"""LoRA fine-tune of DeepSeek-R1-Distill-Qwen-32B on Hanoi traces.

Two regimes:
  - baseline: plain LM loss, fresh LoRA on top of the base model.
  - augmented_steer: continue an existing baseline LoRA with a forward hook
    at L36 that, at supervised anchor positions (closing `]` of each move
    triple), adds a fixed-magnitude unit-vector bump:
        h_new[anchor] = h[anchor] + AUGMENTED_ALPHA * direction[state]
    Hook fires ONLY during training, at supervised positions only. At inference
    the hook is OFF, so the saved checkpoint behaves like a normal LoRA model.

Launch with accelerate; see jobs/finetune_deepseek_r1_qwen32b.job.
"""

from __future__ import annotations

import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from finetune.config import LORA_TARGETS, Config, parse_args, to_dict
from finetune.data import collate_fn, make_datasets


# ══════════════════════════════════════════════════════════════════════════════
# Hyperparameters — HARDCODED, NOT CLI. Edit here if you need to.
# ══════════════════════════════════════════════════════════════════════════════

PROBE_LAYER         = 36       # transformer layer index for the augmented_steer
                               # forward hook (HF convention: hidden_states[L] is
                               # the output of decoder block L; 0 is embeddings).
MAX_GRAD_NORM       = 1.0      # gradient clipping.

# ── augmented_steer path only ────────────────────────────────────────────────
# Continue a baseline LoRA with a FIXED-magnitude steering bump active at L36
# during training only. Hook fires at supervised anchor positions only;
# correction = AUGMENTED_ALPHA · direction[state] (unit vector). At eval the
# hook stays a no-op. Hypothesis: joint training of upstream + downstream LoRA
# layers teaches the LoRA to exploit state information already in h, so the
# perturbation isn't needed at inference.
AUGMENTED_ALPHA           = 5.0      # FIXED, not learned.
AUGMENTED_CONTINUE_LR     = 1e-4     # lower than from-scratch baseline LR.


# ── helpers ──────────────────────────────────────────────────────────────────

def _param_for_layer(model, layer_idx: int):
    """Return the decoder block at 1-indexed `layer_idx`, drilling through
    PEFT / wrapper levels."""
    obj = model
    for _ in range(6):
        if hasattr(obj, "layers"):
            return obj.layers[layer_idx - 1]
        for attr in ("model", "base_model"):
            sub = getattr(obj, attr, None)
            if sub is not None:
                obj = sub
                break
        else:
            break
    raise RuntimeError("Cannot locate decoder layers on this model")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _git_hash() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_ROOT
        ).decode().strip()
    except Exception:
        return None


def _write_tokenisation_sample(path: Path, example: dict, raw_row: dict, tokenizer) -> None:
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "system", "content": raw_row["system_prompt"]},
         {"role": "user", "content": raw_row["user_prompt"]}],
        tokenize=False,
        add_generation_prompt=True,
    )
    lines = [
        "=== TOKENISATION SAMPLE ===",
        f"system_chars={len(raw_row['system_prompt'])}  user_chars={len(raw_row['user_prompt'])}  "
        f"completion_chars={len(raw_row['completion'])}",
        f"total_tokens={len(example['input_ids'])}",
        f"prompt_tokens={sum(1 for x in example['labels'] if x == -100)}",
        f"supervised_positions ({len(example['supervised_positions'])}):",
    ]
    for ti, target in example["supervised_positions"]:
        tok_id = example["input_ids"][ti]
        tok_str = tokenizer.decode([tok_id])
        lines.append(f"  abs_token_idx={ti:4d}  token={tok_str!r}  target_state={target}")
    lines.append("\n=== FULL TEXT ===")
    lines.append(prompt_text + raw_row["completion"])
    path.write_text("\n".join(lines), encoding="utf-8")


def _save_checkpoint(
    out_dir: Path, cfg: Config, model, tokenizer, accelerator,
    *, trainer_state: Optional[dict] = None,
) -> None:
    """Save the LoRA adapter (+ trainer_state + config.json). Same shape for
    both regimes — augmented_steer's adapter is just the continued LoRA."""
    accelerator.unwrap_model(model).save_pretrained(out_dir / "adapter")
    tokenizer.save_pretrained(out_dir / "adapter")
    if trainer_state is not None:
        torch.save(trainer_state, out_dir / "trainer_state.pt")
    (out_dir / "config.json").write_text(json.dumps(to_dict(cfg), indent=2))


def _build_optimizer_and_scheduler(cfg: Config, lora_params, num_training_steps: int):
    """AdamW + cosine schedule with linear warmup, one parameter group."""
    optim = torch.optim.AdamW(
        [{"params": lora_params, "lr": cfg.lr, "weight_decay": cfg.weight_decay}],
        betas=(0.9, 0.999),
    )

    def lr_lambda(step: int) -> float:
        if step < cfg.warmup_steps:
            return step / max(1, cfg.warmup_steps)
        progress = (step - cfg.warmup_steps) / max(1, num_training_steps - cfg.warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)
    return optim, sched


# ── eval ─────────────────────────────────────────────────────────────────────

def _evaluate(model, loader, regime: str, accelerator) -> dict:
    """LM-loss eval over the full test set. Same loop for both regimes.

    For augmented_steer, a SECOND pass runs with the hook ON + oracle sup_pos
    so we can compare against the (headline) hook-OFF pass.
    """
    model.eval()
    tot_lm = 0.0
    n_lm_batches = 0
    # Make sure the hook is OFF for the headline pass (sup_pos=None).
    if regime == "augmented_steer":
        from finetune.augmented_steer import hook_state as aug_hook_state
        aug_hook_state()["sup_pos"] = None
    with torch.no_grad():
        for batch in loader:
            outputs = model(
                input_ids=batch["input_ids"].to(accelerator.device),
                attention_mask=batch["attention_mask"].to(accelerator.device),
                labels=batch["labels"].to(accelerator.device),
                output_hidden_states=False,
                use_cache=False,
            )
            tot_lm += float(outputs.loss.detach().float().item())
            n_lm_batches += 1
    model.train()

    out = {
        "eval_lm_loss": tot_lm / max(1, n_lm_batches),
        "n_eval_batches": n_lm_batches,
    }

    # augmented_steer: diagnostic second pass with hook ON + oracle sup_pos.
    if regime == "augmented_steer":
        from finetune.augmented_steer import (
            filter_sup_pos_to_cache as aug_filter,
            hook_state as aug_hook_state,
        )
        aug_st = aug_hook_state()
        directions = aug_st.get("directions") or {}
        out["eval_lm_loss_hook_OFF"] = out["eval_lm_loss"]
        model.eval()
        tot_lm1 = 0.0
        nb1 = 0
        corr_norm_sum = 0.0
        corr_norm_n   = 0
        try:
            with torch.no_grad():
                for batch in loader:
                    filt = aug_filter(batch["supervised_positions"], directions)
                    aug_st["sup_pos"] = filt
                    o = model(
                        input_ids=batch["input_ids"].to(accelerator.device),
                        attention_mask=batch["attention_mask"].to(accelerator.device),
                        labels=batch["labels"].to(accelerator.device),
                        output_hidden_states=False, use_cache=False,
                    )
                    tot_lm1 += float(o.loss.detach().float().item())
                    nb1 += 1
                    norms = aug_st.get("last_correction_norms") or []
                    if norms:
                        corr_norm_sum += sum(norms)
                        corr_norm_n   += len(norms)
            out["eval_lm_loss_hook_ON_with_oracle"] = tot_lm1 / max(1, nb1)
            out["eval_correction_norm_at_anchor_mean"] = (
                corr_norm_sum / corr_norm_n if corr_norm_n > 0 else 0.0
            )
            out["eval_augmented_alpha"] = float(aug_st.get("alpha") or 0.0)
        finally:
            aug_st["sup_pos"] = None
        model.train()

    return out


# ── startup sanity ──────────────────────────────────────────────────────────

def _startup_sanity_checks(model, tokenizer, first_batch, cfg: Config, total_steps: int) -> None:
    """Loud diagnostics on the first batch. Catches the usual wiring bugs."""
    print("\n" + "█" * 78)
    print("█  STARTUP SANITY CHECKS")
    print("█" * 78)

    # 1. Decoder layer module.
    try:
        block = _param_for_layer(model, PROBE_LAYER)
        block_dev = next(block.parameters()).device
        print(f"  [layer]        decoder block at L{PROBE_LAYER}: {type(block).__name__}  device={block_dev}")
    except Exception as e:
        print(f"  [layer]        COULD NOT LOCATE LAYER — {e!r}")

    # 2. Token at first supervised position.
    sup_pos = first_batch["supervised_positions"]
    if sup_pos:
        b0, t0, target0 = sup_pos[0]
        tok_id = int(first_batch["input_ids"][b0, t0].item())
        tok_str = tokenizer.decode([tok_id])
        marker = "  ✓ looks like a closing bracket" if "]" in tok_str else "  ⚠ NOT a `]`"
        print(f"  [anchor]       supervised_positions[0] = (b={b0}, t={t0}, target={target0})")
        print(f"                 decoded = {tok_str!r}{marker}")
    else:
        print("  [anchor]       ⚠ first batch has NO supervised positions")

    # 3. LoRA / PEFT confirmation.
    pcfg = getattr(model, "peft_config", None)
    if pcfg:
        for name, c in pcfg.items():
            tgt = getattr(c, "target_modules", None)
            r = getattr(c, "r", "?")
            alpha = getattr(c, "lora_alpha", "?")
            print(f"  [LoRA]         adapter '{name}': r={r} α={alpha} target_modules={tgt}")
    else:
        print("  [LoRA]         ⚠ NO PEFT adapter detected on model")

    # 4. augmented_steer-specific: hook differential at L36.
    if cfg.regime == "augmented_steer":
        from finetune.augmented_steer import (
            filter_sup_pos_to_cache as aug_filter,
            hook_state as aug_hook_state,
        )
        aug_st = aug_hook_state()
        directions = aug_st.get("directions") or {}
        alpha_const = float(aug_st.get("alpha") or 0.0)
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  [augmented]    continuing LoRA  trainable_params={n_train:,}  "
              f"α (FIXED) = {alpha_const}")
        print(f"  [augmented]    directions cache: {len(directions)} states  "
              f"‖direction[s]‖ should all be 1.0")
        if directions:
            sample_state, sample_dir = next(iter(directions.items()))
            print(f"  [augmented]    sample state={list(sample_state)}  "
                  f"direction[:5]={sample_dir[:5].tolist()}  "
                  f"‖direction‖={float(sample_dir.norm()):.4f}")
        if sup_pos:
            with torch.no_grad():
                in_dev = model.get_input_embeddings().weight.device
                ids = first_batch["input_ids"][:1].to(in_dev)
                am  = first_batch["attention_mask"][:1].to(in_dev)
                # Off pass.
                aug_st["sup_pos"] = None
                o0 = model(input_ids=ids, attention_mask=am,
                           output_hidden_states=True, use_cache=False)
                h0 = o0.hidden_states[PROBE_LAYER]
                # On pass: row-0 supervised positions only.
                row0_sup = [(0, t, st) for (b, t, st) in sup_pos if b == 0]
                row0_filt = aug_filter(row0_sup, directions)
                aug_st["sup_pos"] = row0_filt
                o1 = model(input_ids=ids, attention_mask=am,
                           output_hidden_states=True, use_cache=False)
                h1 = o1.hidden_states[PROBE_LAYER]
                aug_st["sup_pos"] = None
                diff = float((h1.float() - h0.float()).norm())
                ref  = float(h0.float().norm())
            print(f"  [augmented]    hook differential at L{PROBE_LAYER}: "
                  f"‖h(ON) - h(OFF)‖ = {diff:.4f}  (rel {diff/max(ref,1e-8):.6f})  "
                  f"n_sup_for_test={len(row0_filt)}")
            if row0_filt and diff == 0.0 and alpha_const != 0.0:
                raise RuntimeError(
                    "Augmented hook is a no-op (hidden states identical with sup_pos "
                    "set). Hook isn't modifying the propagated tensor — check tuple "
                    "unpacking and return value."
                )
            elif alpha_const == 0.0:
                print(f"  [augmented]    α=0 — diff=0 expected (control mode).")

    print("█" * 78 + "\n")


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    from accelerate import Accelerator
    from accelerate.utils import set_seed as accel_set_seed
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = parse_args()
    # Layer-ablation hook: --probe_layer overrides the module-level constant
    # for THIS run only.
    global PROBE_LAYER
    if cfg.probe_layer is not None:
        PROBE_LAYER = int(cfg.probe_layer)
        print(f"[INFO] PROBE_LAYER overridden by --probe_layer → {PROBE_LAYER}")
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _seed_everything(cfg.seed)
    accelerator = Accelerator(gradient_accumulation_steps=cfg.grad_accum_steps,
                              mixed_precision="bf16")
    accel_set_seed(cfg.seed)
    is_main = accelerator.is_main_process
    log_path = out_dir / "log.jsonl"

    def _log_jsonl(rec: dict) -> None:
        if not is_main:
            return
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        accelerator.print(json.dumps(rec))

    if is_main:
        resolved = to_dict(cfg)
        resolved["git_hash"] = _git_hash()
        resolved["constants"] = {
            "PROBE_LAYER": PROBE_LAYER,
            "MAX_GRAD_NORM": MAX_GRAD_NORM,
            "AUGMENTED_ALPHA": AUGMENTED_ALPHA,
            "AUGMENTED_CONTINUE_LR": AUGMENTED_CONTINUE_LR,
        }
        (out_dir / "config.json").write_text(json.dumps(resolved, indent=2))

    # ── tokenizer & dataset ────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds, test_ds = make_datasets(Path(cfg.data_dir), tokenizer, cfg.max_seq_len)
    accelerator.print(f"[INFO] train={len(train_ds)}  test={len(test_ds)}")

    if is_main:
        _write_tokenisation_sample(
            out_dir / "tokenisation_sample.txt",
            train_ds[0], train_ds.rows[0], tokenizer,
        )

    train_loader = DataLoader(train_ds, batch_size=cfg.per_device_batch_size,
                              shuffle=True, collate_fn=collate_fn,
                              num_workers=2, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.per_device_batch_size,
                             shuffle=False, collate_fn=collate_fn, num_workers=2)

    # ── model + LoRA ───────────────────────────────────────────────────────
    accelerator.print(f"[INFO] Loading {cfg.model_id} in bf16 with device_map=auto")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    if cfg.regime == "augmented_steer":
        # Continue training the baseline LoRA. Same weights, fresh optimizer.
        adapter_dir = Path(cfg.baseline_checkpoint) / "adapter"
        if not adapter_dir.exists():
            raise FileNotFoundError(
                f"--baseline_checkpoint must point at a dir containing adapter/, "
                f"got: {adapter_dir}"
            )
        accelerator.print(f"[INFO] augmented_steer: continuing LoRA training "
                          f"from {adapter_dir}")
        model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=True)
        for n, p in model.named_parameters():
            if p.dtype != torch.bfloat16 and p.requires_grad:
                p.data = p.data.to(torch.bfloat16)
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        accelerator.print(f"[INFO] augmented_steer: trainable params = {n_train:,}")
    else:
        lora_cfg = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=LORA_TARGETS,
        )
        model = get_peft_model(model, lora_cfg)
        for n, p in model.named_parameters():
            if p.requires_grad and p.dtype != torch.bfloat16:
                p.data = p.data.to(torch.bfloat16)
        if is_main:
            model.print_trainable_parameters()

    # ── augmented_steer: directions cache + hook ───────────────────────────
    augmented_directions = None
    augmented_hook_handle = None
    if cfg.regime == "augmented_steer":
        from finetune.augmented_steer import (
            install_hook as aug_install_hook,
            load_directions_cache,
            set_alpha as aug_set_alpha,
            set_directions as aug_set_directions,
        )
        augmented_directions, aug_global_mean, aug_raw = load_directions_cache(
            cfg.steering_directions,
        )
        aug_set_directions(augmented_directions)
        effective_alpha = (cfg.augmented_alpha if cfg.augmented_alpha is not None
                           else AUGMENTED_ALPHA)
        aug_set_alpha(effective_alpha)
        target_block = _param_for_layer(model, PROBE_LAYER)
        augmented_hook_handle = aug_install_hook(target_block)
        gm_norm = float(aug_global_mean.norm()) if aug_global_mean is not None else float("nan")
        accelerator.print(
            f"[augmented_steer] loaded {len(augmented_directions)} directions from "
            f"{cfg.steering_directions} (probe_layer={aug_raw.get('probe_layer')})"
        )
        accelerator.print(
            f"[augmented_steer] α (FIXED) = {effective_alpha}  "
            f"(override={cfg.augmented_alpha})  "
            f"‖h_global_mean‖ = {gm_norm:.2f}  "
            f"hook attached at L{PROBE_LAYER} ({type(target_block).__name__})"
        )
        if effective_alpha == 0.0:
            accelerator.print(
                f"[augmented_steer] α=0 — control mode: hook fires but adds "
                f"zero perturbation. Functionally identical to plain LoRA "
                f"continuation."
            )

    # ── optimizer ──────────────────────────────────────────────────────────
    lora_params = [p for p in model.parameters() if p.requires_grad]
    if cfg.regime == "augmented_steer":
        # Default to AUGMENTED_CONTINUE_LR for continuation. Caller can still
        # override via --lr if they pass a small enough value.
        if cfg.lr > AUGMENTED_CONTINUE_LR * 2:
            accelerator.print(
                f"[augmented_steer] lowering LR from {cfg.lr} to "
                f"{AUGMENTED_CONTINUE_LR} for continuation training"
            )
            cfg.lr = AUGMENTED_CONTINUE_LR
    steps_per_epoch = math.ceil(len(train_loader) / cfg.grad_accum_steps)
    total_steps = steps_per_epoch * cfg.epochs
    optimizer, scheduler = _build_optimizer_and_scheduler(cfg, lora_params, total_steps)

    optimizer, train_loader, test_loader, scheduler = accelerator.prepare(
        optimizer, train_loader, test_loader, scheduler,
    )

    # ── sanity checks (first batch only) ───────────────────────────────────
    if is_main:
        first_batch = next(iter(train_loader))
        _startup_sanity_checks(model, tokenizer, first_batch, cfg, total_steps)

    # ── training loop ──────────────────────────────────────────────────────
    global_step = 0
    t0 = time.time()
    for epoch in range(cfg.epochs):
        model.train()
        for batch in train_loader:
            with accelerator.accumulate(model):
                # augmented_steer: register sup_pos so the forward hook can
                # find the (b, t) anchors and apply correction.
                n_sup = 0
                if cfg.regime == "augmented_steer":
                    from finetune.augmented_steer import (
                        filter_sup_pos_to_cache as aug_filter,
                        hook_state as aug_hook_state,
                    )
                    _aug_state = aug_hook_state()
                    _aug_filt = aug_filter(
                        batch["supervised_positions"], augmented_directions
                    )
                    _aug_state["sup_pos"] = _aug_filt
                    n_sup = len(_aug_filt)

                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    output_hidden_states=False,
                    use_cache=False,
                )
                lm_loss = outputs.loss
                loss = lm_loss
                accelerator.backward(loss)

                grad_norm_val: Optional[float] = None
                if accelerator.sync_gradients:
                    if lora_params:
                        gn = accelerator.clip_grad_norm_(lora_params, MAX_GRAD_NORM)
                        grad_norm_val = float(gn) if gn is not None else None
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                if cfg.regime == "augmented_steer":
                    from finetune.augmented_steer import hook_state as aug_hook_state
                    aug_hook_state()["sup_pos"] = None

            if accelerator.sync_gradients:
                global_step += 1
                if global_step % cfg.log_every_steps == 0:
                    lm_v = float(lm_loss.detach().float())
                    lr_lora = scheduler.get_last_lr()[0]
                    alpha_v = None
                    correction_norm_mean = None
                    if cfg.regime == "augmented_steer":
                        from finetune.augmented_steer import hook_state as aug_hook_state
                        alpha_v = float(aug_hook_state().get("alpha") or 0.0)
                        norms = aug_hook_state().get("last_correction_norms") or []
                        if norms:
                            correction_norm_mean = sum(norms) / len(norms)
                    _log_jsonl({
                        "step": global_step,
                        "epoch": epoch,
                        "loss_type": cfg.regime,
                        "lm_loss": lm_v,
                        "total_loss": float(loss.detach().float()),
                        "n_sup": int(n_sup),
                        "lr_lora": lr_lora,
                        "grad_norm": grad_norm_val,
                        "alpha": alpha_v,
                        "correction_norm_mean": correction_norm_mean,
                        "elapsed_s": time.time() - t0,
                    })
                    if cfg.regime == "augmented_steer":
                        cn = f"{correction_norm_mean:.2f}" if correction_norm_mean else "—"
                        accelerator.print(
                            f"  step={global_step:5d}  lm={lm_v:.4f}  "
                            f"α={alpha_v}  ‖corr‖={cn}  n_sup={n_sup}"
                        )
                    else:
                        accelerator.print(
                            f"  step={global_step:5d}  lm={lm_v:.4f}  [baseline]"
                        )

                if cfg.eval_every_steps > 0 and global_step % cfg.eval_every_steps == 0:
                    eval_res = _evaluate(model, test_loader, cfg.regime, accelerator)
                    _log_jsonl({"step": global_step, "epoch": epoch, **eval_res})

        # end of epoch
        eval_res = _evaluate(model, test_loader, cfg.regime, accelerator)
        _log_jsonl({"step": global_step, "epoch": epoch, "end_of_epoch": True, **eval_res})

        if cfg.save_every_epoch and is_main:
            ckpt_dir = out_dir / f"epoch_{epoch}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            _save_checkpoint(
                ckpt_dir, cfg, model, tokenizer, accelerator,
                trainer_state={
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "global_step": global_step,
                    "epoch": epoch,
                },
            )

    # final checkpoint
    if is_main:
        final_dir = out_dir / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        _save_checkpoint(final_dir, cfg, model, tokenizer, accelerator)
        accelerator.print(f"[INFO] saved final adapter to {final_dir}")


if __name__ == "__main__":
    main()
