"""LoRA fine-tune of a causal LM on Hanoi traces.

Two regimes:
  - baseline: plain LM loss, fresh LoRA on top of the base model.
  - alignment_loss: fresh LoRA on all layers. LM loss flows everywhere; an
    auxiliary alignment loss restricted to a band of mid-layers
    (ALIGNMENT_BAND, default L30-L42) pulls each supervised-position activation
    in the band toward (h_baseline + α · direction[L][state]). Gradient
    restriction is enforced by re-running the band on a DETACHED pre-band
    activation, so alignment gradient cannot flow into LoRA outside the band.
    No hook at inference — the saved adapter is a plain LoRA.

Launch with accelerate; see jobs/finetune_deepseek_r1_qwen32b.job or
jobs/finetune_qwen36_alignment.job.
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

MAX_GRAD_NORM       = 1.0      # gradient clipping.

# ── alignment_loss path only ─────────────────────────────────────────────────
# The actual band, α, λ_target, λ_warmup_frac come from cfg.* CLI flags.
# Gradient checkpointing is disabled in this regime because the band re-run
# capture hooks need a stable forward graph (checkpointing reruns the forward
# during backward, which can repopulate captures unpredictably).
ALIGNMENT_DEFAULT_BAND_STR = "30,31,32,33,34,35,36,37,38,39,40,41,42"


# ── helpers ──────────────────────────────────────────────────────────────────

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
    both regimes."""
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
    """LM-loss eval over the full test set. Same loop for both regimes."""
    model.eval()
    tot_lm = 0.0
    n_lm_batches = 0
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

    return out


# ── startup sanity ──────────────────────────────────────────────────────────

def _startup_sanity_checks(model, tokenizer, first_batch, cfg: Config, total_steps: int) -> None:
    """Loud diagnostics on the first batch. Catches the usual wiring bugs."""
    print("\n" + "█" * 78)
    print("█  STARTUP SANITY CHECKS")
    print("█" * 78)

    # 1. Token at first supervised position.
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

    # 2. LoRA / PEFT confirmation.
    pcfg = getattr(model, "peft_config", None)
    if pcfg:
        for name, c in pcfg.items():
            tgt = getattr(c, "target_modules", None)
            r = getattr(c, "r", "?")
            alpha = getattr(c, "lora_alpha", "?")
            print(f"  [LoRA]         adapter '{name}': r={r} α={alpha} target_modules={tgt}")
    else:
        print("  [LoRA]         ⚠ NO PEFT adapter detected on model")

    print("█" * 78 + "\n")


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    from accelerate import Accelerator
    from accelerate.utils import set_seed as accel_set_seed
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = parse_args()
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
            "MAX_GRAD_NORM": MAX_GRAD_NORM,
        }
        (out_dir / "config.json").write_text(json.dumps(resolved, indent=2))

    # ── tokenizer & dataset ────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds, test_ds = make_datasets(
        Path(cfg.data_dir), tokenizer, cfg.max_seq_len,
        train_file=cfg.train_file,
    )
    accelerator.print(f"[INFO] train_file = {cfg.train_file}")
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
    # Gradient checkpointing interacts badly with the alignment_loss band
    # re-run (the pre-hook captures get re-fired during backward recompute).
    # For baseline keep it on for memory; turn off otherwise.
    if cfg.regime == "alignment_loss":
        accelerator.print("[INFO] alignment_loss: gradient_checkpointing OFF "
                          "(band re-run requires a stable forward graph).")
    else:
        model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    # ── LoRA install ───────────────────────────────────────────────────────
    # alignment_loss uses the same fresh-LoRA install as baseline. The band
    # restriction is enforced via the gradient path (detach + band re-run),
    # NOT via requires_grad — every LoRA param remains trainable.
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

    # ── alignment_loss: caches + capture hooks ─────────────────────────────
    alignment_state = None
    if cfg.regime == "alignment_loss":
        from finetune.alignment_loss import (
            BandCapture as _AlignBandCapture,
            load_alignment_caches as _align_load_caches,
        )
        align_band = [int(x.strip()) for x in cfg.alignment_band.split(",") if x.strip()]
        align_directions, align_h_baseline, cached_band, cached_hidden_dim = (
            _align_load_caches(cfg.centroids_cache, cfg.h_baseline_cache)
        )
        if cached_band != align_band:
            raise ValueError(
                f"--alignment_band={align_band} but caches were precomputed "
                f"with band={cached_band}. Re-precompute or pass a matching band."
            )
        # Locate the decoder-layer list (drill through PEFT wrappers).
        _layers_ref = model
        for _ in range(6):
            if hasattr(_layers_ref, "layers"):
                break
            for attr in ("model", "base_model"):
                sub = getattr(_layers_ref, attr, None)
                if sub is not None:
                    _layers_ref = sub
                    break
        if not hasattr(_layers_ref, "layers"):
            raise RuntimeError("Could not locate decoder layers for alignment_loss")
        align_layers = _layers_ref.layers
        align_capture = _AlignBandCapture(align_band)
        align_capture.install(align_layers)
        alignment_state = {
            "band": align_band,
            "directions": align_directions,
            "h_baseline": align_h_baseline,
            "hidden_dim": cached_hidden_dim,
            "layers": align_layers,
            "capture": align_capture,
        }
        accelerator.print(
            f"[alignment_loss] band={align_band}  "
            f"hidden_dim={cached_hidden_dim}  "
            f"α={cfg.alpha_alignment}  "
            f"λ_target={cfg.lambda_alignment_target}  "
            f"λ_warmup_frac={cfg.lambda_warmup_frac}"
        )
        accelerator.print(
            f"[alignment_loss] centroids ← {cfg.centroids_cache}  "
            f"({sum(len(v) for v in align_directions.values())} (L,state) directions)"
        )
        accelerator.print(
            f"[alignment_loss] h_baseline ← {cfg.h_baseline_cache}  "
            f"({len(align_h_baseline)} (example,position) entries)"
        )
        # Per-layer direction norm summary.
        for L in align_band:
            norms = torch.stack(
                [v.norm() for v in align_directions[L].values()]
            )
            accelerator.print(
                f"  [alignment_loss]   L{L:2d}  ‖direction‖ mean={norms.mean():.2f}  "
                f"min={norms.min():.2f}  max={norms.max():.2f}"
            )

    # ── optimizer ──────────────────────────────────────────────────────────
    lora_params = [p for p in model.parameters() if p.requires_grad]
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

        # alignment_loss: gradient-flow diagnostic. Aborts on band leakage.
        if cfg.regime == "alignment_loss":
            from finetune.alignment_loss import verify_band_gradient_flow
            verify_band_gradient_flow(
                model,
                alignment_state["layers"],
                alignment_state["band"],
                alignment_state["capture"].captures,
                alignment_state["h_baseline"],
                alignment_state["directions"],
                cfg.alpha_alignment,
                alignment_state["hidden_dim"],
                first_batch,
                accelerator,
            )

    # ── training loop ──────────────────────────────────────────────────────
    global_step = 0
    t0 = time.time()
    for epoch in range(cfg.epochs):
        model.train()
        for batch in train_loader:
            with accelerator.accumulate(model):
                n_sup = 0
                # alignment_loss: clear band-capture state before forward; the
                # forward_pre_hooks repopulate it for layers in the band.
                if cfg.regime == "alignment_loss":
                    alignment_state["capture"].clear()
                    n_sup = sum(1 for _ in batch["supervised_positions"])

                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    output_hidden_states=False,
                    use_cache=False,
                )
                lm_loss = outputs.loss

                loss = lm_loss

                # alignment_loss: band re-run on a detached pre-band activation
                # → MSE to (h_baseline + α · direction[L][state]) at supervised
                # positions, summed over band layers. Added to lm_loss with a
                # warmup-ramped λ. Gradient flow is restricted to LoRA inside
                # the band by the detach() inside run_band_with_grad().
                align_loss_v: Optional[float] = None
                lambda_align_eff_v: Optional[float] = None
                align_perturb_rms: Optional[float] = None
                align_dist_target_rms: Optional[float] = None
                align_per_layer_perturb: Optional[dict] = None
                if cfg.regime == "alignment_loss":
                    from finetune.alignment_loss import (
                        compute_alignment_loss as _align_compute,
                        lambda_eff as _align_lambda,
                        run_band_with_grad as _align_band_rerun,
                    )
                    band_outs = _align_band_rerun(
                        alignment_state["layers"],
                        alignment_state["band"],
                        alignment_state["capture"].captures,
                    )
                    align_raw, align_diag = _align_compute(
                        band_outs,
                        batch["supervised_positions"],
                        batch["example_ids"],
                        alignment_state["h_baseline"],
                        alignment_state["directions"],
                        cfg.alpha_alignment,
                        alignment_state["band"],
                        alignment_state["hidden_dim"],
                        device=lm_loss.device,
                    )
                    lambda_align_eff_v = _align_lambda(
                        global_step, total_steps,
                        cfg.lambda_alignment_target, cfg.lambda_warmup_frac,
                    )
                    if align_raw is not None:
                        align_loss_v = float(align_raw.detach().float())
                        loss = lm_loss + lambda_align_eff_v * align_raw.to(lm_loss.dtype)
                        # Aggregate diagnostics across the band for logging.
                        perturb_vals = list(align_diag["per_layer_perturbation_rms"].values())
                        dist_vals = list(align_diag["per_layer_dist_target_rms"].values())
                        align_perturb_rms = sum(perturb_vals) / max(len(perturb_vals), 1)
                        align_dist_target_rms = sum(dist_vals) / max(len(dist_vals), 1)
                        align_per_layer_perturb = align_diag["per_layer_perturbation_rms"]

                accelerator.backward(loss)

                grad_norm_val: Optional[float] = None
                if accelerator.sync_gradients:
                    if lora_params:
                        gn = accelerator.clip_grad_norm_(lora_params, MAX_GRAD_NORM)
                        grad_norm_val = float(gn) if gn is not None else None
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                if global_step % cfg.log_every_steps == 0:
                    lm_v = float(lm_loss.detach().float())
                    lr_lora = scheduler.get_last_lr()[0]
                    _log_jsonl({
                        "step": global_step,
                        "epoch": epoch,
                        "loss_type": cfg.regime,
                        "lm_loss": lm_v,
                        "total_loss": float(loss.detach().float()),
                        "n_sup": int(n_sup),
                        "lr_lora": lr_lora,
                        "grad_norm": grad_norm_val,
                        # alignment_loss diagnostics (None for other regimes)
                        "alignment_loss_raw": align_loss_v,
                        "lambda_align_eff": lambda_align_eff_v,
                        "perturbation_rms_mean": align_perturb_rms,
                        "distance_to_target_rms_mean": align_dist_target_rms,
                        "per_layer_perturbation_rms": align_per_layer_perturb,
                        "elapsed_s": time.time() - t0,
                    })
                    if cfg.regime == "alignment_loss":
                        al = f"{align_loss_v:.4f}" if align_loss_v is not None else "—"
                        pr = f"{align_perturb_rms:.2f}" if align_perturb_rms is not None else "—"
                        dt = f"{align_dist_target_rms:.2f}" if align_dist_target_rms is not None else "—"
                        lam = f"{lambda_align_eff_v:.4f}" if lambda_align_eff_v is not None else "—"
                        accelerator.print(
                            f"  step={global_step:5d}  lm={lm_v:.4f}  "
                            f"align={al}  λ={lam}  ‖perturb‖={pr}  "
                            f"‖dist_target‖={dt}  n_sup={n_sup}"
                        )
                    else:
                        accelerator.print(
                            f"  step={global_step:5d}  lm={lm_v:.4f}  [baseline]"
                        )

                if cfg.eval_every_steps > 0 and global_step % cfg.eval_every_steps == 0:
                    eval_res = _evaluate(model, test_loader, cfg.regime, accelerator)
                    _log_jsonl({"step": global_step, "epoch": epoch, **eval_res})

                # Hard step cap (for fixed-budget runs, e.g. 648-step matching).
                if cfg.max_steps > 0 and global_step >= cfg.max_steps:
                    accelerator.print(
                        f"[INFO] reached --max_steps={cfg.max_steps}; stopping training."
                    )
                    _stop_training = True
                    break
        if locals().get("_stop_training"):
            break

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
