"""LoRA fine-tune of DeepSeek-R1-Distill-Qwen-32B on the templated Hanoi
traces, with an optional probe-loss term at layer 36.

Launch with accelerate (see launch_train.sh).
"""

from __future__ import annotations

import json
import math
import os
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
from finetune.loss import compute_geom_loss, effective_lambda, total_loss
from finetune.probe import Probe
from finetune.utils import normalised_graph_distance_matrix


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


def _build_optimizer_and_scheduler(
    cfg: Config,
    lora_params,
    probe_params,
    num_training_steps: int,
):
    param_groups = [
        {"params": lora_params, "lr": cfg.lr, "weight_decay": cfg.weight_decay},
    ]
    if probe_params is not None:
        param_groups.append(
            {"params": probe_params, "lr": cfg.probe_lr, "weight_decay": cfg.weight_decay}
        )
    optim = torch.optim.AdamW(param_groups, betas=(0.9, 0.999))

    def lr_lambda(step: int) -> float:
        if step < cfg.warmup_steps:
            return step / max(1, cfg.warmup_steps)
        progress = (step - cfg.warmup_steps) / max(1, num_training_steps - cfg.warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)
    return optim, sched


def _evaluate(
    model,
    probe,
    loader,
    norm_dist,
    layer_idx: int,
    regime: str,
    accelerator,
) -> dict:
    model.eval()
    if probe is not None:
        probe.eval()

    tot_lm = 0.0
    tot_geom = 0.0
    n_lm_batches = 0
    n_geom_batches = 0
    with torch.no_grad():
        for batch in loader:
            # With device_map="auto" the model's first shard (embeddings)
            # lives on cuda:0; HF moves inputs as needed. We send the batch
            # to the accelerator's primary device and let HF route the rest.
            input_ids = batch["input_ids"].to(accelerator.device)
            attention_mask = batch["attention_mask"].to(accelerator.device)
            labels = batch["labels"].to(accelerator.device)
            sup_pos = batch["supervised_positions"]

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                output_hidden_states=(regime == "probe"),
                use_cache=False,
            )
            lm = outputs.loss.detach().float()
            tot_lm += lm.item()
            n_lm_batches += 1

            if regime == "probe":
                hidden = outputs.hidden_states[layer_idx]
                geom, n_sup = compute_geom_loss(hidden, sup_pos, probe, norm_dist)
                if n_sup >= 2:
                    tot_geom += geom.detach().float().item()
                    n_geom_batches += 1

    model.train()
    if probe is not None:
        probe.train()
    return {
        "eval_lm_loss": tot_lm / max(1, n_lm_batches),
        "eval_geom_loss": tot_geom / max(1, n_geom_batches) if n_geom_batches else None,
        "n_eval_batches": n_lm_batches,
    }


def main() -> None:
    # Imports gated here so utils.py can be smoke-tested without HF deps installed.
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
            train_ds[0],
            train_ds.rows[0],
            tokenizer,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.per_device_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.per_device_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
    )

    # ── model + LoRA ───────────────────────────────────────────────────────
    # Frozen 32B base + tiny LoRA adapter ≠ a workload that needs FSDP. Use
    # HF's native `device_map="auto"` to shard the base across the visible
    # GPUs at load time (one shard per GPU, no broadcast spike). Launch with
    # `--num_processes 1` so this is a single-process job; accelerate is
    # kept around purely for grad-accumulation + bf16 plumbing.
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

    lora_cfg = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=LORA_TARGETS,
    )
    model = get_peft_model(model, lora_cfg)
    # Even without FSDP we keep LoRA in bf16 so it matches the frozen base —
    # mixed-precision otherwise adds a fp32 copy of the adapter, which is
    # wasted memory.
    for n, p in model.named_parameters():
        if p.requires_grad and p.dtype != torch.bfloat16:
            p.data = p.data.to(torch.bfloat16)
    if is_main:
        model.print_trainable_parameters()

    # ── probe + graph distances ────────────────────────────────────────────
    probe: Optional[Probe] = None
    norm_dist = normalised_graph_distance_matrix()
    if is_main:
        torch.save(norm_dist, out_dir / "graph_distance_matrix.pt")

    # With device_map="auto", HF's accelerate hooks collate every layer's
    # output back to the model's input-embedding device (usually cuda:0)
    # before returning `outputs.hidden_states`. So even though layer
    # `cfg.layer` itself may sit on a different shard, the hidden states we
    # read for the probe live there. Probe goes on the same device.
    hidden_state_device = model.get_input_embeddings().weight.device

    if cfg.regime == "probe":
        probe = Probe(hidden_dim=cfg.hidden_dim).to(
            device=hidden_state_device, dtype=torch.bfloat16,
        )

    # ── optimizer ──────────────────────────────────────────────────────────
    lora_params = [p for p in model.parameters() if p.requires_grad]
    probe_params = list(probe.parameters()) if probe is not None else None
    steps_per_epoch = math.ceil(len(train_loader) / cfg.grad_accum_steps)
    total_steps = steps_per_epoch * cfg.epochs
    optimizer, scheduler = _build_optimizer_and_scheduler(
        cfg, lora_params, probe_params, total_steps
    )

    # We deliberately do NOT call accelerator.prepare on the model — its
    # device placement is already handled by HF (device_map="auto"). Wrap
    # everything else.
    optimizer, train_loader, test_loader, scheduler = accelerator.prepare(
        optimizer, train_loader, test_loader, scheduler
    )

    norm_dist_dev = norm_dist.to(hidden_state_device)

    # ── training loop ──────────────────────────────────────────────────────
    global_step = 0
    micro_step = 0
    t0 = time.time()
    for epoch in range(cfg.epochs):
        model.train()
        if probe is not None:
            probe.train()
        for batch in train_loader:
            with accelerator.accumulate(model):
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                labels = batch["labels"]
                sup_pos = batch["supervised_positions"]

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    output_hidden_states=(cfg.regime == "probe"),
                    use_cache=False,
                )
                lm_loss = outputs.loss

                geom_loss = torch.tensor(0.0, device=lm_loss.device, dtype=lm_loss.dtype)
                n_sup = 0
                if cfg.regime == "probe":
                    hidden = outputs.hidden_states[cfg.layer]
                    geom_loss, n_sup = compute_geom_loss(
                        hidden, sup_pos, probe, norm_dist_dev
                    )
                    # geom_loss may sit on a different shard from lm_loss
                    # under device_map="auto"; move it (the .to keeps the
                    # autograd graph for CUDA→CUDA copies).
                    geom_loss = geom_loss.to(lm_loss.device)

                # Linear λ-warmup so the LM stabilises before the probe term
                # starts pulling on the representation.
                lambda_eff = effective_lambda(
                    global_step, total_steps, cfg.lambda_geom, cfg.lambda_warmup_frac,
                ) if cfg.regime == "probe" else 0.0
                loss = total_loss(lm_loss, geom_loss, lambda_eff, cfg.regime)
                accelerator.backward(loss)
                grad_norm_val: Optional[float] = None
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(
                        [*lora_params, *(probe_params or [])], cfg.grad_clip
                    )
                    grad_norm_val = float(grad_norm) if grad_norm is not None else None
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            micro_step += 1
            if accelerator.sync_gradients:
                global_step += 1
                if global_step % cfg.log_every_steps == 0:
                    lm_v = float(lm_loss.detach().float())
                    geom_v = float(geom_loss.detach().float()) if cfg.regime == "probe" else None
                    # (lambda_eff * geom_loss) / lm_loss — how much the
                    # probe term is currently pulling vs LM. Want this in
                    # the ~0.1–0.3 ballpark; anything >1 means the probe
                    # dominates and the LM is being deprioritised.
                    ratio = (
                        (lambda_eff * geom_v) / lm_v
                        if (cfg.regime == "probe" and geom_v is not None and lm_v > 1e-9)
                        else None
                    )
                    _log_jsonl({
                        "step": global_step,
                        "epoch": epoch,
                        "lm_loss": lm_v,
                        "geom_loss": geom_v,
                        "lambda_eff": lambda_eff if cfg.regime == "probe" else None,
                        "lambda_geom_target": cfg.lambda_geom if cfg.regime == "probe" else None,
                        "geom_to_lm_ratio": ratio,
                        "total_loss": float(loss.detach().float()),
                        "n_sup": int(n_sup),
                        "lr": scheduler.get_last_lr()[0],
                        "grad_norm": grad_norm_val,
                        "elapsed_s": time.time() - t0,
                    })
                    if cfg.regime == "probe":
                        accelerator.print(
                            f"  step={global_step:5d}  lm={lm_v:.4f}  "
                            f"geom={geom_v:.4f}  λ_eff={lambda_eff:.4f}  "
                            f"λg/lm={ratio:.3f}  n_sup={n_sup}"
                        )
                if (cfg.eval_every_steps > 0
                        and global_step % cfg.eval_every_steps == 0):
                    eval_res = _evaluate(
                        model, probe, test_loader, norm_dist_dev,
                        cfg.layer, cfg.regime, accelerator,
                    )
                    _log_jsonl({"step": global_step, "epoch": epoch, **eval_res})

        # end of epoch: full eval + maybe checkpoint
        eval_res = _evaluate(
            model, probe, test_loader, norm_dist_dev,
            cfg.layer, cfg.regime, accelerator,
        )
        _log_jsonl({"step": global_step, "epoch": epoch, "end_of_epoch": True, **eval_res})

        if cfg.save_every_epoch and is_main:
            ckpt_dir = out_dir / f"epoch_{epoch}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            accelerator.unwrap_model(model).save_pretrained(ckpt_dir / "adapter")
            tokenizer.save_pretrained(ckpt_dir / "adapter")
            if probe is not None:
                torch.save(
                    accelerator.unwrap_model(probe).state_dict(),
                    ckpt_dir / "probe.pt",
                )
            torch.save({
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "global_step": global_step,
                "epoch": epoch,
            }, ckpt_dir / "trainer_state.pt")
            (ckpt_dir / "config.json").write_text(json.dumps(to_dict(cfg), indent=2))

    # final checkpoint
    if is_main:
        final_dir = out_dir / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        accelerator.unwrap_model(model).save_pretrained(final_dir / "adapter")
        tokenizer.save_pretrained(final_dir / "adapter")
        if probe is not None:
            torch.save(
                accelerator.unwrap_model(probe).state_dict(),
                final_dir / "probe.pt",
            )
        torch.save(norm_dist, final_dir / "graph_distance_matrix.pt")
        (final_dir / "config.json").write_text(json.dumps(to_dict(cfg), indent=2))
        accelerator.print(f"[INFO] saved final adapter + probe to {final_dir}")


if __name__ == "__main__":
    main()
