"""Hyperparameter defaults and CLI for the LoRA fine-tune runs.

Two loss types are supported:
  - baseline: plain LM loss, fresh LoRA on top of the base model.
  - augmented_steer: continue a baseline LoRA with a fixed-α steering bump
    applied at L36 supervised anchors during training (hook OFF at inference).
    All augmented_steer hyperparameters except α and the cache paths are
    hardcoded at the top of finetune/train.py.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from typing import Optional


MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
HIDDEN_DIM = 5120
N_LAYERS = 64
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
                "up_proj", "gate_proj", "down_proj"]


@dataclass
class Config:
    regime: str
    output_dir: str
    model_id: str = MODEL_ID
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lr: float = 2e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    epochs: int = 1
    per_device_batch_size: int = 2
    grad_accum_steps: int = 4
    max_seq_len: int = 768
    seed: int = 42
    data_dir: str = "hanoi_data/"
    save_every_epoch: bool = True
    log_every_steps: int = 10
    eval_every_steps: int = 100
    hidden_dim: int = HIDDEN_DIM
    probe_layer: Optional[int] = None  # if set, overrides PROBE_LAYER in train.py
    # augmented_steer only: path to existing LoRA-baseline ckpt to load + continue.
    baseline_checkpoint: Optional[str] = None
    # augmented_steer only: path to precomputed unit-norm steering directions.
    steering_directions: Optional[str] = None
    # augmented_steer only: override the hardcoded α (e.g. 0 for a control
    # run = plain LoRA continuation).
    augmented_alpha: Optional[float] = None


def _str2bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    if v.lower() in ("no", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError(f"Expected bool, got {v!r}")


def parse_args(argv: Optional[list] = None) -> Config:
    p = argparse.ArgumentParser(description="LoRA fine-tune for ToH")
    # --regime is the legacy flag; --loss_type is preferred. Both write to cfg.regime.
    p.add_argument("--regime", choices=["baseline", "augmented_steer"],
                   required=False, default=None)
    p.add_argument("--loss_type", choices=["baseline", "augmented_steer"],
                   required=False, default=None,
                   help="baseline = LM only; "
                        "augmented_steer = continue baseline LoRA with a fixed "
                        "steering bump at L36 anchors during training (hook OFF "
                        "at inference).")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--model_id", default=MODEL_ID)
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--per_device_batch_size", type=int, default=2)
    p.add_argument("--grad_accum_steps", type=int, default=4)
    p.add_argument("--max_seq_len", type=int, default=768)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data_dir", default="hanoi_data/")
    p.add_argument("--save_every_epoch", type=_str2bool, default=True)
    p.add_argument("--log_every_steps", type=int, default=10)
    p.add_argument("--eval_every_steps", type=int, default=100)
    p.add_argument("--probe_layer", type=int, default=None,
                   help="Override the hardcoded PROBE_LAYER for layer ablation.")
    p.add_argument("--baseline_checkpoint", default=None,
                   help="(augmented_steer) path to baseline LoRA ckpt "
                        "(e.g. runs/baseline/final). LoRA continues training "
                        "from these weights.")
    p.add_argument("--steering_directions", default=None,
                   help="(augmented_steer) path to precomputed unit-norm "
                        "{state_tuple → direction_5120} dict + global_mean. "
                        "Build it with "
                        "`python -m finetune.precompute_steering_directions ...`.")
    p.add_argument("--augmented_alpha", type=float, default=None,
                   help="(augmented_steer) override the hardcoded α=5.0. "
                        "Set to 0.0 for a plain-LoRA-continuation control.")
    args = p.parse_args(argv)
    chosen = args.loss_type or args.regime
    if chosen is None:
        p.error("must pass --loss_type {baseline|augmented_steer}")
    if chosen == "augmented_steer":
        if not args.baseline_checkpoint:
            p.error("--loss_type=augmented_steer requires --baseline_checkpoint")
        if not args.steering_directions:
            p.error("--loss_type=augmented_steer requires --steering_directions "
                    "(precompute it first)")
    args.regime = chosen
    del args.loss_type
    return Config(**vars(args))


def to_dict(cfg: Config) -> dict:
    return asdict(cfg)
