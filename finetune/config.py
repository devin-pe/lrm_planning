"""Hyperparameter defaults and CLI for the LoRA fine-tune runs."""

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
    lambda_geom: float = 0.05
    lambda_warmup_frac: float = 0.2
    layer: int = 36
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lr: float = 2e-4
    probe_lr: float = 1e-3
    weight_decay: float = 0.01
    warmup_steps: int = 100
    epochs: int = 2
    per_device_batch_size: int = 2
    grad_accum_steps: int = 4
    max_seq_len: int = 768
    seed: int = 42
    data_dir: str = "hanoi_data/"
    save_every_epoch: bool = True
    log_every_steps: int = 10
    eval_every_steps: int = 100
    grad_clip: float = 1.0
    hidden_dim: int = HIDDEN_DIM


def _str2bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    if v.lower() in ("no", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError(f"Expected bool, got {v!r}")


def parse_args(argv: Optional[list] = None) -> Config:
    p = argparse.ArgumentParser(description="LoRA fine-tune for ToH with optional probe loss")
    p.add_argument("--regime", choices=["baseline", "probe"], required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--model_id", default=MODEL_ID)
    p.add_argument("--lambda_geom", type=float, default=0.05,
                   help="Target λ for the probe loss after warmup.")
    p.add_argument("--lambda_warmup_frac", type=float, default=0.2,
                   help="Fraction of training steps over which λ ramps linearly "
                        "from 0 to --lambda_geom. 0.0 disables warmup.")
    p.add_argument("--layer", type=int, default=36)
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--probe_lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--per_device_batch_size", type=int, default=2)
    p.add_argument("--grad_accum_steps", type=int, default=4)
    p.add_argument("--max_seq_len", type=int, default=768)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data_dir", default="hanoi_data/")
    p.add_argument("--save_every_epoch", type=_str2bool, default=True)
    p.add_argument("--log_every_steps", type=int, default=10)
    p.add_argument("--eval_every_steps", type=int, default=100)
    p.add_argument("--grad_clip", type=float, default=1.0)
    args = p.parse_args(argv)
    return Config(**vars(args))


def to_dict(cfg: Config) -> dict:
    return asdict(cfg)
