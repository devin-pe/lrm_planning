"""Hyperparameter defaults and CLI for the LoRA fine-tune runs.

Two loss types are supported:
  - baseline: plain LM loss, fresh LoRA on top of the base model.
  - alignment_loss: fresh LoRA on all layers; LM loss flows everywhere, but
    an auxiliary alignment loss restricted to a band of mid-layers (default
    L30-L42) pulls each supervised-position activation toward a target
    (h_baseline + α · direction[L][state]). Strict band-gradient mechanism
    via a second band-only forward pass on a detached pre-band activation.
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
    # Override the training-data filename inside data_dir (default `train.jsonl`).
    # Pass `train_multi_puzzle.jsonl` to use the chained-puzzle augmented data.
    train_file: str = "train.jsonl"
    # Stop after this many global steps regardless of epoch count. 0 = unused.
    # Set to e.g. 648 to match the original baseline budget on a larger dataset.
    max_steps: int = 0

    # ── alignment_loss only ────────────────────────────────────────────────
    # CSV of layer indices that receive alignment-loss gradient.
    alignment_band: str = "30,31,32,33,34,35,36,37,38,39,40,41,42"
    # Scaling on the (h_target - h_global_mean) direction. Direction is NOT
    # unit-norm; expect raw magnitudes 50-100 in bf16. Start with α ≈ 1.0
    # then sweep down to 0.1-0.2 if the target is too far from baseline.
    alpha_alignment: float = 1.0
    # End-of-warmup λ weight on the alignment loss term in total = lm + λ·align.
    lambda_alignment_target: float = 0.10
    # Fraction of total steps over which λ ramps linearly from 0 to target.
    lambda_warmup_frac: float = 0.20
    # Paths to the two precompute outputs (precompute_alignment.py).
    centroids_cache: Optional[str] = None      # state_centroids.pt
    h_baseline_cache: Optional[str] = None     # h_baseline_supervised.pt


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
    p.add_argument("--regime",
                   choices=["baseline", "alignment_loss"],
                   required=False, default=None)
    p.add_argument("--loss_type",
                   choices=["baseline", "alignment_loss"],
                   required=False, default=None,
                   help="baseline = LM only; "
                        "alignment_loss = fresh LoRA on all layers, LM loss "
                        "everywhere + band-restricted alignment loss at "
                        "ALIGNMENT_BAND supervised anchors (no hook at inference).")
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
    p.add_argument("--train_file", default="train.jsonl",
                   help="Training data filename inside --data_dir.")
    p.add_argument("--max_steps", type=int, default=0,
                   help="Hard cap on global training steps. 0 = unused.")
    p.add_argument("--alignment_band",
                   default="30,31,32,33,34,35,36,37,38,39,40,41,42",
                   help="(alignment_loss) CSV of layer indices that receive "
                        "alignment-loss gradient.")
    p.add_argument("--alpha_alignment", type=float, default=1.0,
                   help="(alignment_loss) scale on direction vector (NOT unit-norm).")
    p.add_argument("--lambda_alignment_target", type=float, default=0.10,
                   help="(alignment_loss) end-of-warmup λ weight on alignment loss.")
    p.add_argument("--lambda_warmup_frac", type=float, default=0.20,
                   help="(alignment_loss) fraction of total steps for linear λ warmup.")
    p.add_argument("--centroids_cache", default=None,
                   help="(alignment_loss) path to state_centroids.pt "
                        "(precompute_alignment.py output).")
    p.add_argument("--h_baseline_cache", default=None,
                   help="(alignment_loss) path to h_baseline_supervised.pt "
                        "(precompute_alignment.py output).")
    args = p.parse_args(argv)
    chosen = args.loss_type or args.regime
    if chosen is None:
        p.error("must pass --loss_type {baseline|alignment_loss}")
    if chosen == "alignment_loss":
        if not args.centroids_cache:
            p.error("--loss_type=alignment_loss requires --centroids_cache")
        if not args.h_baseline_cache:
            p.error("--loss_type=alignment_loss requires --h_baseline_cache")
    args.regime = chosen
    del args.loss_type
    return Config(**vars(args))


def to_dict(cfg: Config) -> dict:
    return asdict(cfg)
