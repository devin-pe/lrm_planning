"""Configuration and CLI parsing for ToH transformer training."""

from __future__ import annotations

import argparse
from dataclasses import dataclass


def default_model_hparams(n_disks: int) -> dict:
    if n_disks == 3:
        return {
            "n_layers": 4,
            "n_heads": 4,
            "d_model": 128,
            "d_ff": 512,
            "dropout": 0.1,
            "num_epochs": 200,
        }
    if n_disks == 4:
        return {
            "n_layers": 6,
            "n_heads": 4,
            "d_model": 128,
            "d_ff": 512,
            "dropout": 0.1,
            "num_epochs": 80,
        }
    raise ValueError("Only n_disks=3 or n_disks=4 are supported")


def max_seq_len_for_disks(n_disks: int) -> int:
    if n_disks == 3:
        return 40
    if n_disks == 4:
        return 96
    raise ValueError("Only n_disks=3 or n_disks=4 are supported")


@dataclass
class TrainConfig:
    n_disks: int
    n_layers: int
    n_heads: int
    d_model: int
    d_ff: int
    dropout: float
    batch_size: int
    lr: float
    min_lr: float
    beta1: float
    beta2: float
    weight_decay: float
    warmup_ratio: float
    grad_clip: float
    num_epochs: int
    seed: int
    checkpoint_dir: str
    save_every: int
    num_workers: int
    device: str
    max_seq_len: int


def parse_train_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train decoder-only transformer for flat-to-flat ToH")
    parser.add_argument("--n_disks", type=int, required=True, choices=[3, 4])

    parser.add_argument("--n_layers", type=int, default=None)
    parser.add_argument("--n_heads", type=int, default=None)
    parser.add_argument("--d_model", type=int, default=None)
    parser.add_argument("--d_ff", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.98)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.10)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_epochs", type=int, default=None)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint_dir", type=str, default="toh_transformer/checkpoints")
    parser.add_argument("--save_every", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    defaults = default_model_hparams(args.n_disks)
    n_layers = defaults["n_layers"] if args.n_layers is None else args.n_layers
    n_heads = defaults["n_heads"] if args.n_heads is None else args.n_heads
    d_model = defaults["d_model"] if args.d_model is None else args.d_model
    d_ff = defaults["d_ff"] if args.d_ff is None else args.d_ff
    dropout = defaults["dropout"] if args.dropout is None else args.dropout
    num_epochs = defaults["num_epochs"] if args.num_epochs is None else args.num_epochs

    if d_model % n_heads != 0:
        raise ValueError("d_model must be divisible by n_heads")

    return TrainConfig(
        n_disks=args.n_disks,
        n_layers=n_layers,
        n_heads=n_heads,
        d_model=d_model,
        d_ff=d_ff,
        dropout=dropout,
        batch_size=args.batch_size,
        lr=args.lr,
        min_lr=args.min_lr,
        beta1=args.beta1,
        beta2=args.beta2,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        grad_clip=args.grad_clip,
        num_epochs=num_epochs,
        seed=args.seed,
        checkpoint_dir=args.checkpoint_dir,
        save_every=args.save_every,
        num_workers=args.num_workers,
        device=args.device,
        max_seq_len=max_seq_len_for_disks(args.n_disks),
    )
