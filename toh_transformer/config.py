"""Configuration and CLI parsing for ToH transformer training."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List


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
    if n_disks == 5:
        return 96
    if n_disks == 6:
        return 96
    raise ValueError("Only n_disks in {3,4,5,6} are supported")


@dataclass
class TrainConfig:
    n_disks: int
    train_n_disks: List[int]
    test_n_disks: List[int]
    val_ratio: float
    data_mode: str
    tower_data_strategy: str
    tower_train_repeats: int
    tower_test_repeats: int
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
    eval_every: int
    seq_eval_max_examples: int
    seq_eval_max_train_examples: int
    seq_eval_include_train: bool
    seed: int
    checkpoint_dir: str
    save_every: int
    num_workers: int
    device: str
    max_seq_len: int


def parse_train_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train decoder-only transformer for flat-to-flat ToH")
    parser.add_argument("--n_disks", type=int, default=4, choices=[3, 4, 5, 6])
    parser.add_argument(
        "--train_n_disks",
        type=int,
        nargs="+",
        default=None,
        help="Optional multi-n train split. Example: --train_n_disks 3 4 6",
    )
    parser.add_argument(
        "--test_n_disks",
        type=int,
        nargs="+",
        default=None,
        help="Optional held-out test split. Example: --test_n_disks 5",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.2,
        help="Validation split ratio used per train n when --train_n_disks is provided",
    )
    parser.add_argument(
        "--data_mode",
        type=str,
        default="flat-to-flat",
        choices=["flat-to-flat", "tower-to-tower"],
        help="Dataset mode: all-pairs flat-to-flat or canonical tower-to-tower",
    )
    parser.add_argument(
        "--tower_data_strategy",
        type=str,
        default="canonical-repeat",
        choices=["canonical-repeat", "recursive-states"],
        help="For tower-to-tower mode: repeated canonical task or recursive trajectory subproblems",
    )
    parser.add_argument(
        "--tower_train_repeats",
        type=int,
        default=1024,
        help="For tower-to-tower mode, number of repeated examples per train n",
    )
    parser.add_argument(
        "--tower_test_repeats",
        type=int,
        default=1,
        help="For tower-to-tower mode, number of repeated examples per test n",
    )

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
    parser.add_argument("--eval_every", type=int, default=1, help="Run validation/test metrics every N epochs")
    parser.add_argument(
        "--seq_eval_max_examples",
        type=int,
        default=2048,
        help=(
            "Max examples used per split for autoregressive sequence-accuracy eval. "
            "Use <=0 to evaluate the full split."
        ),
    )
    parser.add_argument(
        "--seq_eval_max_train_examples",
        type=int,
        default=256,
        help=(
            "Max training examples used for train sequence-accuracy eval. "
            "Use <=0 to reuse --seq_eval_max_examples."
        ),
    )
    parser.add_argument(
        "--seq_eval_include_train",
        action="store_true",
        help="Also compute sequence accuracy on the training split (expensive for large datasets)",
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint_dir", type=str, default="toh_transformer/checkpoints")
    parser.add_argument("--save_every", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    if args.train_n_disks is not None:
        train_n_disks = sorted(set(int(n) for n in args.train_n_disks))
        invalid = [n for n in train_n_disks if n not in {3, 4, 5, 6}]
        if invalid:
            raise ValueError(f"Unsupported train_n_disks values: {invalid}. Allowed: 3,4,5,6")
    else:
        train_n_disks = [int(args.n_disks)]

    if args.test_n_disks is not None:
        test_n_disks = sorted(set(int(n) for n in args.test_n_disks))
        invalid = [n for n in test_n_disks if n not in {3, 4, 5, 6}]
        if invalid:
            raise ValueError(f"Unsupported test_n_disks values: {invalid}. Allowed: 3,4,5,6")
    else:
        test_n_disks = []

    if not (0.0 <= args.val_ratio < 1.0):
        raise ValueError("val_ratio must satisfy 0 <= val_ratio < 1")
    if args.eval_every < 1:
        raise ValueError("eval_every must be >= 1")
    if args.seq_eval_max_examples == 0:
        args.seq_eval_max_examples = -1
    if args.seq_eval_max_train_examples == 0:
        args.seq_eval_max_train_examples = -1
    if args.tower_train_repeats < 1:
        raise ValueError("tower_train_repeats must be >= 1")
    if args.tower_test_repeats < 1:
        raise ValueError("tower_test_repeats must be >= 1")

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
        train_n_disks=train_n_disks,
        test_n_disks=test_n_disks,
        val_ratio=args.val_ratio,
        data_mode=args.data_mode,
        tower_data_strategy=args.tower_data_strategy,
        tower_train_repeats=args.tower_train_repeats,
        tower_test_repeats=args.tower_test_repeats,
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
        eval_every=args.eval_every,
        seq_eval_max_examples=args.seq_eval_max_examples,
        seq_eval_max_train_examples=args.seq_eval_max_train_examples,
        seq_eval_include_train=bool(args.seq_eval_include_train),
        seed=args.seed,
        checkpoint_dir=args.checkpoint_dir,
        save_every=args.save_every,
        num_workers=args.num_workers,
        device=args.device,
        max_seq_len=max(max_seq_len_for_disks(n) for n in train_n_disks + test_n_disks),
    )
