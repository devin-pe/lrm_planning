"""CLI for the per-checkpoint evaluation pipeline."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class EvalConfig:
    checkpoint_dir: str
    output_dir: str
    data_dir: str = "hanoi_data/"
    max_new_tokens: int = 1024
    temperature: float = 0.0
    probe_layers: List[int] = None       # filled by parse_args
    eval_subset_size: int = 500
    batch_size: int = 8
    seed: int = 42
    skip_solve: bool = False
    skip_probe: bool = False


def _csv_ints(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_args(argv: Optional[List[str]] = None) -> EvalConfig:
    p = argparse.ArgumentParser(description="Evaluate a fine-tuned ToH checkpoint")
    p.add_argument("--checkpoint_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--data_dir", default="hanoi_data/")
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--probe_layers", type=str, default="24,36,48")
    p.add_argument("--eval_subset_size", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=8,
                   help="Problems generated in parallel per generate() call.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip_solve", action="store_true")
    p.add_argument("--skip_probe", action="store_true")
    args = p.parse_args(argv)

    return EvalConfig(
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        probe_layers=_csv_ints(args.probe_layers),
        eval_subset_size=args.eval_subset_size,
        batch_size=args.batch_size,
        seed=args.seed,
        skip_solve=args.skip_solve,
        skip_probe=args.skip_probe,
    )
