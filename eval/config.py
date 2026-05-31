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
    n_disks: int = 4
    max_new_tokens: int = 1024
    temperature: float = 0.0
    probe_layers: List[int] = None       # filled by parse_args
    eval_subset_size: int = 500
    batch_size: int = 8
    seed: int = 42
    skip_solve: bool = False
    skip_probe: bool = False
    checkpoint_type: str = "auto"        # baseline | augmented_steer | auto
    hook_on: bool = False                # augmented_steer only: install hook for diagnostic eval (default OFF)
    alpha_override: Optional[float] = None  # augmented_steer only: override α for eval-time ablation


def _csv_ints(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_args(argv: Optional[List[str]] = None) -> EvalConfig:
    p = argparse.ArgumentParser(description="Evaluate a fine-tuned ToH checkpoint")
    p.add_argument("--checkpoint_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--data_dir", default="hanoi_data/")
    p.add_argument("--n_disks", type=int, choices=[3, 4, 5], default=4,
                   help="Number of disks. 4 is in-distribution (training); 3 and 5 are OOD.")
    p.add_argument("--max_new_tokens", type=int, default=0,
                   help="0 → auto-scale by n_disks (3→2048, 4→3072, 5→4096).")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--probe_layers", type=str, default="24,36,48")
    p.add_argument("--eval_subset_size", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=8,
                   help="Problems generated in parallel per generate() call.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip_solve", "--no_solve", action="store_true",
                   help="Skip the solve-rate sub-eval.")
    p.add_argument("--skip_probe", action="store_true",
                   help="Skip the fresh-probe quality sub-eval.")
    p.add_argument("--hook_on", action="store_true",
                   help="augmented_steer only: install the steering hook "
                        "during eval. Default OFF — the production eval should "
                        "reproduce baseline behavior.")
    p.add_argument("--alpha_override", type=float, default=None,
                   help="augmented_steer only: override α at eval time. "
                        "Useful for ablations (α=0 = no steering, α=10 = "
                        "magnified).")
    p.add_argument("--checkpoint_type",
                   choices=["auto", "baseline", "augmented_steer"],
                   default="auto",
                   help="Override the checkpoint label in the output table. "
                        "Auto-detect: regime field in config.json.")
    args = p.parse_args(argv)

    if args.max_new_tokens <= 0:
        args.max_new_tokens = {3: 2048, 4: 3072, 5: 4096}[args.n_disks]

    return EvalConfig(
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        n_disks=args.n_disks,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        probe_layers=_csv_ints(args.probe_layers),
        eval_subset_size=args.eval_subset_size,
        batch_size=args.batch_size,
        seed=args.seed,
        skip_solve=args.skip_solve,
        skip_probe=args.skip_probe,
        checkpoint_type=args.checkpoint_type,
        hook_on=args.hook_on,
        alpha_override=args.alpha_override,
    )
