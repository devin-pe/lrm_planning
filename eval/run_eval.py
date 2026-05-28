"""Top-level evaluation: runs solve, probe, and steer sub-evaluations and
writes summary.json + a human-readable comparison block to stdout.
"""

from __future__ import annotations

import json
import random
import sys
import traceback
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from eval.config import EvalConfig, parse_args  # noqa: E402
from eval.load import load_checkpoint  # noqa: E402


def _load_test(data_dir: Path) -> List[Dict]:
    rows: List[Dict] = []
    p = Path(data_dir) / "test.jsonl"
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _pct(num: int, denom: int) -> str:
    if denom == 0:
        return "0.0%"
    return f"{100.0 * num / denom:.1f}%"


def _print_summary(cfg: EvalConfig, train_cfg: Dict, summary: Dict) -> str:
    lines: List[str] = []
    lines.append("=" * 65)
    lines.append("EVALUATION SUMMARY")
    lines.append(f"  Checkpoint:        {cfg.checkpoint_dir}")
    lines.append(f"  Regime:            {train_cfg.get('regime')}")
    if train_cfg.get("regime") == "probe":
        lines.append(f"  Lambda:            {train_cfg.get('lambda_geom')}")
        lines.append(f"  Probe layer:       {train_cfg.get('layer')}")
    lines.append("-" * 65)

    solve = summary.get("solve")
    if solve:
        n = solve["n"]
        cc = solve["category_counts"]
        lines.append(f"SOLVE RATE (n={n})")
        for k in ("Optimal", "Suboptimal", "Incorrect", "Illegal"):
            v = cc.get(k, 0)
            lines.append(f"  {k:<12} {v:4d} / {n:<4d}  ({_pct(v, n)})")

    probe = summary.get("probe")
    if probe:
        lines.append("-" * 65)
        lines.append("PROBE QUALITY (fresh per-layer linear probes)")
        for key in sorted(probe["fresh"].keys()):
            m = probe["fresh"][key]
            if m.get("spearman") is None:
                lines.append(f"  {key:<10}  -- insufficient samples --")
                continue
            lines.append(f"  {key:<10}  rho={m['spearman']:.4f}  r={m['pearson']:.4f}  "
                         f"adj={m['adj_acc']:.3f}  n={m['n_states']}")
        if probe["trained_head"]:
            lines.append("-" * 65)
            lines.append(f"TRAINING-TIME PROBE (at layer {probe['train_layer']})")
            for key in sorted(probe["trained_head"].keys()):
                m = probe["trained_head"][key]
                lines.append(f"  {key:<10}  rho={m['spearman']:.4f}  r={m['pearson']:.4f}  "
                             f"adj={m['adj_acc']:.3f}  n={m['n_states']}")

    lines.append("=" * 65)
    return "\n".join(lines)


def main() -> None:
    cfg = parse_args()
    _seed_everything(cfg.seed)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[run_eval] loading checkpoint from {cfg.checkpoint_dir}")
    tokenizer, model, probe_head, train_cfg = load_checkpoint(cfg.checkpoint_dir)

    test_rows = _load_test(Path(cfg.data_dir))
    print(f"[run_eval] loaded {len(test_rows)} test rows")

    summary: Dict = {
        "checkpoint_dir": cfg.checkpoint_dir,
        "train_config": train_cfg,
        "eval_config": cfg.__dict__,
    }

    failed = False

    if not cfg.skip_solve:
        try:
            from eval.solve import run_solve_eval
            print("\n[run_eval] === SOLVE ===")
            summary["solve"] = run_solve_eval(
                model, tokenizer, test_rows,
                max_new_tokens=cfg.max_new_tokens,
                subset_size=cfg.eval_subset_size,
                seed=cfg.seed,
                out_dir=out_dir,
                batch_size=cfg.batch_size,
            )
        except Exception:
            traceback.print_exc()
            failed = True

    if not cfg.skip_probe:
        try:
            from eval.probe_eval import run_probe_eval
            print("\n[run_eval] === PROBE EVAL ===")
            summary["probe"] = run_probe_eval(
                model, tokenizer, probe_head,
                train_layer=int(train_cfg.get("layer", 36)),
                probe_layers=cfg.probe_layers,
                test_rows=test_rows,
                max_new_tokens=cfg.max_new_tokens,
                out_dir=out_dir,
                batch_size=cfg.batch_size,
            )
        except Exception:
            traceback.print_exc()
            failed = True

    pretty = _print_summary(cfg, train_cfg, summary)
    print("\n" + pretty)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    (out_dir / "summary.txt").write_text(pretty)
    print(f"\n[run_eval] wrote {out_dir / 'summary.json'}")

    if failed:
        sys.exit(2)


if __name__ == "__main__":
    main()
