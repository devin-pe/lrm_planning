"""Top-level evaluation: runs solve + probe sub-evaluations and writes
summary.json plus a human-readable comparison block to stdout.
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


def _test_file_for(data_dir: Path, n_disks: int) -> Path:
    """4-disk keeps the legacy `test.jsonl` name; OOD uses `test_n{N}.jsonl`."""
    return Path(data_dir) / ("test.jsonl" if n_disks == 4 else f"test_n{n_disks}.jsonl")


def _load_test(data_dir: Path, n_disks: int = 4) -> List[Dict]:
    rows: List[Dict] = []
    p = _test_file_for(data_dir, n_disks)
    if not p.exists():
        raise FileNotFoundError(
            f"Test file not found: {p}. Generate it with "
            f"`python -m hanoi_data.generate_dataset --n_disks {n_disks}`."
        )
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
    lines.append(f"  Checkpoint type:   {summary.get('checkpoint_type', '?')}")
    lines.append(f"  Regime:            {train_cfg.get('regime')}")
    in_dist = " (in-distribution)" if cfg.n_disks == 4 else " (OOD)"
    lines.append(f"  n_disks:           {cfg.n_disks}{in_dist}")
    lines.append(f"  max_new_tokens:    {cfg.max_new_tokens}")
    augmented = summary.get("augmented_steer")
    if augmented:
        lines.append(f"  Augmented steer:   trained α = {augmented.get('trained_alpha')}  "
                     f"layer={augmented.get('probe_layer')}  "
                     f"({'HOOK ON' if augmented.get('hook_active') else 'HOOK OFF (production)'})")
        if augmented.get("effective_alpha") is not None and \
                augmented["effective_alpha"] != augmented.get("trained_alpha"):
            lines.append(f"  α override:        {augmented['effective_alpha']}")
        lines.append(f"  Baseline ckpt:     {augmented.get('baseline_checkpoint')}")
    lines.append("-" * 65)

    solve = summary.get("solve")
    if solve:
        n = solve["n"]
        cc = solve["category_counts"]
        lines.append(f"SOLVE RATE (n={n})")
        for k in ("Optimal", "Suboptimal", "Incorrect", "Illegal_format", "Illegal_moves"):
            v = cc.get(k, 0)
            lines.append(f"  {k:<15} {v:4d} / {n:<4d}  ({_pct(v, n)})")
        nt = solve.get("n_truncated", 0)
        nfi = solve.get("n_format_invalid", 0)
        if nt or nfi:
            lines.append(f"  (max_new_tokens={solve.get('max_new_tokens', '?')}  "
                         f"truncated={nt}  format_invalid={nfi})")

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

    lines.append("=" * 65)
    return "\n".join(lines)


def main() -> None:
    cfg = parse_args()
    _seed_everything(cfg.seed)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[run_eval] loading checkpoint from {cfg.checkpoint_dir}")
    tokenizer, model, train_cfg = load_checkpoint(cfg.checkpoint_dir)
    augmented_info = getattr(model, "_augmented_info", None)

    # Determine checkpoint_type label for the output table.
    if cfg.checkpoint_type != "auto":
        ckpt_type = cfg.checkpoint_type
    elif augmented_info is not None:
        ckpt_type = "augmented_steer"
    else:
        ckpt_type = "baseline"
    print(f"[run_eval] checkpoint_type = {ckpt_type}")

    # augmented_steer: hook is OFF by default. --hook_on installs and applies
    # at every supervised position in subsequent eval forwards. Note this only
    # affects passes that go through model() with sup_pos set — solve eval is
    # autoregressive and doesn't set sup_pos, so the hook stays a no-op there.
    if augmented_info is not None:
        from finetune.augmented_steer import (
            install_hook as aug_install_hook, load_directions_cache,
            set_alpha as aug_set_alpha, set_directions as aug_set_directions,
        )
        from finetune.train import _param_for_layer
        eff_alpha = (cfg.alpha_override if cfg.alpha_override is not None
                     else augmented_info["trained_alpha"])
        if cfg.hook_on:
            directions, _gm, _raw = load_directions_cache(
                augmented_info["steering_directions_path"]
            )
            aug_set_directions(directions)
            aug_set_alpha(eff_alpha)
            target_block = _param_for_layer(model, augmented_info["probe_layer"])
            handle = aug_install_hook(target_block)
            augmented_info["hook_handle"] = handle
            augmented_info["effective_alpha"] = eff_alpha
            print(f"[run_eval] augmented_steer: --hook_on with α={eff_alpha}  "
                  f"({len(directions)} states cached)")
        else:
            print(f"[run_eval] augmented_steer: hook OFF (production case). "
                  f"Pass --hook_on for the diagnostic eval.")

    # Context-window sanity. DeepSeek-R1-Distill-Qwen-32B has a 32k context.
    try:
        max_ctx = int(getattr(model.config, "max_position_embeddings", 32768))
    except Exception:
        max_ctx = 32768
    _ctx_safe_budget = int(0.8 * max_ctx)
    _est_prompt_tokens = 1024
    if _est_prompt_tokens + cfg.max_new_tokens > _ctx_safe_budget:
        print(f"[run_eval] WARN: prompt(~{_est_prompt_tokens}) + max_new_tokens"
              f"({cfg.max_new_tokens}) = {_est_prompt_tokens + cfg.max_new_tokens}"
              f" exceeds 80% of context ({_ctx_safe_budget} / {max_ctx}). "
              f"Lower max_new_tokens or expect KV-cache spills.")
    else:
        print(f"[run_eval] context fit OK: prompt(~{_est_prompt_tokens}) + "
              f"max_new_tokens({cfg.max_new_tokens}) = "
              f"{_est_prompt_tokens + cfg.max_new_tokens} ≤ 80% of {max_ctx}")

    test_rows = _load_test(Path(cfg.data_dir), n_disks=cfg.n_disks)
    sol_lens = [len(r["moves"]) for r in test_rows]
    if sol_lens:
        from collections import Counter as _C
        bucket = _C(sol_lens)
        dist = "  ".join(f"L={L}:{bucket[L]}" for L in sorted(bucket))
        print(f"[run_eval] loaded {len(test_rows)} test rows  (n_disks={cfg.n_disks})")
        print(f"[run_eval] optimal-length distribution: {dist}")
        print(f"[run_eval] max_new_tokens = {cfg.max_new_tokens}  (auto-scaled by n_disks)")
    else:
        print(f"[run_eval] loaded {len(test_rows)} test rows  (n_disks={cfg.n_disks})")

    summary: Dict = {
        "checkpoint_dir": cfg.checkpoint_dir,
        "checkpoint_type": ckpt_type,
        "train_config": train_cfg,
        "eval_config": cfg.__dict__,
    }
    if augmented_info is not None:
        summary["augmented_steer"] = {
            "trained_alpha": augmented_info["trained_alpha"],
            "effective_alpha": augmented_info.get("effective_alpha"),
            "probe_layer": augmented_info["probe_layer"],
            "baseline_checkpoint": augmented_info["baseline_checkpoint"],
            "steering_directions_path": augmented_info["steering_directions_path"],
            "hook_active": bool(cfg.hook_on),
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
            # Probe eval forwards with output_hidden_states=True, which holds
            # all per-layer activations in memory. Halve the batch_size to
            # avoid OOM on a single GPU at n_disks=5 with max_new_tokens=4096.
            probe_bs = max(1, cfg.batch_size // 4)
            summary["probe"] = run_probe_eval(
                model, tokenizer, None,
                train_layer=int(train_cfg.get("probe_layer") or 36),
                probe_layers=cfg.probe_layers,
                test_rows=test_rows,
                max_new_tokens=cfg.max_new_tokens,
                out_dir=out_dir,
                batch_size=probe_bs,
                n_disks=cfg.n_disks,
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
