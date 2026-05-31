"""Build the augmented-steer steering-directions cache.

Math (per-state s):
    h_target[s]    = mean L36 activation across positions where post-move
                     state was s   (already computed by precompute_canonical_states)
    h_global_mean  = (Σ_s count[s] * h_target[s]) / Σ_s count[s]
                   = mean over ALL individual collected activations
    direction[s]   = (h_target[s] - h_global_mean) / ‖h_target[s] - h_global_mean‖

We reuse the existing canonical cache when it already exists — `h_target[s]`
and `count[s]` are exactly what `precompute_canonical_states.py` stores.
Avoids re-running the (slow) forward-pass collection.

If the canonical cache is missing, we delegate to
`precompute_canonical_states.main` to build it first.

Usage:
    python -m finetune.precompute_steering_directions \\
        --canonical_cache runs/canonical_steer_L36/hidden_states_by_state.pt \\
        --output runs/augmented_steer_L36/steering_directions.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--canonical_cache", required=True,
                   help="Path to hidden_states_by_state.pt (from "
                        "precompute_canonical_states.py).")
    p.add_argument("--output", required=True,
                   help="Where to write steering_directions.pt.")
    p.add_argument("--recompute", action="store_true",
                   help="Overwrite existing output file.")
    args = p.parse_args()

    out_path = Path(args.output)
    if out_path.exists() and not args.recompute:
        print(f"[precompute-steer] {out_path} already exists — pass --recompute to overwrite")
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cache_path = Path(args.canonical_cache)
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Canonical cache not found at {cache_path}. "
            f"Build it first with `python -m finetune.precompute_canonical_states ...`."
        )
    print(f"[precompute-steer] loading canonical cache: {cache_path}")
    raw = torch.load(cache_path, map_location="cpu", weights_only=False)
    if not isinstance(raw, dict) or "states" not in raw:
        raise ValueError(f"unexpected canonical cache shape at {cache_path}: keys={list(raw.keys()) if isinstance(raw, dict) else type(raw)}")
    means: Dict[Tuple[int, ...], torch.Tensor] = {
        tuple(int(x) for x in k): v.detach().to(torch.float32).cpu()
        for k, v in raw["states"].items()
    }
    counts: Dict[Tuple[int, ...], int] = {
        tuple(int(x) for x in k): int(v) for k, v in raw.get("counts", {}).items()
    }
    if not counts or set(counts) != set(means):
        # If counts are missing, fall back to a uniform-weighted mean.
        print(f"[precompute-steer] counts missing or mismatched ({len(counts)} vs "
              f"{len(means)} states) — using uniform mean instead.")
        counts = {k: 1 for k in means}

    n_states = len(means)
    hidden_dim = next(iter(means.values())).shape[-1]
    print(f"[precompute-steer] {n_states} states  hidden_dim={hidden_dim}  "
          f"probe_layer={raw.get('probe_layer')}")

    # h_global_mean = weighted average of h_target[s] by counts[s].
    total_count = sum(counts.values())
    weighted_sum = torch.zeros(hidden_dim, dtype=torch.float32)
    for s, h in means.items():
        weighted_sum += counts[s] * h
    global_mean = weighted_sum / max(total_count, 1)

    # Unit-norm directions per state.
    directions: Dict[Tuple[int, ...], torch.Tensor] = {}
    pre_norms = []
    for s, h in means.items():
        diff = h - global_mean
        n = float(diff.norm())
        pre_norms.append(n)
        if n < 1e-8:
            print(f"[precompute-steer] ⚠ state {s}: ‖h_target - h_global_mean‖={n:.2e} "
                  f"is near zero — direction is ill-defined; using zero vector")
            directions[s] = torch.zeros_like(diff)
        else:
            directions[s] = diff / n

    pre_norms_t = torch.tensor(pre_norms)
    unit_norms = torch.stack([d.norm() for d in directions.values()])
    print(f"[precompute-steer] ‖h_target[s] - h_global_mean‖: "
          f"mean={pre_norms_t.mean():.2f}  std={pre_norms_t.std():.2f}  "
          f"min={pre_norms_t.min():.2f}  max={pre_norms_t.max():.2f}")
    print(f"[precompute-steer] ‖direction[s]‖ (should all be 1.0): "
          f"mean={unit_norms.mean():.4f}  min={unit_norms.min():.4f}  "
          f"max={unit_norms.max():.4f}")
    print(f"[precompute-steer] ‖h_global_mean‖ = {float(global_mean.norm()):.2f}")

    torch.save({
        "directions": directions,
        "global_mean": global_mean,
        "probe_layer": raw.get("probe_layer"),
        "baseline_checkpoint": raw.get("baseline_checkpoint"),
        "canonical_cache_source": str(cache_path),
        "n_states": n_states,
        "hidden_dim": hidden_dim,
    }, out_path)
    out_path.with_suffix(".json").write_text(json.dumps({
        "n_states": n_states,
        "hidden_dim": hidden_dim,
        "probe_layer": raw.get("probe_layer"),
        "canonical_cache_source": str(cache_path),
        "global_mean_norm": float(global_mean.norm()),
        "direction_norms": {
            "mean": float(unit_norms.mean()),
            "min":  float(unit_norms.min()),
            "max":  float(unit_norms.max()),
        },
        "pre_norm_stats": {
            "mean": float(pre_norms_t.mean()),
            "std":  float(pre_norms_t.std()),
            "min":  float(pre_norms_t.min()),
            "max":  float(pre_norms_t.max()),
        },
    }, indent=2))
    print(f"[precompute-steer] saved → {out_path}")


if __name__ == "__main__":
    main()
