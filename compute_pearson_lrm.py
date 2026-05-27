"""Recompute Spearman + Pearson for the LRM SEP probes from saved artifacts.

Reads `outputs/<dir>/hidden_states.pt` and `outputs/<dir>/probe_layer_*.pt`,
applies each saved 2D linear probe to the saved hidden states, and writes a
`pearson_spearman_summary.json` next to them — no GPU, no model load, no
retraining.
"""

from __future__ import annotations

import json
import math
import sys
from collections import deque
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np
import torch
from scipy.stats import spearmanr

State = Tuple[int, ...]


def top_disk_per_peg(state: State, n_disks: int):
    tops: List = [None, None, None]
    for disk in range(1, n_disks + 1):
        peg = state[disk - 1]
        if tops[peg] is None:
            tops[peg] = disk
    return tops


def legal_neighbors(state: State, n_disks: int) -> List[State]:
    tops = top_disk_per_peg(state, n_disks)
    out: List[State] = []
    for src in range(3):
        src_top = tops[src]
        if src_top is None:
            continue
        for dst in range(3):
            if src == dst:
                continue
            dst_top = tops[dst]
            if dst_top is None or src_top < dst_top:
                nxt = list(state)
                nxt[src_top - 1] = dst
                out.append(tuple(int(x) for x in nxt))
    return out


def build_distance_matrix(states: Sequence[State], n_disks: int) -> np.ndarray:
    n = len(states)
    state_to_idx = {s: i for i, s in enumerate(states)}
    adj: List[List[int]] = [[] for _ in range(n)]
    for i, s in enumerate(states):
        for nbr in legal_neighbors(s, n_disks=n_disks):
            adj[i].append(state_to_idx[nbr])

    dist = np.full((n, n), np.inf, dtype=np.float64)
    for src in range(n):
        dist[src, src] = 0.0
        q = deque([src])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if np.isinf(dist[src, v]):
                    dist[src, v] = dist[src, u] + 1.0
                    q.append(v)
    if np.isinf(dist).any():
        raise RuntimeError("Disconnected states")
    return dist


def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    x = x.astype(np.float64) - x.mean()
    y = y.astype(np.float64) - y.mean()
    denom = math.sqrt(float(np.sum(x * x) * np.sum(y * y)))
    return float(np.sum(x * y) / denom) if denom > 0 else 0.0


def apply_probe(state_dict: Dict[str, torch.Tensor], hidden: torch.Tensor) -> np.ndarray:
    W = state_dict["weight"].to(torch.float32)  # (2, d)
    b = state_dict["bias"].to(torch.float32)
    coords = hidden.to(torch.float32) @ W.T + b
    return coords.detach().cpu().numpy().astype(np.float64)


def evaluate_dir(out_dir: Path, n_disks: int = 4) -> Dict:
    hidden_path = out_dir / "hidden_states.pt"
    if not hidden_path.exists():
        raise FileNotFoundError(hidden_path)

    bundle = torch.load(hidden_path, map_location="cpu", weights_only=False)
    layer_ids: List[int] = list(bundle["layer_ids"])
    states: List[State] = [tuple(int(x) for x in s) for s in bundle["states"]]
    hidden_by_layer: Dict[int, torch.Tensor] = bundle["hidden_states"]

    dist = build_distance_matrix(states, n_disks=n_disks)
    triu_i, triu_j = np.triu_indices(len(states), k=1)
    true_pairs = dist[triu_i, triu_j]

    rows: List[Dict] = []
    for layer in layer_ids:
        probe_path = out_dir / f"probe_layer_{layer}.pt"
        if not probe_path.exists():
            print(f"[WARN] missing {probe_path}", file=sys.stderr)
            continue
        payload = torch.load(probe_path, map_location="cpu", weights_only=False)
        coords = apply_probe(payload["state_dict"], hidden_by_layer[layer])

        pred_dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
        pred_pairs = pred_dist[triu_i, triu_j]

        rho, _ = spearmanr(pred_pairs, true_pairs)
        rho = float(rho) if np.isfinite(rho) else float("nan")
        pearson = pearson_r(pred_pairs, true_pairs)

        rows.append({
            "layer": int(layer),
            "spearman_rho": rho,
            "pearson_r": pearson,
            "saved_spearman_rho": float(payload.get("spearman_rho", float("nan"))),
        })

    rows_sorted = sorted(rows, key=lambda r: r["spearman_rho"], reverse=True)
    summary = {
        "model_name": bundle.get("model_name"),
        "layer_ids": layer_ids,
        "by_layer_sorted_by_spearman": rows_sorted,
        "best": rows_sorted[0] if rows_sorted else None,
    }
    out_path = out_dir / "pearson_spearman_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"[INFO] Wrote {out_path}")
    return summary


def main() -> None:
    targets = [
        Path("outputs/qwen_probe"),
        Path("outputs/deepseek_r1_qwen32b_probe"),
    ]
    for d in targets:
        print(f"\n=== {d} ===")
        s = evaluate_dir(d, n_disks=4)
        print(f"{'layer':>5} | {'spearman':>10} | {'pearson':>10}")
        print("-" * 36)
        for r in s["by_layer_sorted_by_spearman"]:
            print(f"{r['layer']:5d} | {r['spearman_rho']:10.4f} | {r['pearson_r']:10.4f}")


if __name__ == "__main__":
    main()
