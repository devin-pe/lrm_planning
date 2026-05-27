#!/usr/bin/env python3
"""Visualize which of the 81 ToH states Qwen3-27B solved on the Sierpinski probe projection.

Loads the layer-36 distance-matching probe (or retrains it), overlays solved/unsolved
status, and highlights the standard tower-to-tower optimal path (0000→2222).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from itertools import product
from pathlib import Path
from typing import Dict, List, Set, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

State = Tuple[int, ...]


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Highlight solved states on Sierpinski probe projection")
    p.add_argument("--probe_dir", default="outputs/qwen_probe",
                   help="Directory containing hidden_states.pt and probe_layer_N.pt")
    p.add_argument("--gen_results", default=None,
                   help="Path to validation_results.json (default: probe_dir/validation_results.json)")
    p.add_argument("--eval_results", default=None,
                   help="Path to evaluate_results.json (or legacy evaluate_qwen_results.json); overrides --gen_results")
    p.add_argument("--n_disks", type=int, default=4)
    p.add_argument("--layer", type=int, default=36)
    p.add_argument("--output_dir", default=None,
                   help="Where to save the figure (default: same as probe_dir)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ── State graph ───────────────────────────────────────────────────────────────

def enumerate_states(n_disks: int) -> List[State]:
    return [tuple(s) for s in product(range(3), repeat=n_disks)]


def state_label(state: State) -> str:
    return "".join(str(x) for x in state)


def top_disk_per_peg(state: State) -> List[int | None]:
    tops: List[int | None] = [None, None, None]
    for disk_idx, peg in enumerate(state):
        disk = disk_idx + 1  # disk numbers are 1-based (disk 1 = smallest)
        if tops[peg] is None:
            tops[peg] = disk
    return tops


def legal_neighbors(state: State) -> List[State]:
    tops = top_disk_per_peg(state)
    out: List[State] = []
    for src in range(3):
        if tops[src] is None:
            continue
        src_top = tops[src]
        for dst in range(3):
            if src == dst:
                continue
            dst_top = tops[dst]
            if dst_top is None or src_top < dst_top:
                nxt = list(state)
                nxt[src_top - 1] = dst
                out.append(tuple(nxt))
    return out


def build_graph(states: List[State]) -> Tuple[np.ndarray, List[Tuple[int, int]], Dict[int, Set[int]]]:
    n = len(states)
    idx = {s: i for i, s in enumerate(states)}
    adj: List[List[int]] = [[] for _ in range(n)]
    edge_set: Set[Tuple[int, int]] = set()
    nbr_sets: Dict[int, Set[int]] = {i: set() for i in range(n)}

    for i, s in enumerate(states):
        for nb in legal_neighbors(s):
            j = idx[nb]
            adj[i].append(j)
            nbr_sets[i].add(j)
            edge_set.add((min(i, j), max(i, j)))

    dist = np.full((n, n), np.inf, dtype=np.float32)
    for src in range(n):
        dist[src, src] = 0.0
        q = deque([src])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if np.isinf(dist[src, v]):
                    dist[src, v] = dist[src, u] + 1.0
                    q.append(v)

    return dist, sorted(edge_set), nbr_sets


def tower_to_tower_path(n_disks: int) -> List[State]:
    """BFS path from (0,...,0) to (2,...,2); returns ordered list of states."""
    start: State = tuple([0] * n_disks)
    goal: State = tuple([2] * n_disks)
    parent: Dict[State, State | None] = {start: None}
    q: deque[State] = deque([start])
    while q:
        curr = q.popleft()
        for nb in legal_neighbors(curr):
            if nb in parent:
                continue
            parent[nb] = curr
            if nb == goal:
                path = []
                s: State | None = goal
                while s is not None:
                    path.append(s)
                    s = parent[s]
                return list(reversed(path))
            q.append(nb)
    raise RuntimeError("No path found from start to goal")


# ── Probe ─────────────────────────────────────────────────────────────────────

def normalize_dist(dist: np.ndarray) -> np.ndarray:
    mu, sigma = dist.mean(), dist.std()
    if sigma < 1e-8:
        sigma = 1.0
    return ((dist - mu) / sigma).astype(np.float32)


def load_or_retrain_probe(
    probe_dir: Path,
    layer: int,
    states: List[State],
    dist: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Return (81, 2) probe coordinates for the given layer."""
    hidden_path = probe_dir / "hidden_states.pt"
    probe_path = probe_dir / f"probe_layer_{layer}.pt"

    if not hidden_path.exists():
        raise FileNotFoundError(f"Missing {hidden_path}")

    hs_data = torch.load(hidden_path, map_location="cpu", weights_only=False)
    hs_by_layer: Dict[int, torch.Tensor] = hs_data["hidden_states"]

    if layer not in hs_by_layer:
        available = list(hs_by_layer.keys())
        raise KeyError(f"Layer {layer} not found. Available: {available}")

    x = hs_by_layer[layer].float()  # (81, d_model)
    d_model = x.shape[1]
    probe = nn.Linear(d_model, 2, bias=True)

    if probe_path.exists():
        probe_data = torch.load(probe_path, map_location="cpu", weights_only=False)
        probe.load_state_dict(probe_data["state_dict"])
        rho = probe_data.get("spearman_rho", float("nan"))
        print(f"[INFO] Loaded probe_layer_{layer}.pt  (spearman={rho:.4f})")
    else:
        print(f"[INFO] probe_layer_{layer}.pt not found — retraining probe...")
        torch.manual_seed(seed)
        dist_norm = torch.tensor(normalize_dist(dist))
        opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
        for ep in range(1, 2001):
            opt.zero_grad(set_to_none=True)
            z = probe(x)
            pred = torch.cdist(z, z, p=2)
            loss = ((pred - dist_norm) ** 2).mean()
            loss.backward()
            opt.step()
            if ep % 400 == 0 or ep == 2000:
                print(f"  epoch {ep:4d}  loss={loss.item():.6f}")

    probe.eval()
    with torch.no_grad():
        coords = probe(x).numpy().astype(np.float32)
    return coords


# ── Analysis helpers ──────────────────────────────────────────────────────────

def min_path_dist(state_i: int, path_indices: Set[int], dist: np.ndarray) -> float:
    return float(min(dist[state_i, p] for p in path_indices))


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    probe_dir = Path(args.probe_dir)
    gen_results_path = (
        Path(args.gen_results) if args.gen_results
        else probe_dir / "validation_results.json"
    )
    out_dir = Path(args.output_dir) if args.output_dir else probe_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ── Build state graph ─────────────────────────────────────────────────────
    states = enumerate_states(args.n_disks)
    state_idx = {s: i for i, s in enumerate(states)}
    print(f"[INFO] {len(states)} states, building graph...")
    dist, edges, nbr_sets = build_graph(states)

    # ── Probe coordinates ─────────────────────────────────────────────────────
    coords = load_or_retrain_probe(probe_dir, args.layer, states, dist, args.seed)

    # ── Generation / evaluation results ──────────────────────────────────────
    def _eval_path_default(d: Path) -> Path:
        """Return evaluate_results.json, falling back to the legacy filename."""
        p = d / "evaluate_results.json"
        return p if p.exists() else d / "evaluate_qwen_results.json"

    eval_path = Path(args.eval_results) if args.eval_results else _eval_path_default(probe_dir)
    gen_results_path = (
        Path(args.gen_results) if args.gen_results
        else probe_dir / "validation_results.json"
    )

    optimal_set: Set[State] = set()
    suboptimal_set: Set[State] = set()

    if eval_path.exists():
        with open(eval_path) as f:
            eval_data = json.load(f)
        for r in eval_data["records"]:
            s = tuple(r["state_tuple"])
            if r["category"] == "CORRECT_OPTIMAL":
                optimal_set.add(s)
            elif r["category"] == "CORRECT_SUBOPTIMAL":
                suboptimal_set.add(s)
        print(f"[INFO] Loaded from {eval_path.name}: "
              f"{len(optimal_set)} optimal + {len(suboptimal_set)} suboptimal")
    elif gen_results_path.exists():
        with open(gen_results_path) as f:
            val_data = json.load(f)
        for r in val_data:
            if r["passed"]:
                optimal_set.add(tuple(r["state_tuple"]))
        print(f"[INFO] Loaded from validation_results.json: {len(optimal_set)} solved")
    else:
        raise FileNotFoundError(
            f"Neither {eval_path} nor {gen_results_path} found. "
            "Run evaluate_model.py first."
        )

    solved_set = optimal_set | suboptimal_set
    print(f"[INFO] Total solved: {len(solved_set)} / {len(states)}")

    # ── Tower-to-tower path ───────────────────────────────────────────────────
    path = tower_to_tower_path(args.n_disks)
    path_set = set(path)
    path_indices = {state_idx[s] for s in path}
    print(f"[INFO] Tower-to-tower path: {len(path)} states, {len(path)-1} moves")

    # ── Goal distance (optimal solution length for each state) ────────────────
    goal: State = tuple([2] * args.n_disks)
    goal_i = state_idx[goal]
    opt_len = {s: int(dist[state_idx[s], goal_i]) for s in states}

    # ─────────────────────────────────────────────────────────────────────────
    # ANALYSIS
    # ─────────────────────────────────────────────────────────────────────────
    optimal_list   = sorted(optimal_set)
    suboptimal_list = sorted(suboptimal_set)
    solved_list    = sorted(solved_set)
    unsolved_list  = [s for s in states if s not in solved_set]

    on_path      = [s for s in solved_list if s in path_set]
    adj_to_path  = [s for s in solved_list
                    if s not in path_set
                    and any(nb in path_set for nb in legal_neighbors(s))]
    within_dist2 = [s for s in solved_list
                    if min_path_dist(state_idx[s], path_indices, dist) <= 2]

    mean_d_solved   = float(np.mean([min_path_dist(state_idx[s], path_indices, dist)
                                     for s in solved_list]))
    mean_d_unsolved = float(np.mean([min_path_dist(state_idx[s], path_indices, dist)
                                     for s in unsolved_list]))

    print()
    print("── Solved state analysis ─────────────────────────────────────────")
    print(f"Solved states on tower path:       {len(on_path):2d} / {len(solved_list)}")
    print(f"Solved states adjacent to path:    {len(adj_to_path):2d} / {len(solved_list)}")
    print(f"Solved states within dist 2:       {len(within_dist2):2d} / {len(solved_list)}")
    print(f"Mean dist (solved   -> path):      {mean_d_solved:.2f}")
    print(f"Mean dist (unsolved -> path):      {mean_d_unsolved:.2f}")

    print()
    print(f"  {'State':<8}  {'OptLen':>6}  {'PathDist':>8}")
    print(f"  {'-----':<8}  {'------':>6}  {'--------':>8}")
    for s in solved_list:
        d = min_path_dist(state_idx[s], path_indices, dist)
        print(f"  {state_label(s):<8}  {opt_len[s]:>6}  {d:>8.0f}")

    # ── Solve rate by optimal solution length ─────────────────────────────────
    buckets = [(1, 3), (4, 7), (8, 11), (12, 15)]
    print()
    print("── Solve rate by optimal solution length ─────────────────────────")
    print(f"  {'Opt Length':<12} {'Total':>6} {'Solved':>7} {'Rate':>7}")
    print(f"  {'-----------':<12} {'------':>6} {'-------':>7} {'------':>7}")
    for lo, hi in buckets:
        in_bucket   = [s for s in states if lo <= opt_len[s] <= hi]
        solved_in   = [s for s in in_bucket if s in solved_set]
        rate = 100.0 * len(solved_in) / len(in_bucket) if in_bucket else 0.0
        print(f"  {f'{lo}-{hi}':<12} {len(in_bucket):>6} {len(solved_in):>7} {rate:>6.1f}%")

    # ─────────────────────────────────────────────────────────────────────────
    # VISUALIZATION
    # ─────────────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 11), constrained_layout=True)

    # Graph edges
    for i, j in edges:
        ax.plot(
            [coords[i, 0], coords[j, 0]],
            [coords[i, 1], coords[j, 1]],
            color="lightgray", linewidth=0.6, alpha=0.4, zorder=1,
        )

    # Tower-to-tower path arrows
    for k in range(len(path) - 1):
        s0, s1 = path[k], path[k + 1]
        i0, i1 = state_idx[s0], state_idx[s1]
        ax.annotate(
            "",
            xy=(coords[i1, 0], coords[i1, 1]),
            xytext=(coords[i0, 0], coords[i0, 1]),
            arrowprops=dict(
                arrowstyle="-|>",
                color="darkorange",
                lw=2.0,
                alpha=0.7,
                mutation_scale=12,
            ),
            zorder=2,
        )

    # Unsolved states
    for s in unsolved_list:
        i = state_idx[s]
        on_tp = s in path_set
        ax.scatter(
            coords[i, 0], coords[i, 1],
            s=80,
            color="lightgray",
            edgecolors="darkorange" if on_tp else "#888888",
            linewidths=2.5 if on_tp else 0.5,
            zorder=3,
        )

    # Suboptimal states (reached goal but used extra moves) — yellow-green fill
    for s in suboptimal_list:
        i = state_idx[s]
        on_tp = s in path_set
        ax.scatter(
            coords[i, 0], coords[i, 1],
            s=120,
            color="#a8e063",
            edgecolors="darkorange" if on_tp else "black",
            linewidths=2.5 if on_tp else 1.2,
            zorder=4,
        )

    # Optimal states — bright green fill
    for s in optimal_list:
        i = state_idx[s]
        on_tp = s in path_set
        ax.scatter(
            coords[i, 0], coords[i, 1],
            s=140,
            color="#2ecc71",
            edgecolors="darkorange" if on_tp else "black",
            linewidths=2.5 if on_tp else 1.2,
            zorder=5,
        )

    # State labels
    for i, s in enumerate(states):
        ax.text(
            coords[i, 0], coords[i, 1],
            state_label(s),
            fontsize=6, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                      alpha=0.75, edgecolor="none"),
            zorder=6,
        )

    # Legend
    n_opt  = len(optimal_list)
    n_sub  = len(suboptimal_list)
    n_fail = len(unsolved_list)
    legend_handles = [
        mpatches.Patch(facecolor="#2ecc71", edgecolor="black",
                       label=f"Correct optimal ({n_opt}/81)"),
        mpatches.Patch(facecolor="#a8e063", edgecolor="black",
                       label=f"Correct suboptimal ({n_sub}/81)"),
        mpatches.Patch(facecolor="lightgray", edgecolor="#888888",
                       label=f"Failed ({n_fail}/81)"),
        mpatches.Patch(facecolor="white", edgecolor="darkorange", linewidth=2.5,
                       label="On tower-to-tower path (orange border)"),
        plt.Line2D([0], [0], color="darkorange", lw=2, alpha=0.8,
                   marker=">", markersize=7,
                   label="Optimal path direction (0000→2222, 15 moves)"),
    ]
    ax.legend(handles=legend_handles, loc="best", fontsize=9, framealpha=0.9)

    ax.set_title(
        f"Qwen3-27B: Solved States on Sierpinski Graph (Layer {args.layer})\n"
        f"(permissive parsing: {n_opt} optimal + {n_sub} suboptimal = {n_opt+n_sub} solved / 81)",
        fontsize=12,
    )
    ax.grid(alpha=0.15)
    ax.set_xlabel("Probe dim 0")
    ax.set_ylabel("Probe dim 1")

    out_path = out_dir / f"highlight_solved_layer_{args.layer}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[INFO] Saved figure to {out_path}")


if __name__ == "__main__":
    main()
