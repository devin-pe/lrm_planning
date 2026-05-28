"""Recompute Spearman + Pearson at positions A, B, C from saved Qwen generation
hidden states.

Loads `outputs/qwen_probe_generation/generation_hidden_states.pt`, re-trains the
2D distance-matching probe per (position, layer), and writes a new
`pearson_spearman_generation.json` next to it. No vLLM, no Qwen model load.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr

State = Tuple[int, ...]


def legal_neighbors(state: State, n_disks: int) -> List[State]:
    tops: List = [None, None, None]
    for disk in range(1, n_disks + 1):
        peg = state[disk - 1]
        if tops[peg] is None:
            tops[peg] = disk
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


def build_distances(states: Sequence[State], n_disks: int):
    n = len(states)
    state_to_idx = {s: i for i, s in enumerate(states)}
    adj = [[] for _ in range(n)]
    nbrs: Dict[int, Set[int]] = {i: set() for i in range(n)}
    for i, s in enumerate(states):
        for nb in legal_neighbors(s, n_disks):
            j = state_to_idx[nb]
            adj[i].append(j)
            nbrs[i].add(j)
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
    return dist, nbrs, state_to_idx


def norm_dist(d: np.ndarray):
    mu = float(d.mean())
    sd = float(np.clip(d.std(), 1e-8, None))
    return (d - mu) / sd, mu, sd


def train_probe(x: torch.Tensor, y_norm: torch.Tensor, epochs: int, lr: float,
                device: torch.device) -> np.ndarray:
    x = x.to(device)
    y = y_norm.to(device)
    probe = nn.Linear(x.shape[1], 2, bias=True).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        z = probe(x)
        loss = torch.mean((torch.cdist(z, z, p=2) - y) ** 2)
        loss.backward()
        opt.step()
    with torch.no_grad():
        return probe(x).detach().cpu().numpy().astype(np.float64)


def evaluate(coords: np.ndarray, true_dist: np.ndarray, nbrs: Dict[int, Set[int]]):
    pred = np.linalg.norm(coords[:, None] - coords[None], axis=2)
    ui, uj = np.triu_indices(pred.shape[0], k=1)
    pred_pairs, true_pairs = pred[ui, uj], true_dist[ui, uj]
    rho, _ = spearmanr(pred_pairs, true_pairs)
    rho = float(rho) if np.isfinite(rho) else float("nan")
    pc = pred_pairs.astype(np.float64) - pred_pairs.mean()
    tc = true_pairs.astype(np.float64) - true_pairs.mean()
    denom = float(np.sqrt(np.sum(pc * pc) * np.sum(tc * tc)))
    pearson = float(np.sum(pc * tc) / denom) if denom > 0 else float("nan")
    n = pred.shape[0]
    correct = 0
    for i in range(n):
        row = pred[i].copy()
        row[i] = np.inf
        if int(np.argmin(row)) in nbrs[i]:
            correct += 1
    return rho, pearson, correct / n


def fit_position_AorB(pos_list, states, true_dist, nbrs, epochs, lr, device):
    idxs = [int(i) for i, _ in pos_list]
    vecs = torch.stack([v for _, v in pos_list]).float()
    sub_dist = true_dist[np.ix_(idxs, idxs)]
    sub_norm, _, _ = norm_dist(sub_dist)
    sub_nbrs = {new_i: {idxs.index(j) for j in nbrs[idxs[new_i]] if j in idxs}
                for new_i in range(len(idxs))}
    coords = train_probe(vecs, torch.tensor(sub_norm, dtype=torch.float32),
                         epochs, lr, device)
    rho, pearson, adj_acc = evaluate(coords, sub_dist, sub_nbrs)
    return rho, pearson, adj_acc, len(idxs)


def fit_position_C_avg(pos_list, states, state_to_idx, true_dist, nbrs,
                       epochs, lr, device):
    bucket: Dict[State, List[torch.Tensor]] = defaultdict(list)
    for st, v in pos_list:
        bucket[tuple(int(x) for x in st)].append(v)
    c_states = sorted(bucket.keys(), key=lambda s: state_to_idx.get(s, 9999))
    c_states_valid = [s for s in c_states if s in state_to_idx]
    if len(c_states_valid) < 4:
        return float("nan"), float("nan"), float("nan"), len(c_states_valid)
    c_idxs = [state_to_idx[s] for s in c_states_valid]
    vecs = torch.stack([torch.stack(bucket[s]).mean(0) for s in c_states_valid]).float()
    c_dist = true_dist[np.ix_(c_idxs, c_idxs)]
    c_norm, _, _ = norm_dist(c_dist)
    c_nbrs = {ni: {c_idxs.index(j) for j in nbrs[c_idxs[ni]] if j in c_idxs}
              for ni in range(len(c_idxs))}
    coords = train_probe(vecs, torch.tensor(c_norm, dtype=torch.float32),
                         epochs, lr, device)
    rho, pearson, adj_acc = evaluate(coords, c_dist, c_nbrs)
    return rho, pearson, adj_acc, len(c_states_valid)


def main():
    out_dir = Path("outputs/qwen_probe_generation")
    bundle = torch.load(out_dir / "generation_hidden_states.pt",
                        map_location="cpu", weights_only=False)
    layer_ids: List[int] = list(bundle["layer_ids"])
    states: List[State] = [tuple(int(x) for x in s) for s in bundle["states"]]
    pos_A = bundle["pos_A"]; pos_B = bundle["pos_B"]; pos_C = bundle["pos_C"]

    n_disks = len(states[0])
    true_dist, nbrs, state_to_idx = build_distances(states, n_disks)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs, lr = 3000, 1e-3

    rows = []
    for l in layer_ids:
        print(f"\n=== Layer {l} ===")
        for pos_name, pos_dict in [("A", pos_A), ("B", pos_B)]:
            vlist = pos_dict.get(l) or []
            if not vlist:
                continue
            rho, pearson, adj, n = fit_position_AorB(
                vlist, states, true_dist, nbrs, epochs, lr, device)
            print(f"  Pos {pos_name}: n={n:3d}  Spearman={rho:.4f}  Pearson={pearson:.4f}  AdjAcc={adj:.4f}")
            rows.append({"pos": pos_name, "layer": int(l), "n": int(n),
                         "spearman_rho": rho, "pearson_r": pearson, "adj_acc": adj})
        clist = pos_C.get(l) or []
        if clist:
            rho, pearson, adj, n = fit_position_C_avg(
                clist, states, state_to_idx, true_dist, nbrs, epochs, lr, device)
            print(f"  Pos C_avg: n_states={n:3d}  Spearman={rho:.4f}  Pearson={pearson:.4f}  AdjAcc={adj:.4f}")
            rows.append({"pos": "C_avg", "layer": int(l), "n": int(n),
                         "spearman_rho": rho, "pearson_r": pearson, "adj_acc": adj})

    summary = {
        "layer_ids": layer_ids,
        "n_disks": n_disks,
        "epochs": epochs,
        "lr": lr,
        "rows": rows,
    }
    out_path = out_dir / "pearson_spearman_generation.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[INFO] Wrote {out_path}")


if __name__ == "__main__":
    main()
