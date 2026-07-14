#!/usr/bin/env python3
"""Train one (regime, h_dim) joint-MLP probe combination and save checkpoint + metrics.

Three regimes:
  - sequential_cls_first  (A): train encoder+per-disk heads on CE; freeze; train 2D head on distance MSE.
  - joint                 (B): train encoder + both heads together with normalised combined loss.
  - sequential_dist_first (C): train encoder+2D head on distance MSE; freeze; train per-disk heads on CE.

The probe operates on the SEP-token (second SEP) residual of the toy ToH transformer
at a chosen layer. Distance targets, normalisation, and the nearest-state-accuracy
metric mirror the existing 2D distance probe (probe.py / toh_transformer/probe.py).

Training core depends on torch only; model loading / SEP extraction reuse the
lightweight toh_transformer.utils helpers.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import deque
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.append(str(_REPO))

from model import MLPEncoder, PerDiskHeads, Proj2D  # noqa: E402
from toh_transformer import utils as utils  # noqa: E402
from toh_transformer.data import Vocabulary  # noqa: E402

State = Tuple[int, ...]

REGIMES = {
    "sequential_cls_first": "A",
    "joint": "B",
    "sequential_dist_first": "C",
}


# ---------------------------------------------------------------------------
# Data: SEP activations, labels, graph distances
# ---------------------------------------------------------------------------
def legal_neighbors(state: State) -> List[State]:
    tops = utils.top_disk_per_peg(state)
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
                nxt[src_top] = dst
                out.append(tuple(nxt))
    return out


def build_distances_and_neighbors(states: Sequence[State]) -> Tuple[np.ndarray, Dict[int, Set[int]]]:
    idx = {s: i for i, s in enumerate(states)}
    n = len(states)
    adj: List[List[int]] = [[] for _ in range(n)]
    neighbor_sets: Dict[int, Set[int]] = {i: set() for i in range(n)}
    for i, s in enumerate(states):
        for nb in legal_neighbors(s):
            j = idx[nb]
            adj[i].append(j)
            neighbor_sets[i].add(j)
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
    if np.isinf(dist).any():
        raise RuntimeError("Graph disconnected")
    return dist, neighbor_sets


@torch.no_grad()
def extract_sep_activations(model, states: Sequence[State], layer: int, vocab: Vocabulary,
                            device: torch.device) -> torch.Tensor:
    goal = tuple(2 for _ in range(len(states[0])))
    rows = [utils.build_context_ids(s, goal, vocab) for s in states]
    sep_idx = len(rows[0]) - 1  # second SEP is the last context token
    inp = torch.tensor(rows, dtype=torch.long, device=device)
    with model.capture_activations(layers=[layer]) as cache:
        _ = model(inp)
    return cache[layer][:, sep_idx, :].detach().float().cpu()  # (81, 128)


# ---------------------------------------------------------------------------
# Metrics (numpy)
# ---------------------------------------------------------------------------
def _rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    return ranks


def spearman(pred: np.ndarray, true: np.ndarray) -> float:
    return pearson(_rankdata(pred), _rankdata(true))


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64) - a.mean()
    b = b.astype(np.float64) - b.mean()
    denom = math.sqrt(float(np.sum(a * a) * np.sum(b * b)))
    return float(np.sum(a * b) / denom) if denom > 0 else 0.0


def evaluate_distance(coords: np.ndarray, true_dist: np.ndarray,
                      neighbor_sets: Dict[int, Set[int]]) -> Tuple[float, float, float]:
    pred = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    iu, ju = np.triu_indices(pred.shape[0], k=1)
    rho = spearman(pred[iu, ju], true_dist[iu, ju])
    pear = pearson(pred[iu, ju], true_dist[iu, ju])
    n = pred.shape[0]
    correct = 0
    for i in range(n):
        row = pred[i].copy()
        row[i] = np.inf
        if int(np.argmin(row)) in neighbor_sets[i]:
            correct += 1
    return rho, pear, correct / n


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------
def cls_loss(logits_list: List[torch.Tensor], labels: torch.Tensor) -> torch.Tensor:
    """Mean cross-entropy over the four per-disk heads."""
    losses = [F.cross_entropy(logits_list[d], labels[:, d]) for d in range(len(logits_list))]
    return torch.stack(losses).mean()


def dist_loss(coords: torch.Tensor, y_dist_norm: torch.Tensor) -> torch.Tensor:
    pred = torch.cdist(coords, coords, p=2)
    return torch.mean((pred - y_dist_norm) ** 2)


def per_disk_accuracy(logits_list: List[torch.Tensor], labels: torch.Tensor) -> List[float]:
    accs = []
    for d in range(len(logits_list)):
        pred = logits_list[d].argmax(dim=1)
        accs.append(float((pred == labels[:, d]).float().mean().item()))
    return accs


# ---------------------------------------------------------------------------
# Training regimes
# ---------------------------------------------------------------------------
def make_opt(params, lr=1e-3, wd=1e-4):
    return torch.optim.AdamW([p for p in params if p.requires_grad], lr=lr, weight_decay=wd)


def train_regime(regime: str, x: torch.Tensor, labels: torch.Tensor, y_dist_norm: torch.Tensor,
                 h_dim: int, epochs: int, lambda_dist: float, device: torch.device):
    torch.manual_seed(42)
    encoder = MLPEncoder(in_dim=x.shape[1], hidden=64, h_dim=h_dim).to(device)

    def log(phase, ep, val):
        if ep == 1 or ep % 200 == 0 or ep == epochs:
            print(f"  [{regime}/{phase}] epoch={ep:4d} loss={val:.6f}")

    if regime == "sequential_cls_first":
        perdisk = PerDiskHeads(h_dim).to(device)
        opt = make_opt(list(encoder.parameters()) + list(perdisk.parameters()))
        for ep in range(1, epochs + 1):
            opt.zero_grad(set_to_none=True)
            loss = cls_loss(perdisk(encoder(x)), labels)
            loss.backward(); opt.step(); log("P1-cls", ep, float(loss))
        # Phase 2: freeze encoder + classifiers, train new 2D head.
        for p in encoder.parameters():
            p.requires_grad = False
        for p in perdisk.parameters():
            p.requires_grad = False
        assert all(not p.requires_grad for p in encoder.parameters()), "A P2: encoder not frozen"
        proj2d = Proj2D(h_dim).to(device)
        opt = make_opt(proj2d.parameters())
        for ep in range(1, epochs + 1):
            opt.zero_grad(set_to_none=True)
            with torch.no_grad():
                z = encoder(x)
            loss = dist_loss(proj2d(z), y_dist_norm)
            loss.backward(); opt.step(); log("P2-dist", ep, float(loss))

    elif regime == "sequential_dist_first":
        proj2d = Proj2D(h_dim).to(device)
        opt = make_opt(list(encoder.parameters()) + list(proj2d.parameters()))
        for ep in range(1, epochs + 1):
            opt.zero_grad(set_to_none=True)
            loss = dist_loss(proj2d(encoder(x)), y_dist_norm)
            loss.backward(); opt.step(); log("P1-dist", ep, float(loss))
        # Phase 2: freeze encoder + 2D head, train new per-disk heads.
        for p in encoder.parameters():
            p.requires_grad = False
        for p in proj2d.parameters():
            p.requires_grad = False
        assert all(not p.requires_grad for p in encoder.parameters()), "C P2: encoder not frozen"
        assert all(not p.requires_grad for p in proj2d.parameters()), "C P2: proj2d not frozen"
        perdisk = PerDiskHeads(h_dim).to(device)
        opt = make_opt(perdisk.parameters())
        for ep in range(1, epochs + 1):
            opt.zero_grad(set_to_none=True)
            with torch.no_grad():
                z = encoder(x)
            loss = cls_loss(perdisk(z), labels)
            loss.backward(); opt.step(); log("P2-cls", ep, float(loss))

    elif regime == "joint":
        perdisk = PerDiskHeads(h_dim).to(device)
        proj2d = Proj2D(h_dim).to(device)
        params = list(encoder.parameters()) + list(perdisk.parameters()) + list(proj2d.parameters())
        assert all(p.requires_grad for p in params), "B: all params must be trainable"
        opt = make_opt(params)
        # Loss normalisation baselines (computed once, before any gradient step).
        cls_baseline = math.log(3.0)
        with torch.no_grad():
            dist_baseline = float(dist_loss(proj2d(encoder(x)), y_dist_norm).item())
            dist_baseline = max(dist_baseline, 1e-8)
        print(f"  [joint] baselines: L_cls={cls_baseline:.6f}  L_dist={dist_baseline:.6f}")
        for ep in range(1, epochs + 1):
            opt.zero_grad(set_to_none=True)
            z = encoder(x)
            lc = cls_loss(perdisk(z), labels)
            ld = dist_loss(proj2d(z), y_dist_norm)
            loss = lc / cls_baseline + lambda_dist * ld / dist_baseline
            loss.backward(); opt.step(); log("joint", ep, float(loss))
    else:
        raise ValueError(f"Unknown regime {regime}")

    return encoder, perdisk, proj2d


def main() -> None:
    ap = argparse.ArgumentParser(description="Train one joint-MLP probe (regime, h_dim)")
    ap.add_argument("--regime", required=True, choices=list(REGIMES.keys()))
    ap.add_argument("--checkpoint", default="toh_transformer/checkpoints/n4/best.pt")
    ap.add_argument("--n_disks", type=int, default=4)
    ap.add_argument("--layer", type=int, default=5)
    ap.add_argument("--h_dim", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--lambda_dist", type=float, default=1.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--output", required=True, help="checkpoint path (.pt); metrics JSON is <stem>.json")
    args = ap.parse_args()

    utils.set_seed(42)
    device = torch.device(args.device)
    vocab = Vocabulary()
    utils.confirm_tokenizer_mapping(vocab)

    ckpt = utils.resolve_checkpoint_path(args.checkpoint, args.n_disks)
    model = utils.load_model(ckpt, args.n_disks, device)
    if args.layer < 1 or args.layer > model.n_layers:
        raise ValueError(f"layer {args.layer} out of range [1, {model.n_layers}]")

    states = utils.enumerate_states(args.n_disks)
    x = extract_sep_activations(model, states, args.layer, vocab, device).to(device)
    labels = torch.tensor([list(s) for s in states], dtype=torch.long, device=device)  # (81, 4)
    true_dist, neighbor_sets = build_distances_and_neighbors(states)
    mu, sigma = float(true_dist.mean()), float(true_dist.std())
    sigma = sigma if sigma > 1e-8 else 1.0
    y_dist_norm = torch.tensor((true_dist - mu) / sigma, dtype=torch.float32, device=device)

    print(f"[INFO] regime={args.regime} layer={args.layer} h_dim={args.h_dim} "
          f"x={tuple(x.shape)} states={len(states)}")

    encoder, perdisk, proj2d = train_regime(
        args.regime, x, labels, y_dist_norm, args.h_dim, args.epochs, args.lambda_dist, device
    )

    # Evaluation on the full 81-state set (the standard geometry/decodability set).
    encoder.eval(); perdisk.eval(); proj2d.eval()
    with torch.no_grad():
        z = encoder(x)
        coords = proj2d(z).detach().cpu().numpy()
        disk_acc = per_disk_accuracy(perdisk(z), labels)
    rho, pear, ns_acc = evaluate_distance(coords, true_dist, neighbor_sets)
    mean_disk = float(np.mean(disk_acc))

    out_pt = Path(args.output)
    out_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "regime": args.regime,
        "layer": args.layer,
        "h_dim": args.h_dim,
        "epochs": args.epochs,
        "lambda_dist": args.lambda_dist,
        "n_disks": args.n_disks,
        "checkpoint": str(ckpt),
        "encoder_state": encoder.state_dict(),
        "perdisk_state": perdisk.state_dict(),
        "proj2d_state": proj2d.state_dict(),
    }, out_pt)

    metrics = {
        "regime": args.regime,
        "layer": args.layer,
        "h_dim": args.h_dim,
        "epochs": args.epochs,
        "lambda_dist": args.lambda_dist if args.regime == "joint" else None,
        "spearman": rho,
        "pearson": pear,
        "nearest_state_acc": ns_acc,
        "per_disk_acc": disk_acc,
        "mean_disk_acc": mean_disk,
        "checkpoint_path": str(out_pt),
    }
    out_json = out_pt.with_suffix(".json")
    out_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"[INFO] spearman={rho:.4f} pearson={pear:.4f} ns_acc={ns_acc:.4f} "
          f"per_disk={['%.3f' % a for a in disk_acc]} mean={mean_disk:.4f}")
    print(f"[INFO] Wrote {out_pt} and {out_json}")


if __name__ == "__main__":
    main()
