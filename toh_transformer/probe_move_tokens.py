#!/usr/bin/env python3
"""Train move-token probes to test intermediate state encoding.

This script trains a new linear 2D probe directly on move-token activations,
then evaluates whether intermediate state identity is decodable.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

if __package__ is None or __package__ == "":
    import sys

    sys.path.append(str(Path(__file__).resolve().parent.parent))

from toh_transformer.config import default_model_hparams, max_seq_len_for_disks
from toh_transformer import utils as utils
from toh_transformer.data import Vocabulary
from toh_transformer.model import ToHTransformer

State = Tuple[int, ...]
Edge = Tuple[int, int]

EXPECTED_TOKEN_MAPPING = utils.EXPECTED_TOKEN_MAPPING


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/evaluate probes on move-token activations")
    parser.add_argument("--checkpoint", type=str, default="best.pt")
    parser.add_argument(
        "--eval_results",
        type=str,
        default="toh_transformer/eval_output/evaluation_results.json",
    )
    parser.add_argument("--n_disks", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="toh_transformer/probe_move_tokens_output")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--spearman_pairs", type=int, default=200000)
    parser.add_argument("--plot_sample", type=int, default=4000)
    parser.add_argument("--min_step_samples", type=int, default=50)
    return parser.parse_args()


def legal_neighbors(state: State) -> List[State]:
    tops = top_disk_per_peg(state)
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


def build_graph_and_distances(n_disks: int) -> Tuple[List[State], np.ndarray, List[Edge]]:
    states = enumerate_states(n_disks)
    state_to_idx = {s: i for i, s in enumerate(states)}
    n_states = len(states)

    adj: List[List[int]] = [[] for _ in range(n_states)]
    edge_set = set()
    for i, s in enumerate(states):
        for nbr in legal_neighbors(s):
            j = state_to_idx[nbr]
            adj[i].append(j)
            a, b = sorted((i, j))
            edge_set.add((a, b))

    dist = np.full((n_states, n_states), np.inf, dtype=np.float32)
    for src in range(n_states):
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
    return states, dist, sorted(edge_set)


@torch.no_grad()
def extract_sep2_activations(
    model: ToHTransformer,
    states: Sequence[State],
    goal: State,
    vocab: Vocabulary,
    device: torch.device,
    batch_size: int = 128,
) -> Dict[int, torch.Tensor]:
    layers = list(range(1, model.n_layers + 1))
    rows = [build_context_ids(s, goal, vocab) for s in states]
    sep2_idx = len(rows[0]) - 1
    inp = torch.tensor(rows, dtype=torch.long, device=device)

    out: Dict[int, List[torch.Tensor]] = {l: [] for l in layers}
    for st in range(0, len(rows), batch_size):
        batch = inp[st : st + batch_size]
        with model.capture_activations(layers=layers) as cache:
            _ = model(batch)
        for layer in layers:
            out[layer].append(cache[layer][:, sep2_idx, :].detach().cpu())

    return {layer: torch.cat(chunks, dim=0) for layer, chunks in out.items()}


@torch.no_grad()
def capture_move_position_activations(
    model: ToHTransformer,
    full_sequence: Sequence[int],
    context_len: int,
    n_move_tokens: int,
    device: torch.device,
) -> Dict[int, np.ndarray]:
    layers = list(range(1, model.n_layers + 1))
    seq_t = torch.tensor(full_sequence, dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        with model.capture_activations(layers=layers) as cache:
            _ = model(seq_t)
    move_start = context_len
    move_end = context_len + n_move_tokens
    return {layer: cache[layer][0, move_start:move_end, :].detach().cpu().numpy() for layer in layers}


def collect_move_token_dataset(
    model: ToHTransformer,
    problems: Sequence[Tuple[State, State]],
    vocab: Vocabulary,
    state_to_idx: Dict[State, int],
    device: torch.device,
    n_disks: int,
) -> Dict[str, object]:
    layers = list(range(1, model.n_layers + 1))
    context_len = 2 * n_disks + 3
    move_order = ["M01", "M02", "M10", "M12", "M20", "M21"]
    move_id_to_class = {vocab.stoi[tok]: i for i, tok in enumerate(move_order)}

    x_by_layer: Dict[int, List[np.ndarray]] = {l: [] for l in layers}
    y_state_idx: List[int] = []
    step_idx: List[int] = []
    move_class_idx: List[int] = []
    traj_len: List[int] = []
    bridge_cross: List[bool] = []

    kept = 0
    for pi, (start, goal) in enumerate(problems):
        context_ids = build_context_ids(start, goal, vocab)
        generated_ids, eos_seen = greedy_decode_ids(model, context_ids, vocab.eos_id, device)

        move_ids = generated_ids[: generated_ids.index(vocab.eos_id)] if eos_seen and vocab.eos_id in generated_ids else generated_ids
        move_tokens = [vocab.itos[i] for i in move_ids]

        current = start
        inter_states: List[State] = []
        valid_move_ids: List[int] = []
        for tok, mid in zip(move_tokens, move_ids):
            mv = decode_move_token(tok)
            if mv is None:
                break
            nxt = apply_move_and_next_state(current, mv)
            if nxt is None:
                break
            inter_states.append(nxt)
            valid_move_ids.append(mid)
            current = nxt

        # Keep only trajectories that reached goal with EOS and all legal moves.
        if not eos_seen or current != goal:
            continue

        full_seq = context_ids + valid_move_ids + [vocab.eos_id]
        if len(full_seq) > model.max_seq_len:
            continue

        layer_acts = capture_move_position_activations(
            model=model,
            full_sequence=full_seq,
            context_len=context_len,
            n_move_tokens=len(valid_move_ids),
            device=device,
        )

        largest_start = start[-1]
        tlen = len(inter_states)
        for s_i, st in enumerate(inter_states):
            y_state_idx.append(state_to_idx[st])
            step_idx.append(s_i + 1)
            move_class_idx.append(move_id_to_class[valid_move_ids[s_i]])
            traj_len.append(tlen)
            bridge_cross.append(st[-1] != largest_start)
            for layer in layers:
                x_by_layer[layer].append(layer_acts[layer][s_i].astype(np.float32, copy=False))

        kept += 1
        if (pi + 1) % 500 == 0:
            print(f"[INFO] Processed {pi + 1}/{len(problems)} problems, kept={kept}")

    if len(y_state_idx) == 0:
        raise RuntimeError("No move-token samples collected")

    x_np = {layer: np.stack(chunks, axis=0) for layer, chunks in x_by_layer.items()}
    return {
        "x_by_layer": x_np,
        "y_state_idx": np.array(y_state_idx, dtype=np.int64),
        "step_idx": np.array(step_idx, dtype=np.int64),
        "move_class_idx": np.array(move_class_idx, dtype=np.int64),
        "traj_len": np.array(traj_len, dtype=np.int64),
        "bridge_cross": np.array(bridge_cross, dtype=bool),
        "n_kept_trajectories": kept,
    }


def train_distance_probe_batched(
    x: np.ndarray,
    y_state_idx: np.ndarray,
    dist_norm: np.ndarray,
    train_idx: np.ndarray,
    device: torch.device,
    lr: float,
    epochs: int,
    batch_size: int,
    seed: int,
) -> Tuple[nn.Linear, float]:
    d_model = x.shape[1]
    probe = nn.Linear(d_model, 2, bias=True).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)

    x_t = torch.tensor(x, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_state_idx, dtype=torch.long, device=device)
    d_t = torch.tensor(dist_norm, dtype=torch.float32, device=device)

    rng = np.random.default_rng(seed)

    final_loss = float("nan")
    for epoch in range(1, epochs + 1):
        order = rng.permutation(train_idx)
        losses: List[float] = []
        for st in range(0, len(order), batch_size):
            b_idx_np = order[st : st + batch_size]
            if b_idx_np.shape[0] < 2:
                continue
            b_idx = torch.tensor(b_idx_np, dtype=torch.long, device=device)

            xb = x_t[b_idx]
            sb = y_t[b_idx]
            true_d = d_t[sb][:, sb]

            opt.zero_grad(set_to_none=True)
            p = probe(xb)
            pred_d = torch.cdist(p, p, p=2)
            loss = torch.mean((pred_d - true_d) ** 2)
            loss.backward()
            opt.step()

            losses.append(float(loss.item()))

        if epoch == 1 or epoch % 20 == 0 or epoch == epochs:
            mean_loss = float(np.mean(losses)) if losses else float("nan")
            final_loss = mean_loss
            print(f"[train] epoch={epoch:3d} loss={mean_loss:.6f}")

    return probe, final_loss


def project_with_probe(probe: nn.Linear, x: np.ndarray, device: torch.device, chunk: int = 4096) -> np.ndarray:
    out: List[np.ndarray] = []
    with torch.no_grad():
        for st in range(0, x.shape[0], chunk):
            xb = torch.tensor(x[st : st + chunk], dtype=torch.float32, device=device)
            out.append(probe(xb).detach().cpu().numpy())
    return np.concatenate(out, axis=0)


def build_token_augmented_features(x: np.ndarray, move_class_idx: np.ndarray) -> np.ndarray:
    n = x.shape[0]
    move_one_hot = np.zeros((n, 6), dtype=np.float32)
    move_one_hot[np.arange(n), move_class_idx] = 1.0
    return np.concatenate([x.astype(np.float32, copy=False), move_one_hot], axis=1)


def build_sep_token_augmented_features(sep_x: np.ndarray) -> np.ndarray:
    n = sep_x.shape[0]
    # SEP activations have no move token; use a zero token-feature vector.
    aux = np.zeros((n, 6), dtype=np.float32)
    return np.concatenate([sep_x.astype(np.float32, copy=False), aux], axis=1)


def nearest_state_accuracy(proj: np.ndarray, ref_positions: np.ndarray, y_state_idx: np.ndarray) -> float:
    d_ref = np.linalg.norm(proj[:, None, :] - ref_positions[None, :, :], axis=-1)
    nearest_idx = np.argmin(d_ref, axis=1)
    return float(np.mean(nearest_idx == y_state_idx))


def weighted_mean(values: Sequence[float], weights: Sequence[int]) -> float:
    v = np.array(values, dtype=np.float64)
    w = np.array(weights, dtype=np.float64)
    mask = np.isfinite(v) & (w > 0)
    if not np.any(mask):
        return float("nan")
    return float(np.sum(v[mask] * w[mask]) / np.sum(w[mask]))


def compute_spearman_heldout(
    proj: np.ndarray,
    y_state_idx: np.ndarray,
    dist_true: np.ndarray,
    val_idx: np.ndarray,
    num_pairs: int,
    seed: int,
) -> float:
    rho, _ = compute_corr_heldout(proj, y_state_idx, dist_true, val_idx, num_pairs, seed)
    return rho


def compute_corr_heldout(
    proj: np.ndarray,
    y_state_idx: np.ndarray,
    dist_true: np.ndarray,
    val_idx: np.ndarray,
    num_pairs: int,
    seed: int,
) -> Tuple[float, float]:
    if val_idx.shape[0] < 2:
        return float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    m = max(1, num_pairs)

    a = rng.integers(0, val_idx.shape[0], size=m)
    b = rng.integers(0, val_idx.shape[0], size=m)
    mask = a != b
    if not np.any(mask):
        return float("nan"), float("nan")
    a = a[mask]
    b = b[mask]

    ia = val_idx[a]
    ib = val_idx[b]

    pred = np.linalg.norm(proj[ia] - proj[ib], axis=1)
    true = dist_true[y_state_idx[ia], y_state_idx[ib]]

    return float(spearman_rho(pred, true)), float(pearson_r(pred, true))


def rankdata_average(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    sorted_x = x[order]
    ranks = np.empty_like(sorted_x, dtype=np.float64)

    i = 0
    n = sorted_x.shape[0]
    while i < n:
        j = i + 1
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        avg_rank = 0.5 * (i + j - 1) + 1.0
        ranks[i:j] = avg_rank
        i = j

    out = np.empty_like(ranks)
    out[order] = ranks
    return out


def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    rx = rankdata_average(x)
    ry = rankdata_average(y)

    rx = rx - rx.mean()
    ry = ry - ry.mean()

    denom = math.sqrt(float(np.sum(rx * rx) * np.sum(ry * ry)))
    if denom == 0.0:
        return 0.0
    return float(np.sum(rx * ry) / denom)


def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    x = x.astype(np.float64) - x.mean()
    y = y.astype(np.float64) - y.mean()
    denom = math.sqrt(float(np.sum(x * x) * np.sum(y * y)))
    if denom == 0.0:
        return 0.0
    return float(np.sum(x * y) / denom)


def stress_metric(pred_d: np.ndarray, true_d: np.ndarray) -> float:
    num = float(np.sum((pred_d - true_d) ** 2))
    den = float(np.sum(true_d**2))
    if den == 0.0:
        return 0.0
    return math.sqrt(num / den)


def state_to_label(state: State) -> str:
    return "".join(str(peg + 1) for peg in state)


def rotate_positions(positions: np.ndarray, degrees_clockwise: float) -> np.ndarray:
    theta = -math.radians(degrees_clockwise)
    c, s = math.cos(theta), math.sin(theta)
    r = np.array([[c, -s], [s, c]], dtype=np.float64)

    p = positions.astype(np.float64)
    center = p.mean(axis=0, keepdims=True)
    rotated = (p - center) @ r.T + center
    return rotated.astype(np.float32)


def plot_labeled_state_map(
    positions: np.ndarray,
    states: Sequence[State],
    edges: Sequence[Edge],
    title: str,
    out_path: Path,
    rotation_deg_clockwise: float = 0.0,
    text_rotation_deg_clockwise: float = 0.0,
) -> None:
    if rotation_deg_clockwise != 0.0:
        positions = rotate_positions(positions, degrees_clockwise=rotation_deg_clockwise)

    fig, ax = plt.subplots(figsize=(14, 12))

    for i, j in edges:
        ax.plot(
            [positions[i, 0], positions[j, 0]],
            [positions[i, 1], positions[j, 1]],
            color="lightgray",
            linewidth=0.5,
            alpha=0.5,
            zorder=1,
        )

    ax.scatter(positions[:, 0], positions[:, 1], c="#1f77b4", s=18, alpha=0.9, zorder=2)

    labels = [state_to_label(state) for state in states]
    for (x, y), label in zip(positions, labels):
        ax.text(
            float(x),
            float(y),
            label,
            fontsize=7,
            ha="center",
            va="center",
            rotation=-text_rotation_deg_clockwise,
            color="black",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.55, "pad": 0.25},
            zorder=3,
        )

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=340)
    plt.close(fig)


def plot_spearman_bar(layer_to_metrics: Dict[int, Dict[str, float]], best_layer: int, out_path: Path) -> None:
    layers = sorted(layer_to_metrics.keys())
    rhos = [layer_to_metrics[layer]["spearman_rho"] for layer in layers]

    colors = ["#4c78a8" if layer != best_layer else "#f58518" for layer in layers]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(layers, rhos, color=colors)
    ax.set_xticks(layers)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Spearman rho")
    ax.set_title("Which Layer Encodes the State Space?")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def split_early_mid_late(step_idx: np.ndarray, traj_len: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    frac = step_idx.astype(np.float32) / np.maximum(traj_len.astype(np.float32), 1.0)
    early = frac <= (1.0 / 3.0)
    mid = (frac > (1.0 / 3.0)) & (frac <= (2.0 / 3.0))
    late = frac > (2.0 / 3.0)
    return early, mid, late


def masked_accuracy(mask: np.ndarray, correct: np.ndarray) -> float:
    if not np.any(mask):
        return float("nan")
    return float(np.mean(correct[mask]))


def evaluate_probe(
    probe: nn.Linear,
    x: np.ndarray,
    y_state_idx: np.ndarray,
    step_idx: np.ndarray,
    traj_len: np.ndarray,
    bridge_cross: np.ndarray,
    reference_positions: np.ndarray,
    dist_true: np.ndarray,
    val_idx: np.ndarray,
    device: torch.device,
    spearman_pairs: int,
    seed: int,
) -> Dict[str, float]:
    proj = project_with_probe(probe, x, device=device)

    d_ref = np.linalg.norm(proj[:, None, :] - reference_positions[None, :, :], axis=-1)
    nearest_idx = np.argmin(d_ref, axis=1)
    correct = nearest_idx == y_state_idx

    early, mid, late = split_early_mid_late(step_idx, traj_len)
    bridge = bridge_cross
    non_bridge = ~bridge

    rho, pearson = compute_corr_heldout(
        proj=proj,
        y_state_idx=y_state_idx,
        dist_true=dist_true,
        val_idx=val_idx,
        num_pairs=spearman_pairs,
        seed=seed,
    )

    return {
        "nearest_state_acc": float(np.mean(correct)),
        "spearman_rho": rho,
        "pearson_r": pearson,
        "early_acc": masked_accuracy(early, correct),
        "mid_acc": masked_accuracy(mid, correct),
        "late_acc": masked_accuracy(late, correct),
        "bridge_acc": masked_accuracy(bridge, correct),
        "non_bridge_acc": masked_accuracy(non_bridge, correct),
    }


def train_sep_probe_for_comparison(
    sep_layer5: torch.Tensor,
    dist_true: np.ndarray,
    device: torch.device,
    epochs: int = 2000,
    lr: float = 1e-3,
) -> Tuple[nn.Linear, np.ndarray]:
    x = sep_layer5.to(device)
    d = torch.tensor(dist_true, dtype=torch.float32, device=device)

    mean = d.mean()
    std = d.std().clamp_min(1e-8)
    d_norm = (d - mean) / std

    probe = nn.Linear(x.size(1), 2, bias=True).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        opt.zero_grad(set_to_none=True)
        p = probe(x)
        pred_d = torch.cdist(p, p, p=2)
        loss = torch.mean((pred_d - d_norm) ** 2)
        loss.backward()
        opt.step()
        if epoch == 1 or epoch % 200 == 0 or epoch == epochs:
            print(f"[sep probe] epoch={epoch:4d} loss={float(loss.item()):.6f}")

    with torch.no_grad():
        ref = probe(x).detach().cpu().numpy()
    return probe, ref


def plot_side_by_side(
    out_path: Path,
    edges: Sequence[Edge],
    sep_ref: np.ndarray,
    move_ref: np.ndarray,
    move_proj: np.ndarray,
    move_state_idx: np.ndarray,
    sample_size: int,
    seed: int,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    for i, j in edges:
        ax.plot(
            [sep_ref[i, 0], sep_ref[j, 0]],
            [sep_ref[i, 1], sep_ref[j, 1]],
            color="lightgray",
            linewidth=0.6,
            alpha=0.45,
            zorder=1,
        )
    ax.scatter(sep_ref[:, 0], sep_ref[:, 1], c=np.arange(sep_ref.shape[0]), cmap="viridis", s=35, zorder=2)
    ax.set_title("Old SEP-Trained Probe Layout")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)

    ax = axes[1]
    for i, j in edges:
        ax.plot(
            [move_ref[i, 0], move_ref[j, 0]],
            [move_ref[i, 1], move_ref[j, 1]],
            color="lightgray",
            linewidth=0.6,
            alpha=0.45,
            zorder=1,
        )
    ax.scatter(move_ref[:, 0], move_ref[:, 1], c="black", s=30, alpha=0.8, zorder=2, label="81 reference states")

    rng = np.random.default_rng(seed)
    n = move_proj.shape[0]
    if n > sample_size:
        idx = rng.choice(n, size=sample_size, replace=False)
    else:
        idx = np.arange(n)

    sc = ax.scatter(
        move_proj[idx, 0],
        move_proj[idx, 1],
        c=move_state_idx[idx],
        cmap="turbo",
        s=8,
        alpha=0.55,
        zorder=3,
        label="move-token projections",
    )
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="True state index")
    ax.set_title("New Move-Trained Probe Layout")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)
    ax.legend(loc="best", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def pct(x: float) -> str:
    if np.isnan(x):
        return "nan"
    return f"{100.0 * x:.2f}%"


set_seed = utils.set_seed
confirm_tokenizer_mapping = utils.confirm_tokenizer_mapping
resolve_checkpoint_path = utils.resolve_checkpoint_path
load_model = utils.load_model
enumerate_states = utils.enumerate_states
build_context_ids = utils.build_context_ids
top_disk_per_peg = utils.top_disk_per_peg
decode_move_token = utils.decode_move_token
apply_move_and_next_state = utils.apply_move_and_next_state
greedy_decode_ids = utils.greedy_decode_ids
load_correct_optimal_problems = utils.load_correct_optimal_problems


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("Requested cuda but CUDA is not available")

    vocab = Vocabulary()
    confirm_tokenizer_mapping(vocab)

    ckpt = resolve_checkpoint_path(args.checkpoint, args.n_disks)
    model = load_model(ckpt, args.n_disks, device)

    states, dist_true, edges = build_graph_and_distances(args.n_disks)
    state_to_idx = {s: i for i, s in enumerate(states)}

    eval_path = Path(args.eval_results)
    if not eval_path.exists():
        raise FileNotFoundError(f"Evaluation results not found: {eval_path}")
    problems = load_correct_optimal_problems(eval_path, args.n_disks)
    if not problems:
        raise RuntimeError("No CORRECT_OPTIMAL problems found")

    print(f"[INFO] Loaded model: {ckpt}")
    print(f"[INFO] CORRECT_OPTIMAL problems: {len(problems)}")

    dataset = collect_move_token_dataset(
        model=model,
        problems=problems,
        vocab=vocab,
        state_to_idx=state_to_idx,
        device=device,
        n_disks=args.n_disks,
    )

    x_by_layer = dataset["x_by_layer"]  # type: ignore[assignment]
    y_state_idx = dataset["y_state_idx"]  # type: ignore[assignment]
    step_idx = dataset["step_idx"]  # type: ignore[assignment]
    move_class_idx = dataset["move_class_idx"]  # type: ignore[assignment]

    n_samples = int(y_state_idx.shape[0])
    print(f"[INFO] Move-token samples: {n_samples}")

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n_samples)
    n_val = max(1, int(0.2 * n_samples))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    d_mean = float(dist_true.mean())
    d_std = float(np.clip(dist_true.std(), 1e-8, None))
    dist_norm = (dist_true - d_mean) / d_std

    fixed_goal = tuple(2 for _ in range(args.n_disks))
    sep_acts = extract_sep2_activations(
        model=model,
        states=states,
        goal=fixed_goal,
        vocab=vocab,
        device=device,
    )

    layer_to_positions: Dict[int, np.ndarray] = {}
    layer_to_metrics: Dict[int, Dict[str, float]] = {}
    comparison_rows: List[Dict[str, object]] = []

    for layer in range(1, model.n_layers + 1):
        print(f"\n[INFO] Training move-token probe for layer {layer}")
        x = x_by_layer[layer]

        # Mode 0: pooled baseline.
        probe, final_loss = train_distance_probe_batched(
            x=x,
            y_state_idx=y_state_idx,
            dist_norm=dist_norm,
            train_idx=train_idx,
            device=device,
            lr=args.lr,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed + layer,
        )

        with torch.no_grad():
            ref = probe(sep_acts[layer].to(device)).detach().cpu().numpy()
        layer_to_positions[layer] = ref

        pooled_proj = project_with_probe(probe, x, device=device)
        pooled_acc = nearest_state_accuracy(pooled_proj, ref, y_state_idx)
        pooled_rho, pooled_pearson = compute_corr_heldout(
            proj=pooled_proj,
            y_state_idx=y_state_idx,
            dist_true=dist_true,
            val_idx=val_idx,
            num_pairs=args.spearman_pairs,
            seed=args.seed + 1000 + layer,
        )

        pred_d = np.linalg.norm(ref[:, None, :] - ref[None, :, :], axis=-1)
        rho = spearman_rho(pred_d.reshape(-1), dist_true.reshape(-1))
        pearson = pearson_r(pred_d.reshape(-1), dist_true.reshape(-1))
        stress = stress_metric(pred_d, dist_true)
        layer_to_metrics[layer] = {
            "final_loss": float(final_loss),
            "stress": float(stress),
            "spearman_rho": float(rho),
            "pearson_r": float(pearson),
        }

        # Mode 1: per-step probes with token augmentation only.
        unique_steps = sorted(int(s) for s in np.unique(step_idx))
        per_step_rows: List[Tuple[int, int, float, float, float]] = []
        for step in unique_steps:
            idx = np.where(step_idx == step)[0]
            n_step = int(idx.shape[0])
            if n_step < args.min_step_samples:
                continue

            x_step = x[idx]
            move_step = move_class_idx[idx]
            y_step = y_state_idx[idx]
            local_train_idx = np.arange(n_step, dtype=np.int64)
            x_step_aug = build_token_augmented_features(x_step, move_step)

            step_probe, step_loss = train_distance_probe_batched(
                x=x_step_aug,
                y_state_idx=y_step,
                dist_norm=dist_norm,
                train_idx=local_train_idx,
                device=device,
                lr=args.lr,
                epochs=args.epochs,
                batch_size=args.batch_size,
                seed=args.seed + 2000 + layer * 100 + step,
            )

            sep_aug = build_sep_token_augmented_features(sep_acts[layer].detach().cpu().numpy())
            with torch.no_grad():
                sep_aug_t = torch.tensor(sep_aug, dtype=torch.float32, device=device)
                step_ref = step_probe(sep_aug_t).detach().cpu().numpy()
            step_proj = project_with_probe(step_probe, x_step_aug, device=device)
            step_acc = nearest_state_accuracy(step_proj, step_ref, y_step)

            local_perm = np.random.default_rng(args.seed + 3000 + layer * 100 + step).permutation(n_step)
            local_n_val = max(1, int(0.2 * n_step))
            local_val_idx = local_perm[:local_n_val]
            step_rho, step_pearson = compute_corr_heldout(
                proj=step_proj,
                y_state_idx=y_step,
                dist_true=dist_true,
                val_idx=local_val_idx,
                num_pairs=args.spearman_pairs,
                seed=args.seed + 4000 + layer * 100 + step,
            )
            per_step_rows.append((step, n_step, step_acc, step_rho, float(step_loss), step_pearson))

        print(f"\n[INFO] Per-step probes (layer {layer}, min samples={args.min_step_samples})")
        print("Step | N_samples | Nearest-State Acc | Loss")
        print("-----+-----------+-------------------+------")
        for step, n_step, step_acc, _step_rho, step_loss, _step_pearson in per_step_rows:
            print(f"{step:4d} | {n_step:9d} | {pct(step_acc):>17} | {step_loss:6.4f}")

        per_step_mean_acc = weighted_mean([r[2] for r in per_step_rows], [r[1] for r in per_step_rows])
        per_step_mean_rho = weighted_mean([r[3] for r in per_step_rows], [r[1] for r in per_step_rows])
        per_step_mean_loss = weighted_mean([r[4] for r in per_step_rows], [r[1] for r in per_step_rows])
        per_step_mean_pearson = weighted_mean([r[5] for r in per_step_rows], [r[1] for r in per_step_rows])

        comparison_rows.append(
            {
                "probe_type": "Pooled",
                "layer": layer,
                "nearest_state_acc": float(pooled_acc),
                "spearman_rho": float(pooled_rho),
                "pearson_r": float(pooled_pearson),
                "final_loss": float(final_loss),
            }
        )
        comparison_rows.append(
            {
                "probe_type": "Per-step+Token (mean)",
                "layer": layer,
                "nearest_state_acc": float(per_step_mean_acc),
                "spearman_rho": float(per_step_mean_rho),
                "pearson_r": float(per_step_mean_pearson),
                "final_loss": float(per_step_mean_loss),
            }
        )

        torch.save(
            {
                "probe_state_dict": probe.state_dict(),
                "layer": layer,
                "dist_mean": d_mean,
                "dist_std": d_std,
                "checkpoint": str(ckpt),
            },
            out_dir / f"move_probe_layer{layer}.pt",
        )

    summary = [
        {
            "layer": layer,
            "final_loss": layer_to_metrics[layer]["final_loss"],
            "stress": layer_to_metrics[layer]["stress"],
            "spearman_rho": layer_to_metrics[layer]["spearman_rho"],
            "pearson_r": layer_to_metrics[layer]["pearson_r"],
        }
        for layer in layer_to_metrics
    ]
    summary.sort(key=lambda x: x["spearman_rho"], reverse=True)

    print("\nSummary (sorted by Spearman rho):")
    print(f"{'layer':>5} | {'final_loss':>12} | {'stress':>8} | {'spearman_rho':>12} | {'pearson_r':>10}")
    print("-" * 64)
    for row in summary:
        print(
            f"{row['layer']:5d} | {row['final_loss']:12.6f} | {row['stress']:8.4f} | {row['spearman_rho']:12.4f} | {row['pearson_r']:10.4f}"
        )

    best_layer = summary[0]["layer"]
    print(f"\nBest layer by Spearman rho: {best_layer}")

    print("\nProbe Type      | Layer | Nearest-State Acc | Spearman rho |  Pearson r | Final Loss")
    print("----------------+-------+-------------------+--------------+------------+-----------")
    probe_order = {"Pooled": 0, "Per-step+Token (mean)": 1}
    comparison_rows_sorted = sorted(
        comparison_rows,
        key=lambda r: (int(r["layer"]), probe_order.get(str(r["probe_type"]), 99)),
    )
    for row in comparison_rows_sorted:
        pearson = float(row.get("pearson_r", float("nan")))
        print(
            f"{str(row['probe_type']):16s} | {int(row['layer']):5d} | "
            f"{pct(float(row['nearest_state_acc'])):>17} | {float(row['spearman_rho']):12.4f} | "
            f"{pearson:10.4f} | {float(row['final_loss']):9.4f}"
        )

    print("[INFO] Saving labeled state maps for all layers")
    for layer in sorted(layer_to_positions.keys()):
        title = (
            f"Layer {layer} - Probe Space (labeled states, rotated 60 deg cw) "
            f"(Spearman={layer_to_metrics[layer]['spearman_rho']:.3f}, "
            f"Stress={layer_to_metrics[layer]['stress']:.3f})"
        )
        plot_labeled_state_map(
            positions=layer_to_positions[layer],
            states=states,
            edges=edges,
            title=title,
            out_path=out_dir / f"layer_{layer:02d}_probe_space_labeled.png",
            rotation_deg_clockwise=60.0,
            text_rotation_deg_clockwise=0.0,
        )

    plot_spearman_bar(
        layer_to_metrics=layer_to_metrics,
        best_layer=best_layer,
        out_path=out_dir / "layer_spearman_bar.png",
    )

    print("[INFO] Saving labeled state map for best layer")
    best_raw_positions = layer_to_positions[best_layer]

    plot_labeled_state_map(
        positions=best_raw_positions,
        states=states,
        edges=edges,
        title=f"Best Layer {best_layer} - Probe Space (labeled states, rotated 60 deg cw)",
        out_path=out_dir / f"best_layer_{best_layer:02d}_probe_space_labeled.png",
        rotation_deg_clockwise=60.0,
        text_rotation_deg_clockwise=0.0,
    )

    (out_dir / "summary_sorted_by_spearman.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.save(out_dir / "true_pairwise_graph_distances.npy", dist_true)
    np.save(out_dir / "graph_edges.npy", np.array(edges, dtype=np.int32))
    print(f"[INFO] Wrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
