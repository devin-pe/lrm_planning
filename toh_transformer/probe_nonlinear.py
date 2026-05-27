#!/usr/bin/env python3
"""Probe nonlinear state encoding at single move-token positions.

This script trains small MLP probes on single-position residual activations to
check whether intermediate state is encoded nonlinearly.
"""

from __future__ import annotations

import argparse
import copy
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
from scipy.stats import spearmanr

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

PROBE_SPECS: Dict[str, str] = {
    "small_mlp": "Small MLP",
    "large_mlp": "Large MLP",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/evaluate nonlinear probes on move-token activations")
    parser.add_argument("--checkpoint", type=str, default="best.pt")
    parser.add_argument(
        "--eval_results",
        type=str,
        default="toh_transformer/eval_output/evaluation_results.json",
    )
    parser.add_argument("--n_disks", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="toh_transformer/probe_nonlinear_output")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--spearman_pairs", type=int, default=200000)
    parser.add_argument("--linear_baseline_acc", type=float, default=0.012)
    parser.add_argument("--linear_baseline_rho", type=float, default=0.289)
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

    x_by_layer: Dict[int, List[np.ndarray]] = {l: [] for l in layers}
    y_state_idx: List[int] = []
    step_idx: List[int] = []

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

        # Keep only trajectories that reached goal with EOS and legal moves.
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

        for s_i, st in enumerate(inter_states):
            y_state_idx.append(state_to_idx[st])
            step_idx.append(s_i + 1)
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
        "n_kept_trajectories": kept,
    }


def make_probe(probe_key: str, d_model: int) -> nn.Module:
    if probe_key == "small_mlp":
        return nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )
    if probe_key == "large_mlp":
        return nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )
    raise ValueError(f"Unknown probe type: {probe_key}")


def pairwise_distance_loss(
    probe: nn.Module,
    x_t: torch.Tensor,
    y_t: torch.Tensor,
    d_t: torch.Tensor,
    idx_np: np.ndarray,
    batch_size: int,
) -> float:
    if idx_np.shape[0] < 2:
        return float("nan")

    losses: List[float] = []
    with torch.no_grad():
        for st in range(0, idx_np.shape[0], batch_size):
            b_idx_np = idx_np[st : st + batch_size]
            if b_idx_np.shape[0] < 2:
                continue
            b_idx = torch.tensor(b_idx_np, dtype=torch.long, device=x_t.device)
            xb = x_t[b_idx]
            sb = y_t[b_idx]
            true_d = d_t[sb][:, sb]
            pred = probe(xb)
            pred_d = torch.cdist(pred, pred, p=2)
            loss = torch.mean((pred_d - true_d) ** 2)
            losses.append(float(loss.item()))

    if not losses:
        return float("nan")
    return float(np.mean(losses))


def train_probe_with_validation(
    probe: nn.Module,
    x: np.ndarray,
    y_state_idx: np.ndarray,
    dist_norm: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    device: torch.device,
    lr: float,
    epochs: int,
    batch_size: int,
    seed: int,
) -> Tuple[nn.Module, List[float], List[float], int, float]:
    probe = probe.to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)

    x_t = torch.tensor(x, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_state_idx, dtype=torch.long, device=device)
    d_t = torch.tensor(dist_norm, dtype=torch.float32, device=device)

    rng = np.random.default_rng(seed)

    best_epoch = 1
    best_val_loss = float("inf")
    best_state = copy.deepcopy(probe.state_dict())
    train_curve: List[float] = []
    val_curve: List[float] = []

    for epoch in range(1, epochs + 1):
        probe.train()
        order = rng.permutation(train_idx)
        batch_losses: List[float] = []

        for st in range(0, order.shape[0], batch_size):
            b_idx_np = order[st : st + batch_size]
            if b_idx_np.shape[0] < 2:
                continue
            b_idx = torch.tensor(b_idx_np, dtype=torch.long, device=device)
            xb = x_t[b_idx]
            sb = y_t[b_idx]
            true_d = d_t[sb][:, sb]

            opt.zero_grad(set_to_none=True)
            pred = probe(xb)
            pred_d = torch.cdist(pred, pred, p=2)
            loss = torch.mean((pred_d - true_d) ** 2)
            loss.backward()
            opt.step()

            batch_losses.append(float(loss.item()))

        train_loss = float(np.mean(batch_losses)) if batch_losses else float("nan")

        probe.eval()
        val_order = rng.permutation(val_idx)
        val_loss = pairwise_distance_loss(
            probe=probe,
            x_t=x_t,
            y_t=y_t,
            d_t=d_t,
            idx_np=val_order,
            batch_size=batch_size,
        )

        train_curve.append(train_loss)
        val_curve.append(val_loss)

        if np.isfinite(val_loss) and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(probe.state_dict())

        if epoch == 1 or epoch % 20 == 0 or epoch == epochs:
            print(f"[train] epoch={epoch:3d} train_loss={train_loss:.6f} val_loss={val_loss:.6f} best_val={best_val_loss:.6f}")

    probe.load_state_dict(best_state)
    probe.eval()
    return probe, train_curve, val_curve, best_epoch, best_val_loss


def project_with_probe(probe: nn.Module, x: np.ndarray, device: torch.device, chunk: int = 4096) -> np.ndarray:
    out: List[np.ndarray] = []
    probe.eval()
    with torch.no_grad():
        for st in range(0, x.shape[0], chunk):
            xb = torch.tensor(x[st : st + chunk], dtype=torch.float32, device=device)
            out.append(probe(xb).detach().cpu().numpy())
    return np.concatenate(out, axis=0)


def nearest_state_accuracy_and_predictions(
    proj: np.ndarray,
    ref_positions: np.ndarray,
    y_state_idx: np.ndarray,
) -> Tuple[float, np.ndarray]:
    d_ref = np.linalg.norm(proj[:, None, :] - ref_positions[None, :, :], axis=-1)
    nearest_idx = np.argmin(d_ref, axis=1)
    acc = float(np.mean(nearest_idx == y_state_idx))
    return acc, nearest_idx


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

    ia = val_idx[a[mask]]
    ib = val_idx[b[mask]]

    pred = np.linalg.norm(proj[ia] - proj[ib], axis=1)
    true = dist_true[y_state_idx[ia], y_state_idx[ib]]

    rho, _ = spearmanr(pred, true)
    if not np.isfinite(rho):
        rho = float("nan")
    pred_c = pred.astype(np.float64) - pred.mean()
    true_c = true.astype(np.float64) - true.mean()
    denom = math.sqrt(float(np.sum(pred_c * pred_c) * np.sum(true_c * true_c)))
    pearson = float(np.sum(pred_c * true_c) / denom) if denom > 0 else float("nan")
    return float(rho), pearson


def split_early_mid_late(step_idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    early = (step_idx >= 1) & (step_idx <= 5)
    mid = (step_idx >= 6) & (step_idx <= 10)
    late = (step_idx >= 11) & (step_idx <= 15)
    return early, mid, late


def masked_accuracy(mask: np.ndarray, correct: np.ndarray) -> float:
    if not np.any(mask):
        return float("nan")
    return float(np.mean(correct[mask]))


def pct(x: float) -> str:
    if np.isnan(x):
        return "nan"
    return f"{100.0 * x:.2f}%"


def state_to_label(state: State) -> str:
    return "".join(str(peg + 1) for peg in state)


def rotate_positions(positions: np.ndarray, degrees_clockwise: float) -> np.ndarray:
    theta = -np.deg2rad(degrees_clockwise)
    c, s = np.cos(theta), np.sin(theta)
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
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_spearman_bar(layer_to_spearman: Dict[int, float], best_layer: int, out_path: Path, title: str) -> None:
    layers = sorted(layer_to_spearman.keys())
    rhos = [layer_to_spearman[layer] for layer in layers]
    colors = ["#4c78a8" if layer != best_layer else "#f58518" for layer in layers]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(layers, rhos, color=colors)
    ax.set_xticks(layers)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Spearman rho")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_best_scatter(
    out_path: Path,
    points: np.ndarray,
    y_state_idx: np.ndarray,
    states: Sequence[State],
    ref_positions: np.ndarray,
    edges: Sequence[Edge],
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))

    for i, j in edges:
        ax.plot(
            [ref_positions[i, 0], ref_positions[j, 0]],
            [ref_positions[i, 1], ref_positions[j, 1]],
            color="lightgray",
            linewidth=0.7,
            alpha=0.55,
            zorder=1,
        )

    disk_colors = np.array([states[idx][3] for idx in y_state_idx], dtype=np.int64)
    sc = ax.scatter(points[:, 0], points[:, 1], c=disk_colors, cmap="viridis", s=10, alpha=0.45, zorder=2)
    ax.scatter(ref_positions[:, 0], ref_positions[:, 1], c="black", s=18, alpha=0.8, zorder=3)

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Largest disk position (state[3])")

    ax.set_title(title)
    ax.set_xlabel("Probe dim 1")
    ax.set_ylabel("Probe dim 2")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_probe_comparison_bars(
    out_path: Path,
    small_best: Dict[str, float],
    large_best: Dict[str, float],
    linear_acc: float,
    linear_rho: float,
) -> None:
    names = ["Linear", "Small MLP", "Large MLP"]
    acc = np.array([linear_acc, small_best["nearest_state_acc"], large_best["nearest_state_acc"]], dtype=np.float32)
    rho = np.array([linear_rho, small_best["spearman_rho"], large_best["spearman_rho"]], dtype=np.float32)

    x = np.arange(len(names))
    w = 0.36

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w / 2, acc, width=w, label="Nearest-state accuracy", color="#4c78a8")
    ax.bar(x + w / 2, rho, width=w, label="Spearman rho", color="#f58518")

    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Best-layer comparison: nonlinear probes vs linear baseline")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_training_curves(
    out_path: Path,
    curves: Dict[str, Dict[str, object]],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)

    probe_keys = ["small_mlp", "large_mlp"]
    for ax, probe_key in zip(axes, probe_keys):
        row = curves[probe_key]
        train_curve = np.array(row["train_curve"], dtype=np.float32)
        val_curve = np.array(row["val_curve"], dtype=np.float32)
        best_epoch = int(row["best_epoch"])
        layer = int(row["layer"])

        epochs = np.arange(1, train_curve.shape[0] + 1)
        ax.plot(epochs, train_curve, label="Train loss", color="#4c78a8")
        ax.plot(epochs, val_curve, label="Val loss", color="#f58518")
        ax.axvline(best_epoch, color="gray", linestyle="--", linewidth=1.0, alpha=0.8)

        ax.set_title(f"{PROBE_SPECS[probe_key]} (layer {layer})")
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.25)

    axes[0].set_ylabel("Pairwise distance MSE")
    axes[0].legend(loc="best")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


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

    if args.n_disks != 4:
        raise ValueError("This script is defined for n_disks=4 (early/mid/late bins are fixed to 1-5/6-10/11-15)")

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

    x_by_layer = dataset["x_by_layer"]
    y_state_idx = dataset["y_state_idx"]
    step_idx = dataset["step_idx"]

    y_state_idx = np.asarray(y_state_idx, dtype=np.int64)
    step_idx = np.asarray(step_idx, dtype=np.int64)

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
    sep_acts_t = extract_sep2_activations(
        model=model,
        states=states,
        goal=fixed_goal,
        vocab=vocab,
        device=device,
    )
    sep_acts = {layer: sep_acts_t[layer].detach().cpu().numpy() for layer in sep_acts_t}

    early, mid, late = split_early_mid_late(step_idx)

    results_rows: List[Dict[str, object]] = []
    best_by_probe: Dict[str, Dict[str, object]] = {}
    best_global: Optional[Dict[str, object]] = None

    for probe_key in PROBE_SPECS:
        probe_rows: List[Dict[str, object]] = []

        for layer in range(1, model.n_layers + 1):
            print(f"\n[INFO] Training {PROBE_SPECS[probe_key]} for layer {layer}")

            x = x_by_layer[layer]
            d_model = int(x.shape[1])
            probe = make_probe(probe_key, d_model=d_model)

            trained_probe, train_curve, val_curve, best_epoch, best_val_loss = train_probe_with_validation(
                probe=probe,
                x=x,
                y_state_idx=y_state_idx,
                dist_norm=dist_norm,
                train_idx=train_idx,
                val_idx=val_idx,
                device=device,
                lr=args.lr,
                epochs=args.epochs,
                batch_size=args.batch_size,
                seed=args.seed + layer * 100 + (0 if probe_key == "small_mlp" else 10000),
            )

            with torch.no_grad():
                sep_x = torch.tensor(sep_acts[layer], dtype=torch.float32, device=device)
                ref_positions = trained_probe(sep_x).detach().cpu().numpy()

            ref_pairwise = np.linalg.norm(ref_positions[:, None, :] - ref_positions[None, :, :], axis=-1)
            layout_spearman, _ = spearmanr(ref_pairwise.reshape(-1), dist_true.reshape(-1))
            if not np.isfinite(layout_spearman):
                layout_spearman = float("nan")
            _ref_p = ref_pairwise.reshape(-1).astype(np.float64)
            _ref_p -= _ref_p.mean()
            _true_p = dist_true.reshape(-1).astype(np.float64)
            _true_p -= _true_p.mean()
            _denom = math.sqrt(float(np.sum(_ref_p * _ref_p) * np.sum(_true_p * _true_p)))
            layout_pearson = float(np.sum(_ref_p * _true_p) / _denom) if _denom > 0 else float("nan")

            proj_all = project_with_probe(trained_probe, x, device=device)
            proj_val = proj_all[val_idx]
            y_val = y_state_idx[val_idx]
            step_val = step_idx[val_idx]

            acc, nearest = nearest_state_accuracy_and_predictions(proj_val, ref_positions, y_val)
            correct = nearest == y_val

            early_val = (step_val >= 1) & (step_val <= 5)
            mid_val = (step_val >= 6) & (step_val <= 10)
            late_val = (step_val >= 11) & (step_val <= 15)

            rho, pearson = compute_corr_heldout(
                proj=proj_all,
                y_state_idx=y_state_idx,
                dist_true=dist_true,
                val_idx=val_idx,
                num_pairs=args.spearman_pairs,
                seed=args.seed + 20000 + layer,
            )

            row: Dict[str, object] = {
                "probe_key": probe_key,
                "probe_name": PROBE_SPECS[probe_key],
                "layer": layer,
                "nearest_state_acc": float(acc),
                "spearman_rho": float(rho),
                "pearson_r": float(pearson),
                "early_acc": masked_accuracy(early_val, correct),
                "mid_acc": masked_accuracy(mid_val, correct),
                "late_acc": masked_accuracy(late_val, correct),
                "val_loss": float(best_val_loss),
                "layout_spearman": float(layout_spearman),
                "layout_pearson": float(layout_pearson),
                "best_epoch": int(best_epoch),
                "train_curve": [float(v) for v in train_curve],
                "val_curve": [float(v) for v in val_curve],
                "ref_positions": ref_positions,
                "val_proj": proj_val,
                "val_y": y_val,
                "trained_probe": trained_probe,
            }

            probe_rows.append(row)
            results_rows.append(row)

            if best_global is None or float(row["nearest_state_acc"]) > float(best_global["nearest_state_acc"]):
                best_global = row

        probe_rows_sorted = sorted(
            probe_rows,
            key=lambda r: (float(r["nearest_state_acc"]), float(r["spearman_rho"])),
            reverse=True,
        )
        best_by_probe[probe_key] = probe_rows_sorted[0]

        layer_to_positions: Dict[int, np.ndarray] = {}
        layer_to_layout_spearman: Dict[int, float] = {}
        for r in probe_rows:
            layer = int(r["layer"])
            layer_to_positions[layer] = np.asarray(r["ref_positions"])
            layer_to_layout_spearman[layer] = float(r["layout_spearman"])

        print(f"[INFO] Saving labeled state maps for all layers ({PROBE_SPECS[probe_key]})")
        for layer in sorted(layer_to_positions.keys()):
            title = (
                f"{PROBE_SPECS[probe_key]} - Layer {layer} - Probe Space (labeled states, rotated 60 deg cw) "
                f"(Spearman={layer_to_layout_spearman[layer]:.3f})"
            )
            plot_labeled_state_map(
                positions=layer_to_positions[layer],
                states=states,
                edges=edges,
                title=title,
                out_path=out_dir / f"{probe_key}_layer_{layer:02d}_probe_space_labeled.png",
                rotation_deg_clockwise=60.0,
                text_rotation_deg_clockwise=0.0,
            )

        best_layer_for_probe = int(best_by_probe[probe_key]["layer"])
        plot_spearman_bar(
            layer_to_spearman=layer_to_layout_spearman,
            best_layer=best_layer_for_probe,
            out_path=out_dir / f"{probe_key}_layer_spearman_bar.png",
            title=f"{PROBE_SPECS[probe_key]}: Which Layer Encodes the State Space?",
        )

        plot_labeled_state_map(
            positions=layer_to_positions[best_layer_for_probe],
            states=states,
            edges=edges,
            title=(
                f"{PROBE_SPECS[probe_key]} - Best Layer {best_layer_for_probe} "
                f"- Probe Space (labeled states, rotated 60 deg cw)"
            ),
            out_path=out_dir / f"{probe_key}_best_layer_{best_layer_for_probe:02d}_probe_space_labeled.png",
            rotation_deg_clockwise=60.0,
            text_rotation_deg_clockwise=0.0,
        )

    print("\nProbe       | Layer | Nearest-State Acc | Spearman rho | Early Acc | Mid Acc | Late Acc | Val Loss")
    print("------------+-------+-------------------+--------------+-----------+---------+----------+---------")
    order = {"small_mlp": 0, "large_mlp": 1}
    rows_sorted = sorted(results_rows, key=lambda r: (order[str(r["probe_key"])], int(r["layer"])))
    for r in rows_sorted:
        print(
            f"{str(r['probe_name']):11s} | {int(r['layer']):5d} | {pct(float(r['nearest_state_acc'])):>17} | "
            f"{float(r['spearman_rho']):12.4f} | {pct(float(r['early_acc'])):>9} | {pct(float(r['mid_acc'])):>7} | "
            f"{pct(float(r['late_acc'])):>8} | {float(r['val_loss']):7.4f}"
        )

    print("\nProbe       | Best Layer | Nearest-State Acc | Spearman rho")
    print("------------+------------+-------------------+--------------")
    for probe_key in ["small_mlp", "large_mlp"]:
        r = best_by_probe[probe_key]
        print(
            f"{str(r['probe_name']):11s} | {int(r['layer']):10d} | {pct(float(r['nearest_state_acc'])):>17} | "
            f"{float(r['spearman_rho']):12.4f}"
        )

    if best_global is None:
        raise RuntimeError("No probe result was produced")

    print(
        "\n[INFO] Best combination by nearest-state accuracy: "
        f"probe={best_global['probe_name']}, layer={best_global['layer']}, "
        f"acc={pct(float(best_global['nearest_state_acc']))}, rho={float(best_global['spearman_rho']):.4f}"
    )

    plot_best_scatter(
        out_path=out_dir / "best_probe_scatter.png",
        points=np.asarray(best_global["val_proj"]),
        y_state_idx=np.asarray(best_global["val_y"]),
        states=states,
        ref_positions=np.asarray(best_global["ref_positions"]),
        edges=edges,
        title=f"Best nonlinear probe: {best_global['probe_name']} (layer {best_global['layer']})",
    )

    plot_probe_comparison_bars(
        out_path=out_dir / "probe_comparison_bars.png",
        small_best=best_by_probe["small_mlp"],
        large_best=best_by_probe["large_mlp"],
        linear_acc=float(args.linear_baseline_acc),
        linear_rho=float(args.linear_baseline_rho),
    )

    curve_payload = {
        "small_mlp": {
            "layer": int(best_by_probe["small_mlp"]["layer"]),
            "best_epoch": int(best_by_probe["small_mlp"]["best_epoch"]),
            "train_curve": best_by_probe["small_mlp"]["train_curve"],
            "val_curve": best_by_probe["small_mlp"]["val_curve"],
        },
        "large_mlp": {
            "layer": int(best_by_probe["large_mlp"]["layer"]),
            "best_epoch": int(best_by_probe["large_mlp"]["best_epoch"]),
            "train_curve": best_by_probe["large_mlp"]["train_curve"],
            "val_curve": best_by_probe["large_mlp"]["val_curve"],
        },
    }
    plot_training_curves(out_path=out_dir / "training_curves_best_layers.png", curves=curve_payload)

    serializable_rows: List[Dict[str, object]] = []
    for r in rows_sorted:
        serializable_rows.append(
            {
                "probe_key": r["probe_key"],
                "probe_name": r["probe_name"],
                "layer": int(r["layer"]),
                "nearest_state_acc": float(r["nearest_state_acc"]),
                "spearman_rho": float(r["spearman_rho"]),
                "pearson_r": float(r["pearson_r"]),
                "layout_spearman": float(r["layout_spearman"]),
                "layout_pearson": float(r["layout_pearson"]),
                "early_acc": float(r["early_acc"]),
                "mid_acc": float(r["mid_acc"]),
                "late_acc": float(r["late_acc"]),
                "val_loss": float(r["val_loss"]),
                "best_epoch": int(r["best_epoch"]),
                "train_curve": r["train_curve"],
                "val_curve": r["val_curve"],
            }
        )
        try:
            torch.save(
                r["trained_probe"].state_dict(),
                out_dir / f"{r['probe_key']}_layer_{int(r['layer']):02d}.pt",
            )
        except Exception as e:
            print(f"[WARN] Could not save probe state dict for {r['probe_key']} layer {r['layer']}: {e}")

    payload = {
        "config": {
            "checkpoint": str(ckpt),
            "eval_results": str(eval_path),
            "n_disks": int(args.n_disks),
            "epochs": int(args.epochs),
            "lr": float(args.lr),
            "batch_size": int(args.batch_size),
            "seed": int(args.seed),
            "linear_baseline_acc": float(args.linear_baseline_acc),
            "linear_baseline_rho": float(args.linear_baseline_rho),
            "train_size": int(train_idx.shape[0]),
            "val_size": int(val_idx.shape[0]),
        },
        "results": serializable_rows,
        "best_by_probe": {
            k: {
                "probe_name": str(best_by_probe[k]["probe_name"]),
                "layer": int(best_by_probe[k]["layer"]),
                "nearest_state_acc": float(best_by_probe[k]["nearest_state_acc"]),
                "spearman_rho": float(best_by_probe[k]["spearman_rho"]),
                "pearson_r": float(best_by_probe[k]["pearson_r"]),
                "best_epoch": int(best_by_probe[k]["best_epoch"]),
                "val_loss": float(best_by_probe[k]["val_loss"]),
            }
            for k in ["small_mlp", "large_mlp"]
        },
        "best_global": {
            "probe_name": str(best_global["probe_name"]),
            "layer": int(best_global["layer"]),
            "nearest_state_acc": float(best_global["nearest_state_acc"]),
            "spearman_rho": float(best_global["spearman_rho"]),
            "best_epoch": int(best_global["best_epoch"]),
            "val_loss": float(best_global["val_loss"]),
        },
    }

    (out_dir / "probe_nonlinear_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[INFO] Wrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
