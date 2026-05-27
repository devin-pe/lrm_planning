#!/usr/bin/env python3
"""Probe distributed state encoding across multiple recent move-token positions.

Hypothesis: intermediate state information may be distributed across a short
history window of recent positions instead of a single move-token position.
"""

from __future__ import annotations

import argparse
import itertools
import json
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

WINDOW_SIZES = [1, 2, 3, 5, 8, 15]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe distributed state encoding with multi-position windows")
    parser.add_argument("--checkpoint", type=str, default="best.pt")
    parser.add_argument(
        "--eval_results",
        type=str,
        default="toh_transformer/eval_output/evaluation_results.json",
    )
    parser.add_argument("--n_disks", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="toh_transformer/probe_multi_position_output")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
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


def build_window_positions(cur_pos: int, k: int) -> List[int]:
    # Use the rightmost K positions from the available prefix [0..cur_pos].
    positions = list(range(cur_pos + 1))
    window = positions[-k:]
    if len(window) < k:
        window = [0] * (k - len(window)) + window
    return window


def capture_full_sequence_activations(
    model: ToHTransformer,
    full_sequence: Sequence[int],
    device: torch.device,
) -> Dict[int, np.ndarray]:
    layers = list(range(1, model.n_layers + 1))
    seq_t = torch.tensor(full_sequence, dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        with model.capture_activations(layers=layers) as cache:
            _ = model(seq_t)
    return {layer: cache[layer][0].detach().cpu().numpy() for layer in layers}


def collect_multi_position_dataset(
    model: ToHTransformer,
    problems: Sequence[Tuple[State, State]],
    vocab: Vocabulary,
    state_to_idx: Dict[State, int],
    device: torch.device,
    n_disks: int,
    window_sizes: Sequence[int],
) -> Dict[str, object]:
    layers = list(range(1, model.n_layers + 1))
    context_len = 2 * n_disks + 3

    x_by_k_layer: Dict[int, Dict[int, List[np.ndarray]]] = {
        k: {layer: [] for layer in layers} for k in window_sizes
    }
    y_state_idx: List[int] = []
    step_idx: List[int] = []

    # Reference candidates requested by the user: first move step from trajectories starting at each state.
    first_step_ref_by_k_layer: Dict[int, Dict[int, Dict[int, np.ndarray]]] = {
        k: {layer: {} for layer in layers} for k in window_sizes
    }

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

        # Keep only full successful trajectories.
        if not eos_seen or current != goal:
            continue

        full_seq = context_ids + valid_move_ids + [vocab.eos_id]
        if len(full_seq) > model.max_seq_len:
            continue

        layer_acts = capture_full_sequence_activations(model=model, full_sequence=full_seq, device=device)

        start_idx = state_to_idx[start]
        for t, st in enumerate(inter_states, start=1):
            state_idx = state_to_idx[st]
            y_state_idx.append(state_idx)
            step_idx.append(t)

            abs_pos = context_len + (t - 1)
            for k in window_sizes:
                pos_window = build_window_positions(abs_pos, k)
                for layer in layers:
                    vec = layer_acts[layer][pos_window, :].reshape(-1).astype(np.float32, copy=False)
                    x_by_k_layer[k][layer].append(vec)
                    if t == 1 and start_idx not in first_step_ref_by_k_layer[k][layer]:
                        first_step_ref_by_k_layer[k][layer][start_idx] = vec.copy()

        kept += 1
        if (pi + 1) % 500 == 0:
            print(f"[INFO] Processed {pi + 1}/{len(problems)} problems, kept={kept}")

    if len(y_state_idx) == 0:
        raise RuntimeError("No samples collected")

    x_np: Dict[int, Dict[int, np.ndarray]] = {}
    for k in window_sizes:
        x_np[k] = {}
        for layer in layers:
            x_np[k][layer] = np.stack(x_by_k_layer[k][layer], axis=0)

    return {
        "x_by_k_layer": x_np,
        "y_state_idx": np.array(y_state_idx, dtype=np.int64),
        "step_idx": np.array(step_idx, dtype=np.int64),
        "first_step_ref_by_k_layer": first_step_ref_by_k_layer,
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
    in_dim = x.shape[1]
    probe = nn.Linear(in_dim, 2, bias=True).to(device)
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
            final_loss = float(np.mean(losses)) if losses else float("nan")
            print(f"[train] epoch={epoch:3d} loss={final_loss:.6f}")

    return probe, final_loss


def project_with_probe(probe: nn.Linear, x: np.ndarray, device: torch.device, chunk: int = 4096) -> np.ndarray:
    out: List[np.ndarray] = []
    with torch.no_grad():
        for st in range(0, x.shape[0], chunk):
            xb = torch.tensor(x[st : st + chunk], dtype=torch.float32, device=device)
            out.append(probe(xb).detach().cpu().numpy())
    return np.concatenate(out, axis=0)


def nearest_state_predictions(
    proj: np.ndarray,
    ref_positions: np.ndarray,
    ref_state_indices: np.ndarray,
) -> np.ndarray:
    d_ref = np.linalg.norm(proj[:, None, :] - ref_positions[None, :, :], axis=-1)
    nearest_ref_idx = np.argmin(d_ref, axis=1)
    return ref_state_indices[nearest_ref_idx]


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    xc = x.astype(np.float64) - x.mean()
    yc = y.astype(np.float64) - y.mean()
    denom = float(np.sqrt(np.sum(xc * xc) * np.sum(yc * yc)))
    return float(np.sum(xc * yc) / denom) if denom > 0 else float("nan")


def compute_spearman_on_validation(
    proj: np.ndarray,
    y_state_idx: np.ndarray,
    dist_true: np.ndarray,
    val_idx: np.ndarray,
    seed: int,
) -> float:
    rho, _ = compute_corr_on_validation(proj, y_state_idx, dist_true, val_idx, seed)
    return rho


def compute_corr_on_validation(
    proj: np.ndarray,
    y_state_idx: np.ndarray,
    dist_true: np.ndarray,
    val_idx: np.ndarray,
    seed: int,
) -> Tuple[float, float]:
    if val_idx.shape[0] < 2:
        return float("nan"), float("nan")

    n = int(val_idx.shape[0])
    max_exact = 2500

    if n <= max_exact:
        x = proj[val_idx]
        y = y_state_idx[val_idx]
        pred_mat = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=-1).reshape(-1)
        true_mat = dist_true[y][:, y].reshape(-1)
        rho, _ = spearmanr(pred_mat, true_mat)
        rho = float(rho) if np.isfinite(rho) else float("nan")
        return rho, _pearson(pred_mat, true_mat)

    rng = np.random.default_rng(seed)
    num_pairs = 200000
    a = rng.integers(0, n, size=num_pairs)
    b = rng.integers(0, n, size=num_pairs)
    mask = a != b
    if not np.any(mask):
        return float("nan"), float("nan")

    a = a[mask]
    b = b[mask]
    ia = val_idx[a]
    ib = val_idx[b]

    pred = np.linalg.norm(proj[ia] - proj[ib], axis=1)
    true = dist_true[y_state_idx[ia], y_state_idx[ib]]
    rho, _ = spearmanr(pred, true)
    rho = float(rho) if np.isfinite(rho) else float("nan")
    return rho, _pearson(pred, true)


def split_early_mid_late(step_idx: np.ndarray, n_disks: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if n_disks == 4:
        early = (step_idx >= 1) & (step_idx <= 5)
        mid = (step_idx >= 6) & (step_idx <= 10)
        late = (step_idx >= 11) & (step_idx <= 15)
        return early, mid, late

    max_steps = (2**n_disks) - 1
    a = max_steps // 3
    b = (2 * max_steps) // 3
    early = step_idx <= a
    mid = (step_idx > a) & (step_idx <= b)
    late = step_idx > b
    return early, mid, late


def masked_accuracy(mask: np.ndarray, correct: np.ndarray) -> float:
    if not np.any(mask):
        return float("nan")
    return float(np.mean(correct[mask]))


def build_reference_positions(
    probe: nn.Linear,
    sep_layer_acts: np.ndarray,
    first_step_ref_for_states: Dict[int, np.ndarray],
    k: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    del sep_layer_acts, k
    ref_state_indices = np.array(sorted(first_step_ref_for_states.keys()), dtype=np.int64)
    if ref_state_indices.size == 0:
        raise RuntimeError("Strict mode: no first-step reference anchors available")
    refs = np.stack([first_step_ref_for_states[int(i)] for i in ref_state_indices], axis=0).astype(np.float32)
    return project_with_probe(probe, refs, device=device), ref_state_indices


def pct(x: float) -> str:
    if not np.isfinite(x):
        return "nan"
    return f"{100.0 * x:.2f}%"


def plot_best_visualizations(
    out_dir: Path,
    best_proj: np.ndarray,
    best_y_state_idx: np.ndarray,
    ref_positions: np.ndarray,
    ref_state_indices: np.ndarray,
    states: Sequence[State],
    edges: Sequence[Edge],
    best_k: int,
    best_layer: int,
    seed: int,
) -> None:
    state_colors = {
        0: "#e74c3c",  # red
        1: "#3498db",  # blue
        2: "#2ecc71",  # green
    }

    ref_color_values = np.array([
        states[int(i)][3] if len(states[int(i)]) >= 4 else states[int(i)][-1] for i in ref_state_indices
    ])
    ref_colors = np.array([state_colors[int(v)] for v in ref_color_values])

    sample_color_values = np.array([
        states[int(idx)][3] if len(states[int(idx)]) >= 4 else states[int(idx)][-1] for idx in best_y_state_idx
    ])
    sample_colors = np.array([state_colors[int(v)] for v in sample_color_values])

    ref_index_map = {int(s_idx): local_i for local_i, s_idx in enumerate(ref_state_indices)}
    filtered_edges = [
        (ref_index_map[i], ref_index_map[j]) for i, j in edges if i in ref_index_map and j in ref_index_map
    ]

    # Figure 1: reference layout only.
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    for i, j in filtered_edges:
        ax1.plot(
            [ref_positions[i, 0], ref_positions[j, 0]],
            [ref_positions[i, 1], ref_positions[j, 1]],
            color="gray",
            linewidth=1.0,
            alpha=0.5,
            zorder=1,
        )

    ax1.scatter(
        ref_positions[:, 0],
        ref_positions[:, 1],
        c=ref_colors,
        s=80,
        alpha=0.95,
        zorder=2,
    )

    for i, state_idx in enumerate(ref_state_indices):
        state = states[int(state_idx)]
        ax1.text(
            ref_positions[i, 0],
            ref_positions[i, 1],
            str(tuple(state)),
            fontsize=6,
            ha="center",
            va="center",
            color="black",
            zorder=3,
        )

    ax1.set_title(f"Reference Sierpinski Layout (K={best_k}, Layer {best_layer})")
    ax1.set_xlabel("Probe dim 1")
    ax1.set_ylabel("Probe dim 2")
    ax1.set_aspect("equal", adjustable="box")
    ax1.grid(alpha=0.2)
    fig1.tight_layout()
    fig1.savefig(out_dir / "best_k_layer_reference_layout.png", dpi=300)
    plt.close(fig1)

    # Figure 2: subsampled move-token overlay.
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    for i, j in filtered_edges:
        ax2.plot(
            [ref_positions[i, 0], ref_positions[j, 0]],
            [ref_positions[i, 1], ref_positions[j, 1]],
            color="gray",
            linewidth=1.0,
            alpha=0.5,
            zorder=1,
        )

    ax2.scatter(
        ref_positions[:, 0],
        ref_positions[:, 1],
        c=ref_colors,
        s=60,
        edgecolors="black",
        linewidths=0.6,
        alpha=0.98,
        zorder=3,
    )

    rng = np.random.default_rng(seed)
    n_total = best_proj.shape[0]
    n_subsample = min(2000, n_total)
    sel = rng.choice(n_total, size=n_subsample, replace=False)
    ax2.scatter(
        best_proj[sel, 0],
        best_proj[sel, 1],
        c=sample_colors[sel],
        s=3,
        alpha=0.15,
        linewidths=0.0,
        zorder=2,
    )

    ax2.set_title("Move-Token Projections Overlaid on Reference Layout")
    ax2.set_xlabel("Probe dim 1")
    ax2.set_ylabel("Probe dim 2")
    ax2.set_aspect("equal", adjustable="box")
    ax2.grid(alpha=0.2)
    fig2.tight_layout()
    fig2.savefig(out_dir / "best_k_layer_overlay_subsample.png", dpi=300)
    plt.close(fig2)

    # Figure 3: density heatmap with references on top.
    fig3, ax3 = plt.subplots(figsize=(10, 8))
    hb = ax3.hexbin(
        best_proj[:, 0],
        best_proj[:, 1],
        gridsize=80,
        cmap="Greys",
        mincnt=1,
        linewidths=0.0,
        alpha=0.9,
        zorder=1,
    )
    cbar = fig3.colorbar(hb, ax=ax3)
    cbar.set_label("Move-token density")

    for i, j in filtered_edges:
        ax3.plot(
            [ref_positions[i, 0], ref_positions[j, 0]],
            [ref_positions[i, 1], ref_positions[j, 1]],
            color="gray",
            linewidth=1.0,
            alpha=0.5,
            zorder=2,
        )

    ax3.scatter(
        ref_positions[:, 0],
        ref_positions[:, 1],
        c=ref_colors,
        s=60,
        edgecolors="black",
        linewidths=0.6,
        alpha=0.98,
        zorder=3,
    )

    ax3.set_title("Density of Move-Token Projections")
    ax3.set_xlabel("Probe dim 1")
    ax3.set_ylabel("Probe dim 2")
    ax3.set_aspect("equal", adjustable="box")
    ax3.grid(alpha=0.2)
    fig3.tight_layout()
    fig3.savefig(out_dir / "best_k_layer_density_hexbin.png", dpi=300)
    plt.close(fig3)


def plot_k_curve(out_path: Path, k_values: Sequence[int], acc_values: Sequence[float], layer: int) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_values, acc_values, marker="o", linewidth=2.0, color="#1f77b4")
    ax.set_xticks(list(k_values))
    ax.set_xlabel("Window size K")
    ax.set_ylabel("Nearest-state accuracy")
    ax.set_title(f"Does adding positions help? (best layer={layer})")
    ax.grid(alpha=0.25)
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
        print("[WARN] Script is tuned for n_disks=4. Proceeding with general fallbacks where possible.")

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

    print(f"[INFO] Model checkpoint: {ckpt}")
    print(f"[INFO] CORRECT_OPTIMAL problems: {len(problems)}")

    dataset = collect_multi_position_dataset(
        model=model,
        problems=problems,
        vocab=vocab,
        state_to_idx=state_to_idx,
        device=device,
        n_disks=args.n_disks,
        window_sizes=WINDOW_SIZES,
    )

    x_by_k_layer = dataset["x_by_k_layer"]
    y_state_idx = dataset["y_state_idx"]
    step_idx = dataset["step_idx"]
    first_step_ref_by_k_layer = dataset["first_step_ref_by_k_layer"]

    y_state_idx = np.asarray(y_state_idx, dtype=np.int64)
    step_idx = np.asarray(step_idx, dtype=np.int64)

    n_samples = int(y_state_idx.shape[0])
    print(f"[INFO] Samples (all steps pooled): {n_samples}")

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

    early_mask, mid_mask, late_mask = split_early_mid_late(step_idx, args.n_disks)

    results_rows: List[Dict[str, float]] = []
    best_by_k: Dict[int, Dict[str, float]] = {}

    best_global: Optional[Dict[str, object]] = None

    for k in WINDOW_SIZES:
        k_rows: List[Dict[str, float]] = []
        for layer in range(1, model.n_layers + 1):
            print(f"\n[INFO] Training probe for K={k}, layer={layer}")
            x = x_by_k_layer[k][layer]

            probe, final_loss = train_distance_probe_batched(
                x=x,
                y_state_idx=y_state_idx,
                dist_norm=dist_norm,
                train_idx=train_idx,
                device=device,
                lr=args.lr,
                epochs=args.epochs,
                batch_size=args.batch_size,
                seed=args.seed + 1000 * k + layer,
            )

            ref_positions = build_reference_positions(
                probe=probe,
                sep_layer_acts=sep_acts[layer],
                first_step_ref_for_states=first_step_ref_by_k_layer[k][layer],
                k=k,
                device=device,
            )
            ref_positions, ref_state_indices = ref_positions

            proj = project_with_probe(probe, x, device=device)
            available_anchor_mask = np.isin(y_state_idx, ref_state_indices)
            nearest = nearest_state_predictions(proj, ref_positions, ref_state_indices)
            correct = nearest == y_state_idx

            acc = masked_accuracy(available_anchor_mask, correct)
            rho, pearson = compute_corr_on_validation(
                proj=proj,
                y_state_idx=y_state_idx,
                dist_true=dist_true,
                val_idx=val_idx,
                seed=args.seed + 5000 + 100 * k + layer,
            )
            early_acc = masked_accuracy(early_mask & available_anchor_mask, correct)
            mid_acc = masked_accuracy(mid_mask & available_anchor_mask, correct)
            late_acc = masked_accuracy(late_mask & available_anchor_mask, correct)
            n_ref_states = int(ref_state_indices.shape[0])
            anchor_coverage = n_ref_states / len(states)

            row = {
                "K": float(k),
                "layer": float(layer),
                "nearest_state_acc": acc,
                "spearman_rho": rho,
                "pearson_r": pearson,
                "early_acc": early_acc,
                "mid_acc": mid_acc,
                "late_acc": late_acc,
                "final_loss": float(final_loss),
                "n_ref_states": float(n_ref_states),
                "anchor_coverage": float(anchor_coverage),
            }
            results_rows.append(row)
            k_rows.append(row)

            try:
                torch.save(
                    probe.state_dict(),
                    out_dir / f"multi_position_K{k}_layer_{layer:02d}.pt",
                )
            except Exception as e:
                print(f"[WARN] Could not save probe for K={k} layer={layer}: {e}")

            if np.isfinite(acc) and (best_global is None or acc > float(best_global["nearest_state_acc"])):
                best_global = {
                    "K": k,
                    "layer": layer,
                    "nearest_state_acc": acc,
                    "spearman_rho": rho,
                    "pearson_r": pearson,
                    "proj": proj,
                    "y_state_idx": y_state_idx.copy(),
                    "ref_positions": ref_positions,
                    "ref_state_indices": ref_state_indices.copy(),
                }

        k_rows_sorted = sorted(k_rows, key=lambda r: (r["nearest_state_acc"], r["spearman_rho"]), reverse=True)
        best_by_k[k] = k_rows_sorted[0]

    print("\nK  | Layer | Nearest-State Acc | Spearman rho | Early Acc | Mid Acc | Late Acc | Anchor Coverage")
    print("---+-------+-------------------+--------------+-----------+---------+---------+----------------")
    rows_sorted = sorted(results_rows, key=lambda r: (int(r["K"]), int(r["layer"])))
    for r in rows_sorted:
        print(
            f"{int(r['K']):2d} | {int(r['layer']):5d} | {pct(r['nearest_state_acc']):>17} | "
            f"{r['spearman_rho']:12.4f} | {pct(r['early_acc']):>9} | {pct(r['mid_acc']):>7} | {pct(r['late_acc']):>8} | "
            f"{pct(r['anchor_coverage']):>14}"
        )

    print("\nK  | Best Layer | Nearest-State Acc | Spearman rho")
    print("---+------------+-------------------+--------------")
    for k in WINDOW_SIZES:
        r = best_by_k[k]
        print(
            f"{k:2d} | {int(r['layer']):10d} | {pct(r['nearest_state_acc']):>17} | {r['spearman_rho']:12.4f}"
        )

    if best_global is None:
        raise RuntimeError("No best result computed")

    best_k = int(best_global["K"])
    best_layer = int(best_global["layer"])
    print(
        "\n[INFO] Best combination by nearest-state accuracy: "
        f"K={best_k}, layer={best_layer}, acc={pct(float(best_global['nearest_state_acc']))}, "
        f"rho={float(best_global['spearman_rho']):.4f}"
    )

    plot_best_visualizations(
        out_dir=out_dir,
        best_proj=np.asarray(best_global["proj"]),
        best_y_state_idx=np.asarray(best_global["y_state_idx"]),
        ref_positions=np.asarray(best_global["ref_positions"]),
        ref_state_indices=np.asarray(best_global["ref_state_indices"]),
        states=states,
        edges=edges,
        best_k=best_k,
        best_layer=best_layer,
        seed=args.seed,
    )

    acc_curve = [best_by_k[k]["nearest_state_acc"] if int(best_by_k[k]["layer"]) == best_layer else float("nan") for k in WINDOW_SIZES]
    # The curve should use one fixed layer: recompute from full table for the best layer.
    acc_curve = []
    for k in WINDOW_SIZES:
        row = next(r for r in rows_sorted if int(r["K"]) == k and int(r["layer"]) == best_layer)
        acc_curve.append(row["nearest_state_acc"])

    plot_k_curve(
        out_path=out_dir / "k_vs_accuracy_best_layer.png",
        k_values=WINDOW_SIZES,
        acc_values=acc_curve,
        layer=best_layer,
    )

    payload = {
        "config": {
            "checkpoint": str(ckpt),
            "eval_results": str(eval_path),
            "n_disks": args.n_disks,
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "window_sizes": WINDOW_SIZES,
        },
        "results": rows_sorted,
        "best_by_k": {str(k): best_by_k[k] for k in WINDOW_SIZES},
        "best_global": {
            "K": best_k,
            "layer": best_layer,
            "nearest_state_acc": float(best_global["nearest_state_acc"]),
            "spearman_rho": float(best_global["spearman_rho"]),
            "pearson_r": float(best_global["pearson_r"]),
            "n_ref_states": int(np.asarray(best_global["ref_state_indices"]).shape[0]),
            "anchor_coverage": float(np.asarray(best_global["ref_state_indices"]).shape[0] / len(states)),
        },
    }
    (out_dir / "multi_position_probe_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"[INFO] Wrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
