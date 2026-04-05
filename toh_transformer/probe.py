#!/usr/bin/env python3
"""Probe transformer layer geometry against Towers of Hanoi graph distances.

For each transformer layer, this script learns a linear projection from hidden states
(d_model -> 2) so projected Euclidean distances match shortest-path distances in the
true Towers of Hanoi state graph.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from collections import deque
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

# Support direct execution: python toh_transformer/probe.py
if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from toh_transformer.config import default_model_hparams, max_seq_len_for_disks
from toh_transformer.data import Vocabulary
from toh_transformer.model import ToHTransformer

State = Tuple[int, ...]
Edge = Tuple[int, int]

EXPECTED_TOKEN_MAPPING = {
    "P0": 0,
    "P1": 1,
    "P2": 2,
    "M01": 3,
    "M02": 4,
    "M10": 5,
    "M12": 6,
    "M20": 7,
    "M21": 8,
    "BOS": 9,
    "SEP": 10,
    "EOS": 11,
    "PAD": 12,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe ToH transformer geometry with 2D linear projections")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--n_disks", type=int, default=4, help="Number of ToH disks (default: 4)")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for figures and metrics")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (default: cuda if available else cpu)",
    )
    parser.add_argument("--epochs", type=int, default=2000, help="Probe training epochs per layer")
    parser.add_argument("--lr", type=float, default=1e-3, help="Probe optimizer learning rate")
    return parser.parse_args()


def confirm_tokenizer_mapping() -> Vocabulary:
    vocab = Vocabulary()
    if vocab.stoi != EXPECTED_TOKEN_MAPPING:
        raise ValueError(
            "Tokenizer mapping mismatch with expected ids. "
            f"Expected={EXPECTED_TOKEN_MAPPING}, actual={vocab.stoi}"
        )
    print("[INFO] Tokenizer mapping confirmed against data.py")
    return vocab


def enumerate_states(n_disks: int) -> List[State]:
    return [tuple(s) for s in itertools.product((0, 1, 2), repeat=n_disks)]


def top_disk_per_peg(state: State) -> List[int | None]:
    """Return top disk index per peg (None if empty). Smaller index means smaller disk."""
    tops: List[int | None] = [None, None, None]
    for disk_idx, peg in enumerate(state):
        top = tops[peg]
        if top is None or disk_idx < top:
            tops[peg] = disk_idx
    return tops


def legal_neighbors(state: State) -> List[State]:
    n_disks = len(state)
    tops = top_disk_per_peg(state)
    neighbors: List[State] = []

    for src in range(3):
        src_top = tops[src]
        if src_top is None:
            continue

        for dst in range(3):
            if src == dst:
                continue

            dst_top = tops[dst]
            if dst_top is None or src_top < dst_top:
                new_state = list(state)
                new_state[src_top] = dst
                neighbors.append(tuple(new_state))

    if not neighbors:
        raise RuntimeError(f"State has no legal moves: {state} with n_disks={n_disks}")
    return neighbors


def build_graph_and_distances(n_disks: int) -> Tuple[List[State], np.ndarray, List[Edge], List[List[int]]]:
    states = enumerate_states(n_disks)
    n_states = len(states)
    state_to_idx = {s: i for i, s in enumerate(states)}

    adjacency: List[List[int]] = [[] for _ in range(n_states)]
    edge_set = set()

    for i, state in enumerate(states):
        for nbr in legal_neighbors(state):
            j = state_to_idx[nbr]
            adjacency[i].append(j)
            a, b = sorted((i, j))
            edge_set.add((a, b))

    edges = sorted(edge_set)

    distances = np.full((n_states, n_states), fill_value=np.inf, dtype=np.float32)
    for src in range(n_states):
        distances[src, src] = 0.0
        q = deque([src])
        while q:
            u = q.popleft()
            for v in adjacency[u]:
                if np.isinf(distances[src, v]):
                    distances[src, v] = distances[src, u] + 1.0
                    q.append(v)

    if np.isinf(distances).any():
        raise RuntimeError("Graph appears disconnected; found infinite shortest-path distances")

    return states, distances, edges, adjacency


def build_probe_input(state: State, goal: State, vocab: Vocabulary) -> List[int]:
    return [
        vocab.bos_id,
        *[vocab.stoi[f"P{peg}"] for peg in state],
        vocab.sep_id,
        *[vocab.stoi[f"P{peg}"] for peg in goal],
        vocab.sep_id,
    ]


def load_model(checkpoint_path: Path, n_disks: int, device: torch.device) -> ToHTransformer:
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg: Dict[str, int | float] = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}

    defaults = default_model_hparams(n_disks)
    n_layers = int(cfg.get("n_layers", defaults["n_layers"]))
    n_heads = int(cfg.get("n_heads", defaults["n_heads"]))
    d_model = int(cfg.get("d_model", defaults["d_model"]))
    d_ff = int(cfg.get("d_ff", defaults["d_ff"]))
    dropout = float(cfg.get("dropout", defaults["dropout"]))
    max_seq_len = int(cfg.get("max_seq_len", max_seq_len_for_disks(n_disks)))

    model = ToHTransformer(
        vocab_size=len(Vocabulary()),
        max_seq_len=max_seq_len,
        n_layers=n_layers,
        n_heads=n_heads,
        d_model=d_model,
        d_ff=d_ff,
        dropout=dropout,
    ).to(device)

    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state_dict = ckpt["model_state"]
    elif isinstance(ckpt, dict):
        state_dict = ckpt
    else:
        raise ValueError("Checkpoint format not recognized")

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    print(
        "[INFO] Loaded model:",
        f"layers={model.n_layers}, d_model={model.token_emb.embedding_dim}, max_seq_len={model.max_seq_len}",
    )
    return model


@torch.no_grad()
def extract_layer_activations(
    model: ToHTransformer,
    states: Sequence[State],
    vocab: Vocabulary,
    n_disks: int,
    device: torch.device,
) -> Dict[int, torch.Tensor]:
    layers = list(range(1, model.n_layers + 1))
    sep2_idx = 2 * n_disks + 2
    if n_disks == 4 and sep2_idx != 10:
        raise AssertionError("Unexpected second SEP position for n_disks=4")

    goal = tuple(2 for _ in range(n_disks))

    per_layer_vectors: Dict[int, List[torch.Tensor]] = {layer: [] for layer in layers}

    for state in states:
        tokens = build_probe_input(state=state, goal=goal, vocab=vocab)
        input_ids = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

        with model.capture_activations(layers=layers) as cache:
            _ = model(input_ids)

        for layer in layers:
            hidden = cache[layer][0, sep2_idx, :].detach().cpu()
            per_layer_vectors[layer].append(hidden)

    return {layer: torch.stack(vecs, dim=0) for layer, vecs in per_layer_vectors.items()}


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


def stress_metric(pred_d: np.ndarray, true_d: np.ndarray) -> float:
    num = float(np.sum((pred_d - true_d) ** 2))
    den = float(np.sum(true_d**2))
    if den == 0.0:
        return 0.0
    return math.sqrt(num / den)


def train_probe_for_layer(
    activations: torch.Tensor,
    true_distances: torch.Tensor,
    epochs: int,
    lr: float,
    device: torch.device,
    log_prefix: str,
) -> Tuple[nn.Linear, np.ndarray, float]:
    a = activations.to(device)
    d = true_distances.to(device)

    d_mean = d.mean()
    d_std = d.std().clamp_min(1e-8)
    d_norm = (d - d_mean) / d_std

    probe = nn.Linear(a.shape[1], 2, bias=True).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)

    final_loss = float("nan")
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        p = probe(a)
        pred_d = torch.cdist(p, p, p=2)
        loss = torch.mean((pred_d - d_norm) ** 2)
        loss.backward()
        optimizer.step()

        final_loss = float(loss.item())
        if epoch % 100 == 0 or epoch == 1 or epoch == epochs:
            print(f"{log_prefix} epoch={epoch:4d} loss={final_loss:.6f}")

    with torch.no_grad():
        p = probe(a)
        proj = p.detach().cpu().numpy()

    return probe, proj, final_loss


def plot_layer_projection(
    positions: np.ndarray,
    states: Sequence[State],
    edges: Sequence[Edge],
    color_disk_idx: int,
    title: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))

    for i, j in edges:
        ax.plot(
            [positions[i, 0], positions[j, 0]],
            [positions[i, 1], positions[j, 1]],
            color="lightgray",
            linewidth=0.6,
            alpha=0.5,
            zorder=1,
        )

    colors = np.array(["#d62728", "#1f77b4", "#2ca02c"])
    labels = np.array([state[color_disk_idx] for state in states], dtype=int)
    ax.scatter(positions[:, 0], positions[:, 1], c=colors[labels], s=32, alpha=0.95, zorder=2)

    ax.set_title(title)
    ax.set_xlabel("proj_x")
    ax.set_ylabel("proj_y")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_all_layers_grid(
    layers: Sequence[int],
    layer_to_positions: Dict[int, np.ndarray],
    layer_to_metrics: Dict[int, Dict[str, float]],
    states: Sequence[State],
    edges: Sequence[Edge],
    color_disk_idx: int,
    out_path: Path,
) -> None:
    n_layers = len(layers)
    n_cols = 3
    n_rows = math.ceil(n_layers / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    axes_arr = np.array(axes).reshape(-1)

    colors = np.array(["#d62728", "#1f77b4", "#2ca02c"])
    labels = np.array([state[color_disk_idx] for state in states], dtype=int)

    for ax_idx, layer in enumerate(layers):
        ax = axes_arr[ax_idx]
        pos = layer_to_positions[layer]
        metrics = layer_to_metrics[layer]

        for i, j in edges:
            ax.plot(
                [pos[i, 0], pos[j, 0]],
                [pos[i, 1], pos[j, 1]],
                color="lightgray",
                linewidth=0.5,
                alpha=0.45,
                zorder=1,
            )

        ax.scatter(pos[:, 0], pos[:, 1], c=colors[labels], s=20, alpha=0.95, zorder=2)
        ax.set_title(
            f"Layer {layer} - Spearman={metrics['spearman_rho']:.3f}, Stress={metrics['stress']:.3f}",
            fontsize=10,
        )
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.2)

    for idx in range(n_layers, len(axes_arr)):
        axes_arr[idx].axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
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


def state_to_label(state: State) -> str:
    """Convert internal 0/1/2 peg ids to display labels 1/2/3 (e.g., 1131)."""
    return "".join(str(peg + 1) for peg in state)


def rotate_positions(positions: np.ndarray, degrees_clockwise: float) -> np.ndarray:
    """Rotate positions clockwise around their centroid."""
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
    """Plot a large, fully labeled graph so every point can be matched to a state id."""
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


def main() -> None:
    args = parse_args()

    torch.manual_seed(0)
    np.random.seed(0)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("Requested cuda but CUDA is not available")

    if args.n_disks != 4:
        print("[WARN] This script is designed for n_disks=4, but will run for provided n_disks")

    vocab = confirm_tokenizer_mapping()

    print(f"[INFO] Building state graph for n_disks={args.n_disks}")
    states, true_d_np, edges, _adjacency = build_graph_and_distances(args.n_disks)
    print(f"[INFO] States={len(states)}, edges={len(edges)}")

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = load_model(checkpoint_path=checkpoint_path, n_disks=args.n_disks, device=device)

    print("[INFO] Extracting activations at second SEP token for all layers")
    layer_to_acts = extract_layer_activations(
        model=model,
        states=states,
        vocab=vocab,
        n_disks=args.n_disks,
        device=device,
    )

    true_d_t = torch.tensor(true_d_np, dtype=torch.float32)

    layer_to_positions: Dict[int, np.ndarray] = {}
    layer_to_metrics: Dict[int, Dict[str, float]] = {}

    print("[INFO] Training linear 2D probes per layer")
    for layer in sorted(layer_to_acts.keys()):
        _, positions, final_loss = train_probe_for_layer(
            activations=layer_to_acts[layer],
            true_distances=true_d_t,
            epochs=args.epochs,
            lr=args.lr,
            device=device,
            log_prefix=f"[Layer {layer}]",
        )

        pred_d = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
        rho = spearman_rho(pred_d.reshape(-1), true_d_np.reshape(-1))
        stress = stress_metric(pred_d, true_d_np)

        layer_to_positions[layer] = positions
        layer_to_metrics[layer] = {
            "final_loss": float(final_loss),
            "stress": float(stress),
            "spearman_rho": float(rho),
        }

    summary = [
        {
            "layer": layer,
            "final_loss": layer_to_metrics[layer]["final_loss"],
            "stress": layer_to_metrics[layer]["stress"],
            "spearman_rho": layer_to_metrics[layer]["spearman_rho"],
        }
        for layer in layer_to_metrics
    ]
    summary.sort(key=lambda x: x["spearman_rho"], reverse=True)

    print("\nSummary (sorted by Spearman rho):")
    print(f"{'layer':>5} | {'final_loss':>12} | {'stress':>8} | {'spearman_rho':>12}")
    print("-" * 50)
    for row in summary:
        print(
            f"{row['layer']:5d} | {row['final_loss']:12.6f} | {row['stress']:8.4f} | {row['spearman_rho']:12.4f}"
        )

    best_layer = summary[0]["layer"]
    print(f"\nBest layer by Spearman rho: {best_layer}")

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
            out_path=output_dir / f"layer_{layer:02d}_probe_space_labeled.png",
            rotation_deg_clockwise=60.0,
            text_rotation_deg_clockwise=0.0,
        )

    plot_spearman_bar(
        layer_to_metrics=layer_to_metrics,
        best_layer=best_layer,
        out_path=output_dir / "layer_spearman_bar.png",
    )

    print("[INFO] Saving labeled state map for best layer")
    best_raw_positions = layer_to_positions[best_layer]

    plot_labeled_state_map(
        positions=best_raw_positions,
        states=states,
        edges=edges,
        title=f"Best Layer {best_layer} - Probe Space (labeled states, rotated 60 deg cw)",
        out_path=output_dir / f"best_layer_{best_layer:02d}_probe_space_labeled.png",
        rotation_deg_clockwise=60.0,
        text_rotation_deg_clockwise=0.0,
    )

    (output_dir / "summary_sorted_by_spearman.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.save(output_dir / "true_pairwise_graph_distances.npy", true_d_np)
    np.save(output_dir / "graph_edges.npy", np.array(edges, dtype=np.int32))
    print(f"[INFO] Wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
