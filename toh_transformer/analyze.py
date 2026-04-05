"""Extract internal representations from a trained ToHTransformer and visualize
the Sierpinski triangle state-space structure via UMAP/PCA projections."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from collections import deque
from pathlib import Path
from typing import Dict, List, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP

from toh_transformer.data import Vocabulary
from toh_transformer.model import ToHTransformer

StateTuple = Tuple[int, ...]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze ToHTransformer internal representations")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to one model .pt file")
    parser.add_argument(
        "--checkpoints",
        type=str,
        nargs="+",
        default=None,
        help="One or more model .pt files; if provided, runs analysis for each checkpoint",
    )
    parser.add_argument("--n_disks", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="toh_transformer/analysis_output")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--probe_epochs", type=int, default=2000, help="Epochs for linear 2D probe training")
    parser.add_argument("--probe_lr", type=float, default=1e-3, help="Learning rate for linear 2D probes")
    parser.add_argument("--probe_log_every", type=int, default=200, help="Probe training log interval")
    return parser.parse_args()


def resolve_checkpoints(args: argparse.Namespace) -> List[str]:
    if args.checkpoints:
        return args.checkpoints
    if args.checkpoint:
        return [args.checkpoint]
    raise ValueError("Provide --checkpoint or --checkpoints")


def checkpoint_tag(checkpoint_path: str) -> str:
    p = Path(checkpoint_path)
    parent = p.parent.name if p.parent.name else "root"
    return f"{parent}_{p.stem}"


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: str, device: torch.device) -> ToHTransformer:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    vocab = Vocabulary()
    model = ToHTransformer(
        vocab_size=len(vocab),
        max_seq_len=cfg["max_seq_len"],
        n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"],
        d_model=cfg["d_model"],
        d_ff=cfg["d_ff"],
        dropout=0.0,  # no dropout at inference
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# State graph
# ---------------------------------------------------------------------------

def enumerate_all_states(n_disks: int) -> List[StateTuple]:
    return [tuple(s) for s in itertools.product(range(3), repeat=n_disks)]


def top_disk_on_peg(state: StateTuple, peg: int) -> int | None:
    """Return the index of the smallest disk on *peg*, or None if empty.
    Disk index 0 is the smallest disk."""
    for disk_idx in range(len(state)):
        if state[disk_idx] == peg:
            return disk_idx
    return None


def get_neighbors(state: StateTuple) -> List[StateTuple]:
    """Return all states reachable from *state* in one legal move."""
    n = len(state)
    neighbors: List[StateTuple] = []
    for src_peg in range(3):
        top_src = top_disk_on_peg(state, src_peg)
        if top_src is None:
            continue
        for dst_peg in range(3):
            if dst_peg == src_peg:
                continue
            top_dst = top_disk_on_peg(state, dst_peg)
            if top_dst is None or top_src < top_dst:  # smaller index = smaller disk
                new_state = list(state)
                new_state[top_src] = dst_peg
                neighbors.append(tuple(new_state))
    return neighbors


def build_adjacency(states: List[StateTuple]) -> Tuple[List[Tuple[int, int]], Dict[StateTuple, Set[int]]]:
    """Return (edge_list, neighbor_index_sets).  Edges are index pairs (i<j)."""
    state_to_idx = {s: i for i, s in enumerate(states)}
    edges: List[Tuple[int, int]] = []
    neighbor_sets: Dict[StateTuple, Set[int]] = {s: set() for s in states}

    for i, s in enumerate(states):
        for nb in get_neighbors(s):
            j = state_to_idx[nb]
            neighbor_sets[s].add(j)
            if i < j:
                edges.append((i, j))
    return edges, neighbor_sets


def shortest_path_matrix(n_states: int, edges: List[Tuple[int, int]]) -> np.ndarray:
    """BFS-based all-pairs shortest paths.  Returns (n_states, n_states) int array."""
    adj: List[List[int]] = [[] for _ in range(n_states)]
    for i, j in edges:
        adj[i].append(j)
        adj[j].append(i)

    dist = np.full((n_states, n_states), -1, dtype=np.int32)
    for src in range(n_states):
        d = dist[src]
        d[src] = 0
        queue = deque([src])
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if d[v] == -1:
                    d[v] = d[u] + 1
                    queue.append(v)
    return dist


# ---------------------------------------------------------------------------
# Activation extraction
# ---------------------------------------------------------------------------

def extract_activations(
    model: ToHTransformer,
    states: List[StateTuple],
    goal: StateTuple,
    vocab: Vocabulary,
    device: torch.device,
) -> Dict[int, np.ndarray]:
    """Run each state through the model and collect per-layer activations.

    Returns {layer_idx: array of shape (n_states, d_model)}.
    """
    n_layers = model.n_layers
    layers = list(range(1, n_layers + 1))

    # Build input tokens for all states at once
    input_rows: List[List[int]] = []
    for state in states:
        tokens = (
            [vocab.bos_id]
            + [vocab.stoi[f"P{p}"] for p in state]
            + [vocab.sep_id]
            + [vocab.stoi[f"P{p}"] for p in goal]
            + [vocab.sep_id]
        )
        input_rows.append(tokens)

    seq_len = len(input_rows[0])  # should be 2*n_disks + 3
    sep2_pos = seq_len - 1  # last token is second SEP

    input_tensor = torch.tensor(input_rows, dtype=torch.long, device=device)

    # Forward pass in batches to avoid OOM on large models
    batch_size = 128
    accum: Dict[int, List[np.ndarray]] = {l: [] for l in layers}

    for start in range(0, len(states), batch_size):
        batch = input_tensor[start : start + batch_size]
        with torch.no_grad(), model.capture_activations(layers) as cache:
            _ = model(batch)
        for l in layers:
            # cache[l] shape: (batch, seq_len, d_model)
            accum[l].append(cache[l][:, sep2_pos, :].cpu().numpy())

    return {l: np.concatenate(accum[l], axis=0) for l in layers}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    activations: Dict[int, np.ndarray],
    sp_matrix: np.ndarray,
    neighbor_sets: Dict[int, Set[int]],
    n_states: int,
) -> List[dict]:
    """Compute Spearman rho, k-NN recall, and PCA explained variance per layer."""
    # Upper-triangular indices for pairwise distances
    triu_i, triu_j = np.triu_indices(n_states, k=1)
    graph_dists = sp_matrix[triu_i, triu_j].astype(float)

    rows: List[dict] = []
    for layer in sorted(activations.keys()):
        X = activations[layer]

        # Pairwise euclidean distances in activation space
        diff = X[triu_i] - X[triu_j]
        act_dists = np.linalg.norm(diff, axis=1)

        pearson_r, _ = pearsonr(act_dists, graph_dists)
        rho, _ = spearmanr(act_dists, graph_dists)

        # PCA explained variance (top 5 components)
        pca5 = PCA(n_components=min(5, X.shape[1]))
        pca5.fit(X)
        ev = pca5.explained_variance_ratio_

        # k-NN recall (k=3)
        k = 3
        # Full pairwise distance matrix
        full_dists = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
        np.fill_diagonal(full_dists, np.inf)
        knn_indices = np.argpartition(full_dists, k, axis=1)[:, :k]

        recalls = []
        for idx in range(n_states):
            true_nbs = neighbor_sets[idx]
            if len(true_nbs) == 0:
                continue
            found = len(set(knn_indices[idx].tolist()) & true_nbs)
            recalls.append(found / len(true_nbs))
        knn_recall = float(np.mean(recalls))

        rows.append({
            "layer": layer,
            "explained_var_2d": f"{ev[0]:.4f}, {ev[1]:.4f}" if len(ev) >= 2 else str(ev),
            "explained_var_top5": ev.tolist(),
            "pearson_r": float(pearson_r),
            "spearman_rho": rho,
            "knn_recall": knn_recall,
        })

    return rows


def save_metrics(metrics: List[dict], goal: StateTuple, output_dir: Path) -> None:
    """Save per-goal layer metrics to CSV and JSON."""
    goal_str = "".join(str(g) for g in goal)
    csv_path = output_dir / f"metrics_goal_{goal_str}.csv"
    json_path = output_dir / f"metrics_goal_{goal_str}.json"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "layer",
            "explained_var_pc1",
            "explained_var_pc2",
            "pearson_r",
            "spearman_rho",
            "knn_recall",
            "explained_var_top5",
        ])
        for m in metrics:
            pc1 = m["explained_var_top5"][0] if len(m["explained_var_top5"]) >= 1 else np.nan
            pc2 = m["explained_var_top5"][1] if len(m["explained_var_top5"]) >= 2 else np.nan
            writer.writerow([
                m["layer"],
                f"{pc1:.8f}" if not np.isnan(pc1) else "nan",
                f"{pc2:.8f}" if not np.isnan(pc2) else "nan",
                f"{m['pearson_r']:.8f}",
                f"{m['spearman_rho']:.8f}",
                f"{m['knn_recall']:.8f}",
                json.dumps(m["explained_var_top5"]),
            ])

    with json_path.open("w", encoding="utf-8") as f:
        json.dump({"goal": list(goal), "metrics": metrics}, f, indent=2)


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

PRIMARY_COLORS = {0: "#e41a1c", 1: "#377eb8", 2: "#4daf4a"}  # red, blue, green
SHADE_ALPHA = {0: 0.55, 1: 0.78, 2: 1.0}  # lighter/darker shades by state[2]
MARKERS = {0: "o", 1: "s", 2: "D"}  # circle, square, diamond by state[2]


def _scatter_with_edges(
    ax,
    coords_2d: np.ndarray,
    states: List[StateTuple],
    edges: List[Tuple[int, int]],
    color_idx: int,
    title: str,
):
    """Draw scatter + edge overlay, coloring by state[color_idx]."""
    # Draw edges
    for i, j in edges:
        ax.plot(
            [coords_2d[i, 0], coords_2d[j, 0]],
            [coords_2d[i, 1], coords_2d[j, 1]],
            color="#cccccc", linewidth=0.3, zorder=1,
        )

    # Draw points — group by (state[color_idx], state[secondary]) for visual clarity
    secondary_idx = color_idx - 1 if color_idx > 0 else 2
    for primary_val in range(3):
        for sec_val in range(3):
            mask = [
                (states[k][color_idx] == primary_val and states[k][secondary_idx] == sec_val)
                for k in range(len(states))
            ]
            idxs = [k for k, m in enumerate(mask) if m]
            if not idxs:
                continue
            ax.scatter(
                coords_2d[idxs, 0],
                coords_2d[idxs, 1],
                c=PRIMARY_COLORS[primary_val],
                marker=MARKERS[sec_val],
                s=28,
                alpha=SHADE_ALPHA[sec_val],
                edgecolors="k",
                linewidths=0.3,
                zorder=2,
            )

    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])


def make_grid_figure(
    activations: Dict[int, np.ndarray],
    states: List[StateTuple],
    edges: List[Tuple[int, int]],
    goal: StateTuple,
    output_dir: Path,
    umap_params: dict,
    tsne_params: dict,
):
    """Produce grid figure: rows = [UMAP, PCA, t-SNE], columns = layers."""
    layers = sorted(activations.keys())
    n_layers = len(layers)

    fig, axes = plt.subplots(3, n_layers, figsize=(3.5 * n_layers, 10), squeeze=False)
    fig.suptitle(f"Goal = {goal}", fontsize=12, y=1.02)

    for col, layer in enumerate(layers):
        X = activations[layer]

        # UMAP
        umap_proj = UMAP(**umap_params).fit_transform(X)
        _scatter_with_edges(
            axes[0, col], umap_proj, states, edges,
            color_idx=len(states[0]) - 1,  # largest disk
            title=f"UMAP — Layer {layer}",
        )

        # PCA
        pca = PCA(n_components=2)
        pca_proj = pca.fit_transform(X)
        _scatter_with_edges(
            axes[1, col], pca_proj, states, edges,
            color_idx=len(states[0]) - 1,
            title=f"PCA — Layer {layer}",
        )

        # t-SNE
        tsne_proj = TSNE(**tsne_params).fit_transform(X)
        _scatter_with_edges(
            axes[2, col], tsne_proj, states, edges,
            color_idx=len(states[0]) - 1,
            title=f"t-SNE — Layer {layer}",
        )

    fig.tight_layout()
    goal_str = "".join(str(g) for g in goal)
    fig.savefig(output_dir / f"grid_goal_{goal_str}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved grid_goal_{goal_str}.png")


def make_fractal_zoom(
    activations: Dict[int, np.ndarray],
    best_layer: int,
    states: List[StateTuple],
    edges: List[Tuple[int, int]],
    goal: StateTuple,
    output_dir: Path,
    umap_params: dict,
    tsne_params: dict,
):
    """Fractal zoom: color by state[-2] (second-largest disk) for best layer."""
    X = activations[best_layer]
    n_disks = len(states[0])
    secondary_color_idx = n_disks - 2  # second-largest disk

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"Fractal Zoom — Layer {best_layer} — Goal {goal}  (color = disk {secondary_color_idx} peg)",
        fontsize=10,
    )

    # UMAP
    umap_proj = UMAP(**umap_params).fit_transform(X)
    _scatter_with_edges(
        axes[0], umap_proj, states, edges,
        color_idx=secondary_color_idx,
        title="UMAP",
    )

    # PCA
    pca_proj = PCA(n_components=2).fit_transform(X)
    _scatter_with_edges(
        axes[1], pca_proj, states, edges,
        color_idx=secondary_color_idx,
        title="PCA",
    )

    # t-SNE
    tsne_proj = TSNE(**tsne_params).fit_transform(X)
    _scatter_with_edges(
        axes[2], tsne_proj, states, edges,
        color_idx=secondary_color_idx,
        title="t-SNE",
    )

    fig.tight_layout()
    goal_str = "".join(str(g) for g in goal)
    fig.savefig(output_dir / f"fractal_zoom_goal_{goal_str}_layer{best_layer}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fractal_zoom_goal_{goal_str}_layer{best_layer}.png")


# ---------------------------------------------------------------------------
# 2D Probe reconstruction
# ---------------------------------------------------------------------------

def _stress(pred_d: np.ndarray, true_d: np.ndarray) -> float:
    num = float(np.sum((pred_d - true_d) ** 2))
    den = float(np.sum(true_d ** 2))
    if den == 0.0:
        return 0.0
    return float(np.sqrt(num / den))


def train_distance_probe(
    layer_acts: np.ndarray,
    true_dists: np.ndarray,
    device: torch.device,
    epochs: int,
    lr: float,
    log_prefix: str,
    log_every: int,
) -> Tuple[np.ndarray, float]:
    """Train a linear d_model->2 probe to match graph shortest-path distances."""
    x = torch.tensor(layer_acts, dtype=torch.float32, device=device)
    d = torch.tensor(true_dists, dtype=torch.float32, device=device)

    d_norm = (d - d.mean()) / d.std().clamp_min(1e-8)

    probe = nn.Linear(x.shape[1], 2, bias=True).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)

    final_loss = float("nan")
    for epoch in range(1, epochs + 1):
        opt.zero_grad(set_to_none=True)
        p = probe(x)
        pred_d = torch.cdist(p, p, p=2)
        loss = torch.mean((pred_d - d_norm) ** 2)
        loss.backward()
        opt.step()

        final_loss = float(loss.item())
        if epoch == 1 or epoch == epochs or epoch % max(1, log_every) == 0:
            print(f"{log_prefix} epoch={epoch:4d} loss={final_loss:.6f}")

    with torch.no_grad():
        coords = probe(x).detach().cpu().numpy()
    return coords, final_loss


def probe_reconstruct_layers(
    activations: Dict[int, np.ndarray],
    sp_matrix: np.ndarray,
    device: torch.device,
    probe_epochs: int,
    probe_lr: float,
    probe_log_every: int,
) -> Tuple[Dict[int, np.ndarray], List[dict], int]:
    """Train and evaluate one 2D distance probe per layer."""
    true_d = sp_matrix.astype(np.float32)

    layer_coords: Dict[int, np.ndarray] = {}
    rows: List[dict] = []

    for layer in sorted(activations.keys()):
        coords, final_loss = train_distance_probe(
            layer_acts=activations[layer],
            true_dists=true_d,
            device=device,
            epochs=probe_epochs,
            lr=probe_lr,
            log_prefix=f"[Probe L{layer}]",
            log_every=probe_log_every,
        )
        pred_d = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
        rho, _ = spearmanr(pred_d.reshape(-1), true_d.reshape(-1))
        stress = _stress(pred_d, true_d)

        layer_coords[layer] = coords
        rows.append(
            {
                "layer": layer,
                "final_loss": float(final_loss),
                "stress": float(stress),
                "spearman_rho": float(rho),
            }
        )

    rows.sort(key=lambda r: r["spearman_rho"], reverse=True)
    best_layer = rows[0]["layer"]
    return layer_coords, rows, best_layer


def save_probe_metrics(rows: List[dict], goal: StateTuple, output_dir: Path) -> None:
    goal_str = "".join(str(g) for g in goal)
    out_path = output_dir / f"probe_metrics_goal_{goal_str}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


def make_probe_grid_figure(
    layer_coords: Dict[int, np.ndarray],
    metrics_rows: List[dict],
    states: List[StateTuple],
    edges: List[Tuple[int, int]],
    goal: StateTuple,
    output_dir: Path,
) -> None:
    layers = sorted(layer_coords.keys())
    metric_by_layer = {m["layer"]: m for m in metrics_rows}

    n_cols = 3
    n_rows = (len(layers) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows), squeeze=False)

    color_idx = len(states[0]) - 1  # largest disk
    flat_axes = axes.reshape(-1)
    for ax_i, layer in enumerate(layers):
        m = metric_by_layer[layer]
        title = f"Layer {layer} - Spearman={m['spearman_rho']:.3f}, Stress={m['stress']:.3f}"
        _scatter_with_edges(flat_axes[ax_i], layer_coords[layer], states, edges, color_idx=color_idx, title=title)

    for ax in flat_axes[len(layers):]:
        ax.axis("off")

    goal_str = "".join(str(g) for g in goal)
    fig.tight_layout()
    fig.savefig(output_dir / f"probe_grid_goal_{goal_str}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_probe_best_second_largest(
    layer_coords: Dict[int, np.ndarray],
    best_layer: int,
    metrics_rows: List[dict],
    states: List[StateTuple],
    edges: List[Tuple[int, int]],
    goal: StateTuple,
    output_dir: Path,
) -> None:
    by_layer = {m["layer"]: m for m in metrics_rows}
    second_idx = len(states[0]) - 2
    m = by_layer[best_layer]

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    _scatter_with_edges(
        ax,
        layer_coords[best_layer],
        states,
        edges,
        color_idx=second_idx,
        title=f"Best probe layer {best_layer} - Spearman={m['spearman_rho']:.3f}, Stress={m['stress']:.3f}",
    )
    goal_str = "".join(str(g) for g in goal)
    fig.tight_layout()
    fig.savefig(output_dir / f"probe_best_goal_{goal_str}_layer{best_layer}_second_largest.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    checkpoints = resolve_checkpoints(args)
    n_disks = args.n_disks
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vocab = Vocabulary()

    # Verify token IDs match expected mapping
    expected = {"P0": 0, "P1": 1, "P2": 2, "M01": 3, "M02": 4, "M10": 5,
                "M12": 6, "M20": 7, "M21": 8, "BOS": 9, "SEP": 10, "EOS": 11, "PAD": 12}
    for tok, idx in expected.items():
        assert vocab.stoi[tok] == idx, f"Token mismatch: {tok} expected {idx}, got {vocab.stoi[tok]}"
    print("Token IDs verified.")

    # Build state graph
    states = enumerate_all_states(n_disks)
    n_states = len(states)
    print(f"States: {n_states}, n_disks={n_disks}")

    edges, neighbor_sets_by_state = build_adjacency(states)
    print(f"Edges: {len(edges)}")

    # Convert neighbor_sets to index-based for metrics
    state_to_idx = {s: i for i, s in enumerate(states)}
    neighbor_sets_by_idx: Dict[int, Set[int]] = {}
    for s, nbs in neighbor_sets_by_state.items():
        neighbor_sets_by_idx[state_to_idx[s]] = nbs

    sp_matrix = shortest_path_matrix(n_states, edges)
    print(f"Shortest-path matrix computed (diameter = {sp_matrix.max()}).")

    # Goals to probe
    goals: List[StateTuple] = [
        tuple([2] * n_disks),
        tuple([0] * n_disks),
        tuple([1] * n_disks),
    ]

    umap_params = dict(n_neighbors=8, min_dist=0.1, metric="euclidean", random_state=42)
    tsne_params = dict(n_components=2, perplexity=12.0, init="pca", learning_rate="auto", random_state=42)
    for checkpoint in checkpoints:
        run_output_dir = output_dir / checkpoint_tag(checkpoint)
        run_output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nLoading model from {checkpoint} ...")
        model = load_model(checkpoint, device)
        n_layers = model.n_layers
        print(f"Model layers: {n_layers}")
        print(f"Output directory for this checkpoint: {run_output_dir}")

        for goal in goals:
            goal_str = "".join(str(g) for g in goal)
            print(f"\n=== Checkpoint {checkpoint} | Goal {goal} ===")

            # Extract activations
            activations = extract_activations(model, states, goal, vocab, device)

            # Compute metrics
            metrics = compute_metrics(activations, sp_matrix, neighbor_sets_by_idx, n_states)

            print(
                f"\n{'Layer':>6} | {'Expl.Var (PC1, PC2)':>22} | {'Pearson r':>10} | {'Spearman ρ':>11} | {'k-NN Recall':>11}"
            )
            print("-" * 77)
            for m in metrics:
                print(
                    f"  {m['layer']:>4} | {m['explained_var_2d']:>22} | {m['pearson_r']:>10.4f} | {m['spearman_rho']:>11.4f} | {m['knn_recall']:>11.4f}"
                )

            # Detailed PCA variance
            print("\n  PCA explained variance (top 5 components):")
            for m in metrics:
                ev_strs = [f"{v:.4f}" for v in m["explained_var_top5"]]
                print(f"    Layer {m['layer']}: [{', '.join(ev_strs)}]")

            save_metrics(metrics, goal, run_output_dir)
            print(f"  Saved metrics_goal_{goal_str}.csv")
            print(f"  Saved metrics_goal_{goal_str}.json")

            # Grid figure
            make_grid_figure(activations, states, edges, goal, run_output_dir, umap_params, tsne_params)

            # Fractal zoom for best layer by Spearman rho
            best = max(metrics, key=lambda m: m["spearman_rho"])
            print(f"\n  Best layer by Spearman ρ: {best['layer']} (ρ = {best['spearman_rho']:.4f})")
            make_fractal_zoom(
                activations,
                best["layer"],
                states,
                edges,
                goal,
                run_output_dir,
                umap_params,
                tsne_params,
            )

            # Train final 2D distance probes and reconstruct graph geometry by layer
            print("\n  Training final linear 2D probes for graph reconstruction...")
            layer_coords, probe_rows, best_probe_layer = probe_reconstruct_layers(
                activations=activations,
                sp_matrix=sp_matrix,
                device=device,
                probe_epochs=args.probe_epochs,
                probe_lr=args.probe_lr,
                probe_log_every=args.probe_log_every,
            )

            print(f"\n  {'Layer':>6} | {'Final loss':>12} | {'Stress':>8} | {'Spearman ρ':>11}")
            print("  " + "-" * 50)
            for row in probe_rows:
                print(
                    f"  {row['layer']:>4} | {row['final_loss']:>12.6f} | {row['stress']:>8.4f} | {row['spearman_rho']:>11.4f}"
                )

            print(f"\n  Best probe layer: {best_probe_layer}")
            save_probe_metrics(probe_rows, goal, run_output_dir)
            make_probe_grid_figure(layer_coords, probe_rows, states, edges, goal, run_output_dir)
            make_probe_best_second_largest(
                layer_coords,
                best_probe_layer,
                probe_rows,
                states,
                edges,
                goal,
                run_output_dir,
            )
            print(f"  Saved probe_metrics_goal_{goal_str}.json")
            print(f"  Saved probe_grid_goal_{goal_str}.png")
            print(f"  Saved probe_best_goal_{goal_str}_layer{best_probe_layer}_second_largest.png")

    print(f"\nAll outputs saved under {output_dir}/")


if __name__ == "__main__":
    main()
