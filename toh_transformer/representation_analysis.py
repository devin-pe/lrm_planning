#!/usr/bin/env python3
"""Compare SEP vs move-token representation structure for ToHTransformer.

Outputs:
- Per-disk probe accuracy comparison (SEP LOOCV vs move-token 80/20 split)
- RSA cosine heatmaps (side-by-side)
- Distance-structure Spearman correlation vs true graph distances
- Subspace principal-angle matrices across disk probes (SEP and move)
- JSON and table files plus PNG figures
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import TwoSlopeNorm
from scipy.linalg import qr, subspace_angles
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, train_test_split

if __package__ is None or __package__ == "":
    import sys

    sys.path.append(str(Path(__file__).resolve().parent.parent))

from toh_transformer.config import default_model_hparams, max_seq_len_for_disks
from toh_transformer import utils as utils
from toh_transformer.data import Vocabulary
from toh_transformer.model import ToHTransformer

State = Tuple[int, ...]
Move = Tuple[int, int]

EXPECTED_TOKEN_MAPPING = utils.EXPECTED_TOKEN_MAPPING


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SEP vs move-token representation analysis")
    parser.add_argument("--checkpoint", type=str, default="best.pt")
    parser.add_argument(
        "--eval_results",
        type=str,
        default="toh_transformer/eval_output/evaluation_results.json",
    )
    parser.add_argument("--n_disks", type=int, default=4)
    parser.add_argument("--layer", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default="toh_transformer/representation_analysis_output")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


@torch.no_grad()
@torch.no_grad()
def extract_sep2_activations_layer(
    model: ToHTransformer,
    states: Sequence[State],
    goal: State,
    vocab: Vocabulary,
    layer: int,
    device: torch.device,
    batch_size: int = 128,
) -> np.ndarray:
    rows = [build_context_ids(s, goal, vocab) for s in states]
    sep2_idx = len(rows[0]) - 1
    inp = torch.tensor(rows, dtype=torch.long, device=device)

    chunks: List[np.ndarray] = []
    for st in range(0, len(rows), batch_size):
        batch = inp[st : st + batch_size]
        with model.capture_activations(layers=[layer]) as cache:
            _ = model(batch)
        chunks.append(cache[layer][:, sep2_idx, :].detach().cpu().numpy().astype(np.float32))
    return np.concatenate(chunks, axis=0)


def collect_move_token_activations_layer(
    model: ToHTransformer,
    problems: Sequence[Tuple[State, State]],
    vocab: Vocabulary,
    layer: int,
    n_disks: int,
    state_to_idx: Dict[State, int],
    device: torch.device,
) -> Dict[str, np.ndarray]:
    context_len = 2 * n_disks + 3

    x_list: List[np.ndarray] = []
    y_state_idx: List[int] = []

    kept = 0
    for pi, (start, goal) in enumerate(problems):
        context_ids = build_context_ids(start, goal, vocab)
        generated_ids, eos_seen = greedy_decode_ids(model, context_ids, vocab.eos_id, device)

        if eos_seen and vocab.eos_id in generated_ids:
            move_ids = generated_ids[: generated_ids.index(vocab.eos_id)]
        else:
            move_ids = generated_ids

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

        if not eos_seen or current != goal:
            continue

        full_seq = context_ids + valid_move_ids + [vocab.eos_id]
        if len(full_seq) > model.max_seq_len:
            continue

        seq_t = torch.tensor(full_seq, dtype=torch.long, device=device).unsqueeze(0)
        with torch.no_grad():
            with model.capture_activations(layers=[layer]) as cache:
                _ = model(seq_t)

        move_start = context_len
        move_end = context_len + len(valid_move_ids)
        acts = cache[layer][0, move_start:move_end, :].detach().cpu().numpy().astype(np.float32)

        for i, st_after in enumerate(inter_states):
            x_list.append(acts[i])
            y_state_idx.append(state_to_idx[st_after])

        kept += 1
        if (pi + 1) % 500 == 0:
            print(f"[INFO] Processed {pi + 1}/{len(problems)} problems, kept={kept}")

    if not x_list:
        raise RuntimeError("No move-token activations collected")

    return {
        "x": np.stack(x_list, axis=0),
        "y_state_idx": np.array(y_state_idx, dtype=np.int64),
    }


def make_logreg() -> LogisticRegression:
    try:
        return LogisticRegression(max_iter=3000, multi_class="multinomial", solver="lbfgs")
    except TypeError:
        return LogisticRegression(max_iter=3000)


def sep_loocv_probe_accuracy(x_sep: np.ndarray, states: Sequence[State]) -> List[float]:
    labels = np.array(states, dtype=np.int64)
    loo = LeaveOneOut()

    accs: List[float] = []
    for disk in range(labels.shape[1]):
        y = labels[:, disk]
        correct = 0
        total = 0
        for train_idx, test_idx in loo.split(x_sep):
            clf = make_logreg()
            clf.fit(x_sep[train_idx], y[train_idx])
            pred = int(clf.predict(x_sep[test_idx])[0])
            if pred == int(y[test_idx][0]):
                correct += 1
            total += 1
        accs.append(float(correct / max(total, 1)))
    return accs


def move_probe_accuracy_split(
    x_move: np.ndarray,
    y_state_idx: np.ndarray,
    states: Sequence[State],
    seed: int,
) -> Tuple[List[float], np.ndarray, np.ndarray]:
    state_array = np.array(states, dtype=np.int64)
    y_all = state_array[y_state_idx]

    idx = np.arange(x_move.shape[0])
    train_idx, val_idx = train_test_split(
        idx,
        test_size=0.2,
        random_state=seed,
        stratify=y_state_idx,
    )

    accs: List[float] = []
    for disk in range(y_all.shape[1]):
        y = y_all[:, disk]
        clf = make_logreg()
        clf.fit(x_move[train_idx], y[train_idx])
        accs.append(float(clf.score(x_move[val_idx], y[val_idx])))
    return accs, train_idx, val_idx


def average_representation_by_state(
    x_move: np.ndarray,
    y_state_idx: np.ndarray,
    n_states: int,
) -> Tuple[np.ndarray, np.ndarray]:
    d_model = x_move.shape[1]
    sums = np.zeros((n_states, d_model), dtype=np.float64)
    counts = np.zeros(n_states, dtype=np.int64)

    for i in range(x_move.shape[0]):
        s = int(y_state_idx[i])
        sums[s] += x_move[i]
        counts[s] += 1

    reps = np.full((n_states, d_model), np.nan, dtype=np.float64)
    valid = counts > 0
    reps[valid] = sums[valid] / counts[valid, None]
    return reps.astype(np.float32), counts


def cosine_similarity_matrix(x: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    n = x.shape[0]
    sim = np.full((n, n), np.nan, dtype=np.float32)
    idx = np.where(valid_mask)[0]
    if idx.size == 0:
        return sim

    xv = x[idx]
    norms = np.linalg.norm(xv, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    xv = xv / norms
    sim_v = xv @ xv.T
    sim[np.ix_(idx, idx)] = sim_v.astype(np.float32)
    return sim


def euclidean_distance_matrix(x: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    n = x.shape[0]
    dist = np.full((n, n), np.nan, dtype=np.float32)
    idx = np.where(valid_mask)[0]
    if idx.size == 0:
        return dist
    xv = x[idx]
    d = cdist(xv, xv, metric="euclidean").astype(np.float32)
    dist[np.ix_(idx, idx)] = d
    return dist


def state_order_indices(states: Sequence[State]) -> List[int]:
    return sorted(
        range(len(states)),
        key=lambda i: (states[i][3], states[i][2], states[i][1], states[i][0]),
    )


def reorder_square_matrix(mat: np.ndarray, order: Sequence[int]) -> np.ndarray:
    return mat[np.ix_(order, order)]


def plot_rsa_side_by_side(
    sep_sim_ord: np.ndarray,
    move_sim_ord: np.ndarray,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.8), constrained_layout=True)

    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad(color="#f0f0f0")
    norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)

    mats = [sep_sim_ord, move_sim_ord]
    titles = ["SEP Representation (Layer 5)", "Move-Token Representation (Layer 5)"]

    im = None
    for ax, mat, title in zip(axes, mats, titles):
        masked = np.ma.masked_invalid(mat)
        im = ax.imshow(masked, cmap=cmap, norm=norm, interpolation="nearest", origin="lower")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

        # Major boundaries for disk-3 groups.
        for b in [27, 54]:
            ax.axhline(b - 0.5, color="black", linewidth=1.0)
            ax.axvline(b - 0.5, color="black", linewidth=1.0)

        # Minor boundaries for disk-2 groups.
        for b in [9, 18, 36, 45, 63, 72]:
            ax.axhline(b - 0.5, color="black", linewidth=0.45, alpha=0.5)
            ax.axvline(b - 0.5, color="black", linewidth=0.45, alpha=0.5)

    if im is not None:
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.9, pad=0.02)
        cbar.set_label("Cosine similarity")

    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_accuracy_bars(sep_acc: Sequence[float], move_acc: Sequence[float], out_path: Path) -> None:
    disks = np.arange(4)
    width = 0.36

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ax.bar(disks - width / 2, 100.0 * np.array(sep_acc), width=width, label="SEP")
    ax.bar(disks + width / 2, 100.0 * np.array(move_acc), width=width, label="Move tokens")
    ax.axhline(33.3333, color="black", linestyle="--", linewidth=1.0, alpha=0.8, label="Chance")

    ax.set_xticks(disks)
    ax.set_xticklabels([f"Disk {d}" for d in disks])
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Per-Disk Probe Accuracy")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


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


def build_graph_distance_matrix(states: Sequence[State]) -> np.ndarray:
    n = len(states)
    state_to_idx = {s: i for i, s in enumerate(states)}
    adj: List[List[int]] = [[] for _ in range(n)]
    for i, s in enumerate(states):
        for nbr in legal_neighbors(s):
            adj[i].append(state_to_idx[nbr])

    dist = np.full((n, n), np.inf, dtype=np.float32)
    for src in range(n):
        dist[src, src] = 0.0
        q = [src]
        head = 0
        while head < len(q):
            u = q[head]
            head += 1
            for v in adj[u]:
                if np.isinf(dist[src, v]):
                    dist[src, v] = dist[src, u] + 1.0
                    q.append(v)

    if np.isinf(dist).any():
        raise RuntimeError("State graph is disconnected")
    return dist


def spearman_dist_vs_graph(
    act_dist: np.ndarray,
    graph_dist: np.ndarray,
    valid_mask: np.ndarray,
) -> float:
    rho, _ = corr_dist_vs_graph(act_dist, graph_dist, valid_mask)
    return rho


def corr_dist_vs_graph(
    act_dist: np.ndarray,
    graph_dist: np.ndarray,
    valid_mask: np.ndarray,
) -> Tuple[float, float]:
    """Return (spearman_rho, pearson_r) on the upper-triangular pairs over
    the valid-mask sub-matrix."""
    idx = np.where(valid_mask)[0]
    if idx.size < 3:
        return float("nan"), float("nan")

    d_act = act_dist[np.ix_(idx, idx)]
    d_graph = graph_dist[np.ix_(idx, idx)]

    iu = np.triu_indices(idx.size, k=1)
    a = d_act[iu]
    g = d_graph[iu]

    finite = np.isfinite(a) & np.isfinite(g)
    if np.sum(finite) < 3:
        return float("nan"), float("nan")

    af = a[finite].astype(np.float64)
    gf = g[finite].astype(np.float64)
    rho, _ = spearmanr(af, gf)
    rho = float(rho) if np.isfinite(rho) else float("nan")
    ac = af - af.mean()
    gc = gf - gf.mean()
    denom = float(np.sqrt(np.sum(ac * ac) * np.sum(gc * gc)))
    pearson = float(np.sum(ac * gc) / denom) if denom > 0 else float("nan")
    return rho, pearson


def fit_disk_probes_and_weights(x: np.ndarray, states_or_labels: np.ndarray) -> Tuple[List[float], List[np.ndarray]]:
    accs: List[float] = []
    weights: List[np.ndarray] = []

    for disk in range(4):
        y = states_or_labels[:, disk]
        clf = make_logreg()
        clf.fit(x, y)
        accs.append(float(clf.score(x, y)))
        coef = np.asarray(clf.coef_, dtype=np.float64)
        weights.append(coef)

    return accs, weights


def principal_angle_deg(w_a: np.ndarray, w_b: np.ndarray) -> float:
    # Subspaces are row-spans of probe weight matrices.
    a = w_a.T
    b = w_b.T
    qa, _ = qr(a, mode="economic")
    qb, _ = qr(b, mode="economic")
    ang = subspace_angles(qa, qb)
    return float(np.degrees(np.min(ang)))


def angle_matrix(weights: Sequence[np.ndarray]) -> np.ndarray:
    n = len(weights)
    out = np.full((n, n), np.nan, dtype=np.float64)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            out[i, j] = principal_angle_deg(weights[i], weights[j])
    return out


def format_pct(x: float) -> str:
    return f"{100.0 * x:.2f}%"


def build_accuracy_table(sep_acc: Sequence[float], move_acc: Sequence[float]) -> str:
    chance = 1.0 / 3.0
    lines = []
    lines.append("Position    | Disk 0 Acc | Disk 1 Acc | Disk 2 Acc | Disk 3 Acc | Mean")
    lines.append("------------+------------+------------+------------+------------+------")
    lines.append(
        "SEP         | "
        + " | ".join(f"{format_pct(sep_acc[d]):>10s}" for d in range(4))
        + f" | {format_pct(float(np.mean(sep_acc))):>6s}"
    )
    lines.append(
        "Move tokens | "
        + " | ".join(f"{format_pct(move_acc[d]):>10s}" for d in range(4))
        + f" | {format_pct(float(np.mean(move_acc))):>6s}"
    )
    lines.append(
        "Chance      | "
        + " | ".join(f"{format_pct(chance):>10s}" for _ in range(4))
        + f" | {format_pct(chance):>6s}"
    )
    return "\n".join(lines) + "\n"


def build_spearman_table(sep_rho: float, move_rho: float,
                         sep_pearson: float = float("nan"),
                         move_pearson: float = float("nan")) -> str:
    """Activation-vs-graph distance correlation table (Spearman + Pearson)."""
    lines = []
    lines.append("Position    | Spearman rho |  Pearson r   (activation dist vs graph dist)")
    lines.append("------------+--------------+-----------------------------------------------")
    lines.append(f"SEP         | {sep_rho:12.4f} | {sep_pearson:10.4f}")
    lines.append(f"Move tokens | {move_rho:12.4f} | {move_pearson:10.4f}")
    return "\n".join(lines) + "\n"


def build_angle_table(name: str, mat: np.ndarray) -> str:
    labels = ["Disk 0", "Disk 1", "Disk 2", "Disk 3"]
    lines = []
    lines.append(name)
    lines.append("        | Disk 0 | Disk 1 | Disk 2 | Disk 3")
    lines.append("--------+--------+--------+--------+-------")
    for i, row_name in enumerate(labels):
        cells = []
        for j in range(4):
            if i == j:
                cells.append("  --  ")
            else:
                cells.append(f"{mat[i, j]:6.2f}")
        lines.append(f"{row_name:7s} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]}")
    return "\n".join(lines) + "\n"


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def save_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


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
        raise ValueError("This analysis is currently defined for n_disks=4")

    vocab = Vocabulary()
    confirm_tokenizer_mapping(vocab)

    ckpt_path = resolve_checkpoint_path(args.checkpoint, args.n_disks)
    model = load_model(ckpt_path, args.n_disks, device)

    if args.layer < 1 or args.layer > model.n_layers:
        raise ValueError(f"Layer {args.layer} is out of range [1, {model.n_layers}]")

    eval_path = Path(args.eval_results)
    if not eval_path.exists():
        raise FileNotFoundError(f"Evaluation results not found: {eval_path}")

    states = enumerate_states(args.n_disks)
    state_to_idx = {s: i for i, s in enumerate(states)}
    goal = tuple(2 for _ in range(args.n_disks))

    print(f"[INFO] Loaded model: {ckpt_path}")
    print(f"[INFO] Layer: {args.layer}")

    # SET A: SEP activations (81 x d_model)
    x_sep = extract_sep2_activations_layer(
        model=model,
        states=states,
        goal=goal,
        vocab=vocab,
        layer=args.layer,
        device=device,
    )
    print(f"[INFO] SEP activations: {x_sep.shape}")

    # SET B: move-token activations (N x d_model), plus state labels.
    problems = load_correct_optimal_problems(eval_path, args.n_disks)
    if not problems:
        raise RuntimeError("No CORRECT_OPTIMAL problems found")
    print(f"[INFO] CORRECT_OPTIMAL problems: {len(problems)}")

    move_data = collect_move_token_activations_layer(
        model=model,
        problems=problems,
        vocab=vocab,
        layer=args.layer,
        n_disks=args.n_disks,
        state_to_idx=state_to_idx,
        device=device,
    )
    x_move = move_data["x"]
    y_state_idx = move_data["y_state_idx"]
    print(f"[INFO] Move-token activations: {x_move.shape}")

    move_avg, move_counts = average_representation_by_state(x_move, y_state_idx, n_states=len(states))
    missing_states = int(np.sum(move_counts == 0))
    print(f"[INFO] Move-token averaged states available: {int(np.sum(move_counts > 0))}/81")

    # Part 1: per-disk probe accuracy.
    sep_acc = sep_loocv_probe_accuracy(x_sep, states)
    move_acc, train_idx, val_idx = move_probe_accuracy_split(x_move, y_state_idx, states, seed=args.seed)
    _ = train_idx
    _ = val_idx

    accuracy_table = build_accuracy_table(sep_acc, move_acc)
    print("\n" + accuracy_table)

    # Part 2: RSA heatmaps.
    sep_valid = np.ones(len(states), dtype=bool)
    move_valid = move_counts > 0

    sep_sim = cosine_similarity_matrix(x_sep, sep_valid)
    move_sim = cosine_similarity_matrix(move_avg, move_valid)

    order = state_order_indices(states)
    sep_sim_ord = reorder_square_matrix(sep_sim, order)
    move_sim_ord = reorder_square_matrix(move_sim, order)

    plot_rsa_side_by_side(
        sep_sim_ord=sep_sim_ord,
        move_sim_ord=move_sim_ord,
        out_path=out_dir / "rsa_heatmaps_sep_vs_move.png",
    )

    # Part 3: distance-structure comparison.
    graph_dist = build_graph_distance_matrix(states)
    sep_euc = euclidean_distance_matrix(x_sep, sep_valid)
    move_euc = euclidean_distance_matrix(move_avg, move_valid)

    sep_rho, sep_pearson = corr_dist_vs_graph(sep_euc, graph_dist, sep_valid)
    move_rho, move_pearson = corr_dist_vs_graph(move_euc, graph_dist, move_valid)

    spearman_table = build_spearman_table(sep_rho, move_rho, sep_pearson, move_pearson)
    print(spearman_table)

    # Part 4: subspace orthogonality.
    sep_labels = np.array(states, dtype=np.int64)
    _, sep_weights = fit_disk_probes_and_weights(x_sep, sep_labels)

    move_labels = sep_labels[y_state_idx]
    _, move_weights = fit_disk_probes_and_weights(x_move, move_labels)

    sep_angles = angle_matrix(sep_weights)
    move_angles = angle_matrix(move_weights)

    sep_angle_table = build_angle_table("SEP subspace principal angles (deg)", sep_angles)
    move_angle_table = build_angle_table("Move-token subspace principal angles (deg)", move_angles)
    print(sep_angle_table)
    print(move_angle_table)

    # Figure 2: grouped per-disk accuracy bars.
    plot_accuracy_bars(sep_acc=sep_acc, move_acc=move_acc, out_path=out_dir / "per_disk_accuracy_bars.png")

    # Save tables.
    save_text(out_dir / "table_accuracy.txt", accuracy_table)
    save_text(out_dir / "table_spearman.txt", spearman_table)
    save_text(out_dir / "table_subspace_angles_sep.txt", sep_angle_table)
    save_text(out_dir / "table_subspace_angles_move.txt", move_angle_table)

    # Save structured results.
    results = {
        "config": {
            "checkpoint": str(ckpt_path),
            "eval_results": str(eval_path),
            "n_disks": args.n_disks,
            "layer": args.layer,
            "device": str(device),
            "seed": args.seed,
        },
        "data_summary": {
            "num_states_total": len(states),
            "num_sep_samples": int(x_sep.shape[0]),
            "num_move_samples": int(x_move.shape[0]),
            "num_move_states_with_average": int(np.sum(move_counts > 0)),
            "num_move_states_missing": missing_states,
            "missing_state_indices": [int(i) for i in np.where(move_counts == 0)[0]],
            "move_counts_per_state": move_counts.tolist(),
        },
        "accuracy": {
            "sep": {f"disk_{i}": float(sep_acc[i]) for i in range(4)},
            "move_tokens": {f"disk_{i}": float(move_acc[i]) for i in range(4)},
            "mean_sep": float(np.mean(sep_acc)),
            "mean_move_tokens": float(np.mean(move_acc)),
            "chance": 1.0 / 3.0,
        },
        "distance_structure": {
            "spearman_sep": float(sep_rho),
            "spearman_move_tokens": float(move_rho),
            "pearson_sep": float(sep_pearson),
            "pearson_move_tokens": float(move_pearson),
        },
        "subspace_angles_deg": {
            "sep": sep_angles.tolist(),
            "move_tokens": move_angles.tolist(),
        },
        "state_order_for_heatmap": [int(i) for i in order],
        "states_in_order": [list(states[i]) for i in order],
    }
    save_json(out_dir / "representation_analysis_results.json", results)

    print(f"[INFO] Saved outputs to {out_dir}")


if __name__ == "__main__":
    main()
