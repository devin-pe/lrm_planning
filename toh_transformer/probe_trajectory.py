#!/usr/bin/env python3
"""Probe trajectory state tracking from residual stream activations.

This script:
1) Retrains and saves a layer-5 linear 2D probe from second-SEP activations.
2) Uses evaluate.py results to select CORRECT_OPTIMAL problems.
3) Greedily decodes each selected problem and captures per-layer activations at move-token positions.
4) Applies the frozen layer-5 probe to every layer's activations to test transfer.
5) Reports nearest-state decoding metrics and saves figures + full per-step JSON.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

if __package__ is None or __package__ == "":
    import sys

    sys.path.append(str(Path(__file__).resolve().parent.parent))

from toh_transformer.config import default_model_hparams, max_seq_len_for_disks
from toh_transformer.data import Vocabulary
from toh_transformer.model import ToHTransformer

State = Tuple[int, ...]
Move = Tuple[int, int]
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
    parser = argparse.ArgumentParser(
        description="Probe whether the residual stream tracks current ToH state through generation"
    )
    parser.add_argument("--checkpoint", type=str, default="best.pt", help="Path to model checkpoint")
    parser.add_argument(
        "--probe_checkpoint",
        type=str,
        default="",
        help="Path to save probe checkpoint (default: <output_dir>/probe_layer5.pt)",
    )
    parser.add_argument("--n_disks", type=int, default=4, help="Number of disks")
    parser.add_argument("--layer", type=int, default=5, help="Primary layer index")
    parser.add_argument(
        "--eval_results",
        type=str,
        default="toh_transformer/eval_output/evaluation_results.json",
        help="Path to evaluate.py JSON output used to filter CORRECT_OPTIMAL problems",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="toh_transformer/probe_trajectory_output",
        help="Directory for figures and result files",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Compute device",
    )
    parser.add_argument(
        "--max_trajectories",
        type=int,
        default=-1,
        help="Optional cap for selected CORRECT_OPTIMAL trajectories; <=0 means all",
    )
    return parser.parse_args()


def confirm_tokenizer_mapping(vocab: Vocabulary) -> None:
    if vocab.stoi != EXPECTED_TOKEN_MAPPING:
        raise ValueError(
            "Tokenizer mapping mismatch. "
            f"Expected={EXPECTED_TOKEN_MAPPING}, actual={vocab.stoi}"
        )


def resolve_checkpoint_path(path_str: str, n_disks: int) -> Path:
    p = Path(path_str)
    if p.exists():
        return p

    if path_str == "best.pt":
        candidates = [
            Path(f"toh_transformer/checkpoints/n{n_disks}/best.pt"),
            Path(f"toh_transformer/checkpoints/flat_train_3-4-6__test_5/best.pt"),
        ]
        for c in candidates:
            if c.exists():
                print(f"[INFO] Checkpoint 'best.pt' not found; using {c}")
                return c

    raise FileNotFoundError(f"Checkpoint not found: {path_str}")


def load_model(checkpoint_path: Path, n_disks: int, device: torch.device) -> ToHTransformer:
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}

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
    return model


def enumerate_states(n_disks: int) -> List[State]:
    return [tuple(s) for s in itertools.product((0, 1, 2), repeat=n_disks)]


def top_disk_per_peg(state: State) -> List[Optional[int]]:
    tops: List[Optional[int]] = [None, None, None]
    for disk_idx, peg in enumerate(state):
        top = tops[peg]
        if top is None or disk_idx < top:
            tops[peg] = disk_idx
    return tops


def legal_neighbors(state: State) -> List[State]:
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
                nxt = list(state)
                nxt[src_top] = dst
                neighbors.append(tuple(nxt))

    if not neighbors:
        raise RuntimeError(f"State has no legal neighbors: {state}")
    return neighbors


def build_graph_and_distances(n_disks: int) -> Tuple[List[State], np.ndarray, List[Edge]]:
    states = enumerate_states(n_disks)
    n_states = len(states)
    state_to_idx = {s: i for i, s in enumerate(states)}

    adjacency: List[List[int]] = [[] for _ in range(n_states)]
    edge_set = set()

    for i, s in enumerate(states):
        for nbr in legal_neighbors(s):
            j = state_to_idx[nbr]
            adjacency[i].append(j)
            a, b = sorted((i, j))
            edge_set.add((a, b))

    dist = np.full((n_states, n_states), np.inf, dtype=np.float32)
    for src in range(n_states):
        dist[src, src] = 0.0
        q = deque([src])
        while q:
            u = q.popleft()
            for v in adjacency[u]:
                if np.isinf(dist[src, v]):
                    dist[src, v] = dist[src, u] + 1.0
                    q.append(v)

    if np.isinf(dist).any():
        raise RuntimeError("Found infinite graph distances; graph should be connected")

    return states, dist, sorted(edge_set)


def build_context_ids(start: State, goal: State, vocab: Vocabulary) -> List[int]:
    return [
        vocab.bos_id,
        *[vocab.stoi[f"P{p}"] for p in start],
        vocab.sep_id,
        *[vocab.stoi[f"P{p}"] for p in goal],
        vocab.sep_id,
    ]


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
    seq_len = len(rows[0])
    sep2_idx = seq_len - 1

    inp = torch.tensor(rows, dtype=torch.long, device=device)
    per_layer: Dict[int, List[torch.Tensor]] = {l: [] for l in layers}

    for st in range(0, len(rows), batch_size):
        batch = inp[st : st + batch_size]
        with model.capture_activations(layers=layers) as cache:
            _ = model(batch)
        for l in layers:
            per_layer[l].append(cache[l][:, sep2_idx, :].detach().cpu())

    return {l: torch.cat(per_layer[l], dim=0) for l in layers}


def train_layer5_probe(
    activations: torch.Tensor,
    true_distances: torch.Tensor,
    device: torch.device,
    epochs: int = 2000,
    lr: float = 1e-3,
) -> Tuple[nn.Linear, float, float, np.ndarray]:
    a = activations.to(device)
    d = true_distances.to(device)

    d_mean = float(d.mean().item())
    d_std = float(d.std().clamp_min(1e-8).item())
    d_norm = (d - d_mean) / d_std

    probe = nn.Linear(a.shape[1], 2, bias=True).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        opt.zero_grad(set_to_none=True)
        p = probe(a)
        pred_d = torch.cdist(p, p, p=2)
        loss = torch.mean((pred_d - d_norm) ** 2)
        loss.backward()
        opt.step()

        if epoch == 1 or epoch % 200 == 0 or epoch == epochs:
            print(f"[Probe layer 5] epoch={epoch:4d} loss={float(loss.item()):.6f}")

    with torch.no_grad():
        ref = probe(a).detach().cpu().numpy()

    return probe, d_mean, d_std, ref


@torch.no_grad()
def greedy_decode_ids(
    model: ToHTransformer,
    context_ids: Sequence[int],
    eos_id: int,
    device: torch.device,
) -> Tuple[List[int], bool]:
    seq = torch.tensor(context_ids, dtype=torch.long, device=device).unsqueeze(0)
    out: List[int] = []

    while seq.size(1) < model.max_seq_len:
        logits = model(seq)
        next_id = int(torch.argmax(logits[:, -1, :], dim=-1).item())
        out.append(next_id)
        seq = torch.cat([seq, torch.tensor([[next_id]], dtype=torch.long, device=device)], dim=1)
        if next_id == eos_id:
            return out, True

    return out, False


def decode_move_token(tok: str) -> Optional[Move]:
    if len(tok) != 3 or tok[0] != "M":
        return None
    if tok[1] not in "012" or tok[2] not in "012":
        return None
    src = int(tok[1])
    dst = int(tok[2])
    if src == dst:
        return None
    return src, dst


def apply_move_and_next_state(state: State, move: Move) -> Optional[State]:
    src, dst = move
    tops = top_disk_per_peg(state)
    src_top = tops[src]
    dst_top = tops[dst]

    if src_top is None:
        return None
    if dst_top is not None and src_top > dst_top:
        return None

    nxt = list(state)
    nxt[src_top] = dst
    return tuple(nxt)


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

    with model.capture_activations(layers=layers) as cache:
        _ = model(seq_t)

    out: Dict[int, np.ndarray] = {}
    move_start = context_len
    move_end = context_len + n_move_tokens
    for l in layers:
        # Shape: (n_moves, d_model)
        out[l] = cache[l][0, move_start:move_end, :].detach().cpu().numpy()
    return out


def load_eval_problems(eval_results_path: Path, n_disks: int) -> List[Dict[str, object]]:
    if not eval_results_path.exists():
        raise FileNotFoundError(f"Evaluation results not found: {eval_results_path}")

    with eval_results_path.open("r", encoding="utf-8") as f:
        rows = json.load(f)

    if not isinstance(rows, list):
        raise ValueError("Evaluation results JSON must be a list")

    problems: List[Dict[str, object]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue

        start_raw = row.get("start")
        goal_raw = row.get("goal")
        category = str(row.get("category", "UNKNOWN"))
        if not isinstance(start_raw, list) or not isinstance(goal_raw, list):
            continue
        if len(start_raw) != n_disks or len(goal_raw) != n_disks:
            continue

        start = tuple(int(x) for x in start_raw)
        goal = tuple(int(x) for x in goal_raw)
        problems.append({"start": start, "goal": goal, "category": category})

    # Preserve order while deduplicating.
    seen = set()
    dedup: List[Dict[str, object]] = []
    for item in problems:
        pair = (item["start"], item["goal"])
        if pair in seen:
            continue
        seen.add(pair)
        dedup.append(item)
    return dedup


def pick_diverse_trajectories(lengths: List[int], max_count: int = 12) -> List[int]:
    eligible = [i for i, ln in enumerate(lengths) if ln > 0]
    if not eligible:
        return []
    if len(eligible) <= max_count:
        return eligible

    eligible_sorted = sorted(eligible, key=lambda i: lengths[i])
    picks = np.linspace(0, len(eligible_sorted) - 1, num=max_count)
    idxs = sorted(set(int(round(x)) for x in picks))
    return [eligible_sorted[i] for i in idxs]


def plot_trajectory_map(
    out_path: Path,
    reference_positions: np.ndarray,
    edges: Sequence[Edge],
    chosen_traj_ids: Sequence[int],
    traj_pred_positions_primary: Dict[int, np.ndarray],
) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))

    for i, j in edges:
        ax.plot(
            [reference_positions[i, 0], reference_positions[j, 0]],
            [reference_positions[i, 1], reference_positions[j, 1]],
            color="lightgray",
            linewidth=0.6,
            alpha=0.35,
            zorder=1,
        )

    ax.scatter(
        reference_positions[:, 0],
        reference_positions[:, 1],
        s=18,
        c="#bdbdbd",
        alpha=0.55,
        zorder=2,
        label="Reference states",
    )

    cmap = plt.get_cmap("tab20")
    for k, traj_id in enumerate(chosen_traj_ids):
        pos = traj_pred_positions_primary.get(traj_id)
        if pos is None or len(pos) == 0:
            continue

        color = cmap(k % 20)
        ax.plot(pos[:, 0], pos[:, 1], color=color, linewidth=1.6, alpha=0.95, zorder=3)
        ax.scatter(pos[:, 0], pos[:, 1], s=18, color=color, alpha=0.95, zorder=4)

        for i in range(len(pos) - 1):
            ax.annotate(
                "",
                xy=(pos[i + 1, 0], pos[i + 1, 1]),
                xytext=(pos[i, 0], pos[i, 1]),
                arrowprops={"arrowstyle": "->", "color": color, "lw": 0.8, "alpha": 0.7},
                zorder=5,
            )

        ax.scatter(pos[0, 0], pos[0, 1], s=70, marker="o", edgecolor="black", facecolor=color, zorder=6)
        ax.scatter(pos[-1, 0], pos[-1, 1], s=80, marker="s", edgecolor="black", facecolor=color, zorder=6)

    ax.set_title("Probed Trajectories on Reference State Layout")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_accuracy_by_step(
    out_path: Path,
    per_step_layer_correct: Dict[int, Dict[int, Tuple[int, int]]],
    n_layers: int,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))

    for layer in range(1, n_layers + 1):
        step_dict = per_step_layer_correct.get(layer, {})
        if not step_dict:
            continue
        steps = sorted(step_dict.keys())
        ys = [step_dict[s][0] / max(step_dict[s][1], 1) for s in steps]
        ax.plot(steps, ys, marker="o", markersize=3, linewidth=1.5, label=f"Layer {layer}")

    ax.set_title("Nearest-State Accuracy by Move Position")
    ax.set_xlabel("Move step index")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(ncol=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_state_error_heatmap(
    out_path: Path,
    reference_positions: np.ndarray,
    edges: Sequence[Edge],
    state_correct: np.ndarray,
    state_total: np.ndarray,
    title: str,
) -> None:
    acc = np.divide(state_correct, np.maximum(state_total, 1), where=state_total >= 0)

    fig, ax = plt.subplots(figsize=(9, 7.5))
    for i, j in edges:
        ax.plot(
            [reference_positions[i, 0], reference_positions[j, 0]],
            [reference_positions[i, 1], reference_positions[j, 1]],
            color="lightgray",
            linewidth=0.5,
            alpha=0.4,
            zorder=1,
        )

    sc = ax.scatter(
        reference_positions[:, 0],
        reference_positions[:, 1],
        c=acc,
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        s=70,
        edgecolor="black",
        linewidth=0.2,
        zorder=2,
    )
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("Decode accuracy when state appears")

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def format_pct(x: float) -> str:
    return f"{100.0 * x:.2f}%"


def correct_to_sep_position(
    activations: torch.Tensor,
    model: ToHTransformer,
    move_start_pos: int,
    sep_pos: int,
) -> torch.Tensor:
    """Shift move-position activations into the probe's SEP-position frame.

    corrected[t] = act[t] - pos_emb[pos_t] + pos_emb[sep_pos]
    """
    n_steps = activations.size(0)
    if n_steps == 0:
        return activations

    pos_ids = torch.arange(move_start_pos, move_start_pos + n_steps, device=activations.device, dtype=torch.long)
    pos_w = model.pos_emb.weight
    delta = pos_w[sep_pos].unsqueeze(0) - pos_w[pos_ids]
    return activations + delta


def main() -> None:
    args = parse_args()

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("Requested cuda but CUDA is not available")

    vocab = Vocabulary()
    confirm_tokenizer_mapping(vocab)

    ckpt_path = resolve_checkpoint_path(args.checkpoint, args.n_disks)
    model = load_model(ckpt_path, args.n_disks, device)

    if args.layer < 1 or args.layer > model.n_layers:
        raise ValueError(f"--layer must be in [1, {model.n_layers}]")

    print(f"[INFO] Loaded model from {ckpt_path}")
    print(f"[INFO] Model layers={model.n_layers}, d_model={model.token_emb.embedding_dim}, max_seq_len={model.max_seq_len}")

    # ------------------------------------------------------------------
    # Re-train and save layer-5 probe on second-SEP activations.
    # ------------------------------------------------------------------
    states, true_dist_np, edges = build_graph_and_distances(args.n_disks)
    state_to_idx = {s: i for i, s in enumerate(states)}

    fixed_goal = tuple(2 for _ in range(args.n_disks))
    print("[INFO] Extracting second-SEP activations for all states")
    layer_sep_acts = extract_sep2_activations(
        model=model,
        states=states,
        goal=fixed_goal,
        vocab=vocab,
        device=device,
        batch_size=128,
    )

    if 5 not in layer_sep_acts:
        raise ValueError("Model does not have layer 5; cannot train requested probe")

    print("[INFO] Training layer-5 probe (2000 epochs, Adam lr=1e-3)")
    probe, dist_mean, dist_std, reference_positions = train_layer5_probe(
        activations=layer_sep_acts[5],
        true_distances=torch.tensor(true_dist_np, dtype=torch.float32),
        device=device,
        epochs=2000,
        lr=1e-3,
    )
    probe.eval()
    for p in probe.parameters():
        p.requires_grad_(False)

    reference_pos_map = {
        tuple(int(x) for x in s): (float(reference_positions[i, 0]), float(reference_positions[i, 1]))
        for i, s in enumerate(states)
    }

    probe_ckpt_path = Path(args.probe_checkpoint) if args.probe_checkpoint else (out_dir / "probe_layer5.pt")
    torch.save(
        {
            "probe_state_dict": probe.state_dict(),
            "dist_mean": float(dist_mean),
            "dist_std": float(dist_std),
            "reference_positions": reference_pos_map,
            "layer": 5,
            "n_disks": args.n_disks,
            "checkpoint": str(ckpt_path),
        },
        probe_ckpt_path,
    )
    print(f"[INFO] Saved probe checkpoint to {probe_ckpt_path}")

    # ------------------------------------------------------------------
    # Load problems from evaluate.py output and split by optimality.
    # ------------------------------------------------------------------
    eval_results_path = Path(args.eval_results)
    eval_problems = load_eval_problems(eval_results_path, args.n_disks)

    optimal_problems = [
        (p["start"], p["goal"]) for p in eval_problems if str(p["category"]) == "CORRECT_OPTIMAL"
    ]
    nonoptimal_problems = [
        (p["start"], p["goal"]) for p in eval_problems if str(p["category"]) != "CORRECT_OPTIMAL"
    ]

    if args.max_trajectories > 0:
        optimal_problems = optimal_problems[: args.max_trajectories]
        nonoptimal_problems = nonoptimal_problems[: args.max_trajectories]

    if not optimal_problems:
        raise RuntimeError("No CORRECT_OPTIMAL problems found in evaluation results")
    if not nonoptimal_problems:
        print("[WARN] No non-optimal problems found; only optimal cohort will be analyzed")

    print(
        f"[INFO] Loaded {len(optimal_problems)} CORRECT_OPTIMAL and "
        f"{len(nonoptimal_problems)} NOT_OPTIMAL problems from {eval_results_path}"
    )

    # ------------------------------------------------------------------
    # Decode trajectories, capture activations at move-token positions,
    # and apply frozen probe to all layers, separately by cohort.
    # ------------------------------------------------------------------
    context_len = 2 * args.n_disks + 3
    sep2_pos = context_len - 1
    ref_pos = reference_positions.astype(np.float32)
    ref_pairwise = np.linalg.norm(ref_pos[:, None, :] - ref_pos[None, :, :], axis=-1)
    iu = np.triu_indices(ref_pairwise.shape[0], k=1)
    avg_interstate_dist = float(np.mean(ref_pairwise[iu]))

    def analyze_cohort(
        cohort_name: str,
        problems: List[Tuple[State, State]],
    ) -> Optional[Dict[str, object]]:
        if not problems:
            return None

        true_state_indices: List[int] = []
        step_indices: List[int] = []
        trajectory_ids_for_points: List[int] = []
        traj_lengths: List[int] = []
        kept_problem_indices: List[int] = []

        pred_positions_by_layer: Dict[int, List[np.ndarray]] = {l: [] for l in range(1, model.n_layers + 1)}
        per_step_json_rows: List[Dict[str, object]] = []

        traj_pred_positions_primary: Dict[int, np.ndarray] = {}
        per_step_layer_correct: Dict[int, Dict[int, Tuple[int, int]]] = {l: {} for l in range(1, model.n_layers + 1)}

        for traj_id, (start, goal) in enumerate(problems):
            context_ids = build_context_ids(start, goal, vocab)
            generated_ids, eos_seen = greedy_decode_ids(model, context_ids, vocab.eos_id, device)

            if eos_seen:
                eos_pos = generated_ids.index(vocab.eos_id)
                move_ids = generated_ids[:eos_pos]
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

            full_seq = context_ids + valid_move_ids + ([vocab.eos_id] if eos_seen else [])
            if len(full_seq) > model.max_seq_len:
                continue

            layer_to_acts = capture_move_position_activations(
                model=model,
                full_sequence=full_seq,
                context_len=context_len,
                n_move_tokens=len(valid_move_ids),
                device=device,
            )

            kept_idx = len(traj_lengths)
            traj_lengths.append(len(inter_states))
            kept_problem_indices.append(traj_id)

            primary_positions_this_traj = None
            if len(inter_states) > 0:
                for layer in range(1, model.n_layers + 1):
                    acts = torch.tensor(layer_to_acts[layer], dtype=torch.float32, device=device)
                    acts = correct_to_sep_position(
                        activations=acts,
                        model=model,
                        move_start_pos=context_len,
                        sep_pos=sep2_pos,
                    )
                    with torch.no_grad():
                        pred = probe(acts).detach().cpu().numpy().astype(np.float32)
                    pred_positions_by_layer[layer].append(pred)
                    if layer == args.layer:
                        primary_positions_this_traj = pred

                for step0, state in enumerate(inter_states):
                    step = step0 + 1
                    s_idx = state_to_idx[state]

                    row = {
                        "cohort": cohort_name,
                        "trajectory_id": kept_idx,
                        "source_problem_index": traj_id,
                        "start": list(start),
                        "goal": list(goal),
                        "step": step,
                        "trajectory_length": len(inter_states),
                        "true_state": list(state),
                        "predicted_positions": {},
                    }

                    for layer in range(1, model.n_layers + 1):
                        p = pred_positions_by_layer[layer][-1][step0]
                        row["predicted_positions"][str(layer)] = [float(p[0]), float(p[1])]

                    per_step_json_rows.append(row)
                    true_state_indices.append(s_idx)
                    step_indices.append(step)
                    trajectory_ids_for_points.append(kept_idx)
            else:
                for layer in range(1, model.n_layers + 1):
                    pred_positions_by_layer[layer].append(np.zeros((0, 2), dtype=np.float32))

            if primary_positions_this_traj is not None:
                traj_pred_positions_primary[kept_idx] = primary_positions_this_traj

            if (traj_id + 1) % 200 == 0:
                print(f"[INFO][{cohort_name}] Processed {traj_id + 1}/{len(problems)} trajectories")

        if not true_state_indices:
            print(f"[WARN][{cohort_name}] No valid intermediate steps collected")
            return {
                "cohort": cohort_name,
                "summary_by_layer": [],
                "n_trajectories": len(traj_lengths),
                "n_total_steps": 0,
                "source_problem_indices": kept_problem_indices,
                "per_step": [],
                "figures": {},
            }

        true_state_idx_arr = np.array(true_state_indices, dtype=np.int64)
        step_idx_arr = np.array(step_indices, dtype=np.int64)
        traj_id_arr = np.array(trajectory_ids_for_points, dtype=np.int64)
        traj_len_lookup = {tid: ln for tid, ln in enumerate(traj_lengths)}

        pred_flat_by_layer: Dict[int, np.ndarray] = {}
        for layer in range(1, model.n_layers + 1):
            chunks: List[np.ndarray] = [pred_positions_by_layer[layer][tid] for tid in range(len(traj_lengths))]
            pred_flat_by_layer[layer] = np.concatenate(chunks, axis=0)

        if pred_flat_by_layer[1].shape[0] != true_state_idx_arr.shape[0]:
            raise RuntimeError(f"[{cohort_name}] Mismatch between flattened predictions and true state list")

        summary_rows: List[Dict[str, object]] = []
        state_correct_primary = np.zeros(len(states), dtype=np.int64)
        state_total_primary = np.zeros(len(states), dtype=np.int64)

        for layer in range(1, model.n_layers + 1):
            pred = pred_flat_by_layer[layer]
            dists_to_ref = np.linalg.norm(pred[:, None, :] - ref_pos[None, :, :], axis=-1)
            nearest_idx = np.argmin(dists_to_ref, axis=1)
            nearest_correct = nearest_idx == true_state_idx_arr

            true_ref = ref_pos[true_state_idx_arr]
            l2_err = np.linalg.norm(pred - true_ref, axis=1)

            overall_acc = float(np.mean(nearest_correct))
            mean_l2 = float(np.mean(l2_err))

            early_mask = np.zeros_like(nearest_correct, dtype=bool)
            mid_mask = np.zeros_like(nearest_correct, dtype=bool)
            late_mask = np.zeros_like(nearest_correct, dtype=bool)

            for i in range(len(nearest_correct)):
                tid = int(traj_id_arr[i])
                step = int(step_idx_arr[i])
                tlen = max(traj_len_lookup.get(tid, 1), 1)
                frac = step / tlen
                if frac <= (1.0 / 3.0):
                    early_mask[i] = True
                elif frac <= (2.0 / 3.0):
                    mid_mask[i] = True
                else:
                    late_mask[i] = True

            def masked_acc(mask: np.ndarray) -> float:
                if not np.any(mask):
                    return float("nan")
                return float(np.mean(nearest_correct[mask]))

            early_acc = masked_acc(early_mask)
            mid_acc = masked_acc(mid_mask)
            late_acc = masked_acc(late_mask)

            summary_rows.append(
                {
                    "layer": layer,
                    "nearest_state_accuracy": overall_acc,
                    "mean_l2_error": mean_l2,
                    "mean_l2_over_avg_interstate": float(mean_l2 / max(avg_interstate_dist, 1e-8)),
                    "early_accuracy": early_acc,
                    "mid_accuracy": mid_acc,
                    "late_accuracy": late_acc,
                }
            )

            step_stats: Dict[int, List[int]] = defaultdict(lambda: [0, 0])
            for i in range(len(nearest_correct)):
                s = int(step_idx_arr[i])
                step_stats[s][0] += int(nearest_correct[i])
                step_stats[s][1] += 1
            per_step_layer_correct[layer] = {s: (c[0], c[1]) for s, c in step_stats.items()}

            if layer == args.layer:
                for idx, ok in zip(true_state_idx_arr, nearest_correct):
                    state_total_primary[idx] += 1
                    state_correct_primary[idx] += int(ok)

            for i, row in enumerate(per_step_json_rows):
                if "per_layer" not in row:
                    row["per_layer"] = {}
                row["per_layer"][str(layer)] = {
                    "nearest_state": [int(x) for x in states[int(nearest_idx[i])]],
                    "nearest_correct": bool(nearest_correct[i]),
                    "l2_error": float(l2_err[i]),
                }

        summary_rows_sorted = sorted(summary_rows, key=lambda r: int(r["layer"]))
        cohort_tag = cohort_name.lower().replace(" ", "_")

        chosen = pick_diverse_trajectories(traj_lengths, max_count=12)

        trajectory_plot_path = out_dir / f"trajectory_probe_paths_{cohort_tag}.png"
        plot_trajectory_map(
            out_path=trajectory_plot_path,
            reference_positions=ref_pos,
            edges=edges,
            chosen_traj_ids=chosen,
            traj_pred_positions_primary=traj_pred_positions_primary,
        )

        step_plot_path = out_dir / f"accuracy_by_step_layer_{cohort_tag}.png"
        plot_accuracy_by_step(
            out_path=step_plot_path,
            per_step_layer_correct=per_step_layer_correct,
            n_layers=model.n_layers,
        )

        heatmap_path = out_dir / f"state_decode_heatmap_layer_{args.layer}_{cohort_tag}.png"
        plot_state_error_heatmap(
            out_path=heatmap_path,
            reference_positions=ref_pos,
            edges=edges,
            state_correct=state_correct_primary,
            state_total=state_total_primary,
            title=f"State Decode Accuracy Heatmap (Layer {args.layer}, {cohort_name})",
        )

        return {
            "cohort": cohort_name,
            "summary_by_layer": summary_rows_sorted,
            "n_trajectories": len(traj_lengths),
            "n_total_steps": int(len(per_step_json_rows)),
            "source_problem_indices": kept_problem_indices,
            "per_step": per_step_json_rows,
            "figures": {
                "trajectory_plot": str(trajectory_plot_path),
                "accuracy_by_step": str(step_plot_path),
                "error_heatmap": str(heatmap_path),
            },
        }

    optimal_results = analyze_cohort("CORRECT_OPTIMAL", optimal_problems)
    nonoptimal_results = analyze_cohort("NOT_OPTIMAL", nonoptimal_problems) if nonoptimal_problems else None

    if optimal_results is None:
        raise RuntimeError("Failed to compute optimal cohort probing results")

    full_results = {
        "checkpoint": str(ckpt_path),
        "probe_checkpoint": str(probe_ckpt_path),
        "n_disks": args.n_disks,
        "primary_layer": args.layer,
        "eval_results": str(eval_results_path),
        "avg_interstate_distance_reference": avg_interstate_dist,
        "cohorts": {
            "CORRECT_OPTIMAL": optimal_results,
            "NOT_OPTIMAL": nonoptimal_results,
        },
    }

    json_path = out_dir / "trajectory_probe_results.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2)

    def summary_lookup(res: Optional[Dict[str, object]], layer: int) -> Optional[Dict[str, object]]:
        if res is None:
            return None
        rows = res.get("summary_by_layer", [])
        if not isinstance(rows, list):
            return None
        for row in rows:
            if int(row.get("layer", -1)) == layer:
                return row
        return None

    comparison_rows: List[Dict[str, object]] = []
    for layer in range(1, model.n_layers + 1):
        opt_row = summary_lookup(optimal_results, layer)
        non_row = summary_lookup(nonoptimal_results, layer)
        if opt_row is None:
            continue
        delta_acc = None
        delta_l2 = None
        if non_row is not None:
            delta_acc = float(opt_row["nearest_state_accuracy"]) - float(non_row["nearest_state_accuracy"])
            delta_l2 = float(opt_row["mean_l2_error"]) - float(non_row["mean_l2_error"])
        comparison_rows.append(
            {
                "layer": layer,
                "optimal": opt_row,
                "nonoptimal": non_row,
                "delta_nearest_state_accuracy": delta_acc,
                "delta_mean_l2_error": delta_l2,
            }
        )

    summary_path = out_dir / "trajectory_probe_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "avg_interstate_distance_reference": avg_interstate_dist,
                "comparison_by_layer": comparison_rows,
                "n_optimal_trajectories": int(optimal_results["n_trajectories"]),
                "n_nonoptimal_trajectories": int(nonoptimal_results["n_trajectories"]) if nonoptimal_results else 0,
            },
            f,
            indent=2,
        )

    def print_table(title: str, rows: List[Dict[str, object]]) -> None:
        print(f"\n{title}")
        print("Layer | Nearest-State Acc | Mean L2 Error | Early Acc | Mid Acc | Late Acc")
        print("------+-------------------+---------------+-----------+---------+---------")
        for row in rows:
            layer = int(row["layer"])
            overall = float(row["nearest_state_accuracy"])
            l2 = float(row["mean_l2_error"])
            early = float(row["early_accuracy"])
            mid = float(row["mid_accuracy"])
            late = float(row["late_accuracy"])
            print(
                f"{layer:5d} | {format_pct(overall):>17} | {l2:13.6f} | "
                f"{format_pct(early):>9} | {format_pct(mid):>7} | {format_pct(late):>8}"
            )

    print_table("CORRECT_OPTIMAL", optimal_results["summary_by_layer"])  # type: ignore[arg-type]
    if nonoptimal_results is not None:
        print_table("NOT_OPTIMAL", nonoptimal_results["summary_by_layer"])  # type: ignore[arg-type]

    print("\n[INFO] Mean inter-state distance in reference layout:", f"{avg_interstate_dist:.6f}")
    print(f"[INFO] Saved full per-step JSON to {json_path}")
    print(f"[INFO] Saved comparison summary JSON to {summary_path}")


if __name__ == "__main__":
    main()
