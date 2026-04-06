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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def confirm_tokenizer_mapping(vocab: Vocabulary) -> None:
    if vocab.stoi != EXPECTED_TOKEN_MAPPING:
        raise ValueError(f"Tokenizer mapping mismatch: {vocab.stoi}")


def resolve_checkpoint_path(path_str: str, n_disks: int) -> Path:
    p = Path(path_str)
    if p.exists():
        return p
    if path_str == "best.pt":
        candidates = [
            Path(f"toh_transformer/checkpoints/n{n_disks}/best.pt"),
            Path("toh_transformer/checkpoints/flat_train_3-4-6__test_5/best.pt"),
        ]
        for cand in candidates:
            if cand.exists():
                print(f"[INFO] Checkpoint 'best.pt' not found; using {cand}")
                return cand
    raise FileNotFoundError(f"Checkpoint not found: {path_str}")


def load_model(checkpoint_path: Path, n_disks: int, device: torch.device) -> ToHTransformer:
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}

    defaults = default_model_hparams(n_disks)
    model = ToHTransformer(
        vocab_size=len(Vocabulary()),
        max_seq_len=int(cfg.get("max_seq_len", max_seq_len_for_disks(n_disks))),
        n_layers=int(cfg.get("n_layers", defaults["n_layers"])),
        n_heads=int(cfg.get("n_heads", defaults["n_heads"])),
        d_model=int(cfg.get("d_model", defaults["d_model"])),
        d_ff=int(cfg.get("d_ff", defaults["d_ff"])),
        dropout=float(cfg.get("dropout", defaults["dropout"])),
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


def build_context_ids(start: State, goal: State, vocab: Vocabulary) -> List[int]:
    return [
        vocab.bos_id,
        *[vocab.stoi[f"P{p}"] for p in start],
        vocab.sep_id,
        *[vocab.stoi[f"P{p}"] for p in goal],
        vocab.sep_id,
    ]


@torch.no_grad()
def greedy_decode_ids(
    model: ToHTransformer,
    context_ids: Sequence[int],
    eos_id: int,
    device: torch.device,
) -> Tuple[List[int], bool]:
    seq = torch.tensor(context_ids, dtype=torch.long, device=device).unsqueeze(0)
    generated: List[int] = []
    while seq.size(1) < model.max_seq_len:
        logits = model(seq)
        next_id = int(torch.argmax(logits[:, -1, :], dim=-1).item())
        generated.append(next_id)
        seq = torch.cat([seq, torch.tensor([[next_id]], dtype=torch.long, device=device)], dim=1)
        if next_id == eos_id:
            return generated, True
    return generated, False


def decode_move_token(tok: str) -> Optional[Tuple[int, int]]:
    if len(tok) != 3 or tok[0] != "M":
        return None
    if tok[1] not in "012" or tok[2] not in "012":
        return None
    src, dst = int(tok[1]), int(tok[2])
    if src == dst:
        return None
    return src, dst


def apply_move_and_next_state(state: State, move: Tuple[int, int]) -> Optional[State]:
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


def load_correct_optimal_problems(eval_results_path: Path, n_disks: int) -> List[Tuple[State, State]]:
    with eval_results_path.open("r", encoding="utf-8") as f:
        rows = json.load(f)

    out: List[Tuple[State, State]] = []
    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        if row.get("category") != "CORRECT_OPTIMAL":
            continue
        s_raw = row.get("start")
        g_raw = row.get("goal")
        if not isinstance(s_raw, list) or not isinstance(g_raw, list):
            continue
        if len(s_raw) != n_disks or len(g_raw) != n_disks:
            continue
        pair = (tuple(int(x) for x in s_raw), tuple(int(x) for x in g_raw))
        if pair in seen:
            continue
        seen.add(pair)
        out.append(pair)
    return out


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


def nearest_state_predictions(proj: np.ndarray, ref_positions: np.ndarray) -> np.ndarray:
    d_ref = np.linalg.norm(proj[:, None, :] - ref_positions[None, :, :], axis=-1)
    return np.argmin(d_ref, axis=1)


def compute_spearman_on_validation(
    proj: np.ndarray,
    y_state_idx: np.ndarray,
    dist_true: np.ndarray,
    val_idx: np.ndarray,
    seed: int,
) -> float:
    if val_idx.shape[0] < 2:
        return float("nan")

    n = int(val_idx.shape[0])
    max_exact = 2500

    if n <= max_exact:
        x = proj[val_idx]
        y = y_state_idx[val_idx]
        pred_mat = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=-1)
        true_mat = dist_true[y][:, y]
        rho, _ = spearmanr(pred_mat.reshape(-1), true_mat.reshape(-1))
        return float(rho) if np.isfinite(rho) else float("nan")

    rng = np.random.default_rng(seed)
    num_pairs = 200000
    a = rng.integers(0, n, size=num_pairs)
    b = rng.integers(0, n, size=num_pairs)
    mask = a != b
    if not np.any(mask):
        return float("nan")

    a = a[mask]
    b = b[mask]
    ia = val_idx[a]
    ib = val_idx[b]

    pred = np.linalg.norm(proj[ia] - proj[ib], axis=1)
    true = dist_true[y_state_idx[ia], y_state_idx[ib]]
    rho, _ = spearmanr(pred, true)
    return float(rho) if np.isfinite(rho) else float("nan")


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
) -> np.ndarray:
    n_states, d_model = sep_layer_acts.shape
    refs = np.zeros((n_states, k * d_model), dtype=np.float32)

    for state_idx in range(n_states):
        if state_idx in first_step_ref_for_states:
            refs[state_idx] = first_step_ref_for_states[state_idx]
        else:
            refs[state_idx] = np.tile(sep_layer_acts[state_idx], k)

    return project_with_probe(probe, refs, device=device)


def pct(x: float) -> str:
    if not np.isfinite(x):
        return "nan"
    return f"{100.0 * x:.2f}%"


def plot_best_scatter(
    out_path: Path,
    best_proj: np.ndarray,
    best_y_state_idx: np.ndarray,
    ref_positions: np.ndarray,
    states: Sequence[State],
    edges: Sequence[Edge],
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

    colors = np.array([states[idx][3] if len(states[idx]) >= 4 else states[idx][-1] for idx in best_y_state_idx])
    sc = ax.scatter(
        best_proj[:, 0],
        best_proj[:, 1],
        c=colors,
        cmap="viridis",
        s=8,
        alpha=0.35,
        zorder=2,
    )
    ax.scatter(ref_positions[:, 0], ref_positions[:, 1], c="black", s=18, alpha=0.8, zorder=3)

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Largest disk position (state[3])")

    ax.set_title("Best (K, layer): projected points with state-graph edges")
    ax.set_xlabel("Probe dim 1")
    ax.set_ylabel("Probe dim 2")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


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

            proj = project_with_probe(probe, x, device=device)
            nearest = nearest_state_predictions(proj, ref_positions)
            correct = nearest == y_state_idx

            acc = float(np.mean(correct))
            rho = compute_spearman_on_validation(
                proj=proj,
                y_state_idx=y_state_idx,
                dist_true=dist_true,
                val_idx=val_idx,
                seed=args.seed + 5000 + 100 * k + layer,
            )
            early_acc = masked_accuracy(early_mask, correct)
            mid_acc = masked_accuracy(mid_mask, correct)
            late_acc = masked_accuracy(late_mask, correct)

            row = {
                "K": float(k),
                "layer": float(layer),
                "nearest_state_acc": acc,
                "spearman_rho": rho,
                "early_acc": early_acc,
                "mid_acc": mid_acc,
                "late_acc": late_acc,
                "final_loss": float(final_loss),
            }
            results_rows.append(row)
            k_rows.append(row)

            if best_global is None or acc > float(best_global["nearest_state_acc"]):
                best_global = {
                    "K": k,
                    "layer": layer,
                    "nearest_state_acc": acc,
                    "spearman_rho": rho,
                    "proj": proj,
                    "y_state_idx": y_state_idx.copy(),
                    "ref_positions": ref_positions,
                }

        k_rows_sorted = sorted(k_rows, key=lambda r: (r["nearest_state_acc"], r["spearman_rho"]), reverse=True)
        best_by_k[k] = k_rows_sorted[0]

    print("\nK  | Layer | Nearest-State Acc | Spearman rho | Early Acc | Mid Acc | Late Acc")
    print("---+-------+-------------------+--------------+-----------+---------+---------")
    rows_sorted = sorted(results_rows, key=lambda r: (int(r["K"]), int(r["layer"])))
    for r in rows_sorted:
        print(
            f"{int(r['K']):2d} | {int(r['layer']):5d} | {pct(r['nearest_state_acc']):>17} | "
            f"{r['spearman_rho']:12.4f} | {pct(r['early_acc']):>9} | {pct(r['mid_acc']):>7} | {pct(r['late_acc']):>8}"
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

    plot_best_scatter(
        out_path=out_dir / "best_k_layer_scatter.png",
        best_proj=np.asarray(best_global["proj"]),
        best_y_state_idx=np.asarray(best_global["y_state_idx"]),
        ref_positions=np.asarray(best_global["ref_positions"]),
        states=states,
        edges=edges,
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
        },
    }
    (out_dir / "multi_position_probe_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"[INFO] Wrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
