#!/usr/bin/env python3
"""Activation patching for SEP-token world-model representations.

This script tests whether the representation at the second SEP token causally drives
move generation by swapping activations between donor/recipient problems.

Protocol summary:
- Fixed goal: (2, 2, 2, 2)
- Enumerate all 81 start states for n_disks=4
- Keep only starts whose clean greedy decode is correct (optimal or suboptimal)
- Run full ordered donor->recipient sweeps for selected layers (default: 4, 5, 6)
- Select best causal layer by full-transfer rate
- Emit tables, figures, and raw per-pair results
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
from collections import deque
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import BoundaryNorm, ListedColormap

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from toh_transformer.config import default_model_hparams, max_seq_len_for_disks
from toh_transformer.data import Vocabulary
from toh_transformer.model import ToHTransformer

State = Tuple[int, ...]
Move = Tuple[int, int]
Pair = Tuple[State, State]

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

CATEGORY_FULL_TRANSFER = "FULL_TRANSFER"
CATEGORY_PARTIAL_TRANSFER = "PARTIAL_TRANSFER"
CATEGORY_RECIPIENT_UNCHANGED = "RECIPIENT_UNCHANGED"
CATEGORY_NOVEL_CORRECT = "NOVEL_CORRECT"
CATEGORY_NOVEL_INCORRECT = "NOVEL_INCORRECT"
CATEGORY_UNAVAILABLE = "UNAVAILABLE"

CATEGORY_ORDER = [
    CATEGORY_FULL_TRANSFER,
    CATEGORY_PARTIAL_TRANSFER,
    CATEGORY_RECIPIENT_UNCHANGED,
    CATEGORY_NOVEL_CORRECT,
    CATEGORY_NOVEL_INCORRECT,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Activation patching at the second SEP token")
    parser.add_argument("--checkpoint", type=str, default="best.pt")
    parser.add_argument("--n_disks", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="toh_transformer/activation_patching_output")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--full_sweep_layers",
        type=str,
        default="4,5,6",
        help="Comma-separated layer ids for full 81x80 donor->recipient sweeps",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_layer_list(spec: str) -> List[int]:
    values = []
    for part in spec.split(","):
        token = part.strip()
        if not token:
            continue
        values.append(int(token))
    if not values:
        raise ValueError("Layer list is empty")
    # Preserve order while dropping duplicates.
    return list(dict.fromkeys(values))


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


def build_context_ids(start: State, goal: State, vocab: Vocabulary) -> List[int]:
    return [
        vocab.bos_id,
        *[vocab.stoi[f"P{p}"] for p in start],
        vocab.sep_id,
        *[vocab.stoi[f"P{p}"] for p in goal],
        vocab.sep_id,
    ]


def top_disk_per_peg(state: State) -> List[Optional[int]]:
    tops: List[Optional[int]] = [None, None, None]
    for disk_idx, peg in enumerate(state):
        top = tops[peg]
        if top is None or disk_idx < top:
            tops[peg] = disk_idx
    return tops


def decode_move_token(tok: str) -> Optional[Move]:
    if len(tok) != 3 or tok[0] != "M":
        return None
    if tok[1] not in "012" or tok[2] not in "012":
        return None
    src, dst = int(tok[1]), int(tok[2])
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

    out = list(state)
    out[src_top] = dst
    return tuple(out)


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


@torch.no_grad()
def forward_with_sep_patch(
    model: ToHTransformer,
    input_ids: torch.Tensor,
    target_layer: int,
    sep_index: int,
    donor_sep_activation: torch.Tensor,
) -> torch.Tensor:
    """Forward pass with post-layer residual patch at one position.

    donor_sep_activation can be shape [d_model] or [batch, d_model].
    """
    bsz, seq_len = input_ids.shape
    if seq_len > model.max_seq_len:
        raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len={model.max_seq_len}")

    if target_layer < 1 or target_layer > model.n_layers:
        raise ValueError(f"target_layer must be in [1, {model.n_layers}], got {target_layer}")

    if donor_sep_activation.dim() == 1:
        donor_act = donor_sep_activation.unsqueeze(0).expand(bsz, -1)
    elif donor_sep_activation.dim() == 2 and donor_sep_activation.size(0) == bsz:
        donor_act = donor_sep_activation
    else:
        raise ValueError("donor_sep_activation must be shape [d_model] or [batch, d_model]")

    positions = torch.arange(0, seq_len, device=input_ids.device, dtype=torch.long)
    x = model.token_emb(input_ids) + model.pos_emb(positions).unsqueeze(0)
    x = model.emb_dropout(x)

    for layer_idx, block in enumerate(model.blocks, start=1):
        x = block(x)
        if layer_idx == target_layer:
            x = x.clone()
            x[:, sep_index, :] = donor_act

    x = model.ln_f(x)
    logits = model.lm_head(x)
    return logits


@torch.no_grad()
def greedy_decode_ids_with_patch(
    model: ToHTransformer,
    context_ids: Sequence[int],
    target_layer: int,
    sep_index: int,
    donor_sep_activation: torch.Tensor,
    eos_id: int,
    device: torch.device,
) -> Tuple[List[int], bool]:
    """Decode autoregressively while applying SEP patch on every step.

    As requested, this intentionally recomputes the full forward pass each step and
    reapplies the patch at the second SEP position.
    """
    seq = torch.tensor(context_ids, dtype=torch.long, device=device).unsqueeze(0)
    generated: List[int] = []

    donor_act = donor_sep_activation.to(device=device)

    while seq.size(1) < model.max_seq_len:
        logits = forward_with_sep_patch(
            model=model,
            input_ids=seq,
            target_layer=target_layer,
            sep_index=sep_index,
            donor_sep_activation=donor_act,
        )
        next_id = int(torch.argmax(logits[:, -1, :], dim=-1).item())
        generated.append(next_id)
        seq = torch.cat([seq, torch.tensor([[next_id]], dtype=torch.long, device=device)], dim=1)
        if next_id == eos_id:
            return generated, True

    return generated, False


def parse_generation(
    start: State,
    goal: State,
    generated_ids: Sequence[int],
    eos_seen: bool,
    vocab: Vocabulary,
) -> Dict[str, object]:
    """Parse generated sequence and evaluate legality/goal correctness."""
    toks = [vocab.itos[i] for i in generated_ids]

    if eos_seen and vocab.eos_id in generated_ids:
        eos_pos = generated_ids.index(vocab.eos_id)
        move_tokens = toks[:eos_pos]
    else:
        move_tokens = toks

    current = start
    legal = True
    for tok in move_tokens:
        mv = decode_move_token(tok)
        if mv is None:
            legal = False
            break
        nxt = apply_move_and_next_state(current, mv)
        if nxt is None:
            legal = False
            break
        current = nxt

    reaches_goal = legal and eos_seen and current == goal
    return {
        "moves": move_tokens,
        "legal": legal,
        "eos_seen": eos_seen,
        "final_state": current,
        "reaches_goal": reaches_goal,
    }


@torch.no_grad()
def extract_sep2_activations(
    model: ToHTransformer,
    starts: Sequence[State],
    goal: State,
    vocab: Vocabulary,
    device: torch.device,
    batch_size: int = 128,
) -> Dict[int, torch.Tensor]:
    layers = list(range(1, model.n_layers + 1))
    rows = [build_context_ids(s, goal, vocab) for s in starts]
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


def common_prefix_len(a: Sequence[str], b: Sequence[str]) -> int:
    k = 0
    n = min(len(a), len(b))
    while k < n and a[k] == b[k]:
        k += 1
    return k


def classify_patched_output(
    patched_moves: Sequence[str],
    donor_moves: Sequence[str],
    recipient_moves: Sequence[str],
    patched_is_correct: bool,
) -> Tuple[str, Optional[int]]:
    if list(patched_moves) == list(donor_moves):
        return CATEGORY_FULL_TRANSFER, None

    if list(patched_moves) == list(recipient_moves):
        return CATEGORY_RECIPIENT_UNCHANGED, None

    k = common_prefix_len(patched_moves, donor_moves)
    if k >= 1:
        return CATEGORY_PARTIAL_TRANSFER, int(k)

    if patched_is_correct:
        return CATEGORY_NOVEL_CORRECT, None

    return CATEGORY_NOVEL_INCORRECT, None


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


def build_distances(states: Sequence[State]) -> np.ndarray:
    state_to_idx = {s: i for i, s in enumerate(states)}
    n_states = len(states)

    adj: List[List[int]] = [[] for _ in range(n_states)]
    for i, s in enumerate(states):
        for nbr in legal_neighbors(s):
            adj[i].append(state_to_idx[nbr])

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
        raise RuntimeError("State graph is disconnected")
    return dist


def summarize_layer(results: Sequence[Dict[str, object]]) -> Dict[str, object]:
    total = len(results)
    counts = {cat: 0 for cat in CATEGORY_ORDER}
    partial_ks: List[int] = []

    for row in results:
        cat = str(row["category"])
        counts[cat] += 1
        if cat == CATEGORY_PARTIAL_TRANSFER and row.get("partial_k") is not None:
            partial_ks.append(int(row["partial_k"]))

    rates = {cat: (counts[cat] / total if total > 0 else 0.0) for cat in CATEGORY_ORDER}
    mean_k = float(np.mean(partial_ks)) if partial_ks else 0.0

    return {
        "n_pairs": total,
        "counts": counts,
        "rates": rates,
        "partial_mean_k": mean_k,
    }


def format_pct(x: float) -> str:
    return f"{100.0 * x:.2f}%"


def write_table_1(layer_summaries: Dict[int, Dict[str, object]], output_path: Path, title: str) -> str:
    lines = []
    lines.append(title)
    lines.append("")
    lines.append("Layer | N pairs | Full Transfer | Partial (mean K) | Unchanged | Novel Correct | Disrupted")
    lines.append("------+---------+---------------+------------------+-----------+---------------+----------")

    for layer in sorted(layer_summaries.keys()):
        s = layer_summaries[layer]
        rates = s["rates"]
        partial_rate = format_pct(float(rates[CATEGORY_PARTIAL_TRANSFER]))
        mean_k = float(s["partial_mean_k"])
        lines.append(
            f"{layer:>5d} | "
            f"{int(s['n_pairs']):>7d} | "
            f"{format_pct(float(rates[CATEGORY_FULL_TRANSFER])):>13s} | "
            f"{partial_rate:>8s} (K={mean_k:.2f}) | "
            f"{format_pct(float(rates[CATEGORY_RECIPIENT_UNCHANGED])):>9s} | "
            f"{format_pct(float(rates[CATEGORY_NOVEL_CORRECT])):>13s} | "
            f"{format_pct(float(rates[CATEGORY_NOVEL_INCORRECT])):>8s}"
        )

    text = "\n".join(lines) + "\n"
    output_path.write_text(text, encoding="utf-8")
    return text


def compute_distance_bucket_table(
    best_layer_results: Sequence[Dict[str, object]],
    distances: np.ndarray,
    state_to_idx: Dict[State, int],
) -> str:
    buckets = [
        ("1-3", 1, 3),
        ("4-6", 4, 6),
        ("7-9", 7, 9),
        ("10-15", 10, 15),
    ]

    rows = []
    for label, lo, hi in buckets:
        bucket_rows = []
        for r in best_layer_results:
            donor = tuple(int(x) for x in r["donor"])
            recipient = tuple(int(x) for x in r["recipient"])
            d = int(distances[state_to_idx[donor], state_to_idx[recipient]])
            if lo <= d <= hi:
                bucket_rows.append(r)

        n = len(bucket_rows)
        if n == 0:
            full_rate = 0.0
            disrupted_rate = 0.0
        else:
            full_rate = sum(1 for x in bucket_rows if x["category"] == CATEGORY_FULL_TRANSFER) / n
            disrupted_rate = sum(1 for x in bucket_rows if x["category"] == CATEGORY_NOVEL_INCORRECT) / n

        rows.append((label, n, full_rate, disrupted_rate))

    lines = []
    lines.append("Table 2 - Transfer by donor-recipient start-state graph distance (best layer)")
    lines.append("")
    lines.append("Distance | N pairs | Full Transfer | Disrupted")
    lines.append("---------+---------+---------------+----------")
    for label, n, full_rate, disrupted_rate in rows:
        lines.append(f"{label:>8s} | {n:>7d} | {format_pct(full_rate):>13s} | {format_pct(disrupted_rate):>8s}")
    return "\n".join(lines) + "\n"


def compute_donor_length_bucket_table(best_layer_results: Sequence[Dict[str, object]]) -> str:
    buckets = [
        ("1-3", 1, 3),
        ("4-7", 4, 7),
        ("8-11", 8, 11),
        ("12-15", 12, 15),
    ]

    rows = []
    for label, lo, hi in buckets:
        bucket_rows = [
            r
            for r in best_layer_results
            if lo <= len(r["donor_moves"]) <= hi
        ]
        n = len(bucket_rows)
        if n == 0:
            full_rate = 0.0
            disrupted_rate = 0.0
        else:
            full_rate = sum(1 for x in bucket_rows if x["category"] == CATEGORY_FULL_TRANSFER) / n
            disrupted_rate = sum(1 for x in bucket_rows if x["category"] == CATEGORY_NOVEL_INCORRECT) / n
        rows.append((label, n, full_rate, disrupted_rate))

    lines = []
    lines.append("Table 3 - Transfer by donor clean solution length (best layer)")
    lines.append("")
    lines.append("Donor Length | N pairs | Full Transfer | Disrupted")
    lines.append("-------------+---------+---------------+----------")
    for label, n, full_rate, disrupted_rate in rows:
        lines.append(f"{label:>12s} | {n:>7d} | {format_pct(full_rate):>13s} | {format_pct(disrupted_rate):>8s}")
    return "\n".join(lines) + "\n"


def plot_layer_stacked_bar(layer_summaries: Dict[int, Dict[str, object]], out_path: Path) -> None:
    layers = sorted(layer_summaries.keys())

    categories = [
        CATEGORY_FULL_TRANSFER,
        CATEGORY_PARTIAL_TRANSFER,
        CATEGORY_RECIPIENT_UNCHANGED,
        CATEGORY_NOVEL_CORRECT,
        CATEGORY_NOVEL_INCORRECT,
    ]
    colors = {
        CATEGORY_FULL_TRANSFER: "#2ca02c",  # green
        CATEGORY_PARTIAL_TRANSFER: "#f1c40f",  # yellow
        CATEGORY_RECIPIENT_UNCHANGED: "#1f77b4",  # blue
        CATEGORY_NOVEL_CORRECT: "#8e44ad",  # purple
        CATEGORY_NOVEL_INCORRECT: "#d62728",  # red
    }

    fig, ax = plt.subplots(figsize=(10, 5.8))
    bottom = np.zeros(len(layers), dtype=np.float64)

    for cat in categories:
        vals = np.array([float(layer_summaries[l]["rates"][cat]) for l in layers], dtype=np.float64)
        ax.bar(layers, vals, bottom=bottom, color=colors[cat], label=cat.replace("_", " ").title())
        bottom += vals

    ax.set_title("Activation Patching: Which Layer Drives Planning?")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Proportion")
    ax.set_xticks(layers)
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_best_layer_heatmap(
    best_layer: int,
    all_states: Sequence[State],
    valid_states: Sequence[State],
    best_layer_results: Sequence[Dict[str, object]],
    out_path: Path,
) -> None:
    state_order = sorted(all_states, key=lambda s: tuple(reversed(s)))
    idx_in_order = {s: i for i, s in enumerate(state_order)}

    n = len(state_order)
    mat = np.zeros((n, n), dtype=np.int32)

    code_of = {
        CATEGORY_UNAVAILABLE: 0,
        CATEGORY_RECIPIENT_UNCHANGED: 1,
        CATEGORY_FULL_TRANSFER: 2,
        CATEGORY_PARTIAL_TRANSFER: 3,
        CATEGORY_NOVEL_CORRECT: 4,
        CATEGORY_NOVEL_INCORRECT: 5,
    }

    valid_set = set(valid_states)
    for donor in state_order:
        for recipient in state_order:
            if donor == recipient:
                continue
            if donor not in valid_set or recipient not in valid_set:
                continue
            mat[idx_in_order[donor], idx_in_order[recipient]] = code_of[CATEGORY_UNAVAILABLE]

    for row in best_layer_results:
        donor = tuple(int(x) for x in row["donor"])
        recipient = tuple(int(x) for x in row["recipient"])
        cat = str(row["category"])
        mat[idx_in_order[donor], idx_in_order[recipient]] = code_of[cat]

    colors = [
        "#c7c7c7",  # unavailable
        "#1f77b4",  # unchanged blue
        "#2ca02c",  # full transfer green
        "#f1c40f",  # partial yellow
        "#8e44ad",  # novel correct purple
        "#d62728",  # novel incorrect red
    ]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, len(colors) + 0.5, 1), cmap.N)

    fig, ax = plt.subplots(figsize=(8.8, 7.6))
    im = ax.imshow(mat, cmap=cmap, norm=norm, interpolation="nearest", origin="lower")
    ax.set_title(f"Best Layer {best_layer} Patching Outcomes (Donor x Recipient)")
    ax.set_xlabel("Recipient start state (Sierpinski-ordered)")
    ax.set_ylabel("Donor start state (Sierpinski-ordered)")

    cbar = fig.colorbar(im, ax=ax, ticks=list(range(len(colors))), fraction=0.046, pad=0.04)
    cbar.ax.set_yticklabels([
        "Unavailable",
        "Recipient unchanged",
        "Full transfer",
        "Partial transfer",
        "Novel correct",
        "Disrupted",
    ])

    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def run_layer_on_pairs(
    model: ToHTransformer,
    vocab: Vocabulary,
    goal: State,
    sep2_idx: int,
    layer: int,
    pairs: Sequence[Pair],
    clean_by_state: Dict[State, Dict[str, object]],
    cached_sep_by_layer: Dict[int, torch.Tensor],
    state_to_all_index: Dict[State, int],
    device: torch.device,
) -> List[Dict[str, object]]:
    layer_cache = cached_sep_by_layer[layer]

    rows: List[Dict[str, object]] = []
    for donor, recipient in pairs:
        donor_idx = state_to_all_index[donor]
        donor_vec = layer_cache[donor_idx]

        recipient_context = build_context_ids(recipient, goal, vocab)
        patched_ids, patched_eos = greedy_decode_ids_with_patch(
            model=model,
            context_ids=recipient_context,
            target_layer=layer,
            sep_index=sep2_idx,
            donor_sep_activation=donor_vec,
            eos_id=vocab.eos_id,
            device=device,
        )

        parsed = parse_generation(
            start=recipient,
            goal=goal,
            generated_ids=patched_ids,
            eos_seen=patched_eos,
            vocab=vocab,
        )

        donor_moves = list(clean_by_state[donor]["moves"])
        recipient_moves = list(clean_by_state[recipient]["moves"])
        patched_moves = list(parsed["moves"])

        category, partial_k = classify_patched_output(
            patched_moves=patched_moves,
            donor_moves=donor_moves,
            recipient_moves=recipient_moves,
            patched_is_correct=bool(parsed["reaches_goal"]),
        )

        rows.append(
            {
                "donor": list(donor),
                "recipient": list(recipient),
                "layer": layer,
                "category": category,
                "patched_moves": patched_moves,
                "donor_moves": donor_moves,
                "recipient_moves": recipient_moves,
                "partial_k": partial_k,
                "patched_reaches_goal": bool(parsed["reaches_goal"]),
                "patched_eos_seen": bool(parsed["eos_seen"]),
                "patched_legal": bool(parsed["legal"]),
            }
        )

    return rows


def main() -> None:
    args = parse_args()
    if args.n_disks != 4:
        raise ValueError("This experiment is defined for n_disks=4")

    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    vocab = Vocabulary()
    confirm_tokenizer_mapping(vocab)

    checkpoint_path = resolve_checkpoint_path(args.checkpoint, args.n_disks)
    model = load_model(checkpoint_path, args.n_disks, device)

    goal: State = tuple(2 for _ in range(args.n_disks))
    all_states = enumerate_states(args.n_disks)
    state_to_all_index = {s: i for i, s in enumerate(all_states)}

    print("[INFO] Caching clean second-SEP activations for all starts and layers...")
    cached_sep_by_layer = extract_sep2_activations(
        model=model,
        starts=all_states,
        goal=goal,
        vocab=vocab,
        device=device,
        batch_size=128,
    )

    sep2_idx = 2 * args.n_disks + 2
    if sep2_idx != 10:
        raise AssertionError(f"Expected second SEP index 10 for n=4, got {sep2_idx}")

    print("[INFO] Running clean greedy decode for all 81 starts...")
    clean_by_state: Dict[State, Dict[str, object]] = {}
    valid_states: List[State] = []

    for st in all_states:
        context = build_context_ids(st, goal, vocab)
        gen_ids, eos_seen = greedy_decode_ids(
            model=model,
            context_ids=context,
            eos_id=vocab.eos_id,
            device=device,
        )

        parsed = parse_generation(
            start=st,
            goal=goal,
            generated_ids=gen_ids,
            eos_seen=eos_seen,
            vocab=vocab,
        )
        clean_by_state[st] = parsed
        if bool(parsed["reaches_goal"]):
            valid_states.append(st)

    print(f"[INFO] Correct clean solutions: {len(valid_states)} / {len(all_states)}")
    if len(valid_states) < 2:
        raise RuntimeError("Need at least 2 correct states for donor/recipient patching pairs")

    full_sweep_layers = parse_layer_list(args.full_sweep_layers)
    for layer in full_sweep_layers:
        if layer < 1 or layer > model.n_layers:
            raise RuntimeError(
                f"Requested layer {layer} outside model range [1, {model.n_layers}]"
            )

    all_valid_pairs = [(a, b) for a in valid_states for b in valid_states if a != b]
    if not all_valid_pairs:
        raise RuntimeError("No valid donor/recipient pairs available")

    print(
        "[INFO] Full sweeps configured for layers "
        f"{full_sweep_layers} with {len(all_valid_pairs)} ordered pairs per layer"
    )

    full_results_by_layer: Dict[int, List[Dict[str, object]]] = {}
    for layer in full_sweep_layers:
        print(f"[INFO] Running full donor->recipient patch sweep for layer {layer}...")
        full_results_by_layer[layer] = run_layer_on_pairs(
            model=model,
            vocab=vocab,
            goal=goal,
            sep2_idx=sep2_idx,
            layer=layer,
            pairs=all_valid_pairs,
            clean_by_state=clean_by_state,
            cached_sep_by_layer=cached_sep_by_layer,
            state_to_all_index=state_to_all_index,
            device=device,
        )

    layer_summaries = {layer: summarize_layer(rows) for layer, rows in full_results_by_layer.items()}

    best_layer = max(
        full_sweep_layers,
        key=lambda l: (
            float(layer_summaries[l]["rates"][CATEGORY_FULL_TRANSFER]),
            -float(layer_summaries[l]["rates"][CATEGORY_NOVEL_INCORRECT]),
        ),
    )
    best_rows = full_results_by_layer[best_layer]
    raw_rows: List[Dict[str, object]] = []
    for layer in full_sweep_layers:
        raw_rows.extend(full_results_by_layer[layer])

    raw_json_path = output_dir / "activation_patching_results.json"
    with raw_json_path.open("w", encoding="utf-8") as f:
        json.dump(raw_rows, f, indent=2)

    # Tables
    table1_text = write_table_1(
        layer_summaries,
        output_dir / "table1_layers.txt",
        title="Table 1 - Per-layer summary (full donor->recipient sweep)",
    )

    distances = build_distances(all_states)
    state_to_idx = {s: i for i, s in enumerate(all_states)}

    best_layer_only_rows = [r for r in best_rows if int(r["layer"]) == best_layer]
    table2_text = compute_distance_bucket_table(best_layer_only_rows, distances, state_to_idx)
    table3_text = compute_donor_length_bucket_table(best_layer_only_rows)

    (output_dir / "table2_distance_buckets.txt").write_text(table2_text, encoding="utf-8")
    (output_dir / "table3_donor_length_buckets.txt").write_text(table3_text, encoding="utf-8")

    combined_tables = (
        f"{table1_text}\n"
        f"{table2_text}\n"
        f"{table3_text}\n"
    )
    (output_dir / "tables_summary.txt").write_text(combined_tables, encoding="utf-8")

    # Figures
    plot_layer_stacked_bar(layer_summaries, output_dir / "layer_category_stacked_bar.png")
    plot_best_layer_heatmap(
        best_layer=best_layer,
        all_states=all_states,
        valid_states=valid_states,
        best_layer_results=best_layer_only_rows,
        out_path=output_dir / "best_layer_81x81_heatmap.png",
    )

    metadata = {
        "checkpoint": str(checkpoint_path),
        "n_disks": args.n_disks,
        "goal": list(goal),
        "num_all_states": len(all_states),
        "num_valid_states": len(valid_states),
        "full_sweep_layers": full_sweep_layers,
        "pairs_per_full_layer": len(all_valid_pairs),
        "best_layer": best_layer,
        "best_layer_full_transfer_rate": float(layer_summaries[best_layer]["rates"][CATEGORY_FULL_TRANSFER]),
        "raw_rows": len(raw_rows),
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\n" + combined_tables)
    print(f"[INFO] Raw rows saved to {raw_json_path}")
    print(f"[INFO] Best layer: {best_layer}")
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
