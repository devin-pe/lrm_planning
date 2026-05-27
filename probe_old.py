#!/usr/bin/env python3
"""Stage 2: HF hidden-state extraction + linear probing for ToH geometry.

This script expects Stage 1 outputs in outputs/qwen_probe/:
- generated_texts.json
- valid_states.json
- validation_results.json (optional, for diagnostics)

It does not run generation. Use generate.py first.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from collections import deque
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.linalg import orthogonal_procrustes
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

State = Tuple[int, int, int, int]
Edge = Tuple[int, int]

SYSTEM_PROMPT = """You are a helpful assistant. Solve this puzzle for me.
There are three pegs and n disks of different sizes stacked on the first peg.
The disks are numbered from 1 (smallest) to n (largest). Disk moves in this puzzle should follow:
1. Only one disk can be moved at a time.
2. Each move consists of taking the upper disk from one stack and placing it on top of another stack.
3. A larger disk may not be placed on top of a smaller disk.
The goal is to move the entire stack to the third peg.

Example: With 3 disks numbered 1 (smallest), 2, and 3 (largest), the initial state is [[3, 2, 1], [], []], and a solution might be:

initial_state = [[3, 2, 1], [], []]
moves = [[1 , 0, 2], [2, 0, 1], [1, 2, 1], [3, 0, 2], [1, 1, 0], [2, 1, 2], [1, 0, 2]]

This means: Move disk 1 from peg 0 to peg 2, then move disk 2 from peg 0 to peg 1, and so on.

Requirements:
- When exploring potential solutions in your thinking process, always include the corresponding
complete list of moves.
- The positions are 0-indexed (the leftmost peg is 0).
- First restate the initial state in the same format as the question according to disk_id
- Ensure your final answer includes the complete list of moves in the format: moves = [[disk id, from peg, to peg], ...]"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 2 HF extraction + probe training")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3.6-27B")
    parser.add_argument("--n_disks", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="outputs/qwen_probe")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dtype_from_arg(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    return torch.float32


def state_key(state: State) -> str:
    return str(tuple(int(x) for x in state))


def parse_state_key(key: str) -> State:
    vals = [int(x.strip()) for x in key.strip().strip("()").split(",") if x.strip()]
    if len(vals) != 4:
        raise ValueError(f"Invalid state key format: {key}")
    return tuple(vals)  # type: ignore[return-value]


def state_tuple_to_pegs(state: State, n_disks: int) -> List[List[int]]:
    pegs = [[], [], []]
    for disk in range(n_disks, 0, -1):
        peg = int(state[disk - 1])
        pegs[peg].append(disk)
    return pegs


def enumerate_states(n_disks: int) -> List[State]:
    return [tuple(int(v) for v in s) for s in product(range(3), repeat=n_disks)]


def top_disk_per_peg(state: State, n_disks: int) -> List[Optional[int]]:
    tops: List[Optional[int]] = [None, None, None]
    for disk in range(1, n_disks + 1):
        peg = state[disk - 1]
        if tops[peg] is None:
            tops[peg] = disk
    return tops


def legal_neighbors(state: State, n_disks: int) -> List[State]:
    tops = top_disk_per_peg(state, n_disks=n_disks)
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
                nxt[src_top - 1] = dst
                out.append(tuple(int(x) for x in nxt))
    return out


def build_graph_and_distances(states: Sequence[State], n_disks: int) -> Tuple[np.ndarray, List[Edge]]:
    n = len(states)
    state_to_idx = {s: i for i, s in enumerate(states)}

    adjacency: List[List[int]] = [[] for _ in range(n)]
    edge_set = set()

    for i, s in enumerate(states):
        for nbr in legal_neighbors(s, n_disks=n_disks):
            j = state_to_idx[nbr]
            adjacency[i].append(j)
            a, b = sorted((i, j))
            edge_set.add((a, b))

    dist = np.full((n, n), np.inf, dtype=np.float32)
    for src in range(n):
        dist[src, src] = 0.0
        q = deque([src])
        while q:
            u = q.popleft()
            for v in adjacency[u]:
                if np.isinf(dist[src, v]):
                    dist[src, v] = dist[src, u] + 1.0
                    q.append(v)

    if np.isinf(dist).any():
        raise RuntimeError("Found disconnected states in ToH graph")

    return dist, sorted(edge_set)


def sierpinski_coords(states: Sequence[State], n_disks: int) -> np.ndarray:
    vertices = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, math.sqrt(3.0) / 2.0],
        ],
        dtype=np.float64,
    )

    coords = np.zeros((len(states), 2), dtype=np.float64)
    for i, st in enumerate(states):
        p = np.zeros(2, dtype=np.float64)
        scale = 1.0
        for disk in range(n_disks, 0, -1):
            peg = int(st[disk - 1])
            p += scale * vertices[peg]
            scale *= 0.5
        coords[i] = p

    coords -= coords.mean(axis=0, keepdims=True)
    coords /= np.std(coords, axis=0, keepdims=True).clip(min=1e-8)
    return coords.astype(np.float32)


def make_user_prompt(state_pegs: List[List[int]], n_disks: int) -> str:
    return (
        f"Solve this Tower of Hanoi problem with {n_disks} disks. "
        f"The initial state is {state_pegs}."
    )


def extract_initial_state_line(generated_text: str) -> Tuple[Optional[str], Optional[int], Optional[int]]:
    think_pos = generated_text.find("</think>")
    search_start = think_pos + len("</think>") if think_pos >= 0 else 0
    post = generated_text[search_start:]

    pat = re.compile(r"(?m)^\s*initial_state\s*=\s*\[\[.*\]\]\s*$")
    m = pat.search(post)
    if m is None:
        return None, None, None

    line = m.group(0)
    start_char = search_start + m.start()
    end_char_no_newline = search_start + m.end()
    return line, start_char, end_char_no_newline


def truncate_through_initial_line(generated_text: str) -> Tuple[str, Optional[str], bool]:
    line, start_char, end_char_no_newline = extract_initial_state_line(generated_text)
    if line is None or start_char is None or end_char_no_newline is None:
        return generated_text, None, False

    end = end_char_no_newline
    if end < len(generated_text) and generated_text[end] == "\n":
        end += 1
    return generated_text[:end], line, True


def locate_line_token_index(
    tokenizer: AutoTokenizer,
    system_prompt: str,
    user_prompt: str,
    assistant_truncated: str,
    line: str,
) -> int:
    def _to_token_list(ids_obj: object) -> List[int]:
        # apply_chat_template(tokenize=True) can return different container
        # types across tokenizer implementations (list, tensor, BatchEncoding).
        if isinstance(ids_obj, list):
            if ids_obj and isinstance(ids_obj[0], list):
                return [int(x) for x in ids_obj[0]]
            return [int(x) for x in ids_obj]

        # torch.Tensor path
        if hasattr(ids_obj, "tolist"):
            vals = ids_obj.tolist()
            if isinstance(vals, list) and vals and isinstance(vals[0], list):
                return [int(x) for x in vals[0]]
            if isinstance(vals, list):
                return [int(x) for x in vals]

        # BatchEncoding / dict-like path
        if hasattr(ids_obj, "get"):
            input_ids = ids_obj.get("input_ids")
            if input_ids is not None:
                return _to_token_list(input_ids)

        if hasattr(ids_obj, "input_ids"):
            return _to_token_list(getattr(ids_obj, "input_ids"))

        raise TypeError(f"Unsupported token container type: {type(ids_obj)!r}")

    def _find_subsequence(haystack: Sequence[int], needle: Sequence[int]) -> int:
        if not needle or len(needle) > len(haystack):
            return -1
        end = len(haystack) - len(needle) + 1
        for i in range(end):
            if haystack[i : i + len(needle)] == list(needle):
                return i
        return -1

    line_start = assistant_truncated.find(line)
    if line_start < 0:
        raise ValueError("Could not locate initial_state line in assistant text")

    assistant_before = assistant_truncated[:line_start]
    assistant_through_line = assistant_truncated[: line_start + len(line)]

    msgs_before = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": assistant_before},
    ]
    msgs_through = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": assistant_through_line},
    ]

    ids_before = _to_token_list(
        tokenizer.apply_chat_template(msgs_before, tokenize=True, add_generation_prompt=False)
    )
    ids_through = _to_token_list(
        tokenizer.apply_chat_template(msgs_through, tokenize=True, add_generation_prompt=False)
    )
    if len(ids_through) <= len(ids_before):
        # Fallback for tokenizers/chat templates that normalize whitespace/newlines
        # and make prefix-length differencing ambiguous.
        msgs_full = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_truncated},
        ]
        ids_full = _to_token_list(
            tokenizer.apply_chat_template(msgs_full, tokenize=True, add_generation_prompt=False)
        )

        # Try a few variants to survive newline normalization.
        candidate_lines = [line, f"{line}\n", line.rstrip(), f"{line.rstrip()}\n"]
        for cand in candidate_lines:
            line_ids = tokenizer.encode(cand, add_special_tokens=False)
            if not line_ids:
                continue
            start = _find_subsequence(ids_full, line_ids)
            if start >= 0:
                return start + len(line_ids) - 1

        # Last-resort fallback for chat templates/tokenizers that normalize the
        # assistant text so aggressively that exact subsequence matching fails.
        # Probe the final non-special token rather than aborting the entire run.
        special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
        for idx in range(len(ids_full) - 1, -1, -1):
            if ids_full[idx] not in special_ids:
                return idx

        if ids_full:
            return len(ids_full) - 1
        raise RuntimeError("Tokenization produced an empty sequence")
    return len(ids_through) - 1


def load_stage1_artifacts(output_dir: Path) -> Tuple[Dict[str, str], List[State], np.ndarray]:
    generated_path = output_dir / "generated_texts.json"
    valid_states_path = output_dir / "valid_states.json"

    if not valid_states_path.exists() or not generated_path.exists():
        print(
            "[ERROR] Missing Stage 1 artifacts. Run generate.py first to create "
            "valid_states.json and generated_texts.json.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    generated_obj = json.loads(generated_path.read_text(encoding="utf-8"))
    if not isinstance(generated_obj, dict):
        raise RuntimeError("generated_texts.json must be a dict mapping state tuple string -> text")

    generated_texts: Dict[str, str] = {}
    for k, v in generated_obj.items():
        generated_texts[str(k)] = str(v)

    valid_obj = json.loads(valid_states_path.read_text(encoding="utf-8"))
    if not isinstance(valid_obj, list):
        raise RuntimeError("valid_states.json must be a list")

    valid_states: List[State] = []
    for row in valid_obj:
        if isinstance(row, list) and len(row) == 4:
            valid_states.append(tuple(int(x) for x in row))
        elif isinstance(row, str):
            valid_states.append(parse_state_key(row))
        else:
            raise RuntimeError(f"Invalid valid_states entry: {row!r}")

    all_states = enumerate_states(4)
    state_to_idx = {s: i for i, s in enumerate(all_states)}
    valid_mask = np.zeros(len(all_states), dtype=bool)
    for st in valid_states:
        if st not in state_to_idx:
            raise RuntimeError(f"Unknown state in valid_states.json: {st}")
        valid_mask[state_to_idx[st]] = True

    return generated_texts, valid_states, valid_mask


def extract_hidden_states_for_valid(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    valid_states: Sequence[State],
    generated_texts: Dict[str, str],
    layers_to_probe: Sequence[int],
    n_disks: int,
    device: torch.device,
) -> torch.Tensor:
    n_valid = len(valid_states)
    n_layers = len(layers_to_probe)
    hidden_dim = int(model.config.hidden_size)
    hidden = torch.full((n_valid, n_layers, hidden_dim), torch.nan, dtype=torch.float32)

    pbar = tqdm(range(n_valid), desc="Extracting hidden states", unit="state")
    for out_i in pbar:
        st = valid_states[out_i]
        key = state_key(st)
        if key not in generated_texts:
            raise RuntimeError(f"Missing generated text for valid state {st}")

        full_generated = generated_texts[key]
        truncated_assistant, line, found_line = truncate_through_initial_line(full_generated)
        if not found_line or line is None:
            raise RuntimeError(f"No initial_state line in generated text for state {st}")

        user_prompt = make_user_prompt(state_tuple_to_pegs(st, n_disks=n_disks), n_disks=n_disks)
        token_index = locate_line_token_index(
            tokenizer=tokenizer,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            assistant_truncated=truncated_assistant,
            line=line,
        )

        msgs_full = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": truncated_assistant},
        ]
        full_ids = tokenizer.apply_chat_template(
            msgs_full,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            out = model(full_ids, output_hidden_states=True, use_cache=False)

        hs = out.hidden_states
        if hs is None:
            raise RuntimeError("Model did not return hidden_states")

        for li, layer_idx in enumerate(layers_to_probe):
            h = hs[layer_idx + 1][0, token_index, :].detach().float().cpu()
            hidden[out_i, li] = h

        if out_i < 3:
            lo = max(0, token_index - 6)
            hi = min(full_ids.shape[1], token_index + 7)
            window_ids = full_ids[0, lo:hi].tolist()
            window_text = tokenizer.decode(window_ids, skip_special_tokens=False)
            print(
                f"[SANITY] state={st} token_index={token_index} line={line!r}\n"
                f"[SANITY] token-window: {window_text!r}"
            )

    return hidden


def train_probe(
    x: np.ndarray,
    y: np.ndarray,
    epochs: int,
    lr: float,
    device: torch.device,
    log_prefix: str,
) -> Tuple[nn.Linear, np.ndarray, float]:
    xb = torch.tensor(x, dtype=torch.float32, device=device)
    yb = torch.tensor(y, dtype=torch.float32, device=device)

    probe = nn.Linear(x.shape[1], 2, bias=True).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)

    final_loss = float("nan")
    for ep in range(1, epochs + 1):
        opt.zero_grad(set_to_none=True)
        pred = probe(xb)
        loss = F.mse_loss(pred, yb)
        loss.backward()
        opt.step()

        final_loss = float(loss.item())
        if ep == 1 or ep % 200 == 0 or ep == epochs:
            print(f"{log_prefix} epoch={ep:4d} loss={final_loss:.8f}")

    with torch.no_grad():
        pred = probe(xb).detach().cpu().numpy().astype(np.float32)

    return probe, pred, final_loss


def procrustes_align(pred: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, float]:
    pred = pred.astype(np.float64)
    target = target.astype(np.float64)

    pred_mean = pred.mean(axis=0, keepdims=True)
    target_mean = target.mean(axis=0, keepdims=True)
    pred_c = pred - pred_mean
    target_c = target - target_mean

    pred_norm = np.linalg.norm(pred_c)
    target_norm = np.linalg.norm(target_c)
    if pred_norm < 1e-12 or target_norm < 1e-12:
        aligned = pred.copy()
        mse = float(np.mean((aligned - target) ** 2))
        return aligned.astype(np.float32), mse

    pred_u = pred_c / pred_norm
    target_u = target_c / target_norm

    r, _ = orthogonal_procrustes(pred_u, target_u)
    pred_r = pred_u @ r

    denom = float(np.sum(pred_r * pred_r))
    alpha = float(np.sum(pred_r * target_u) / denom) if denom > 1e-12 else 1.0

    aligned_u = alpha * pred_r
    aligned = aligned_u * target_norm + target_mean

    mse = float(np.mean((aligned - target) ** 2))
    return aligned.astype(np.float32), mse


def plot_layer_projection(
    pred_coords: np.ndarray,
    true_coords: np.ndarray,
    valid_mask: np.ndarray,
    states: Sequence[State],
    edges: Sequence[Edge],
    title: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)

    for i, j in edges:
        ax.plot(
            [true_coords[i, 0], true_coords[j, 0]],
            [true_coords[i, 1], true_coords[j, 1]],
            color="gray",
            linewidth=0.8,
            alpha=0.35,
            zorder=1,
        )

    largest_disk_peg = np.array([s[3] for s in states], dtype=np.int64)
    peg_colors = {0: "red", 1: "blue", 2: "green"}

    missing_mask = ~valid_mask
    if np.any(missing_mask):
        ax.scatter(
            true_coords[missing_mask, 0],
            true_coords[missing_mask, 1],
            s=70,
            facecolors="none",
            edgecolors="gray",
            linewidths=1.2,
            alpha=0.9,
            zorder=2,
            label="missing / failed",
        )

    for peg in [0, 1, 2]:
        mask = (largest_disk_peg == peg) & valid_mask & np.all(np.isfinite(pred_coords), axis=1)
        ax.scatter(
            pred_coords[mask, 0],
            pred_coords[mask, 1],
            s=80,
            alpha=0.95,
            color=peg_colors[peg],
            zorder=3,
            label=f"largest disk on peg {peg}",
        )

    ax.text(
        0.01,
        0.98,
        f"{int(valid_mask.sum())}/81 optimal solutions",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )

    ax.set_title(title)
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_loss_summary(layer_ids: Sequence[int], losses: Sequence[float], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax.plot(layer_ids, losses, marker="o", linewidth=2)
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Post-Procrustes MSE")
    ax.set_title("Qwen ToH Probe Loss by Layer")
    ax.grid(alpha=0.25)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.n_disks != 4:
        raise ValueError("This script is configured for 4-disk ToH only")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    generated_texts, valid_states, valid_mask = load_stage1_artifacts(out_dir)

    dtype = dtype_from_arg(args.dtype)
    device = torch.device(args.device)

    print("[INFO] Loading tokenizer/model (HF for extraction)")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        device_map="auto" if device.type == "cuda" else None,
        trust_remote_code=True,
    )
    if device.type != "cuda":
        model.to(device)
    model.eval()

    n_layers_total = int(model.config.num_hidden_layers)
    requested = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44]
    if (n_layers_total - 1) not in requested:
        requested.append(n_layers_total - 1)
    layers_to_probe = sorted([l for l in requested if 0 <= l < n_layers_total])

    print(f"[INFO] num_hidden_layers={n_layers_total}")
    print(f"[INFO] Probing layers: {layers_to_probe}")

    states = enumerate_states(args.n_disks)
    state_to_idx = {s: i for i, s in enumerate(states)}
    if len(states) != 3 ** args.n_disks:
        raise RuntimeError("State enumeration mismatch")

    valid_indices = [state_to_idx[s] for s in valid_states]
    if len(valid_indices) < 1:
        raise RuntimeError("No valid optimal states available for probing")

    if len(valid_indices) < 40:
        print(
            f"[WARN] Only {len(valid_indices)} optimal solutions passed validation (<40). "
            "Probe results may be unreliable."
        )

    goal = (2, 2, 2, 2)
    print(f"[INFO] Enumerated {len(states)} states; fixed goal={goal}")

    _graph_dist, edges = build_graph_and_distances(states, n_disks=args.n_disks)
    target_coords = sierpinski_coords(states, n_disks=args.n_disks)

    hs_t0 = time.time()
    hidden_states = extract_hidden_states_for_valid(
        model=model,
        tokenizer=tokenizer,
        valid_states=valid_states,
        generated_texts=generated_texts,
        layers_to_probe=layers_to_probe,
        n_disks=args.n_disks,
        device=device,
    )
    print(
        f"[INFO] Hidden extraction done in {(time.time() - hs_t0) / 60.0:.2f} min; "
        f"valid={len(valid_indices)}/{len(states)}"
    )

    torch.save(
        {
            "hidden_states": hidden_states,
            "layers": layers_to_probe,
            "states": [list(map(int, s)) for s in states],
            "valid_state_indices": valid_indices,
            "valid_mask": valid_mask.tolist(),
            "goal_state": list(goal),
        },
        out_dir / "hidden_states.pt",
    )

    if len(valid_indices) < 10:
        raise RuntimeError("Too few valid samples to train probes")

    x_all = hidden_states.numpy()
    y_all = target_coords[np.array(valid_indices, dtype=np.int64), :]

    per_layer_mse: Dict[int, float] = {}

    probe_t0 = time.time()
    for layer_pos, layer_idx in enumerate(layers_to_probe):
        layer_start = time.time()
        x = x_all[:, layer_pos, :]
        y = y_all

        probe, pred, train_loss = train_probe(
            x=x,
            y=y,
            epochs=args.epochs,
            lr=args.lr,
            device=device,
            log_prefix=f"[Layer {layer_idx}]",
        )

        pred_aligned_valid, aligned_mse = procrustes_align(pred, y)
        per_layer_mse[layer_idx] = float(aligned_mse)

        layer_pred_coords = np.full((len(states), 2), np.nan, dtype=np.float32)
        layer_pred_coords[np.array(valid_indices, dtype=np.int64)] = pred_aligned_valid

        plot_layer_projection(
            pred_coords=layer_pred_coords,
            true_coords=target_coords,
            valid_mask=valid_mask,
            states=states,
            edges=edges,
            title=f"Layer {layer_idx} - post-Procrustes MSE={aligned_mse:.6f}",
            out_path=out_dir / f"layer_{layer_idx:02d}_probe_projection.png",
        )

        torch.save(
            {
                "layer": int(layer_idx),
                "state_dict": probe.state_dict(),
                "train_loss": float(train_loss),
                "aligned_mse": float(aligned_mse),
                "valid_indices": valid_indices,
            },
            out_dir / f"probe_layer_{layer_idx}.pt",
        )

        print(
            f"[INFO] layer={layer_idx:2d} train_loss={train_loss:.8f} "
            f"aligned_mse={aligned_mse:.8f} time={(time.time() - layer_start):.1f}s"
        )

    layer_ids_sorted = sorted(per_layer_mse.keys())
    losses_sorted = [per_layer_mse[l] for l in layer_ids_sorted]
    plot_loss_summary(layer_ids_sorted, losses_sorted, out_dir / "probe_loss_vs_layer.png")

    summary = {
        "model_name": args.model_name,
        "n_disks": args.n_disks,
        "goal_state": list(goal),
        "num_states": len(states),
        "num_valid_states": int(valid_mask.sum()),
        "layers": layer_ids_sorted,
        "post_procrustes_mse": {str(k): float(v) for k, v in per_layer_mse.items()},
        "graph_num_edges": len(edges),
        "runtime_probe_seconds": float(time.time() - probe_t0),
        "stage1_generated_texts": str(out_dir / "generated_texts.json"),
        "stage1_valid_states": str(out_dir / "valid_states.json"),
    }
    (out_dir / "probe_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[INFO] Saved artifacts to {out_dir}")


if __name__ == "__main__":
    main()
