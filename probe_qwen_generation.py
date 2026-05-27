#!/usr/bin/env python3
"""Probe Qwen3-27B hidden states at key positions during generation.

For each of the 81 4-disk ToH problems, loads prompt + already-generated text,
runs ONE forward pass, and extracts hidden states at:

  A: last prompt token (baseline)
  B: token just before the final "moves = [" (model commits to answer)
  C: closing "]" of each move entry [d, f, t] in the final moves list

Trains distance-matching linear probes and per-disk logistic classifiers,
then produces comparison tables and Sierpinski plots.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from collections import defaultdict, deque
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from prompts import create_nonstandard_prompt

State = Tuple[int, ...]


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Probe Qwen hidden states during generation")
    p.add_argument("--model_name", default="Qwen/Qwen3-27B")
    p.add_argument("--generated_texts", default="outputs/qwen_probe/generated_texts.json")
    p.add_argument("--n_disks", type=int, default=4)
    p.add_argument("--layers", default="24,36,48")
    p.add_argument("--output_dir", default="outputs/qwen_probe_generation")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def dtype_from_str(s: str) -> torch.dtype:
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[s]


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── State / graph helpers ──────────────────────────────────────────────────────

def enumerate_states(n_disks: int) -> List[State]:
    return [tuple(int(v) for v in s) for s in product(range(3), repeat=n_disks)]


def state_tuple_to_pegs(state: State, n_disks: int) -> List[List[int]]:
    pegs: List[List[int]] = [[], [], []]
    for disk in range(n_disks, 0, -1):
        pegs[int(state[disk - 1])].append(disk)
    return pegs


def pegs_to_state(pegs: List[List[int]], n_disks: int) -> State:
    s = [0] * n_disks
    for pi, peg in enumerate(pegs):
        for disk in peg:
            s[disk - 1] = pi
    return tuple(s)


def state_label(state: State) -> str:
    return "".join(str(x) for x in state)


def top_disk_per_peg(state: State) -> List[Optional[int]]:
    tops: List[Optional[int]] = [None, None, None]
    for disk in range(1, len(state) + 1):
        peg = state[disk - 1]
        if tops[peg] is None:
            tops[peg] = disk
    return tops


def legal_neighbors(state: State, n_disks: int) -> List[State]:
    tops = top_disk_per_peg(state)
    out: List[State] = []
    for src in range(3):
        d = tops[src]
        if d is None:
            continue
        for dst in range(3):
            if src == dst:
                continue
            if tops[dst] is None or d < tops[dst]:
                nxt = list(state)
                nxt[d - 1] = dst
                out.append(tuple(nxt))
    return out


def build_graph(states: List[State], n_disks: int):
    n = len(states)
    idx = {s: i for i, s in enumerate(states)}
    adj: List[List[int]] = [[] for _ in range(n)]
    edges: Set[Tuple[int, int]] = set()
    nbrs: Dict[int, Set[int]] = {i: set() for i in range(n)}
    for i, s in enumerate(states):
        for nb in legal_neighbors(s, n_disks):
            j = idx[nb]
            adj[i].append(j)
            nbrs[i].add(j)
            a, b = sorted((i, j))
            edges.add((a, b))
    dist = np.full((n, n), np.inf, dtype=np.float32)
    for src in range(n):
        dist[src, src] = 0.0
        q = deque([src])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if np.isinf(dist[src, v]):
                    dist[src, v] = dist[src, u] + 1.0
                    q.append(v)
    return dist, sorted(edges), nbrs


def norm_dist(dist: np.ndarray) -> Tuple[np.ndarray, float, float]:
    mu, sigma = float(dist.mean()), float(dist.std())
    if sigma < 1e-8:
        sigma = 1.0
    return ((dist - mu) / sigma).astype(np.float32), mu, sigma


# ── Prompt building (mirrors probe.py / generate.py) ──────────────────────────

def build_prompt_str(tokenizer, state: State, state_idx: int, n_disks: int) -> str:
    goal_pegs = [[], [], list(range(n_disks, 0, -1))]
    pegs = state_tuple_to_pegs(state, n_disks)
    system_prompt, user_prompt, _ = create_nonstandard_prompt(
        num_disks=n_disks, problem_id=state_idx, seed=0,
        initial_state_override=pegs, goal_state_override=goal_pegs,
    )
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": system_prompt},
         {"role": "user", "content": user_prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


# ── Position finding via offset mapping ───────────────────────────────────────

def char_to_token(offset_mapping: List[Tuple[int, int]], char_pos: int) -> Optional[int]:
    """Return the index of the token whose span contains char_pos."""
    for i, (s, e) in enumerate(offset_mapping):
        if s <= char_pos < e:
            return i
    # fallback: last token that starts at or before char_pos
    best: Optional[int] = None
    for i, (s, e) in enumerate(offset_mapping):
        if s <= char_pos:
            best = i
    return best


def find_position_B(
    gen_text: str,
    prompt_len_chars: int,
    offset_mapping: List[Tuple[int, int]],
) -> Optional[int]:
    """Token index just before the final 'moves = [' in the full sequence."""
    pattern = re.compile(r'moves\s*=\s*\[', re.IGNORECASE)
    matches = list(pattern.finditer(gen_text))
    if not matches:
        return None
    char_in_gen = matches[-1].start()
    char_in_full = prompt_len_chars + char_in_gen
    tok = char_to_token(offset_mapping, char_in_full)
    if tok is None or tok == 0:
        return None
    return tok - 1


def find_positions_C(
    gen_text: str,
    prompt_len_chars: int,
    offset_mapping: List[Tuple[int, int]],
    start_pegs: List[List[int]],
    n_disks: int,
) -> List[Tuple[int, State]]:
    """
    For each move entry in the LAST moves = [...] block, return
    (token_index_of_closing_bracket, true_state_after_move).
    Only includes moves that are legal.
    """
    # Find the last complete moves = [...] block
    last_block_match: Optional[re.Match] = None
    last_block_text: Optional[str] = None
    last_block_start: Optional[int] = None  # char offset in gen_text

    for m in re.finditer(r'moves\s*=\s*\[', gen_text, re.IGNORECASE):
        bracket_start = gen_text.find('[', m.start())
        if bracket_start < 0:
            continue
        depth = 0
        for idx in range(bracket_start, len(gen_text)):
            c = gen_text[idx]
            if c == '[':
                depth += 1
            elif c == ']':
                depth -= 1
                if depth == 0:
                    candidate = gen_text[bracket_start:idx + 1]
                    if candidate != '[]':
                        last_block_match = m
                        last_block_text = candidate
                        last_block_start = bracket_start
                    break

    if last_block_text is None or last_block_start is None:
        return []

    # Find all [d, f, t] move entries within the block
    move_re = re.compile(r'\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]')
    pegs = [list(p) for p in start_pegs]
    results: List[Tuple[int, State]] = []

    for move_match in move_re.finditer(last_block_text):
        disk = int(move_match.group(1))
        from_peg = int(move_match.group(2))
        to_peg = int(move_match.group(3))

        # Validate and apply move
        if (1 <= disk <= n_disks and 0 <= from_peg <= 2 and 0 <= to_peg <= 2
                and pegs[from_peg] and pegs[from_peg][-1] == disk
                and (not pegs[to_peg] or pegs[to_peg][-1] > disk)):
            pegs[to_peg].append(pegs[from_peg].pop())
            true_state = pegs_to_state(pegs, n_disks)

            # Char position of the closing ']' of this move entry (in gen_text)
            close_char_in_block = move_match.end() - 1  # the ']'
            close_char_in_gen = last_block_start + close_char_in_block
            close_char_in_full = prompt_len_chars + close_char_in_gen
            tok = char_to_token(offset_mapping, close_char_in_full)
            if tok is not None:
                results.append((tok, true_state))
        else:
            break  # first illegal move: stop tracking

    return results


# ── Model loading ──────────────────────────────────────────────────────────────

def load_model_and_tokenizer(model_name: str, dtype: torch.dtype, device_str: str):
    print(f"[INFO] Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[INFO] Loading model: {model_name}")
    load_kw = dict(
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map="auto" if device_str == "cuda" else None,
    )
    try:
        raw = AutoModelForCausalLM.from_pretrained(model_name, **load_kw)
    except (ValueError, KeyError, AttributeError):
        raw = AutoModel.from_pretrained(model_name, **load_kw)

    if hasattr(raw, "model") and hasattr(raw.model, "language_model"):
        print("[INFO] VLM detected — using model.model.language_model")
        model = raw.model.language_model
        cfg = getattr(raw.config, "text_config", raw.config)
    else:
        model = raw
        cfg = raw.config

    device = torch.device(device_str)
    if device_str != "cuda":
        model.to(device)
    model.eval()
    return model, tokenizer, cfg, device


def get_layers_list(model: nn.Module):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "layers"):
        return model.layers
    raise RuntimeError("Cannot find transformer layers")


# ── Hidden state extraction (one problem at a time) ───────────────────────────

def extract_at_positions(
    model: nn.Module,
    input_ids: torch.Tensor,  # [1, seq_len]
    layer_ids: List[int],      # 1-based
    positions: List[int],      # token indices to extract
    device: torch.device,
    hidden_size: int,
) -> Dict[int, Dict[int, torch.Tensor]]:
    """
    Returns: {layer_id: {position: hidden_vec (float32 CPU)}}
    """
    layers_list = get_layers_list(model)
    captured: Dict[int, torch.Tensor] = {}

    def make_hook(lid: int):
        def hook_fn(module, input, output):
            h = output[0] if isinstance(output, tuple) else output
            captured[lid] = h.detach().float().cpu()
        return hook_fn

    hooks = [layers_list[l - 1].register_forward_hook(make_hook(l)) for l in layer_ids]
    try:
        input_ids = input_ids.to(device)
        with torch.no_grad():
            model(input_ids=input_ids, use_cache=False)
        result: Dict[int, Dict[int, torch.Tensor]] = {}
        for lid in layer_ids:
            result[lid] = {}
            hs = captured[lid]  # [1, seq_len, hidden_size]
            for pos in positions:
                if 0 <= pos < hs.shape[1]:
                    result[lid][pos] = hs[0, pos, :].clone()  # clone breaks the view so we don't carry the full seq_len storage
                else:
                    result[lid][pos] = torch.zeros(hidden_size)
    finally:
        for h in hooks:
            h.remove()
    return result


# ── Distance-matching probe ────────────────────────────────────────────────────

def train_probe(
    x: torch.Tensor,
    y_dist_norm: torch.Tensor,
    epochs: int,
    lr: float,
    device: torch.device,
) -> Tuple[np.ndarray, float]:
    x = x.to(device)
    y = y_dist_norm.to(device)
    probe = nn.Linear(x.shape[1], 2, bias=True).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    for ep in range(1, epochs + 1):
        opt.zero_grad(set_to_none=True)
        z = probe(x)
        loss = torch.mean((torch.cdist(z, z, p=2) - y) ** 2)
        loss.backward()
        opt.step()
        if ep % 500 == 0 or ep == epochs:
            print(f"  epoch={ep} loss={loss.item():.6f}")
    with torch.no_grad():
        coords = probe(x).detach().cpu().numpy().astype(np.float32)
    return coords, float(loss.item())


def evaluate_probe(
    coords: np.ndarray,
    true_dist: np.ndarray,
    nbrs: Dict[int, Set[int]],
) -> Tuple[float, float]:
    pred = np.linalg.norm(coords[:, None] - coords[None], axis=2)
    ui, uj = np.triu_indices(pred.shape[0], k=1)
    rho, _ = spearmanr(pred[ui, uj], true_dist[ui, uj])
    correct = sum(int(np.argmin(np.where(np.arange(len(coords)) == i,
                                         np.inf, pred[i]))) in nbrs[i]
                  for i in range(len(coords)))
    return float(rho), correct / len(coords)


# ── Per-disk logistic regression ───────────────────────────────────────────────

def per_disk_accuracy(
    vecs: np.ndarray,    # [N, D]
    states: List[State], # [N]
    n_disks: int,
) -> List[float]:
    """Return cross-validated accuracy for each disk's peg classification."""
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print("[WARN] sklearn not available; skipping per-disk accuracy")
        return [float("nan")] * n_disks

    scaler = StandardScaler()
    X = scaler.fit_transform(vecs)
    accs = []
    for disk in range(n_disks):
        y = np.array([s[disk] for s in states])
        n_splits = min(5, len(np.unique(y, return_counts=True)[1]))
        if n_splits < 2 or len(X) < 10:
            accs.append(float("nan"))
            continue
        clf = LogisticRegression(max_iter=1000, C=1.0)
        scores = cross_val_score(clf, X, y, cv=n_splits, scoring="accuracy")
        accs.append(float(scores.mean()))
    return accs


# ── Visualization ──────────────────────────────────────────────────────────────

def plot_sierpinski(
    coords: np.ndarray,
    states: List[State],
    edges: List[Tuple[int, int]],
    title: str,
    out_path: Path,
    color_by_disk: int = 3,
    highlight: Optional[Dict[int, str]] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    for i, j in edges:
        ax.plot([coords[i, 0], coords[j, 0]], [coords[i, 1], coords[j, 1]],
                color="gray", lw=0.8, alpha=0.5, zorder=1)

    if highlight is not None:
        for i, color in highlight.items():
            ax.scatter(coords[i, 0], coords[i, 1], s=100, color=color,
                       alpha=0.9, zorder=2)
    else:
        palette = {0: "red", 1: "blue", 2: "green"}
        disk_vals = np.array([s[color_by_disk] for s in states])
        for peg in range(3):
            mask = disk_vals == peg
            ax.scatter(coords[mask, 0], coords[mask, 1], s=80,
                       color=palette[peg], alpha=0.9, zorder=2,
                       label=f"disk[{color_by_disk}] on peg {peg}")
        ax.legend(loc="best")

    for i, st in enumerate(states):
        ax.text(coords[i, 0], coords[i, 1], state_label(st), fontsize=6,
                ha="center", va="center", zorder=3,
                bbox={"boxstyle": "round,pad=0.15", "facecolor": "white",
                      "alpha": 0.75, "edgecolor": "none"})

    ax.set_title(title)
    ax.grid(alpha=0.2)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_side_by_side(
    coords_list: List[np.ndarray],
    titles: List[str],
    states_list: List[List[State]],   # one per subplot
    out_path: Path,
    color_by_disk: int = 3,
) -> None:
    n = len(coords_list)
    fig, axes = plt.subplots(1, n, figsize=(10 * n, 8), constrained_layout=True)
    if n == 1:
        axes = [axes]
    palette = {0: "red", 1: "blue", 2: "green"}
    for ax, coords, title, sub_states in zip(axes, coords_list, titles, states_list):
        disk_vals = np.array([s[color_by_disk] for s in sub_states])
        for peg in range(3):
            mask = disk_vals == peg
            ax.scatter(coords[mask, 0], coords[mask, 1], s=60,
                       color=palette[peg], alpha=0.9, zorder=2)
        for i, st in enumerate(sub_states):
            ax.text(coords[i, 0], coords[i, 1], state_label(st), fontsize=5,
                    ha="center", va="center", zorder=3,
                    bbox={"boxstyle": "round,pad=0.1", "facecolor": "white",
                          "alpha": 0.7, "edgecolor": "none"})
        ax.set_title(title)
        ax.grid(alpha=0.2)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    layer_ids = [int(x) for x in args.layers.split(",")]
    n_disks = args.n_disks
    t0 = time.time()

    print(f"[INFO] Loading generated texts from {args.generated_texts}")
    with open(args.generated_texts) as f:
        generated_texts: Dict[str, str] = json.load(f)

    states = enumerate_states(n_disks)
    if len(states) != 81:
        raise RuntimeError(f"Expected 81 states, got {len(states)}")

    state_to_idx = {s: i for i, s in enumerate(states)}

    print("[INFO] Building ToH graph distances")
    true_dist, edges, nbrs = build_graph(states, n_disks)

    model, tokenizer, cfg, device = load_model_and_tokenizer(
        args.model_name, dtype_from_str(args.dtype), args.device)

    n_layers_total = int(cfg.num_hidden_layers)
    hidden_size = int(cfg.hidden_size)
    print(f"[INFO] num_hidden_layers={n_layers_total}  hidden_size={hidden_size}")
    print(f"[INFO] probing layers={layer_ids}")

    # Storage: {layer: {pos_name: list of (state_idx, vec)}}
    pos_A: Dict[int, List[Tuple[int, torch.Tensor]]] = {l: [] for l in layer_ids}
    pos_B: Dict[int, List[Tuple[int, torch.Tensor]]] = {l: [] for l in layer_ids}
    # pos_C: {layer: list of (true_state, vec)}
    pos_C: Dict[int, List[Tuple[State, torch.Tensor]]] = {l: [] for l in layer_ids}

    # Track which problems have position B/C
    state_has_B: List[bool] = [False] * len(states)
    state_is_correct: List[bool] = [False] * len(states)

    # Load evaluation results to know correct/failed
    _eval_dir = Path(args.generated_texts).parent
    eval_path = (
        _eval_dir / "evaluate_results.json"
        if (_eval_dir / "evaluate_results.json").exists()
        else _eval_dir / "evaluate_qwen_results.json"
    )
    correct_states: Set[str] = set()
    if eval_path.exists():
        with open(eval_path) as f:
            eval_data = json.load(f)
        correct_cats = {"CORRECT_OPTIMAL", "CORRECT_SUBOPTIMAL"}
        for rec in eval_data.get("records", []):
            if rec["category"] in correct_cats:
                correct_states.add(rec["state"])
    else:
        print(f"[WARN] {eval_path} not found; correct/failed split unavailable")

    print(f"\n[INFO] Processing {len(states)} problems (one forward pass each)...")

    for prob_idx, state in enumerate(states):
        state_key = str(tuple(int(x) for x in state))
        gen_text = generated_texts.get(state_key, "")
        if not gen_text:
            print(f"  [{prob_idx+1:3d}/81] {state_key}: no generated text, skipping")
            continue

        state_is_correct[prob_idx] = state_key in correct_states

        print(f"  [{prob_idx+1:3d}/81] {state_key}  correct={state_is_correct[prob_idx]}", end=" ")

        # Build prompt string
        prompt_text = build_prompt_str(tokenizer, state, prob_idx, n_disks)
        full_text = prompt_text + gen_text

        # Tokenize for forward pass
        try:
            enc = tokenizer(full_text, return_tensors="pt", add_special_tokens=False,
                            truncation=False)
            input_ids = enc["input_ids"]
            seq_len = input_ids.shape[1]

            # Tokenize with offset mapping (same settings) for position lookup
            enc_off = tokenizer(full_text, return_offsets_mapping=True,
                                add_special_tokens=False, truncation=False)
            offset_mapping: List[Tuple[int, int]] = enc_off["offset_mapping"]
        except Exception as e:
            print(f"[tokenization error: {e}]")
            continue

        # Prompt length in tokens
        try:
            prompt_enc = tokenizer(prompt_text, return_tensors="pt",
                                   add_special_tokens=False, truncation=False)
            prompt_tok_len = prompt_enc["input_ids"].shape[1]
        except Exception as e:
            print(f"[prompt tok error: {e}]")
            continue

        pos_a = prompt_tok_len - 1

        # Find position B
        p_b = find_position_B(gen_text, len(prompt_text), offset_mapping)

        # Find positions C with true intermediate states
        start_pegs = state_tuple_to_pegs(state, n_disks)
        pos_c_list = find_positions_C(gen_text, len(prompt_text), offset_mapping,
                                      start_pegs, n_disks)

        positions_to_extract = {pos_a}
        if p_b is not None:
            positions_to_extract.add(p_b)
        for tok_idx, _ in pos_c_list:
            positions_to_extract.add(tok_idx)

        positions_list = sorted(positions_to_extract)
        print(f"seqlen={seq_len}  posA={pos_a}  posB={p_b}  nC={len(pos_c_list)}")

        # Forward pass
        try:
            extracted = extract_at_positions(
                model, input_ids, layer_ids, positions_list, device, hidden_size)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"    [OOM on seq_len={seq_len}; skipping]")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            raise

        # Store position A
        for l in layer_ids:
            if pos_a in extracted[l]:
                pos_A[l].append((prob_idx, extracted[l][pos_a]))

        # Store position B
        if p_b is not None and p_b in extracted.get(layer_ids[0], {}):
            state_has_B[prob_idx] = True
            for l in layer_ids:
                if p_b in extracted[l]:
                    pos_B[l].append((prob_idx, extracted[l][p_b]))

        # Store position C
        for tok_idx, true_state in pos_c_list:
            for l in layer_ids:
                if tok_idx in extracted[l]:
                    pos_C[l].append((true_state, extracted[l][tok_idx]))

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n[INFO] Extraction done in {time.time()-t0:.0f}s")
    print(f"  Position A: {len(pos_A[layer_ids[0]])} vectors")
    print(f"  Position B: {len(pos_B[layer_ids[0]])} vectors")
    print(f"  Position C: {len(pos_C[layer_ids[0]])} vectors")

    # Save raw hidden states
    torch.save({
        "pos_A": {l: pos_A[l] for l in layer_ids},
        "pos_B": {l: pos_B[l] for l in layer_ids},
        "pos_C": {l: pos_C[l] for l in layer_ids},
        "states": [list(s) for s in states],
        "state_has_B": state_has_B,
        "state_is_correct": state_is_correct,
        "layer_ids": layer_ids,
    }, out_dir / "generation_hidden_states.pt")
    print(f"[INFO] Saved hidden states to {out_dir / 'generation_hidden_states.pt'}")

    # ── Probe each position ────────────────────────────────────────────────────

    # We'll store results for the comparison table
    # row: {position, layer, spearman, adj_acc, disk_accs}
    table_rows = []

    for l in layer_ids:
        print(f"\n{'='*60}")
        print(f"Layer {l}")
        print(f"{'='*60}")

        # ── Position A ────────────────────────────────────────────────────────
        if pos_A[l]:
            print(f"\n[Layer {l}] Position A ({len(pos_A[l])} vectors)")
            idxs_A = [i for i, _ in pos_A[l]]
            vecs_A = torch.stack([v for _, v in pos_A[l]]).float()
            states_A = [states[i] for i in idxs_A]

            # Distance submatrix
            sub_dist = true_dist[np.ix_(idxs_A, idxs_A)]
            sub_norm, _, _ = norm_dist(sub_dist)
            y_A = torch.tensor(sub_norm, dtype=torch.float32)

            sub_nbrs = {new_i: {idxs_A.index(j) for j in nbrs[idxs_A[new_i]] if j in idxs_A}
                        for new_i in range(len(idxs_A))}

            coords_A, _ = train_probe(vecs_A, y_A, args.epochs, args.lr, device)
            rho_A, adj_A = evaluate_probe(coords_A, sub_dist, sub_nbrs)
            print(f"  Spearman={rho_A:.4f}  AdjAcc={adj_A:.4f}")

            disk_acc_A = per_disk_accuracy(vecs_A.numpy(), states_A, n_disks)
            print(f"  Per-disk acc: {[f'{a:.3f}' for a in disk_acc_A]}")

            table_rows.append({"pos": "A", "layer": l, "spearman": rho_A,
                                "adj_acc": adj_A, "disk_accs": disk_acc_A,
                                "n": len(idxs_A)})

            if l == layer_ids[0]:
                plot_sierpinski(coords_A, states_A, [], f"Layer {l} | Pos A | ρ={rho_A:.3f}",
                                out_dir / f"L{l}_posA_sierpinski.png", color_by_disk=3)

        # ── Position B ────────────────────────────────────────────────────────
        if pos_B[l]:
            print(f"\n[Layer {l}] Position B ({len(pos_B[l])} vectors)")
            idxs_B = [i for i, _ in pos_B[l]]
            vecs_B = torch.stack([v for _, v in pos_B[l]]).float()
            states_B = [states[i] for i in idxs_B]

            sub_dist_B = true_dist[np.ix_(idxs_B, idxs_B)]
            sub_norm_B, _, _ = norm_dist(sub_dist_B)
            y_B = torch.tensor(sub_norm_B, dtype=torch.float32)

            sub_nbrs_B = {new_i: {idxs_B.index(j) for j in nbrs[idxs_B[new_i]] if j in idxs_B}
                          for new_i in range(len(idxs_B))}

            coords_B, _ = train_probe(vecs_B, y_B, args.epochs, args.lr, device)
            rho_B, adj_B = evaluate_probe(coords_B, sub_dist_B, sub_nbrs_B)
            print(f"  Spearman={rho_B:.4f}  AdjAcc={adj_B:.4f}")

            disk_acc_B = per_disk_accuracy(vecs_B.numpy(), states_B, n_disks)
            print(f"  Per-disk acc: {[f'{a:.3f}' for a in disk_acc_B]}")

            table_rows.append({"pos": "B", "layer": l, "spearman": rho_B,
                                "adj_acc": adj_B, "disk_accs": disk_acc_B,
                                "n": len(idxs_B)})

            if l == layer_ids[0]:
                # Standard Sierpinski
                plot_sierpinski(
                    coords_B, states_B, [],
                    f"Layer {l} | Pos B (before moves=[) | ρ={rho_B:.3f}",
                    out_dir / f"L{l}_posB_sierpinski.png", color_by_disk=3)

                # Correct vs failed coloring
                highlight = {new_i: ("green" if state_is_correct[idxs_B[new_i]] else "red")
                             for new_i in range(len(idxs_B))}
                plot_sierpinski(
                    coords_B, states_B, [],
                    f"Layer {l} | Pos B | green=correct  red=failed",
                    out_dir / f"L{l}_posB_correct_vs_failed.png",
                    highlight=highlight)

                # Correct/failed split probe metrics
                corr_mask = [state_is_correct[idxs_B[i]] for i in range(len(idxs_B))]
                fail_mask = [not state_is_correct[idxs_B[i]] for i in range(len(idxs_B))]
                corr_idx = [i for i, c in enumerate(corr_mask) if c]
                fail_idx = [i for i, c in enumerate(fail_mask) if c]
                print(f"\n  Position B: correct/failed split")
                idxs_B_set = set(idxs_B)
                idxs_B_pos = {v: i for i, v in enumerate(idxs_B)}
                for group_name, gidx in [("Correct", corr_idx), ("Failed", fail_idx)]:
                    if len(gidx) < 4:
                        print(f"  {group_name}: too few samples ({len(gidx)})")
                        continue
                    g_sub = sub_dist_B[np.ix_(gidx, gidx)]
                    gidx_set = set(gidx)
                    gidx_pos = {v: i for i, v in enumerate(gidx)}
                    g_nbrs = {ni: {gidx_pos[idxs_B_pos[j]]
                                   for j in nbrs[idxs_B[gidx[ni]]]
                                   if j in idxs_B_set and idxs_B_pos[j] in gidx_set}
                              for ni in range(len(gidx))}
                    g_vecs = vecs_B[gidx]
                    g_states = [states_B[i] for i in gidx]
                    g_norm, _, _ = norm_dist(g_sub)
                    g_coords, _ = train_probe(g_vecs, torch.tensor(g_norm), args.epochs, args.lr, device)
                    g_rho, g_adj = evaluate_probe(g_coords, g_sub, g_nbrs)
                    g_disk_acc = per_disk_accuracy(g_vecs.numpy(), g_states, n_disks)
                    print(f"  {group_name:8s} n={len(gidx):2d}  ρ={g_rho:.4f}  AdjAcc={g_adj:.4f}"
                          f"  disk={[f'{a:.3f}' for a in g_disk_acc]}")

        # ── Position C (averaged per true state) ──────────────────────────────
        if pos_C[l]:
            print(f"\n[Layer {l}] Position C averaged ({len(pos_C[l])} raw vectors)")
            # Group by true state
            state_vecs: Dict[State, List[torch.Tensor]] = defaultdict(list)
            for true_st, vec in pos_C[l]:
                state_vecs[true_st].append(vec)

            c_states = sorted(state_vecs.keys(), key=lambda s: state_to_idx.get(s, 9999))
            c_idxs = [state_to_idx[s] for s in c_states if s in state_to_idx]
            c_states_valid = [s for s in c_states if s in state_to_idx]

            if len(c_states_valid) < 4:
                print("  Too few unique states for C-averaged probe")
            else:
                c_vecs_avg = torch.stack([
                    torch.stack(state_vecs[s]).mean(0) for s in c_states_valid
                ]).float()

                c_dist = true_dist[np.ix_(c_idxs, c_idxs)]
                c_norm, _, _ = norm_dist(c_dist)
                c_nbrs = {ni: {c_idxs.index(j) for j in nbrs[c_idxs[ni]] if j in c_idxs}
                          for ni in range(len(c_idxs))}

                c_coords, _ = train_probe(c_vecs_avg, torch.tensor(c_norm), args.epochs, args.lr, device)
                rho_C, adj_C = evaluate_probe(c_coords, c_dist, c_nbrs)
                print(f"  Spearman={rho_C:.4f}  AdjAcc={adj_C:.4f}  n_states={len(c_states_valid)}")

                table_rows.append({"pos": "C_avg", "layer": l, "spearman": rho_C,
                                   "adj_acc": adj_C, "disk_accs": [float("nan")] * n_disks,
                                   "n": len(c_states_valid)})

                if l == layer_ids[0]:
                    plot_sierpinski(c_coords, c_states_valid, [],
                                    f"Layer {l} | Pos C averaged ({len(c_states_valid)} states) | ρ={rho_C:.3f}",
                                    out_dir / f"L{l}_posC_avg_sierpinski.png", color_by_disk=3)

            # Per-disk logistic regression on individual C vectors
            all_c_vecs = np.stack([v.numpy() for _, v in pos_C[l]])
            all_c_states = [s for s, _ in pos_C[l]]
            disk_acc_C = per_disk_accuracy(all_c_vecs, all_c_states, n_disks)
            print(f"  C per-disk acc (N={len(all_c_vecs)}): {[f'{a:.3f}' for a in disk_acc_C]}")

            table_rows.append({"pos": "C_perdisk", "layer": l, "spearman": float("nan"),
                               "adj_acc": float("nan"), "disk_accs": disk_acc_C,
                               "n": len(all_c_vecs)})

    # ── Side-by-side Sierpinski (A, B, C-avg for best layer) ──────────────────
    best_layer = layer_ids[1] if len(layer_ids) > 1 else layer_ids[0]  # default to middle
    # pick layer with best Spearman at pos A
    a_rows = [r for r in table_rows if r["pos"] == "A"]
    if a_rows:
        best_layer = max(a_rows, key=lambda r: r["spearman"])["layer"]

    multi_coords: List[np.ndarray] = []
    multi_titles: List[str] = []
    multi_states: List[List[State]] = []

    if pos_A[best_layer]:
        idxs = [i for i, _ in pos_A[best_layer]]
        vecs = torch.stack([v for _, v in pos_A[best_layer]]).float()
        sub = true_dist[np.ix_(idxs, idxs)]
        sn, _, _ = norm_dist(sub)
        c, _ = train_probe(vecs, torch.tensor(sn), args.epochs, args.lr, device)
        multi_coords.append(c)
        multi_titles.append("Pos A (last prompt tok)")
        multi_states.append([states[i] for i in idxs])

    if pos_B[best_layer]:
        idxs = [i for i, _ in pos_B[best_layer]]
        vecs = torch.stack([v for _, v in pos_B[best_layer]]).float()
        sub = true_dist[np.ix_(idxs, idxs)]
        sn, _, _ = norm_dist(sub)
        c, _ = train_probe(vecs, torch.tensor(sn), args.epochs, args.lr, device)
        multi_coords.append(c)
        multi_titles.append("Pos B (before moves=[)")
        multi_states.append([states[i] for i in idxs])

    c_avg_rows = [r for r in table_rows if r["pos"] == "C_avg" and r["layer"] == best_layer]
    if c_avg_rows and pos_C[best_layer]:
        sv: Dict[State, List[torch.Tensor]] = defaultdict(list)
        for ts, vec in pos_C[best_layer]:
            sv[ts].append(vec)
        cv = sorted([s for s in sv if s in state_to_idx], key=lambda s: state_to_idx[s])
        if cv:
            ci = [state_to_idx[s] for s in cv]
            avg_vecs = torch.stack([torch.stack(sv[s]).mean(0) for s in cv]).float()
            sub = true_dist[np.ix_(ci, ci)]
            sn, _, _ = norm_dist(sub)
            c, _ = train_probe(avg_vecs, torch.tensor(sn), args.epochs, args.lr, device)
            multi_coords.append(c)
            multi_titles.append(f"Pos C avg ({len(cv)} states)")
            multi_states.append(cv)

    if len(multi_coords) >= 2:
        plot_side_by_side(multi_coords, multi_titles, multi_states,
                          out_dir / f"L{best_layer}_positions_comparison.png")

    # ── Print comparison table ─────────────────────────────────────────────────
    print(f"\n{'='*90}")
    print("COMPARISON TABLE")
    print(f"{'='*90}")
    hdr = (f"{'Position':<20} {'Layer':>5} {'N':>4} {'Spearman':>10} "
           f"{'AdjAcc':>8} {'Disk0':>7} {'Disk1':>7} {'Disk2':>7} {'Disk3':>7}")
    print(hdr)
    print("-" * 90)
    for r in table_rows:
        da = r["disk_accs"]
        def fmt(x):
            return f"{x:.3f}" if not math.isnan(x) else "  -  "
        spearman_str = f"{r['spearman']:.4f}" if not math.isnan(r["spearman"]) else "   -   "
        adj_str = f"{r['adj_acc']:.4f}" if not math.isnan(r["adj_acc"]) else "  -  "
        print(f"{r['pos']:<20} {r['layer']:>5} {r['n']:>4} {spearman_str:>10} "
              f"{adj_str:>8} {fmt(da[0]):>7} {fmt(da[1]):>7} {fmt(da[2]):>7} {fmt(da[3]):>7}")

    # Save table as JSON
    summary = {
        "model_name": args.model_name,
        "layers": layer_ids,
        "n_disks": n_disks,
        "table_rows": [
            {k: (v if not isinstance(v, float) or not math.isnan(v) else None)
             for k, v in r.items()}
            for r in table_rows
        ],
        "runtime_seconds": time.time() - t0,
    }
    (out_dir / "probe_generation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[INFO] Results saved to {out_dir}")
    print(f"[INFO] Total runtime: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
