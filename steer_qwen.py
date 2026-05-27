#!/usr/bin/env python3
"""Activation steering for Qwen3-27B on flat-to-flat Tower of Hanoi.

Idea: the model encodes the state space cleanly at position A (last prompt
token) with Spearman ~0.935 between probe-distance and graph-distance, but
that signal degrades during long CoT. Use the position-A activations as
clean steering targets and inject them into the residual stream during
generation, updating the target each time the model commits a legal move.

Pipeline:
  1. Load layer-L position-A hidden states from outputs/qwen_probe/hidden_states.pt
     → dict state_tuple → activation vector, plus the mean over all 81 states.
  2. Install a forward hook on layer-L that modifies the LAST-token residual
     at every forward pass (prefill + each generation step):
        directional: h[-1] += alpha * unit(target - mean)
        blend:       h[-1] = (1 - beta) * h[-1] + beta * target
  3. Generate token-by-token with KV cache. When a ']' lands in the freshly
     decoded text, re-parse the LAST moves=[...] block from scratch, replay
     against the start state, and if the resulting board changes update the
     hook's target activation.
  4. Classify the final answer (CORRECT_OPTIMAL / CORRECT_SUBOPTIMAL /
     ILLEGAL_* / PARSE_ERROR / WRONG_GOAL_STATE).
  5. Sweep alpha, beta, layer choices and print a comparison table.

Optional oracle baseline (--run_oracle) generates ONE move at a time with
the current state explicitly written into the prompt — an upper bound on
what state-feedback alone could buy.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict, deque
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from prompts import create_nonstandard_prompt

State = Tuple[int, ...]

# Categories — mirror evaluate_model.py
CORRECT_OPTIMAL = "CORRECT_OPTIMAL"
CORRECT_SUBOPTIMAL = "CORRECT_SUBOPTIMAL"
PARSE_ERROR = "PARSE_ERROR"
ILLEGAL_EMPTY_SOURCE = "ILLEGAL_EMPTY_SOURCE"
ILLEGAL_LARGER_ON_SMALLER = "ILLEGAL_LARGER_ON_SMALLER"
ILLEGAL_WRONG_DISK = "ILLEGAL_WRONG_DISK"
ILLEGAL_BAD_FORMAT = "ILLEGAL_BAD_FORMAT"
WRONG_GOAL_STATE = "WRONG_GOAL_STATE"
PREMATURE_STOP = "PREMATURE_STOP"
EXCESSIVE_MOVES = "EXCESSIVE_MOVES"
ILLEGAL_CATEGORIES = {ILLEGAL_EMPTY_SOURCE, ILLEGAL_LARGER_ON_SMALLER,
                      ILLEGAL_WRONG_DISK, ILLEGAL_BAD_FORMAT}
CORRECT_CATEGORIES = {CORRECT_OPTIMAL, CORRECT_SUBOPTIMAL}


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Activation steering for Qwen ToH solving")
    p.add_argument("--model_name", default="Qwen/Qwen3-27B")
    p.add_argument("--hidden_states", default="outputs/qwen_probe/hidden_states.pt",
                   help="Position-A hidden states from probe.py")
    p.add_argument("--n_disks", type=int, default=4)
    p.add_argument("--layer", type=int, default=36,
                   help="Primary steering layer (1-indexed)")
    p.add_argument("--extra_layers", default="",
                   help="Comma-separated extra layers for joint steering, e.g. '48'")
    p.add_argument("--alphas", default="0.5,1,2,5,10,20")
    p.add_argument("--betas", default="0.01,0.05,0.1,0.2,0.5")
    p.add_argument("--modes", default="directional,blend",
                   help="Comma-separated subset of {directional, blend}")
    p.add_argument("--run_baseline", action="store_true",
                   help="Run a no-steering pass for comparison")
    p.add_argument("--run_oracle", action="store_true",
                   help="Run the oracle (state-in-prompt) baseline")
    p.add_argument("--output_dir", default="outputs/qwen_steering")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default="bfloat16",
                   choices=["float16", "bfloat16", "float32"])
    p.add_argument("--max_new_tokens", type=int, default=13000)
    p.add_argument("--max_oracle_steps", type=int, default=30,
                   help="Per-problem move cap for the oracle baseline")
    p.add_argument("--oracle_max_new_tokens", type=int, default=2048,
                   help="Token budget per oracle step")
    p.add_argument("--n_problems", type=int, default=81)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--probe_dir", default="outputs/qwen_probe",
                   help="Used by the Sierpinski plot of flipped problems")
    p.add_argument("--decode_every", type=int, default=1,
                   help="Decode/parse-state cadence in tokens. 1 == every step.")
    p.add_argument("--save_texts", action="store_true",
                   help="Save full generated text per (condition, problem) — large")
    p.add_argument("--eval_results",
                   default="outputs/qwen_probe/evaluate_qwen_results.json",
                   help="Path to evaluate_*_results.json used by --failed_only")
    p.add_argument("--failed_only", action="store_true",
                   help="Skip the states the model already solves in --eval_results")
    p.add_argument("--always_steer", action="store_true",
                   help="Steer during CoT too (legacy). Default: hook is dormant "
                        "until the model emits 'moves = [' in its output.")
    return p.parse_args()


def dtype_from_str(s: str) -> torch.dtype:
    return {"float16": torch.float16, "bfloat16": torch.bfloat16,
            "float32": torch.float32}[s]


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
    for i, s in enumerate(states):
        for nb in legal_neighbors(s, n_disks):
            j = idx[nb]
            adj[i].append(j)
            edges.add((min(i, j), max(i, j)))
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
    return dist, sorted(edges)


def optimal_length(state: State, n_disks: int) -> int:
    start = tuple(int(x) for x in state)
    goal: State = tuple([2] * n_disks)
    if start == goal:
        return 0
    seen = {start}
    q: deque = deque([(start, 0)])
    while q:
        s, d = q.popleft()
        for nb in legal_neighbors(s, n_disks):
            if nb in seen:
                continue
            if nb == goal:
                return d + 1
            seen.add(nb)
            q.append((nb, d + 1))
    return -1


# ── Move parsing & simulation ──────────────────────────────────────────────────

MOVES_BLOCK_RE = re.compile(r'moves\s*=\s*\[', re.IGNORECASE)
MOVE_RE = re.compile(r'\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]')


def _illegal_category(disk: int, src: int, dst: int, n_disks: int,
                      pegs: List[List[int]]) -> str:
    if not (1 <= disk <= n_disks and 0 <= src <= 2 and 0 <= dst <= 2 and src != dst):
        return ILLEGAL_BAD_FORMAT
    if not pegs[src]:
        return ILLEGAL_EMPTY_SOURCE
    if pegs[src][-1] != disk:
        return ILLEGAL_WRONG_DISK
    if pegs[dst] and pegs[dst][-1] < disk:
        return ILLEGAL_LARGER_ON_SMALLER
    return ILLEGAL_BAD_FORMAT


def find_last_moves_block(text: str) -> Optional[Tuple[int, str]]:
    """Return (bracket_start_offset, block_text) for the last complete 'moves=[...]'."""
    best: Optional[Tuple[int, str]] = None
    for m in MOVES_BLOCK_RE.finditer(text):
        bracket = text.find('[', m.start())
        if bracket < 0:
            continue
        depth = 0
        for idx in range(bracket, len(text)):
            c = text[idx]
            if c == '[':
                depth += 1
            elif c == ']':
                depth -= 1
                if depth == 0:
                    cand = text[bracket:idx + 1]
                    if cand.strip() != '[]':
                        best = (bracket, cand)
                    break
    return best


def replay_block(block_text: str, start_pegs: List[List[int]], n_disks: int
                 ) -> Tuple[List[List[int]], int, Optional[str], int]:
    """Replay moves in `block_text` from `start_pegs`. Returns
    (final_pegs, n_legal_prefix, first_illegal_category_or_None, n_moves_total).
    Stops at the first illegal move (mirrors evaluate_model.simulate_moves).
    """
    pegs = [list(p) for p in start_pegs]
    n_legal = 0
    n_total = 0
    illegal_cat: Optional[str] = None
    for m in MOVE_RE.finditer(block_text):
        n_total += 1
        disk, src, dst = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if (1 <= disk <= n_disks and 0 <= src <= 2 and 0 <= dst <= 2 and src != dst
                and pegs[src] and pegs[src][-1] == disk
                and (not pegs[dst] or pegs[dst][-1] > disk)):
            pegs[dst].append(pegs[src].pop())
            n_legal += 1
        else:
            illegal_cat = _illegal_category(disk, src, dst, n_disks, pegs)
            break
    return pegs, n_legal, illegal_cat, n_total


def classify(gen_text: str, start_pegs: List[List[int]], n_disks: int,
             opt_len: int, goal_pegs: List[List[int]]) -> Dict[str, object]:
    """Categorize a generated text following evaluate_model.simulate_moves."""
    block = find_last_moves_block(gen_text)
    if block is None:
        return {
            "category": PARSE_ERROR,
            "n_moves": 0,
            "n_legal": 0,
            "first_illegal_at": None,
            "final_state": list(pegs_to_state(start_pegs, n_disks)),
            "reached_goal": False,
            "opt_len": opt_len,
        }
    _, block_text = block
    pegs, n_legal, illegal_cat, n_total = replay_block(block_text, start_pegs, n_disks)
    reached = pegs == goal_pegs
    if illegal_cat is not None:
        category = illegal_cat
    elif reached:
        category = CORRECT_OPTIMAL if n_total == opt_len else CORRECT_SUBOPTIMAL
    elif n_total < opt_len:
        category = PREMATURE_STOP
    elif n_total > opt_len:
        category = EXCESSIVE_MOVES
    else:
        category = WRONG_GOAL_STATE
    return {
        "category": category,
        "n_moves": n_total,
        "n_legal": n_legal,
        "first_illegal_at": n_legal + 1 if illegal_cat is not None else None,
        "final_state": list(pegs_to_state(pegs, n_disks)),
        "reached_goal": reached,
        "opt_len": opt_len,
    }


# ── Activation lookup ──────────────────────────────────────────────────────────

def load_state_activations(path: Path, layer: int
                           ) -> Tuple[Dict[State, torch.Tensor], torch.Tensor]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    states_raw = data["states"]
    states = [tuple(int(x) for x in s) for s in states_raw]
    hs_by_layer = data["hidden_states"]
    if layer not in hs_by_layer:
        raise KeyError(f"Layer {layer} not in hidden_states; "
                       f"available={sorted(hs_by_layer.keys())}")
    hs = hs_by_layer[layer].float()  # (n_states, d_model)
    state_to_vec = {s: hs[i].clone() for i, s in enumerate(states)}
    mean_vec = hs.mean(dim=0)
    return state_to_vec, mean_vec


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
        text_model = raw.model.language_model
        cfg = getattr(raw.config, "text_config", raw.config)
    else:
        text_model = raw
        cfg = raw.config

    device = torch.device(device_str)
    if device_str != "cuda":
        text_model.to(device)
    raw.eval()
    return raw, text_model, tokenizer, cfg, device


def get_layers_list(text_model: nn.Module):
    if hasattr(text_model, "model") and hasattr(text_model.model, "layers"):
        return text_model.model.layers
    if hasattr(text_model, "layers"):
        return text_model.layers
    raise RuntimeError("Cannot find transformer layers")


# ── Steering hook ──────────────────────────────────────────────────────────────

class SteeringHook:
    """Forward hook on one transformer layer; rewrites residual at the LAST tok.

    mode='directional':  h[-1] += scale * unit(target - mean)
    mode='blend':        h[-1]  = (1 - scale) * h[-1] + scale * target

    The hook is REGISTERED throughout but stays *dormant* (no-op) until
    `moves_phase_active` is flipped on. The generation loop turns it on the
    moment the model emits the final-answer header `moves = [`. This keeps
    the model's CoT untouched and steers only the move-output tokens, cutting
    steered forward passes from ~12k/problem to ~50–100.
    """

    def __init__(self, layer_module: nn.Module, layer_id: int, mode: str, scale: float,
                 moves_phase_active: bool = False):
        assert mode in ("directional", "blend"), mode
        self.layer_id = layer_id
        self.mode = mode
        self.scale = float(scale)
        self.target_vec: Optional[torch.Tensor] = None
        self.mean_vec: Optional[torch.Tensor] = None
        self.steering_unit: Optional[torch.Tensor] = None
        self.enabled = True
        self.moves_phase_active = moves_phase_active
        self.fired = 0
        self.handle = layer_module.register_forward_hook(self._hook)

    def update_target(self, target_vec: torch.Tensor, mean_vec: torch.Tensor) -> None:
        self.target_vec = target_vec
        self.mean_vec = mean_vec
        if self.mode == "directional":
            diff = target_vec - mean_vec
            n = float(diff.norm())
            self.steering_unit = diff / n if n > 1e-8 else diff.clone()

    def activate_moves_phase(self) -> None:
        self.moves_phase_active = True

    def reset_moves_phase(self, active: bool = False) -> None:
        self.moves_phase_active = active
        self.fired = 0

    def _hook(self, module, inputs, output):
        if (not self.enabled
                or not self.moves_phase_active
                or self.target_vec is None):
            return None
        h = output[0] if isinstance(output, tuple) else output
        device, dtype = h.device, h.dtype
        if self.mode == "directional":
            sv = self.steering_unit.to(device=device, dtype=dtype)
            h[:, -1:, :] = h[:, -1:, :] + self.scale * sv
        else:
            tv = self.target_vec.to(device=device, dtype=dtype)
            h[:, -1:, :] = (1.0 - self.scale) * h[:, -1:, :] + self.scale * tv
        self.fired += 1
        if isinstance(output, tuple):
            return (h,) + output[1:]
        return h

    def remove(self) -> None:
        self.handle.remove()


# ── Prompt builders ────────────────────────────────────────────────────────────

def build_prompt(tokenizer, state: State, idx: int, n_disks: int) -> str:
    goal_pegs = [[], [], list(range(n_disks, 0, -1))]
    pegs = state_tuple_to_pegs(state, n_disks)
    sysp, userp, _ = create_nonstandard_prompt(
        num_disks=n_disks, problem_id=idx, seed=0,
        initial_state_override=pegs, goal_state_override=goal_pegs,
    )
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": sysp},
         {"role": "user", "content": userp}],
        tokenize=False, add_generation_prompt=True,
    )


def build_oracle_step_prompt(tokenizer, current_pegs: List[List[int]],
                             goal_pegs: List[List[int]], n_disks: int,
                             moves_so_far: List[List[int]]) -> str:
    sysp, userp, _ = create_nonstandard_prompt(
        num_disks=n_disks, problem_id=0, seed=0,
        initial_state_override=current_pegs, goal_state_override=goal_pegs,
    )
    moves_str = ", ".join(f"[{m[0]}, {m[1]}, {m[2]}]" for m in moves_so_far) if moves_so_far else "(none)"
    addendum = (
        f"\n\nMoves already executed: {moves_str}.\n"
        f"Current peg configuration: peg 0={current_pegs[0]}, "
        f"peg 1={current_pegs[1]}, peg 2={current_pegs[2]}.\n"
        "Output ONLY the single next move toward the goal in the exact "
        "format `moves = [[disk, from, to]]`. Do not output any other moves."
    )
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": sysp},
         {"role": "user", "content": userp + addendum}],
        tokenize=False, add_generation_prompt=True,
    )


# ── Generation ─────────────────────────────────────────────────────────────────

def _stop_token_ids(tokenizer) -> Set[int]:
    ids: Set[int] = set()
    if tokenizer.eos_token_id is not None:
        ids.add(int(tokenizer.eos_token_id))
    for tok in ("<|im_end|>", "<|endoftext|>"):
        tid = tokenizer.convert_tokens_to_ids(tok)
        if isinstance(tid, int) and tid >= 0 and tid != tokenizer.unk_token_id:
            ids.add(tid)
    return ids


_MOVES_HEADER_RE = re.compile(r'moves\s*=\s*\[', re.IGNORECASE)


def generate_with_steering(
    raw_model: nn.Module,
    tokenizer,
    prompt_text: str,
    start_state: State,
    n_disks: int,
    layer_state_to_vec: Dict[int, Dict[State, torch.Tensor]],
    layer_mean_vec: Dict[int, torch.Tensor],
    hooks: Sequence[SteeringHook],
    max_new_tokens: int,
    device: torch.device,
    eos_ids: Set[int],
    decode_every: int = 1,
    always_steer: bool = False,
) -> Tuple[str, State, int, int]:
    """Run autoregressive greedy generation with hooks dormant during CoT.

    The hook is REGISTERED before this call but stays no-op until either:
      * `always_steer=True` (legacy behavior — active from the first token), or
      * the model emits the final-answer header `moves = [` in its output, at
        which point all hooks are flipped active for the rest of generation.

    Returns (generated_text, current_simulated_state, n_legal_moves_seen,
             moves_phase_step). `moves_phase_step` is the generation step at
    which the header was detected (-1 if never).
    """
    enc = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"].to(device)

    start_pegs = state_tuple_to_pegs(start_state, n_disks)
    current_state = tuple(int(x) for x in start_state)

    def push_targets(state: State) -> None:
        for h in hooks:
            stv = layer_state_to_vec.get(h.layer_id)
            mv = layer_mean_vec.get(h.layer_id)
            if stv is None or mv is None or state not in stv:
                continue
            h.update_target(stv[state], mv)

    if hooks:
        push_targets(current_state)
        # Each new problem starts with a clean dormant hook (or active if the
        # user asked for legacy "steer during CoT too").
        for h in hooks:
            h.reset_moves_phase(active=always_steer)

    moves_phase = always_steer
    moves_phase_step = -1

    generated_ids: List[int] = []
    cur_input = input_ids
    past = None
    n_legal_total = 0

    with torch.no_grad():
        for step in range(max_new_tokens):
            out = raw_model(input_ids=cur_input, past_key_values=past, use_cache=True)
            past = out.past_key_values
            next_logits = out.logits[:, -1, :]
            next_id = int(next_logits.argmax(dim=-1).item())
            generated_ids.append(next_id)
            if next_id in eos_ids:
                break
            cur_input = torch.tensor([[next_id]], device=device)

            # ── Detect transition into the moves-output phase ────────────────
            # Run only while hooks are still dormant; stop checking once flipped.
            # A 32-token tail comfortably contains the ~8-char header.
            if hooks and not moves_phase:
                tail = tokenizer.decode(
                    generated_ids[-min(32, len(generated_ids)):],
                    skip_special_tokens=False,
                )
                if _MOVES_HEADER_RE.search(tail):
                    moves_phase = True
                    moves_phase_step = step
                    for h in hooks:
                        h.activate_moves_phase()

            # ── State tracking via newly-appeared ']' ────────────────────────
            # Runs throughout, so by the time the hook activates its target is
            # already in sync with whatever the simulator says the board is.
            if decode_every <= 1 or (step % decode_every == 0):
                tail = tokenizer.decode(
                    generated_ids[-min(8, len(generated_ids)):],
                    skip_special_tokens=False,
                )
                if ']' not in tail:
                    continue
                full_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
                block = find_last_moves_block(full_text)
                if block is None:
                    continue
                pegs, n_legal, _, _ = replay_block(block[1], start_pegs, n_disks)
                new_state = pegs_to_state(pegs, n_disks)
                if new_state != current_state:
                    current_state = new_state
                    n_legal_total = n_legal
                    if hooks:
                        push_targets(current_state)

    gen_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
    return gen_text, current_state, n_legal_total, moves_phase_step


def generate_plain(
    raw_model: nn.Module,
    tokenizer,
    prompt_text: str,
    max_new_tokens: int,
    device: torch.device,
    eos_ids: Set[int],
) -> str:
    """Vanilla greedy generation (no hooks)."""
    enc = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"].to(device)
    generated_ids: List[int] = []
    cur_input = input_ids
    past = None
    with torch.no_grad():
        for _ in range(max_new_tokens):
            out = raw_model(input_ids=cur_input, past_key_values=past, use_cache=True)
            past = out.past_key_values
            next_id = int(out.logits[:, -1, :].argmax(dim=-1).item())
            generated_ids.append(next_id)
            if next_id in eos_ids:
                break
            cur_input = torch.tensor([[next_id]], device=device)
    return tokenizer.decode(generated_ids, skip_special_tokens=False)


def run_oracle_for_state(
    raw_model: nn.Module,
    tokenizer,
    start_state: State,
    n_disks: int,
    max_oracle_steps: int,
    max_new_tokens: int,
    device: torch.device,
    eos_ids: Set[int],
) -> Tuple[str, List[List[int]], List[List[int]], str]:
    """Iteratively ask the model for the next single move with the current state
    in the prompt. Returns (concat_log, final_pegs, moves_executed, category).
    """
    goal_pegs = [[], [], list(range(n_disks, 0, -1))]
    pegs = state_tuple_to_pegs(start_state, n_disks)
    moves_executed: List[List[int]] = []
    log_chunks: List[str] = []
    category = WRONG_GOAL_STATE

    for step in range(max_oracle_steps):
        if pegs == goal_pegs:
            category = (CORRECT_OPTIMAL
                        if len(moves_executed) == optimal_length(start_state, n_disks)
                        else CORRECT_SUBOPTIMAL)
            break
        prompt = build_oracle_step_prompt(tokenizer, pegs, goal_pegs, n_disks, moves_executed)
        out_text = generate_plain(raw_model, tokenizer, prompt, max_new_tokens, device, eos_ids)
        log_chunks.append(f"=== step {step} ===\n{out_text}\n")

        block = find_last_moves_block(out_text)
        if block is None:
            category = PARSE_ERROR
            break
        moves_in_block: List[Tuple[int, int, int]] = [
            (int(m.group(1)), int(m.group(2)), int(m.group(3)))
            for m in MOVE_RE.finditer(block[1])
        ]
        if not moves_in_block:
            category = PARSE_ERROR
            break
        disk, src, dst = moves_in_block[0]
        if (1 <= disk <= n_disks and 0 <= src <= 2 and 0 <= dst <= 2 and src != dst
                and pegs[src] and pegs[src][-1] == disk
                and (not pegs[dst] or pegs[dst][-1] > disk)):
            pegs[dst].append(pegs[src].pop())
            moves_executed.append([disk, src, dst])
        else:
            category = _illegal_category(disk, src, dst, n_disks, pegs)
            break
    else:
        category = EXCESSIVE_MOVES

    if pegs == goal_pegs and category not in CORRECT_CATEGORIES:
        category = (CORRECT_OPTIMAL
                    if len(moves_executed) == optimal_length(start_state, n_disks)
                    else CORRECT_SUBOPTIMAL)
    return "\n".join(log_chunks), pegs, moves_executed, category


# ── Sweep runner ───────────────────────────────────────────────────────────────

def run_condition(
    label: str,
    states: List[State],
    raw_model: nn.Module,
    text_model: nn.Module,
    tokenizer,
    n_disks: int,
    opt_lens: Dict[State, int],
    goal_pegs: List[List[int]],
    layer_ids: List[int],
    layer_state_to_vec: Dict[int, Dict[State, torch.Tensor]],
    layer_mean_vec: Dict[int, torch.Tensor],
    mode: Optional[str],
    scale: Optional[float],
    max_new_tokens: int,
    device: torch.device,
    eos_ids: Set[int],
    decode_every: int,
    save_texts: bool,
    always_steer: bool = False,
) -> Dict[str, object]:
    """Run a single sweep cell. mode=None → no steering."""
    hooks: List[SteeringHook] = []
    if mode is not None and scale is not None:
        layers_list = get_layers_list(text_model)
        for lid in layer_ids:
            hooks.append(SteeringHook(
                layers_list[lid - 1], lid, mode, float(scale),
                moves_phase_active=always_steer,
            ))

    per_problem: List[Dict[str, object]] = []
    cat_counter: Counter = Counter()
    t0 = time.time()
    try:
        for prob_idx, state in enumerate(states):
            start_pegs = state_tuple_to_pegs(state, n_disks)
            prompt = build_prompt(tokenizer, state, prob_idx, n_disks)
            try:
                gen_text, sim_state, n_legal, moves_step = generate_with_steering(
                    raw_model, tokenizer, prompt, state, n_disks,
                    layer_state_to_vec, layer_mean_vec, hooks,
                    max_new_tokens, device, eos_ids, decode_every,
                    always_steer=always_steer,
                )
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"  [OOM on prob {prob_idx} state={state}; recovering]")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gen_text = ""
                    sim_state = state
                    n_legal = 0
                    moves_step = -1
                else:
                    raise

            cls = classify(gen_text, start_pegs, n_disks,
                           opt_lens[state], goal_pegs)
            cat_counter[cls["category"]] += 1
            hook_fired = hooks[0].fired if hooks else 0
            rec = {
                "prob_idx": prob_idx,
                "state": list(state),
                **cls,
                "sim_state": list(sim_state),
                "n_legal_via_steering": n_legal,
                "moves_phase_step": moves_step,
                "hook_fired_steps": hook_fired,
            }
            if save_texts:
                rec["generated_text"] = gen_text
            per_problem.append(rec)

            print(f"    [{label}] {prob_idx+1:3d}/{len(states)}  state={state}  "
                  f"cat={cls['category']:24s}  n_moves={cls['n_moves']:2d}  "
                  f"opt={cls['opt_len']:2d}  legal_pref={cls['n_legal']}  "
                  f"moves_at={moves_step}  hook_fired={hook_fired}  "
                  f"sim={sim_state}")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        for h in hooks:
            h.remove()

    return {
        "label": label,
        "mode": mode,
        "scale": scale,
        "layer_ids": layer_ids,
        "category_counts": dict(cat_counter),
        "n_solved": sum(cat_counter[c] for c in CORRECT_CATEGORIES),
        "n_optimal": cat_counter.get(CORRECT_OPTIMAL, 0),
        "n_illegal": sum(cat_counter[c] for c in ILLEGAL_CATEGORIES),
        "n_parse_error": cat_counter.get(PARSE_ERROR, 0),
        "runtime_seconds": time.time() - t0,
        "per_problem": per_problem,
    }


def run_oracle_condition(
    states: List[State],
    raw_model: nn.Module,
    tokenizer,
    n_disks: int,
    opt_lens: Dict[State, int],
    args: argparse.Namespace,
    device: torch.device,
    eos_ids: Set[int],
) -> Dict[str, object]:
    cat_counter: Counter = Counter()
    per_problem: List[Dict[str, object]] = []
    t0 = time.time()
    for prob_idx, state in enumerate(states):
        try:
            log, final_pegs, moves, cat = run_oracle_for_state(
                raw_model, tokenizer, state, n_disks,
                args.max_oracle_steps, args.oracle_max_new_tokens,
                device, eos_ids,
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  [oracle OOM at prob {prob_idx} state={state}]")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            raise
        cat_counter[cat] += 1
        rec = {
            "prob_idx": prob_idx,
            "state": list(state),
            "category": cat,
            "n_moves": len(moves),
            "opt_len": opt_lens[state],
            "final_pegs": final_pegs,
        }
        if args.save_texts:
            rec["log"] = log
        per_problem.append(rec)
        print(f"    [oracle] {prob_idx+1:3d}/{len(states)}  state={state}  cat={cat:24s}  "
              f"moves={len(moves):2d}  opt={opt_lens[state]:2d}")
    return {
        "label": "oracle",
        "mode": "oracle",
        "scale": None,
        "layer_ids": [],
        "category_counts": dict(cat_counter),
        "n_solved": sum(cat_counter[c] for c in CORRECT_CATEGORIES),
        "n_optimal": cat_counter.get(CORRECT_OPTIMAL, 0),
        "n_illegal": sum(cat_counter[c] for c in ILLEGAL_CATEGORIES),
        "n_parse_error": cat_counter.get(PARSE_ERROR, 0),
        "runtime_seconds": time.time() - t0,
        "per_problem": per_problem,
    }


# ── Reporting & visualization ──────────────────────────────────────────────────

def print_comparison_table(results: List[Dict[str, object]], n_states: int) -> None:
    print("\n" + "=" * 96)
    print("COMPARISON TABLE")
    print("=" * 96)
    hdr = (f"{'Condition':<28} {'Layers':>10} {'α/β':>8} "
           f"{'Solved':>8} {'Optimal':>8} {'Illegal':>8} {'Parse':>7} {'Wrong':>7}")
    print(hdr)
    print("-" * 96)
    for r in results:
        scale_str = f"{r['scale']:.3f}" if isinstance(r["scale"], (int, float)) else "  -"
        layers_str = ",".join(str(l) for l in r["layer_ids"]) if r["layer_ids"] else "-"
        cc = r["category_counts"]
        wrong = (cc.get(WRONG_GOAL_STATE, 0) + cc.get(PREMATURE_STOP, 0)
                 + cc.get(EXCESSIVE_MOVES, 0))
        print(f"{r['label']:<28} {layers_str:>10} {scale_str:>8} "
              f"{r['n_solved']:>4d}/{n_states:<3d}  "
              f"{r['n_optimal']:>4d}/{n_states:<3d} "
              f"{r['n_illegal']:>4d}/{n_states:<3d} "
              f"{r['n_parse_error']:>3d}/{n_states:<3d} "
              f"{wrong:>3d}/{n_states:<3d}")


def plot_flipped(
    baseline: Optional[Dict[str, object]],
    steered: Dict[str, object],
    states: List[State],
    edges: List[Tuple[int, int]],
    coords: Optional[np.ndarray],
    out_path: Path,
    title: str,
    known_solved: Optional[Set[State]] = None,
) -> None:
    """`known_solved` are states pre-marked correct (e.g. from --failed_only)
    so they show up as 'stable solved' even though we never re-ran them.
    """
    if coords is None:
        return
    state_idx = {s: i for i, s in enumerate(states)}
    known_solved = known_solved or set()
    baseline_solved: Set[State] = set(known_solved)
    if baseline is not None:
        for r in baseline["per_problem"]:
            if r["category"] in CORRECT_CATEGORIES:
                baseline_solved.add(tuple(r["state"]))
    steered_solved: Set[State] = set(known_solved)
    for r in steered["per_problem"]:
        if r["category"] in CORRECT_CATEGORIES:
            steered_solved.add(tuple(r["state"]))

    gained = steered_solved - baseline_solved
    lost = baseline_solved - steered_solved
    stable = steered_solved & baseline_solved
    failed = set(states) - steered_solved - baseline_solved

    fig, ax = plt.subplots(figsize=(12, 10), constrained_layout=True)
    for i, j in edges:
        ax.plot([coords[i, 0], coords[j, 0]], [coords[i, 1], coords[j, 1]],
                color="lightgray", lw=0.6, alpha=0.4, zorder=1)

    def _plot(states_subset: Set[State], color: str, edge: str, size: int):
        if not states_subset:
            return
        idxs = [state_idx[s] for s in states_subset]
        ax.scatter(coords[idxs, 0], coords[idxs, 1], s=size, color=color,
                   edgecolors=edge, linewidths=1.2, zorder=4)

    _plot(failed, "lightgray", "#888", 80)
    _plot(stable, "#2ecc71", "black", 110)
    _plot(lost,   "#e74c3c", "black", 130)
    _plot(gained, "#3498db", "black", 150)

    for i, s in enumerate(states):
        ax.text(coords[i, 0], coords[i, 1], state_label(s), fontsize=6,
                ha="center", va="center", zorder=5,
                bbox={"boxstyle": "round,pad=0.15", "facecolor": "white",
                      "alpha": 0.75, "edgecolor": "none"})

    handles = [
        mpatches.Patch(facecolor="#3498db", edgecolor="black",
                       label=f"Gained ({len(gained)})"),
        mpatches.Patch(facecolor="#2ecc71", edgecolor="black",
                       label=f"Solved in both ({len(stable)})"),
        mpatches.Patch(facecolor="#e74c3c", edgecolor="black",
                       label=f"Lost ({len(lost)})"),
        mpatches.Patch(facecolor="lightgray", edgecolor="#888",
                       label=f"Failed in both ({len(failed)})"),
    ]
    ax.legend(handles=handles, loc="best", fontsize=9)
    ax.set_title(title)
    ax.grid(alpha=0.15)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def load_probe_coords(probe_dir: Path, layer: int, states: List[State]
                      ) -> Optional[np.ndarray]:
    hidden_path = probe_dir / "hidden_states.pt"
    probe_path = probe_dir / f"probe_layer_{layer}.pt"
    if not hidden_path.exists() or not probe_path.exists():
        print(f"[WARN] {hidden_path} or {probe_path} missing; skipping flipped plot")
        return None
    hs_data = torch.load(hidden_path, map_location="cpu", weights_only=False)
    probe_data = torch.load(probe_path, map_location="cpu", weights_only=False)
    hs = hs_data["hidden_states"][layer].float()
    sd = probe_data["state_dict"]
    d_model = hs.shape[1]
    probe = nn.Linear(d_model, 2, bias=True)
    probe.load_state_dict(sd)
    probe.eval()
    with torch.no_grad():
        coords = probe(hs).numpy().astype(np.float32)
    # Make sure ordering matches `states` (it does — probe.py uses the same enumeration)
    return coords


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    n_disks = args.n_disks
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    states_all = enumerate_states(n_disks)
    if n_disks == 4 and len(states_all) != 81:
        raise RuntimeError(f"Expected 81 states, got {len(states_all)}")

    # Optionally drop the problems the model already solves (per eval results).
    skipped_solved: Set[State] = set()
    if args.failed_only:
        eval_path = Path(args.eval_results)
        if not eval_path.exists():
            raise FileNotFoundError(
                f"--failed_only needs {eval_path}; run evaluate_model.py first")
        with open(eval_path) as f:
            eval_data = json.load(f)
        for r in eval_data.get("records", []):
            if r["category"] in CORRECT_CATEGORIES:
                skipped_solved.add(tuple(int(x) for x in r["state_tuple"]))
        before = len(states_all)
        states_all_for_run = [s for s in states_all if s not in skipped_solved]
        print(f"[INFO] --failed_only: dropped {before - len(states_all_for_run)} "
              f"already-solved states; running on {len(states_all_for_run)}")
    else:
        states_all_for_run = states_all

    states = states_all_for_run[: args.n_problems]
    n_states = len(states)
    goal_pegs = [[], [], list(range(n_disks, 0, -1))]

    print("[INFO] Computing optimal lengths")
    opt_lens = {s: optimal_length(s, n_disks) for s in states_all}

    print("[INFO] Building state graph for visualization")
    _, edges = build_graph(states_all, n_disks)

    layer_ids = [args.layer]
    if args.extra_layers:
        for x in args.extra_layers.split(","):
            x = x.strip()
            if x:
                lid = int(x)
                if lid not in layer_ids:
                    layer_ids.append(lid)
    print(f"[INFO] Steering layers (1-indexed): {layer_ids}")

    print(f"[INFO] Loading position-A activations from {args.hidden_states}")
    layer_state_to_vec: Dict[int, Dict[State, torch.Tensor]] = {}
    layer_mean_vec: Dict[int, torch.Tensor] = {}
    for lid in layer_ids:
        stv, mv = load_state_activations(Path(args.hidden_states), lid)
        layer_state_to_vec[lid] = stv
        layer_mean_vec[lid] = mv

    raw_model, text_model, tokenizer, cfg, device = load_model_and_tokenizer(
        args.model_name, dtype_from_str(args.dtype), args.device)
    eos_ids = _stop_token_ids(tokenizer)
    print(f"[INFO] EOS token ids: {sorted(eos_ids)}")
    print(f"[INFO] num_hidden_layers={int(cfg.num_hidden_layers)}  "
          f"hidden_size={int(cfg.hidden_size)}")

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    alphas = [float(x) for x in args.alphas.split(",") if x.strip()]
    betas = [float(x) for x in args.betas.split(",") if x.strip()]

    results: List[Dict[str, object]] = []
    baseline_result: Optional[Dict[str, object]] = None

    if args.run_baseline:
        print("\n[INFO] === Condition: no steering (baseline) ===")
        baseline_result = run_condition(
            label="baseline",
            states=states, raw_model=raw_model, text_model=text_model,
            tokenizer=tokenizer, n_disks=n_disks, opt_lens=opt_lens,
            goal_pegs=goal_pegs, layer_ids=[],
            layer_state_to_vec=layer_state_to_vec,
            layer_mean_vec=layer_mean_vec,
            mode=None, scale=None,
            max_new_tokens=args.max_new_tokens, device=device,
            eos_ids=eos_ids, decode_every=args.decode_every,
            save_texts=args.save_texts,
            always_steer=args.always_steer,
        )
        results.append(baseline_result)

    for mode in modes:
        scales = alphas if mode == "directional" else betas
        for scale in scales:
            tag = f"{mode}_{scale}_L{'-'.join(str(l) for l in layer_ids)}"
            print(f"\n[INFO] === Condition: {tag} ===")
            r = run_condition(
                label=tag,
                states=states, raw_model=raw_model, text_model=text_model,
                tokenizer=tokenizer, n_disks=n_disks, opt_lens=opt_lens,
                goal_pegs=goal_pegs, layer_ids=layer_ids,
                layer_state_to_vec=layer_state_to_vec,
                layer_mean_vec=layer_mean_vec,
                mode=mode, scale=scale,
                max_new_tokens=args.max_new_tokens, device=device,
                eos_ids=eos_ids, decode_every=args.decode_every,
                save_texts=args.save_texts,
                always_steer=args.always_steer,
            )
            results.append(r)

    if args.run_oracle:
        print("\n[INFO] === Condition: oracle (state in prompt) ===")
        oracle_result = run_oracle_condition(
            states=states, raw_model=raw_model, tokenizer=tokenizer,
            n_disks=n_disks, opt_lens=opt_lens, args=args,
            device=device, eos_ids=eos_ids,
        )
        results.append(oracle_result)

    print_comparison_table(results, n_states)

    # Save per-condition results (without texts unless --save_texts)
    summary = {
        "model_name": args.model_name,
        "n_disks": n_disks,
        "n_problems": n_states,
        "layer_ids": layer_ids,
        "modes": modes,
        "alphas": alphas,
        "betas": betas,
        "max_new_tokens": args.max_new_tokens,
        "results": [
            {k: v for k, v in r.items()
             if k not in ("per_problem",) or args.save_texts}
            for r in results
        ],
    }
    (out_dir / "steering_summary.json").write_text(
        json.dumps(summary, indent=2, default=lambda o: list(o) if isinstance(o, tuple) else o))
    for r in results:
        slug = r["label"].replace("/", "_")
        (out_dir / f"perproblem_{slug}.json").write_text(
            json.dumps(r["per_problem"], indent=2,
                       default=lambda o: list(o) if isinstance(o, tuple) else o))
    print(f"\n[INFO] Saved results to {out_dir}")

    # Sierpinski flipped-states plot for the best steering cell.
    # When --failed_only is set the baseline run may be absent, so we feed the
    # pre-known solved set in as `known_solved` to keep the plot honest.
    coords = load_probe_coords(Path(args.probe_dir), args.layer, states_all)
    if coords is not None:
        scored = [r for r in results if r["label"] != "baseline" and r["label"] != "oracle"]
        if scored and (baseline_result is not None or skipped_solved):
            best = max(scored, key=lambda r: r["n_solved"])
            base_n = (baseline_result["n_solved"] if baseline_result is not None
                      else len(skipped_solved))
            tot_known = len(skipped_solved) + best["n_solved"]
            plot_flipped(
                baseline=baseline_result,
                steered=best,
                states=states_all,
                edges=edges,
                coords=coords,
                out_path=out_dir / f"flipped_{best['label']}.png",
                title=(f"Steering effect: {best['label']}  "
                       f"baseline={base_n}/{len(states_all)}  "
                       f"steered={tot_known}/{len(states_all)}"),
                known_solved=skipped_solved,
            )
            print(f"[INFO] Best steered condition: {best['label']} "
                  f"({best['n_solved']}/{n_states} of run subset solved; "
                  f"{tot_known}/{len(states_all)} including pre-known solved)")

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
