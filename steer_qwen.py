#!/usr/bin/env python3
"""Directional activation steering for LRMs on flat-to-flat Tower of Hanoi.

Works with Qwen3-27B and DeepSeek-R1-Distill-Qwen-32B (any model whose
hidden_states.pt + probe_layer_*.pt artifacts were produced by probe.py).

Idea: the model encodes the state space cleanly at position A (last prompt
token) with Spearman ~0.935 between probe-distance and graph-distance, but
that signal degrades during long CoT. Use the position-A activations as
clean steering targets and inject them into the residual stream during
generation, updating the target each time the model commits a legal move.

Pipeline:
  1. Load layer-L position-A hidden states from <hidden_states.pt>
     → dict state_tuple → activation vector, plus the mean over all 81 states.
  2. Install a forward hook on layer-L that modifies the LAST-token residual
     at every forward pass (prefill + each generation step) in DIRECTIONAL
     mode:  h[-1] += alpha * unit(target - mean)
  3. Generate token-by-token with KV cache. When a ']' lands in the freshly
     decoded text, re-parse the LAST moves=[...] block from scratch, replay
     against the start state, and if the resulting board changes update the
     hook's target activation.
  4. Classify the final answer (CORRECT_OPTIMAL / CORRECT_SUBOPTIMAL /
     ILLEGAL_* / PARSE_ERROR / WRONG_GOAL_STATE).
  5. Sweep alpha and layer choices and print a comparison table.
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
    p = argparse.ArgumentParser(description="Directional activation steering for LRM ToH solving")
    p.add_argument("--model_name", default="Qwen/Qwen3-27B")
    p.add_argument("--hidden_states", default="outputs/qwen_probe/hidden_states.pt",
                   help="Position-A hidden states from probe.py")
    p.add_argument("--n_disks", type=int, default=4)
    p.add_argument("--layer", type=int, default=36,
                   help="Primary steering layer (1-indexed)")
    p.add_argument("--extra_layers", default="",
                   help="Comma-separated extra layers for joint steering, e.g. '48'")
    p.add_argument("--alphas", default="0.5,1,2,5,10,20")
    p.add_argument("--run_baseline", action="store_true",
                   help="Run a no-steering pass for comparison")
    p.add_argument("--output_dir", default="outputs/qwen_steering")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default="bfloat16",
                   choices=["float16", "bfloat16", "float32"])
    p.add_argument("--max_new_tokens", type=int, default=13000)
    p.add_argument("--n_problems", type=int, default=81)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--probe_dir", default="outputs/qwen_probe",
                   help="Used by the Sierpinski plot of flipped problems")
    p.add_argument("--decode_every", type=int, default=1,
                   help="Decode/parse-state cadence in tokens. 1 == every step.")
    p.add_argument("--batch_size", type=int, default=8,
                   help="Number of problems to generate in parallel per sweep cell. "
                        "1 = legacy single-problem path. KV-cache scales linearly.")
    p.add_argument("--save_texts", action="store_true",
                   help="Save full generated text per (condition, problem) — large")
    p.add_argument("--eval_results",
                   default="outputs/qwen_probe/evaluate_qwen_results.json",
                   help="Path to per-state eval results used by --failed_only. "
                        "Accepts either {records: [{state_tuple, category}]} (Qwen) "
                        "or a list of {state_tuple, passed} (DeepSeek validation_results.json).")
    p.add_argument("--failed_only", action="store_true",
                   help="Skip the states the model already solves in --eval_results")
    p.add_argument("--always_steer", action="store_true",
                   help="Steer during CoT too (legacy). Default: hook is dormant "
                        "until the model emits 'moves = [' in its output.")
    p.add_argument("--debug_steering", action="store_true",
                   help="Print one-shot diagnostics on the FIRST hook call: "
                        "module class, output tuple structure, dtype/shape/norm "
                        "of hidden state, direction, delta, before/after diff. "
                        "Off by default to keep production logs clean.")
    p.add_argument("--skip_sanity", action="store_true",
                   help="Skip the alpha=0 vs alpha=20 differential test that "
                        "runs once at startup. Default: the test always runs "
                        "and ABORTS if outputs are byte-identical.")
    p.add_argument("--gate_on_sanity", action="store_true",
                   help="Run an ADDITIONAL diagnostic at startup: generate one "
                        "full sequence (up to --gate_sanity_max_tokens) with "
                        "the gate active (steering activates only after "
                        "</think>+moves=[) at α=0 and at α=200, then compare "
                        "the first 30 post-gate tokens. Classifies the layer "
                        "into BRANCH 1 (steerable, gate doesn't block), "
                        "BRANCH 2 (steerable but gate-blocked → cache "
                        "dominance), or BRANCH 3 (layer not steerable). "
                        "Aborts on BRANCH 2 or 3.")
    p.add_argument("--gate_sanity_max_tokens", type=int, default=8000,
                   help="Token budget for each gate-on sanity generation "
                        "(needs to be long enough to pass </think>+moves=[). "
                        "Default 8000 (~2-3 min per generation on H100).")
    p.add_argument("--gate_sanity_escalate_alpha", type=float, default=500.0,
                   help="If α=200 fails BOTH gate-on and gate-off (existing "
                        "sanity), retry once at this α before classifying as "
                        "BRANCH 3.")
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

# Module-level guard so the first-call debug printout happens exactly once
# across ALL SteeringHook instances (hooks get created/destroyed per sweep
# cell). The hook short-circuits past the diagnostic block once flipped.
_STEERING_DEBUG_PRINTED: bool = False
# Toggled by main() based on --debug_steering. Hooks read this on every
# call (cheap) and only do the heavy printing when both flags align.
_STEERING_DEBUG_ENABLED: bool = False


def _module_path_in_model(model: nn.Module, target: nn.Module) -> str:
    """Best-effort name lookup so the debug block can print where the hook
    actually attached (e.g. `model.model.layers[35].mlp` would expose a
    layer-vs-sub-module mistake)."""
    for name, mod in model.named_modules():
        if mod is target:
            return name or "<root>"
    return "<not-found-in-model>"


def _hook_debug_dump(
    module: nn.Module, output, direction: torch.Tensor, alpha: float,
    *, model_for_lookup: Optional[nn.Module] = None,
) -> None:
    """Verifies every link in the chain on the very first hook call:
    module identity → output structure → hidden state stats → direction
    stats → delta after dtype cast → actual before/after diff norm.

    Prints with the [steer-debug] prefix so a `grep steer-debug <out>`
    yields the full sanity record from a single forward.
    """
    print("[steer-debug] === HOOK FIRST CALL ===")
    print(f"[steer-debug] module class: {type(module).__name__}")
    if model_for_lookup is not None:
        path = _module_path_in_model(model_for_lookup, module)
        print(f"[steer-debug] module name in model: {path}")
    print(f"[steer-debug] output type: {type(output).__name__}")
    if isinstance(output, tuple):
        print(f"[steer-debug] output tuple len: {len(output)}")
        for i, item in enumerate(output):
            if torch.is_tensor(item):
                print(f"[steer-debug]   output[{i}]: tensor shape={tuple(item.shape)} "
                      f"dtype={item.dtype} norm={float(item.float().norm()):.4f}")
            else:
                print(f"[steer-debug]   output[{i}]: {type(item).__name__} (non-tensor)")
        hs = output[0]
    else:
        hs = output
    print(f"[steer-debug] hidden_state to modify: shape={tuple(hs.shape)} dtype={hs.dtype}")
    print(f"[steer-debug]   norm={float(hs.float().norm()):.4f}  "
          f"abs_mean={float(hs.float().abs().mean()):.6f}")
    print(f"[steer-debug] direction: shape={tuple(direction.shape)}  "
          f"dtype={direction.dtype}  device={direction.device}")
    print(f"[steer-debug]   norm={float(direction.float().norm()):.4f}  "
          f"abs_mean={float(direction.float().abs().mean()):.6f}")
    print(f"[steer-debug]   first 5 values: {direction.float().flatten()[:5].tolist()}")
    print(f"[steer-debug] alpha={alpha}  type={type(alpha).__name__}")

    # Reproduce the same arithmetic the hook itself will apply.
    delta_fp32 = (alpha * direction.float()).to(direction.device)
    delta_cast = delta_fp32.to(hs.dtype)
    print(f"[steer-debug] delta (alpha·direction):")
    print(f"[steer-debug]   fp32  abs_mean={float(delta_fp32.abs().mean()):.6f}  "
          f"abs_max={float(delta_fp32.abs().max()):.6f}  "
          f"norm={float(delta_fp32.norm()):.4f}")
    print(f"[steer-debug]   cast to {hs.dtype}: "
          f"abs_mean={float(delta_cast.float().abs().mean()):.6f}  "
          f"abs_max={float(delta_cast.float().abs().max()):.6f}  "
          f"norm={float(delta_cast.float().norm()):.4f}")
    nz_frac = float((delta_cast != 0).float().mean())
    print(f"[steer-debug]   nonzero fraction after cast: {nz_frac:.4f}  "
          f"({'underflow!' if nz_frac < 0.5 else 'ok'})")

    # Apply against the LAST token slice — the same slice the real hook
    # mutates. Detach + clone so we don't side-effect.
    hs_last_before = hs[..., -1:, :].detach().clone()
    hs_last_after = hs_last_before + delta_cast.to(hs_last_before.dtype)
    actual_diff = float((hs_last_after - hs_last_before).float().norm())
    norm_before = float(hs_last_before.float().norm())
    ratio = actual_diff / norm_before if norm_before > 0 else float("nan")
    print(f"[steer-debug] last-token slice norm: before={norm_before:.4f}  "
          f"diff_after_add={actual_diff:.4f}  ratio={ratio:.6f}")
    if actual_diff == 0.0:
        print("[steer-debug] ⚠ actual_diff == 0 — bf16 underflow or wrong slot. "
              "Hook is a no-op on the residual stream.")
    print("[steer-debug] === END HOOK DEBUG ===\n")


class SteeringHook:
    """Forward hook on one transformer layer; rewrites residual at the LAST tok.

    Directional steering:  h[-1] += alpha * unit(target - mean)

    The hook is REGISTERED throughout but stays *dormant* (no-op) until
    `moves_phase_active` is flipped on. The generation loop turns it on
    AFTER the model first emits `</think>` (R1's CoT-close token) and then
    the final-answer header `moves = [`. Gating on </think> matters because
    R1's CoT routinely speculates with phrases like "let me try moves = […]";
    triggering on those would steer the entire CoT and collapse the reasoning.
    With the gate, the hook fires only on the final-answer tokens
    (~50–100/problem instead of ~12k).
    """

    def __init__(self, layer_module: nn.Module, layer_id: int, alpha: float,
                 moves_phase_active: bool = False):
        self.layer_id = layer_id
        self.alpha = float(alpha)
        # Scalar / single-row state (kept for backwards compat).
        self.target_vec: Optional[torch.Tensor] = None
        self.mean_vec: Optional[torch.Tensor] = None
        self.steering_unit: Optional[torch.Tensor] = None
        # Batched state (overrides scalar when set).
        self.steering_units_batched: Optional[torch.Tensor] = None  # (B, H)
        self.moves_phase_mask: Optional[torch.Tensor] = None        # (B,) bool
        self.enabled = True
        self.moves_phase_active = moves_phase_active
        self.fired = 0
        # Per-instance debug print flag. Each hook instance prints once on
        # its first call (so an α=0 sanity hook doesn't silence the α=20 one).
        self._debug_printed_this_instance = False
        self.handle = layer_module.register_forward_hook(self._hook)

    def update_target(self, target_vec: torch.Tensor, mean_vec: torch.Tensor) -> None:
        self.target_vec = target_vec
        self.mean_vec = mean_vec
        diff = target_vec - mean_vec
        n = float(diff.norm())
        self.steering_unit = diff / n if n > 1e-8 else diff.clone()

    def update_targets_batched(self, units: torch.Tensor) -> None:
        """units: (B, hidden_dim) precomputed unit vectors (target - mean) / ||·||.
        Active rows are controlled by `set_moves_phase_mask`.
        """
        self.steering_units_batched = units

    def set_moves_phase_mask(self, mask: torch.Tensor) -> None:
        """mask: (B,) bool — True for rows that should receive the steering bump."""
        self.moves_phase_mask = mask

    def activate_moves_phase(self) -> None:
        self.moves_phase_active = True

    def reset_moves_phase(self, active: bool = False) -> None:
        self.moves_phase_active = active
        self.fired = 0

    def _hook(self, module, inputs, output):
        global _STEERING_DEBUG_PRINTED
        if not self.enabled:
            return None
        h = output[0] if isinstance(output, tuple) else output

        # Batched path takes precedence when units have been registered.
        if self.steering_units_batched is not None and self.moves_phase_mask is not None:
            mask = self.moves_phase_mask.to(h.device)
            if not mask.any():
                return None
            sv = self.steering_units_batched.to(device=h.device, dtype=h.dtype)
            should_debug = (_STEERING_DEBUG_ENABLED
                            and not self._debug_printed_this_instance
                            and sv.numel() > 0)
            if should_debug:
                _hook_debug_dump(module, output, sv[0], self.alpha,
                                 model_for_lookup=None)
                self._debug_printed_this_instance = True
                _STEERING_DEBUG_PRINTED = True  # keep global for back-compat
            delta = self.alpha * sv * mask.to(dtype=h.dtype).unsqueeze(-1)  # (B, H)
            # Clone-and-return: avoid any in-place ambiguity when the parent
            # module's forward returns a bare Tensor (newer transformers).
            new_h = h.clone()
            new_h[:, -1:, :] = h[:, -1:, :] + delta.unsqueeze(1)
            self.fired += int(mask.sum().item())
            if should_debug:
                actual = float((new_h[:, -1:, :] - h[:, -1:, :]).float().norm())
                print(f"[steer-debug] post-modify check: ‖new[-1] - old[-1]‖ = "
                      f"{actual:.4f}  alpha={self.alpha}  layer={self.layer_id}")
            if isinstance(output, tuple):
                return (new_h,) + output[1:]
            return new_h

        # Scalar / single-row path (legacy).
        if (not self.moves_phase_active or self.steering_unit is None):
            return None
        sv = self.steering_unit.to(device=h.device, dtype=h.dtype)
        should_debug = (_STEERING_DEBUG_ENABLED
                        and not self._debug_printed_this_instance)
        if should_debug:
            _hook_debug_dump(module, output, sv, self.alpha)
            self._debug_printed_this_instance = True
            _STEERING_DEBUG_PRINTED = True  # keep global for back-compat
        new_h = h.clone()
        new_h[:, -1:, :] = h[:, -1:, :] + self.alpha * sv
        self.fired += 1
        if should_debug:
            actual = float((new_h[:, -1:, :] - h[:, -1:, :]).float().norm())
            print(f"[steer-debug] post-modify check: ‖new[-1] - old[-1]‖ = "
                  f"{actual:.4f}  alpha={self.alpha}  layer={self.layer_id}  "
                  f"(if alpha>0 and this is 0, hook math is wrong)")
        if isinstance(output, tuple):
            return (new_h,) + output[1:]
        return new_h

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
# R1 emits a full <think>…</think> CoT before the final-answer block. The CoT
# routinely contains speculative "moves = [(0, 0, 2), …]" fragments as the
# model reasons. We must NOT activate the steering hook on those — only on the
# post-think final-answer header. Gate the trigger on </think> having appeared
# in the generation so far.
_THINK_END_RE = re.compile(r'</think>', re.IGNORECASE)


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
    think_closed = always_steer  # if forcing steer-during-CoT, skip the gate

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
            # R1's CoT routinely contains speculative "moves = [..." fragments.
            # First wait for </think> to close the CoT, THEN look for the real
            # final-answer header in subsequent tokens.
            if hooks and not moves_phase:
                tail = tokenizer.decode(
                    generated_ids[-min(32, len(generated_ids)):],
                    skip_special_tokens=False,
                )
                if not think_closed and _THINK_END_RE.search(tail):
                    think_closed = True
                if think_closed and _MOVES_HEADER_RE.search(tail):
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


def _compute_unit(target: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:
    diff = target - mean
    n = float(diff.norm())
    return diff / n if n > 1e-8 else diff.clone()


def generate_batched_with_steering(
    raw_model: nn.Module,
    tokenizer,
    prompt_texts: List[str],
    start_states: List[State],
    n_disks: int,
    layer_state_to_vec: Dict[int, Dict[State, torch.Tensor]],
    layer_mean_vec: Dict[int, torch.Tensor],
    hooks: Sequence[SteeringHook],
    max_new_tokens: int,
    device: torch.device,
    eos_ids: Set[int],
    decode_every: int = 1,
    always_steer: bool = False,
) -> List[Tuple[str, State, int, int]]:
    """Batched analogue of `generate_with_steering`. All B problems decode in
    lockstep with KV cache; the hook applies a per-row steering bump only to
    rows whose moves-phase mask is on, using a per-row unit vector.

    Returns a list of (gen_text, current_sim_state, n_legal, moves_phase_step)
    in the same order as `prompt_texts`.
    """
    B = len(prompt_texts)
    if B == 0:
        return []

    # Left-pad so all prompts end at the same token offset → generation
    # appends new tokens at position -1 for every row.
    saved_padding_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "left"
    try:
        enc = tokenizer(
            prompt_texts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
    finally:
        tokenizer.padding_side = saved_padding_side

    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    start_pegs_list = [state_tuple_to_pegs(s, n_disks) for s in start_states]
    current_states = [tuple(int(x) for x in s) for s in start_states]
    moves_phase = [bool(always_steer)] * B
    moves_phase_step = [-1] * B
    think_closed = [bool(always_steer)] * B  # per-row </think>-seen gate
    finished = [False] * B
    generated_ids_per_row: List[List[int]] = [[] for _ in range(B)]
    n_legal_total = [0] * B

    # Pre-fetch the (state -> unit-vector) cache per hook, then build the
    # initial per-row unit tensor.
    hook_state_units: Dict[int, Dict[State, torch.Tensor]] = {}
    for h in hooks:
        stv = layer_state_to_vec.get(h.layer_id)
        mv = layer_mean_vec.get(h.layer_id)
        if stv is None or mv is None:
            continue
        hook_state_units[h.layer_id] = {s: _compute_unit(v, mv) for s, v in stv.items()}

    def _refresh_hook_units() -> None:
        if not hooks:
            return
        for h in hooks:
            units_map = hook_state_units.get(h.layer_id)
            if units_map is None:
                continue
            rows = []
            for s in current_states:
                rows.append(units_map.get(s, torch.zeros_like(next(iter(units_map.values())))))
            stacked = torch.stack(rows, dim=0)  # (B, H)
            h.update_targets_batched(stacked)
        mask = torch.tensor(moves_phase, dtype=torch.bool)
        for h in hooks:
            h.set_moves_phase_mask(mask)

    _refresh_hook_units()
    for h in hooks:
        h.reset_moves_phase(active=False)  # we drive the mask explicitly per-row

    cur_input = input_ids
    cur_attn = attention_mask
    past = None
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    with torch.no_grad():
        for step in range(max_new_tokens):
            if all(finished):
                break
            out = raw_model(
                input_ids=cur_input,
                attention_mask=cur_attn,
                past_key_values=past,
                use_cache=True,
            )
            past = out.past_key_values
            next_logits = out.logits[:, -1, :]
            next_ids = next_logits.argmax(dim=-1)  # (B,)

            for b in range(B):
                if finished[b]:
                    continue
                tok_id = int(next_ids[b].item())
                generated_ids_per_row[b].append(tok_id)
                if tok_id in eos_ids:
                    finished[b] = True

            # Substitute finished rows with pad so KV-cache stays aligned but
            # outputs don't affect anything we read from them.
            substituted = next_ids.clone()
            for b in range(B):
                if finished[b]:
                    substituted[b] = pad_id
            cur_input = substituted.unsqueeze(-1)  # (B, 1)
            cur_attn = torch.cat(
                [cur_attn, torch.ones((B, 1), device=device, dtype=cur_attn.dtype)],
                dim=1,
            )

            # ── Per-row moves-phase detection + state tracking ───────────────
            state_changed = False
            phase_changed = False
            for b in range(B):
                if finished[b]:
                    continue
                ids_b = generated_ids_per_row[b]
                if not moves_phase[b]:
                    tail = tokenizer.decode(
                        ids_b[-min(32, len(ids_b)):],
                        skip_special_tokens=False,
                    )
                    if not think_closed[b] and _THINK_END_RE.search(tail):
                        think_closed[b] = True
                        print(f"    [gate] row {b}: </think> seen at step {step}")
                    if think_closed[b] and _MOVES_HEADER_RE.search(tail):
                        moves_phase[b] = True
                        moves_phase_step[b] = step
                        phase_changed = True
                        print(f"    [gate] row {b}: moves=[ opened at step {step} "
                              f"→ steering ON")

                if decode_every <= 1 or (step % decode_every == 0):
                    tail = tokenizer.decode(
                        ids_b[-min(8, len(ids_b)):],
                        skip_special_tokens=False,
                    )
                    if ']' not in tail:
                        continue
                    full_text = tokenizer.decode(ids_b, skip_special_tokens=False)
                    block = find_last_moves_block(full_text)
                    if block is None:
                        continue
                    pegs, n_legal, _, _ = replay_block(block[1], start_pegs_list[b], n_disks)
                    new_state = pegs_to_state(pegs, n_disks)
                    if new_state != current_states[b]:
                        current_states[b] = new_state
                        n_legal_total[b] = n_legal
                        state_changed = True

            if hooks and (state_changed or phase_changed):
                _refresh_hook_units()

    out_rows: List[Tuple[str, State, int, int]] = []
    for b in range(B):
        gen_text = tokenizer.decode(generated_ids_per_row[b], skip_special_tokens=False)
        out_rows.append((gen_text, current_states[b], n_legal_total[b], moves_phase_step[b]))
    return out_rows


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
    alpha: Optional[float],
    max_new_tokens: int,
    device: torch.device,
    eos_ids: Set[int],
    decode_every: int,
    save_texts: bool,
    always_steer: bool = False,
    batch_size: int = 1,
) -> Dict[str, object]:
    """Run a single sweep cell. alpha=None → no steering.
    `batch_size > 1` enables parallel multi-problem decoding via
    `generate_batched_with_steering`.
    """
    hooks: List[SteeringHook] = []
    if alpha is not None:
        layers_list = get_layers_list(text_model)
        for lid in layer_ids:
            hooks.append(SteeringHook(
                layers_list[lid - 1], lid, float(alpha),
                moves_phase_active=always_steer,
            ))

    per_problem: List[Dict[str, object]] = []
    cat_counter: Counter = Counter()
    t0 = time.time()
    bs = max(1, batch_size)
    try:
        idx = 0
        while idx < len(states):
            chunk = states[idx: idx + bs]
            prompts = [build_prompt(tokenizer, s, idx + j, n_disks)
                       for j, s in enumerate(chunk)]
            try:
                if bs > 1:
                    results = generate_batched_with_steering(
                        raw_model, tokenizer, prompts, list(chunk), n_disks,
                        layer_state_to_vec, layer_mean_vec, hooks,
                        max_new_tokens, device, eos_ids, decode_every,
                        always_steer=always_steer,
                    )
                else:
                    g_text, g_state, g_legal, g_step = generate_with_steering(
                        raw_model, tokenizer, prompts[0], chunk[0], n_disks,
                        layer_state_to_vec, layer_mean_vec, hooks,
                        max_new_tokens, device, eos_ids, decode_every,
                        always_steer=always_steer,
                    )
                    results = [(g_text, g_state, g_legal, g_step)]
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"  [OOM on batch starting at prob {idx} "
                          f"(size {len(chunk)}); recovering]")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    results = [("", s, 0, -1) for s in chunk]
                else:
                    raise

            for j, ((gen_text, sim_state, n_legal, moves_step), state) in enumerate(
                zip(results, chunk)
            ):
                prob_idx = idx + j
                start_pegs = state_tuple_to_pegs(state, n_disks)
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

            idx += bs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        for h in hooks:
            h.remove()

    return {
        "label": label,
        "alpha": alpha,
        "layer_ids": layer_ids,
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
    print("\n" + "=" * 116)
    print("COMPARISON TABLE")
    print("=" * 116)
    hdr = (f"{'Condition':<28} {'Layers':>8} {'α':>7} "
           f"{'Solved':>8} {'Optimal':>8} {'Illegal':>8} {'Parse':>7} {'Wrong':>7} "
           f"{'≠ref':>7}")
    print(hdr)
    print("-" * 116)
    # Pick the reference: first result with alpha=0.5 if present, else lowest
    # non-None alpha. "≠ref" counts per-problem (category, sim_state) tuples
    # that differ from the reference — i.e. how many problems higher α actually
    # changed.
    ref_idx = None
    for i, r in enumerate(results):
        a = r.get("alpha")
        if isinstance(a, (int, float)) and abs(a - 0.5) < 1e-6:
            ref_idx = i
            break
    if ref_idx is None:
        non_none = [(i, r["alpha"]) for i, r in enumerate(results)
                    if isinstance(r.get("alpha"), (int, float))]
        if non_none:
            ref_idx = min(non_none, key=lambda x: x[1])[0]
    ref_per_prob: Dict[int, Tuple[str, Tuple[int, ...]]] = {}
    if ref_idx is not None:
        for rec in results[ref_idx]["per_problem"]:
            ref_per_prob[int(rec["prob_idx"])] = (
                rec.get("category", ""),
                tuple(rec.get("sim_state", ())),
            )

    for i, r in enumerate(results):
        alpha_str = f"{r['alpha']:.3f}" if isinstance(r["alpha"], (int, float)) else "  -"
        layers_str = ",".join(str(l) for l in r["layer_ids"]) if r["layer_ids"] else "-"
        cc = r["category_counts"]
        wrong = (cc.get(WRONG_GOAL_STATE, 0) + cc.get(PREMATURE_STOP, 0)
                 + cc.get(EXCESSIVE_MOVES, 0))
        if i == ref_idx or ref_idx is None:
            diff_str = "  -"
        else:
            n_diff = sum(
                1 for rec in r["per_problem"]
                if (rec.get("category", ""), tuple(rec.get("sim_state", ())))
                != ref_per_prob.get(int(rec["prob_idx"]), (None, None))
            )
            diff_str = f"{n_diff:>3d}/{n_states:<3d}"
        print(f"{r['label']:<28} {layers_str:>8} {alpha_str:>7} "
              f"{r['n_solved']:>4d}/{n_states:<3d}  "
              f"{r['n_optimal']:>4d}/{n_states:<3d} "
              f"{r['n_illegal']:>4d}/{n_states:<3d} "
              f"{r['n_parse_error']:>3d}/{n_states:<3d} "
              f"{wrong:>3d}/{n_states:<3d} "
              f"{diff_str:>7}")
    if ref_idx is not None:
        ref_alpha = results[ref_idx]["alpha"]
        print(f"\n  '≠ref' = problems whose (category, sim_state) differs from "
              f"α={ref_alpha} (the reference). Near-zero values across all rows "
              f"mean higher α didn't change anything — same failure pattern as "
              f"the L36 run.")


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
    # Accepts two schemas:
    #   Qwen-style: {records: [{state_tuple, category, ...}]} where success means
    #               category in CORRECT_CATEGORIES.
    #   DeepSeek-style: list of {state_tuple, passed, ...} where success is passed=True.
    skipped_solved: Set[State] = set()
    if args.failed_only:
        eval_path = Path(args.eval_results)
        if not eval_path.exists():
            raise FileNotFoundError(
                f"--failed_only needs {eval_path}; run evaluate_model.py first")
        with open(eval_path) as f:
            eval_data = json.load(f)
        if isinstance(eval_data, dict) and "records" in eval_data:
            for r in eval_data["records"]:
                if r.get("category") in CORRECT_CATEGORIES:
                    skipped_solved.add(tuple(int(x) for x in r["state_tuple"]))
        elif isinstance(eval_data, list):
            for r in eval_data:
                if r.get("passed"):
                    st = r.get("state_tuple") or r.get("state")
                    if st is not None:
                        skipped_solved.add(tuple(int(x) for x in st))
        else:
            raise ValueError(f"Unrecognized schema in {eval_path}")
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

    # ── Flip global debug flag so the hook prints on its first call. ───
    global _STEERING_DEBUG_ENABLED
    _STEERING_DEBUG_ENABLED = bool(args.debug_steering)

    # ── hidden_states.pt provenance check (always; cheap, critical) ────
    hs_path = Path(args.hidden_states)
    if not hs_path.exists():
        raise FileNotFoundError(f"hidden_states not found: {hs_path}")
    import hashlib, datetime as _dt
    h = hashlib.md5()
    with hs_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    mtime = _dt.datetime.fromtimestamp(hs_path.stat().st_mtime).isoformat(timespec="seconds")
    print(f"[steer-startup] hidden_states.pt = {hs_path}")
    print(f"[steer-startup]   md5={h.hexdigest()}  mtime={mtime}  "
          f"size={hs_path.stat().st_size:,} bytes")

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

    # ── Direction sanity + dim match against the model ─────────────────
    primary_layer = layer_ids[0]
    primary_stv = layer_state_to_vec[primary_layer]
    primary_mv = layer_mean_vec[primary_layer]
    sample_state = next(iter(primary_stv))
    sample_diff = (primary_stv[sample_state] - primary_mv)
    sample_norm = float(sample_diff.norm())
    print(f"[steer-startup] model class = {type(raw_model).__name__}  "
          f"hidden_size={int(cfg.hidden_size)}")
    print(f"[steer-startup] direction (target-mean) at L{primary_layer}: "
          f"shape={tuple(sample_diff.shape)}  dtype={sample_diff.dtype}  "
          f"norm={sample_norm:.4f}  abs_mean={float(sample_diff.abs().mean()):.6f}")
    if sample_norm < 0.1:
        print("[steer-startup] ⚠ direction norm < 0.1 — vector looks near-zero. "
              "Probably a stale hidden_states.pt (wrong model / corrupted).")
    if sample_diff.shape[-1] != int(cfg.hidden_size):
        raise RuntimeError(
            f"DIRECTION DIM MISMATCH: hidden_states gives {sample_diff.shape[-1]} "
            f"but model.config.hidden_size is {int(cfg.hidden_size)}. "
            f"hidden_states.pt is from a different model — regenerate it."
        )
    # Layer module identity (catches off-by-one + wrong-sub-module bugs).
    layers_list = get_layers_list(text_model)
    target_module = layers_list[primary_layer - 1]
    print(f"[steer-startup] layer index {primary_layer} (1-indexed) → "
          f"{type(target_module).__name__} "
          f"at depth {primary_layer}/{int(cfg.num_hidden_layers)} "
          f"({100.0 * primary_layer / int(cfg.num_hidden_layers):.1f}% through net)")

    alphas = [float(x) for x in args.alphas.split(",") if x.strip()]

    # ── Differential sanity: alpha=0 vs alpha=20 on one prompt ─────────
    # If outputs are byte-identical we abort BEFORE the (expensive) sweep,
    # because something downstream is silently swallowing the modification
    # (most often: bf16 underflow, wrong tuple element, or non-returning hook).
    if not args.skip_sanity:
        print("\n[steer-startup] running α=0 vs α=20 differential test on "
              f"state={states[0]} (this is the empty-vs-loud probe)…")
        from_state = states[0]
        sanity_prompt = build_prompt(tokenizer, from_state, 0, n_disks)
        sanity_max_tokens = min(96, args.max_new_tokens)

        def _sanity_one(alpha_val: float) -> Tuple[List[int], List[int]]:
            hooks_local: List[SteeringHook] = []
            for lid in layer_ids:
                hk = SteeringHook(
                    layers_list[lid - 1], lid, float(alpha_val),
                    moves_phase_active=True,  # force-on for this test
                )
                hk.update_target(layer_state_to_vec[lid][from_state],
                                 layer_mean_vec[lid])
                hooks_local.append(hk)
            try:
                enc = tokenizer(sanity_prompt, return_tensors="pt",
                                add_special_tokens=False)
                input_ids = enc["input_ids"].to(device)
                gen_ids: List[int] = []
                cur = input_ids
                past = None
                with torch.no_grad():
                    for _ in range(sanity_max_tokens):
                        out = raw_model(input_ids=cur, past_key_values=past, use_cache=True)
                        past = out.past_key_values
                        nxt = int(out.logits[:, -1, :].argmax(dim=-1).item())
                        gen_ids.append(nxt)
                        if nxt in eos_ids:
                            break
                        cur = torch.tensor([[nxt]], device=device)
                fired_counts = [hk.fired for hk in hooks_local]
                return gen_ids, fired_counts
            finally:
                for hk in hooks_local:
                    hk.remove()

        ids_a0,  fired_a0  = _sanity_one(0.0)
        ids_a20, fired_a20 = _sanity_one(20.0)
        head_a0  = tokenizer.decode(ids_a0[:30],  skip_special_tokens=False)
        head_a20 = tokenizer.decode(ids_a20[:30], skip_special_tokens=False)
        print(f"[steer-startup]   α=0   fired per-layer: {fired_a0}")
        print(f"[steer-startup]   α=0   first 30 tok ids: {ids_a0[:30]}")
        print(f"[steer-startup]   α=0   decoded head: {head_a0!r}")
        print(f"[steer-startup]   α=20  fired per-layer: {fired_a20}")
        print(f"[steer-startup]   α=20  first 30 tok ids: {ids_a20[:30]}")
        print(f"[steer-startup]   α=20  decoded head: {head_a20!r}")

        same_20 = ids_a0 == ids_a20
        if same_20:
            # Identical at α=20 doesn't yet prove the hook is broken — R1's
            # opener "Okay, so I have …" is extremely confident, and a 20-unit
            # bump on a ~180-norm residual slice may not cross the argmax
            # margin for the first ~30 tokens. Escalate to α=200 to
            # disambiguate true no-op vs subthreshold perturbation.
            print("[steer-startup] α=20 produced byte-identical tokens — "
                  "escalating to α=200 to distinguish a broken hook from a "
                  "subthreshold bump (R1 opens with very confident tokens).")
            ids_a200, fired_a200 = _sanity_one(200.0)
            head_a200 = tokenizer.decode(ids_a200[:30], skip_special_tokens=False)
            print(f"[steer-startup]   α=200 fired per-layer: {fired_a200}")
            print(f"[steer-startup]   α=200 first 30 tok ids: {ids_a200[:30]}")
            print(f"[steer-startup]   α=200 decoded head: {head_a200!r}")
            if ids_a0 == ids_a200:
                raise RuntimeError(
                    "Steering has no effect even at α=200 — hook is a no-op. "
                    "fired_a200={} (zeros mean the hook never ran). Re-run "
                    "with --debug_steering to see the [steer-debug] block for "
                    "the α=200 sanity call.".format(fired_a200)
                )
            else:
                print("[steer-startup] ✓ outputs DIFFER at α=200 — hook is "
                      "functional. α=20 was simply below R1's argmax-flip "
                      "threshold on the opener. Pick a larger production α "
                      "or accept that early-token effects will be muted.")
        else:
            print("[steer-startup] ✓ outputs differ at α=20 — steering modifies the residual stream.")
    else:
        print("[steer-startup] --skip_sanity set; differential α=0/α=20 test skipped.")

    # ── Gate-on diagnostic ─────────────────────────────────────────────────
    # The existing sanity above runs the hook with moves_phase_active=True
    # (force-on / gate OFF). That tests whether the layer is steerable in
    # principle. But the production sweep gates the hook ON only AFTER
    # </think>+moves=[, by which point thousands of tokens of KV cache may
    # dominate attention. This second diagnostic generates one full sequence
    # with the gate ON at α=0 and at α=200 and compares the first 30 post-
    # gate tokens. If they're identical, the gate blocks the effect →
    # cache-dominance is real and the sweep would waste compute.
    if args.gate_on_sanity:
        print(f"\n[steer-startup] GATE-ON differential test at L{primary_layer} "
              f"(α=0 vs α=200, max_new_tokens={args.gate_sanity_max_tokens}). "
              f"This generates two ~3-min sequences sequentially.")
        gate_prompt = build_prompt(tokenizer, states[0], 0, n_disks)

        def _gate_on_one(alpha_val: float) -> Tuple[List[int], int, int]:
            """Generate with gate-on hook. Returns (gen_ids, gate_step, fired).
            gate_step = step at which moves=[ triggered the hook, -1 if never."""
            hooks_local: List[SteeringHook] = []
            for lid in layer_ids:
                hk = SteeringHook(
                    layers_list[lid - 1], lid, float(alpha_val),
                    moves_phase_active=False,  # ← GATE ON (dormant until trigger)
                )
                hk.update_target(layer_state_to_vec[lid][states[0]],
                                 layer_mean_vec[lid])
                hooks_local.append(hk)
            try:
                enc = tokenizer(gate_prompt, return_tensors="pt",
                                add_special_tokens=False)
                input_ids = enc["input_ids"].to(device)
                gen_ids: List[int] = []
                think_closed = False
                gate_step = -1
                cur = input_ids
                past = None
                with torch.no_grad():
                    for step in range(args.gate_sanity_max_tokens):
                        out = raw_model(input_ids=cur, past_key_values=past,
                                        use_cache=True)
                        past = out.past_key_values
                        nxt = int(out.logits[:, -1, :].argmax(dim=-1).item())
                        gen_ids.append(nxt)
                        if nxt in eos_ids:
                            break
                        cur = torch.tensor([[nxt]], device=device)
                        # Trigger detection — same two-stage gate as production.
                        if gate_step < 0:
                            tail = tokenizer.decode(
                                gen_ids[-min(32, len(gen_ids)):],
                                skip_special_tokens=False,
                            )
                            if not think_closed and _THINK_END_RE.search(tail):
                                think_closed = True
                            if think_closed and _MOVES_HEADER_RE.search(tail):
                                gate_step = step
                                for hk in hooks_local:
                                    hk.activate_moves_phase()
                fired = hooks_local[0].fired if hooks_local else 0
                return gen_ids, gate_step, fired
            finally:
                for hk in hooks_local:
                    hk.remove()

        ids_g0, gate0, fired_g0 = _gate_on_one(0.0)
        ids_g200, gate200, fired_g200 = _gate_on_one(200.0)

        def _post_gate(ids: List[int], gate: int, n: int = 30) -> List[int]:
            if gate < 0:
                return []
            return ids[gate: gate + n]

        post0 = _post_gate(ids_g0, gate0)
        post200 = _post_gate(ids_g200, gate200)
        post0_text = tokenizer.decode(post0, skip_special_tokens=False) if post0 else "<gate never fired>"
        post200_text = tokenizer.decode(post200, skip_special_tokens=False) if post200 else "<gate never fired>"
        print(f"[steer-startup]   GATE-ON α=0   total_gen={len(ids_g0):5d}  "
              f"gate_at={gate0}  fired={fired_g0}")
        print(f"[steer-startup]   GATE-ON α=0   post-gate (first 30): {post0_text!r}")
        print(f"[steer-startup]   GATE-ON α=200 total_gen={len(ids_g200):5d}  "
              f"gate_at={gate200}  fired={fired_g200}")
        print(f"[steer-startup]   GATE-ON α=200 post-gate (first 30): {post200_text!r}")

        # Branch classification.
        gate_on_differs = bool(post0) and bool(post200) and (post0 != post200)
        # gate-off result from existing sanity: any α we tested that differed
        # from α=0. ids_a20/ids_a200 are populated above; reconstruct here.
        # If --skip_sanity was passed, we can't classify — bail with a message.
        if args.skip_sanity:
            print("[steer-startup] ⚠ --skip_sanity was set; cannot determine "
                  "gate-off baseline. Skipping branch classification.")
        else:
            ids_a200_in_scope = ids_a200 if 'ids_a200' in dir() else ids_a20
            gate_off_differs = (ids_a0 != ids_a200_in_scope)
            if gate_off_differs and gate_on_differs:
                print(f"[steer-startup] ✓ BRANCH 1: L{primary_layer} is steerable "
                      f"AND gate-on still has an effect. Proceeding to full eval.")
            elif gate_off_differs and not gate_on_differs:
                print(f"[steer-startup] ⚠ BRANCH 2: L{primary_layer} is steerable "
                      f"force-on but gate-on produces identical post-gate tokens. "
                      f"This confirms KV-cache dominance — layer choice doesn't help.")
                print(f"[steer-startup] Aborting BEFORE the {len(alphas)}-α sweep "
                      f"to save compute. Re-run without --gate_on_sanity to force.")
                sys.exit(2)
            else:
                # Try one more α at the escalation level.
                esc = args.gate_sanity_escalate_alpha
                print(f"[steer-startup] α=200 didn't move either gate-off or "
                      f"gate-on. Retrying gate-on at α={esc}…")
                ids_gE, gateE, fired_gE = _gate_on_one(esc)
                postE = _post_gate(ids_gE, gateE)
                postE_text = tokenizer.decode(postE, skip_special_tokens=False) if postE else "<gate never fired>"
                print(f"[steer-startup]   GATE-ON α={esc} gate_at={gateE}  "
                      f"fired={fired_gE}  post-gate (first 30): {postE_text!r}")
                if post0 and postE and post0 != postE:
                    print(f"[steer-startup] ✓ BRANCH 1 at α={esc}.")
                else:
                    print(f"[steer-startup] ✗ BRANCH 3: L{primary_layer} not "
                          f"steerable at any α we tried (20, 200, {esc}). "
                          f"Aborting.")
                    sys.exit(2)

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
            alpha=None,
            max_new_tokens=args.max_new_tokens, device=device,
            eos_ids=eos_ids, decode_every=args.decode_every,
            save_texts=args.save_texts,
            always_steer=args.always_steer,
            batch_size=args.batch_size,
        )
        results.append(baseline_result)

    for alpha in alphas:
        tag = f"directional_{alpha}_L{'-'.join(str(l) for l in layer_ids)}"
        print(f"\n[INFO] === Condition: {tag} ===")
        r = run_condition(
            label=tag,
            states=states, raw_model=raw_model, text_model=text_model,
            tokenizer=tokenizer, n_disks=n_disks, opt_lens=opt_lens,
            goal_pegs=goal_pegs, layer_ids=layer_ids,
            layer_state_to_vec=layer_state_to_vec,
            layer_mean_vec=layer_mean_vec,
            alpha=alpha,
            max_new_tokens=args.max_new_tokens, device=device,
            eos_ids=eos_ids, decode_every=args.decode_every,
            save_texts=args.save_texts,
            always_steer=args.always_steer,
            batch_size=args.batch_size,
        )
        results.append(r)

    print_comparison_table(results, n_states)

    # Save per-condition results (without texts unless --save_texts)
    summary = {
        "model_name": args.model_name,
        "n_disks": n_disks,
        "n_problems": n_states,
        "layer_ids": layer_ids,
        "alphas": alphas,
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
        scored = [r for r in results if r["label"] != "baseline"]
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
