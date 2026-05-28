"""Position A / B / C probe quality on the fine-tuned model.

Two distinct measurements:

1. Fresh per-layer linear probe — train a 2D distance-matching probe on the
   hidden states collected at each position × layer, report Spearman /
   Pearson / AdjAcc.

2. Training-time probe head — for the `probe` regime only, apply the FROZEN
   2D probe (loaded from probe.pt) to the same hidden states and report the
   same metrics. Tells us whether the head generalises beyond the supervised
   move-emission positions it saw during training.
"""

from __future__ import annotations

import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from compute_pearson_qwen_generation import build_distances  # noqa: E402
from eval.solve import build_prompt  # noqa: E402
from finetune.utils import (  # noqa: E402
    N_STATES,
    normalised_graph_distance_matrix,
    state_to_idx,
)

State = Tuple[int, ...]


_MOVE_LINE = re.compile(r"\bMove\s+(\d+)\s*:")
_SOLUTION_HEADER = re.compile(r"Solution\s*:", re.IGNORECASE)


# ── Hidden-state collection ───────────────────────────────────────────────────

def _state_after_k_moves(s_I: State, moves: Sequence[Sequence[int]], k: int) -> State:
    """Apply the first `k` 2-tuple moves against s_I (1-indexed disks internally)."""
    from hanoi_data.template import N_DISKS, state_to_pegs
    pegs = [list(p) for p in state_to_pegs(tuple(int(x) for x in s_I))]
    for i in range(min(k, len(moves))):
        m = moves[i]
        if not (isinstance(m, list) and len(m) == 2):
            break
        fp, tp = int(m[0]), int(m[1])
        if not (0 <= fp <= 2 and 0 <= tp <= 2) or not pegs[fp]:
            break
        disk = pegs[fp][-1]
        if pegs[tp] and pegs[tp][-1] < disk:
            break
        pegs[fp].pop()
        pegs[tp].append(disk)
    out = [0] * N_DISKS
    for p, peg_list in enumerate(pegs):
        for d in peg_list:
            out[d - 1] = p
    return tuple(out)


def _emitted_moves_from_text(text: str) -> List[List[int]]:
    """Extract intra-think emitted moves like 'moving to peg T' — but the cleanest
    signal is the final 'Solution: [[…]]' block which the template always emits.
    We use that as ground-truth for what the model actually committed to.
    """
    from planning import _extract_moves_block, _parse_moves_json
    m = _SOLUTION_HEADER.search(text)
    if m is None:
        return []
    after = text[m.end():]
    block = _extract_moves_block("moves = " + after)
    if block is None:
        return []
    try:
        moves = _parse_moves_json(block)
        return [[int(x) for x in mv] for mv in moves if len(mv) == 2]
    except Exception:
        return []


def _model_input_device(model) -> torch.device:
    return model.get_input_embeddings().weight.device


@torch.no_grad()
def _generate_then_forward(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    layers: Sequence[int],
) -> Tuple[str, List[Tuple[int, int]], Dict[int, torch.Tensor], int]:
    """Single-prompt wrapper that re-uses the batched implementation."""
    gen_texts, offsets, hidden, prompt_lens = _generate_then_forward_batched(
        model, tokenizer, [prompt], max_new_tokens, layers,
    )
    return gen_texts[0], offsets[0], hidden[0], prompt_lens[0]


@torch.no_grad()
def _generate_then_forward_batched(
    model,
    tokenizer,
    prompts: Sequence[str],
    max_new_tokens: int,
    layers: Sequence[int],
) -> Tuple[List[str],
           List[List[Tuple[int, int]]],
           List[Dict[int, torch.Tensor]],
           List[int]]:
    """Batched analogue of `_generate_then_forward`. Returns per-row lists.

    Strategy:
      - Left-pad prompts so they share a common prompt_len in the batched tensor.
      - Each row's "real" prompt_len is the count of attention-mask=1 in that row.
      - Generate the batch, then re-run a full forward with
        output_hidden_states=True on the same batched tensor.
      - For each row, slice out the non-pad portion of hidden states using the
        attention mask, so per-row indexing matches the un-padded re-tokenised
        completion offsets.
    """
    device = _model_input_device(model)
    saved_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "left"
    try:
        enc = tokenizer(list(prompts), return_tensors="pt", padding=True,
                        add_special_tokens=False)
    finally:
        tokenizer.padding_side = saved_side
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    pad_prompt_len = input_ids.shape[1]
    real_prompt_lens = attention_mask.sum(dim=1).tolist()

    gen = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )  # (B, pad_prompt_len + N_gen)
    full_ids = gen  # keep on device for the next forward
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    # Build attention mask for the full (prompt+gen) sequence — prompt padding
    # stays zeros; everything from real_prompt_len onward is real (generation
    # never produces pads on its own). For finished-early rows HF emits
    # pad/eos tokens; mask those out too for cleanliness.
    full_attn = (full_ids != pad_id).to(attention_mask.dtype)
    # But we want the original prompt's real positions kept exactly. Override:
    full_attn[:, :pad_prompt_len] = attention_mask

    # Second forward pass — get hidden states.
    out = model(
        input_ids=full_ids,
        attention_mask=full_attn,
        output_hidden_states=True,
        use_cache=False,
    )

    # Per-row outputs.
    gen_texts: List[str] = []
    per_row_offsets: List[List[Tuple[int, int]]] = []
    per_row_hidden: List[Dict[int, torch.Tensor]] = []
    per_row_prompt_len: List[int] = []
    for b in range(full_ids.shape[0]):
        # Strip the left-pad. After stripping, the row has:
        #   [real prompt tokens (length real_prompt_len[b])]
        #   [generated tokens (length n_new_total - pad_drop)]
        n_pad_left = pad_prompt_len - int(real_prompt_lens[b])
        row_ids = full_ids[b, n_pad_left:]
        # Generation tokens start at index real_prompt_len[b] in `row_ids`.
        gen_ids = full_ids[b, pad_prompt_len:].tolist()
        # Strip trailing pad tokens that HF may have produced after the row
        # hit EOS.
        while gen_ids and gen_ids[-1] == pad_id:
            gen_ids.pop()
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

        comp_enc = tokenizer(
            gen_text, add_special_tokens=False, return_offsets_mapping=True,
        )

        layer_to_hidden: Dict[int, torch.Tensor] = {}
        for L in layers:
            # hidden_states[L] is (B, T_full, H). Index [b, n_pad_left:] to
            # drop left-pad rows. T_keep = real_prompt_len + len(gen_ids).
            row_h = out.hidden_states[L][b, n_pad_left: n_pad_left + int(real_prompt_lens[b]) + len(gen_ids)]
            layer_to_hidden[L] = row_h.detach().to(torch.float32).cpu()

        gen_texts.append(gen_text)
        per_row_offsets.append(comp_enc["offset_mapping"])
        per_row_hidden.append(layer_to_hidden)
        per_row_prompt_len.append(int(real_prompt_lens[b]))

    return gen_texts, per_row_offsets, per_row_hidden, per_row_prompt_len


def _char_to_token(offset_mapping: List[Tuple[int, int]], char_pos: int) -> Optional[int]:
    """Last token whose [s, e) contains char_pos, with fallback to last
    token starting <= char_pos."""
    best: Optional[int] = None
    for i, (s, e) in enumerate(offset_mapping):
        if s == e:
            continue
        if s <= char_pos < e:
            best = i
            break
        if s <= char_pos:
            best = i
    return best


def _extract_positions(
    s_I: State,
    s_G: State,
    gen_text: str,
    comp_offsets: List[Tuple[int, int]],
    prompt_len_tokens: int,
    layer_hidden: torch.Tensor,
) -> Tuple[Optional[torch.Tensor],
           List[Tuple[torch.Tensor, State]],
           List[Tuple[torch.Tensor, State]]]:
    """Pull layer-K hidden vectors at Positions A / B / C.

    Returns (h_A, [(h_B, target_state)], [(h_C, target_state), …]). Any of
    these may be None / [] when the relevant landmark isn't found.
    """
    # Position A: last prompt token.
    h_A = layer_hidden[prompt_len_tokens - 1].clone()

    # Position B: token just before "Solution:".
    h_B_list: List[Tuple[torch.Tensor, State]] = []
    sol_match = _SOLUTION_HEADER.search(gen_text)
    if sol_match is not None:
        tok_idx = _char_to_token(comp_offsets, sol_match.start())
        if tok_idx is not None and tok_idx >= 1:
            # token-just-before is index (tok_idx - 1) in completion; in full seq
            # add prompt_len_tokens.
            full_idx = prompt_len_tokens + (tok_idx - 1)
            if 0 <= full_idx < layer_hidden.shape[0]:
                # State at Position B is the final s_G (model has done all its
                # internal reasoning and is about to emit the final move list).
                # We use s_G_t = s_I (model hasn't moved anything) as the
                # "starting" state? No — the model has committed mental moves
                # in CoT. Use the model's predicted final state derived from
                # its emitted Solution moves; if extraction fails, fall back to
                # s_I.
                emitted = _emitted_moves_from_text(gen_text)
                target = _state_after_k_moves(s_I, emitted, len(emitted)) if emitted else tuple(s_I)
                h_B_list.append((layer_hidden[full_idx].clone(), target))

    # Position C: token just before each "Move N:" header.
    h_C_list: List[Tuple[torch.Tensor, State]] = []
    emitted = _emitted_moves_from_text(gen_text)
    for m in _MOVE_LINE.finditer(gen_text):
        n_move = int(m.group(1))
        tok_idx = _char_to_token(comp_offsets, m.start())
        if tok_idx is None or tok_idx < 1:
            continue
        full_idx = prompt_len_tokens + (tok_idx - 1)
        if not (0 <= full_idx < layer_hidden.shape[0]):
            continue
        # The model is about to emit move N (1-indexed). The relevant true
        # state is the one BEFORE move N is applied, i.e. after (N-1) moves.
        true_state = _state_after_k_moves(s_I, emitted, n_move - 1)
        h_C_list.append((layer_hidden[full_idx].clone(), true_state))

    return h_A, h_B_list, h_C_list


# ── Probe training + metrics ─────────────────────────────────────────────────

def _train_distance_probe(
    x: torch.Tensor,            # (N, D)
    y_norm: torch.Tensor,       # (N, N) normalised pairwise distance target
    epochs: int = 2000,
    lr: float = 1e-3,
    device: str = "cpu",
) -> np.ndarray:
    x = x.to(device).float()
    y = y_norm.to(device).float()
    probe = nn.Linear(x.shape[1], 2, bias=True).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        z = probe(x)
        loss = torch.mean((torch.cdist(z, z, p=2) - y) ** 2)
        loss.backward()
        opt.step()
    with torch.no_grad():
        return probe(x).detach().cpu().numpy().astype(np.float64)


def _pairwise_metrics(coords: np.ndarray, true_dist: np.ndarray, nbrs: Dict[int, set]) -> Dict[str, float]:
    pred = np.linalg.norm(coords[:, None] - coords[None], axis=2)
    ui, uj = np.triu_indices(pred.shape[0], k=1)
    pred_p, true_p = pred[ui, uj], true_dist[ui, uj]
    rho, _ = spearmanr(pred_p, true_p)
    rho = float(rho) if np.isfinite(rho) else float("nan")
    pc = pred_p.astype(np.float64) - pred_p.mean()
    tc = true_p.astype(np.float64) - true_p.mean()
    denom = float(np.sqrt(np.sum(pc * pc) * np.sum(tc * tc)))
    pearson = float(np.sum(pc * tc) / denom) if denom > 0 else float("nan")
    correct = 0
    n = pred.shape[0]
    for i in range(n):
        row = pred[i].copy()
        row[i] = np.inf
        if int(np.argmin(row)) in nbrs.get(i, set()):
            correct += 1
    return {"spearman": rho, "pearson": pearson, "adj_acc": correct / max(n, 1)}


def _normalised_subdist(states: List[State], true_dist_full: np.ndarray,
                       norm_factor: float) -> np.ndarray:
    """Sub-matrix over the given list of states, normalised by `norm_factor`."""
    idxs = [state_to_idx(s) for s in states]
    return true_dist_full[np.ix_(idxs, idxs)] / norm_factor


# ── Top-level evaluation entry ────────────────────────────────────────────────

def run_probe_eval(
    model,
    tokenizer,
    probe_head: Optional[nn.Module],
    train_layer: int,
    probe_layers: Sequence[int],
    test_rows: List[Dict],
    max_new_tokens: int,
    out_dir: Path,
    sample_n: int = 81,
    batch_size: int = 8,
) -> Dict:
    """Collect hidden states across `sample_n` problems × per-layer, then train
    fresh per-layer probes and optionally apply the loaded probe head.

    `sample_n` defaults to 81 (one per s_I) which gives the cleanest probe
    fits; raise to use more problems if desired.
    """
    # Stratify by s_I so we cover all 81 starts.
    by_start: Dict[Tuple, List[Dict]] = defaultdict(list)
    for r in test_rows:
        by_start[tuple(r["s_I"])].append(r)
    chosen: List[Dict] = []
    for st, bucket in by_start.items():
        chosen.append(bucket[0])
    if len(chosen) < sample_n:
        for r in test_rows:
            if r not in chosen:
                chosen.append(r)
            if len(chosen) >= sample_n:
                break
    chosen = chosen[:sample_n]
    print(f"[probe_eval] collecting hidden states on {len(chosen)} problems "
          f"(layers={list(probe_layers)})")

    # layer -> position -> list of (hidden_vec, state)
    bucket: Dict[int, Dict[str, List[Tuple[torch.Tensor, State]]]] = {
        L: {"A": [], "B": [], "C": []} for L in probe_layers
    }

    bs = max(1, batch_size)
    i = 0
    while i < len(chosen):
        chunk = chosen[i: i + bs]
        prompts = [build_prompt(tokenizer, tuple(r["s_I"]), tuple(r["s_G"])) for r in chunk]
        gen_texts, comp_offs, layer_h_list, prompt_lens = _generate_then_forward_batched(
            model, tokenizer, prompts, max_new_tokens, probe_layers,
        )
        for j, row in enumerate(chunk):
            s_I = tuple(row["s_I"])
            s_G = tuple(row["s_G"])
            for L in probe_layers:
                h_A, h_B_list, h_C_list = _extract_positions(
                    s_I, s_G, gen_texts[j], comp_offs[j], prompt_lens[j], layer_h_list[j][L],
                )
                if h_A is not None:
                    bucket[L]["A"].append((h_A, s_I))
                bucket[L]["B"].extend(h_B_list)
                bucket[L]["C"].extend(h_C_list)
        i += bs
        print(f"  [probe_eval] {min(i, len(chosen)):4d}/{len(chosen)}")

    # graph-distance for the probe target
    norm_dist = normalised_graph_distance_matrix().numpy()
    raw_dist = (norm_dist * (norm_dist[~np.eye(N_STATES, dtype=bool)].std() or 1.0))
    # We need the per-bucket sub-matrix; build it from the same scale used at training.
    norm_factor = (raw_dist[~np.eye(N_STATES, dtype=bool)].std() or 1.0)

    nbrs = build_distances(  # reuse to get the adjacency dict for AdjAcc
        [tuple(int(x) for x in r["s_I"]) for r in test_rows[:1]], 4
    )[1]  # build_distances expects a state list, but its nbrs depend only on n_disks.
    # Cleaner: build directly from finetune.utils adjacency.
    from finetune.utils import build_graph_adjacency
    adj = build_graph_adjacency()

    fresh_results: Dict[str, Dict] = {}
    trained_head_results: Dict[str, Dict] = {}

    def _bucket_states_with_avg(items: List[Tuple[torch.Tensor, State]]
                                ) -> Tuple[torch.Tensor, List[State]]:
        """Group vectors by state, average within group → unique-state matrix."""
        groups: Dict[State, List[torch.Tensor]] = defaultdict(list)
        for h, st in items:
            groups[st].append(h)
        states = sorted(groups.keys(), key=state_to_idx)
        x = torch.stack([torch.stack(groups[s]).mean(0) for s in states])
        return x, states

    for L in probe_layers:
        for pos_name in ("A", "B", "C"):
            items = bucket[L][pos_name]
            if len(items) < 4:
                print(f"  [probe_eval] L={L} pos={pos_name}: only {len(items)} samples, skip")
                fresh_results[f"L{L}_{pos_name}"] = {
                    "n_states": len(items), "spearman": None, "pearson": None,
                    "adj_acc": None,
                }
                continue
            x, states = _bucket_states_with_avg(items)
            sub_dist = np.array([
                [float(adj_path_len(adj, a, b)) for b in states]
                for a in states
            ], dtype=np.float64)
            sub_norm = sub_dist / max(sub_dist[sub_dist > 0].std(), 1e-8)
            # Adjacency-acc neighbour sets (in the subset's index space).
            state_to_local = {s: i for i, s in enumerate(states)}
            local_nbrs = {
                i: {state_to_local[t] for t in adj[s] if t in state_to_local}
                for i, s in enumerate(states)
            }
            coords = _train_distance_probe(x, torch.tensor(sub_norm), device="cpu")
            metrics = _pairwise_metrics(coords, sub_dist, local_nbrs)
            metrics["n_states"] = len(states)
            fresh_results[f"L{L}_{pos_name}"] = metrics
            print(f"  [probe_eval/fresh]   L={L:2d}  pos={pos_name}  n={len(states):3d}  "
                  f"rho={metrics['spearman']:.4f}  r={metrics['pearson']:.4f}  "
                  f"adj={metrics['adj_acc']:.3f}")

            if probe_head is not None and L == train_layer:
                with torch.no_grad():
                    x_in = x.to(dtype=next(probe_head.parameters()).dtype,
                                device=next(probe_head.parameters()).device)
                    coords_trained = probe_head(x_in).detach().float().cpu().numpy().astype(np.float64)
                trained_metrics = _pairwise_metrics(coords_trained, sub_dist, local_nbrs)
                trained_metrics["n_states"] = len(states)
                trained_head_results[f"L{L}_{pos_name}"] = trained_metrics
                print(f"  [probe_eval/trained] L={L:2d}  pos={pos_name}  n={len(states):3d}  "
                      f"rho={trained_metrics['spearman']:.4f}  r={trained_metrics['pearson']:.4f}  "
                      f"adj={trained_metrics['adj_acc']:.3f}")

    out = {
        "n_problems": len(chosen),
        "probe_layers": list(probe_layers),
        "train_layer": train_layer,
        "fresh": fresh_results,
        "trained_head": trained_head_results,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "probe_results.json").write_text(json.dumps(out, indent=2))
    return out


def adj_path_len(adj: Dict[State, List[State]], a: State, b: State) -> int:
    """BFS path length from a to b in the adjacency graph (cached via finetune.utils)."""
    # We re-derive via normalised matrix to avoid an extra BFS implementation.
    # Use raw distance matrix (one-time build, cached as module-level singleton).
    return _RAW_DIST[state_to_idx(a), state_to_idx(b)]


# module-level cached raw distance matrix
from finetune.utils import graph_distance_matrix as _gdm
_RAW_DIST = _gdm().numpy()
