"""Shared utilities for ToH transformer scripts and analyses."""

from __future__ import annotations

import itertools
import json
from pathlib import Path
import random
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

from toh_transformer.config import default_model_hparams, max_seq_len_for_disks
from toh_transformer.data import Vocabulary
from toh_transformer.model import ToHTransformer

State = Tuple[int, ...]
Move = Tuple[int, int]

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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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
