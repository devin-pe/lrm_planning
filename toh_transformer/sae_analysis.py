#!/usr/bin/env python3
"""Sparse autoencoder analysis on move-token residual activations.

This script:
1) Loads CORRECT_OPTIMAL problems from evaluate.py output.
2) Collects residual-stream activations at move-token positions (and SEP2 references).
3) Trains sparse SAEs with L1 sweeps and optional TopK SAEs.
4) Filters to truly sparse models (L0 in [5, 50]) and selects the best by val recon.
5) Analyzes latent features, probes, and causal interventions.
6) Extracts 2D geometry from raw residuals vs SAE latents (SEP and move-token).
7) Saves JSON summaries and PNG figures.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mutual_info_score
from sklearn.model_selection import train_test_split

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

MOVE_TOKENS = ["M01", "M02", "M10", "M12", "M20", "M21"]


@dataclass
class SampleMeta:
    start: State
    goal: State
    step_index: int
    move_id: int
    move_token: str
    state_after: State
    prefix_ids_including_step: List[int]
    next_move_id: Optional[int]
    current_state_after_step: State


@dataclass
class NormalizationStats:
    mean: np.ndarray
    std: np.ndarray


@dataclass
class SAETrainResult:
    variant: str
    layer: int
    expansion: int
    d_hidden: int
    l1_lambda: Optional[float]
    topk_k: Optional[int]
    train_recon: float
    train_l1: float
    train_l0: float
    val_recon: float
    val_l1: float
    val_l0: float
    score: float
    model_path: Path


class SparseAutoencoder(nn.Module):
    def __init__(self, d_model: int, d_hidden: int):
        super().__init__()
        self.encoder = nn.Linear(d_model, d_hidden)
        self.decoder = nn.Linear(d_hidden, d_model)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x_centered = x - self.decoder.bias
        z = F.relu(self.encoder(x_centered))
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z


class TopKSparseAutoencoder(nn.Module):
    def __init__(self, d_model: int, d_hidden: int, k: int):
        super().__init__()
        self.encoder = nn.Linear(d_model, d_hidden)
        self.decoder = nn.Linear(d_hidden, d_model)
        self.k = int(k)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x_centered = x - self.decoder.bias
        z_pre = self.encoder(x_centered)
        topk_vals, topk_idx = torch.topk(z_pre, self.k, dim=-1)
        z = torch.zeros_like(z_pre)
        z.scatter_(-1, topk_idx, F.relu(topk_vals))
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SAE analysis on ToH residual stream move-token positions")
    parser.add_argument("--checkpoint", type=str, default="best.pt")
    parser.add_argument(
        "--eval_results",
        type=str,
        default="toh_transformer/eval_output/evaluation_results.json",
    )
    parser.add_argument("--n_disks", type=int, default=4)
    parser.add_argument("--layers", type=str, default="4,5")
    parser.add_argument("--output_dir", type=str, default="toh_transformer/sae_analysis_output")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--topk",
        action="store_true",
        help="Also train TopK SAEs (k in {5,10,20,30}, expansion fixed at 8x)",
    )
    return parser.parse_args()


def parse_layer_list(spec: str) -> List[int]:
    vals = []
    for p in spec.split(","):
        token = p.strip()
        if not token:
            continue
        vals.append(int(token))
    if not vals:
        raise ValueError("--layers cannot be empty")
    return list(dict.fromkeys(vals))


@torch.no_grad()
def extract_sep2_activations(
    model: ToHTransformer,
    states: Sequence[State],
    goal: State,
    vocab: Vocabulary,
    layers: Sequence[int],
    device: torch.device,
    batch_size: int = 128,
) -> Dict[int, np.ndarray]:
    rows = [build_context_ids(s, goal, vocab) for s in states]
    sep2_idx = len(rows[0]) - 1
    inp = torch.tensor(rows, dtype=torch.long, device=device)

    out: Dict[int, List[np.ndarray]] = {l: [] for l in layers}
    for st in range(0, len(rows), batch_size):
        batch = inp[st : st + batch_size]
        with model.capture_activations(layers=layers) as cache:
            _ = model(batch)
        for layer in layers:
            out[layer].append(cache[layer][:, sep2_idx, :].detach().cpu().numpy().astype(np.float32))

    return {layer: np.concatenate(chunks, axis=0) for layer, chunks in out.items()}


def collect_move_token_activations(
    model: ToHTransformer,
    problems: Sequence[Tuple[State, State]],
    vocab: Vocabulary,
    layers: Sequence[int],
    device: torch.device,
    n_disks: int,
) -> Tuple[Dict[int, np.ndarray], List[SampleMeta], Dict[str, int]]:
    context_len = 2 * n_disks + 3
    x_by_layer: Dict[int, List[np.ndarray]] = {l: [] for l in layers}
    metas: List[SampleMeta] = []

    kept_trajectories = 0
    skipped_illegal = 0

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
        valid_move_tokens: List[str] = []

        for tok, mid in zip(move_tokens, move_ids):
            mv = decode_move_token(tok)
            if mv is None:
                break
            nxt = apply_move_and_next_state(current, mv)
            if nxt is None:
                break
            inter_states.append(nxt)
            valid_move_ids.append(mid)
            valid_move_tokens.append(tok)
            current = nxt

        if not eos_seen or current != goal:
            skipped_illegal += 1
            continue

        full_seq = context_ids + valid_move_ids + [vocab.eos_id]
        if len(full_seq) > model.max_seq_len:
            skipped_illegal += 1
            continue

        seq_t = torch.tensor(full_seq, dtype=torch.long, device=device).unsqueeze(0)
        with torch.no_grad():
            with model.capture_activations(layers=layers) as cache:
                _ = model(seq_t)

        move_start = context_len
        move_end = context_len + len(valid_move_ids)

        for step_i, state_after in enumerate(inter_states):
            for layer in layers:
                vec = cache[layer][0, move_start:move_end, :][step_i].detach().cpu().numpy().astype(np.float32)
                x_by_layer[layer].append(vec)

            step = step_i + 1
            prefix_including_step = context_ids + valid_move_ids[:step]
            next_move_id = valid_move_ids[step] if step < len(valid_move_ids) else None
            metas.append(
                SampleMeta(
                    start=start,
                    goal=goal,
                    step_index=step,
                    move_id=valid_move_ids[step_i],
                    move_token=valid_move_tokens[step_i],
                    state_after=state_after,
                    prefix_ids_including_step=prefix_including_step,
                    next_move_id=next_move_id,
                    current_state_after_step=state_after,
                )
            )

        kept_trajectories += 1
        if (pi + 1) % 500 == 0:
            print(f"[INFO] Processed {pi + 1}/{len(problems)} problems, kept={kept_trajectories}")

    if not metas:
        raise RuntimeError("No move-token activations collected")

    stacked = {layer: np.stack(chunks, axis=0) for layer, chunks in x_by_layer.items()}
    stats = {
        "kept_trajectories": kept_trajectories,
        "skipped_trajectories": skipped_illegal,
        "num_samples": len(metas),
    }
    return stacked, metas, stats


def compute_normalization(x: np.ndarray) -> NormalizationStats:
    mean = x.mean(axis=0, keepdims=False).astype(np.float32)
    std = x.std(axis=0, keepdims=False).astype(np.float32)
    std = np.clip(std, 1e-6, None)
    return NormalizationStats(mean=mean, std=std)


def normalize_x(x: np.ndarray, stats: NormalizationStats) -> np.ndarray:
    return ((x - stats.mean) / stats.std).astype(np.float32)


def iterate_batches(x: np.ndarray, batch_size: int, rng: np.random.Generator) -> Sequence[np.ndarray]:
    idx = rng.permutation(x.shape[0])
    for st in range(0, x.shape[0], batch_size):
        yield idx[st : st + batch_size]


@torch.no_grad()
def evaluate_sae(sae: nn.Module, x: np.ndarray, device: torch.device) -> Tuple[float, float, float]:
    sae.eval()
    xb = torch.tensor(x, dtype=torch.float32, device=device)
    x_hat, z = sae(xb)
    recon = torch.mean((xb - x_hat) ** 2).item()
    l1 = torch.mean(torch.abs(z)).item()
    l0 = torch.mean((z > 0).float().sum(dim=1)).item()
    return float(recon), float(l1), float(l0)


def make_logreg(max_iter: int = 2000) -> LogisticRegression:
    try:
        return LogisticRegression(max_iter=max_iter, multi_class="multinomial", solver="lbfgs")
    except TypeError:
        return LogisticRegression(max_iter=max_iter)


def train_single_sae(
    x_train: np.ndarray,
    x_val: np.ndarray,
    d_model: int,
    d_hidden: int,
    l1_lambda: float,
    device: torch.device,
    seed: int,
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 3e-4,
) -> Tuple[SparseAutoencoder, Dict[str, float]]:
    sae = SparseAutoencoder(d_model=d_model, d_hidden=d_hidden).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    rng = np.random.default_rng(seed)

    x_train_t = x_train

    for ep in range(1, epochs + 1):
        sae.train()
        for bidx in iterate_batches(x_train_t, batch_size=batch_size, rng=rng):
            xb = torch.tensor(x_train_t[bidx], dtype=torch.float32, device=device)
            x_hat, z = sae(xb)
            recon = torch.mean((xb - x_hat) ** 2)
            l1 = torch.mean(torch.abs(z))
            loss = recon + l1_lambda * l1

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        if ep == 1 or ep % 20 == 0 or ep == epochs:
            tr_recon, tr_l1, tr_l0 = evaluate_sae(sae, x_train_t, device=device)
            va_recon, va_l1, va_l0 = evaluate_sae(sae, x_val, device=device)
            print(
                f"[SAE] ep={ep:3d} d_hidden={d_hidden:4d} lambda={l1_lambda:.1e} "
                f"train(recon={tr_recon:.6f}, l1={tr_l1:.6f}, l0={tr_l0:.2f}) "
                f"val(recon={va_recon:.6f}, l1={va_l1:.6f}, l0={va_l0:.2f})"
            )

    tr_recon, tr_l1, tr_l0 = evaluate_sae(sae, x_train_t, device=device)
    va_recon, va_l1, va_l0 = evaluate_sae(sae, x_val, device=device)
    metrics = {
        "train_recon": tr_recon,
        "train_l1": tr_l1,
        "train_l0": tr_l0,
        "val_recon": va_recon,
        "val_l1": va_l1,
        "val_l0": va_l0,
    }
    return sae, metrics


def train_single_topk_sae(
    x_train: np.ndarray,
    x_val: np.ndarray,
    d_model: int,
    d_hidden: int,
    k: int,
    device: torch.device,
    seed: int,
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 3e-4,
) -> Tuple[TopKSparseAutoencoder, Dict[str, float]]:
    sae = TopKSparseAutoencoder(d_model=d_model, d_hidden=d_hidden, k=k).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    rng = np.random.default_rng(seed)

    x_train_t = x_train

    for ep in range(1, epochs + 1):
        sae.train()
        for bidx in iterate_batches(x_train_t, batch_size=batch_size, rng=rng):
            xb = torch.tensor(x_train_t[bidx], dtype=torch.float32, device=device)
            x_hat, _ = sae(xb)
            recon = torch.mean((xb - x_hat) ** 2)

            opt.zero_grad(set_to_none=True)
            recon.backward()
            opt.step()

        if ep == 1 or ep % 20 == 0 or ep == epochs:
            tr_recon, tr_l1, tr_l0 = evaluate_sae(sae, x_train_t, device=device)
            va_recon, va_l1, va_l0 = evaluate_sae(sae, x_val, device=device)
            print(
                f"[TopK SAE] ep={ep:3d} d_hidden={d_hidden:4d} k={k:2d} "
                f"train(recon={tr_recon:.6f}, l1={tr_l1:.6f}, l0={tr_l0:.2f}) "
                f"val(recon={va_recon:.6f}, l1={va_l1:.6f}, l0={va_l0:.2f})"
            )

    tr_recon, tr_l1, tr_l0 = evaluate_sae(sae, x_train_t, device=device)
    va_recon, va_l1, va_l0 = evaluate_sae(sae, x_val, device=device)
    metrics = {
        "train_recon": tr_recon,
        "train_l1": tr_l1,
        "train_l0": tr_l0,
        "val_recon": va_recon,
        "val_l1": va_l1,
        "val_l0": va_l0,
    }
    return sae, metrics


def score_sae(val_recon: float, val_l0: float) -> float:
    lo, hi = 5.0, 30.0
    penalty = 0.0
    if val_l0 < lo:
        penalty += (lo - val_l0) * 0.02
    if val_l0 > hi:
        penalty += (val_l0 - hi) * 0.02
    return float(val_recon + penalty)


@torch.no_grad()
def extract_latents(sae: SparseAutoencoder, x_norm: np.ndarray, device: torch.device, chunk: int = 4096) -> np.ndarray:
    sae.eval()
    outs = []
    for st in range(0, x_norm.shape[0], chunk):
        xb = torch.tensor(x_norm[st : st + chunk], dtype=torch.float32, device=device)
        z = sae.encode(xb)
        outs.append(z.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(outs, axis=0)


def quantile_thresholds(z: np.ndarray, q_lo: float, q_hi: float) -> Tuple[np.ndarray, np.ndarray]:
    lo = np.quantile(z, q_lo, axis=0)
    hi = np.quantile(z, q_hi, axis=0)
    return lo.astype(np.float32), hi.astype(np.float32)


def safe_mode(values: np.ndarray) -> Tuple[int, float]:
    if values.size == 0:
        return -1, 0.0
    counts = np.bincount(values.astype(np.int64))
    mode = int(np.argmax(counts))
    concentration = float(counts[mode] / np.sum(counts))
    return mode, concentration


def build_feature_analysis(
    z: np.ndarray,
    metas: Sequence[SampleMeta],
    vocab: Vocabulary,
) -> Dict[str, object]:
    n_samples, n_features = z.shape

    disk_vals = np.array([m.state_after for m in metas], dtype=np.int64)
    move_vals = np.array([MOVE_TOKENS.index(m.move_token) for m in metas], dtype=np.int64)
    step_vals = np.array([m.step_index for m in metas], dtype=np.int64)

    feature_rows: List[Dict[str, object]] = []
    label_counts = Counter()

    for fi in range(n_features):
        zf = z[:, fi]
        active = zf > 0
        act_freq = float(np.mean(active))
        act_vals = zf[active]
        mean_active = float(np.mean(act_vals)) if act_vals.size > 0 else 0.0
        max_active = float(np.max(act_vals)) if act_vals.size > 0 else 0.0
        uninformative = bool(act_freq < 0.01 or act_freq > 0.95)

        q80 = float(np.quantile(zf, 0.8))
        top_mask = zf >= q80
        top_idx = np.where(top_mask)[0]

        best_label = "uninterpretable"
        best_score = -1.0
        best_meta: Dict[str, object] = {}

        # Disks 0..3
        for d in range(disk_vals.shape[1]):
            vals = disk_vals[top_idx, d] if top_idx.size > 0 else np.array([], dtype=np.int64)
            mode, conc = safe_mode(vals)
            mi = float(mutual_info_score((zf > 0).astype(np.int64), disk_vals[:, d]))
            if conc > best_score:
                best_score = conc
                best_label = f"disk_{d}_on_peg_{mode}"
                best_meta = {
                    "type": "disk_position",
                    "disk": int(d),
                    "value": int(mode),
                    "concentration": float(conc),
                    "mutual_info": mi,
                }

        # Move token selectivity
        vals_move = move_vals[top_idx] if top_idx.size > 0 else np.array([], dtype=np.int64)
        mode_move, conc_move = safe_mode(vals_move)
        mi_move = float(mutual_info_score((zf > 0).astype(np.int64), move_vals))
        if conc_move > best_score:
            best_score = conc_move
            move_token = MOVE_TOKENS[mode_move] if mode_move >= 0 else "UNKNOWN"
            best_label = f"move_{move_token}"
            best_meta = {
                "type": "move_token",
                "value": move_token,
                "concentration": float(conc_move),
                "mutual_info": mi_move,
            }

        # Step-index monotonicity/selectivity
        if np.std(zf) > 0 and np.std(step_vals) > 0:
            rho, _ = spearmanr(zf, step_vals)
            rho = float(rho) if np.isfinite(rho) else 0.0
        else:
            rho = 0.0

        top_steps = step_vals[top_idx] if top_idx.size > 0 else np.array([], dtype=np.int64)
        mode_step, conc_step = safe_mode(top_steps)
        if conc_step > best_score:
            best_score = conc_step
            best_label = f"step_{mode_step}"
            best_meta = {
                "type": "step_index",
                "value": int(mode_step),
                "concentration": float(conc_step),
                "spearman_rho": rho,
            }

        interpretable = bool((best_score >= 0.70) and (not uninformative))
        if not interpretable:
            best_label = "uninterpretable"
            best_meta = {"type": "uninterpretable", "concentration": float(best_score)}

        label_counts[str(best_meta.get("type", "uninterpretable"))] += 1

        feature_rows.append(
            {
                "feature": int(fi),
                "activation_frequency": act_freq,
                "mean_active": mean_active,
                "max_active": max_active,
                "uninformative": uninformative,
                "q80_threshold": q80,
                "interpretable": interpretable,
                "label": best_label,
                "label_meta": best_meta,
                "top_count": int(top_idx.size),
            }
        )

    interpretable_count = int(sum(1 for r in feature_rows if bool(r["interpretable"])))
    return {
        "num_features": int(n_features),
        "num_interpretable": interpretable_count,
        "num_uninterpretable": int(n_features - interpretable_count),
        "breakdown": dict(label_counts),
        "features": feature_rows,
    }


def fit_state_reconstruction(
    x_raw: np.ndarray,
    z: np.ndarray,
    metas: Sequence[SampleMeta],
    seed: int,
) -> Dict[str, object]:
    disk_vals = np.array([m.state_after for m in metas], dtype=np.int64)

    raw_acc = []
    sae_acc = []

    for d in range(disk_vals.shape[1]):
        y = disk_vals[:, d]
        x_train_raw, x_test_raw, y_train, y_test = train_test_split(
            x_raw,
            y,
            test_size=0.2,
            random_state=seed,
            stratify=y,
        )
        z_train, z_test, _, _ = train_test_split(
            z,
            y,
            test_size=0.2,
            random_state=seed,
            stratify=y,
        )

        
        raw_clf = make_logreg(max_iter=2000)
        sae_clf = make_logreg(max_iter=2000)

        raw_clf.fit(x_train_raw, y_train)
        sae_clf.fit(z_train, y_train)

        raw_acc.append(float(raw_clf.score(x_test_raw, y_test)))
        sae_acc.append(float(sae_clf.score(z_test, y_test)))

    chance = 1.0 / 3.0
    rows = []
    for d in range(disk_vals.shape[1]):
        rows.append(
            {
                "disk": d,
                "linear_probe_raw": raw_acc[d],
                "logistic_sae_z": sae_acc[d],
                "chance": chance,
            }
        )
    return {
        "rows": rows,
        "mean_linear_probe_raw": float(np.mean(raw_acc)),
        "mean_logistic_sae_z": float(np.mean(sae_acc)),
        "chance": chance,
    }


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
    return neighbors


def build_graph_and_distances(states: Sequence[State]) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
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
        raise RuntimeError("Found infinite graph distances")

    return dist, sorted(edge_set)


def train_distance_probe_full(
    x: np.ndarray,
    target_dist: np.ndarray,
    device: torch.device,
    epochs: int = 2000,
    lr: float = 1e-3,
) -> Tuple[nn.Linear, np.ndarray]:
    xb = torch.tensor(x, dtype=torch.float32, device=device)
    d = torch.tensor(target_dist, dtype=torch.float32, device=device)
    d_norm = (d - d.mean()) / d.std().clamp_min(1e-8)

    probe = nn.Linear(x.shape[1], 2, bias=True).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)

    for ep in range(1, epochs + 1):
        opt.zero_grad(set_to_none=True)
        p = probe(xb)
        pred_d = torch.cdist(p, p, p=2)
        loss = torch.mean((pred_d - d_norm) ** 2)
        loss.backward()
        opt.step()

        if ep == 1 or ep % 200 == 0 or ep == epochs:
            print(f"[2D probe full] epoch={ep:4d} loss={float(loss.item()):.6f}")

    with torch.no_grad():
        coords = probe(xb).detach().cpu().numpy().astype(np.float32)
    return probe, coords


def train_distance_probe_batched(
    x: np.ndarray,
    y_state_idx: np.ndarray,
    graph_dist: np.ndarray,
    device: torch.device,
    epochs: int = 2000,
    batch_size: int = 1024,
    lr: float = 1e-3,
    seed: int = 42,
) -> Tuple[nn.Linear, np.ndarray]:
    xb = torch.tensor(x, dtype=torch.float32, device=device)
    yb = torch.tensor(y_state_idx, dtype=torch.long, device=device)

    d_graph = torch.tensor(graph_dist, dtype=torch.float32, device=device)
    d_norm = (d_graph - d_graph.mean()) / d_graph.std().clamp_min(1e-8)

    probe = nn.Linear(x.shape[1], 2, bias=True).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)

    g = torch.Generator(device=device)
    g.manual_seed(seed)

    n = xb.shape[0]
    for ep in range(1, epochs + 1):
        idx = torch.randint(0, n, size=(min(batch_size, n),), generator=g, device=device)
        x_batch = xb[idx]
        y_batch = yb[idx]

        opt.zero_grad(set_to_none=True)
        p = probe(x_batch)
        pred_d = torch.cdist(p, p, p=2)
        target = d_norm[y_batch][:, y_batch]
        loss = torch.mean((pred_d - target) ** 2)
        loss.backward()
        opt.step()

        if ep == 1 or ep % 200 == 0 or ep == epochs:
            print(f"[2D probe batched] epoch={ep:4d} loss={float(loss.item()):.6f}")

    with torch.no_grad():
        coords = probe(xb).detach().cpu().numpy().astype(np.float32)
    return probe, coords


@torch.no_grad()
def apply_probe(x: np.ndarray, probe: nn.Linear, device: torch.device, chunk: int = 4096) -> np.ndarray:
    outs = []
    for st in range(0, x.shape[0], chunk):
        xb = torch.tensor(x[st : st + chunk], dtype=torch.float32, device=device)
        outs.append(probe(xb).detach().cpu().numpy().astype(np.float32))
    return np.concatenate(outs, axis=0)


def pairwise_spearman_from_coords(coords: np.ndarray, graph_dist: np.ndarray) -> float:
    rho, _ = pairwise_corr_from_coords(coords, graph_dist)
    return rho


def pairwise_corr_from_coords(coords: np.ndarray, graph_dist: np.ndarray) -> Tuple[float, float]:
    eu = cdist(coords, coords, metric="euclidean")
    iu = np.triu_indices(coords.shape[0], k=1)
    pred = eu[iu]
    true = graph_dist[iu]
    rho, _ = spearmanr(pred, true)
    rho = float(rho) if np.isfinite(rho) else 0.0
    pc = pred.astype(np.float64) - pred.mean()
    tc = true.astype(np.float64) - true.mean()
    denom = float(np.sqrt(np.sum(pc * pc) * np.sum(tc * tc)))
    pearson = float(np.sum(pc * tc) / denom) if denom > 0 else 0.0
    return rho, pearson


def nearest_state_accuracy(sample_coords: np.ndarray, sample_state_idx: np.ndarray, state_coords: np.ndarray) -> float:
    d = cdist(sample_coords, state_coords, metric="euclidean")
    pred = np.argmin(d, axis=1)
    return float(np.mean(pred == sample_state_idx))


def averaged_neighbor_metrics(coords: np.ndarray, graph_dist: np.ndarray) -> Dict[str, float]:
    n = coords.shape[0]
    if graph_dist.shape != (n, n):
        raise ValueError("graph_dist shape must match coords")

    d2 = cdist(coords, coords, metric="euclidean")
    top1_ok = 0
    top3_ok = 0
    neighbor_ranks: List[int] = []

    for i in range(n):
        order = np.argsort(d2[i])
        order = order[order != i]
        nbrs = set(np.where(graph_dist[i] == 1.0)[0].tolist())
        if not nbrs:
            continue

        if int(order[0]) in nbrs:
            top1_ok += 1
        if any(int(j) in nbrs for j in order[:3]):
            top3_ok += 1

        rank_by_state = {int(s): int(r + 1) for r, s in enumerate(order)}
        for nb in nbrs:
            neighbor_ranks.append(rank_by_state[int(nb)])

    if not neighbor_ranks:
        return {
            "top1_neighbor_adjacent": 0.0,
            "top3_neighbor_adjacent": 0.0,
            "mean_neighbor_rank": float("inf"),
        }

    return {
        "top1_neighbor_adjacent": float(top1_ok / n),
        "top3_neighbor_adjacent": float(top3_ok / n),
        "mean_neighbor_rank": float(np.mean(np.array(neighbor_ranks, dtype=np.float64))),
    }


def average_state_coords(
    sample_coords: np.ndarray,
    sample_state_idx: np.ndarray,
    n_states: int,
) -> Tuple[np.ndarray, np.ndarray]:
    sums = np.zeros((n_states, 2), dtype=np.float64)
    counts = np.zeros(n_states, dtype=np.int64)
    for i in range(sample_coords.shape[0]):
        s = int(sample_state_idx[i])
        sums[s] += sample_coords[i]
        counts[s] += 1
    out = np.full((n_states, 2), np.nan, dtype=np.float32)
    valid = counts > 0
    out[valid] = (sums[valid] / counts[valid, None]).astype(np.float32)
    return out, counts


def average_vectors_by_state(
    x: np.ndarray,
    state_idx: np.ndarray,
    n_states: int,
) -> Tuple[np.ndarray, np.ndarray]:
    sums = np.zeros((n_states, x.shape[1]), dtype=np.float64)
    counts = np.zeros(n_states, dtype=np.int64)
    for i in range(x.shape[0]):
        s = int(state_idx[i])
        sums[s] += x[i]
        counts[s] += 1

    out = np.zeros((n_states, x.shape[1]), dtype=np.float32)
    valid = counts > 0
    out[valid] = (sums[valid] / counts[valid, None]).astype(np.float32)
    return out, counts


def plot_labeled_sierpinski_layout(
    states: Sequence[State],
    edges: Sequence[Tuple[int, int]],
    coords: np.ndarray,
    title: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)

    for i, j in edges:
        ax.plot(
            [coords[i, 0], coords[j, 0]],
            [coords[i, 1], coords[j, 1]],
            color="gray",
            linewidth=0.8,
            alpha=0.5,
            zorder=1,
        )

    largest_disk_peg = np.array([s[len(s) - 1] for s in states], dtype=np.int64)
    peg_colors = {0: "red", 1: "blue", 2: "green"}
    for peg in [0, 1, 2]:
        mask = largest_disk_peg == peg
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=80,
            alpha=0.95,
            color=peg_colors[peg],
            zorder=2,
            label=f"largest disk on peg {peg}",
        )

    for i, st in enumerate(states):
        st_label = "".join(str(v) for v in st)
        ax.text(
            float(coords[i, 0]),
            float(coords[i, 1]),
            st_label,
            fontsize=6,
            ha="center",
            va="center",
            zorder=3,
            bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
        )

    ax.set_title(title)
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_move_layout_side_by_side(
    states: Sequence[State],
    edges: Sequence[Tuple[int, int]],
    raw_state_coords: np.ndarray,
    sae_state_coords: np.ndarray,
    raw_move_coords: np.ndarray,
    sae_move_coords: np.ndarray,
    y_state_idx: np.ndarray,
    out_path: Path,
    max_points: int = 12000,
    seed: int = 42,
) -> None:
    rng = np.random.default_rng(seed)
    idx_all = np.arange(raw_move_coords.shape[0])
    if idx_all.size > max_points:
        idx = rng.choice(idx_all, size=max_points, replace=False)
    else:
        idx = idx_all

    largest_disk_peg = np.array([states[int(s)][len(states[int(s)]) - 1] for s in y_state_idx[idx]], dtype=np.int64)
    cmap = plt.get_cmap("Set1")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    payload = [
        (axes[0], raw_state_coords, raw_move_coords[idx], "Move-Token 2D (Raw residual)"),
        (axes[1], sae_state_coords, sae_move_coords[idx], "Move-Token 2D (SAE latent)"),
    ]

    for ax, state_coords, move_coords, title in payload:
        for i, j in edges:
            ax.plot(
                [state_coords[i, 0], state_coords[j, 0]],
                [state_coords[i, 1], state_coords[j, 1]],
                color="gray",
                linewidth=0.6,
                alpha=0.35,
                zorder=1,
            )
        for peg in [0, 1, 2]:
            mask = largest_disk_peg == peg
            ax.scatter(
                move_coords[mask, 0],
                move_coords[mask, 1],
                s=8,
                alpha=0.25,
                color=cmap(peg),
                zorder=2,
            )
        ax.scatter(state_coords[:, 0], state_coords[:, 1], s=26, color="black", alpha=0.85, zorder=3)
        ax.set_title(title)
        ax.grid(alpha=0.2)

    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_l0_recon_tradeoff(training_rows: Sequence[Dict[str, object]], out_path: Path) -> None:
    l1_rows = [r for r in training_rows if str(r.get("variant")) == "l1"]
    if not l1_rows:
        return

    expansions = sorted(set(int(r["expansion"]) for r in l1_rows))
    fig, axes = plt.subplots(1, len(expansions), figsize=(5.5 * len(expansions), 4.8), constrained_layout=True)
    if len(expansions) == 1:
        axes = [axes]

    colors = {4: "#1f77b4", 5: "#ff7f0e", 6: "#2ca02c"}
    for ax, exp in zip(axes, expansions):
        rows = [r for r in l1_rows if int(r["expansion"]) == exp]
        layers = sorted(set(int(r["layer"]) for r in rows))
        for layer in layers:
            lr = [r for r in rows if int(r["layer"]) == layer]
            xs = [float(r["val_l0"]) for r in lr]
            ys = [float(r["val_recon"]) for r in lr]
            ax.plot(xs, ys, marker="o", linewidth=1.5, color=colors.get(layer, None), label=f"layer {layer}")
            for r in lr:
                ax.annotate(f"{float(r['lambda']):.2g}", (float(r["val_l0"]), float(r["val_recon"])), fontsize=8)
        ax.axvline(5.0, color="gray", linestyle="--", linewidth=1.0)
        ax.axvline(50.0, color="gray", linestyle="--", linewidth=1.0)
        ax.set_title(f"Expansion {exp}x")
        ax.set_xlabel("Validation L0")
        ax.set_ylabel("Validation Reconstruction MSE")
        ax.grid(alpha=0.2)
        ax.legend(loc="best")

    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def forward_with_position_patch(
    model: ToHTransformer,
    input_ids: torch.Tensor,
    target_layer: int,
    patch_position: int,
    patched_residual: torch.Tensor,
) -> torch.Tensor:
    bsz, seq_len = input_ids.shape
    if seq_len > model.max_seq_len:
        raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len={model.max_seq_len}")

    if patched_residual.dim() == 1:
        donor = patched_residual.unsqueeze(0).expand(bsz, -1)
    elif patched_residual.dim() == 2 and patched_residual.size(0) == bsz:
        donor = patched_residual
    else:
        raise ValueError("patched_residual must be shape [d_model] or [batch, d_model]")

    positions = torch.arange(0, seq_len, device=input_ids.device, dtype=torch.long)
    x = model.token_emb(input_ids) + model.pos_emb(positions).unsqueeze(0)
    x = model.emb_dropout(x)

    for li, block in enumerate(model.blocks, start=1):
        x = block(x)
        if li == target_layer:
            x = x.clone()
            x[:, patch_position, :] = donor

    x = model.ln_f(x)
    logits = model.lm_head(x)
    return logits


@torch.no_grad()
def decode_from_prefix(
    model: ToHTransformer,
    prefix_ids: Sequence[int],
    eos_id: int,
    device: torch.device,
) -> List[int]:
    seq = torch.tensor(prefix_ids, dtype=torch.long, device=device).unsqueeze(0)
    out = []
    while seq.size(1) < model.max_seq_len:
        logits = model(seq)
        next_id = int(torch.argmax(logits[:, -1, :], dim=-1).item())
        out.append(next_id)
        seq = torch.cat([seq, torch.tensor([[next_id]], dtype=torch.long, device=device)], dim=1)
        if next_id == eos_id:
            break
    return out


@torch.no_grad()
def intervene_next_token_and_rollout(
    model: ToHTransformer,
    prefix_ids: Sequence[int],
    target_layer: int,
    patch_position: int,
    patched_residual: np.ndarray,
    eos_id: int,
    device: torch.device,
) -> Tuple[int, List[int]]:
    seq = torch.tensor(prefix_ids, dtype=torch.long, device=device).unsqueeze(0)
    patch_vec = torch.tensor(patched_residual, dtype=torch.float32, device=device)

    logits = forward_with_position_patch(
        model=model,
        input_ids=seq,
        target_layer=target_layer,
        patch_position=patch_position,
        patched_residual=patch_vec,
    )
    next_id = int(torch.argmax(logits[:, -1, :], dim=-1).item())

    seq2 = torch.cat([seq, torch.tensor([[next_id]], dtype=torch.long, device=device)], dim=1)
    rest = decode_from_prefix(model, seq2.squeeze(0).tolist(), eos_id=eos_id, device=device)
    return next_id, [next_id] + rest


def simulate_suffix(
    start_state: State,
    generated_ids: Sequence[int],
    goal: State,
    vocab: Vocabulary,
) -> Dict[str, object]:
    toks = [vocab.itos[i] for i in generated_ids]
    if vocab.eos_id in generated_ids:
        eos_pos = generated_ids.index(vocab.eos_id)
        move_tokens = toks[:eos_pos]
        eos_seen = True
    else:
        move_tokens = toks
        eos_seen = False

    cur = start_state
    legal = True
    for tok in move_tokens:
        mv = decode_move_token(tok)
        if mv is None:
            legal = False
            break
        nxt = apply_move_and_next_state(cur, mv)
        if nxt is None:
            legal = False
            break
        cur = nxt

    reaches_goal = bool(legal and eos_seen and cur == goal)
    return {
        "legal": bool(legal),
        "eos_seen": bool(eos_seen),
        "reaches_goal": reaches_goal,
        "final_state": list(cur),
        "moves": move_tokens,
    }


def make_feature_patch(
    sae: SparseAutoencoder,
    x_raw: np.ndarray,
    norm: NormalizationStats,
    feature_idx: int,
    target_value: float,
    device: torch.device,
) -> np.ndarray:
    x_norm = normalize_x(x_raw[None, :], norm)
    xb = torch.tensor(x_norm, dtype=torch.float32, device=device)
    with torch.no_grad():
        z = sae.encode(xb)
        z_mod = z.clone()
        z_mod[:, feature_idx] = float(target_value)
        x_mod_norm = sae.decode(z_mod)
    x_mod = x_mod_norm.detach().cpu().numpy()[0] * norm.std + norm.mean
    return x_mod.astype(np.float32)


def run_causal_interventions(
    model: ToHTransformer,
    vocab: Vocabulary,
    layer: int,
    sae: SparseAutoencoder,
    norm: NormalizationStats,
    x_raw: np.ndarray,
    z: np.ndarray,
    metas: Sequence[SampleMeta],
    feature_rows: Sequence[Dict[str, object]],
    seed: int,
    top_k_features: int = 20,
    n_examples_per_feature: int = 50,
) -> Dict[str, object]:
    rng = np.random.default_rng(seed)

    with_next = np.array([m.next_move_id is not None for m in metas], dtype=bool)
    next_move_vals = np.array([m.next_move_id if m.next_move_id is not None else -1 for m in metas], dtype=np.int64)

    # Rank interpretable features by label concentration.
    candidates = [r for r in feature_rows if bool(r["interpretable"])]
    candidates = sorted(
        candidates,
        key=lambda r: float(r.get("label_meta", {}).get("concentration", 0.0)),
        reverse=True,
    )
    top_features = candidates[:top_k_features]

    if not top_features:
        return {
            "top_features": [],
            "activation_success_rate": 0.0,
            "suppression_success_rate": 0.0,
            "disruption_rate": 0.0,
            "examples": [],
        }

    q20, q80 = quantile_thresholds(z, q_lo=0.2, q_hi=0.8)
    q90 = np.quantile(z, 0.9, axis=0).astype(np.float32)

    activation_success = 0
    activation_total = 0
    suppression_success = 0
    suppression_total = 0
    disruptions = 0
    disruptions_total = 0

    feature_reports: List[Dict[str, object]] = []
    intervention_examples: List[Dict[str, object]] = []

    for feat in top_features:
        fi = int(feat["feature"])
        zf = z[:, fi]

        # Preferred next move in feature's high-activation regime.
        top_mask = (zf >= q80[fi]) & with_next & (next_move_vals >= 0)
        preferred_next = -1
        preferred_conc = 0.0
        if np.any(top_mask):
            vals = next_move_vals[top_mask]
            mode, conc = safe_mode(vals)
            preferred_next = mode
            preferred_conc = conc

        active_pool = np.where((zf >= q80[fi]) & with_next)[0]
        inactive_pool = np.where((zf <= q20[fi]) & with_next)[0]

        if active_pool.size == 0 or inactive_pool.size == 0:
            continue

        chosen_on = rng.choice(inactive_pool, size=min(n_examples_per_feature, inactive_pool.size), replace=False)
        chosen_off = rng.choice(active_pool, size=min(n_examples_per_feature, active_pool.size), replace=False)

        feature_on_success = 0
        feature_on_total = 0
        feature_off_success = 0
        feature_off_total = 0
        feature_disrupt = 0
        feature_disrupt_total = 0

        # Activation intervention: force feature ON from inactive states.
        for idx in chosen_on:
            meta = metas[int(idx)]
            if meta.next_move_id is None:
                continue

            patch_pos = len(meta.prefix_ids_including_step) - 1
            patched_resid = make_feature_patch(
                sae=sae,
                x_raw=x_raw[int(idx)],
                norm=norm,
                feature_idx=fi,
                target_value=float(q90[fi]),
                device=next(model.parameters()).device,
            )

            clean_next = int(meta.next_move_id)
            patched_next, rollout = intervene_next_token_and_rollout(
                model=model,
                prefix_ids=meta.prefix_ids_including_step,
                target_layer=layer,
                patch_position=patch_pos,
                patched_residual=patched_resid,
                eos_id=vocab.eos_id,
                device=next(model.parameters()).device,
            )
            suffix_eval = simulate_suffix(
                start_state=meta.current_state_after_step,
                generated_ids=rollout,
                goal=meta.goal,
                vocab=vocab,
            )

            feature_on_total += 1
            activation_total += 1

            if preferred_next >= 0 and patched_next != clean_next and patched_next == preferred_next:
                feature_on_success += 1
                activation_success += 1

            feature_disrupt_total += 1
            disruptions_total += 1
            if (not bool(suffix_eval["legal"])) or (not bool(suffix_eval["reaches_goal"])):
                feature_disrupt += 1
                disruptions += 1

            if len(intervention_examples) < 5:
                intervention_examples.append(
                    {
                        "feature": fi,
                        "mode": "activation",
                        "start": list(meta.start),
                        "goal": list(meta.goal),
                        "step_index": int(meta.step_index),
                        "clean_next": vocab.itos[clean_next],
                        "patched_next": vocab.itos[patched_next],
                        "suffix_legal": bool(suffix_eval["legal"]),
                        "suffix_reaches_goal": bool(suffix_eval["reaches_goal"]),
                        "patched_moves": list(suffix_eval["moves"]),
                    }
                )

        # Suppression intervention: force feature OFF from active states.
        for idx in chosen_off:
            meta = metas[int(idx)]
            if meta.next_move_id is None:
                continue

            patch_pos = len(meta.prefix_ids_including_step) - 1
            patched_resid = make_feature_patch(
                sae=sae,
                x_raw=x_raw[int(idx)],
                norm=norm,
                feature_idx=fi,
                target_value=0.0,
                device=next(model.parameters()).device,
            )

            clean_next = int(meta.next_move_id)
            patched_next, rollout = intervene_next_token_and_rollout(
                model=model,
                prefix_ids=meta.prefix_ids_including_step,
                target_layer=layer,
                patch_position=patch_pos,
                patched_residual=patched_resid,
                eos_id=vocab.eos_id,
                device=next(model.parameters()).device,
            )
            suffix_eval = simulate_suffix(
                start_state=meta.current_state_after_step,
                generated_ids=rollout,
                goal=meta.goal,
                vocab=vocab,
            )

            feature_off_total += 1
            suppression_total += 1
            if preferred_next >= 0 and clean_next == preferred_next and patched_next != clean_next:
                feature_off_success += 1
                suppression_success += 1

            feature_disrupt_total += 1
            disruptions_total += 1
            if (not bool(suffix_eval["legal"])) or (not bool(suffix_eval["reaches_goal"])):
                feature_disrupt += 1
                disruptions += 1

            if len(intervention_examples) < 5:
                intervention_examples.append(
                    {
                        "feature": fi,
                        "mode": "suppression",
                        "start": list(meta.start),
                        "goal": list(meta.goal),
                        "step_index": int(meta.step_index),
                        "clean_next": vocab.itos[clean_next],
                        "patched_next": vocab.itos[patched_next],
                        "suffix_legal": bool(suffix_eval["legal"]),
                        "suffix_reaches_goal": bool(suffix_eval["reaches_goal"]),
                        "patched_moves": list(suffix_eval["moves"]),
                    }
                )

        if preferred_next >= 0 and preferred_next < len(vocab.itos):
            preferred_next_token = vocab.itos[preferred_next]
        else:
            preferred_next_token = None

        feature_reports.append(
            {
                "feature": fi,
                "label": feat["label"],
                "label_meta": feat["label_meta"],
                "preferred_next_move": preferred_next_token,
                "preferred_next_concentration": preferred_conc,
                "activation_success": feature_on_success,
                "activation_total": feature_on_total,
                "suppression_success": feature_off_success,
                "suppression_total": feature_off_total,
                "disruption_count": feature_disrupt,
                "disruption_total": feature_disrupt_total,
            }
        )

    return {
        "top_features": feature_reports,
        "activation_success_rate": float(activation_success / activation_total) if activation_total > 0 else 0.0,
        "suppression_success_rate": float(suppression_success / suppression_total) if suppression_total > 0 else 0.0,
        "disruption_rate": float(disruptions / disruptions_total) if disruptions_total > 0 else 0.0,
        "activation_trials": int(activation_total),
        "suppression_trials": int(suppression_total),
        "disruption_trials": int(disruptions_total),
        "examples": intervention_examples,
    }


def save_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def plot_activation_frequency_hist(feature_rows: Sequence[Dict[str, object]], out_path: Path) -> None:
    freqs = np.array([float(r["activation_frequency"]) for r in feature_rows], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(freqs, bins=30, color="#2a9d8f", edgecolor="black", alpha=0.85)
    ax.set_title("Feature Activation Frequency Histogram")
    ax.set_xlabel("Fraction of samples with z_i > 0")
    ax.set_ylabel("Feature count")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_top_feature_bars(
    z: np.ndarray,
    metas: Sequence[SampleMeta],
    feature_rows: Sequence[Dict[str, object]],
    out_path: Path,
    top_k: int = 10,
) -> None:
    disk_vals = np.array([m.state_after for m in metas], dtype=np.int64)

    ranked = [r for r in feature_rows if bool(r["interpretable"])]
    ranked = sorted(
        ranked,
        key=lambda r: float(r.get("label_meta", {}).get("concentration", 0.0)),
        reverse=True,
    )[:top_k]

    if not ranked:
        return

    n_cols = 5
    n_rows = int(math.ceil(len(ranked) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.8 * n_cols, 3.2 * n_rows))
    axes = np.array(axes).reshape(-1)

    for i, row in enumerate(ranked):
        ax = axes[i]
        fi = int(row["feature"])
        meta = row["label_meta"]
        label = str(row["label"])

        if meta.get("type") == "disk_position":
            disk = int(meta["disk"])
            weights = z[:, fi]
            bins = [
                float(np.mean(weights[disk_vals[:, disk] == peg])) if np.any(disk_vals[:, disk] == peg) else 0.0
                for peg in [0, 1, 2]
            ]
            ax.bar(["peg0", "peg1", "peg2"], bins, color="#457b9d")
            ax.set_title(f"f{fi}: {label}")
            ax.set_ylabel("mean z")
        else:
            # Fallback: show top-20% vs rest activation mean.
            zf = z[:, fi]
            q80 = np.quantile(zf, 0.8)
            hi = zf[zf >= q80]
            lo = zf[zf < q80]
            ax.bar(["bottom80", "top20"], [float(np.mean(lo)), float(np.mean(hi))], color="#e76f51")
            ax.set_title(f"f{fi}: {label}")
            ax.set_ylabel("mean z")

        ax.grid(alpha=0.2)

    for j in range(len(ranked), len(axes)):
        axes[j].axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_causal_examples(examples: Sequence[Dict[str, object]], out_path: Path) -> None:
    if not examples:
        return

    n = min(len(examples), 5)
    fig, axes = plt.subplots(n, 1, figsize=(13, 2.5 * n))
    if n == 1:
        axes = [axes]

    for i in range(n):
        ex = examples[i]
        ax = axes[i]
        ax.axis("off")
        txt = (
            f"Feature {ex['feature']} ({ex['mode']}) | step={ex['step_index']} | "
            f"start={ex['start']} goal={ex['goal']}\n"
            f"clean next={ex['clean_next']} -> patched next={ex['patched_next']} | "
            f"legal={ex['suffix_legal']} reaches_goal={ex['suffix_reaches_goal']}\n"
            f"patched suffix: {' '.join(ex['patched_moves'])}"
        )
        ax.text(0.01, 0.65, txt, fontsize=10, va="center", ha="left", family="monospace")

    fig.suptitle("Causal Intervention Examples", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def print_state_table(state_rec: Dict[str, object]) -> None:
    rows = state_rec["rows"]
    print("\nMethod comparison: state reconstruction")
    print("Method              | Disk 0 Acc | Disk 1 Acc | Disk 2 Acc | Disk 3 Acc | Mean")
    print("--------------------+------------+------------+------------+------------+------")

    raw_vals = [float(r["linear_probe_raw"]) for r in rows]
    sae_vals = [float(r["logistic_sae_z"]) for r in rows]
    chance = float(state_rec["chance"])

    print(
        "Linear probe (raw)  | "
        + " | ".join([f"{100.0 * v:8.2f}%" for v in raw_vals])
        + f" | {100.0 * float(np.mean(raw_vals)):5.2f}%"
    )
    print(
        "Logistic (SAE z)    | "
        + " | ".join([f"{100.0 * v:8.2f}%" for v in sae_vals])
        + f" | {100.0 * float(np.mean(sae_vals)):5.2f}%"
    )
    print(
        "Chance (1/3)        | "
        + " | ".join([f"{100.0 * chance:8.2f}%" for _ in range(4)])
        + f" | {100.0 * chance:5.2f}%"
    )


def print_2d_comparison_table(rows: Sequence[Dict[str, object]]) -> None:
    print("\n2D probe quality comparison")
    print("Input to 2D probe    | Nearest-State Acc | Spearman rho")
    print("---------------------+-------------------+-------------")
    for r in rows:
        print(
            f"{str(r['input']):21s} | {100.0 * float(r['nearest_state_acc']):16.2f}% | {float(r['spearman_rho']):11.4f}"
        )


def print_move_metric_matrix(
    layers: Sequence[int],
    move_metric_summary: Dict[str, Dict[str, Dict[str, float]]],
) -> None:
    if len(layers) < 2:
        return

    l1, l2 = int(layers[0]), int(layers[1])

    def get(layer: int, variant: str, key: str) -> float:
        return float(move_metric_summary[str(layer)][variant][key])

    print("\nMove-token metric summary (raw vs SAE)")
    print(f"Metric                          | Layer {l1} Raw | Layer {l1} SAE | Layer {l2} Raw | Layer {l2} SAE")
    print("--------------------------------+-------------+-------------+-------------+------------")
    print(
        "Averaged top-1 neighbor correct | "
        f"{100.0 * get(l1, 'raw', 'avg_top1_neighbor_adj'):11.2f}% | "
        f"{100.0 * get(l1, 'sae', 'avg_top1_neighbor_adj'):11.2f}% | "
        f"{100.0 * get(l2, 'raw', 'avg_top1_neighbor_adj'):11.2f}% | "
        f"{100.0 * get(l2, 'sae', 'avg_top1_neighbor_adj'):10.2f}%"
    )
    print(
        "Averaged top-3 neighbor correct | "
        f"{100.0 * get(l1, 'raw', 'avg_top3_neighbor_adj'):11.2f}% | "
        f"{100.0 * get(l1, 'sae', 'avg_top3_neighbor_adj'):11.2f}% | "
        f"{100.0 * get(l2, 'raw', 'avg_top3_neighbor_adj'):11.2f}% | "
        f"{100.0 * get(l2, 'sae', 'avg_top3_neighbor_adj'):10.2f}%"
    )
    print(
        "Averaged mean neighbor rank     | "
        f"{get(l1, 'raw', 'avg_mean_neighbor_rank'):11.2f} | "
        f"{get(l1, 'sae', 'avg_mean_neighbor_rank'):11.2f} | "
        f"{get(l2, 'raw', 'avg_mean_neighbor_rank'):11.2f} | "
        f"{get(l2, 'sae', 'avg_mean_neighbor_rank'):10.2f}"
    )
    print(
        "Per-sample nearest-state acc    | "
        f"{100.0 * get(l1, 'raw', 'per_sample_acc'):11.2f}% | "
        f"{100.0 * get(l1, 'sae', 'per_sample_acc'):11.2f}% | "
        f"{100.0 * get(l2, 'raw', 'per_sample_acc'):11.2f}% | "
        f"{100.0 * get(l2, 'sae', 'per_sample_acc'):10.2f}%"
    )
    print(
        "Spearman rho                    | "
        f"{get(l1, 'raw', 'spearman_rho'):11.4f} | "
        f"{get(l1, 'sae', 'spearman_rho'):11.4f} | "
        f"{get(l2, 'raw', 'spearman_rho'):11.4f} | "
        f"{get(l2, 'sae', 'spearman_rho'):10.4f}"
    )


def print_space_diagnostic_matrix(
    layers: Sequence[int],
    move_metric_summary: Dict[str, Dict[str, Dict[str, float]]],
) -> None:
    print("\nSpace diagnostic (move-token queries vs 81 averaged refs)")
    for layer in layers:
        layer_key = str(layer)
        raw = move_metric_summary[layer_key]["raw"]
        sae = move_metric_summary[layer_key]["sae"]

        print(f"\nLayer {layer}")
        print("Space                | Nearest-state exact match")
        print("---------------------+-------------------------------------------")
        print(f"Full d_model (raw)   | {100.0 * float(raw['full_space_sample_acc']):9.2f}%")
        print(f"Full d_hidden (SAE)  | {100.0 * float(sae['full_space_sample_acc']):9.2f}%")
        print(f"2D projected (raw)   | {100.0 * float(raw['per_sample_acc']):9.2f}%")
        print(f"2D projected (SAE)   | {100.0 * float(sae['per_sample_acc']):9.2f}%")


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
    models_dir = out_dir / "sae_models"
    models_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("Requested cuda but CUDA is not available")

    layers = parse_layer_list(args.layers)
    vocab = Vocabulary()
    confirm_tokenizer_mapping(vocab)

    ckpt_path = resolve_checkpoint_path(args.checkpoint, args.n_disks)
    model = load_model(ckpt_path, args.n_disks, device)

    for layer in layers:
        if layer < 1 or layer > model.n_layers:
            raise ValueError(f"Requested layer {layer} out of range [1, {model.n_layers}]")

    eval_path = Path(args.eval_results)
    if not eval_path.exists():
        raise FileNotFoundError(f"Evaluation results not found: {eval_path}")

    problems = load_correct_optimal_problems(eval_path, args.n_disks)
    if not problems:
        raise RuntimeError("No CORRECT_OPTIMAL problems found")

    print(f"[INFO] Loaded model: {ckpt_path}")
    print(f"[INFO] CORRECT_OPTIMAL problems: {len(problems)}")
    print(f"[INFO] Target layers for SAE: {layers}")

    x_by_layer, metas, collection_stats = collect_move_token_activations(
        model=model,
        problems=problems,
        vocab=vocab,
        layers=layers,
        device=device,
        n_disks=args.n_disks,
    )
    print(
        f"[INFO] Collected {collection_stats['num_samples']} move-token samples "
        f"from {collection_stats['kept_trajectories']} trajectories"
    )

    goal = tuple(2 for _ in range(args.n_disks))
    all_states = enumerate_states(args.n_disks)
    sep_acts = extract_sep2_activations(
        model=model,
        states=all_states,
        goal=goal,
        vocab=vocab,
        layers=layers,
        device=device,
    )
    print(f"[INFO] Collected {len(all_states)} SEP-position activations per layer")

    expansions = [4, 8, 16]
    lambdas = [0.03, 0.1, 0.3, 1.0]
    topk_values = [5, 10, 20, 30]

    training_rows: List[Dict[str, object]] = []
    all_results: List[SAETrainResult] = []
    best_per_layer_expansion: Dict[Tuple[str, int, int], SAETrainResult] = {}
    best_global: Optional[SAETrainResult] = None

    normalization_by_layer: Dict[int, NormalizationStats] = {}

    for layer in layers:
        x_raw = x_by_layer[layer]
        norm = compute_normalization(x_raw)
        normalization_by_layer[layer] = norm
        x_norm = normalize_x(x_raw, norm)

        idx_all = np.arange(x_norm.shape[0])
        train_idx, val_idx = train_test_split(idx_all, test_size=0.1, random_state=args.seed)
        x_train = x_norm[train_idx]
        x_val = x_norm[val_idx]
        d_model = x_norm.shape[1]
        seen_lambdas: set = set()
        lambda_queue: List[float] = list(lambdas)
        layer_has_l0_lt50 = False

        while True:
            pending = [lam for lam in lambda_queue if lam not in seen_lambdas]
            if not pending:
                break

            for lam in pending:
                seen_lambdas.add(lam)

                for expansion in expansions:
                    d_hidden = expansion * d_model
                    print(
                        f"\n[INFO] Training SAE layer={layer} expansion={expansion}x "
                        f"d_hidden={d_hidden} lambda={lam:.2g}"
                    )
                    sae, metrics = train_single_sae(
                        x_train=x_train,
                        x_val=x_val,
                        d_model=d_model,
                        d_hidden=d_hidden,
                        l1_lambda=lam,
                        device=device,
                        seed=args.seed + layer * 1000 + expansion * 100 + int(lam * 1e5),
                        epochs=100,
                        batch_size=256,
                        lr=3e-4,
                    )

                    val_l0 = metrics["val_l0"]
                    if val_l0 < 50.0:
                        layer_has_l0_lt50 = True

                    score = score_sae(metrics["val_recon"], val_l0)
                    model_path = models_dir / f"sae_l1_layer{layer}_exp{expansion}_lam{lam:.3g}.pt"
                    torch.save(
                        {
                            "variant": "l1",
                            "layer": layer,
                            "expansion": expansion,
                            "d_hidden": d_hidden,
                            "lambda": lam,
                            "topk_k": None,
                            "state_dict": sae.state_dict(),
                            "norm_mean": norm.mean.tolist(),
                            "norm_std": norm.std.tolist(),
                            "metrics": metrics,
                        },
                        model_path,
                    )

                    row = SAETrainResult(
                        variant="l1",
                        layer=layer,
                        expansion=expansion,
                        d_hidden=d_hidden,
                        l1_lambda=lam,
                        topk_k=None,
                        train_recon=metrics["train_recon"],
                        train_l1=metrics["train_l1"],
                        train_l0=metrics["train_l0"],
                        val_recon=metrics["val_recon"],
                        val_l1=metrics["val_l1"],
                        val_l0=metrics["val_l0"],
                        score=score,
                        model_path=model_path,
                    )
                    all_results.append(row)

                    training_rows.append(
                        {
                            "variant": "l1",
                            "layer": layer,
                            "expansion": expansion,
                            "d_hidden": d_hidden,
                            "lambda": lam,
                            "topk_k": None,
                            "train_recon": row.train_recon,
                            "train_l1": row.train_l1,
                            "train_l0": row.train_l0,
                            "val_recon": row.val_recon,
                            "val_l1": row.val_l1,
                            "val_l0": row.val_l0,
                            "l0_in_analysis_range": bool(5.0 <= row.val_l0 <= 50.0),
                            "score": row.score,
                            "model_path": str(model_path),
                        }
                    )

                    key = ("l1", layer, expansion)
                    prev = best_per_layer_expansion.get(key)
                    if prev is None or row.score < prev.score:
                        best_per_layer_expansion[key] = row

            if layer_has_l0_lt50:
                break

            max_seen = max(seen_lambdas)
            if max_seen >= 100.0:
                print(
                    f"[WARN] Layer {layer}: no L0 < 50 even at lambda={max_seen:.2g}. "
                    "Stopping lambda escalation for this layer."
                )
                break
            next_lam = max_seen * 2.0
            print(f"[INFO] Layer {layer}: no L0 < 50 yet, escalating lambda to {next_lam:.2g}")
            lambda_queue.append(next_lam)

        if args.topk:
            expansion = 8
            d_hidden = expansion * d_model
            for k in topk_values:
                print(
                    f"\n[INFO] Training TopK SAE layer={layer} expansion={expansion}x "
                    f"d_hidden={d_hidden} k={k}"
                )
                topk_sae, metrics = train_single_topk_sae(
                    x_train=x_train,
                    x_val=x_val,
                    d_model=d_model,
                    d_hidden=d_hidden,
                    k=k,
                    device=device,
                    seed=args.seed + layer * 1000 + 7000 + k,
                    epochs=100,
                    batch_size=256,
                    lr=3e-4,
                )

                score = score_sae(metrics["val_recon"], metrics["val_l0"])
                model_path = models_dir / f"sae_topk_layer{layer}_exp8_k{k}.pt"
                torch.save(
                    {
                        "variant": "topk",
                        "layer": layer,
                        "expansion": expansion,
                        "d_hidden": d_hidden,
                        "lambda": None,
                        "topk_k": k,
                        "state_dict": topk_sae.state_dict(),
                        "norm_mean": norm.mean.tolist(),
                        "norm_std": norm.std.tolist(),
                        "metrics": metrics,
                    },
                    model_path,
                )

                row = SAETrainResult(
                    variant="topk",
                    layer=layer,
                    expansion=expansion,
                    d_hidden=d_hidden,
                    l1_lambda=None,
                    topk_k=k,
                    train_recon=metrics["train_recon"],
                    train_l1=metrics["train_l1"],
                    train_l0=metrics["train_l0"],
                    val_recon=metrics["val_recon"],
                    val_l1=metrics["val_l1"],
                    val_l0=metrics["val_l0"],
                    score=score,
                    model_path=model_path,
                )
                all_results.append(row)

                training_rows.append(
                    {
                        "variant": "topk",
                        "layer": layer,
                        "expansion": expansion,
                        "d_hidden": d_hidden,
                        "lambda": None,
                        "topk_k": k,
                        "train_recon": row.train_recon,
                        "train_l1": row.train_l1,
                        "train_l0": row.train_l0,
                        "val_recon": row.val_recon,
                        "val_l1": row.val_l1,
                        "val_l0": row.val_l0,
                        "l0_in_analysis_range": bool(5.0 <= row.val_l0 <= 50.0),
                        "score": row.score,
                        "model_path": str(model_path),
                    }
                )

                key = ("topk", layer, expansion)
                prev = best_per_layer_expansion.get(key)
                if prev is None or row.score < prev.score:
                    best_per_layer_expansion[key] = row

    eligible_results = [r for r in all_results if 5.0 <= r.val_l0 <= 50.0]
    if not eligible_results:
        raise RuntimeError(
            "No SAE configuration reached analysis sparsity range L0 in [5, 50]. "
            "Try --topk or broader lambda escalation."
        )

    best_global = min(eligible_results, key=lambda r: r.val_recon)
    best_per_layer: Dict[int, SAETrainResult] = {}
    for layer in layers:
        layer_rows = [r for r in eligible_results if r.layer == layer]
        if not layer_rows:
            raise RuntimeError(
                f"No eligible SAE with L0 in [5, 50] for layer {layer}. "
                "Increase lambda range or enable --topk."
            )
        best_per_layer[layer] = min(layer_rows, key=lambda r: r.val_recon)

    # Print required training summary.
    print("\nSAE training summary")
    print("Variant | Layer | Expansion | d_hidden | lambda/k | val_recon | val_l1 | val_l0 | in[5,50]")
    print("--------+-------+-----------+----------+----------+-----------+--------+--------+---------")
    for row in sorted(
        all_results,
        key=lambda r: (r.variant, r.layer, r.expansion, -1.0 if r.l1_lambda is None else r.l1_lambda, r.topk_k or -1),
    ):
        if row.variant == "l1":
            lk = f"lam={row.l1_lambda:.2g}"
        else:
            lk = f"k={row.topk_k:d}"
        print(
            f"{row.variant:7s} | {row.layer:5d} | {row.expansion:9d} | {row.d_hidden:8d} | {lk:8s} | "
            f"{row.val_recon:9.6f} | {row.val_l1:6.4f} | {row.val_l0:6.2f} | {str(5.0 <= row.val_l0 <= 50.0):>7s}"
        )

    # Load best SAE and run downstream analysis.
    best_ckpt = torch.load(best_global.model_path, map_location=device)
    if best_global.variant == "topk":
        if best_global.topk_k is None:
            raise RuntimeError("Best TopK SAE is missing k")
        best_sae: nn.Module = TopKSparseAutoencoder(
            d_model=x_by_layer[best_global.layer].shape[1],
            d_hidden=best_global.d_hidden,
            k=best_global.topk_k,
        )
    else:
        best_sae = SparseAutoencoder(d_model=x_by_layer[best_global.layer].shape[1], d_hidden=best_global.d_hidden)
    best_sae.load_state_dict(best_ckpt["state_dict"])
    best_sae.to(device)
    best_sae.eval()

    best_layer = best_global.layer
    x_best_raw = x_by_layer[best_layer]
    best_norm = normalization_by_layer[best_layer]
    x_best_norm = normalize_x(x_best_raw, best_norm)
    z_best = extract_latents(best_sae, x_best_norm, device=device)

    # 2D geometry analysis on raw residuals vs SAE latents, per layer.
    states = enumerate_states(args.n_disks)
    state_to_idx = {s: i for i, s in enumerate(states)}
    y_move_state_idx = np.array([state_to_idx[m.state_after] for m in metas], dtype=np.int64)
    graph_dist, edges = build_graph_and_distances(states)
    geometry_rows_by_layer: Dict[str, List[Dict[str, object]]] = {}
    move_metric_summary: Dict[str, Dict[str, Dict[str, float]]] = {}

    for layer in layers:
        layer_best = best_per_layer[layer]
        layer_ckpt = torch.load(layer_best.model_path, map_location=device)
        if layer_best.variant == "topk":
            if layer_best.topk_k is None:
                raise RuntimeError(f"Layer {layer} TopK SAE is missing k")
            layer_sae: nn.Module = TopKSparseAutoencoder(
                d_model=x_by_layer[layer].shape[1],
                d_hidden=layer_best.d_hidden,
                k=layer_best.topk_k,
            )
        else:
            layer_sae = SparseAutoencoder(d_model=x_by_layer[layer].shape[1], d_hidden=layer_best.d_hidden)
        layer_sae.load_state_dict(layer_ckpt["state_dict"])
        layer_sae.to(device)
        layer_sae.eval()

        x_layer_raw = x_by_layer[layer]
        layer_norm = normalization_by_layer[layer]
        z_layer = extract_latents(layer_sae, normalize_x(x_layer_raw, layer_norm), device=device)

        x_sep_raw = sep_acts[layer]
        x_sep_sae = extract_latents(layer_sae, normalize_x(x_sep_raw, layer_norm), device=device)

        sep_probe_raw, sep_coords_raw = train_distance_probe_full(x_sep_raw, graph_dist, device=device)
        sep_probe_sae, sep_coords_sae = train_distance_probe_full(x_sep_sae, graph_dist, device=device)
        move_probe_raw, move_sample_coords_raw = train_distance_probe_batched(
            x=x_layer_raw,
            y_state_idx=y_move_state_idx,
            graph_dist=graph_dist,
            device=device,
            epochs=2000,
            batch_size=1024,
            lr=1e-3,
            seed=args.seed,
        )
        move_probe_sae, move_sample_coords_sae = train_distance_probe_batched(
            x=z_layer,
            y_state_idx=y_move_state_idx,
            graph_dist=graph_dist,
            device=device,
            epochs=2000,
            batch_size=1024,
            lr=1e-3,
            seed=args.seed,
        )

        # Reference 81-state coordinates for move probes come from applying the learned
        # move-token probe to SEP state activations (one per state, no averaging).
        move_state_coords_raw = apply_probe(x_sep_raw, move_probe_raw, device=device)
        move_state_coords_sae = apply_probe(x_sep_sae, move_probe_sae, device=device)

        sep_raw_acc = nearest_state_accuracy(sep_coords_raw, np.arange(len(states), dtype=np.int64), sep_coords_raw)
        sep_sae_acc = nearest_state_accuracy(sep_coords_sae, np.arange(len(states), dtype=np.int64), sep_coords_sae)
        move_raw_acc = nearest_state_accuracy(move_sample_coords_raw, y_move_state_idx, move_state_coords_raw)
        move_sae_acc = nearest_state_accuracy(move_sample_coords_sae, y_move_state_idx, move_state_coords_sae)

        move_raw_full_space_acc = nearest_state_accuracy(x_layer_raw, y_move_state_idx, x_sep_raw)
        move_sae_full_space_acc = nearest_state_accuracy(z_layer, y_move_state_idx, x_sep_sae)

        move_avg_topo_raw = averaged_neighbor_metrics(move_state_coords_raw, graph_dist)
        move_avg_topo_sae = averaged_neighbor_metrics(move_state_coords_sae, graph_dist)

        sep_raw_rho, sep_raw_pearson = pairwise_corr_from_coords(sep_coords_raw, graph_dist)
        sep_sae_rho, sep_sae_pearson = pairwise_corr_from_coords(sep_coords_sae, graph_dist)
        move_raw_rho, move_raw_pearson = pairwise_corr_from_coords(move_state_coords_raw, graph_dist)
        move_sae_rho, move_sae_pearson = pairwise_corr_from_coords(move_state_coords_sae, graph_dist)

        geometry_rows = [
            {
                "input": "Raw residual (SEP)",
                "nearest_state_acc": sep_raw_acc,
                "spearman_rho": sep_raw_rho,
                "pearson_r": sep_raw_pearson,
            },
            {
                "input": "SAE latent (SEP)",
                "nearest_state_acc": sep_sae_acc,
                "spearman_rho": sep_sae_rho,
                "pearson_r": sep_sae_pearson,
            },
            {
                "input": "Raw residual (move)",
                "nearest_state_acc": move_raw_acc,
                "spearman_rho": move_raw_rho,
                "pearson_r": move_raw_pearson,
                "avg_top1_neighbor_adj": move_avg_topo_raw["top1_neighbor_adjacent"],
                "avg_top3_neighbor_adj": move_avg_topo_raw["top3_neighbor_adjacent"],
                "avg_mean_neighbor_rank": move_avg_topo_raw["mean_neighbor_rank"],
                "per_sample_acc": move_raw_acc,
                "full_space_sample_acc": move_raw_full_space_acc,
            },
            {
                "input": "SAE latent (move)",
                "nearest_state_acc": move_sae_acc,
                "spearman_rho": move_sae_rho,
                "pearson_r": move_sae_pearson,
                "avg_top1_neighbor_adj": move_avg_topo_sae["top1_neighbor_adjacent"],
                "avg_top3_neighbor_adj": move_avg_topo_sae["top3_neighbor_adjacent"],
                "avg_mean_neighbor_rank": move_avg_topo_sae["mean_neighbor_rank"],
                "per_sample_acc": move_sae_acc,
                "full_space_sample_acc": move_sae_full_space_acc,
            },
        ]
        geometry_rows_by_layer[str(layer)] = geometry_rows
        move_metric_summary[str(layer)] = {
            "raw": {
                "avg_top1_neighbor_adj": float(move_avg_topo_raw["top1_neighbor_adjacent"]),
                "avg_top3_neighbor_adj": float(move_avg_topo_raw["top3_neighbor_adjacent"]),
                "avg_mean_neighbor_rank": float(move_avg_topo_raw["mean_neighbor_rank"]),
                "per_sample_acc": float(move_raw_acc),
                "spearman_rho": float(move_raw_rho),
                "pearson_r": float(move_raw_pearson),
                "full_space_sample_acc": float(move_raw_full_space_acc),
            },
            "sae": {
                "avg_top1_neighbor_adj": float(move_avg_topo_sae["top1_neighbor_adjacent"]),
                "avg_top3_neighbor_adj": float(move_avg_topo_sae["top3_neighbor_adjacent"]),
                "avg_mean_neighbor_rank": float(move_avg_topo_sae["mean_neighbor_rank"]),
                "per_sample_acc": float(move_sae_acc),
                "spearman_rho": float(move_sae_rho),
                "pearson_r": float(move_sae_pearson),
                "full_space_sample_acc": float(move_sae_full_space_acc),
            },
        }

        plot_labeled_sierpinski_layout(
            states=states,
            edges=edges,
            coords=sep_coords_raw,
            title=f"Raw SEP Sierpinski - Layer {layer} - Spearman={sep_raw_rho:.3f}",
            out_path=out_dir / f"figure4_layer{layer}_raw_sep_sierpinski.png",
        )
        plot_labeled_sierpinski_layout(
            states=states,
            edges=edges,
            coords=sep_coords_sae,
            title=f"SAE SEP Sierpinski - Layer {layer} - Spearman={sep_sae_rho:.3f}",
            out_path=out_dir / f"figure5_layer{layer}_sae_sep_sierpinski.png",
        )
        plot_labeled_sierpinski_layout(
            states=states,
            edges=edges,
            coords=move_state_coords_raw,
            title=f"Raw Move-Token Sierpinski - Layer {layer} - Spearman={move_raw_rho:.3f}",
            out_path=out_dir / f"figure6_layer{layer}_raw_move_sierpinski.png",
        )
        plot_labeled_sierpinski_layout(
            states=states,
            edges=edges,
            coords=move_state_coords_sae,
            title=f"SAE Move-Token Sierpinski - Layer {layer} - Spearman={move_sae_rho:.3f}",
            out_path=out_dir / f"figure7_layer{layer}_sae_move_sierpinski.png",
        )

    feature_analysis = build_feature_analysis(z=z_best, metas=metas, vocab=vocab)
    state_recon = fit_state_reconstruction(x_raw=x_best_raw, z=z_best, metas=metas, seed=args.seed)

    interventions = run_causal_interventions(
        model=model,
        vocab=vocab,
        layer=best_layer,
        sae=best_sae,
        norm=best_norm,
        x_raw=x_best_raw,
        z=z_best,
        metas=metas,
        feature_rows=feature_analysis["features"],
        seed=args.seed,
        top_k_features=20,
        n_examples_per_feature=50,
    )

    # Save outputs.
    save_json(
        out_dir / "sae_training_summary.json",
        {
            "checkpoint": str(ckpt_path),
            "eval_results": str(eval_path),
            "layers": layers,
            "collection_stats": collection_stats,
            "sep_samples_per_layer": {str(k): int(v.shape[0]) for k, v in sep_acts.items()},
            "rows": training_rows,
            "eligible_rows_l0_5_to_50": [
                {
                    "variant": r.variant,
                    "layer": r.layer,
                    "expansion": r.expansion,
                    "d_hidden": r.d_hidden,
                    "lambda": r.l1_lambda,
                    "topk_k": r.topk_k,
                    "val_recon": r.val_recon,
                    "val_l0": r.val_l0,
                    "model_path": str(r.model_path),
                }
                for r in eligible_results
            ],
            "best_global": {
                "variant": best_global.variant,
                "layer": best_global.layer,
                "expansion": best_global.expansion,
                "d_hidden": best_global.d_hidden,
                "lambda": best_global.l1_lambda,
                "topk_k": best_global.topk_k,
                "score": best_global.score,
                "model_path": str(best_global.model_path),
            },
            "best_per_layer": {
                str(layer): {
                    "variant": best_per_layer[layer].variant,
                    "layer": best_per_layer[layer].layer,
                    "expansion": best_per_layer[layer].expansion,
                    "d_hidden": best_per_layer[layer].d_hidden,
                    "lambda": best_per_layer[layer].l1_lambda,
                    "topk_k": best_per_layer[layer].topk_k,
                    "score": best_per_layer[layer].score,
                    "model_path": str(best_per_layer[layer].model_path),
                }
                for layer in layers
            },
        },
    )
    save_json(
        out_dir / "feature_analysis_best.json",
        {
            "best_layer": best_layer,
            "best_expansion": best_global.expansion,
            "best_lambda": best_global.l1_lambda,
            "analysis": feature_analysis,
        },
    )
    save_json(out_dir / "state_reconstruction_comparison.json", state_recon)
    save_json(out_dir / "causal_interventions.json", interventions)
    save_json(
        out_dir / "geometry_probe_comparison.json",
        {
            "rows_by_layer": geometry_rows_by_layer,
            "move_metric_summary": move_metric_summary,
            "best_per_layer": {
                str(layer): {
                    "variant": best_per_layer[layer].variant,
                    "layer": best_per_layer[layer].layer,
                    "expansion": best_per_layer[layer].expansion,
                    "d_hidden": best_per_layer[layer].d_hidden,
                    "lambda": best_per_layer[layer].l1_lambda,
                    "topk_k": best_per_layer[layer].topk_k,
                    "model_path": str(best_per_layer[layer].model_path),
                }
                for layer in layers
            },
        },
    )

    # Figures.
    plot_activation_frequency_hist(feature_analysis["features"], out_dir / "figure1_activation_frequency_hist.png")
    plot_top_feature_bars(
        z=z_best,
        metas=metas,
        feature_rows=feature_analysis["features"],
        out_path=out_dir / "figure2_top10_interpretable_features.png",
        top_k=10,
    )
    plot_causal_examples(interventions.get("examples", []), out_dir / "figure3_causal_intervention_examples.png")
    plot_l0_recon_tradeoff(training_rows, out_dir / "figure8_l0_vs_recon_tradeoff.png")

    # Required printed summaries.
    print("\nFeature analysis summary")
    print(
        f"Best SAE: variant={best_global.variant}, layer={best_layer}, expansion={best_global.expansion}x, "
        f"d_hidden={best_global.d_hidden}, "
        f"lambda={best_global.l1_lambda if best_global.l1_lambda is not None else 'n/a'}, "
        f"k={best_global.topk_k if best_global.topk_k is not None else 'n/a'}"
    )
    print(
        f"Interpretable features: {feature_analysis['num_interpretable']} / {feature_analysis['num_features']}"
    )
    print(f"Breakdown: {feature_analysis['breakdown']}")

    print_state_table(state_recon)

    print("\nCausal intervention results (top interpretable features)")
    print(f"Activation intervention success rate: {100.0 * float(interventions['activation_success_rate']):.2f}%")
    print(f"Suppression intervention success rate: {100.0 * float(interventions['suppression_success_rate']):.2f}%")
    print(f"Disruption rate: {100.0 * float(interventions['disruption_rate']):.2f}%")
    if float(interventions["activation_success_rate"]) > float(interventions["suppression_success_rate"]):
        print("Observed asymmetry: activation > suppression (consistent with maze SAE findings).")
    else:
        print("Observed asymmetry: suppression >= activation.")

    for layer in layers:
        layer_rows = geometry_rows_by_layer[str(layer)]
        print(f"\nLayer {layer}: 2D geometry results")
        print_2d_comparison_table(layer_rows)
        move_raw_row = next(r for r in layer_rows if str(r["input"]) == "Raw residual (move)")
        move_sae_row = next(r for r in layer_rows if str(r["input"]) == "SAE latent (move)")

        if float(move_sae_row["nearest_state_acc"]) > float(move_raw_row["nearest_state_acc"]):
            print("Move-token nearest-state accuracy improved in SAE latent space.")
        else:
            print("Move-token nearest-state accuracy did not improve over raw residual space.")

        if float(move_sae_row["spearman_rho"]) > float(move_raw_row["spearman_rho"]):
            print("Move-token 2D distance structure (Spearman rho) improved in SAE latent space.")
        else:
            print("Move-token 2D distance structure (Spearman rho) did not improve over raw residual space.")

    print_move_metric_matrix(layers=layers, move_metric_summary=move_metric_summary)
    print_space_diagnostic_matrix(layers=layers, move_metric_summary=move_metric_summary)

    print(f"\nSaved outputs to {out_dir}")


if __name__ == "__main__":
    main()
