"""L_total = L_LM + lambda_geom * L_geom.

L_LM comes from the HF model's forward pass when labels are provided.

L_geom: for every (batch_idx, token_idx, target_state) in the batch,
project the hidden state at the configured layer onto 2-D via the probe,
build the n×n pairwise predicted-distance matrix, compare it to the
normalised graph-distance matrix at the corresponding indices via MSE on
off-diagonal entries.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn


def _state_idx_b3(state: Sequence[int]) -> int:
    idx = 0
    for i in range(len(state) - 1, -1, -1):
        idx = idx * 3 + int(state[i])
    return idx


def compute_geom_loss(
    hidden_states_layer: torch.Tensor,         # (B, T, H), already from the chosen layer
    supervised_positions: Sequence[Tuple[int, int, Sequence[int]]],
    probe: nn.Module,
    norm_dist: torch.Tensor,                   # (81, 81)
) -> Tuple[torch.Tensor, int]:
    """Return (L_geom, n_supervised). L_geom is a scalar tensor with grad.

    Normalisation: mean per supervised position. Each of the n positions
    appears in exactly (n−1) pairwise distance terms, so dividing by
    n·(n−1) yields a per-position MSE that is O(1) regardless of how many
    supervised positions fall into the batch. With fewer than 2 supervised
    positions we have no pairwise structure → return a zero-grad scalar.

    With `device_map="auto"`, HF collates per-layer hidden-state outputs
    onto the model's embedding device. The probe head may live on a
    different shard, so we move `h_sel` to the probe's device — the .to()
    preserves the autograd graph.
    """
    hs_device = hidden_states_layer.device
    n_sup = len(supervised_positions)
    if n_sup < 2:
        # Safe-guard: no signal possible. Return a true scalar so the
        # caller can backward through it without changing graph state.
        return (
            torch.zeros((), device=hs_device, dtype=hidden_states_layer.dtype),
            n_sup,
        )

    bs_idx = torch.tensor([b for b, _, _ in supervised_positions], dtype=torch.long, device=hs_device)
    tok_idx = torch.tensor([t for _, t, _ in supervised_positions], dtype=torch.long, device=hs_device)
    targets = [tuple(int(x) for x in s) for _, _, s in supervised_positions]

    probe_device = next(probe.parameters()).device

    h_sel = hidden_states_layer[bs_idx, tok_idx]                       # (n, H)  @ hs_device
    h_sel = h_sel.to(probe_device)                                     # autograd-preserving move
    z = probe(h_sel).float()                                           # (n, 2)  @ probe_device
    pred = torch.cdist(z, z, p=2)                                      # (n, n)
    state_idx = torch.tensor([_state_idx_b3(t) for t in targets], dtype=torch.long, device=probe_device)
    true = norm_dist.to(device=probe_device, dtype=pred.dtype)[state_idx][:, state_idx]

    n = z.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool, device=probe_device)
    diff = (pred - true)[mask]
    # `.mean()` over n·(n−1) off-diagonal entries == mean per supervised
    # position (each position contributes to (n−1) pairs).
    loss = (diff * diff).mean()
    return loss, n


def effective_lambda(
    step: int,
    total_steps: int,
    lambda_target: float,
    warmup_frac: float = 0.2,
) -> float:
    """Linear warmup of λ from 0 → `lambda_target` over the first
    `warmup_frac` of training, then constant at the target.

    Pass `warmup_frac=0.0` to disable warmup (always returns the target).
    """
    if lambda_target == 0.0 or warmup_frac <= 0.0 or total_steps <= 0:
        return float(lambda_target)
    warmup_steps = max(1, int(round(total_steps * warmup_frac)))
    if step >= warmup_steps:
        return float(lambda_target)
    return float(lambda_target) * (float(step) / float(warmup_steps))


def total_loss(
    lm_loss: torch.Tensor,
    geom_loss: torch.Tensor,
    lambda_geom: float,
    regime: str,
) -> torch.Tensor:
    if regime == "probe":
        return lm_loss + lambda_geom * geom_loss
    return lm_loss
