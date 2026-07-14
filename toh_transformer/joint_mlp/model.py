#!/usr/bin/env python3
"""Model components for joint MLP probes.

A shared nonlinear encoder maps the 128-d SEP residual to an h_dim bottleneck,
feeding two heads:
  - four per-disk 3-way classifiers (which peg each disk sits on), and
  - a 2D distance-matching projection (bias-free).

Dependencies: torch only.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MLPEncoder(nn.Module):
    """Linear(in_dim -> hidden) -> ReLU -> Linear(hidden -> h_dim)."""

    def __init__(self, in_dim: int = 128, hidden: int = 64, h_dim: int = 8):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, hidden)
        self.act = nn.ReLU()
        self.linear2 = nn.Linear(hidden, h_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.act(self.linear1(x)))


class PerDiskHeads(nn.Module):
    """Four independent Linear(h_dim -> 3) classifiers, one per disk."""

    def __init__(self, h_dim: int, n_disks: int = 4, n_pegs: int = 3):
        super().__init__()
        self.heads = nn.ModuleList([nn.Linear(h_dim, n_pegs) for _ in range(n_disks)])

    def forward(self, z: torch.Tensor):
        return [head(z) for head in self.heads]  # list of (N, 3)


class Proj2D(nn.Module):
    """Bias-free Linear(h_dim -> 2) for distance-matching."""

    def __init__(self, h_dim: int):
        super().__init__()
        self.proj = nn.Linear(h_dim, 2, bias=False)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.proj(z)


def effective_linear_map(encoder: MLPEncoder) -> torch.Tensor:
    """W_effective = W_enc_2 @ W_enc_1, the LINEAR part of the encoder (ignoring
    the ReLU). Shape (h_dim, in_dim)."""
    W1 = encoder.linear1.weight  # (hidden, in_dim)
    W2 = encoder.linear2.weight  # (h_dim, hidden)
    return W2 @ W1  # (h_dim, in_dim)
