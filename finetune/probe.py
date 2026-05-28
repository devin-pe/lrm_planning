"""2D linear probe head trained jointly with the LM to project layer-L
hidden states onto a 2-D plane matching graph distances."""

from __future__ import annotations

import torch
import torch.nn as nn


class Probe(nn.Module):
    def __init__(self, hidden_dim: int, out_dim: int = 2):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, out_dim, bias=False)
        nn.init.normal_(self.proj.weight, std=0.02)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.proj(h)
