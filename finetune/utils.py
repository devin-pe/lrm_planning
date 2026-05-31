"""Graph distance precomputation + state <-> index helpers.

State encoding: a 4-tuple (p_0, p_1, p_2, p_3) over {0,1,2} is mapped to an
integer in [0, 81) via base-3 packing on the disk index. Disk 0 is the least
significant trit.
"""

from __future__ import annotations

import sys
from collections import deque
from pathlib import Path
from typing import Iterable, Tuple

import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from hanoi_data.template import (  # noqa: E402
    N_DISKS,
    enumerate_states,
    legal_moves,
)

State = Tuple[int, ...]
N_STATES = 3 ** N_DISKS  # 81


def state_to_idx(state: State) -> int:
    """Base-3 packing: disk 0 is the least significant trit."""
    idx = 0
    for i in range(len(state) - 1, -1, -1):
        idx = idx * 3 + int(state[i])
    return idx


def idx_to_state(idx: int, n_disks: int = N_DISKS) -> State:
    """Inverse of state_to_idx."""
    out = []
    for _ in range(n_disks):
        out.append(idx % 3)
        idx //= 3
    return tuple(out)


def build_graph_adjacency(n_disks: int = N_DISKS) -> dict:
    """state_tuple -> list of neighbouring state_tuples via one legal move."""
    adj = {}
    for s in enumerate_states(n_disks=n_disks):
        nbrs = []
        for fp, tp in legal_moves(s):
            # Inline-apply to avoid the planning.py round-trip for every move:
            # the topmost disk on fp is the smallest d with s[d] == fp.
            for d in range(len(s)):
                if s[d] == fp:
                    nxt = list(s)
                    nxt[d] = tp
                    nbrs.append(tuple(nxt))
                    break
        adj[s] = nbrs
    return adj


def graph_distance_matrix(n_disks: int = N_DISKS) -> torch.Tensor:
    """Unnormalised 3^n × 3^n BFS distance matrix.

    Indexed by state_to_idx. Build with one BFS per source state.
    """
    adj = build_graph_adjacency(n_disks=n_disks)
    n = 3 ** n_disks
    dist = torch.full((n, n), float("inf"), dtype=torch.float32)
    for src in enumerate_states(n_disks=n_disks):
        src_idx = state_to_idx(src)
        dist[src_idx, src_idx] = 0.0
        q = deque([src])
        while q:
            u = q.popleft()
            u_idx = state_to_idx(u)
            for v in adj[u]:
                v_idx = state_to_idx(v)
                if torch.isinf(dist[src_idx, v_idx]):
                    dist[src_idx, v_idx] = dist[src_idx, u_idx] + 1.0
                    q.append(v)
    if torch.isinf(dist).any():
        raise RuntimeError(f"Disconnected states in {n_disks}-disk Hanoi graph")
    return dist


def normalised_graph_distance_matrix(n_disks: int = N_DISKS) -> torch.Tensor:
    """Divide by std-dev over off-diagonal entries so the MSE target is well-conditioned."""
    dist = graph_distance_matrix(n_disks=n_disks)
    n = dist.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool)
    sd = float(dist[mask].std().clamp(min=1e-8))
    return dist / sd


def states_to_idx_tensor(states: Iterable[State]) -> torch.Tensor:
    return torch.tensor([state_to_idx(tuple(int(x) for x in s)) for s in states],
                        dtype=torch.long)


if __name__ == "__main__":
    # Smoke test: verify bijection and a couple of known distances.
    for s in enumerate_states():
        assert idx_to_state(state_to_idx(s)) == s, s
    raw = graph_distance_matrix()
    assert raw.shape == (81, 81)
    assert (raw.diagonal() == 0).all()
    # The diameter of 4-disk flat-to-flat is 15.
    assert raw.max() == 15.0, raw.max()
    norm = normalised_graph_distance_matrix()
    print(f"raw  max={raw.max().item():.0f}  mean off-diag={raw[~torch.eye(81, dtype=torch.bool)].mean():.3f}")
    print(f"norm max={norm.max().item():.3f}  std off-diag={norm[~torch.eye(81, dtype=torch.bool)].std():.3f}")
