"""Augmented-steer hook + steering-direction cache helpers.

At each supervised position (b, t, state) the hook adds a FIXED-MAGNITUDE
unit-vector bump along the state-specific direction:

    h_new[b, t] = h[b, t] + alpha * direction[state]

Where `direction[state]` is the precomputed unit-norm steering vector
(target-mean over global-mean, normalized). `alpha` is a constant
hyperparameter (NOT a learned scalar — unlike canonical_steer).

Hook contract:
  - Training: hook fires at supervised anchor positions only. Set sup_pos
    before each forward; clear after.
  - Eval/inference: caller leaves sup_pos=None so the hook is a no-op.

Position-aware via indexed scatter, same pattern as canonical_loss.py:
only (b, t) positions in sup_pos are touched; every other token passes
through unchanged from h.clone().
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn


StateTuple = Tuple[int, ...]
SupPos = Tuple[int, int, Sequence[int]]


# Module-level state read by the hook. Same single-process assumption as
# canonical_loss.py (we run with `accelerate launch --num_processes 1`).
_HOOK_STATE: Dict[str, object] = {
    "sup_pos":    None,  # list of (b, t, state_tuple) — filtered to cached
    "alpha":      None,  # float (constant; NOT a Parameter)
    "directions": None,  # Dict[tuple, Tensor(H,)] unit-norm, CPU fp32
    "last_correction_norms": [],
}


def hook_state() -> Dict[str, object]:
    return _HOOK_STATE


def set_directions(directions: Dict[StateTuple, torch.Tensor]) -> None:
    _HOOK_STATE["directions"] = directions


def set_alpha(alpha: float) -> None:
    _HOOK_STATE["alpha"] = float(alpha)


def set_sup_pos(sup_pos: Optional[List[SupPos]]) -> None:
    _HOOK_STATE["sup_pos"] = sup_pos


def filter_sup_pos_to_cache(
    sup_pos: Sequence[SupPos], directions: Dict[StateTuple, torch.Tensor],
) -> List[SupPos]:
    return [(b, t, st) for (b, t, st) in sup_pos
            if tuple(int(x) for x in st) in directions]


def _augmented_hook(module, inputs, output):
    is_tuple = isinstance(output, tuple)
    h = output[0] if is_tuple else output

    sup_pos    = _HOOK_STATE.get("sup_pos")
    alpha      = _HOOK_STATE.get("alpha")
    directions = _HOOK_STATE.get("directions")
    if not sup_pos or alpha is None or directions is None:
        return None  # no-op — eval / inference path

    device = h.device
    dtype  = h.dtype
    bs_idx  = torch.tensor([b for (b, _, _) in sup_pos], dtype=torch.long, device=device)
    tok_idx = torch.tensor([t for (_, t, _) in sup_pos], dtype=torch.long, device=device)
    direction_stack = torch.stack([
        directions[tuple(int(x) for x in st)] for (_, _, st) in sup_pos
    ], dim=0).to(device=device, dtype=dtype)                # (n_sup, H)

    new_h = h.clone()
    correction = float(alpha) * direction_stack             # (n_sup, H)
    new_h[bs_idx, tok_idx] = h[bs_idx, tok_idx] + correction

    with torch.no_grad():
        norms = correction.float().norm(dim=-1)
        _HOOK_STATE["last_correction_norms"] = norms.tolist()

    if is_tuple:
        return (new_h,) + output[1:]
    return new_h


def install_hook(layer_module: nn.Module) -> torch.utils.hooks.RemovableHandle:
    """Attach _augmented_hook to `layer_module` (typically decoder block at
    PROBE_LAYER). Returns the handle so the caller can .remove() if needed."""
    return layer_module.register_forward_hook(_augmented_hook)


# ── Cache loading ────────────────────────────────────────────────────────────

def load_directions_cache(
    path: str,
) -> Tuple[Dict[StateTuple, torch.Tensor], torch.Tensor, dict]:
    """Returns (directions, global_mean, raw_dict). `directions[s]` is the
    unit-norm steering vector; `global_mean` is the activation mean across
    ALL collected supervised positions; raw_dict has provenance metadata."""
    raw = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(raw, dict):
        raise ValueError(f"steering_directions.pt should be a dict, got {type(raw)}")
    if "directions" not in raw:
        raise ValueError(
            f"steering_directions.pt at {path} is missing 'directions' — was it "
            f"built by finetune/precompute_steering_directions.py?"
        )
    directions = {
        tuple(int(x) for x in k): v.detach().to(torch.float32).cpu()
        for k, v in raw["directions"].items()
    }
    global_mean = raw.get("global_mean")
    if global_mean is not None:
        global_mean = global_mean.detach().to(torch.float32).cpu()
    return directions, global_mean, raw
