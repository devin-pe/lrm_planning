"""Band-restricted alignment loss for the alignment_loss training regime.

Mechanism: during a single forward pass for LM loss, we register
forward_pre_hooks with_kwargs=True on each band layer to capture both the
hidden_states input and the kwargs (position_embeddings, attention_mask,
position_ids) the layer was about to receive. After computing lm_loss we
*detach* the captured input to layer band[0] and re-run band layers
band[0]..band[-1] on that detached tensor with the captured kwargs.

The detach breaks the gradient chain back through layers 0..band[0]-1, so
gradient from alignment_loss can only enter LoRA weights inside the band
(plus any unfrozen non-LoRA params at those layers, but here only LoRA is
trainable). Layers band[-1]+1..N are not in the alignment computation path
at all and naturally receive no alignment gradient.

This file contains pure functions; the orchestration (hook install, two
losses + combined backward, logging) lives in finetune/train.py.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch


# ── Cache loading ────────────────────────────────────────────────────────────

def load_alignment_caches(
    centroids_path: str,
    h_baseline_path: str,
) -> Tuple[Dict[int, Dict[Tuple, torch.Tensor]],
           Dict[Tuple[int, int], torch.Tensor],
           List[int],
           int]:
    """Returns (directions, h_baseline_data, band_layers, hidden_dim).

    `directions[L][state_tuple]` is a fp32 CPU tensor[hidden_dim].
    `h_baseline_data[(example_id, ti)]` is a bf16 CPU tensor[n_band, hidden_dim],
    in the same band-layer order as `band_layers`.
    """
    cents = torch.load(centroids_path, map_location="cpu", weights_only=False)
    base  = torch.load(h_baseline_path, map_location="cpu", weights_only=False)
    if cents["band_layers"] != base["band_layers"]:
        raise ValueError(
            f"centroids band={cents['band_layers']} but h_baseline band="
            f"{base['band_layers']} — caches were precomputed with mismatched bands"
        )
    band = list(cents["band_layers"])
    hidden_dim = int(cents["hidden_dim"])
    directions = cents["directions"]            # {L: {state: tensor}}
    h_baseline = base["data"]                   # {(ex_id, ti): tensor[n_band, hidden]}
    return directions, h_baseline, band, hidden_dim


# ── Hook capture ─────────────────────────────────────────────────────────────

class BandCapture:
    """Captures, per band layer, the kwargs and input hidden_states as they
    enter the layer during the first forward pass.

    After pass 1, the captured `hidden_states_in` for `band[0]` is the
    activation right BEFORE the first band layer — exactly what we detach and
    feed into the band re-run.
    """

    def __init__(self, band: Sequence[int]):
        self.band = list(band)
        self.captures: Dict[int, Dict[str, Any]] = {}
        self._handles: List[Any] = []

    def install(self, layers) -> None:
        for L in self.band:
            self._handles.append(
                layers[L].register_forward_pre_hook(
                    self._make_pre_hook(L), with_kwargs=True
                )
            )

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    def clear(self) -> None:
        self.captures = {}

    def _make_pre_hook(self, layer_idx: int):
        def hook(_module, args, kwargs):
            # `hidden_states` is the first positional. Capture as-is (no detach;
            # detach happens in the caller for the band[0] entry).
            self.captures[layer_idx] = {
                "hidden_states_in": args[0] if args else kwargs.get("hidden_states"),
                "kwargs": dict(kwargs),
            }
            # Returning None lets the original call proceed unchanged.
            return None
        return hook


# ── Band re-run ──────────────────────────────────────────────────────────────

def run_band_with_grad(
    layers,
    band: Sequence[int],
    captures: Dict[int, Dict[str, Any]],
) -> Dict[int, torch.Tensor]:
    """Re-run band layers on a detached copy of the band[0] input, returning
    a dict {L: hidden_states_out[B, T, hidden]} with gradient ancestry
    restricted to LoRA params inside the band.

    NB: each band layer uses ITS OWN captured kwargs (the attention mask
    differs between linear- and full-attention layers in Qwen3_5).
    """
    if not band:
        return {}
    # Detach the input to the first band layer — this is the critical step
    # that severs the alignment-loss gradient chain from upstream LoRA. The
    # detached tensor has requires_grad=False; the subsequent layer.forward
    # calls regenerate a grad-tracked output via the LoRA params inside the
    # band, which IS the only path alignment_loss.backward() can travel.
    h = captures[band[0]]["hidden_states_in"].detach()
    outputs: Dict[int, torch.Tensor] = {}
    for L in band:
        kwargs = captures[L]["kwargs"]
        # Drop past_key_values / use_cache — band re-run is stateless.
        kwargs = {k: v for k, v in kwargs.items()
                  if k not in ("past_key_values", "use_cache", "past_key_value")}
        h = layers[L](h, **kwargs)
        if isinstance(h, tuple):
            h = h[0]
        outputs[L] = h
    return outputs


# ── Alignment loss ───────────────────────────────────────────────────────────

def compute_alignment_loss(
    band_outputs: Dict[int, torch.Tensor],
    supervised_positions: Sequence[Tuple[int, int, Tuple]],
    example_ids: Sequence[int],
    h_baseline: Dict[Tuple[int, int], torch.Tensor],
    directions: Dict[int, Dict[Tuple, torch.Tensor]],
    alpha: float,
    band: Sequence[int],
    hidden_dim: int,
    device: torch.device,
) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
    """Mean-squared distance from h_current to (h_baseline + α · direction).

    Returns (loss, diagnostics_dict). loss is None if no usable sup positions
    in this batch.
    """
    if not supervised_positions:
        return None, {"n_sup_used": 0, "n_sup_skipped": 0}

    band_idx_of = {L: i for i, L in enumerate(band)}
    total_sq_err = 0.0  # accumulated tensor; init as Python 0 and use += tensor below
    total_terms = 0
    n_used = 0
    n_skipped = 0
    per_layer_perturb_sq: Dict[int, float] = {L: 0.0 for L in band}
    per_layer_dist_target_sq: Dict[int, float] = {L: 0.0 for L in band}

    # Accumulate losses in a list and stack/mean at the end — cheaper than
    # repeated scalar broadcasts.
    loss_terms: List[torch.Tensor] = []
    for (b, ti, state) in supervised_positions:
        ex_id = int(example_ids[b])
        key = (ex_id, int(ti))
        baseline_block = h_baseline.get(key)
        if baseline_block is None:
            n_skipped += 1
            continue
        state_t = tuple(int(x) for x in state)
        for L in band:
            dir_vec = directions[L].get(state_t)
            if dir_vec is None:
                n_skipped += 1
                continue
            i_band = band_idx_of[L]
            h_cur = band_outputs[L][b, ti]                                 # [hidden]  (bf16)
            # With device_map="auto" the band layers may sit on a different
            # GPU than lm_head — place targets on h_cur's actual device.
            tgt_dev = h_cur.device
            h_base = baseline_block[i_band].to(device=tgt_dev,
                                               dtype=h_cur.dtype)           # [hidden]
            target = h_base + alpha * dir_vec.to(device=tgt_dev,
                                                 dtype=h_cur.dtype)         # [hidden]
            diff = (h_cur - target).float()
            loss_term = (diff * diff).mean()       # scalar, fp32
            loss_terms.append(loss_term)
            per_layer_dist_target_sq[L] += float(loss_term.detach())
            with torch.no_grad():
                perturb = (h_cur - h_base).float()
                per_layer_perturb_sq[L] += float((perturb * perturb).mean())
            total_terms += 1
        n_used += 1

    if not loss_terms:
        return None, {"n_sup_used": n_used, "n_sup_skipped": n_skipped}
    # Band layers can be split across GPUs by device_map="auto"; collect onto
    # a single device (the caller's `device`, typically lm_loss.device) before
    # reduction so the result lands on the same GPU as lm_loss for summing.
    loss = torch.stack([t.to(device) for t in loss_terms]).mean()
    # Per-layer means (over positions visited): sqrt(mean(mse_per_layer)).
    diag = {
        "n_sup_used": n_used,
        "n_sup_skipped": n_skipped,
        "n_terms": total_terms,
        "per_layer_dist_target_rms": {
            L: (per_layer_dist_target_sq[L] / max(n_used, 1)) ** 0.5
            for L in band
        },
        "per_layer_perturbation_rms": {
            L: (per_layer_perturb_sq[L] / max(n_used, 1)) ** 0.5
            for L in band
        },
    }
    return loss, diag


# ── λ warmup ─────────────────────────────────────────────────────────────────

def lambda_eff(step: int, total_steps: int, target: float, warmup_frac: float) -> float:
    """Linear ramp from 0 to `target` over the first warmup_frac * total_steps
    steps, then constant at `target`."""
    if warmup_frac <= 0:
        return float(target)
    warm_steps = max(1, int(warmup_frac * total_steps))
    if step >= warm_steps:
        return float(target)
    return float(target) * (step / warm_steps)


# ── Gradient-flow diagnostic ─────────────────────────────────────────────────

def _layer_idx_from_param_name(name: str, n_layers: int) -> Optional[int]:
    """Return the layer idx encoded in a parameter name like
    'base_model.model.model.layers.31.self_attn.q_proj.lora_A.default.weight',
    or None if the name doesn't reference a layer (embed_tokens, lm_head)."""
    parts = name.split(".")
    for i, tok in enumerate(parts[:-1]):
        if tok == "layers":
            nxt = parts[i + 1]
            if nxt.isdigit():
                L = int(nxt)
                if 0 <= L < n_layers:
                    return L
    return None


def verify_band_gradient_flow(
    model,
    layers,
    band: Sequence[int],
    captures: Dict[int, Dict[str, Any]],
    h_baseline_cache,
    directions_cache,
    alpha: float,
    hidden_dim: int,
    batch: Dict[str, Any],
    accelerator,
) -> None:
    """Run two separate backward passes on a SHARED forward graph (lm_loss
    only, then alignment_loss only) and assert that alignment_loss puts
    gradient ONLY on LoRA params at band layers.

    Raises RuntimeError if band leakage is detected.
    """
    print("\n[grad-diag] verifying band-only alignment-loss gradient flow …")
    model.train()
    captures.clear()

    # The batch may come from a CPU dataloader; move to the model's input device.
    in_dev = model.get_input_embeddings().weight.device
    outputs = model(
        input_ids=batch["input_ids"].to(in_dev),
        attention_mask=batch["attention_mask"].to(in_dev),
        labels=batch["labels"].to(in_dev),
        use_cache=False,
    )
    lm_loss = outputs.loss
    band_outputs = run_band_with_grad(layers, band, captures)
    align_loss, _ = compute_alignment_loss(
        band_outputs, batch["supervised_positions"], batch["example_ids"],
        h_baseline_cache, directions_cache, alpha, band, hidden_dim,
        device=lm_loss.device,
    )
    if align_loss is None:
        print("[grad-diag] no usable sup positions in diagnostic batch — skipping.")
        return

    n_layers = len(layers)

    def _layers_with_grad() -> set:
        s: set = set()
        for n, p in model.named_parameters():
            if not p.requires_grad or p.grad is None:
                continue
            if not torch.any(p.grad != 0):
                continue
            L = _layer_idx_from_param_name(n, n_layers)
            if L is not None:
                s.add(L)
        return s

    # LM-only backward (retain_graph for the second pass).
    for prm in model.parameters():
        if prm.grad is not None:
            prm.grad = None
    accelerator.backward(lm_loss, retain_graph=True)
    lm_layers = _layers_with_grad()

    for prm in model.parameters():
        if prm.grad is not None:
            prm.grad = None
    accelerator.backward(align_loss)
    align_layers = _layers_with_grad()

    for prm in model.parameters():
        if prm.grad is not None:
            prm.grad = None

    band_set = {int(L) for L in band}
    align_leak = align_layers - band_set

    print(f"[grad-diag] LM-only backward    → {len(lm_layers)}/{n_layers} layers received grad")
    print(f"[grad-diag] align-only backward → layers with grad = {sorted(align_layers)}")
    print(f"[grad-diag] band                 = {sorted(band_set)}")
    if align_leak:
        raise RuntimeError(
            f"[grad-diag] BAND LEAKAGE: alignment_loss deposited gradient on "
            f"layers OUTSIDE the band: {sorted(align_leak)}. The detach() in "
            f"run_band_with_grad() failed — abort."
        )
    if not (align_layers <= band_set):
        # Should not happen given align_leak == ∅, but belt+braces.
        raise RuntimeError("[grad-diag] band-membership check failed.")
    if len(lm_layers) < n_layers // 2:
        print(f"[grad-diag] WARN: LM loss touched only {len(lm_layers)} layers — "
              f"expected ~all. Check LoRA target_modules.")
    print("[grad-diag] OK — alignment loss gradient stayed strictly inside the band.")
