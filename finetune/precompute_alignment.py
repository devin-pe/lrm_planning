"""Precompute the two caches alignment_loss needs from a RAW PRETRAINED model
(no LoRA): per-state activation centroids and per-(example, position)
h_baseline reference activations, both at every layer in ALIGNMENT_BAND.

Outputs:
    <out_dir>/state_centroids.pt   -- dict with keys
        'band_layers':     list[int]   (the captured layers)
        'hidden_dim':      int
        'h_target':        {L: {state_tuple: fp32 tensor[hidden_dim]}}
        'h_global_mean':   {L: fp32 tensor[hidden_dim]}
        'directions':      {L: {state_tuple: fp32 tensor[hidden_dim]}}
                            # direction = h_target - h_global_mean  (NOT unit-norm)
        'counts':          {L: {state_tuple: int}}
        'model_id':        str

    <out_dir>/h_baseline_supervised.pt -- dict with keys
        'band_layers':     list[int]
        'hidden_dim':      int
        'data':            {(example_id, ti): bf16 tensor[n_band_layers, hidden_dim]}
        'model_id':        str
        'train_file':      str

The h_baseline cache is keyed by (dataset_row_idx, abs_token_idx). At training
time, batch['example_ids'][b] + sup_pos[i][1] picks the right entry.

Usage:
    python -m finetune.precompute_alignment \\
        --model_id /scratch-shared/dpereira/qwen3.6-27b \\
        --data_dir hanoi_data/ \\
        --train_file train.jsonl \\
        --output_dir runs/qwen36_alignment_loss_L30to42 \\
        --alignment_band 30,31,32,33,34,35,36,37,38,39,40,41,42 \\
        --max_seq_len 768 \\
        --batch_size 2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from finetune.data import collate_fn, make_datasets


def _parse_csv_ints(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _decoder_layers(model):
    """Drill through wrappers to the layer list. Works for raw HF and PEFT."""
    obj = model
    for _ in range(6):
        if hasattr(obj, "layers"):
            return obj.layers
        for attr in ("model", "base_model"):
            sub = getattr(obj, attr, None)
            if sub is not None:
                obj = sub
                break
        else:
            break
    raise RuntimeError("Cannot locate decoder layers on this model")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", required=True,
                   help="Path or HF id of the RAW pretrained model (no LoRA).")
    p.add_argument("--data_dir", default="hanoi_data/")
    p.add_argument("--train_file", default="train.jsonl")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--alignment_band", default="30,31,32,33,34,35,36,37,38,39,40,41,42",
                   help="Comma-separated layer indices to capture.")
    p.add_argument("--max_seq_len", type=int, default=768)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    band = _parse_csv_ints(args.alignment_band)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load raw pretrained — no LoRA, no hook, just the base model in bf16.
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"[precompute_alignment] loading raw model from {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map="auto" if args.device == "cuda" else None,
    )
    model.eval()
    for prm in model.parameters():
        prm.requires_grad_(False)

    layers = _decoder_layers(model)
    hidden_dim = int(model.config.text_config.hidden_size) \
        if hasattr(model.config, "text_config") else int(model.config.hidden_size)
    print(f"[precompute_alignment] hidden_dim = {hidden_dim}")
    print(f"[precompute_alignment] alignment_band = {band}")
    print(f"[precompute_alignment] total layers = {len(layers)}")

    # Register forward hooks on each band layer to capture the OUTPUT
    # (post-residual hidden state). DecoderLayer.forward returns a Tensor on
    # Qwen3_5; if some HF version returns a tuple we take element 0.
    captured: Dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(layer_idx):
        def hook(_module, _args, output):
            captured[layer_idx] = output[0] if isinstance(output, tuple) else output
        return hook

    for L in band:
        handles.append(layers[L].register_forward_hook(make_hook(L)))

    # Dataset.
    train_ds, _ = make_datasets(
        Path(args.data_dir), tokenizer, args.max_seq_len,
        train_file=args.train_file,
    )
    print(f"[precompute_alignment] train rows = {len(train_ds)}")
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate_fn, num_workers=2)

    # Centroid accumulators: per-layer per-state running sum + count (fp32).
    sums: Dict[int, Dict[Tuple, torch.Tensor]] = {
        L: defaultdict(lambda: torch.zeros(hidden_dim, dtype=torch.float32))
        for L in band
    }
    counts: Dict[int, Dict[Tuple, int]] = {L: defaultdict(int) for L in band}
    # Per-layer total sum + total count for the global mean.
    global_sum: Dict[int, torch.Tensor] = {
        L: torch.zeros(hidden_dim, dtype=torch.float32) for L in band
    }
    global_count = 0

    # h_baseline storage. Keys are (example_id, ti); values are bf16 [n_band, hidden].
    h_baseline: Dict[Tuple[int, int], torch.Tensor] = {}

    t0 = time.time()
    n_positions = 0
    n_batches = 0
    in_dev = model.get_input_embeddings().weight.device

    with torch.no_grad():
        for batch in loader:
            captured.clear()
            _ = model(
                input_ids=batch["input_ids"].to(in_dev),
                attention_mask=batch["attention_mask"].to(in_dev),
                output_hidden_states=False,
                use_cache=False,
            )
            ex_ids = batch["example_ids"]
            sup = batch["supervised_positions"]
            for (b, ti, state) in sup:
                key = tuple(int(x) for x in state)
                # h_baseline: stack across band layers.
                vecs = torch.stack(
                    [captured[L][b, ti].detach().to(torch.bfloat16).cpu() for L in band],
                    dim=0,
                )
                h_baseline[(int(ex_ids[b]), int(ti))] = vecs
                # Centroids: per-layer per-state running sum.
                for L in band:
                    v = captured[L][b, ti].detach().to(torch.float32).cpu()
                    sums[L][key] = sums[L][key] + v
                    counts[L][key] = counts[L][key] + 1
                    global_sum[L] = global_sum[L] + v
                n_positions += 1
                global_count += 1
            n_batches += 1
            if n_batches % 20 == 0:
                print(f"  [precompute_alignment] batch {n_batches}/{len(loader)}  "
                      f"sup_positions={n_positions}  unique_states={len(sums[band[0]])}  "
                      f"elapsed={time.time() - t0:.1f}s")

    for h in handles:
        h.remove()

    # Compute centroids + global means + directions.
    h_target = {L: {k: (sums[L][k] / counts[L][k]).contiguous()
                    for k in sums[L]} for L in band}
    h_global_mean = {L: (global_sum[L] / max(global_count, 1)).contiguous() for L in band}
    directions = {L: {k: (h_target[L][k] - h_global_mean[L]).contiguous()
                       for k in h_target[L]} for L in band}

    # Sanity: per-layer direction norm distribution.
    for L in band:
        norms = torch.stack([directions[L][k].norm() for k in directions[L]])
        print(f"  [direction norms]  L={L:2d}  mean={norms.mean():.2f}  "
              f"std={norms.std():.2f}  min={norms.min():.2f}  max={norms.max():.2f}")

    centroids_out = {
        "band_layers": band,
        "hidden_dim": hidden_dim,
        "h_target": h_target,
        "h_global_mean": h_global_mean,
        "directions": directions,
        "counts": {L: dict(counts[L]) for L in band},
        "model_id": args.model_id,
    }
    centroids_path = out_dir / "state_centroids.pt"
    torch.save(centroids_out, centroids_path)
    print(f"[precompute_alignment] saved → {centroids_path}")

    h_baseline_out = {
        "band_layers": band,
        "hidden_dim": hidden_dim,
        "data": h_baseline,
        "model_id": args.model_id,
        "train_file": args.train_file,
    }
    h_baseline_path = out_dir / "h_baseline_supervised.pt"
    torch.save(h_baseline_out, h_baseline_path)
    print(f"[precompute_alignment] saved → {h_baseline_path}")
    print(f"[precompute_alignment] h_baseline entries = {len(h_baseline)}  "
          f"estimated size = {len(h_baseline) * len(band) * hidden_dim * 2 / 1e9:.2f} GB")

    summary = {
        "model_id": args.model_id,
        "train_file": args.train_file,
        "band_layers": band,
        "hidden_dim": hidden_dim,
        "n_positions": n_positions,
        "n_unique_states_per_layer": {L: len(h_target[L]) for L in band},
        "elapsed_s": time.time() - t0,
        "direction_norm_stats": {
            L: {
                "mean": float(torch.stack([directions[L][k].norm() for k in directions[L]]).mean()),
                "std": float(torch.stack([directions[L][k].norm() for k in directions[L]]).std()),
            } for L in band
        },
    }
    (out_dir / "precompute_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[precompute_alignment] done. elapsed={time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
