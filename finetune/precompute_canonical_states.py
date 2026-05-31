"""Precompute canonical hidden states per board state.

For each training example, do a forward pass through baseline+LoRA (no hook
attached). At each supervised position (closing `]` of each move triple),
record (post-move-state, hidden_state at PROBE_LAYER). Group by state, take
the per-state mean.

Output (saved with torch.save):
    {state_tuple: tensor(hidden_dim,) on CPU, fp32}

Usage:
    python -m finetune.precompute_canonical_states \\
        --baseline_checkpoint runs/baseline/final \\
        --data_dir hanoi_data/ \\
        --output runs/canonical_steer_L36/hidden_states_by_state.pt \\
        --probe_layer 36 \\
        --max_seq_len 768 \\
        --batch_size 4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from finetune.data import collate_fn, make_datasets
from finetune.config import HIDDEN_DIM


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline_checkpoint", required=True,
                   help="Path to a LoRA-baseline ckpt dir, e.g. runs/baseline/final.")
    p.add_argument("--data_dir", default="hanoi_data/")
    p.add_argument("--output", required=True,
                   help="Where to write the {state_tuple: h_target} dict.")
    p.add_argument("--probe_layer", type=int, default=36)
    p.add_argument("--max_seq_len", type=int, default=768)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load baseline + LoRA via the eval-time loader; it sets up dtype, device_map,
    # and freezes everything for us.
    from eval.load import load_checkpoint
    tokenizer, model, _probe_head, train_cfg = load_checkpoint(
        args.baseline_checkpoint, device=args.device
    )
    print(f"[precompute] baseline ckpt = {args.baseline_checkpoint}")
    print(f"[precompute] probe_layer  = {args.probe_layer}  hidden_dim = {HIDDEN_DIM}")

    train_ds, _ = make_datasets(Path(args.data_dir), tokenizer, args.max_seq_len)
    print(f"[precompute] train rows  = {len(train_ds)}")
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate_fn, num_workers=2)

    # Aggregate per-state: running sum + count, then mean. fp32 to avoid bf16
    # accumulation drift.
    sums:   "dict[tuple, torch.Tensor]" = defaultdict(
        lambda: torch.zeros(HIDDEN_DIM, dtype=torch.float32)
    )
    counts: "dict[tuple, int]" = defaultdict(int)

    n_batches = 0
    n_positions = 0
    t0 = time.time()
    model.eval()
    with torch.no_grad():
        for batch in loader:
            in_dev = model.get_input_embeddings().weight.device
            outputs = model(
                input_ids=batch["input_ids"].to(in_dev),
                attention_mask=batch["attention_mask"].to(in_dev),
                output_hidden_states=True,
                use_cache=False,
            )
            h = outputs.hidden_states[args.probe_layer]   # (B, T, H), bf16
            for (b, t, state) in batch["supervised_positions"]:
                key = tuple(int(x) for x in state)
                vec = h[b, t].detach().to(torch.float32).cpu()
                sums[key]   = sums[key] + vec
                counts[key] = counts[key] + 1
                n_positions += 1
            n_batches += 1
            if n_batches % 20 == 0:
                print(f"  [precompute] batch {n_batches}/{len(loader)}  "
                      f"sup_positions_so_far={n_positions}  "
                      f"unique_states={len(sums)}  "
                      f"elapsed={time.time() - t0:.1f}s")

    means = {k: (sums[k] / max(counts[k], 1)).contiguous() for k in sums}
    print(f"[precompute] done. unique states={len(means)}  total positions={n_positions}  "
          f"elapsed={time.time() - t0:.1f}s")
    # Norm distribution for a sanity check.
    norms = torch.stack([v.norm() for v in means.values()])
    print(f"[precompute] ‖h_target‖: mean={norms.mean():.2f}  "
          f"std={norms.std():.2f}  min={norms.min():.2f}  max={norms.max():.2f}")

    torch.save(
        {"states": means, "counts": dict(counts), "probe_layer": args.probe_layer,
         "baseline_checkpoint": args.baseline_checkpoint},
        out_path,
    )
    # Also dump a small json summary for at-a-glance inspection.
    summary = {
        "n_states": len(means),
        "n_positions": n_positions,
        "probe_layer": args.probe_layer,
        "baseline_checkpoint": args.baseline_checkpoint,
        "states_sample": [list(k) for k in list(means.keys())[:10]],
        "norm_stats": {
            "mean": float(norms.mean()), "std": float(norms.std()),
            "min": float(norms.min()), "max": float(norms.max()),
        },
    }
    out_path.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    print(f"[precompute] saved → {out_path}")
    print(f"[precompute] saved summary → {out_path.with_suffix('.json')}")


if __name__ == "__main__":
    main()
