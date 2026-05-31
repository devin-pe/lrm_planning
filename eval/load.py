"""Load a fine-tuned checkpoint: base model + LoRA adapter.

Two checkpoint shapes are supported:
  - baseline: local adapter/ dir.
  - augmented_steer: local adapter/ dir + the train cfg records
    {baseline_checkpoint, steering_directions, probe_layer}. The hook is
    OFF at eval by default; run_eval can install it on --hook_on for the
    diagnostic eval.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Tuple

import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def load_checkpoint(
    checkpoint_dir: str | Path,
    device: str = "cuda",
) -> Tuple[object, object, dict]:
    """Returns (tokenizer, model, train_config_dict)."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ckpt = Path(checkpoint_dir)
    cfg_path = ckpt / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing {cfg_path}")
    train_cfg = json.loads(cfg_path.read_text())

    adapter_dir = ckpt / "adapter"
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Missing LoRA adapter dir: {adapter_dir}")

    base_id = train_cfg.get("model_id")
    if not base_id:
        raise ValueError(f"config.json at {cfg_path} has no model_id")

    print(f"[load] base model = {base_id}")
    tokenizer = AutoTokenizer.from_pretrained(base_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        base_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map="auto" if device == "cuda" else None,
    )
    base.config.use_cache = True  # caching on for eval; was off during training
    print(f"[load] applying adapter from {adapter_dir}")
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # augmented_steer: standalone LoRA adapter. The hook is OFF by default —
    # caller can install it via the eval --hook_on flag. We stash the
    # directions cache path so run_eval can install on demand.
    if train_cfg.get("regime") == "augmented_steer":
        model._augmented_info = {
            "probe_layer": int(train_cfg.get("probe_layer") or 36),
            "steering_directions_path": train_cfg.get("steering_directions"),
            "baseline_checkpoint": train_cfg.get("baseline_checkpoint"),
            "trained_alpha": 5.0,  # AUGMENTED_ALPHA — also recorded in train_cfg["constants"]
        }
        print(f"[load] augmented_steer: standalone LoRA adapter; hook OFF by "
              f"default. Directions cache: {train_cfg.get('steering_directions')}")

    return tokenizer, model, train_cfg
