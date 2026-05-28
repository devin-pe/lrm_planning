"""Load a fine-tuned checkpoint: base model + LoRA adapter + (optional) probe head."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional, Tuple

import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from finetune.probe import Probe  # noqa: E402


def load_checkpoint(
    checkpoint_dir: str | Path,
    device: str = "cuda",
) -> Tuple[object, object, Optional[Probe], dict]:
    """Returns (tokenizer, model, probe_or_None, train_config_dict)."""
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

    probe: Optional[Probe] = None
    if train_cfg.get("regime") == "probe":
        probe_path = ckpt / "probe.pt"
        if probe_path.exists():
            sd = torch.load(probe_path, map_location="cpu")
            probe = Probe(hidden_dim=train_cfg.get("hidden_dim", 5120))
            probe.load_state_dict(sd)
            probe.eval()
            for p in probe.parameters():
                p.requires_grad_(False)
            target_device = next(model.parameters()).device
            probe.to(device=target_device, dtype=torch.bfloat16)
            print(f"[load] probe head loaded from {probe_path}")
        else:
            print(f"[load] WARN regime=probe but no probe.pt found in {ckpt}")

    return tokenizer, model, probe, train_cfg
