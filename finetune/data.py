"""Dataset loading + tokenisation + supervision-position mapping.

Strategy for supervised positions:
- Tokenise prompt and completion separately (avoids BPE merges across the
  prompt/completion boundary swallowing the first completion token).
- `return_offsets_mapping=True` on the completion gives character spans for
  each completion token, in completion-local coordinates — exactly what the
  supervisions field already uses.
- For each supervision span (cs, ce, target_state), pick all completion
  tokens whose offset range is contained in [cs, ce] (or strictly overlaps
  the span when no token is fully contained, which can happen near the brackets);
  use the LAST such token. Its position in input_ids is prompt_len + token_idx.
- Skip the supervision if no token survives or its absolute position would
  exceed max_seq_len.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
from torch.utils.data import Dataset

State = Tuple[int, int, int, int]


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _map_supervisions_to_tokens(
    sup_spans: Sequence[Sequence],
    comp_offsets: Sequence[Tuple[int, int]],
) -> List[Tuple[int, State]]:
    """Return list of (completion_token_idx, target_state). token_idx is local
    to the completion (caller adds prompt_len to get absolute position).
    """
    out: List[Tuple[int, State]] = []
    for cs, ce, target in sup_spans:
        last_tok: int = -1
        for ti, (s, e) in enumerate(comp_offsets):
            # Empty offsets (s == e) are special-token slots; skip.
            if s == e:
                continue
            # Token fully contained in the supervision span:
            if s >= cs and e <= ce:
                last_tok = ti
        if last_tok >= 0:
            out.append((last_tok, tuple(int(x) for x in target)))
    return out


class HanoiSFTDataset(Dataset):
    """Per-example output:
        input_ids: List[int]
        attention_mask: List[int]
        labels: List[int]   # -100 over prompt, real ids over completion
        supervised_positions: List[(abs_token_idx, target_state)]
    """

    def __init__(
        self,
        jsonl_path: Path,
        tokenizer,
        max_seq_len: int = 768,
    ):
        self.rows = _load_jsonl(Path(jsonl_path))
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.rows[idx]
        completion = row["completion"]
        sup_spans = row["supervisions"]

        # Build the chat-templated prompt from the (system, user) pair stored
        # in the JSONL. We deliberately did not bake the chat template into
        # the dataset since it's tokenizer-specific.
        prompt = self.tokenizer.apply_chat_template(
            [{"role": "system", "content": row["system_prompt"]},
             {"role": "user", "content": row["user_prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
        )

        # Separate-tokenise so the BPE boundary cannot eat completion tokens.
        prompt_ids = self.tokenizer(
            prompt,
            add_special_tokens=False,
            return_attention_mask=False,
        )["input_ids"]

        comp_enc = self.tokenizer(
            completion,
            add_special_tokens=False,
            return_offsets_mapping=True,
            return_attention_mask=False,
        )
        comp_ids = comp_enc["input_ids"]
        comp_offsets = comp_enc["offset_mapping"]

        input_ids = (prompt_ids + comp_ids)[: self.max_seq_len]
        labels = [-100] * len(prompt_ids) + list(comp_ids)
        labels = labels[: self.max_seq_len]
        attention_mask = [1] * len(input_ids)

        comp_supervised = _map_supervisions_to_tokens(sup_spans, comp_offsets)
        prompt_len = len(prompt_ids)
        supervised_positions: List[Tuple[int, State]] = []
        for ti, target in comp_supervised:
            abs_idx = prompt_len + ti
            if abs_idx < len(input_ids):
                supervised_positions.append((abs_idx, target))

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "supervised_positions": supervised_positions,
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Pad input_ids / attention_mask / labels to the batch max length, and
    flatten supervised_positions to a single list of (batch_idx, token_idx,
    target_state) tuples.
    """
    max_len = max(len(ex["input_ids"]) for ex in batch)
    input_ids = torch.zeros((len(batch), max_len), dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)

    flat_sup: List[Tuple[int, int, State]] = []
    for bi, ex in enumerate(batch):
        L = len(ex["input_ids"])
        input_ids[bi, :L] = torch.tensor(ex["input_ids"], dtype=torch.long)
        attention_mask[bi, :L] = torch.tensor(ex["attention_mask"], dtype=torch.long)
        labels[bi, :L] = torch.tensor(ex["labels"], dtype=torch.long)
        for ti, target in ex["supervised_positions"]:
            flat_sup.append((bi, ti, target))

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "supervised_positions": flat_sup,
    }


def make_datasets(data_dir: Path, tokenizer, max_seq_len: int):
    data_dir = Path(data_dir)
    train = HanoiSFTDataset(data_dir / "train.jsonl", tokenizer, max_seq_len)
    test = HanoiSFTDataset(data_dir / "test.jsonl", tokenizer, max_seq_len)
    return train, test
