"""Solve-rate evaluation: stratified test subset, greedy decode, optimality classification."""

from __future__ import annotations

import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from eval.optimality import classify_moves  # noqa: E402
from hanoi_data.template import get_prompts  # noqa: E402
from planning import _extract_moves_block, _parse_moves_json  # noqa: E402

# Regex captures everything from "Solution:" to end of generated text; the
# bracket-balanced extractor then pulls out the outer list.
_SOLUTION_HEADER = re.compile(r"Solution\s*:", re.IGNORECASE)


def build_prompt(tokenizer, s_I, s_G) -> str:
    """Apply the model's chat template to the canonical (system, user) pair."""
    sys_p, user_p = get_prompts(s_I, s_G)
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": sys_p},
         {"role": "user", "content": user_p}],
        tokenize=False,
        add_generation_prompt=True,
    )


def extract_solution(text: str) -> Optional[List[List[int]]]:
    m = _SOLUTION_HEADER.search(text)
    if m is None:
        return None
    after = text[m.end():]
    block = _extract_moves_block("moves = " + after)  # synthesize header for the extractor
    if block is None:
        return None
    try:
        return _parse_moves_json(block)
    except Exception:
        return None


def stratified_subset(rows: List[Dict], n: int, seed: int) -> List[Dict]:
    by_len: Dict[int, List[Dict]] = defaultdict(list)
    for r in rows:
        by_len[len(r["moves"])].append(r)
    rng = random.Random(seed)
    for v in by_len.values():
        rng.shuffle(v)
    out: List[Dict] = []
    total = sum(len(v) for v in by_len.values())
    for L, bucket in by_len.items():
        # proportional allocation, at least 1 per stratum if non-empty
        take = max(1, round(len(bucket) * n / total)) if bucket else 0
        out.extend(bucket[:take])
    if len(out) > n:
        rng.shuffle(out)
        out = out[:n]
    return out


def _model_input_device(model) -> torch.device:
    """Where to send input_ids under device_map='auto'."""
    return model.get_input_embeddings().weight.device


@torch.no_grad()
def _greedy_generate_batched(
    model, tokenizer, prompts: List[str], max_new_tokens: int,
) -> List[str]:
    """Batched greedy generation. Left-pads so all prompts end at the same
    position; decode each row's new tokens after the original prompt length.
    """
    saved_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "left"
    try:
        enc = tokenizer(prompts, return_tensors="pt", padding=True,
                        add_special_tokens=False)
    finally:
        tokenizer.padding_side = saved_side
    device = _model_input_device(model)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    gen = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    prompt_len = input_ids.shape[1]
    new_tok_batch = gen[:, prompt_len:]
    return tokenizer.batch_decode(new_tok_batch, skip_special_tokens=True)


def run_solve_eval(
    model,
    tokenizer,
    test_rows: List[Dict],
    max_new_tokens: int,
    subset_size: int,
    seed: int,
    out_dir: Path,
    batch_size: int = 8,
) -> Dict:
    sub = stratified_subset(test_rows, subset_size, seed)
    per_problem: List[Dict] = []
    cat_counter: Counter = Counter()
    by_len_cats: Dict[int, Counter] = defaultdict(Counter)
    bs = max(1, batch_size)

    i = 0
    while i < len(sub):
        chunk = sub[i: i + bs]
        prompts = [build_prompt(tokenizer, tuple(r["s_I"]), tuple(r["s_G"])) for r in chunk]
        gen_texts = _greedy_generate_batched(model, tokenizer, prompts, max_new_tokens)

        for j, (row, gen_text) in enumerate(zip(chunk, gen_texts)):
            s_I = tuple(row["s_I"])
            s_G = tuple(row["s_G"])
            moves = extract_solution(gen_text)
            cls = classify_moves(s_I, s_G, moves)
            cat_counter[cls["category"]] += 1
            by_len_cats[cls["optimal_len"]][cls["category"]] += 1

            per_problem.append({
                "s_I": list(s_I), "s_G": list(s_G),
                "optimal_len": cls["optimal_len"],
                "generated_moves": moves,
                **{k: v for k, v in cls.items() if k != "optimal_len"},
            })

        i += bs
        running = " ".join(f"{k}={cat_counter[k]}" for k in ("Optimal", "Suboptimal", "Incorrect", "Illegal"))
        print(f"  [solve] {min(i, len(sub)):4d}/{len(sub)}  {running}")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "solve_results.json").write_text(json.dumps({
        "n": len(sub),
        "category_counts": dict(cat_counter),
        "by_optimal_length": {str(L): dict(c) for L, c in sorted(by_len_cats.items())},
        "per_problem": per_problem,
    }, indent=2))

    return {
        "n": len(sub),
        "category_counts": dict(cat_counter),
        "by_optimal_length": {str(L): dict(c) for L, c in sorted(by_len_cats.items())},
    }
