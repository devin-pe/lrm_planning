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

# Regex captures everything from "moves = " to end of generated text; the
# bracket-balanced extractor then pulls out the outer list.
_SOLUTION_HEADER = re.compile(r"\bmoves\s*=\s*\[", re.IGNORECASE)


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
    # _extract_moves_block already does the right thing: it finds the LAST
    # complete `moves = [...]` block in the text (bracket-balanced).
    block = _extract_moves_block(text)
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
    leftovers: List[Dict] = []
    total = sum(len(v) for v in by_len.values())
    for L, bucket in by_len.items():
        # proportional allocation, at least 1 per stratum if non-empty
        take = max(1, round(len(bucket) * n / total)) if bucket else 0
        take = min(take, len(bucket))
        out.extend(bucket[:take])
        leftovers.extend(bucket[take:])
    if len(out) > n:
        rng.shuffle(out)
        out = out[:n]
    elif len(out) < n and leftovers:
        # per-stratum round() can leave the allocation short of n; top up to
        # exactly n from the unused remainder (deterministic via rng).
        rng.shuffle(leftovers)
        out.extend(leftovers[: n - len(out)])
    return out


def _model_input_device(model) -> torch.device:
    """Where to send input_ids under device_map='auto'."""
    return model.get_input_embeddings().weight.device


@torch.no_grad()
def _greedy_generate_batched(
    model, tokenizer, prompts: List[str], max_new_tokens: int,
) -> Tuple[List[str], List[bool]]:
    """Batched greedy generation. Returns (decoded_texts, truncated_flags).
    `truncated_flags[i]` is True iff that row's completion ran into the
    `max_new_tokens` cap without ever emitting an EOS token.
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
    # Truncation = row never emitted EOS within max_new_tokens.
    eos_id = tokenizer.eos_token_id
    truncated = [
        bool(eos_id is None or (new_tok_batch[i] != eos_id).all().item())
        for i in range(new_tok_batch.shape[0])
    ]
    return tokenizer.batch_decode(new_tok_batch, skip_special_tokens=True), truncated


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
    n_truncated = 0
    n_format_invalid = 0  # subset: truncated AND no parsed `moves = […]`

    i = 0
    cats_for_log = ("Optimal", "Suboptimal", "Incorrect", "Illegal_format", "Illegal_moves")
    while i < len(sub):
        chunk = sub[i: i + bs]
        prompts = [build_prompt(tokenizer, tuple(r["s_I"]), tuple(r["s_G"])) for r in chunk]
        gen_texts, truncated_flags = _greedy_generate_batched(
            model, tokenizer, prompts, max_new_tokens,
        )

        for j, (row, gen_text, was_truncated) in enumerate(
            zip(chunk, gen_texts, truncated_flags)
        ):
            s_I = tuple(row["s_I"])
            s_G = tuple(row["s_G"])
            moves = extract_solution(gen_text)
            cls = classify_moves(s_I, s_G, moves)
            if (i + j) < 2:
                tail = gen_text[-200:].replace("\n", "\\n")
                print(f"  [solve/sanity] prob {i+j}  s_I={s_I} s_G={s_G}  "
                      f"len(gen_text)={len(gen_text)}  parsed_moves={moves}")
                print(f"  [solve/sanity]   tail: …{tail!r}")
            cat_counter[cls["category"]] += 1
            by_len_cats[cls["optimal_len"]][cls["category"]] += 1
            if was_truncated:
                n_truncated += 1
            # format_invalid: ran out of tokens BEFORE the model emitted
            # `moves = […]`. Distinct from the model emitting a malformed
            # move sequence inside an otherwise-present block (that's
            # Illegal_moves).
            this_format_invalid = was_truncated and moves is None
            if this_format_invalid:
                n_format_invalid += 1

            per_problem.append({
                "s_I": list(s_I), "s_G": list(s_G),
                "optimal_len": cls["optimal_len"],
                "generated_moves": moves,
                "truncated": was_truncated,
                "format_invalid": this_format_invalid,
                **{k: v for k, v in cls.items() if k != "optimal_len"},
            })

        i += bs
        running = " ".join(f"{k}={cat_counter[k]}" for k in cats_for_log)
        print(f"  [solve] {min(i, len(sub)):4d}/{len(sub)}  {running}  "
              f"trunc={n_truncated}  fmt_invalid={n_format_invalid}")

    if n_truncated:
        print(f"\n[solve] WARNING: {n_truncated}/{len(sub)} completions hit "
              f"max_new_tokens={max_new_tokens} without emitting EOS  "
              f"({n_format_invalid} of those never emitted a `moves = […]` block — "
              f"counted as Illegal_format).")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "solve_results.json").write_text(json.dumps({
        "n": len(sub),
        "category_counts": dict(cat_counter),
        "by_optimal_length": {str(L): dict(c) for L, c in sorted(by_len_cats.items())},
        "n_truncated": n_truncated,
        "n_format_invalid": n_format_invalid,
        "max_new_tokens": max_new_tokens,
        "per_problem": per_problem,
    }, indent=2))

    return {
        "n": len(sub),
        "category_counts": dict(cat_counter),
        "by_optimal_length": {str(L): dict(c) for L, c in sorted(by_len_cats.items())},
        "n_truncated": n_truncated,
        "n_format_invalid": n_format_invalid,
        "max_new_tokens": max_new_tokens,
    }
