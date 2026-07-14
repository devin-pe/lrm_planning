#!/usr/bin/env python3
"""Stage 1: vLLM generation + solution validation for ToH probing.

This script uses vLLM only (no HF model forward passes) to generate full responses
for all 81 4-disk ToH start states, validates solutions, and writes:
- outputs/qwen_probe/generated_texts.json
- outputs/qwen_probe/validation_results.json
- outputs/qwen_probe/valid_states.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from planning import MoveParser, NonStandardValidator
from prompts import create_nonstandard_prompt
from transformers import AutoTokenizer

try:
    from vllm import LLM, SamplingParams
except ImportError as e:  # pragma: no cover
    raise SystemExit("vLLM is required for generate.py. Install with: pip install vllm") from e

State = Tuple[int, int, int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 1 vLLM generation + validation")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.6-27B")
    parser.add_argument("--tokenizer", type=str, default="Qwen/Qwen3.6-27B")
    parser.add_argument("--n_disks", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="outputs/qwen_probe")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--max_model_len", type=int, default=262144)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--download_dir", type=str, default=None)
    parser.add_argument("--max_tokens", type=int, default=32000)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--min_p", type=float, default=0.0)
    parser.add_argument("--presence_penalty", type=float, default=1.5)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--disable_guided_regex",
        action="store_true",
        help="Disable vLLM guided regex decoding for final output format.",
    )
    parser.add_argument(
        "--hide_reasoning",
        action="store_true",
        help=(
            "Use vLLM reasoning parser mode (reasoning hidden from generated_texts). "
            "By default reasoning is kept visible in generated_texts."
        ),
    )
    return parser.parse_args()


def build_final_output_regex(n_disks: int) -> str:
    if n_disks <= 9:
        disk_pat = rf"[1-{n_disks}]"
    else:
        disk_pat = r"[1-9][0-9]*"

    peg_pat = r"[0-2]"
    ws = r"[ \t]*"
    move_pat = rf"\[{ws}{disk_pat}{ws},{ws}{peg_pat}{ws},{ws}{peg_pat}{ws}\]"
    # Require at least one move; this blocks frequent `moves = []` failures.
    moves_pat = rf"moves{ws}={ws}\[{ws}{move_pat}(?:{ws},{ws}{move_pat})*{ws}\]"
    return rf"(?s)^.*?{moves_pat}\n?$"


def create_sampling_params(args: argparse.Namespace) -> SamplingParams:
    sampling_kwargs = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "min_p": args.min_p,
        "presence_penalty": args.presence_penalty,
        "repetition_penalty": args.repetition_penalty,
        "max_tokens": args.max_tokens,
    }

    if args.disable_guided_regex:
        print("[INFO] Guided regex decoding disabled.")
        return SamplingParams(**sampling_kwargs)

    guided_regex = build_final_output_regex(args.n_disks)

    try:
        print("[INFO] Enabling guided regex decoding via SamplingParams(guided_regex=...).")
        return SamplingParams(**sampling_kwargs, guided_regex=guided_regex)
    except TypeError:
        pass

    try:
        from vllm.sampling_params import StructuredOutputsParams  # type: ignore

        so = StructuredOutputsParams(regex=guided_regex)
        print("[INFO] Enabling guided regex decoding via SamplingParams(structured_outputs=...).")
        return SamplingParams(**sampling_kwargs, structured_outputs=so)
    except Exception:
        pass

    try:
        from vllm.sampling_params import GuidedDecodingParams  # type: ignore

        guided = GuidedDecodingParams(regex=guided_regex)
        print("[INFO] Enabling guided regex decoding via SamplingParams(guided_decoding=...).")
        return SamplingParams(**sampling_kwargs, guided_decoding=guided)
    except Exception as e:
        raise RuntimeError(
            "This vLLM build does not support guided regex decoding. "
            "Upgrade vLLM or run with --disable_guided_regex."
        ) from e


def state_key(state: State) -> str:
    return str(tuple(int(x) for x in state))


def enumerate_states(n_disks: int) -> List[State]:
    return [tuple(int(v) for v in s) for s in product(range(3), repeat=n_disks)]


def state_tuple_to_pegs(state: State, n_disks: int) -> List[List[int]]:
    pegs = [[], [], []]
    for disk in range(n_disks, 0, -1):
        peg = int(state[disk - 1])
        pegs[peg].append(disk)
    return pegs


def extract_post_think_text(generated_text: str) -> str:
    think_pos = generated_text.find("</think>")
    if think_pos < 0:
        return generated_text
    return generated_text[think_pos + len("</think>") :]


def build_prompts(tokenizer: AutoTokenizer, states: Sequence[State], n_disks: int) -> Tuple[List[str], List[List[int]]]:
    prompts: List[str] = []
    state_pegs: List[List[int]] = []
    goal_pegs = [[], [], list(range(n_disks, 0, -1))]

    for idx, st in enumerate(states):
        pegs = state_tuple_to_pegs(st, n_disks=n_disks)
        state_pegs.append(pegs)
        system_prompt, user_prompt, _ = create_nonstandard_prompt(
            num_disks=n_disks,
            problem_id=idx,
            seed=0,
            initial_state_override=pegs,
            goal_state_override=goal_pegs,
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(prompt_text)

    return prompts, state_pegs


def validate_outputs(
    states: Sequence[State],
    state_pegs: Sequence[List[int]],
    generated_texts: Dict[str, str],
    n_disks: int,
) -> Tuple[List[Dict[str, object]], List[State]]:
    validator = NonStandardValidator()
    parser = MoveParser()
    goal_state = [[], [], list(range(n_disks, 0, -1))]

    validation_rows: List[Dict[str, object]] = []
    valid_states: List[State] = []

    for st, pegs in zip(states, state_pegs):
        key = state_key(st)
        text = generated_texts[key]
        post = extract_post_think_text(text)

        reason: Optional[str] = None
        passed = False
        model_move_count: Optional[int] = None
        optimal_move_count: Optional[int] = None
        legality_ok = False
        reaches_goal_ok = False
        optimal_ok = False

        parsed_moves = parser.parse_final_moves(post)
        if parsed_moves is not None:
            model_move_count = len(parsed_moves)

        val = validator.validate(
            response=post,
            initial_state=pegs,
            goal_state=goal_state,
            num_disks=n_disks,
        )

        optimal_move_count = val.get("optimal_moves")
        if not val.get("success", False):
            reason = "parse error"
        else:
            violations = int(val.get("violations", 0))
            solved = bool(val.get("solved", False))
            is_optimal = bool(val.get("is_optimal", False))

            legality_ok = violations == 0
            reaches_goal_ok = solved
            optimal_ok = is_optimal
            model_move_count = int(val.get("num_moves", model_move_count or 0))

            if not legality_ok:
                reason = "illegal move"
            elif not reaches_goal_ok:
                reason = "wrong goal"
            elif not optimal_ok:
                reason = "suboptimal"
            else:
                passed = True

        if passed:
            valid_states.append(st)

        validation_rows.append(
            {
                "state_tuple": list(st),
                "passed": passed,
                "failure_reason": reason,
                "model_move_count": model_move_count,
                "optimal_move_count": optimal_move_count,
                "legality_ok": legality_ok,
                "reaches_goal_ok": reaches_goal_ok,
                "optimal_ok": optimal_ok,
            }
        )

    return validation_rows, valid_states


def main() -> None:
    args = parse_args()
    if args.n_disks != 4:
        raise ValueError("This pipeline is configured for 4 disks")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    print("[INFO] Enumerating 81 states")
    states = enumerate_states(args.n_disks)
    prompts, state_pegs = build_prompts(tokenizer=tokenizer, states=states, n_disks=args.n_disks)

    print("[INFO] Initializing vLLM engine")
    t0 = time.time()
    llm_kwargs = {
        "model": args.model,
        "tokenizer": args.tokenizer,
        "tensor_parallel_size": args.tensor_parallel_size,
        "dtype": args.dtype,
        "max_model_len": args.max_model_len,
        "trust_remote_code": True,
        "seed": args.seed,
    }
    if args.download_dir:
        llm_kwargs["download_dir"] = args.download_dir

    if args.hide_reasoning:
        try:
            from vllm.config.structured_outputs import StructuredOutputsConfig
        except Exception as e:
            raise RuntimeError(
                "Reasoning is required, but this vLLM build does not expose "
                "StructuredOutputsConfig."
            ) from e

        so_cfg = StructuredOutputsConfig(reasoning_parser="qwen3", enable_in_reasoning=True)
        try:
            llm = LLM(**llm_kwargs, structured_outputs_config=so_cfg)
            print("[INFO] Reasoning enabled via structured_outputs_config(reasoning_parser='qwen3').")
            print("[INFO] Reasoning visibility: hidden (parser mode).")
        except TypeError as e:
            raise RuntimeError(
                "Reasoning is required, but this vLLM build does not support "
                "structured_outputs_config with reasoning_parser."
            ) from e
    else:
        llm = LLM(**llm_kwargs)
        print("[INFO] Reasoning visibility: visible in raw generated_texts (no reasoning parser mode).")

    sampling_params = create_sampling_params(args)

    print("[INFO] Running batched generation for 81 prompts")
    try:
        outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
    except TypeError:
        outputs = llm.generate(prompts, sampling_params)

    generated_texts: Dict[str, str] = {}
    for i, out in enumerate(outputs):
        if not out.outputs:
            text = ""
        else:
            text = out.outputs[0].text
        generated_texts[state_key(states[i])] = text

    print("[INFO] Validating generated solutions")
    validation_rows, valid_states = validate_outputs(
        states=states,
        state_pegs=state_pegs,
        generated_texts=generated_texts,
        n_disks=args.n_disks,
    )

    n_valid = len(valid_states)
    n_total = len(states)
    print(f"[INFO] Total problems: {n_total}")
    print(f"[INFO] Passed validation: {n_valid}")
    print(f"[INFO] Failed validation: {n_total - n_valid}")

    for row in validation_rows:
        if not bool(row["passed"]):
            print(f"[WARN] state={row['state_tuple']} failed: {row['failure_reason']}")

    (out_dir / "generated_texts.json").write_text(json.dumps(generated_texts, indent=2), encoding="utf-8")
    (out_dir / "validation_results.json").write_text(json.dumps(validation_rows, indent=2), encoding="utf-8")
    (out_dir / "valid_states.json").write_text(
        json.dumps([list(s) for s in valid_states], indent=2),
        encoding="utf-8",
    )

    summary = {
        "model": args.model,
        "tokenizer": args.tokenizer,
        "n_states": n_total,
        "n_valid": n_valid,
        "n_failed": n_total - n_valid,
        "max_tokens": args.max_tokens,
        "max_model_len": args.max_model_len,
        "download_dir": args.download_dir,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "min_p": args.min_p,
        "presence_penalty": args.presence_penalty,
        "repetition_penalty": args.repetition_penalty,
        "hide_reasoning": args.hide_reasoning,
        "runtime_seconds": time.time() - t0,
    }
    (out_dir / "generation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[INFO] Saved outputs to {out_dir}")


if __name__ == "__main__":
    main()
