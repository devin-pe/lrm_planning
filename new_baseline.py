"""
New Baseline Evaluation: Free Reasoning Model via OpenRouter on Towers of Hanoi

This script generates 10 random TOH problems per disk count (3-5 disks) with
valid random start/goal states and queries a reasoning model through OpenRouter.
"""

import os
import re
import json
import time
from datetime import datetime
from openai import OpenAI
from prompts import SYSTEM_PROMPT, create_nonstandard_prompt


# ============================================================================
# Configuration
# ============================================================================

NUM_PROBLEMS = 10
DISK_RANGE = [3, 4, 5]
SEED = 42

# OpenRouter configuration
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
MODEL_NAME = "deepseek/deepseek-r1-0528"
BASE_URL = "https://openrouter.ai/api/v1"

# Output configuration
OUTPUT_DIR = "./new_baseline_results"
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")


# ============================================================================
# Client Setup
# ============================================================================

client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url=BASE_URL,
)


# ============================================================================
# Problem Generation
# ============================================================================

def generate_problems():
    """Generate 10 TOH problems stratified across 3-5 disks (3+3+4 or similar)."""
    # Stratified: 10 problems across 3 disk counts -> 3 each + 1 extra for the last
    counts_per_disk = {d: NUM_PROBLEMS // len(DISK_RANGE) for d in DISK_RANGE}
    remainder = NUM_PROBLEMS - sum(counts_per_disk.values())
    for d in DISK_RANGE[:remainder]:
        counts_per_disk[d] += 1

    problems = []
    global_id = 0
    for num_disks in DISK_RANGE:
        for i in range(counts_per_disk[num_disks]):
            system_prompt, user_prompt, problem_info = create_nonstandard_prompt(
                num_disks=num_disks,
                problem_id=i,
                seed=SEED,
            )
            problems.append({
                "problem_id": global_id,
                "num_disks": num_disks,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "problem_info": problem_info,
            })
            global_id += 1
    return problems


# ============================================================================
# Solution Generation
# ============================================================================

def extract_think_tags(text: str) -> tuple[str, str]:
    """
    Extract reasoning from <think> tags and return (reasoning, answer).
    If no think tags found, return ("", original_text).
    """
    pattern = r"<think>(.*?)</think>"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        reasoning = "\n\n".join(m.strip() for m in matches)
        answer = re.sub(pattern, "", text, flags=re.DOTALL).strip()
        return reasoning, answer
    return "", text


def query_model(system_prompt: str, user_prompt: str) -> dict:
    """
    Query the reasoning model via OpenRouter.

    Returns:
        Dict with 'reasoning', 'answer', and 'usage' fields.
    """
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=1.0,
            max_completion_tokens=32_000,
            extra_headers={
                "HTTP-Referer": "https://github.com/lrm-planning",
                "X-Title": "LRM Planning Baseline",
            },
            extra_body={
                "reasoning": {
                    "effort": "high",
                },
            },
        )

        raw_content = response.choices[0].message.content or ""

        # Try to get reasoning from reasoning_details array (OpenRouter format)
        message = response.choices[0].message
        reasoning_details = (
            getattr(message, "reasoning_details", None)
            or (message.model_extra or {}).get("reasoning_details", None)
        )
        reasoning = ""
        if reasoning_details and isinstance(reasoning_details, list):
            # Each item has a 'content' (or 'text') field with the reasoning text
            parts = []
            for detail in reasoning_details:
                if isinstance(detail, dict):
                    text = detail.get("content") or detail.get("text") or ""
                else:
                    text = getattr(detail, "content", "") or getattr(detail, "text", "") or ""
                if text:
                    parts.append(text.strip())
            reasoning = "\n\n".join(parts)

        # Fallback: try reasoning_content attribute
        if not reasoning:
            reasoning = getattr(message, "reasoning_content", None) or ""

        answer = raw_content

        # If still no reasoning, parse <think> tags from content
        if not reasoning:
            reasoning, answer = extract_think_tags(raw_content)

        finish_reason = response.choices[0].finish_reason or "unknown"

        usage = {
            "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
            "completion_tokens": response.usage.completion_tokens if response.usage else 0,
            "total_tokens": response.usage.total_tokens if response.usage else 0,
        }

        return {
            "reasoning": reasoning,
            "answer": answer,
            "raw_content": raw_content,
            "finish_reason": finish_reason,
            "usage": usage,
        }

    except Exception as e:
        print(f"  Error querying model: {e}")
        return {"reasoning": "", "answer": "", "usage": {}, "error": str(e)}


# ============================================================================
# Main
# ============================================================================

def main():
    if not OPENROUTER_API_KEY:
        print("ERROR: Set OPENROUTER_API_KEY environment variable.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    run_dir = os.path.join(OUTPUT_DIR, f"run_{TIMESTAMP}")
    os.makedirs(run_dir, exist_ok=True)

    # Generate problems
    problems = generate_problems()

    # Save problems manifest
    manifest_path = os.path.join(run_dir, "problems.json")
    with open(manifest_path, "w") as f:
        json.dump(problems, f, indent=2)
    print(f"Saved {len(problems)} problems to {manifest_path}")

    total_problems = len(problems)

    # Query model for each problem
    results = []
    for p in problems:
        pid = p["problem_id"]
        print(f"\n[Problem {pid + 1}/{total_problems}] "
              f"disks={p['num_disks']} "
              f"initial={p['problem_info']['initial_state']} "
              f"goal={p['problem_info']['goal_state']}")

        start_time = time.time()
        response = query_model(p["system_prompt"], p["user_prompt"])
        elapsed = time.time() - start_time

        result = {
            **p,
            "response": response,
            "elapsed_seconds": round(elapsed, 2),
        }
        results.append(result)

        # Save individual result
        out_path = os.path.join(run_dir, f"problem_{pid:03d}.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)

        print(f"  Time: {elapsed:.1f}s | Tokens: {response.get('usage', {}).get('total_tokens', '?')} | Finish: {response.get('finish_reason', '?')}")

        # Brief pause to be nice to the free tier
        time.sleep(2)

    # Save combined results
    combined_path = os.path.join(run_dir, "all_results.json")
    with open(combined_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll results saved to {combined_path}")


if __name__ == "__main__":
    main()
