"""
Extract Attempts: Use GPT-4.1-mini via OpenRouter to parse reasoning traces.

Given a problem result JSON (from new_baseline.py), this script feeds the
original reasoning trace to GPT-4.1-mini and asks it to identify every
distinct solution attempt the reasoning model explored, listing the moves
for each attempt.
"""

import os
import re
import json
import sys
import time
from datetime import datetime
from openai import OpenAI


# ============================================================================
# Configuration
# ============================================================================

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
MODEL_NAME = "openai/gpt-4.1-mini"
BASE_URL = "https://openrouter.ai/api/v1"

OUTPUT_DIR = "./extract_attempts_results"
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")


# ============================================================================
# Client Setup
# ============================================================================

client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url=BASE_URL,
)


# ============================================================================
# Prompt
# ============================================================================

EXTRACTION_SYSTEM_PROMPT = """\
You are analyzing reasoning traces from language models solving \
Towers of Hanoi puzzles.

Your task: given a model's internal reasoning trace, identify EVERY distinct \
solution attempt the model explored. An "attempt" is any sequence of moves \
the model tried, whether it was abandoned, led to a dead end, or was the \
final chosen solution.

Rules for identifying attempts:
1. An attempt starts when the model begins exploring a new sequence of moves \
   (often after phrases like "let me try", "alternatively", "let's start over", \
   "another approach", etc.).
2. An attempt ends when the model abandons it, realizes it's wrong, or \
   declares it as the solution.
3. Even partial attempts (where the model stops mid-sequence) should be \
   captured — list whatever moves were stated before the attempt was abandoned.
4. The FINAL attempt (the one the model settles on) should be marked as such.

Output format — you MUST output valid JSON and nothing else:
{
  "attempts": [
    {
      "attempt_number": 1,
      "description": "Brief description of this attempt (e.g. 'First exploration, moved disk1 to peg2 first')",
      "moves": [[disk_id, from_peg, to_peg], ...],
      "outcome": "abandoned" | "dead_end" | "incorrect" | "final_solution",
      "reason": "Why this attempt was abandoned or chosen (brief)"
    },
    ...
  ],
  "total_attempts": <number>,
  "final_moves": [[disk_id, from_peg, to_peg], ...]
}

Important:
- Each move is [disk_id, from_peg, to_peg] with 0-indexed pegs.
- Include ALL attempts, even if they share moves with other attempts.
- If the model re-derives the same sequence, count it as a separate attempt.
- "final_moves" should be the moves from the attempt marked "final_solution".\
"""


def build_user_prompt(problem_data: dict) -> str:
    """Build the user prompt from a problem result JSON."""
    problem_info = problem_data["problem_info"]
    reasoning = problem_data["response"]["reasoning"]
    answer = problem_data["response"].get("answer", "")

    user_prompt = f"""\
Here is a Towers of Hanoi problem and the reasoning trace from a model that \
tried to solve it. Please extract all solution attempts from the reasoning.

## Problem Setup
- Number of disks: {problem_info['num_disks']}
- Initial state: {problem_info['initial_state']}
- Goal state: {problem_info['goal_state']}

## Model's Reasoning Trace
{reasoning}

## Model's Final Answer
{answer}

Now extract every distinct attempt the model made at solving this problem. \
Output valid JSON only."""

    return user_prompt


# ============================================================================
# Query
# ============================================================================

def query_extractor(problem_data: dict) -> dict:
    """
    Send the reasoning trace to GPT-4.1-mini for attempt extraction.

    Returns:
        Dict with parsed attempts or error info.
    """
    user_prompt = build_user_prompt(problem_data)

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=8_000,
            extra_headers={
                "HTTP-Referer": "https://github.com/lrm-planning",
                "X-Title": "LRM Planning Extract",
            },
        )

        raw_content = response.choices[0].message.content or ""
        finish_reason = response.choices[0].finish_reason or "unknown"

        usage = {
            "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
            "completion_tokens": response.usage.completion_tokens if response.usage else 0,
            "total_tokens": response.usage.total_tokens if response.usage else 0,
        }

        # Try to parse JSON from the response
        parsed = None
        try:
            # Strip markdown code fences if present
            cleaned = re.sub(r"^```(?:json)?\s*", "", raw_content.strip())
            cleaned = re.sub(r"\s*```$", "", cleaned)
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            # Try to find JSON object in the response
            match = re.search(r"\{.*\}", raw_content, re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group())
                except json.JSONDecodeError:
                    pass

        return {
            "parsed": parsed,
            "raw_content": raw_content,
            "finish_reason": finish_reason,
            "usage": usage,
        }

    except Exception as e:
        print(f"  Error querying model: {e}")
        return {"parsed": None, "raw_content": "", "usage": {}, "error": str(e)}


# ============================================================================
# Main
# ============================================================================

def main():
    if not OPENROUTER_API_KEY:
        print("ERROR: Set OPENROUTER_API_KEY environment variable.")
        return

    # Accept a run directory or individual problem file
    if len(sys.argv) < 2:
        print("Usage: python extract_attempts.py <problem_file_or_run_dir>")
        print("  e.g. python extract_attempts.py new_baseline_results/run_20260207_122942/problem_000.json")
        print("  e.g. python extract_attempts.py new_baseline_results/run_20260207_122942/")
        return

    input_path = sys.argv[1]

    # Collect problem files
    problem_files = []
    if os.path.isdir(input_path):
        for f in sorted(os.listdir(input_path)):
            if f.startswith("problem_") and f.endswith(".json"):
                problem_files.append(os.path.join(input_path, f))
    elif os.path.isfile(input_path):
        problem_files.append(input_path)
    else:
        print(f"ERROR: {input_path} not found.")
        return

    if not problem_files:
        print("No problem files found.")
        return

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    run_dir = os.path.join(OUTPUT_DIR, f"run_{TIMESTAMP}")
    os.makedirs(run_dir, exist_ok=True)

    print(f"Processing {len(problem_files)} problem file(s)...")
    print(f"Output directory: {run_dir}")

    all_results = []
    for pf in problem_files:
        fname = os.path.basename(pf)
        print(f"\n{'='*60}")
        print(f"Processing: {fname}")

        with open(pf) as f:
            problem_data = json.load(f)

        reasoning = problem_data.get("response", {}).get("reasoning", "")
        if not reasoning:
            print("  Skipping — no reasoning trace found.")
            continue

        start_time = time.time()
        result = query_extractor(problem_data)
        elapsed = time.time() - start_time

        output = {
            "source_file": pf,
            "problem_id": problem_data.get("problem_id"),
            "num_disks": problem_data.get("num_disks", problem_data.get("problem_info", {}).get("num_disks")),
            "problem_info": problem_data.get("problem_info"),
            "extraction": result,
            "elapsed_seconds": round(elapsed, 2),
        }
        all_results.append(output)

        # Save individual extraction result
        out_path = os.path.join(run_dir, f"extraction_{fname}")
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)

        parsed = result.get("parsed")
        if parsed:
            n_attempts = parsed.get("total_attempts", len(parsed.get("attempts", [])))
            print(f"  Found {n_attempts} attempt(s)")
            for att in parsed.get("attempts", []):
                n_moves = len(att.get("moves", []))
                print(f"    Attempt {att['attempt_number']}: {n_moves} moves — {att['outcome']}")
        else:
            print("  WARNING: Could not parse structured output.")

        print(f"  Time: {elapsed:.1f}s | Tokens: {result.get('usage', {}).get('total_tokens', '?')} | Finish: {result.get('finish_reason', '?')}")

        time.sleep(1)

    # Save combined results
    combined_path = os.path.join(run_dir, "all_extractions.json")
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to {combined_path}")


if __name__ == "__main__":
    main()
