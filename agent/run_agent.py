"""
Run the code-executing agent on Towers of Hanoi planning problems.

Uses the same prompts as new_baseline.py, but instead of pure reasoning,
the agent can write and execute Python code in an E2B sandbox to solve each
problem programmatically.

Usage:
    # Generate new problems and solve them
    python agent/run_agent.py

    # Load problems from a previous baseline run
    python agent/run_agent.py --problems new_baseline_results/run_20260207_122942/problems.json

    # Specify disk range and number of problems
    python agent/run_agent.py --disks 3 4 5 --num-problems 10
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Ensure project root is importable
_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from typing import List, Optional

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import Annotated, TypedDict

from dotenv import load_dotenv

load_dotenv()

# Import prompts and problem generation from the shared module
from prompts import CODE_SYSTEM_PROMPT, SYSTEM_PROMPT, create_nonstandard_prompt

# Import tools from the existing agent module
_agent_dir = str(Path(__file__).resolve().parent)
if _agent_dir not in sys.path:
    sys.path.insert(0, _agent_dir)
from agent import execute_python, run_shell, write_file, read_file, tools


# ============================================================================
# Configuration
# ============================================================================

NUM_PROBLEMS = 10
DISK_RANGE = [3, 4, 5]
SEED = 42
MAX_ITERATIONS = 10  # safety cap on agent loops per problem

OUTPUT_DIR = os.path.join(str(_root), "agent_results")
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")


# ============================================================================
# Combined System Prompt
# ============================================================================

# The agent's system prompt: code-execution capabilities + TOH puzzle rules
AGENT_SYSTEM_PROMPT = CODE_SYSTEM_PROMPT.rstrip() + "\n\n" + SYSTEM_PROMPT


# ============================================================================
# LangGraph Agent (with configurable system prompt)
# ============================================================================

class State(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]


model = ChatOpenAI(
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0.0,
    model="deepseek/deepseek-r1",
    timeout=1200,
    stream_usage=True,
).bind_tools(tools, parallel_tool_calls=False)


def agent_node(state: State):
    """LLM invocation with the combined system prompt."""
    messages = state["messages"]
    sys_msg = SystemMessage(content=AGENT_SYSTEM_PROMPT)
    response = model.invoke([sys_msg] + messages)
    return {"messages": [response]}


def tool_router(state: State):
    last = state["messages"][-1]
    if last.tool_calls:
        return "tools"
    return END


tool_node = ToolNode(tools)

graph_builder = StateGraph(State)
graph_builder.add_node("agent", agent_node)
graph_builder.add_node("tools", tool_node)
graph_builder.add_edge(START, "agent")
graph_builder.add_conditional_edges("agent", tool_router, ["tools", END])
graph_builder.add_edge("tools", "agent")
graph = graph_builder.compile()


# ============================================================================
# Problem Generation (mirrors new_baseline.py)
# ============================================================================

def generate_problems(disk_range: list, num_problems: int, seed: int) -> list:
    """Generate TOH problems stratified across disk counts."""
    counts_per_disk = {d: num_problems // len(disk_range) for d in disk_range}
    remainder = num_problems - sum(counts_per_disk.values())
    for d in disk_range[:remainder]:
        counts_per_disk[d] += 1

    problems = []
    global_id = 0
    for num_disks in disk_range:
        for i in range(counts_per_disk[num_disks]):
            system_prompt, user_prompt, problem_info = create_nonstandard_prompt(
                num_disks=num_disks,
                problem_id=i,
                seed=seed,
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
# Run Agent on a Single Problem
# ============================================================================

def _print_flush(*args, **kwargs):
    """Print with immediate flush so output appears in real time."""
    print(*args, **kwargs, flush=True)


def run_problem(problem: dict, run_dir: str = None) -> dict:
    """
    Run the code agent on a single TOH problem.

    Prints every agent / tool message in real time so the user can follow
    progress and Ctrl-C if something goes wrong.

    Returns a dict with the problem info, the agent's messages, and timing.
    """
    user_prompt = problem["user_prompt"]
    initial_messages = [HumanMessage(content=user_prompt)]

    _print_flush("  Invoking agent...")
    start = time.time()

    # Track messages we have already printed so we only show new ones
    printed_count = 0
    final_state = None
    step = 0
    code_snippet_counter = 0  # counter for saved code files

    for event in graph.stream(
        {"messages": initial_messages},
        config={"recursion_limit": MAX_ITERATIONS * 2},
        stream_mode="values",
    ):
        final_state = event
        msgs = event.get("messages", [])

        # Print only the new messages we haven't seen yet
        for msg in msgs[printed_count:]:
            step += 1
            role = msg.type  # "human", "ai", "tool"

            if role == "human":
                _print_flush(f"\n  ── Step {step} | USER ──")
                content = getattr(msg, "content", "")
                # Truncate very long prompts in the live view
                if len(content) > 500:
                    _print_flush(f"  {content[:500]}...")
                else:
                    _print_flush(f"  {content}")

            elif role == "ai":
                content = getattr(msg, "content", "")
                tool_calls = getattr(msg, "tool_calls", [])

                if tool_calls:
                    for tc in tool_calls:
                        name = tc.get("name", "?")
                        args = tc.get("args", {})
                        _print_flush(f"\n  ── Step {step} | AGENT → tool: {name} ──")
                        # Show full tool call args (no truncation)
                        args_preview = json.dumps(args, ensure_ascii=False, indent=2)
                        _print_flush(f"  {args_preview}")

                        # Save code snippets to .py files for easy review
                        if run_dir and name in ("execute_python", "write_file"):
                            code_snippet_counter += 1
                            code_text = args.get("code") or args.get("content", "")
                            pid = problem.get("problem_id", 0)
                            snippet_path = os.path.join(
                                run_dir,
                                f"problem_{pid:03d}_code_{code_snippet_counter:03d}.py",
                            )
                            with open(snippet_path, "w") as sf:
                                sf.write(f"# Tool: {name}\n")
                                if name == "write_file":
                                    sf.write(f"# Target path: {args.get('path', '?')}\n")
                                sf.write(f"# Step: {step}\n\n")
                                sf.write(code_text)
                            _print_flush(f"  [Code saved to {snippet_path}]")
                else:
                    _print_flush(f"\n  ── Step {step} | AGENT (final answer) ──")
                    _print_flush(f"  {content}")

            elif role == "tool":
                content = getattr(msg, "content", "")
                tool_name = getattr(msg, "name", "?")
                _print_flush(f"\n  ── Step {step} | TOOL RESULT ({tool_name}) ──")
                # Show full tool output (no truncation)
                _print_flush(f"  {content}")

        printed_count = len(msgs)

    elapsed = time.time() - start

    # Extract conversation history for saving
    all_messages = final_state["messages"] if final_state else []
    serialised_messages = []
    for msg in all_messages:
        entry = {
            "role": msg.type,
            "content": getattr(msg, "content", ""),
        }
        # Preserve tool call info
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            entry["tool_calls"] = [
                {"name": tc.get("name"), "args": tc.get("args")}
                for tc in msg.tool_calls
            ]
        if hasattr(msg, "name") and msg.type == "tool":
            entry["tool_name"] = msg.name
        serialised_messages.append(entry)

    # The last AI message is the final answer
    final_answer = ""
    for msg in reversed(all_messages):
        if msg.type == "ai" and getattr(msg, "content", ""):
            final_answer = msg.content
            break

    return {
        **problem,
        "agent_response": {
            "final_answer": final_answer,
            "messages": serialised_messages,
            "num_messages": len(all_messages),
        },
        "elapsed_seconds": round(elapsed, 2),
    }


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run the code agent on TOH planning problems"
    )
    parser.add_argument(
        "--problems",
        type=str,
        default=None,
        help="Path to a problems.json file from a previous run (skips generation)",
    )
    parser.add_argument(
        "--disks",
        type=int,
        nargs="+",
        default=DISK_RANGE,
        help="Disk counts to generate problems for (default: 3 4 5)",
    )
    parser.add_argument(
        "--num-problems",
        type=int,
        default=NUM_PROBLEMS,
        help="Total number of problems to generate (default: 10)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Random seed for problem generation (default: 42)",
    )
    parser.add_argument(
        "--max-problems",
        type=int,
        default=None,
        help="Only run the first N problems (default: run all)",
    )
    args = parser.parse_args()

    if not os.getenv("OPENROUTER_API_KEY"):
        print("ERROR: Set OPENROUTER_API_KEY environment variable.")
        sys.exit(1)

    # Load or generate problems
    if args.problems:
        print(f"Loading problems from {args.problems}")
        with open(args.problems) as f:
            problems = json.load(f)
    else:
        print(f"Generating {args.num_problems} problems for disks {args.disks}")
        problems = generate_problems(args.disks, args.num_problems, args.seed)

    # Optionally limit the number of problems
    if args.max_problems is not None:
        problems = problems[:args.max_problems]

    # Create output directory
    run_dir = os.path.join(OUTPUT_DIR, f"run_{TIMESTAMP}")
    os.makedirs(run_dir, exist_ok=True)

    # Save problems manifest
    manifest_path = os.path.join(run_dir, "problems.json")
    with open(manifest_path, "w") as f:
        json.dump(problems, f, indent=2)
    print(f"Saved {len(problems)} problems to {manifest_path}")

    # Run agent on each problem
    results = []
    total = len(problems)
    for p in problems:
        pid = p["problem_id"]

        # ── Section header ──
        header = f"PROBLEM {pid + 1}/{total}"
        _print_flush(f"\n{'#' * 60}")
        _print_flush(f"#### {header} | disks={p['num_disks']} ####")
        _print_flush(f"  Initial: {p['problem_info']['initial_state']}")
        _print_flush(f"  Goal:    {p['problem_info']['goal_state']}")
        _print_flush(f"{'#' * 60}")

        result = run_problem(p, run_dir=run_dir)
        results.append(result)

        # Save individual result
        out_path = os.path.join(run_dir, f"problem_{pid:03d}.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)

        _print_flush(
            f"\n  ✓ Problem {pid + 1} done in {result['elapsed_seconds']:.1f}s  |  "
            f"Messages: {result['agent_response']['num_messages']}"
        )

    # Save combined results
    combined_path = os.path.join(run_dir, "all_results.json")
    with open(combined_path, "w") as f:
        json.dump(results, f, indent=2)
    _print_flush(f"\n{'=' * 60}")
    _print_flush(f"All {total} results saved to {combined_path}")
    _print_flush(f"{'=' * 60}")


if __name__ == "__main__":
    main()
