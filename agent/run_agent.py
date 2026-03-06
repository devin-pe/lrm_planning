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

from typing import Dict, List, Optional

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import Annotated, TypedDict

from dotenv import load_dotenv

load_dotenv()

# Import prompts and problem generation from the shared module
from prompts import CODE_SYSTEM_PROMPT, SYSTEM_PROMPT
from planning import BaselineProblemGenerator
from planning import NonStandardValidator, TowersOfHanoiValidator

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
PROBLEM_MODE = "nonstandard"  # "standard" or "nonstandard"
MODEL_NAME = "deepseek/deepseek-r1"

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
    model=MODEL_NAME,
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


def _configure_agent_model(model_name: str) -> None:
    global model, graph

    model = ChatOpenAI(
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0.0,
        model=model_name,
        timeout=1200,
        stream_usage=True,
    ).bind_tools(tools, parallel_tool_calls=False)

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

def generate_problems(disk_range: list, num_problems: int, seed: int, mode: str) -> list:
    """Generate TOH problems via shared baseline generator."""
    mode = mode.strip().lower()
    if mode not in {"standard", "nonstandard"}:
        raise ValueError(f"Invalid problem mode '{mode}'. Expected 'standard' or 'nonstandard'.")

    return BaselineProblemGenerator.generate_problems(
        num_problems=num_problems,
        min_disks=min(disk_range),
        max_disks=max(disk_range),
        seed=seed,
        mode=mode,
    )


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


def validate_agent_solution(
    problem: Dict,
    final_answer: str,
    message_history: List[Dict],
    standard_validator: TowersOfHanoiValidator,
    nonstandard_validator: NonStandardValidator,
) -> Dict:
    response_text_for_validation = final_answer
    if message_history:
        history_text = "\n\n".join(
            str(msg.get("content", ""))
            for msg in message_history
            if msg.get("role") in {"ai", "tool"}
        )
        if history_text.strip():
            response_text_for_validation = f"{final_answer}\n\n{history_text}"

    config_type = (
        problem.get("config_type")
        or problem.get("problem_info", {}).get("config_type")
        or "nonstandard"
    ).strip().lower()

    num_disks = int(problem.get("num_disks", problem.get("problem_info", {}).get("num_disks", 3)))

    if config_type == "standard":
        goal_peg = problem.get("goal_peg", problem.get("problem_info", {}).get("goal_peg", 2))

        try:
            moves = standard_validator.parse_moves(response_text_for_validation)
            num_moves = len(moves)
            parse_ok = True
        except ValueError:
            num_moves = 0
            parse_ok = False

        reward, violations = standard_validator.validate_trace(
            response_text_for_validation,
            {"num_disks": num_disks, "goal_peg": goal_peg},
        )

        solved = reward >= 1.0
        optimal_moves = 2 ** num_disks - 1
        is_optimal = solved and violations == 0 and parse_ok and num_moves == optimal_moves

        return {
            "success": parse_ok,
            "config_type": "standard",
            "violations": violations,
            "num_moves": num_moves,
            "solved": solved,
            "reward": reward,
            "optimal_moves": optimal_moves,
            "is_optimal": is_optimal,
            "extra_moves": (num_moves - optimal_moves) if solved and parse_ok else None,
        }

    info = problem.get("problem_info", {})
    initial_state = problem.get("initial_state", info.get("initial_state"))
    goal_state = problem.get("goal_state", info.get("goal_state"))

    validation = nonstandard_validator.validate(
        response_text_for_validation,
        initial_state,
        goal_state,
        num_disks,
    )
    if "is_optimal" not in validation:
        validation["is_optimal"] = False
    validation["config_type"] = "nonstandard"
    return validation


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
    parser.add_argument(
        "--problem-mode",
        type=str,
        default=os.getenv("PROBLEM_MODE", PROBLEM_MODE),
        choices=["standard", "nonstandard"],
        help="Problem mode to generate when --problems is not provided.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=os.getenv("MODEL_NAME", MODEL_NAME),
        help="OpenRouter model name to use for the agent.",
    )
    args = parser.parse_args()

    if not os.getenv("OPENROUTER_API_KEY"):
        print("ERROR: Set OPENROUTER_API_KEY environment variable.")
        sys.exit(1)

    _configure_agent_model(args.model_name)
    print(f"Using model: {args.model_name}")

    # Load or generate problems
    if args.problems:
        print(f"Loading problems from {args.problems}")
        with open(args.problems) as f:
            problems = json.load(f)
    else:
        print(
            f"Generating {args.num_problems} {args.problem_mode} problems "
            f"for disks {args.disks}"
        )
        problems = generate_problems(args.disks, args.num_problems, args.seed, args.problem_mode)

    # Optionally limit the number of problems
    if args.max_problems is not None:
        problems = problems[:args.max_problems]

    # Create output directory
    run_dir = os.path.join(OUTPUT_DIR, f"run_{TIMESTAMP}")
    os.makedirs(run_dir, exist_ok=True)

    standard_validator = TowersOfHanoiValidator()
    nonstandard_validator = NonStandardValidator()

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
        validation = validate_agent_solution(
            problem=p,
            final_answer=result["agent_response"].get("final_answer", ""),
            message_history=result["agent_response"].get("messages", []),
            standard_validator=standard_validator,
            nonstandard_validator=nonstandard_validator,
        )
        result["validation"] = validation
        result["is_optimal"] = bool(validation.get("is_optimal", False))
        results.append(result)

        # Save individual result
        out_path = os.path.join(run_dir, f"problem_{pid:03d}.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)

        _print_flush(
            f"\n  ✓ Problem {pid + 1} done in {result['elapsed_seconds']:.1f}s  |  "
            f"Messages: {result['agent_response']['num_messages']}  |  "
            f"Solved: {validation.get('solved', False)}  |  "
            f"Optimal: {validation.get('is_optimal', False)}"
        )

    # Save combined results
    combined_path = os.path.join(run_dir, "all_results.json")
    with open(combined_path, "w") as f:
        json.dump(results, f, indent=2)

    solved_count = sum(1 for r in results if r.get("validation", {}).get("solved", False))
    optimal_count = sum(1 for r in results if r.get("validation", {}).get("is_optimal", False))

    per_disk_total = {}
    per_disk_solved = {}
    per_disk_optimal = {}
    for r in results:
        disk = r.get("num_disks")
        if disk is None:
            continue
        per_disk_total[disk] = per_disk_total.get(disk, 0) + 1
        if r.get("validation", {}).get("solved", False):
            per_disk_solved[disk] = per_disk_solved.get(disk, 0) + 1
        if r.get("validation", {}).get("is_optimal", False):
            per_disk_optimal[disk] = per_disk_optimal.get(disk, 0) + 1

    _print_flush(f"\n{'=' * 60}")
    _print_flush(f"All {total} results saved to {combined_path}")
    _print_flush(f"Solved: {solved_count}/{total} ({100 * solved_count / total:.1f}%)")
    _print_flush(f"Optimal: {optimal_count}/{total} ({100 * optimal_count / total:.1f}%)")
    if per_disk_total:
        _print_flush("Solved/Optimal by disk:")
        for disk in sorted(per_disk_total):
            solved = per_disk_solved.get(disk, 0)
            optimal = per_disk_optimal.get(disk, 0)
            total_disk = per_disk_total[disk]
            _print_flush(
                f"  {disk} disks: solved {solved}/{total_disk} ({100 * solved / total_disk:.1f}%), "
                f"optimal {optimal}/{total_disk} ({100 * optimal / total_disk:.1f}%)"
            )
    _print_flush(f"{'=' * 60}")


if __name__ == "__main__":
    main()
