"""
Verify and report optimality statistics for Towers of Hanoi result folders.

By default, it checks these folders under new_baseline_results:
- gpt-oss-120b-flat
- kimi-k2-think-flat
- deepseek-r1-flat

For each folder, it re-validates every problem_*.json with NonStandardValidator,
then reports solved/optimal counts overall and per disk count.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

from planning import NonStandardValidator

DEFAULT_FOLDERS = [
    "new_baseline_results/gpt-oss-120b-flat",
    "new_baseline_results/kimi-k2-think-flat",
    "new_baseline_results/deepseek-r1-flat",
]

MODEL_SLUG_ALIASES = {
    "gpt-oss-120b": "gpt-oss-120b",
    "kimi-k2-thinking": "kimi-k2-think",
    "kimi-k2-think": "kimi-k2-think",
    "deepseek-r1-0528": "deepseek-r1",
    "deepseek-r1": "deepseek-r1",
}


def _load_response_text(record: Dict) -> str:
    response = record.get("response", {})
    return response.get("raw_content") or response.get("answer") or ""


def _resolve_states(record: Dict):
    info = record.get("problem_info", {})
    initial_state = record.get("initial_state", info.get("initial_state"))
    goal_state = record.get("goal_state", info.get("goal_state"))
    num_disks = record.get("num_disks", info.get("num_disks"))
    return initial_state, goal_state, num_disks


def _safe_pct(numerator: int, denominator: int) -> float:
    return (100.0 * numerator / denominator) if denominator else 0.0


def _mode_suffix(problem_mode: str) -> str:
    mode = (problem_mode or "nonstandard").strip().lower()
    return "flat" if mode == "nonstandard" else "tower"


def _model_to_slug(model_name: str) -> str:
    raw = model_name.split("/")[-1].strip().lower()
    if raw in MODEL_SLUG_ALIASES:
        return MODEL_SLUG_ALIASES[raw]

    normalized = re.sub(r"[^a-z0-9-]+", "-", raw).strip("-")
    if normalized in MODEL_SLUG_ALIASES:
        return MODEL_SLUG_ALIASES[normalized]

    return normalized


def resolve_model_folders(models: List[str], results_root: Path, problem_mode: str) -> List[Path]:
    suffix = _mode_suffix(problem_mode)
    folders: List[Path] = []

    for model in models:
        slug = _model_to_slug(model)
        candidate = results_root / f"{slug}-{suffix}"
        folders.append(candidate)

    return folders


def summarize_folder(folder: Path, validator: NonStandardValidator) -> Dict:
    files = sorted(folder.glob("problem_*.json"))

    summary = {
        "folder": str(folder),
        "total": len(files),
        "solved": 0,
        "optimal": 0,
        "parse_fail": 0,
        "per_disk": {},
    }

    for file_path in files:
        with file_path.open() as f:
            record = json.load(f)

        initial_state, goal_state, num_disks = _resolve_states(record)
        response_text = _load_response_text(record)

        validation = validator.validate(
            response_text,
            initial_state,
            goal_state,
            num_disks,
        )

        solved = bool(validation.get("solved", False))
        optimal = bool(validation.get("is_optimal", False))
        parse_failed = not bool(validation.get("success", False))

        if solved:
            summary["solved"] += 1
        if optimal:
            summary["optimal"] += 1
        if parse_failed:
            summary["parse_fail"] += 1

        per_disk = summary["per_disk"].setdefault(
            int(num_disks),
            {"total": 0, "solved": 0, "optimal": 0},
        )
        per_disk["total"] += 1
        if solved:
            per_disk["solved"] += 1
        if optimal:
            per_disk["optimal"] += 1

    return summary


def print_summary(summary: Dict) -> None:
    total = summary["total"]
    solved = summary["solved"]
    optimal = summary["optimal"]

    print("=" * 88)
    print(f"Folder: {summary['folder']}")
    print(
        f"Total: {total} | Solved: {solved}/{total} ({_safe_pct(solved, total):.1f}%) "
        f"| Optimal: {optimal}/{total} ({_safe_pct(optimal, total):.1f}%) "
        f"| Parse failures: {summary['parse_fail']}"
    )

    if summary["per_disk"]:
        print("Per-disk optimality:")
        print(f"{'Disks':<8} {'Optimal':<14} {'Solved':<14} {'Total':<8}")
        print("-" * 50)
        for disk in sorted(summary["per_disk"]):
            stats = summary["per_disk"][disk]
            print(
                f"{disk:<8} "
                f"{stats['optimal']}/{stats['total']} ({_safe_pct(stats['optimal'], stats['total']):>5.1f}%) "
                f"{stats['solved']}/{stats['total']} ({_safe_pct(stats['solved'], stats['total']):>5.1f}%) "
                f"{stats['total']:<8}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify optimality stats for result folders (problem_*.json)."
    )
    parser.add_argument(
        "folders",
        nargs="*",
        help="Folders to evaluate. If omitted, uses the 3 default flat folders.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="Model IDs (e.g., openai/gpt-oss-120b). Resolved to results folders.",
    )
    parser.add_argument(
        "--results-root",
        default="new_baseline_results",
        help="Base directory containing model result folders.",
    )
    parser.add_argument(
        "--problem-mode",
        default="nonstandard",
        choices=["standard", "nonstandard"],
        help="Used with --models to select folder suffix: nonstandard->flat, standard->tower.",
    )
    args = parser.parse_args()

    if args.models:
        folders = resolve_model_folders(
            models=args.models,
            results_root=Path(args.results_root),
            problem_mode=args.problem_mode,
        )
    else:
        folders = [Path(p) for p in (args.folders or DEFAULT_FOLDERS)]

    validator = NonStandardValidator()

    for folder in folders:
        if not folder.exists() or not folder.is_dir():
            print("=" * 88)
            print(f"Folder: {folder}")
            print("ERROR: Folder not found or not a directory")
            continue

        summary = summarize_folder(folder, validator)
        print_summary(summary)


if __name__ == "__main__":
    main()
