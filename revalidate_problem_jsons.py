import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

from planning import TowersOfHanoiValidator, NonStandardValidator


def _load_response_text(record: Dict) -> str:
    response = record.get("response", {})
    return response.get("raw_content") or response.get("answer") or ""


def _validate_standard(record: Dict, validator: TowersOfHanoiValidator) -> Tuple[Dict, bool]:
    num_disks = record.get("num_disks") or record.get("problem_info", {}).get("num_disks", 3)
    goal_peg = record.get("goal_peg", record.get("problem_info", {}).get("goal_peg", 2))
    response_text = _load_response_text(record)

    parse_ok = True
    num_moves = 0
    try:
        moves = validator.parse_moves(response_text)
        num_moves = len(moves)
    except ValueError:
        parse_ok = False

    reward, violations = validator.validate_trace(
        response_text,
        {"num_disks": num_disks, "goal_peg": goal_peg},
    )

    solved = reward >= 1.0
    validation = {
        "success": parse_ok,
        "violations": violations,
        "num_moves": num_moves,
        "solved": solved,
        "reward": reward,
    }
    return validation, solved


def _validate_nonstandard(record: Dict, validator: NonStandardValidator) -> Tuple[Dict, bool]:
    response_text = _load_response_text(record)
    info = record.get("problem_info", {})
    initial_state = record.get("initial_state", info.get("initial_state"))
    goal_state = record.get("goal_state", info.get("goal_state"))
    num_disks = record.get("num_disks", info.get("num_disks"))

    validation = validator.validate(response_text, initial_state, goal_state, num_disks)
    return validation, bool(validation.get("solved", False))


def revalidate_directory(dir_path: Path) -> Dict:
    std_validator = TowersOfHanoiValidator()
    nonstd_validator = NonStandardValidator()

    problem_files = sorted(dir_path.glob("problem_*.json"))
    if not problem_files:
        return {"updated": 0, "total": 0, "solved": 0}

    updated = 0
    solved_count = 0

    for file_path in problem_files:
        with file_path.open() as f:
            record = json.load(f)

        mode = (record.get("config_type") or record.get("problem_info", {}).get("config_type") or "standard").lower()
        if mode == "standard":
            validation, solved = _validate_standard(record, std_validator)
        else:
            validation, solved = _validate_nonstandard(record, nonstd_validator)

        old_validation = record.get("validation")
        record["validation"] = validation

        if old_validation != validation:
            updated += 1

        if solved:
            solved_count += 1

        with file_path.open("w") as f:
            json.dump(record, f, indent=2)

    all_results_path = dir_path / "all_results.json"
    if all_results_path.exists():
        records = []
        for file_path in problem_files:
            with file_path.open() as f:
                records.append(json.load(f))
        with all_results_path.open("w") as f:
            json.dump(records, f, indent=2)

    return {"updated": updated, "total": len(problem_files), "solved": solved_count}


def main() -> None:
    parser = argparse.ArgumentParser(description="Revalidate problem_*.json files and refresh validation fields.")
    parser.add_argument("dirs", nargs="+", help="Directories containing problem_*.json files")
    args = parser.parse_args()

    for folder in args.dirs:
        dir_path = Path(folder)
        stats = revalidate_directory(dir_path)
        print(f"{dir_path}: updated={stats['updated']}/{stats['total']}, solved={stats['solved']}/{stats['total']}")


if __name__ == "__main__":
    main()
