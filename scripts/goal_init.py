#!/usr/bin/env python3
"""Initialize autonomous retrieval-goal state and directories."""

import argparse
from pathlib import Path
from typing import Any, Dict

from goal_common import (
    DEFAULT_WIN_MARGIN,
    FINAL_TASKS,
    PRIMARY_METRIC,
    atomic_write_json,
    ensure_dir,
    load_json,
    recursive_fill_missing,
    repo_status,
    validate_baseline_artifacts,
)


def baseline_state() -> Dict[str, Any]:
    output_dir = Path("outputs/baselines/standard_frozen")
    summary = output_dir / "results_summary.csv"
    manifest = output_dir / "baseline_manifest.json"
    validation = validate_baseline_artifacts(summary, manifest)
    if validation["valid"]:
        return {
            "status": "frozen",
            "path": str(summary),
            "manifest": str(manifest),
            "sha256": validation["sha256"],
        }
    return {"status": "missing", "path": None, "manifest": None, "sha256": None, "reason": validation["reason"]}


def default_state(max_concurrent_gpu_jobs: int, max_gpu_hours_per_batch: float) -> Dict[str, Any]:
    return {
        "phase": "BOOTSTRAP",
        "repo": repo_status(),
        "primary_metric": PRIMARY_METRIC,
        "win_margin": DEFAULT_WIN_MARGIN,
        "final_tasks": list(FINAL_TASKS),
        "baseline": baseline_state(),
        "budget": {
            "max_concurrent_gpu_jobs": max_concurrent_gpu_jobs,
            "max_gpu_hours_per_batch": max_gpu_hours_per_batch,
        },
        "current_batch": None,
        "open_jobs": [],
        "best_candidate": None,
        "last_scoreboard": None,
        "next_required_action": "Freeze or validate baseline before autonomous experiments.",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Initialize goal-control directories and state.")
    parser.add_argument("--state", default="outputs/goal/state.json")
    parser.add_argument("--max-concurrent-gpu-jobs", type=int, default=4)
    parser.add_argument("--max-gpu-hours-per-batch", type=float, default=24.0)
    parser.add_argument("--force", action="store_true", help="Rewrite state instead of filling missing fields.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for directory in [
        "outputs/goal",
        "outputs/goal/runs",
        "outputs/goal/eval",
        "outputs/baselines",
    ]:
        ensure_dir(directory)

    state_path = Path(args.state)
    defaults = default_state(args.max_concurrent_gpu_jobs, args.max_gpu_hours_per_batch)
    if state_path.exists() and not args.force:
        state = load_json(state_path)
        recursive_fill_missing(state, defaults)
        state["repo"] = repo_status()
        state["baseline"] = baseline_state()
        if "budget" not in state or not isinstance(state["budget"], dict):
            state["budget"] = defaults["budget"]
        else:
            state["budget"].setdefault("max_concurrent_gpu_jobs", args.max_concurrent_gpu_jobs)
            state["budget"].setdefault("max_gpu_hours_per_batch", args.max_gpu_hours_per_batch)
    else:
        state = defaults

    atomic_write_json(state_path, state)
    print("Initialized goal state: {}".format(state_path))
    print("Baseline status: {}".format(state["baseline"]["status"]))


if __name__ == "__main__":
    main()
