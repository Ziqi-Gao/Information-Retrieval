#!/usr/bin/env python3
"""Freeze a standard baseline summary for future autonomous comparisons."""

import argparse
import shutil
from pathlib import Path
from typing import Any, Dict, List

from goal_common import (
    FINAL_TASKS,
    PRIMARY_METRIC,
    atomic_write_json,
    ensure_dir,
    load_json,
    metric_float,
    now_utc,
    parse_task_list,
    read_csv_rows,
    repo_status,
    sha256_file,
)


def validate_standard_rows(rows: List[Dict[str, str]], tasks: List[str], metric: str) -> Dict[str, float]:
    by_task: Dict[str, List[Dict[str, str]]] = {task: [] for task in tasks}
    for row in rows:
        if row.get("version") == "standard" and row.get("task") in by_task:
            by_task[row.get("task")].append(row)

    values: Dict[str, float] = {}
    problems: List[str] = []
    for task in tasks:
        task_rows = by_task[task]
        if len(task_rows) != 1:
            problems.append("{} has {} standard rows; expected exactly 1".format(task, len(task_rows)))
            continue
        value = metric_float(task_rows[0].get(metric))
        if value is None:
            problems.append("{} has invalid {}".format(task, metric))
            continue
        values[task] = value
    if problems:
        raise ValueError("; ".join(problems))
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze standard baseline summary for goal scoreboard.")
    parser.add_argument("--source-summary", required=True, help="Existing CSV summary containing standard rows.")
    parser.add_argument("--output-dir", required=True, help="Baseline freeze directory.")
    parser.add_argument("--tasks", default=",".join(FINAL_TASKS), help="Required task list.")
    parser.add_argument("--metric", default=PRIMARY_METRIC)
    parser.add_argument("--force", action="store_true", help="Overwrite existing frozen summary/manifest.")
    parser.add_argument("--state", default="outputs/goal/state.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = Path(args.source_summary)
    output_dir = Path(args.output_dir)
    frozen_summary = output_dir / "results_summary.csv"
    manifest_path = output_dir / "baseline_manifest.json"
    tasks = parse_task_list(args.tasks)

    if args.metric != PRIMARY_METRIC:
        raise SystemExit("Only {} is supported for the primary protocol.".format(PRIMARY_METRIC))
    if tasks != FINAL_TASKS:
        raise SystemExit("Task list must match final protocol order: {}".format(",".join(FINAL_TASKS)))
    if not source.exists():
        print("Source summary is absent: {}".format(source))
        print("Run or locate a completed standard evaluation summary first, for example:")
        print("  python scripts/goal_freeze_baseline.py --source-summary outputs/final_grid_experiment/eval/summaries/standard_results_summary.csv --output-dir outputs/baselines/standard_frozen")
        raise SystemExit(1)

    if not args.force and (frozen_summary.exists() or manifest_path.exists()):
        raise SystemExit("Refusing to overwrite existing frozen baseline. Use --force only if intentional.")

    rows = read_csv_rows(source)
    try:
        per_task = validate_standard_rows(rows, tasks, args.metric)
    except ValueError as exc:
        raise SystemExit("Invalid standard baseline source: {}".format(exc))

    ensure_dir(output_dir)
    shutil.copy2(str(source), str(frozen_summary))
    summary_hash = sha256_file(frozen_summary)
    manifest: Dict[str, Any] = {
        "created_at": now_utc(),
        "repo": repo_status(),
        "source_summary": str(source),
        "frozen_summary": str(frozen_summary),
        "sha256": summary_hash,
        "metric": args.metric,
        "tasks": tasks,
        "per_task_baseline": per_task,
    }
    atomic_write_json(manifest_path, manifest)

    state_path = Path(args.state)
    if state_path.exists():
        state = load_json(state_path)
        state["baseline"] = {
            "status": "frozen",
            "path": str(frozen_summary),
            "manifest": str(manifest_path),
            "sha256": summary_hash,
        }
        state["next_required_action"] = "Design and validate a manifest before autonomous experiments."
        atomic_write_json(state_path, state)

    print("Frozen baseline summary: {}".format(frozen_summary))
    print("Baseline manifest: {}".format(manifest_path))
    print("SHA256: {}".format(summary_hash))


if __name__ == "__main__":
    main()
