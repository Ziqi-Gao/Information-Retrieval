#!/usr/bin/env python3
"""Compare collected candidates against the frozen standard baseline."""

import argparse
import statistics
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

from goal_common import (
    DEFAULT_WIN_MARGIN,
    FINAL_TASKS,
    PRIMARY_METRIC,
    atomic_write_json,
    load_json,
    metric_float,
    now_utc,
    read_csv_rows,
    repo_status,
    write_csv_rows,
)


AGG_COLUMNS = [
    "candidate_id",
    "run_id",
    "tasks_total",
    "tasks_valid",
    "tasks_won",
    "tasks_lost",
    "min_delta",
    "mean_delta",
    "pass_all_tasks",
    "failure_reasons",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score candidates against a frozen standard baseline.")
    parser.add_argument("--baseline", default=None)
    parser.add_argument("--results", default=None)
    parser.add_argument("--metric", default=PRIMARY_METRIC)
    parser.add_argument("--margin", type=float, default=DEFAULT_WIN_MARGIN)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--state", default="outputs/goal/state.json")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def load_baseline(path: Path, metric: str) -> Dict[str, float]:
    rows = read_csv_rows(path)
    values: Dict[str, List[float]] = {task: [] for task in FINAL_TASKS}
    for row in rows:
        if row.get("version") != "standard":
            continue
        task = row.get("task")
        if task not in values:
            continue
        value = metric_float(row.get(metric))
        if value is not None:
            values[task].append(value)

    baseline: Dict[str, float] = {}
    problems: List[str] = []
    for task in FINAL_TASKS:
        if len(values[task]) != 1:
            problems.append("{} has {} valid standard rows; expected exactly 1".format(task, len(values[task])))
        else:
            baseline[task] = values[task][0]
    if problems:
        raise ValueError("; ".join(problems))
    return baseline


def load_candidates(path: Path, metric: str) -> Dict[str, List[Dict[str, Any]]]:
    rows = read_csv_rows(path)
    candidates: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        if row.get("metric") and row.get("metric") != metric:
            continue
        candidate_id = row.get("candidate_id") or row.get("run_id") or row.get("version")
        if not candidate_id:
            continue
        candidates.setdefault(candidate_id, []).append(row)
    return candidates


def evaluate_candidate(candidate_id: str, rows: List[Dict[str, Any]], baseline: Dict[str, float], metric: str, margin: float) -> Dict[str, Any]:
    by_task: Dict[str, List[Dict[str, Any]]] = {task: [] for task in FINAL_TASKS}
    for row in rows:
        task = row.get("task")
        if task in by_task:
            by_task[task].append(row)

    details: List[Dict[str, Any]] = []
    valid_deltas: List[float] = []
    failure_reasons: List[str] = []
    run_ids = sorted(set(row.get("run_id", "") for row in rows if row.get("run_id")))
    for task in FINAL_TASKS:
        task_rows = by_task[task]
        reason = ""
        candidate_value = None
        status = "completed"
        if len(task_rows) == 0:
            status = "missing_result"
            reason = "missing task result"
        elif len(task_rows) > 1:
            status = "invalid_metric"
            reason = "multiple rows for candidate/task"
        else:
            row = task_rows[0]
            row_status = row.get("status") or "completed"
            raw_value = row.get("value") if row.get("value") != "" else row.get(metric)
            candidate_value = metric_float(raw_value)
            if row_status != "completed":
                status = row_status
                reason = row.get("reason") or row_status
            elif candidate_value is None:
                status = "invalid_metric"
                reason = "invalid {}".format(metric)

        baseline_value = baseline[task]
        delta = None if candidate_value is None else candidate_value - baseline_value
        won_task = bool(delta is not None and delta >= margin and status == "completed")
        if delta is not None and status == "completed":
            valid_deltas.append(delta)
        if not won_task:
            if not reason and delta is not None:
                reason = "delta below margin"
            failure_reasons.append("{}: {}".format(task, reason or "not won"))
        details.append(
            {
                "candidate_id": candidate_id,
                "task": task,
                "baseline_value": baseline_value,
                "candidate_value": candidate_value,
                "delta": delta,
                "won_task": won_task,
                "status": status,
                "reason": reason,
            }
        )

    tasks_valid = len(valid_deltas)
    tasks_won = sum(1 for item in details if item["won_task"])
    tasks_lost = len(FINAL_TASKS) - tasks_won
    pass_all = tasks_won == len(FINAL_TASKS) and tasks_valid == len(FINAL_TASKS)
    aggregate = {
        "candidate_id": candidate_id,
        "run_id": ",".join(run_ids),
        "tasks_total": len(FINAL_TASKS),
        "tasks_valid": tasks_valid,
        "tasks_won": tasks_won,
        "tasks_lost": tasks_lost,
        "min_delta": min(valid_deltas) if valid_deltas else "",
        "mean_delta": statistics.mean(valid_deltas) if valid_deltas else "",
        "pass_all_tasks": pass_all,
        "failure_reasons": "; ".join(failure_reasons),
    }
    return {"aggregate": aggregate, "details": details}


def compare(baseline_path: Path, results_path: Path, metric: str, margin: float) -> Dict[str, Any]:
    if metric != PRIMARY_METRIC:
        raise ValueError("Unsupported metric {}; expected {}".format(metric, PRIMARY_METRIC))
    baseline = load_baseline(baseline_path, metric)
    candidates = load_candidates(results_path, metric)
    aggregates: List[Dict[str, Any]] = []
    details: List[Dict[str, Any]] = []
    for candidate_id, rows in candidates.items():
        scored = evaluate_candidate(candidate_id, rows, baseline, metric, margin)
        aggregates.append(scored["aggregate"])
        details.extend(scored["details"])
    aggregates.sort(
        key=lambda row: (
            1 if row["pass_all_tasks"] else 0,
            row["min_delta"] if row["min_delta"] != "" else float("-inf"),
            row["mean_delta"] if row["mean_delta"] != "" else float("-inf"),
        ),
        reverse=True,
    )
    return {
        "created_at": now_utc(),
        "repo": repo_status(),
        "metric": metric,
        "margin": margin,
        "final_tasks": list(FINAL_TASKS),
        "baseline": str(baseline_path),
        "results": str(results_path),
        "aggregates": aggregates,
        "details": details,
    }


def update_state(state_path: Path, result: Dict[str, Any], output_json: str) -> None:
    if not state_path.exists():
        return
    state = load_json(state_path)
    state["last_scoreboard"] = output_json
    passing = [row for row in result["aggregates"] if row["pass_all_tasks"]]
    if passing:
        state["best_candidate"] = passing[0]
        state["next_required_action"] = "Final-validate the passing candidate before making any claim."
    else:
        state["next_required_action"] = "No candidate beat every final task; design the next batch or inspect failures."
    atomic_write_json(state_path, state)


def self_test() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        baseline_path = tmpdir / "baseline.csv"
        results_path = tmpdir / "results.csv"
        baseline_rows = []
        pass_rows = []
        fail_rows = []
        for idx, task in enumerate(FINAL_TASKS):
            base = 0.2 + idx * 0.01
            baseline_rows.append({"version": "standard", "task": task, PRIMARY_METRIC: base})
            pass_rows.append(
                {
                    "candidate_id": "candidate_pass__loop1",
                    "run_id": "candidate_pass",
                    "task": task,
                    "metric": PRIMARY_METRIC,
                    "value": base + 0.002,
                    "status": "completed",
                }
            )
            fail_rows.append(
                {
                    "candidate_id": "candidate_fail__loop1",
                    "run_id": "candidate_fail",
                    "task": task,
                    "metric": PRIMARY_METRIC,
                    "value": base + (0.002 if idx else 0.0001),
                    "status": "completed",
                }
            )
        write_csv_rows(baseline_path, baseline_rows, ["version", "task", PRIMARY_METRIC])
        write_csv_rows(
            results_path,
            pass_rows + fail_rows,
            ["candidate_id", "run_id", "task", "metric", "value", "status"],
        )
        result = compare(baseline_path, results_path, PRIMARY_METRIC, DEFAULT_WIN_MARGIN)
        by_id = {row["candidate_id"]: row for row in result["aggregates"]}
        assert by_id["candidate_pass__loop1"]["pass_all_tasks"] is True
        assert by_id["candidate_fail__loop1"]["pass_all_tasks"] is False
    print("goal_scoreboard self-test passed")


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    if not args.baseline or not args.results or not args.output_csv or not args.output_json:
        raise SystemExit("--baseline, --results, --output-csv, and --output-json are required unless --self-test is used")
    result = compare(Path(args.baseline), Path(args.results), args.metric, args.margin)
    write_csv_rows(Path(args.output_csv), result["aggregates"], AGG_COLUMNS)
    atomic_write_json(args.output_json, result)
    update_state(Path(args.state), result, args.output_json)
    print("Wrote scoreboard CSV: {}".format(args.output_csv))
    print("Wrote scoreboard JSON: {}".format(args.output_json))
    if result["aggregates"]:
        top = result["aggregates"][0]
        print("Top candidate: {} pass_all_tasks={}".format(top["candidate_id"], top["pass_all_tasks"]))
    else:
        print("No candidates found in results.")


if __name__ == "__main__":
    main()
