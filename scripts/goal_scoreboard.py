#!/usr/bin/env python3
"""Compare collected candidates against the frozen standard baseline."""

import argparse
import statistics
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from goal_common import (
    DEFAULT_WIN_MARGIN,
    FINAL_TASKS,
    PRIMARY_METRIC,
    atomic_write_json,
    load_json,
    load_yaml,
    metric_float,
    now_utc,
    read_csv_rows,
    repo_status,
    write_yaml,
    write_csv_rows,
)


DIAGNOSTIC_MARGIN = 0.001
MAIN_TASK_DELTA_THRESHOLD = 0.002
MAIN_MEAN_DELTA_THRESHOLD = 0.005
PUBLISHABLE_MEAN_DELTA_THRESHOLD = 0.008
VALID_CANDIDATE_TRACKS = {"standalone_main", "fusion_diagnostic", "diagnostic"}
FUSION_ROW_KEYS = {"fusion_standard_checkpoint_dir", "fusion_alpha", "fusion_scope"}
FUSION_EVAL_KEYS = {"fusion_standard_checkpoint_dir", "fusion_alpha", "fusion_scope"}
FUSION_TEXT_MARKERS = [
    "standard+loop",
    "standard + loop",
    "weighted concat",
    "weighted concatenation",
    "standard embedding",
    "standard embeddings",
    "standard score",
    "standard scores",
    "frozen standard plus",
    "ensemble with the frozen standard",
    "explicit ensemble",
    "score fusion",
    "fusion_scope",
    "fusion_alpha",
]
SIGNIFICANCE_FIELDS = {"bootstrap_p_value", "p_value", "significant", "significance", "significance_status"}


AGG_COLUMNS = [
    "candidate_id",
    "run_id",
    "purpose",
    "candidate_track",
    "tasks_total",
    "tasks_valid",
    "tasks_won",
    "tasks_lost",
    "tasks_at_main_margin",
    "min_delta",
    "mean_delta",
    "no_task_regression",
    "minimal_positive_signal",
    "research_grade_threshold_pass",
    "fusion_diagnostic_pass",
    "main_goal_success",
    "publishable_score_candidate",
    "publishable_certification",
    "pass_all_tasks",
    "main_failure_reasons",
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
    parser.add_argument("--manifest", default=None, help="Optional submitted manifest. Defaults to results directory manifest when available.")
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


def _present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, dict, set)):
        return bool(value)
    return True


def _string_values(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        values: List[str] = []
        for key, item in value.items():
            values.append(str(key))
            values.extend(_string_values(item))
        return values
    if isinstance(value, (list, tuple, set)):
        values = []
        for item in value:
            values.extend(_string_values(item))
        return values
    return []


def _experiment_fusion_evidence(experiment: Dict[str, Any]) -> List[str]:
    evidence: List[str] = []
    eval_config = experiment.get("eval") or {}
    for key in sorted(FUSION_EVAL_KEYS):
        if _present(eval_config.get(key)):
            evidence.append("eval.{}".format(key))
    text = " ".join(_string_values({key: experiment.get(key) for key in [
        "hypothesis",
        "mechanism",
        "candidate_rule",
        "expected_effect",
        "fallback",
    ]})).lower()
    for marker in FUSION_TEXT_MARKERS:
        if marker in text:
            evidence.append("text:{}".format(marker))
    return sorted(set(evidence))


def _explicit_track(experiment: Dict[str, Any]) -> Optional[str]:
    track = experiment.get("claim_track")
    if track is None:
        track = experiment.get("candidate_track")
    return track if isinstance(track, str) else None


def _infer_experiment_track(experiment: Dict[str, Any], purpose: Optional[str]) -> str:
    explicit = _explicit_track(experiment)
    if explicit in VALID_CANDIDATE_TRACKS:
        return explicit
    if _experiment_fusion_evidence(experiment):
        return "fusion_diagnostic"
    if purpose == "final":
        return "standalone_main"
    return "diagnostic"


def default_manifest_path(results_path: Path) -> Optional[Path]:
    candidate = results_path.parent / "batch_manifest.submitted.yaml"
    if candidate.exists():
        return candidate
    return None


def load_candidate_metadata(results_path: Path, manifest_path: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    metadata: Dict[str, Dict[str, Any]] = {}
    path = manifest_path or default_manifest_path(results_path)
    if path is None or not path.exists():
        return metadata
    manifest = load_yaml(path)
    purpose = manifest.get("purpose")
    for experiment in manifest.get("experiments", []):
        if not isinstance(experiment, dict):
            continue
        run_id = experiment.get("run_id")
        if not run_id:
            continue
        metadata[run_id] = {
            "purpose": purpose,
            "candidate_track": _infer_experiment_track(experiment, purpose),
            "fusion_diagnostic_evidence": _experiment_fusion_evidence(experiment),
            "manifest": str(path),
        }
    return metadata


def rows_use_fusion(rows: List[Dict[str, Any]]) -> bool:
    return any(_present(row.get(key)) for row in rows for key in FUSION_ROW_KEYS)


def rows_have_significance(rows: List[Dict[str, Any]]) -> bool:
    return any(_present(row.get(key)) for row in rows for key in SIGNIFICANCE_FIELDS)


def candidate_metadata(candidate_id: str, rows: List[Dict[str, Any]], metadata: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    run_ids = sorted(set(row.get("run_id", "") for row in rows if row.get("run_id")))
    meta: Dict[str, Any] = {}
    for run_id in run_ids:
        if run_id in metadata:
            meta = dict(metadata[run_id])
            break
    if rows_use_fusion(rows):
        meta["candidate_track"] = "fusion_diagnostic"
        meta.setdefault("fusion_diagnostic_evidence", []).append("collected_results.fusion_*")
    meta.setdefault("candidate_track", "diagnostic")
    meta.setdefault("purpose", "")
    return meta


def evaluate_candidate(
    candidate_id: str,
    rows: List[Dict[str, Any]],
    baseline: Dict[str, float],
    metric: str,
    margin: float,
    metadata: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    by_task: Dict[str, List[Dict[str, Any]]] = {task: [] for task in FINAL_TASKS}
    for row in rows:
        task = row.get("task")
        if task in by_task:
            by_task[task].append(row)

    meta = candidate_metadata(candidate_id, rows, metadata)
    candidate_track = meta.get("candidate_track", "diagnostic")
    purpose = meta.get("purpose", "")
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
        diagnostic_won_task = bool(delta is not None and delta >= DIAGNOSTIC_MARGIN and status == "completed")
        main_margin_task = bool(delta is not None and delta >= MAIN_TASK_DELTA_THRESHOLD and status == "completed")
        if delta is not None and status == "completed":
            valid_deltas.append(delta)
        if not diagnostic_won_task:
            if not reason and delta is not None:
                reason = "delta below diagnostic margin"
            failure_reasons.append("{}: {}".format(task, reason or "not won"))
        details.append(
            {
                "candidate_id": candidate_id,
                "task": task,
                "baseline_value": baseline_value,
                "candidate_value": candidate_value,
                "delta": delta,
                "won_task": diagnostic_won_task,
                "diagnostic_won_task": diagnostic_won_task,
                "main_margin_task": main_margin_task,
                "status": status,
                "reason": reason,
            }
        )

    tasks_valid = len(valid_deltas)
    tasks_won = sum(1 for item in details if item["diagnostic_won_task"])
    tasks_at_main_margin = sum(1 for item in details if item["main_margin_task"])
    tasks_lost = len(FINAL_TASKS) - tasks_won
    all_final_tasks_valid = tasks_valid == len(FINAL_TASKS)
    min_delta = min(valid_deltas) if valid_deltas else ""
    mean_delta = statistics.mean(valid_deltas) if valid_deltas else ""
    no_task_regression = bool(all_final_tasks_valid and valid_deltas and min_delta >= 0.0)
    minimal_positive_signal = bool(all_final_tasks_valid and tasks_won == len(FINAL_TASKS))
    research_grade_threshold_pass = bool(
        all_final_tasks_valid
        and tasks_at_main_margin == len(FINAL_TASKS)
        and mean_delta != ""
        and mean_delta >= MAIN_MEAN_DELTA_THRESHOLD
        and no_task_regression
    )
    standalone_final = candidate_track == "standalone_main" and purpose == "final"
    main_goal_success = bool(standalone_final and research_grade_threshold_pass)
    fusion_diagnostic_pass = bool(candidate_track == "fusion_diagnostic" and minimal_positive_signal)
    publishable_score_candidate = bool(
        standalone_final
        and all_final_tasks_valid
        and tasks_at_main_margin == len(FINAL_TASKS)
        and mean_delta != ""
        and mean_delta >= PUBLISHABLE_MEAN_DELTA_THRESHOLD
    )
    if publishable_score_candidate:
        publishable_certification = "significance evidence present" if rows_have_significance(rows) else "score-only, not statistically certified"
    else:
        publishable_certification = ""
    main_failure_reasons: List[str] = []
    if not standalone_final:
        main_failure_reasons.append("candidate_track={} purpose={} is not standalone_main final".format(candidate_track, purpose or "unknown"))
    if not all_final_tasks_valid:
        main_failure_reasons.append("not all final tasks are valid")
    if tasks_at_main_margin != len(FINAL_TASKS):
        main_failure_reasons.append("one or more final-task deltas are below +{:.3f}".format(MAIN_TASK_DELTA_THRESHOLD))
    if mean_delta == "" or mean_delta < MAIN_MEAN_DELTA_THRESHOLD:
        main_failure_reasons.append("macro mean delta is below +{:.3f}".format(MAIN_MEAN_DELTA_THRESHOLD))
    if not no_task_regression:
        main_failure_reasons.append("one or more tasks regress or are invalid")
    pass_all = minimal_positive_signal
    aggregate = {
        "candidate_id": candidate_id,
        "run_id": ",".join(run_ids),
        "purpose": purpose,
        "candidate_track": candidate_track,
        "tasks_total": len(FINAL_TASKS),
        "tasks_valid": tasks_valid,
        "tasks_won": tasks_won,
        "tasks_lost": tasks_lost,
        "tasks_at_main_margin": tasks_at_main_margin,
        "min_delta": min_delta,
        "mean_delta": mean_delta,
        "no_task_regression": no_task_regression,
        "minimal_positive_signal": minimal_positive_signal,
        "research_grade_threshold_pass": research_grade_threshold_pass,
        "fusion_diagnostic_pass": fusion_diagnostic_pass,
        "main_goal_success": main_goal_success,
        "publishable_score_candidate": publishable_score_candidate,
        "publishable_certification": publishable_certification,
        "pass_all_tasks": pass_all,
        "main_failure_reasons": "" if main_goal_success else "; ".join(main_failure_reasons),
        "failure_reasons": "; ".join(failure_reasons),
    }
    return {"aggregate": aggregate, "details": details}


def compare(baseline_path: Path, results_path: Path, metric: str, margin: float, manifest_path: Optional[Path] = None) -> Dict[str, Any]:
    if metric != PRIMARY_METRIC:
        raise ValueError("Unsupported metric {}; expected {}".format(metric, PRIMARY_METRIC))
    baseline = load_baseline(baseline_path, metric)
    candidates = load_candidates(results_path, metric)
    metadata = load_candidate_metadata(results_path, manifest_path)
    aggregates: List[Dict[str, Any]] = []
    details: List[Dict[str, Any]] = []
    for candidate_id, rows in candidates.items():
        scored = evaluate_candidate(candidate_id, rows, baseline, metric, margin, metadata)
        aggregates.append(scored["aggregate"])
        details.extend(scored["details"])
    aggregates.sort(
        key=lambda row: (
            1 if row["main_goal_success"] else 0,
            1 if row["publishable_score_candidate"] else 0,
            1 if row["research_grade_threshold_pass"] else 0,
            1 if row["fusion_diagnostic_pass"] else 0,
            1 if row["minimal_positive_signal"] else 0,
            row["min_delta"] if row["min_delta"] != "" else float("-inf"),
            row["mean_delta"] if row["mean_delta"] != "" else float("-inf"),
        ),
        reverse=True,
    )
    return {
        "created_at": now_utc(),
        "repo": repo_status(),
        "metric": metric,
        "diagnostic_margin": DIAGNOSTIC_MARGIN,
        "requested_margin": margin,
        "main_task_delta_threshold": MAIN_TASK_DELTA_THRESHOLD,
        "main_mean_delta_threshold": MAIN_MEAN_DELTA_THRESHOLD,
        "publishable_mean_delta_threshold": PUBLISHABLE_MEAN_DELTA_THRESHOLD,
        "final_tasks": list(FINAL_TASKS),
        "baseline": str(baseline_path),
        "results": str(results_path),
        "manifest": str(manifest_path or default_manifest_path(results_path) or ""),
        "aggregates": aggregates,
        "details": details,
    }


def update_state(state_path: Path, result: Dict[str, Any], output_json: str) -> None:
    if not state_path.exists():
        return
    state = load_json(state_path)
    state["last_scoreboard"] = output_json
    main_success = [row for row in result["aggregates"] if row["main_goal_success"]]
    diagnostics = [row for row in result["aggregates"] if row["minimal_positive_signal"]]
    if main_success:
        state["best_candidate"] = main_success[0]
        state["next_required_action"] = "Standalone main candidate passed stricter final thresholds; review significance evidence before any publishable claim."
    elif diagnostics:
        state["best_diagnostic_candidate"] = diagnostics[0]
        state["next_required_action"] = "Only diagnostic or sub-threshold signals passed; do not claim main goal success."
    else:
        state["next_required_action"] = "No candidate cleared the diagnostic final threshold; inspect failures before designing another batch."
    atomic_write_json(state_path, state)


def self_test() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        baseline_path = tmpdir / "baseline.csv"
        results_path = tmpdir / "results.csv"
        manifest_path = tmpdir / "manifest.yaml"
        baseline_rows = []
        standalone_main_rows = []
        standalone_weak_rows = []
        standalone_publishable_rows = []
        fusion_rows = []
        for idx, task in enumerate(FINAL_TASKS):
            base = 0.2 + idx * 0.01
            baseline_rows.append({"version": "standard", "task": task, PRIMARY_METRIC: base})
            standalone_main_rows.append(
                {
                    "candidate_id": "standalone_main__loop1",
                    "run_id": "standalone_main",
                    "task": task,
                    "metric": PRIMARY_METRIC,
                    "value": base + 0.006,
                    "status": "completed",
                }
            )
            standalone_weak_rows.append(
                {
                    "candidate_id": "standalone_weak__loop1",
                    "run_id": "standalone_weak",
                    "task": task,
                    "metric": PRIMARY_METRIC,
                    "value": base + 0.0015,
                    "status": "completed",
                }
            )
            standalone_publishable_rows.append(
                {
                    "candidate_id": "standalone_publishable__loop1",
                    "run_id": "standalone_publishable",
                    "task": task,
                    "metric": PRIMARY_METRIC,
                    "value": base + 0.009,
                    "status": "completed",
                }
            )
            fusion_rows.append(
                {
                    "candidate_id": "fusion_pass__loop3",
                    "run_id": "fusion_pass",
                    "task": task,
                    "metric": PRIMARY_METRIC,
                    "value": base + 0.010,
                    "status": "completed",
                    "fusion_standard_checkpoint_dir": "outputs/final_grid_experiment/checkpoints/standard",
                    "fusion_alpha": "0.2",
                    "fusion_scope": "query_only",
                }
            )
        write_csv_rows(baseline_path, baseline_rows, ["version", "task", PRIMARY_METRIC])
        write_csv_rows(
            results_path,
            standalone_main_rows + standalone_weak_rows + standalone_publishable_rows + fusion_rows,
            [
                "candidate_id",
                "run_id",
                "task",
                "metric",
                "value",
                "status",
                "fusion_standard_checkpoint_dir",
                "fusion_alpha",
                "fusion_scope",
            ],
        )
        manifest = {
            "purpose": "final",
            "experiments": [
                {"run_id": "standalone_main", "version": "loop_matryoshka", "eval": {"task_names": list(FINAL_TASKS)}},
                {"run_id": "standalone_weak", "version": "loop_matryoshka", "eval": {"task_names": list(FINAL_TASKS)}},
                {"run_id": "standalone_publishable", "version": "loop_matryoshka", "eval": {"task_names": list(FINAL_TASKS)}},
                {
                    "run_id": "fusion_pass",
                    "version": "loop_matryoshka",
                    "mechanism": "standard+loop weighted concat",
                    "eval": {
                        "task_names": list(FINAL_TASKS),
                        "fusion_standard_checkpoint_dir": "outputs/final_grid_experiment/checkpoints/standard",
                        "fusion_alpha": 0.2,
                        "fusion_scope": "query_only",
                    },
                },
            ],
        }
        write_yaml(manifest_path, manifest)
        result = compare(baseline_path, results_path, PRIMARY_METRIC, DEFAULT_WIN_MARGIN, manifest_path=manifest_path)
        by_id = {row["candidate_id"]: row for row in result["aggregates"]}
        assert by_id["standalone_main__loop1"]["minimal_positive_signal"] is True
        assert by_id["standalone_main__loop1"]["main_goal_success"] is True
        assert by_id["standalone_main__loop1"]["publishable_score_candidate"] is False
        assert by_id["standalone_weak__loop1"]["minimal_positive_signal"] is True
        assert by_id["standalone_weak__loop1"]["main_goal_success"] is False
        assert by_id["standalone_publishable__loop1"]["publishable_score_candidate"] is True
        assert by_id["standalone_publishable__loop1"]["publishable_certification"] == "score-only, not statistically certified"
        assert by_id["fusion_pass__loop3"]["minimal_positive_signal"] is True
        assert by_id["fusion_pass__loop3"]["fusion_diagnostic_pass"] is True
        assert by_id["fusion_pass__loop3"]["main_goal_success"] is False
    print("goal_scoreboard self-test passed")


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    if not args.baseline or not args.results or not args.output_csv or not args.output_json:
        raise SystemExit("--baseline, --results, --output-csv, and --output-json are required unless --self-test is used")
    manifest_path = Path(args.manifest) if args.manifest else None
    result = compare(Path(args.baseline), Path(args.results), args.metric, args.margin, manifest_path=manifest_path)
    write_csv_rows(Path(args.output_csv), result["aggregates"], AGG_COLUMNS)
    atomic_write_json(args.output_json, result)
    update_state(Path(args.state), result, args.output_json)
    print("Wrote scoreboard CSV: {}".format(args.output_csv))
    print("Wrote scoreboard JSON: {}".format(args.output_json))
    if result["aggregates"]:
        top = result["aggregates"][0]
        print(
            "Top candidate: {} track={} main_goal_success={} minimal_positive_signal={}".format(
                top["candidate_id"],
                top["candidate_track"],
                top["main_goal_success"],
                top["minimal_positive_signal"],
            )
        )
    else:
        print("No candidates found in results.")


if __name__ == "__main__":
    main()
