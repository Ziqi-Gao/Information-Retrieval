#!/usr/bin/env python3
"""Collect and validate per-run evaluation summaries for a goal batch."""

import argparse
from pathlib import Path
from typing import Any, Dict, List

from goal_common import (
    PRIMARY_METRIC,
    atomic_write_json,
    load_json,
    load_yaml,
    metric_float,
    parse_task_list,
    read_csv_rows,
    write_csv_rows,
)


COLLECT_COLUMNS = [
    "batch_id",
    "run_id",
    "candidate_id",
    "version",
    "task",
    "loop_idx",
    "metric",
    "value",
    "status",
    "reason",
    "checkpoint_dir",
    "raw_result_path",
    "fusion_standard_checkpoint_dir",
    "fusion_alpha",
    "fusion_scope",
    "self_query_alpha",
    "self_query_source_loop",
    "doc_chunk_words",
    "doc_chunk_stride",
    "doc_chunk_max_chunks",
    "lexical_hash_dim",
    "lexical_weight",
    "source_summary",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect goal evaluation results without scoring them.")
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--eval-root", default="outputs/goal/eval")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load_manifest_for_batch(batch_id: str) -> Dict[str, Any]:
    path = Path("outputs/goal/runs") / batch_id / "batch_manifest.submitted.yaml"
    if not path.exists():
        raise SystemExit("Missing submitted manifest: {}".format(path))
    return load_yaml(path)


def expected_tasks(manifest: Dict[str, Any], experiment: Dict[str, Any]) -> List[str]:
    exp_tasks = parse_task_list((experiment.get("eval") or {}).get("task_names"))
    if exp_tasks:
        return exp_tasks
    tasks = manifest.get("tasks") or {}
    if manifest.get("purpose") == "final":
        return parse_task_list(tasks.get("final"))
    return parse_task_list(tasks.get("dev")) or ["SciFact"]


def row_candidate_id(run_id: str, loop_idx: Any) -> str:
    loop = str(loop_idx).strip() if loop_idx is not None else ""
    return "{}__loop{}".format(run_id, loop or "unknown")


def expected_loop_indices(manifest: Dict[str, Any], experiment: Dict[str, Any]) -> List[str]:
    eval_config = experiment.get("eval") or {}
    declared = eval_config.get("candidate_loop_indices")
    if manifest.get("purpose") == "final":
        if not isinstance(declared, list) or not declared:
            raise SystemExit("Final batch experiment {} must declare eval.candidate_loop_indices".format(experiment.get("run_id")))
        return [str(value) for value in declared]
    if isinstance(declared, list) and declared:
        return [str(value) for value in declared]
    return []


def collect_experiment(batch_id: str, eval_root: Path, manifest: Dict[str, Any], experiment: Dict[str, Any]) -> Dict[str, Any]:
    run_id = experiment["run_id"]
    version = experiment.get("version", "")
    tasks = expected_tasks(manifest, experiment)
    expected_loops = expected_loop_indices(manifest, experiment)
    summary_path = eval_root / batch_id / run_id / "results_summary.csv"
    rows: List[Dict[str, Any]] = []
    validation = {
        "run_id": run_id,
        "summary": str(summary_path),
        "expected_tasks": tasks,
        "status": "completed",
        "reasons": [],
    }

    if not summary_path.exists():
        validation["status"] = "missing_result"
        validation["reasons"].append("missing results_summary.csv")
        loop_values = expected_loops or [None]
        for task in tasks:
            for loop_idx in loop_values:
                rows.append(
                    {
                        "batch_id": batch_id,
                        "run_id": run_id,
                        "candidate_id": row_candidate_id(run_id, loop_idx),
                        "version": version,
                        "task": task,
                        "loop_idx": "" if loop_idx is None else loop_idx,
                        "metric": PRIMARY_METRIC,
                        "value": "",
                        "status": "missing_result",
                        "reason": "missing results_summary.csv",
                        "source_summary": str(summary_path),
                    }
                )
        return {"rows": rows, "validation": validation}

    raw_rows = read_csv_rows(summary_path)
    seen_keys = set()
    task_hits = {task: 0 for task in tasks}
    expected_pairs = set()
    if expected_loops:
        expected_pairs = {(task, loop_idx) for task in tasks for loop_idx in expected_loops}
    for raw in raw_rows:
        task = raw.get("task")
        if task not in task_hits:
            continue
        loop_idx = raw.get("loop_idx")
        key = (task, str(loop_idx))
        value = metric_float(raw.get(PRIMARY_METRIC))
        status = "completed"
        reason = ""
        if expected_pairs and key not in expected_pairs:
            status = "invalid_metric"
            reason = "unexpected task/loop candidate"
        elif key in seen_keys:
            status = "invalid_metric"
            reason = "duplicate task/loop row"
        elif value is None:
            status = "invalid_metric"
            reason = "invalid {}".format(PRIMARY_METRIC)
        seen_keys.add(key)
        if not expected_pairs or key in expected_pairs:
            task_hits[task] += 1
        rows.append(
            {
                "batch_id": batch_id,
                "run_id": run_id,
                "candidate_id": row_candidate_id(run_id, loop_idx),
                "version": raw.get("version") or version,
                "task": task,
                "loop_idx": loop_idx,
                "metric": PRIMARY_METRIC,
                "value": "" if value is None else value,
                "status": status,
                "reason": reason,
                "checkpoint_dir": raw.get("checkpoint_dir", ""),
                "raw_result_path": raw.get("raw_result_path", ""),
                "fusion_standard_checkpoint_dir": raw.get("fusion_standard_checkpoint_dir", ""),
                "fusion_alpha": raw.get("fusion_alpha", ""),
                "fusion_scope": raw.get("fusion_scope", ""),
                "self_query_alpha": raw.get("self_query_alpha", ""),
                "self_query_source_loop": raw.get("self_query_source_loop", ""),
                "doc_chunk_words": raw.get("doc_chunk_words", ""),
                "doc_chunk_stride": raw.get("doc_chunk_stride", ""),
                "doc_chunk_max_chunks": raw.get("doc_chunk_max_chunks", ""),
                "lexical_hash_dim": raw.get("lexical_hash_dim", ""),
                "lexical_weight": raw.get("lexical_weight", ""),
                "source_summary": str(summary_path),
            }
        )

    missing_pairs = []
    if expected_pairs:
        missing_pairs = sorted(expected_pairs - seen_keys)
    else:
        missing_pairs = [(task, None) for task, count in sorted(task_hits.items()) if count == 0]
    for task, loop_idx in missing_pairs:
        validation["status"] = "partial_tasks"
        validation["reasons"].append("missing task {} loop {}".format(task, loop_idx or "unknown"))
        rows.append(
            {
                "batch_id": batch_id,
                "run_id": run_id,
                "candidate_id": row_candidate_id(run_id, loop_idx),
                "version": version,
                "task": task,
                "loop_idx": "" if loop_idx is None else loop_idx,
                "metric": PRIMARY_METRIC,
                "value": "",
                "status": "missing_result",
                "reason": "missing task/loop result",
                "source_summary": str(summary_path),
            }
        )

    if any(row["status"] != "completed" for row in rows):
        if validation["status"] == "completed":
            validation["status"] = "invalid_metric"
        validation["reasons"].append("one or more collected rows are invalid")
    return {"rows": rows, "validation": validation}


def main() -> None:
    args = parse_args()
    batch_id = args.batch_id
    eval_root = Path(args.eval_root)
    output_csv = Path(args.output)
    output_json = output_csv.with_suffix(".json")
    per_run_json = output_csv.parent / "per_run_validation.json"
    manifest = load_manifest_for_batch(batch_id)

    all_rows: List[Dict[str, Any]] = []
    validations: List[Dict[str, Any]] = []
    for experiment in manifest.get("experiments", []):
        result = collect_experiment(batch_id, eval_root, manifest, experiment)
        all_rows.extend(result["rows"])
        validations.append(result["validation"])

    write_csv_rows(output_csv, all_rows, COLLECT_COLUMNS)
    atomic_write_json(output_json, {"batch_id": batch_id, "rows": all_rows})
    atomic_write_json(per_run_json, {"batch_id": batch_id, "runs": validations})
    print("Collected {} row(s) into {}".format(len(all_rows), output_csv))
    print("Per-run validation: {}".format(per_run_json))


if __name__ == "__main__":
    main()
