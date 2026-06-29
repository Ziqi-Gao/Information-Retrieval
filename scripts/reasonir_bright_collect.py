#!/usr/bin/env python3
"""Collect BRIGHT evaluation summaries for a ReasonIR-BRIGHT batch."""

import argparse
from pathlib import Path
from typing import Any, Dict, List

from goal_common import atomic_write_json, load_yaml, metric_float, parse_task_list, read_csv_rows, write_csv_rows


METRIC = "ndcg_at_10"
COLLECT_COLUMNS = [
    "batch_id",
    "run_id",
    "candidate_id",
    "version",
    "domain",
    "task",
    "loop_idx",
    "metric",
    "value",
    "status",
    "reason",
    "checkpoint_dir",
    "raw_result_path",
    "source_summary",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect ReasonIR-BRIGHT evaluation results.")
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--run-root", default="outputs/reasonir_bright/runs")
    parser.add_argument("--eval-root", default="outputs/reasonir_bright/eval")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load_manifest(run_root: Path, batch_id: str) -> Dict[str, Any]:
    path = run_root / batch_id / "batch_manifest.submitted.yaml"
    if not path.exists():
        path = run_root / batch_id / "batch_manifest.dry_run.yaml"
    if not path.exists():
        raise SystemExit(f"Missing ReasonIR-BRIGHT batch manifest copy under {run_root / batch_id}")
    return load_yaml(path)


def domains_for_experiment(manifest: Dict[str, Any], experiment: Dict[str, Any]) -> List[str]:
    eval_config = experiment.get("eval") or {}
    domains = parse_task_list(eval_config.get("domains"))
    if domains:
        return domains
    return parse_task_list(((manifest.get("domains") or {}).get("dev"))) or ["biology"]


def expected_loop_indices(experiment: Dict[str, Any]) -> List[str]:
    eval_config = experiment.get("eval") or {}
    if eval_config.get("eval_all_loops"):
        tmax = int((experiment.get("train") or {}).get("tmax", 10))
        return [str(idx) for idx in range(1, tmax + 1)]
    if eval_config.get("loop_idx") is not None:
        return [str(eval_config.get("loop_idx"))]
    return ["1"]


def candidate_id(run_id: str, loop_idx: Any) -> str:
    return f"{run_id}__loop{loop_idx}"


def collect_experiment(batch_id: str, eval_root: Path, manifest: Dict[str, Any], experiment: Dict[str, Any]) -> Dict[str, Any]:
    run_id = experiment["run_id"]
    version = experiment.get("version", run_id)
    domains = domains_for_experiment(manifest, experiment)
    loops = expected_loop_indices(experiment)
    summary_path = eval_root / batch_id / run_id / "results_summary.csv"
    rows: List[Dict[str, Any]] = []
    validation = {
        "run_id": run_id,
        "summary": str(summary_path),
        "expected_domains": domains,
        "expected_loop_indices": loops,
        "status": "completed",
        "reasons": [],
    }

    expected_pairs = {(domain, loop_idx) for domain in domains for loop_idx in loops}
    seen = set()
    if not summary_path.exists():
        validation["status"] = "missing_result"
        validation["reasons"].append("missing results_summary.csv")
        for domain, loop_idx in sorted(expected_pairs):
            rows.append(
                {
                    "batch_id": batch_id,
                    "run_id": run_id,
                    "candidate_id": candidate_id(run_id, loop_idx),
                    "version": version,
                    "domain": domain,
                    "task": domain,
                    "loop_idx": loop_idx,
                    "metric": METRIC,
                    "value": "",
                    "status": "missing_result",
                    "reason": "missing results_summary.csv",
                    "source_summary": str(summary_path),
                }
            )
        return {"rows": rows, "validation": validation}

    for raw in read_csv_rows(summary_path):
        domain = raw.get("domain") or raw.get("task")
        loop_idx = str(raw.get("loop_idx", ""))
        if (domain, loop_idx) not in expected_pairs:
            continue
        key = (domain, loop_idx)
        value = metric_float(raw.get(METRIC))
        status = "completed"
        reason = ""
        if key in seen:
            status = "invalid_metric"
            reason = "duplicate domain/loop row"
        elif value is None:
            status = "invalid_metric"
            reason = f"invalid {METRIC}"
        seen.add(key)
        rows.append(
            {
                "batch_id": batch_id,
                "run_id": run_id,
                "candidate_id": candidate_id(run_id, loop_idx),
                "version": raw.get("version") or version,
                "domain": domain,
                "task": domain,
                "loop_idx": loop_idx,
                "metric": METRIC,
                "value": "" if value is None else value,
                "status": status,
                "reason": reason,
                "checkpoint_dir": raw.get("checkpoint_dir", ""),
                "raw_result_path": raw.get("raw_result_path", ""),
                "source_summary": str(summary_path),
            }
        )

    for domain, loop_idx in sorted(expected_pairs - seen):
        validation["status"] = "partial_domains"
        validation["reasons"].append(f"missing domain {domain} loop {loop_idx}")
        rows.append(
            {
                "batch_id": batch_id,
                "run_id": run_id,
                "candidate_id": candidate_id(run_id, loop_idx),
                "version": version,
                "domain": domain,
                "task": domain,
                "loop_idx": loop_idx,
                "metric": METRIC,
                "value": "",
                "status": "missing_result",
                "reason": "missing domain/loop result",
                "source_summary": str(summary_path),
            }
        )
    if any(row["status"] != "completed" for row in rows) and validation["status"] == "completed":
        validation["status"] = "invalid_metric"
        validation["reasons"].append("one or more collected rows are invalid")
    return {"rows": rows, "validation": validation}


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root)
    eval_root = Path(args.eval_root)
    output_csv = Path(args.output)
    manifest = load_manifest(run_root, args.batch_id)
    all_rows: List[Dict[str, Any]] = []
    validations: List[Dict[str, Any]] = []
    for experiment in manifest.get("experiments", []):
        result = collect_experiment(args.batch_id, eval_root, manifest, experiment)
        all_rows.extend(result["rows"])
        validations.append(result["validation"])
    write_csv_rows(output_csv, all_rows, COLLECT_COLUMNS)
    atomic_write_json(output_csv.with_suffix(".json"), {"batch_id": args.batch_id, "rows": all_rows})
    atomic_write_json(output_csv.parent / "per_run_validation.json", {"batch_id": args.batch_id, "runs": validations})
    print(f"Collected {len(all_rows)} row(s) into {output_csv}")


if __name__ == "__main__":
    main()
