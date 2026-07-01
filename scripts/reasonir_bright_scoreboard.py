#!/usr/bin/env python3
"""Score ReasonIR-BRIGHT loop methods against the batch-local standard run."""

import argparse
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional

from goal_common import atomic_write_json, load_yaml, metric_float, now_utc, parse_task_list, read_csv_rows, repo_status, write_csv_rows


METRIC = "ndcg_at_10"
AGG_COLUMNS = [
    "candidate_id",
    "run_id",
    "version",
    "loop_idx",
    "is_predeclared_candidate_loop",
    "domains_total",
    "domains_valid",
    "domains_non_regressing",
    "domains_regressed",
    "min_delta",
    "mean_delta",
    "macro_ndcg_at_10",
    "every_domain_non_regressing",
    "helps_bright_macro",
    "failure_reasons",
]
DETAIL_COLUMNS = [
    "candidate_id",
    "domain",
    "baseline_value",
    "candidate_value",
    "delta",
    "non_regressing",
    "status",
    "reason",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare ReasonIR-BRIGHT results against the batch-local standard baseline.")
    parser.add_argument("--results", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--metric", default=METRIC)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def domains_from_manifest(manifest: Dict[str, Any]) -> List[str]:
    return parse_task_list(((manifest.get("domains") or {}).get("dev"))) or ["biology", "economics", "psychology", "stackoverflow"]


def candidate_loop_indices(experiment: Dict[str, Any]) -> List[str]:
    eval_config = experiment.get("eval") or {}
    declared = eval_config.get("candidate_loop_indices")
    if isinstance(declared, list) and declared:
        return [str(item) for item in declared]
    if eval_config.get("loop_idx") is not None:
        return [str(eval_config.get("loop_idx"))]
    return ["1"]


def metadata_by_run(manifest: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    metadata = {}
    for experiment in manifest.get("experiments", []):
        run_id = experiment.get("run_id")
        if not run_id:
            continue
        metadata[run_id] = {
            "version": experiment.get("version", run_id),
            "candidate_loop_indices": candidate_loop_indices(experiment),
        }
    return metadata


def load_rows(results_path: Path, metric: str) -> List[Dict[str, Any]]:
    rows = []
    for row in read_csv_rows(results_path):
        if row.get("metric") and row.get("metric") != metric:
            continue
        rows.append(row)
    return rows


def standard_baseline(rows: List[Dict[str, Any]], domains: List[str], metric: str) -> Dict[str, float]:
    del metric
    values: Dict[str, List[float]] = {domain: [] for domain in domains}
    for row in rows:
        if row.get("run_id") != "standard":
            continue
        if str(row.get("loop_idx")) != "1":
            continue
        domain = row.get("domain") or row.get("task")
        if domain not in values:
            continue
        value = metric_float(row.get("value"))
        if row.get("status") == "completed" and value is not None:
            values[domain].append(value)
    problems = []
    baseline = {}
    for domain in domains:
        if len(values[domain]) != 1:
            problems.append(f"{domain} has {len(values[domain])} valid standard baseline rows; expected exactly 1")
        else:
            baseline[domain] = values[domain][0]
    if problems:
        raise SystemExit("; ".join(problems))
    return baseline


def evaluate_candidate(
    candidate_id: str,
    candidate_rows: List[Dict[str, Any]],
    baseline: Dict[str, float],
    domains: List[str],
    metadata: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    run_ids = sorted({row.get("run_id", "") for row in candidate_rows if row.get("run_id")})
    run_id = run_ids[0] if run_ids else ""
    version = metadata.get(run_id, {}).get("version", candidate_rows[0].get("version", ""))
    loop_idx = str(candidate_rows[0].get("loop_idx", ""))
    predeclared = loop_idx in metadata.get(run_id, {}).get("candidate_loop_indices", [])
    by_domain = {domain: [] for domain in domains}
    for row in candidate_rows:
        domain = row.get("domain") or row.get("task")
        if domain in by_domain:
            by_domain[domain].append(row)

    details = []
    deltas = []
    candidate_values = []
    failure_reasons = []
    for domain in domains:
        rows = by_domain[domain]
        status = "completed"
        reason = ""
        candidate_value: Optional[float] = None
        if not rows:
            status = "missing_result"
            reason = "missing domain result"
        elif len(rows) > 1:
            status = "invalid_metric"
            reason = "multiple rows for candidate/domain"
        else:
            row = rows[0]
            status = row.get("status") or "completed"
            candidate_value = metric_float(row.get("value"))
            if status != "completed":
                reason = row.get("reason") or status
            elif candidate_value is None:
                status = "invalid_metric"
                reason = "invalid ndcg_at_10"
        baseline_value = baseline[domain]
        delta = None if candidate_value is None or status != "completed" else candidate_value - baseline_value
        non_regressing = bool(delta is not None and delta >= 0.0)
        if delta is not None:
            deltas.append(delta)
            candidate_values.append(candidate_value)
        if not non_regressing:
            failure_reasons.append(f"{domain}: {reason or 'regressed'}")
        details.append(
            {
                "candidate_id": candidate_id,
                "domain": domain,
                "baseline_value": baseline_value,
                "candidate_value": candidate_value,
                "delta": delta,
                "non_regressing": non_regressing,
                "status": status,
                "reason": reason,
            }
        )

    domains_valid = len(deltas)
    domains_non_regressing = sum(1 for item in details if item["non_regressing"])
    domains_regressed = len(domains) - domains_non_regressing
    mean_delta = statistics.mean(deltas) if deltas else ""
    min_delta = min(deltas) if deltas else ""
    macro_value = statistics.mean(candidate_values) if candidate_values else ""
    aggregate = {
        "candidate_id": candidate_id,
        "run_id": run_id,
        "version": version,
        "loop_idx": loop_idx,
        "is_predeclared_candidate_loop": predeclared,
        "domains_total": len(domains),
        "domains_valid": domains_valid,
        "domains_non_regressing": domains_non_regressing,
        "domains_regressed": domains_regressed,
        "min_delta": min_delta,
        "mean_delta": mean_delta,
        "macro_ndcg_at_10": macro_value,
        "every_domain_non_regressing": bool(domains_valid == len(domains) and domains_regressed == 0),
        "helps_bright_macro": bool(mean_delta != "" and mean_delta > 0.0),
        "failure_reasons": "; ".join(failure_reasons),
    }
    return {"aggregate": aggregate, "details": details}


def compare(results_path: Path, manifest_path: Path, metric: str) -> Dict[str, Any]:
    if metric != METRIC:
        raise ValueError(f"Unsupported metric {metric!r}; expected {METRIC}.")
    manifest = load_yaml(manifest_path)
    domains = domains_from_manifest(manifest)
    metadata = metadata_by_run(manifest)
    rows = load_rows(results_path, metric)
    baseline = standard_baseline(rows, domains, metric)
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        if row.get("run_id") == "standard":
            continue
        grouped.setdefault(row.get("candidate_id") or f"{row.get('run_id')}__loop{row.get('loop_idx')}", []).append(row)

    aggregates = []
    details = []
    for candidate_id, candidate_rows in grouped.items():
        result = evaluate_candidate(candidate_id, candidate_rows, baseline, domains, metadata)
        aggregates.append(result["aggregate"])
        details.extend(result["details"])
    aggregates.sort(
        key=lambda row: (
            1 if row["is_predeclared_candidate_loop"] else 0,
            1 if row["every_domain_non_regressing"] else 0,
            row["mean_delta"] if row["mean_delta"] != "" else float("-inf"),
            row["min_delta"] if row["min_delta"] != "" else float("-inf"),
        ),
        reverse=True,
    )
    return {
        "created_at": now_utc(),
        "repo": repo_status(),
        "metric": metric,
        "domains": domains,
        "baseline_run_id": "standard",
        "baseline_values": baseline,
        "results": str(results_path),
        "manifest": str(manifest_path),
        "aggregates": aggregates,
        "details": details,
    }


def main() -> None:
    args = parse_args()
    result = compare(Path(args.results), Path(args.manifest), args.metric)
    write_csv_rows(Path(args.output_csv), result["aggregates"], AGG_COLUMNS)
    atomic_write_json(args.output_json, result)
    detail_path = Path(args.output_csv).with_name("scoreboard_details.csv")
    write_csv_rows(detail_path, result["details"], DETAIL_COLUMNS)
    print(f"Wrote ReasonIR-BRIGHT scoreboard CSV: {args.output_csv}")
    print(f"Wrote ReasonIR-BRIGHT scoreboard JSON: {args.output_json}")
    print(f"Wrote ReasonIR-BRIGHT details CSV: {detail_path}")
    if result["aggregates"]:
        top = result["aggregates"][0]
        print(
            "Top candidate: {} loop={} mean_delta={} every_domain_non_regressing={}".format(
                top["candidate_id"],
                top["loop_idx"],
                top["mean_delta"],
                top["every_domain_non_regressing"],
            )
        )


if __name__ == "__main__":
    main()
