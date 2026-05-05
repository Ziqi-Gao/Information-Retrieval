import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt

from .experiments import baseline_versions, curve_versions, get_version_spec
from .utils import ensure_dir


PLOT_METRICS = [
    ("ndcg_at_10", "nDCG@10", "loop_depth_vs_ndcg10.png"),
    ("recall_at_10", "Recall@10", "loop_depth_vs_recall10.png"),
    ("recall_at_100", "Recall@100", "loop_depth_vs_recall100.png"),
    ("mrr_at_10", "MRR@10", "loop_depth_vs_mrr10.png"),
]
def numeric(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def read_summary(summary_csv: Path) -> Tuple[List[Dict[str, Any]], List[str]]:
    with open(summary_csv, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    for row in rows:
        if row.get("loop_idx") not in (None, ""):
            row["loop_idx"] = int(row["loop_idx"])
        for metric, _, _ in PLOT_METRICS:
            row[metric] = numeric(row.get(metric))
    return rows, fieldnames


def latest_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped: Dict[tuple, Dict[str, Any]] = {}
    for row in rows:
        key = (row.get("version"), row.get("task"), str(row.get("loop_idx")), row.get("checkpoint_dir"))
        deduped[key] = row
    return list(deduped.values())


def write_summary(rows: List[Dict[str, Any]], fieldnames: List[str], output_path: Path) -> None:
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{key: row.get(key) for key in fieldnames} for row in rows])


def plot_metric(rows: List[Dict[str, Any]], metric: str, ylabel: str, output_path: Path) -> None:
    plt.figure(figsize=(7.5, 4.8))
    for baseline_version in baseline_versions():
        baseline_rows = [row for row in rows if row.get("version") == baseline_version and row.get(metric) is not None]
        if baseline_rows:
            color = get_version_spec(baseline_version).color
            baseline = float(baseline_rows[-1][metric])
            plt.axhline(baseline, color=color, linestyle="--", linewidth=1.4, label=baseline_version)

    for version in curve_versions():
        sub = [row for row in rows if row.get("version") == version and row.get(metric) is not None]
        if not sub:
            continue
        sub = sorted(sub, key=lambda row: int(row["loop_idx"]))
        plt.plot(
            [int(row["loop_idx"]) for row in sub],
            [float(row[metric]) for row in sub],
            marker="o",
            linewidth=2.0,
            label=version,
            color=get_version_spec(version).color,
        )

    plt.xlabel("loop depth t")
    plt.ylabel(ylabel)
    plt.xticks(range(1, 11))
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def best_loop_summary(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, float]]]:
    summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    for version in curve_versions():
        summary[version] = {}
        sub = [row for row in rows if row.get("version") == version]
        if not sub:
            continue
        for metric, _, _ in PLOT_METRICS:
            metric_sub = [row for row in sub if row.get(metric) is not None]
            if not metric_sub:
                continue
            best = max(metric_sub, key=lambda row: float(row[metric]))
            summary[version][metric] = {
                "best_loop_idx": int(best["loop_idx"]),
                "best_value": float(best[metric]),
            }
    for baseline_version in baseline_versions():
        baseline_rows = [row for row in rows if row.get("version") == baseline_version]
        if not baseline_rows:
            continue
        summary[baseline_version] = {}
        for metric, _, _ in PLOT_METRICS:
            metric_sub = [row for row in baseline_rows if row.get(metric) is not None]
            if metric_sub:
                summary[baseline_version][metric] = {"value": float(metric_sub[-1][metric])}
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot loop-depth retrieval metrics.")
    parser.add_argument("--summary_csv", required=True)
    parser.add_argument("--output_dir", default="outputs/plots")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    rows, fieldnames = read_summary(Path(args.summary_csv))
    rows = latest_rows(rows)
    write_summary(rows, fieldnames, output_dir / "loop_depth_metrics.csv")

    for metric, ylabel, filename in PLOT_METRICS:
        plot_metric(rows, metric, ylabel, output_dir / filename)

    with open(output_dir / "best_loop_summary.json", "w", encoding="utf-8") as handle:
        json.dump(best_loop_summary(rows), handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"Wrote plots and summaries to {output_dir}")


if __name__ == "__main__":
    main()
