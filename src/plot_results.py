import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt

from .utils import ensure_dir


PLOT_METRICS = [
    ("ndcg_at_10", "nDCG@10", "loop_depth_vs_ndcg10.png"),
    ("recall_at_10", "Recall@10", "loop_depth_vs_recall10.png"),
    ("recall_at_100", "Recall@100", "loop_depth_vs_recall100.png"),
    ("mrr_at_10", "MRR@10", "loop_depth_vs_mrr10.png"),
]
METRIC_SUFFIXES = {
    "ndcg_at_10": "ndcg10",
    "recall_at_10": "recall10",
    "recall_at_100": "recall100",
    "mrr_at_10": "mrr10",
}

MECHANISM_METRIC = ("ndcg_at_10", "nDCG@10")
METADATA_COLUMNS = [
    "series_name",
    "series_label",
    "run_name",
    "loss_group",
    "memory_mode",
    "query_mode",
]
SUMMARY_COLUMNS = [
    "version",
    "task",
    "loop_idx",
    "ndcg_at_10",
    "recall_at_10",
    "recall_at_100",
    "mrr_at_10",
    "map_at_10",
    "checkpoint_dir",
    "raw_result_path",
]
LOSS_LABELS = {
    "standard": "Standard",
    "final": "Final-only supervision",
    "matryoshka": "Loopwise supervision",
}
SHORT_LOSS_LABELS = {
    "standard": "standard",
    "final": "final-only",
    "matryoshka": "loopwise-supervised",
}
MEMORY_LABELS = {
    "standard": "standard",
    "first_token": "first token",
    "mean_pool": "mean pool",
    "token_concat": "token concat",
    "none": "no memory",
    "unknown": "unknown memory",
}
QUERY_LABELS = {
    "standard": "standard",
    "original": "original query",
    "recurrent_hidden": "recurrent hidden",
    "unknown": "unknown query",
}
MEMORY_COLORS = {
    "standard": "black",
    "first_token": "#4e79a7",
    "mean_pool": "#59a14f",
    "token_concat": "#f28e2b",
    "none": "#af7aa1",
    "unknown": "#7f7f7f",
}
RECURRENT_MEAN_POOL_COLOR = "#e15759"
LOSS_LINESTYLES = {
    "standard": "--",
    "final": "--",
    "matryoshka": "-",
}
LOSS_MARKERS = {
    "standard": None,
    "final": "o",
    "matryoshka": "s",
}
LOSS_ORDER = {"standard": 0, "final": 1, "matryoshka": 2, "unknown": 99}
QUERY_ORDER = {"standard": 0, "original": 1, "recurrent_hidden": 2, "unknown": 99}
MEMORY_ORDER = {
    "standard": 0,
    "first_token": 1,
    "mean_pool": 2,
    "token_concat": 3,
    "none": 4,
    "unknown": 99,
}


def numeric(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def infer_run_name(row: Dict[str, Any], source_path: Path) -> str:
    for key in ("series_name", "run_name"):
        value = row.get(key)
        if value:
            return str(value)

    checkpoint_dir = str(row.get("checkpoint_dir") or "")
    if checkpoint_dir:
        parts = Path(checkpoint_dir).parts
        for marker in ("preexp", "outputs"):
            if marker in parts:
                idx = parts.index(marker)
                if marker == "preexp" and idx + 1 < len(parts):
                    return parts[idx + 1]
                if marker == "outputs" and idx + 2 < len(parts) and parts[idx + 1] == "preexp":
                    return parts[idx + 2]

    if source_path.name == "results_summary.csv" and source_path.parent.name:
        return source_path.parent.name
    return str(row.get("version") or "unknown")


def parse_run_metadata(run_name: str, version: str) -> Dict[str, str]:
    if run_name == "standard" or version == "standard":
        return {
            "loss_group": "standard",
            "memory_mode": "standard",
            "query_mode": "standard",
        }

    if run_name.startswith("loop_final"):
        loss_group = "final"
        suffix = run_name[len("loop_final") :].strip("_")
    elif run_name.startswith("loop_matryoshka"):
        loss_group = "matryoshka"
        suffix = run_name[len("loop_matryoshka") :].strip("_")
    elif version.startswith("loop_final"):
        loss_group = "final"
        suffix = run_name
    elif version.startswith("loop_matryoshka"):
        loss_group = "matryoshka"
        suffix = run_name
    else:
        loss_group = "unknown"
        suffix = run_name

    if suffix.startswith("recurrent_no_memory"):
        memory_mode = "none"
        query_mode = "recurrent_hidden"
    elif suffix.startswith("recurrent_mean_pool"):
        memory_mode = "mean_pool"
        query_mode = "recurrent_hidden"
    elif suffix in {"first_token", "mean_pool", "token_concat"}:
        memory_mode = suffix
        query_mode = "original"
    elif suffix == "":
        memory_mode = "unknown"
        query_mode = "unknown"
    else:
        memory_mode = suffix
        query_mode = "unknown"

    return {
        "loss_group": loss_group,
        "memory_mode": memory_mode,
        "query_mode": query_mode,
    }


def series_label(metadata: Dict[str, str]) -> str:
    loss = SHORT_LOSS_LABELS.get(metadata["loss_group"], metadata["loss_group"])
    memory = MEMORY_LABELS.get(metadata["memory_mode"], metadata["memory_mode"])
    query = QUERY_LABELS.get(metadata["query_mode"], metadata["query_mode"])
    if metadata["loss_group"] == "standard":
        return "standard"
    if metadata["query_mode"] == "original":
        return f"{loss} / {memory}"
    return f"{loss} / {memory} / {query}"


def annotate_row(row: Dict[str, Any], source_path: Path) -> Dict[str, Any]:
    run_name = infer_run_name(row, source_path)
    metadata = parse_run_metadata(run_name, str(row.get("version") or ""))
    row["run_name"] = run_name
    row["series_name"] = run_name
    row["loss_group"] = metadata["loss_group"]
    row["memory_mode"] = metadata["memory_mode"]
    row["query_mode"] = metadata["query_mode"]
    row["series_label"] = series_label(metadata)
    return row


def read_summary_file(summary_csv: Path) -> Tuple[List[Dict[str, Any]], List[str]]:
    with open(summary_csv, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    for row in rows:
        annotate_row(row, summary_csv)
        if row.get("loop_idx") not in (None, ""):
            row["loop_idx"] = int(row["loop_idx"])
        for metric, _, _ in PLOT_METRICS:
            row[metric] = numeric(row.get(metric))
    return rows, fieldnames


def merge_fieldnames(fieldnames_groups: Iterable[Sequence[str]]) -> List[str]:
    merged: List[str] = []
    for name in METADATA_COLUMNS + SUMMARY_COLUMNS:
        if name not in merged:
            merged.append(name)
    for group in fieldnames_groups:
        for name in group:
            if name not in merged:
                merged.append(name)
    return merged


def summary_paths_from_dir(summary_dir: Path) -> List[Path]:
    paths = sorted(summary_dir.glob("*/results_summary.csv"))
    if not paths:
        raise FileNotFoundError(f"No per-method results_summary.csv files found under {summary_dir}")
    return paths


def read_summaries(paths: Sequence[Path]) -> Tuple[List[Dict[str, Any]], List[str]]:
    rows: List[Dict[str, Any]] = []
    fieldname_groups: List[Sequence[str]] = []
    for path in paths:
        file_rows, file_fieldnames = read_summary_file(path)
        rows.extend(file_rows)
        fieldname_groups.append(file_fieldnames)
    return rows, merge_fieldnames(fieldname_groups)


def latest_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped: Dict[tuple, Dict[str, Any]] = {}
    for row in rows:
        key = (
            row.get("series_name"),
            row.get("task"),
            str(row.get("loop_idx")),
            row.get("checkpoint_dir"),
        )
        deduped[key] = row
    return list(deduped.values())


def write_summary(rows: List[Dict[str, Any]], fieldnames: List[str], output_path: Path) -> None:
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{key: row.get(key) for key in fieldnames} for row in rows])


def series_sort_key(series_name: str, rows: List[Dict[str, Any]]) -> Tuple[int, int, int, str]:
    first = next(row for row in rows if row.get("series_name") == series_name)
    return (
        LOSS_ORDER.get(str(first.get("loss_group")), 99),
        QUERY_ORDER.get(str(first.get("query_mode")), 99),
        MEMORY_ORDER.get(str(first.get("memory_mode")), 99),
        series_name,
    )


def series_color(row: Dict[str, Any]) -> str:
    if row.get("query_mode") == "recurrent_hidden" and row.get("memory_mode") == "mean_pool":
        return RECURRENT_MEAN_POOL_COLOR
    return MEMORY_COLORS.get(str(row.get("memory_mode")), MEMORY_COLORS["unknown"])


def plot_series(
    ax: Any,
    rows: List[Dict[str, Any]],
    metric: str,
    label: str,
    color: str,
    linestyle: str = "-",
    marker: Optional[str] = "o",
    linewidth: float = 2.0,
) -> None:
    sub = [row for row in rows if row.get(metric) is not None]
    if not sub:
        return
    sub = sorted(sub, key=lambda row: int(row["loop_idx"]))
    ax.plot(
        [int(row["loop_idx"]) for row in sub],
        [float(row[metric]) for row in sub],
        marker=marker,
        linewidth=linewidth,
        linestyle=linestyle,
        label=label,
        color=color,
    )


def baseline_value(rows: List[Dict[str, Any]], metric: str) -> Optional[float]:
    baseline_rows = [
        row
        for row in rows
        if row.get("series_name") == "standard" and row.get(metric) is not None
    ]
    if not baseline_rows:
        return None
    return float(baseline_rows[-1][metric])


def add_standard_baseline(ax: Any, rows: List[Dict[str, Any]], metric: str) -> None:
    baseline = baseline_value(rows, metric)
    if baseline is not None:
        ax.axhline(
            baseline,
            color=MEMORY_COLORS["standard"],
            linestyle=":",
            linewidth=1.4,
            label="standard",
        )


def setup_loop_axis(ax: Any, ylabel: Optional[str] = None) -> None:
    ax.set_xlabel("loop depth t")
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.set_xticks(range(1, 11))
    ax.grid(True, alpha=0.25)


def plot_metric(rows: List[Dict[str, Any]], metric: str, ylabel: str, output_path: Path) -> None:
    _, ax = plt.subplots(figsize=(9.2, 5.4))
    add_standard_baseline(ax, rows, metric)

    series_names = sorted(
        {str(row.get("series_name")) for row in rows if row.get("loss_group") != "standard"},
        key=lambda name: series_sort_key(name, rows),
    )
    for name in series_names:
        sub = [row for row in rows if row.get("series_name") == name and row.get(metric) is not None]
        if not sub:
            continue
        first = sub[0]
        plot_series(
            ax,
            sub,
            metric,
            label=str(first.get("series_label")),
            color=series_color(first),
            linestyle=LOSS_LINESTYLES.get(str(first.get("loss_group")), "-"),
            marker=LOSS_MARKERS.get(str(first.get("loss_group")), "o"),
        )

    setup_loop_axis(ax, ylabel)
    ax.set_title(f"All controlled runs: {ylabel}")
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_grouped_runs(
    rows: List[Dict[str, Any]],
    metric: str,
    ylabel: str,
    output_path: Path,
    title: str,
    recurrent: bool,
) -> None:
    _, ax = plt.subplots(figsize=(8.2, 5.0))
    add_standard_baseline(ax, rows, metric)
    series_names = sorted(
        {
            str(row.get("series_name"))
            for row in rows
            if row.get("loss_group") != "standard"
            and (row.get("query_mode") == "recurrent_hidden") == recurrent
        },
        key=lambda name: series_sort_key(name, rows),
    )
    for name in series_names:
        sub = [row for row in rows if row.get("series_name") == name and row.get(metric) is not None]
        if not sub:
            continue
        first = sub[0]
        plot_series(
            ax,
            sub,
            metric,
            label=str(first.get("series_label")),
            color=series_color(first),
            linestyle=LOSS_LINESTYLES.get(str(first.get("loss_group")), "-"),
            marker=LOSS_MARKERS.get(str(first.get("loss_group")), "o"),
        )

    setup_loop_axis(ax, ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def select_series(
    rows: List[Dict[str, Any]],
    loss_group: str,
    memory_mode: str,
    query_mode: str,
) -> List[Dict[str, Any]]:
    return [
        row
        for row in rows
        if row.get("loss_group") == loss_group
        and row.get("memory_mode") == memory_mode
        and row.get("query_mode") == query_mode
    ]


def plot_memory_construction(rows: List[Dict[str, Any]], metric: str, ylabel: str, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), sharey=True)
    for ax, loss_group in zip(axes, ("final", "matryoshka")):
        add_standard_baseline(ax, rows, metric)
        for memory_mode in ("first_token", "mean_pool", "token_concat"):
            sub = select_series(rows, loss_group, memory_mode, "original")
            if not sub:
                continue
            plot_series(
                ax,
                sub,
                metric,
                label=MEMORY_LABELS[memory_mode],
                color=MEMORY_COLORS[memory_mode],
                linestyle="-",
                marker=LOSS_MARKERS[loss_group],
            )
        ax.set_title(LOSS_LABELS[loss_group])
        setup_loop_axis(ax, ylabel if ax is axes[0] else None)
        ax.legend(fontsize=8)
    fig.suptitle("Memory construction ablation, original-query loop")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_loss_supervision(rows: List[Dict[str, Any]], metric: str, ylabel: str, output_path: Path) -> None:
    conditions = [
        ("first_token", "original", "first-token memory"),
        ("mean_pool", "original", "mean-pool memory"),
        ("token_concat", "original", "token-concat memory"),
        ("mean_pool", "recurrent_hidden", "recurrent query + mean pool"),
        ("none", "recurrent_hidden", "recurrent query + no memory"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13.0, 7.0), sharex=True, sharey=True)
    flat_axes = list(axes.flatten())
    for ax, (memory_mode, query_mode, title) in zip(flat_axes, conditions):
        add_standard_baseline(ax, rows, metric)
        for loss_group in ("final", "matryoshka"):
            sub = select_series(rows, loss_group, memory_mode, query_mode)
            if not sub:
                continue
            color = "#4e79a7" if loss_group == "final" else "#e15759"
            plot_series(
                ax,
                sub,
                metric,
                label=LOSS_LABELS[loss_group],
                color=color,
                linestyle=LOSS_LINESTYLES[loss_group],
                marker=LOSS_MARKERS[loss_group],
            )
        ax.set_title(title)
        setup_loop_axis(ax, ylabel if ax in (flat_axes[0], flat_axes[3]) else None)
        ax.legend(fontsize=8)
    flat_axes[-1].axis("off")
    fig.suptitle("Training-supervision ablation at fixed memory/query settings")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_recurrent_query(rows: List[Dict[str, Any]], metric: str, ylabel: str, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), sharey=True)
    for ax, loss_group in zip(axes, ("final", "matryoshka")):
        add_standard_baseline(ax, rows, metric)
        for query_mode, color in (("original", MEMORY_COLORS["mean_pool"]), ("recurrent_hidden", RECURRENT_MEAN_POOL_COLOR)):
            sub = select_series(rows, loss_group, "mean_pool", query_mode)
            if not sub:
                continue
            plot_series(
                ax,
                sub,
                metric,
                label=QUERY_LABELS[query_mode],
                color=color,
                linestyle="-",
                marker=LOSS_MARKERS[loss_group],
            )
        ax.set_title(LOSS_LABELS[loss_group])
        setup_loop_axis(ax, ylabel if ax is axes[0] else None)
        ax.legend(fontsize=8)
    fig.suptitle("Query feedback ablation, mean-pool memory")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_recurrent_memory(rows: List[Dict[str, Any]], metric: str, ylabel: str, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), sharey=True)
    for ax, loss_group in zip(axes, ("final", "matryoshka")):
        add_standard_baseline(ax, rows, metric)
        for memory_mode in ("none", "mean_pool"):
            sub = select_series(rows, loss_group, memory_mode, "recurrent_hidden")
            if not sub:
                continue
            plot_series(
                ax,
                sub,
                metric,
                label=MEMORY_LABELS[memory_mode],
                color=series_color(sub[0]),
                linestyle="-",
                marker=LOSS_MARKERS[loss_group],
            )
        ax.set_title(LOSS_LABELS[loss_group])
        setup_loop_axis(ax, ylabel if ax is axes[0] else None)
        ax.legend(fontsize=8)
    fig.suptitle("Memory-token ablation under recurrent query feedback")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def best_loop_summary(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, float]]]:
    summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    for series_name in sorted({str(row.get("series_name")) for row in rows}, key=lambda name: series_sort_key(name, rows)):
        summary[series_name] = {}
        sub = [row for row in rows if row.get("series_name") == series_name]
        if not sub:
            continue
        for metric, _, _ in PLOT_METRICS:
            metric_sub = [row for row in sub if row.get(metric) is not None]
            if not metric_sub:
                continue
            best = max(metric_sub, key=lambda row: float(row[metric]))
            summary[series_name][metric] = {
                "best_loop_idx": int(best["loop_idx"]),
                "best_value": float(best[metric]),
            }
    return summary


def write_mechanism_summary(rows: List[Dict[str, Any]], output_path: Path) -> None:
    fieldnames = [
        "series_name",
        "series_label",
        "loss_group",
        "memory_mode",
        "query_mode",
        "metric",
        "loop1_value",
        "best_loop_idx",
        "best_value",
        "final_loop_idx",
        "final_value",
        "delta_best_vs_standard",
        "delta_final_vs_standard",
    ]
    summary_rows = []
    series_names = sorted({str(row.get("series_name")) for row in rows}, key=lambda name: series_sort_key(name, rows))
    for series_name in series_names:
        sub = [row for row in rows if row.get("series_name") == series_name]
        if not sub:
            continue
        sub = sorted(sub, key=lambda row: int(row["loop_idx"]))
        first = sub[0]
        for metric, _, _ in PLOT_METRICS:
            metric_sub = [row for row in sub if row.get(metric) is not None]
            if not metric_sub:
                continue
            best = max(metric_sub, key=lambda row: float(row[metric]))
            final = metric_sub[-1]
            standard = baseline_value(rows, metric)
            best_value = float(best[metric])
            final_value = float(final[metric])
            summary_rows.append(
                {
                    "series_name": series_name,
                    "series_label": first.get("series_label"),
                    "loss_group": first.get("loss_group"),
                    "memory_mode": first.get("memory_mode"),
                    "query_mode": first.get("query_mode"),
                    "metric": metric,
                    "loop1_value": float(metric_sub[0][metric]),
                    "best_loop_idx": int(best["loop_idx"]),
                    "best_value": best_value,
                    "final_loop_idx": int(final["loop_idx"]),
                    "final_value": final_value,
                    "delta_best_vs_standard": "" if standard is None else best_value - standard,
                    "delta_final_vs_standard": "" if standard is None else final_value - standard,
                }
            )

    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot loop-depth retrieval metrics.")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--summary_csv")
    input_group.add_argument("--summary_dir", help="Directory containing */results_summary.csv files.")
    parser.add_argument("--output_dir", default="outputs/plots")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    if args.summary_dir:
        paths = summary_paths_from_dir(Path(args.summary_dir))
    else:
        paths = [Path(args.summary_csv)]
    rows, fieldnames = read_summaries(paths)
    rows = latest_rows(rows)
    rows = sorted(rows, key=lambda row: series_sort_key(str(row.get("series_name")), rows) + (int(row["loop_idx"]),))
    write_summary(rows, fieldnames, output_dir / "results_summary_all.csv")
    write_summary(rows, fieldnames, output_dir / "loop_depth_metrics.csv")

    for metric, ylabel, filename in PLOT_METRICS:
        plot_metric(rows, metric, ylabel, output_dir / filename)
        suffix = METRIC_SUFFIXES[metric]
        plot_grouped_runs(
            rows,
            metric,
            ylabel,
            output_dir / f"recurrent_runs_{suffix}.png",
            f"Recurrent-hidden runs: {ylabel}",
            recurrent=True,
        )
        plot_grouped_runs(
            rows,
            metric,
            ylabel,
            output_dir / f"non_recurrent_runs_{suffix}.png",
            f"Non-recurrent runs plus standard: {ylabel}",
            recurrent=False,
        )

    mechanism_metric, mechanism_ylabel = MECHANISM_METRIC
    plot_memory_construction(
        rows,
        mechanism_metric,
        mechanism_ylabel,
        output_dir / "controlled_memory_construction_ndcg10.png",
    )
    plot_loss_supervision(
        rows,
        mechanism_metric,
        mechanism_ylabel,
        output_dir / "controlled_supervision_objective_ndcg10.png",
    )
    plot_recurrent_query(
        rows,
        mechanism_metric,
        mechanism_ylabel,
        output_dir / "controlled_recurrent_query_ndcg10.png",
    )
    plot_recurrent_memory(
        rows,
        mechanism_metric,
        mechanism_ylabel,
        output_dir / "controlled_recurrent_memory_ndcg10.png",
    )
    write_mechanism_summary(rows, output_dir / "mechanism_curve_summary.csv")

    with open(output_dir / "best_loop_summary.json", "w", encoding="utf-8") as handle:
        json.dump(best_loop_summary(rows), handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"Wrote plots and summaries to {output_dir}")


if __name__ == "__main__":
    main()
