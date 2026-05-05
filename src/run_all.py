import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path
from typing import List

from .experiments import MAIN_VERSIONS, get_version_spec
from .utils import ensure_dir, load_yaml


def run(cmd: List[str]) -> None:
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def infer_paths(config_path: str) -> tuple[str, str]:
    stem = Path(config_path).stem
    if stem == "smoke":
        return "outputs/smoke", "outputs/smoke_eval"
    return "outputs/preexp", "outputs/preexp_eval"


def dedupe_summary(summary_csv: Path) -> None:
    if not summary_csv.exists():
        return
    with open(summary_csv, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        columns = reader.fieldnames or []
    deduped = {}
    for row in rows:
        key = (row.get("version"), row.get("task"), str(row.get("loop_idx")), row.get("checkpoint_dir"))
        deduped[key] = row
    with open(summary_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(deduped.values())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run train/eval/plot stages for loop Matryoshka retrieval.")
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument("--stage", choices=["train", "eval", "plot", "all"], default="all")
    parser.add_argument("--output_base", default=None)
    parser.add_argument("--eval_output_dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    task_name = config.get("task_name", "SciFact")
    output_base, eval_output_dir = infer_paths(args.config)
    output_base = args.output_base or output_base
    eval_output_dir = args.eval_output_dir or eval_output_dir
    ensure_dir(output_base)
    ensure_dir(eval_output_dir)
    python_bin = os.environ.get("PYTHON_BIN", sys.executable)

    if args.stage in {"train", "all"}:
        for version in MAIN_VERSIONS:
            run(
                [
                    python_bin,
                    "-m",
                    "src.train",
                    "--config",
                    args.config,
                    "--version",
                    version,
                    "--output_dir",
                    f"{output_base}/{version}",
                ]
            )

    if args.stage in {"eval", "all"}:
        for version in MAIN_VERSIONS:
            run(
                [
                    python_bin,
                    "-m",
                    "src.eval_mteb",
                    "--checkpoint_dir",
                    f"{output_base}/{version}/final",
                    "--version",
                    version,
                    "--task_name",
                    task_name,
                    "--eval_all_loops",
                    str(get_version_spec(version).eval_all_loops).lower(),
                    "--output_dir",
                    eval_output_dir,
                ]
            )
        dedupe_summary(Path(eval_output_dir) / "results_summary.csv")

    if args.stage in {"plot", "all"}:
        run(
            [
                python_bin,
                "-m",
                "src.plot_results",
                "--summary_csv",
                f"{eval_output_dir}/results_summary.csv",
                "--output_dir",
                f"{eval_output_dir}/plots",
            ]
        )


if __name__ == "__main__":
    main()
