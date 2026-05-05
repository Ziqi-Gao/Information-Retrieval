import csv
import shutil
from pathlib import Path
from typing import Iterable, List, Optional


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


def rewrite_summary_paths(summary_path: Path, method: str, output_root: Path = Path("outputs")) -> None:
    with summary_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames or SUMMARY_COLUMNS

    for row in rows:
        loop_idx = int(row["loop_idx"])
        row["checkpoint_dir"] = str(output_root / method / "model")
        row["raw_result_path"] = str(output_root / method / "eval" / f"raw_mteb_results_loop{loop_idx}.json")

    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def combine_summaries(summary_paths: Iterable[Path], output_path: Path) -> None:
    rows = []
    fieldnames: Optional[List[str]] = None
    for path in summary_paths:
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if fieldnames is None:
                fieldnames = reader.fieldnames or SUMMARY_COLUMNS
            rows.extend(reader)

    if fieldnames is None:
        raise FileNotFoundError("No results_summary.csv files found to combine.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def finalize_single_method_eval(run_dir: Path, method: str) -> None:
    final_dir = run_dir / "final"
    model_dir = run_dir / "model"
    if not model_dir.exists() and final_dir.exists():
        shutil.move(str(final_dir), str(model_dir))

    eval_tmp = run_dir / "eval_tmp"
    method_eval_dir = eval_tmp / method
    if not method_eval_dir.exists():
        raise FileNotFoundError(f"Missing temporary eval directory: {method_eval_dir}")

    eval_dir = run_dir / "eval"
    if eval_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing eval directory: {eval_dir}")

    shutil.move(str(method_eval_dir), str(eval_dir))
    shutil.move(str(eval_tmp / "results_summary.csv"), str(run_dir / "results_summary.csv"))
    eval_tmp.rmdir()
    rewrite_summary_paths(run_dir / "results_summary.csv", method)
