import argparse
import csv
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch

from .experiments import get_version_spec, version_names
from .model import load_model
from .utils import acquire_file_lock, ensure_dir, make_jsonable, release_file_lock, str2bool, write_json


METRIC_ALIASES = {
    "ndcg_at_10": {"ndcgat10", "ndcg10"},
    "recall_at_10": {"recallat10", "recall10"},
    "recall_at_100": {"recallat100", "recall100"},
    "mrr_at_10": {"mrrat10", "mrr10"},
    "map_at_10": {"mapat10", "map10"},
}


def normalize_metric_key(key: str) -> str:
    return "".join(ch for ch in key.lower() if ch.isalnum())


def numeric_value(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        if math.isnan(float(value)):
            return None
        return float(value)
    return None


def parse_metrics_recursive(obj: Any) -> Dict[str, Optional[float]]:
    found: Dict[str, Optional[float]] = {key: None for key in METRIC_ALIASES}

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for raw_key, raw_value in value.items():
                norm_key = normalize_metric_key(str(raw_key))
                for canonical, aliases in METRIC_ALIASES.items():
                    if found[canonical] is None and norm_key in aliases:
                        parsed = numeric_value(raw_value)
                        if parsed is not None:
                            found[canonical] = parsed
                visit(raw_value)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)

    visit(make_jsonable(obj))
    return found


def corpus_to_texts(corpus: Any) -> List[str]:
    if isinstance(corpus, dict):
        values = list(corpus.values())
    else:
        values = list(corpus)

    texts: List[str] = []
    for doc in values:
        if isinstance(doc, str):
            texts.append(doc)
        elif isinstance(doc, dict):
            title = str(doc.get("title", "") or "").strip()
            text = str(doc.get("text", doc.get("contents", doc.get("content", ""))) or "").strip()
            texts.append(f"{title}\n{text}" if title else text)
        else:
            texts.append(str(doc))
    return texts


class LoopRetrieverMTEBWrapper:
    def __init__(self, model, loop_idx: int, device: torch.device, batch_size: int) -> None:
        self.model = model
        self.loop_idx = loop_idx
        self.device = device
        self.batch_size = batch_size

    def encode_queries(self, queries, batch_size: int = 32, **kwargs):
        del kwargs
        with torch.no_grad():
            return (
                self.model.encode_queries(
                    list(queries),
                    batch_size=batch_size or self.batch_size,
                    loop_idx=self.loop_idx,
                    return_all_loops=False,
                    device=self.device,
                )
                .detach()
                .cpu()
                .numpy()
            )

    def encode_corpus(self, corpus, batch_size: int = 32, **kwargs):
        del kwargs
        texts = corpus_to_texts(corpus)
        with torch.no_grad():
            return (
                self.model.encode_docs(
                    texts,
                    batch_size=batch_size or self.batch_size,
                    device=self.device,
                )
                .detach()
                .cpu()
                .numpy()
            )

    def encode(self, sentences, batch_size: int = 32, **kwargs):
        return self.encode_queries(sentences, batch_size=batch_size, **kwargs)


def append_summary_rows(output_dir: Path, rows: List[Dict[str, Any]]) -> None:
    summary_path = output_dir / "results_summary.csv"
    lock_path = output_dir / ".results_summary.lock"
    columns = [
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
    acquire_file_lock(lock_path)
    try:
        combined_rows: List[Dict[str, Any]] = []
        if summary_path.exists():
            with open(summary_path, "r", newline="", encoding="utf-8") as handle:
                combined_rows.extend(csv.DictReader(handle))
        combined_rows.extend(rows)

        deduped: Dict[tuple, Dict[str, Any]] = {}
        for row in combined_rows:
            key = (row.get("version"), row.get("task"), str(row.get("loop_idx")), row.get("checkpoint_dir"))
            deduped[key] = row

        with open(summary_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows([{key: row.get(key) for key in columns} for row in deduped.values()])
    finally:
        release_file_lock(lock_path)


def evaluate_one_loop(args: argparse.Namespace, model, device: torch.device, loop_idx: int) -> Dict[str, Any]:
    try:
        import mteb
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "MTEB is required for evaluation but is not installed in the active Python environment. "
            "Use a PYTHON_BIN that has mteb installed, or install only the missing evaluation dependency "
            "into the intended project-space environment."
        ) from exc

    output_dir = ensure_dir(args.output_dir)
    artifact_dir = ensure_dir(output_dir / args.version)
    wrapper = LoopRetrieverMTEBWrapper(model, loop_idx=loop_idx, device=device, batch_size=args.batch_size)
    tasks = mteb.get_tasks(tasks=[args.task_name])
    evaluator = mteb.MTEB(tasks=tasks)
    mteb_output = artifact_dir / f"mteb_loop{loop_idx}"
    with torch.no_grad():
        raw_results = evaluator.run(wrapper, output_folder=str(mteb_output))

    raw_json = make_jsonable(raw_results)
    raw_path = artifact_dir / f"raw_mteb_results_loop{loop_idx}.json"
    write_json(raw_path, raw_json)

    parsed = parse_metrics_recursive(raw_json)
    parsed_path = artifact_dir / f"parsed_metrics_loop{loop_idx}.json"
    write_json(parsed_path, parsed)

    return {
        "version": args.version,
        "task": args.task_name,
        "loop_idx": loop_idx,
        "ndcg_at_10": parsed.get("ndcg_at_10"),
        "recall_at_10": parsed.get("recall_at_10"),
        "recall_at_100": parsed.get("recall_at_100"),
        "mrr_at_10": parsed.get("mrr_at_10"),
        "map_at_10": parsed.get("map_at_10"),
        "checkpoint_dir": str(args.checkpoint_dir),
        "raw_result_path": str(raw_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate loop-wise retrievers with MTEB.")
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--version", choices=version_names(), required=True)
    parser.add_argument("--task_name", default="SciFact")
    parser.add_argument("--loop_idx", type=int, default=None)
    parser.add_argument("--eval_all_loops", type=str2bool, default=False)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)
    requested_device = torch.device(args.device)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA was requested for evaluation, but it is unavailable. Falling back to CPU.")
        requested_device = torch.device("cpu")

    model = load_model(args.checkpoint_dir, map_location="cpu").to(requested_device)
    model.eval()

    if get_version_spec(args.version).is_standard_family:
        loop_indices = [1]
    elif args.loop_idx is not None:
        loop_indices = [int(args.loop_idx)]
    elif args.eval_all_loops:
        loop_indices = list(range(1, model.tmax + 1))
    else:
        loop_indices = [model.tmax]

    rows = []
    for loop_idx in loop_indices:
        print(f"Evaluating {args.version} loop {loop_idx} on {args.task_name}")
        rows.append(evaluate_one_loop(args, model, requested_device, loop_idx))

    append_summary_rows(Path(args.output_dir), rows)
    print(f"Wrote summary rows to {Path(args.output_dir) / 'results_summary.csv'}")


if __name__ == "__main__":
    main()
