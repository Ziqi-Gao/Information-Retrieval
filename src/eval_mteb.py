import argparse
import csv
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch

from .experiments import get_version_spec, version_names
from .model import load_model
from .utils import (
    acquire_file_lock,
    ensure_dir,
    make_jsonable,
    release_file_lock,
    safe_task_dir_name,
    split_task_names,
    str2bool,
    write_json,
)


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


def metadata_value(task: Any, key: str) -> Any:
    metadata = getattr(task, "metadata", None)
    if isinstance(metadata, dict):
        return metadata.get(key)
    if metadata is not None and hasattr(metadata, key):
        return getattr(metadata, key)
    description = getattr(task, "description", None)
    if isinstance(description, dict):
        return description.get(key)
    return getattr(task, key, None)


def assert_retrieval_tasks(tasks: List[Any], requested_name: str) -> None:
    if not tasks:
        raise ValueError(f"MTEB returned no tasks for {requested_name!r}.")
    for task in tasks:
        task_type = metadata_value(task, "type")
        task_type_text = str(getattr(task_type, "value", task_type)).split(".")[-1].lower()
        if task_type is not None and task_type_text != "retrieval":
            task_name = metadata_value(task, "name") or requested_name
            raise ValueError(f"{task_name!r} is a {task_type!r} task, but this evaluator only supports Retrieval tasks.")


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


def weighted_concat(left: torch.Tensor, right: torch.Tensor, alpha: float) -> torch.Tensor:
    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError(f"fusion_alpha must be in [0, 1], got {alpha}.")
    left_weight = math.sqrt(1.0 - float(alpha))
    right_weight = math.sqrt(float(alpha))
    return torch.cat([left * left_weight, right * right_weight], dim=-1)


class StandardLoopFusionMTEBWrapper:
    def __init__(
        self,
        standard_model,
        loop_model,
        loop_idx: int,
        fusion_alpha: float,
        device: torch.device,
        batch_size: int,
    ) -> None:
        self.standard_model = standard_model
        self.loop_model = loop_model
        self.loop_idx = loop_idx
        self.fusion_alpha = float(fusion_alpha)
        self.device = device
        self.batch_size = batch_size

    def encode_queries(self, queries, batch_size: int = 32, **kwargs):
        del kwargs
        queries = list(queries)
        batch_size = batch_size or self.batch_size
        with torch.no_grad():
            standard_emb = self.standard_model.encode_queries(
                queries,
                batch_size=batch_size,
                loop_idx=1,
                return_all_loops=False,
                device=self.device,
            )
            loop_emb = self.loop_model.encode_queries(
                queries,
                batch_size=batch_size,
                loop_idx=self.loop_idx,
                return_all_loops=False,
                device=self.device,
            )
            return weighted_concat(standard_emb, loop_emb, self.fusion_alpha).detach().cpu().numpy()

    def encode_corpus(self, corpus, batch_size: int = 32, **kwargs):
        del kwargs
        texts = corpus_to_texts(corpus)
        batch_size = batch_size or self.batch_size
        with torch.no_grad():
            standard_emb = self.standard_model.encode_docs(texts, batch_size=batch_size, device=self.device)
            loop_emb = self.loop_model.encode_docs(texts, batch_size=batch_size, device=self.device)
            return weighted_concat(standard_emb, loop_emb, self.fusion_alpha).detach().cpu().numpy()

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
        "fusion_standard_checkpoint_dir",
        "fusion_alpha",
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
            key = (
                row.get("version"),
                row.get("task"),
                str(row.get("loop_idx")),
                row.get("checkpoint_dir"),
                row.get("fusion_standard_checkpoint_dir", ""),
                str(row.get("fusion_alpha", "")),
            )
            deduped[key] = row

        with open(summary_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows([{key: row.get(key) for key in columns} for row in deduped.values()])
    finally:
        release_file_lock(lock_path)


def evaluate_one_loop(
    args: argparse.Namespace,
    model,
    device: torch.device,
    task_name: str,
    loop_idx: int,
    standard_model=None,
) -> Dict[str, Any]:
    try:
        import mteb
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "MTEB is required for evaluation but is not installed in the active Python environment. "
            "Use a PYTHON_BIN that has mteb installed, or install only the missing evaluation dependency "
            "into the intended project-space environment."
        ) from exc

    output_dir = ensure_dir(args.output_dir)
    artifact_dir = ensure_dir(output_dir / args.version / safe_task_dir_name(task_name))
    if standard_model is not None:
        wrapper = StandardLoopFusionMTEBWrapper(
            standard_model=standard_model,
            loop_model=model,
            loop_idx=loop_idx,
            fusion_alpha=args.fusion_alpha,
            device=device,
            batch_size=args.batch_size,
        )
    else:
        wrapper = LoopRetrieverMTEBWrapper(model, loop_idx=loop_idx, device=device, batch_size=args.batch_size)
    tasks = mteb.get_tasks(tasks=[task_name])
    assert_retrieval_tasks(tasks, task_name)
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
        "task": task_name,
        "loop_idx": loop_idx,
        "ndcg_at_10": parsed.get("ndcg_at_10"),
        "recall_at_10": parsed.get("recall_at_10"),
        "recall_at_100": parsed.get("recall_at_100"),
        "mrr_at_10": parsed.get("mrr_at_10"),
        "map_at_10": parsed.get("map_at_10"),
        "checkpoint_dir": str(args.checkpoint_dir),
        "raw_result_path": str(raw_path),
        "fusion_standard_checkpoint_dir": args.fusion_standard_checkpoint_dir or "",
        "fusion_alpha": "" if args.fusion_alpha is None else args.fusion_alpha,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate loop-wise retrievers with MTEB.")
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--version", choices=version_names(), required=True)
    parser.add_argument("--task_name", default="SciFact")
    parser.add_argument(
        "--task_names",
        nargs="*",
        default=None,
        help="One or more MTEB retrieval tasks. Accepts repeated values or comma-separated lists.",
    )
    parser.add_argument("--loop_idx", type=int, default=None)
    parser.add_argument("--eval_all_loops", type=str2bool, default=False)
    parser.add_argument(
        "--fusion_standard_checkpoint_dir",
        default=None,
        help="Optional standard checkpoint for standard+loop weighted-concat retrieval-time fusion.",
    )
    parser.add_argument(
        "--fusion_alpha",
        type=float,
        default=None,
        help="Loop-side weight for standard+loop weighted-concat fusion. Requires --fusion_standard_checkpoint_dir.",
    )
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

    if bool(args.fusion_standard_checkpoint_dir) != (args.fusion_alpha is not None):
        raise ValueError("--fusion_standard_checkpoint_dir and --fusion_alpha must be provided together.")
    if args.fusion_alpha is not None and not 0.0 <= float(args.fusion_alpha) <= 1.0:
        raise ValueError("--fusion_alpha must be in [0, 1].")

    model = load_model(args.checkpoint_dir, map_location="cpu").to(requested_device)
    model.eval()
    standard_model = None
    if args.fusion_standard_checkpoint_dir:
        standard_model = load_model(args.fusion_standard_checkpoint_dir, map_location="cpu").to(requested_device)
        standard_model.eval()
    task_names = split_task_names(args.task_names, args.task_name)

    if get_version_spec(args.version).is_standard_family:
        loop_indices = [1]
    elif args.loop_idx is not None:
        loop_indices = [int(args.loop_idx)]
    elif args.eval_all_loops:
        loop_indices = list(range(1, model.tmax + 1))
    else:
        loop_indices = [model.tmax]

    rows = []
    for task_name in task_names:
        for loop_idx in loop_indices:
            print(f"Evaluating {args.version} loop {loop_idx} on {task_name}")
            rows.append(evaluate_one_loop(args, model, requested_device, task_name, loop_idx, standard_model=standard_model))

    append_summary_rows(Path(args.output_dir), rows)
    print(f"Wrote summary rows to {Path(args.output_dir) / 'results_summary.csv'}")


if __name__ == "__main__":
    main()
