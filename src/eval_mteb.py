import argparse
import csv
import hashlib
import math
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

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


def fusion_artifact_dir_name(scope: str, alpha: float) -> str:
    alpha_text = str(float(alpha)).replace(".", "p")
    return "fusion_{}_a{}".format(scope, alpha_text)


TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def eval_alpha_text(alpha: float) -> str:
    return str(float(alpha)).replace(".", "p")


def doc_chunk_artifact_dir_name(chunk_words: int, chunk_stride: int, max_chunks: int) -> str:
    return "doc_chunks_w{}_s{}_m{}".format(int(chunk_words), int(chunk_stride), int(max_chunks))


def lexical_artifact_dir_name(hash_dim: int, weight: float) -> str:
    return "lexhash_d{}_a{}".format(int(hash_dim), eval_alpha_text(float(weight)))


def split_word_chunks(text: str, chunk_words: int, chunk_stride: int, max_chunks: int) -> List[str]:
    words = str(text or "").split()
    if not words:
        return [""]
    if len(words) <= int(chunk_words):
        return [" ".join(words)]

    chunks: List[str] = []
    start = 0
    while start < len(words):
        chunk = " ".join(words[start : start + int(chunk_words)]).strip()
        if chunk:
            chunks.append(chunk)
        if start + int(chunk_words) >= len(words):
            break
        if int(max_chunks) > 0 and len(chunks) >= int(max_chunks):
            break
        start += int(chunk_stride)
    return chunks or [" ".join(words[: int(chunk_words)])]


def lexical_hash_features(texts: List[str], hash_dim: int) -> torch.Tensor:
    features = np.zeros((len(texts), int(hash_dim)), dtype=np.float32)
    for row_idx, text in enumerate(texts):
        for token in TOKEN_RE.findall(str(text or "").lower()):
            if len(token) < 2:
                continue
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, byteorder="little", signed=False)
            col = value % int(hash_dim)
            sign = -1.0 if (value >> 63) else 1.0
            features[row_idx, col] += sign
        norm = float(np.linalg.norm(features[row_idx]))
        if norm > 0.0:
            features[row_idx] /= norm
    return torch.from_numpy(features)


class LoopRetrieverMTEBWrapper:
    def __init__(
        self,
        model,
        loop_idx: int,
        device: torch.device,
        batch_size: int,
        loop_docs: bool = False,
        doc_loop_idx: Optional[int] = None,
        self_query_alpha: Optional[float] = None,
        self_query_source_loop: int = 1,
        doc_chunk_words: int = 0,
        doc_chunk_stride: int = 0,
        doc_chunk_max_chunks: int = 0,
        lexical_hash_dim: int = 0,
        lexical_weight: float = 0.0,
    ) -> None:
        self.model = model
        self.loop_idx = loop_idx
        self.device = device
        self.batch_size = batch_size
        self.loop_docs = bool(loop_docs)
        self.doc_loop_idx = int(doc_loop_idx) if doc_loop_idx is not None else int(loop_idx)
        self.self_query_alpha = None if self_query_alpha is None else float(self_query_alpha)
        self.self_query_source_loop = int(self_query_source_loop)
        self.doc_chunk_words = int(doc_chunk_words or 0)
        self.doc_chunk_stride = int(doc_chunk_stride or self.doc_chunk_words or 0)
        self.doc_chunk_max_chunks = int(doc_chunk_max_chunks or 0)
        self.lexical_hash_dim = int(lexical_hash_dim or 0)
        self.lexical_weight = float(lexical_weight or 0.0)

    @property
    def lexical_enabled(self) -> bool:
        return self.lexical_hash_dim > 0 and self.lexical_weight > 0.0

    @property
    def doc_chunking_enabled(self) -> bool:
        return self.doc_chunk_words > 0

    def _combine_lexical(self, dense_emb: torch.Tensor, texts: List[str]) -> torch.Tensor:
        if not self.lexical_enabled:
            return dense_emb
        lexical = lexical_hash_features(texts, self.lexical_hash_dim).to(dense_emb.device)
        dense_weight = math.sqrt(max(0.0, 1.0 - self.lexical_weight))
        lexical_weight = math.sqrt(self.lexical_weight)
        combined = torch.cat([dense_emb * dense_weight, lexical * lexical_weight], dim=-1)
        return F.normalize(combined, p=2, dim=-1)

    def _encode_chunked_docs(self, texts: List[str], batch_size: int) -> torch.Tensor:
        chunk_texts: List[str] = []
        slices: List[tuple[int, int]] = []
        for text in texts:
            start = len(chunk_texts)
            chunks = split_word_chunks(
                text,
                chunk_words=self.doc_chunk_words,
                chunk_stride=self.doc_chunk_stride,
                max_chunks=self.doc_chunk_max_chunks,
            )
            chunk_texts.extend(chunks)
            slices.append((start, len(chunk_texts)))

        chunk_emb = self.model.encode_docs(
            chunk_texts,
            batch_size=batch_size or self.batch_size,
            device=self.device,
        )
        doc_embeddings = []
        for start, end in slices:
            pooled = chunk_emb[start:end].mean(dim=0)
            doc_embeddings.append(F.normalize(pooled, p=2, dim=-1))
        return torch.stack(doc_embeddings, dim=0)

    def encode_queries(self, queries, batch_size: int = 32, **kwargs):
        del kwargs
        queries = list(queries)
        with torch.no_grad():
            if self.self_query_alpha is None:
                query_emb = self.model.encode_queries(
                    queries,
                    batch_size=batch_size or self.batch_size,
                    loop_idx=self.loop_idx,
                    return_all_loops=False,
                    device=self.device,
                )
            else:
                source_emb = self.model.encode_queries(
                    queries,
                    batch_size=batch_size or self.batch_size,
                    loop_idx=self.self_query_source_loop,
                    return_all_loops=False,
                    device=self.device,
                )
                target_emb = self.model.encode_queries(
                    queries,
                    batch_size=batch_size or self.batch_size,
                    loop_idx=self.loop_idx,
                    return_all_loops=False,
                    device=self.device,
                )
                query_emb = F.normalize(
                    source_emb * (1.0 - self.self_query_alpha) + target_emb * self.self_query_alpha,
                    p=2,
                    dim=-1,
                )
            query_emb = self._combine_lexical(query_emb, queries)
            return query_emb.detach().cpu().numpy()

    def encode_corpus(self, corpus, batch_size: int = 32, **kwargs):
        del kwargs
        texts = corpus_to_texts(corpus)
        with torch.no_grad():
            if self.doc_chunking_enabled:
                doc_emb = self._encode_chunked_docs(texts, batch_size=batch_size or self.batch_size)
            elif self.loop_docs:
                doc_emb = self.model.encode_docs_looped(
                    texts,
                    batch_size=batch_size or self.batch_size,
                    loop_idx=self.doc_loop_idx,
                    device=self.device,
                )
            else:
                doc_emb = self.model.encode_docs(
                    texts,
                    batch_size=batch_size or self.batch_size,
                    device=self.device,
                )
            doc_emb = self._combine_lexical(doc_emb, texts)
            return doc_emb.detach().cpu().numpy()

    def encode(self, sentences, batch_size: int = 32, **kwargs):
        return self.encode_queries(sentences, batch_size=batch_size, **kwargs)


def weighted_concat(left: torch.Tensor, right: torch.Tensor, alpha: float) -> torch.Tensor:
    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError(f"fusion_alpha must be in [0, 1], got {alpha}.")
    left_weight = math.sqrt(1.0 - float(alpha))
    right_weight = math.sqrt(float(alpha))
    return torch.cat([left * left_weight, right * right_weight], dim=-1)


def self_query_artifact_dir_name(source_loop: int, alpha: float) -> str:
    alpha_text = str(float(alpha)).replace(".", "p")
    return "self_query_s{}_a{}".format(int(source_loop), alpha_text)


class StandardLoopFusionMTEBWrapper:
    def __init__(
        self,
        standard_model,
        loop_model,
        loop_idx: int,
        fusion_alpha: float,
        fusion_scope: str,
        device: torch.device,
        batch_size: int,
    ) -> None:
        if fusion_scope not in {"both", "query_only", "doc_only"}:
            raise ValueError(f"Unsupported fusion_scope: {fusion_scope}")
        self.standard_model = standard_model
        self.loop_model = loop_model
        self.loop_idx = loop_idx
        self.fusion_alpha = float(fusion_alpha)
        self.fusion_scope = fusion_scope
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
            if self.fusion_scope == "doc_only":
                loop_emb = standard_emb
            else:
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
            if self.fusion_scope == "query_only":
                loop_emb = standard_emb
            else:
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
        "fusion_scope",
        "self_query_alpha",
        "self_query_source_loop",
        "doc_chunk_words",
        "doc_chunk_stride",
        "doc_chunk_max_chunks",
        "lexical_hash_dim",
        "lexical_weight",
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
                row.get("fusion_scope", ""),
                str(row.get("self_query_alpha", "")),
                str(row.get("self_query_source_loop", "")),
                str(row.get("doc_chunk_words", "")),
                str(row.get("doc_chunk_stride", "")),
                str(row.get("doc_chunk_max_chunks", "")),
                str(row.get("lexical_hash_dim", "")),
                str(row.get("lexical_weight", "")),
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
    if args.loop_docs:
        artifact_dir = ensure_dir(artifact_dir / f"doc_loop{args.doc_loop_idx or loop_idx}")
    if args.self_query_alpha is not None:
        artifact_dir = ensure_dir(artifact_dir / self_query_artifact_dir_name(args.self_query_source_loop, args.self_query_alpha))
    if args.doc_chunk_words:
        artifact_dir = ensure_dir(
            artifact_dir
            / doc_chunk_artifact_dir_name(args.doc_chunk_words, args.doc_chunk_stride, args.doc_chunk_max_chunks)
        )
    if args.lexical_weight:
        artifact_dir = ensure_dir(artifact_dir / lexical_artifact_dir_name(args.lexical_hash_dim, args.lexical_weight))
    if standard_model is not None:
        artifact_dir = ensure_dir(artifact_dir / fusion_artifact_dir_name(args.fusion_scope, args.fusion_alpha))
    if standard_model is not None:
        wrapper = StandardLoopFusionMTEBWrapper(
            standard_model=standard_model,
            loop_model=model,
            loop_idx=loop_idx,
            fusion_alpha=args.fusion_alpha,
            fusion_scope=args.fusion_scope,
            device=device,
            batch_size=args.batch_size,
        )
    else:
        wrapper = LoopRetrieverMTEBWrapper(
            model,
            loop_idx=loop_idx,
            device=device,
            batch_size=args.batch_size,
            loop_docs=args.loop_docs,
            doc_loop_idx=args.doc_loop_idx,
            self_query_alpha=args.self_query_alpha,
            self_query_source_loop=args.self_query_source_loop,
            doc_chunk_words=args.doc_chunk_words,
            doc_chunk_stride=args.doc_chunk_stride,
            doc_chunk_max_chunks=args.doc_chunk_max_chunks,
            lexical_hash_dim=args.lexical_hash_dim,
            lexical_weight=args.lexical_weight,
        )
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
        "fusion_scope": args.fusion_scope if args.fusion_standard_checkpoint_dir else "",
        "self_query_alpha": "" if args.self_query_alpha is None else args.self_query_alpha,
        "self_query_source_loop": "" if args.self_query_alpha is None else args.self_query_source_loop,
        "doc_chunk_words": "" if not args.doc_chunk_words else args.doc_chunk_words,
        "doc_chunk_stride": "" if not args.doc_chunk_words else args.doc_chunk_stride,
        "doc_chunk_max_chunks": "" if not args.doc_chunk_words else args.doc_chunk_max_chunks,
        "lexical_hash_dim": "" if not args.lexical_weight else args.lexical_hash_dim,
        "lexical_weight": "" if not args.lexical_weight else args.lexical_weight,
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
    parser.add_argument("--loop_docs", type=str2bool, default=False)
    parser.add_argument("--doc_loop_idx", type=int, default=None)
    parser.add_argument("--self_query_alpha", type=float, default=None)
    parser.add_argument("--self_query_source_loop", type=int, default=1)
    parser.add_argument("--doc_chunk_words", type=int, default=0)
    parser.add_argument("--doc_chunk_stride", type=int, default=0)
    parser.add_argument("--doc_chunk_max_chunks", type=int, default=8)
    parser.add_argument("--lexical_hash_dim", type=int, default=0)
    parser.add_argument("--lexical_weight", type=float, default=0.0)
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
    parser.add_argument(
        "--fusion_scope",
        choices=["both", "query_only", "doc_only"],
        default="both",
        help="Which side uses loop embeddings during fusion. Default preserves standard+loop fusion on both sides.",
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
    if args.fusion_standard_checkpoint_dir and args.self_query_alpha is not None:
        raise ValueError("--self_query_alpha cannot be combined with frozen-standard fusion.")
    if args.fusion_standard_checkpoint_dir and (
        args.doc_chunk_words or args.lexical_weight > 0.0 or args.lexical_hash_dim > 0
    ):
        raise ValueError("--doc_chunk_* and --lexical_* cannot be combined with frozen-standard fusion.")
    if not args.fusion_standard_checkpoint_dir and args.fusion_scope != "both":
        raise ValueError("--fusion_scope may only be set when fusion is enabled.")
    if args.fusion_alpha is not None and not 0.0 <= float(args.fusion_alpha) <= 1.0:
        raise ValueError("--fusion_alpha must be in [0, 1].")
    if args.doc_loop_idx is not None and args.doc_loop_idx <= 0:
        raise ValueError("--doc_loop_idx must be a positive integer.")
    if args.doc_loop_idx is not None and not args.loop_docs:
        raise ValueError("--doc_loop_idx may only be set when --loop_docs true.")
    if args.self_query_alpha is not None and not 0.0 <= float(args.self_query_alpha) <= 1.0:
        raise ValueError("--self_query_alpha must be in [0, 1].")
    if args.self_query_source_loop <= 0:
        raise ValueError("--self_query_source_loop must be a positive integer.")
    if args.doc_chunk_words < 0:
        raise ValueError("--doc_chunk_words must be non-negative.")
    if args.doc_chunk_words and args.loop_docs:
        raise ValueError("--doc_chunk_words cannot be combined with --loop_docs.")
    if args.doc_chunk_words and args.doc_chunk_stride <= 0:
        args.doc_chunk_stride = args.doc_chunk_words
    if args.doc_chunk_stride < 0:
        raise ValueError("--doc_chunk_stride must be non-negative.")
    if args.doc_chunk_max_chunks < 0:
        raise ValueError("--doc_chunk_max_chunks must be non-negative.")
    if args.lexical_hash_dim < 0:
        raise ValueError("--lexical_hash_dim must be non-negative.")
    if not 0.0 <= float(args.lexical_weight) <= 1.0:
        raise ValueError("--lexical_weight must be in [0, 1].")
    if args.lexical_weight > 0.0 and args.lexical_hash_dim <= 0:
        raise ValueError("--lexical_hash_dim must be positive when --lexical_weight is positive.")

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
