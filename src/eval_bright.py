import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch
import torch.nn.functional as F
from datasets import load_dataset

from .experiments import get_version_spec, version_names
from .model import load_model
from .utils import ensure_dir, safe_task_dir_name, str2bool, write_json


DEFAULT_DOMAINS = ["biology", "economics", "psychology", "stackoverflow"]
SUMMARY_COLUMNS = [
    "version",
    "domain",
    "task",
    "loop_idx",
    "ndcg_at_10",
    "num_queries",
    "num_docs",
    "use_long_documents",
    "checkpoint_dir",
    "raw_result_path",
]


def split_csv(value: Optional[str], fallback: Sequence[str]) -> List[str]:
    if value is None or not str(value).strip():
        return list(fallback)
    items: List[str] = []
    seen = set()
    for part in str(value).replace(";", ",").split(","):
        item = part.strip()
        if item and item not in seen:
            seen.add(item)
            items.append(item)
    return items


def normalize_excluded_ids(value: Any) -> Set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        text = value.strip()
        return set() if not text or text.upper() == "N/A" else {text}
    if isinstance(value, (list, tuple, set)):
        output = set()
        for item in value:
            output.update(normalize_excluded_ids(item))
        return output
    return {str(value)}


def normalize_gold_ids(value: Any) -> Set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        text = value.strip()
        return {text} if text else set()
    if isinstance(value, (list, tuple, set)):
        output = set()
        for item in value:
            output.update(normalize_gold_ids(item))
        return output
    return {str(value)}


def ndcg_at_k(ranked_doc_ids: Sequence[str], gold_ids: Set[str], k: int = 10) -> float:
    if not gold_ids:
        return 0.0
    dcg = 0.0
    for rank, doc_id in enumerate(ranked_doc_ids[:k], start=1):
        if doc_id in gold_ids:
            dcg += 1.0 / math.log2(rank + 1)
    ideal_hits = min(len(gold_ids), int(k))
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return 0.0 if idcg <= 0.0 else dcg / idcg


def append_summary_rows(output_dir: Path, rows: List[Dict[str, Any]]) -> None:
    summary_path = output_dir / "results_summary.csv"
    combined: List[Dict[str, Any]] = []
    if summary_path.exists():
        with open(summary_path, "r", newline="", encoding="utf-8") as handle:
            combined.extend(csv.DictReader(handle))
    combined.extend(rows)

    deduped: Dict[Tuple[str, str, str, str, str], Dict[str, Any]] = {}
    for row in combined:
        key = (
            str(row.get("version", "")),
            str(row.get("domain", "")),
            str(row.get("task", "")),
            str(row.get("loop_idx", "")),
            str(row.get("checkpoint_dir", "")),
        )
        deduped[key] = row

    with open(summary_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in deduped.values():
            writer.writerow({column: row.get(column, "") for column in SUMMARY_COLUMNS})


def write_domain_summary(artifact_dir: Path, row: Dict[str, Any]) -> None:
    with open(artifact_dir / "results_summary.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerow({column: row.get(column, "") for column in SUMMARY_COLUMNS})


def load_bright_domain(domain: str, use_long_documents: bool, max_queries: Optional[int], max_docs: Optional[int]) -> Dict[str, Any]:
    examples = load_dataset("xlangai/BRIGHT", "examples")[domain]
    document_config = "long_documents" if use_long_documents else "documents"
    documents = load_dataset("xlangai/BRIGHT", document_config)[domain]
    if max_queries is not None:
        examples = examples.select(range(min(int(max_queries), len(examples))))
    if max_docs is not None:
        documents = documents.select(range(min(int(max_docs), len(documents))))

    doc_ids = [str(doc["id"]) for doc in documents]
    doc_texts = [str(doc.get("content", "") or "") for doc in documents]
    query_rows = []
    gold_field = "gold_ids_long" if use_long_documents else "gold_ids"
    for row in examples:
        query_rows.append(
            {
                "id": str(row.get("id", "")),
                "query": str(row.get("query", "") or ""),
                "gold_ids": normalize_gold_ids(row.get(gold_field)),
                "excluded_ids": normalize_excluded_ids(row.get("excluded_ids")),
            }
        )
    return {"doc_ids": doc_ids, "doc_texts": doc_texts, "queries": query_rows}


def encode_corpus(model, texts: List[str], batch_size: int, device: torch.device) -> torch.Tensor:
    chunks = []
    for start in range(0, len(texts), int(batch_size)):
        with torch.no_grad():
            emb = model.encode_docs(texts[start : start + int(batch_size)], batch_size=int(batch_size), device=device)
        chunks.append(emb.detach().cpu())
    if not chunks:
        return torch.empty(0, model.embedding_dim)
    return torch.cat(chunks, dim=0)


def encode_queries(model, texts: List[str], loop_idx: int, batch_size: int, device: torch.device) -> torch.Tensor:
    chunks = []
    for start in range(0, len(texts), int(batch_size)):
        with torch.no_grad():
            emb = model.encode_queries(
                texts[start : start + int(batch_size)],
                batch_size=int(batch_size),
                loop_idx=int(loop_idx),
                return_all_loops=False,
                device=device,
            )
        chunks.append(emb.detach().cpu())
    if not chunks:
        return torch.empty(0, model.embedding_dim)
    return torch.cat(chunks, dim=0)


def topk_rankings(
    query_emb: torch.Tensor,
    corpus_emb: torch.Tensor,
    doc_ids: List[str],
    query_rows: List[Dict[str, Any]],
    device: torch.device,
    score_chunk_size: int,
    k: int = 10,
) -> List[List[Tuple[str, float]]]:
    if corpus_emb.size(0) != len(doc_ids):
        raise ValueError("corpus_emb and doc_ids length mismatch.")

    doc_index = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}
    excluded_indices = [
        {doc_index[doc_id] for doc_id in row["excluded_ids"] if doc_id in doc_index}
        for row in query_rows
    ]
    rankings: List[List[Tuple[str, float]]] = []
    for q_start in range(0, query_emb.size(0), 16):
        q_chunk = query_emb[q_start : q_start + 16].to(device)
        batch_size = q_chunk.size(0)
        top_scores = torch.full((batch_size, k), float("-inf"), device=device)
        top_indices = torch.full((batch_size, k), -1, dtype=torch.long, device=device)
        for d_start in range(0, corpus_emb.size(0), int(score_chunk_size)):
            d_end = min(d_start + int(score_chunk_size), corpus_emb.size(0))
            d_chunk = corpus_emb[d_start:d_end].to(device)
            scores = torch.matmul(q_chunk, d_chunk.transpose(0, 1))
            for row_idx in range(batch_size):
                excluded = [
                    idx - d_start
                    for idx in excluded_indices[q_start + row_idx]
                    if d_start <= idx < d_end
                ]
                if excluded:
                    scores[row_idx, torch.tensor(excluded, dtype=torch.long, device=device)] = float("-inf")
            chunk_k = min(k, scores.size(1))
            chunk_scores, chunk_local_indices = torch.topk(scores, k=chunk_k, dim=1)
            chunk_global_indices = chunk_local_indices + d_start
            combined_scores = torch.cat([top_scores, chunk_scores], dim=1)
            combined_indices = torch.cat([top_indices, chunk_global_indices], dim=1)
            top_scores, select = torch.topk(combined_scores, k=k, dim=1)
            top_indices = torch.gather(combined_indices, dim=1, index=select)

        for row_idx in range(batch_size):
            row = []
            for score, doc_idx in zip(top_scores[row_idx].detach().cpu().tolist(), top_indices[row_idx].detach().cpu().tolist()):
                if doc_idx >= 0:
                    row.append((doc_ids[int(doc_idx)], float(score)))
            rankings.append(row)
    return rankings


def evaluate_domain(args: argparse.Namespace, model, device: torch.device, domain: str, loop_idx: int) -> Dict[str, Any]:
    data = load_bright_domain(domain, args.use_long_documents, args.max_queries, args.max_docs)
    doc_ids = data["doc_ids"]
    doc_texts = data["doc_texts"]
    query_rows = data["queries"]
    queries = [row["query"] for row in query_rows]

    artifact_dir = ensure_dir(Path(args.output_dir) / args.version / safe_task_dir_name(domain) / f"loop{loop_idx}")
    corpus_emb = encode_corpus(model, doc_texts, args.corpus_batch_size, device)
    query_emb = encode_queries(model, queries, loop_idx, args.query_batch_size, device)
    corpus_emb = F.normalize(corpus_emb, p=2, dim=-1)
    query_emb = F.normalize(query_emb, p=2, dim=-1)

    rankings = topk_rankings(
        query_emb=query_emb,
        corpus_emb=corpus_emb,
        doc_ids=doc_ids,
        query_rows=query_rows,
        device=device,
        score_chunk_size=args.score_chunk_size,
        k=10,
    )
    diagnostics = []
    values = []
    for row, ranking in zip(query_rows, rankings):
        ranked_ids = [doc_id for doc_id, _ in ranking]
        value = ndcg_at_k(ranked_ids, row["gold_ids"], k=10)
        values.append(value)
        diagnostics.append(
            {
                "query_id": row["id"],
                "gold_ids": sorted(row["gold_ids"]),
                "excluded_ids": sorted(row["excluded_ids"]),
                "ndcg_at_10": value,
                "top10": [{"doc_id": doc_id, "score": score} for doc_id, score in ranking],
            }
        )

    raw_path = artifact_dir / "raw_query_rankings.jsonl"
    with open(raw_path, "w", encoding="utf-8") as handle:
        for item in diagnostics:
            handle.write(json.dumps(item, sort_keys=True) + "\n")
    metrics = {
        "domain": domain,
        "loop_idx": int(loop_idx),
        "ndcg_at_10": sum(values) / len(values) if values else float("nan"),
        "num_queries": len(query_rows),
        "num_docs": len(doc_ids),
        "use_long_documents": bool(args.use_long_documents),
    }
    write_json(artifact_dir / "parsed_metrics.json", metrics)

    row = {
        "version": args.version,
        "domain": domain,
        "task": domain,
        "loop_idx": int(loop_idx),
        "ndcg_at_10": metrics["ndcg_at_10"],
        "num_queries": len(query_rows),
        "num_docs": len(doc_ids),
        "use_long_documents": str(bool(args.use_long_documents)).lower(),
        "checkpoint_dir": str(args.checkpoint_dir),
        "raw_result_path": str(raw_path),
    }
    write_domain_summary(artifact_dir, row)
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate loop-wise retrievers on BRIGHT.")
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--version", choices=version_names(), required=True)
    parser.add_argument("--domains", default=",".join(DEFAULT_DOMAINS))
    parser.add_argument("--use-long-documents", type=str2bool, default=False)
    parser.add_argument("--metric", default="ndcg_at_10")
    parser.add_argument("--loop_idx", type=int, default=None)
    parser.add_argument("--eval_all_loops", type=str2bool, default=False)
    parser.add_argument("--query_batch_size", type=int, default=32)
    parser.add_argument("--corpus_batch_size", type=int, default=64)
    parser.add_argument("--score_chunk_size", type=int, default=8192)
    parser.add_argument("--max_queries", type=int, default=None)
    parser.add_argument("--max_docs", type=int, default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.metric != "ndcg_at_10":
        raise ValueError("BRIGHT evaluator currently supports only --metric ndcg_at_10.")
    ensure_dir(args.output_dir)
    domains = split_csv(args.domains, DEFAULT_DOMAINS)
    requested_device = torch.device(args.device)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA was requested for BRIGHT evaluation, but it is unavailable. Falling back to CPU.")
        requested_device = torch.device("cpu")

    model = load_model(args.checkpoint_dir, map_location="cpu").to(requested_device)
    model.eval()
    if get_version_spec(args.version).is_standard_family:
        loop_indices = [1]
    elif args.loop_idx is not None and args.eval_all_loops:
        loop_indices = list(range(1, model.tmax + 1))
    elif args.loop_idx is not None:
        loop_indices = [int(args.loop_idx)]
    elif args.eval_all_loops:
        loop_indices = list(range(1, model.tmax + 1))
    else:
        loop_indices = [model.tmax]

    rows = []
    for loop_idx in loop_indices:
        for domain in domains:
            print(f"Evaluating {args.version} loop {loop_idx} on BRIGHT/{domain}")
            rows.append(evaluate_domain(args, model, requested_device, domain, loop_idx))
    append_summary_rows(Path(args.output_dir), rows)
    macro_rows = [row for row in rows if row.get("ndcg_at_10") == row.get("ndcg_at_10")]
    macro = sum(float(row["ndcg_at_10"]) for row in macro_rows) / len(macro_rows) if macro_rows else float("nan")
    write_json(
        Path(args.output_dir) / "macro_summary.json",
        {
            "version": args.version,
            "domains": domains,
            "loop_indices": loop_indices,
            "metric": args.metric,
            "macro_ndcg_at_10": macro,
        },
    )
    print(f"Wrote BRIGHT summary rows to {Path(args.output_dir) / 'results_summary.csv'}")
    print(f"Macro ndcg_at_10 over produced rows: {macro:.6f}")


if __name__ == "__main__":
    main()
