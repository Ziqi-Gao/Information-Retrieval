import json
import random
from typing import Any, Dict, List, Optional

import torch
from datasets import load_dataset


def _preview_sample(sample: Dict[str, Any]) -> str:
    preview = {}
    for key, value in sample.items():
        if isinstance(value, list):
            preview[key] = value[:2]
        else:
            preview[key] = value
    return json.dumps(preview, ensure_ascii=False, default=str)[:4000]


def passage_to_text(passage: Any, field_name: str = "passage") -> str:
    if isinstance(passage, str):
        text = passage.strip()
        if text:
            return text
        raise ValueError(f"{field_name} is an empty string.")

    if isinstance(passage, dict):
        title = str(passage.get("title", "") or "").strip()
        text = passage.get("text", None)
        if text is None:
            text = passage.get("passage", passage.get("contents", passage.get("content", None)))
        if text is None:
            raise ValueError(
                f"{field_name} dict must contain a text-like field. Available keys: {sorted(passage.keys())}"
            )
        text = str(text).strip()
        if not text and not title:
            raise ValueError(f"{field_name} has neither title nor text content.")
        return f"{title}\n{text}" if title else text

    raise ValueError(
        f"{field_name} must be a string or dict with title/text fields, got {type(passage).__name__}."
    )


def _first_positive(item: Dict[str, Any]) -> str:
    positives = item.get("positive_passages", None)
    if not isinstance(positives, (list, tuple)) or not positives:
        raise ValueError("positive_passages must be a non-empty list.")
    return passage_to_text(positives[0], field_name="positive_passages[0]")


def _first_k_negatives(item: Dict[str, Any], k: int) -> Optional[List[str]]:
    negatives = item.get("negative_passages", None)
    if not isinstance(negatives, (list, tuple)):
        raise ValueError("negative_passages must be a list.")
    if len(negatives) < k:
        return None
    return [passage_to_text(negative, field_name=f"negative_passages[{idx}]") for idx, negative in enumerate(negatives[:k])]


class RLHNRetrievalDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_name: str = "rlhn/rlhn-680K",
        train_sample_size: Optional[int] = None,
        smoke_test_sample_size: Optional[int] = None,
        num_negatives: int = 7,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.dataset_name = dataset_name
        self.num_negatives = num_negatives
        target_size = smoke_test_sample_size if smoke_test_sample_size is not None else train_sample_size

        raw = load_dataset(dataset_name, split="train")
        order = list(range(len(raw)))
        random.Random(seed).shuffle(order)

        records: List[Dict[str, Any]] = []
        first_error: Optional[Exception] = None
        first_error_sample: Optional[Dict[str, Any]] = None

        for idx in order:
            item = raw[int(idx)]
            try:
                query = item.get("query", None)
                if not isinstance(query, str) or not query.strip():
                    raise ValueError("query must be a non-empty string.")
                positive = _first_positive(item)
                negatives = _first_k_negatives(item, num_negatives)
            except Exception as exc:  # noqa: BLE001 - include sample structure in the final error.
                if first_error is None:
                    first_error = exc
                    first_error_sample = item
                continue

            if negatives is None:
                continue

            records.append(
                {
                    "query": query.strip(),
                    "positive": positive,
                    "negatives": negatives,
                }
            )
            if target_size is not None and len(records) >= int(target_size):
                break

        if not records:
            detail = _preview_sample(first_error_sample) if first_error_sample is not None else "<no sample available>"
            raise RuntimeError(
                f"No usable samples were loaded from {dataset_name}. First parsing error: {first_error}. "
                f"Sample structure: {detail}"
            )

        if target_size is not None and len(records) < int(target_size):
            raise RuntimeError(
                f"Requested {target_size} usable samples from {dataset_name}, but only found {len(records)} "
                f"with at least {num_negatives} negatives."
            )

        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.records[idx]


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "queries": [item["query"] for item in batch],
        "positives": [item["positive"] for item in batch],
        "negatives": [item["negatives"] for item in batch],
    }
