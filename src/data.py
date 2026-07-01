import json
import re
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

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


PASSAGE_SAMPLING_STRATEGIES = {"first", "middle_negatives", "seeded_random"}


def _positive_passages(item: Dict[str, Any]) -> Sequence[Any]:
    positives = item.get("positive_passages", None)
    if not isinstance(positives, (list, tuple)) or not positives:
        raise ValueError("positive_passages must be a non-empty list.")
    return positives


def _negative_passages(item: Dict[str, Any]) -> Sequence[Any]:
    negatives = item.get("negative_passages", None)
    if not isinstance(negatives, (list, tuple)):
        raise ValueError("negative_passages must be a list.")
    return negatives


def _first_positive(item: Dict[str, Any]) -> str:
    positives = _positive_passages(item)
    return passage_to_text(positives[0], field_name="positive_passages[0]")


def _first_k_negatives(item: Dict[str, Any], k: int) -> Optional[List[str]]:
    negatives = _negative_passages(item)
    if len(negatives) < k:
        return None
    return [passage_to_text(negative, field_name=f"negative_passages[{idx}]") for idx, negative in enumerate(negatives[:k])]


def _middle_k_negatives(item: Dict[str, Any], k: int) -> Optional[List[str]]:
    negatives = _negative_passages(item)
    if len(negatives) < k:
        return None
    start = max(0, (len(negatives) - k) // 2)
    selected = list(range(start, start + k))
    return [passage_to_text(negatives[idx], field_name=f"negative_passages[{idx}]") for idx in selected]


def _sampled_positive_and_negatives(item: Dict[str, Any], k: int, seed: int, idx: int) -> Optional[tuple[str, List[str]]]:
    positives = _positive_passages(item)
    negatives = _negative_passages(item)
    if len(negatives) < k:
        return None

    rng = random.Random((int(seed) + 1) * 1_000_003 + int(idx))
    positive_idx = rng.randrange(len(positives))
    negative_indices = rng.sample(range(len(negatives)), k)
    positive = passage_to_text(positives[positive_idx], field_name=f"positive_passages[{positive_idx}]")
    sampled_negatives = [
        passage_to_text(negatives[negative_idx], field_name=f"negative_passages[{negative_idx}]")
        for negative_idx in negative_indices
    ]
    return positive, sampled_negatives


class RLHNRetrievalDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_name: str = "rlhn/rlhn-680K",
        train_sample_size: Optional[int] = None,
        smoke_test_sample_size: Optional[int] = None,
        num_negatives: int = 7,
        seed: int = 42,
        passage_sampling_strategy: str = "first",
    ) -> None:
        super().__init__()
        self.dataset_name = dataset_name
        self.num_negatives = num_negatives
        if passage_sampling_strategy not in PASSAGE_SAMPLING_STRATEGIES:
            known = ", ".join(sorted(PASSAGE_SAMPLING_STRATEGIES))
            raise ValueError(f"passage_sampling_strategy must be one of {{{known}}}, got {passage_sampling_strategy!r}.")
        self.passage_sampling_strategy = str(passage_sampling_strategy)
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
                if self.passage_sampling_strategy == "seeded_random":
                    sampled = _sampled_positive_and_negatives(item, num_negatives, seed=seed, idx=int(idx))
                    if sampled is None:
                        negatives = None
                        positive = ""
                    else:
                        positive, negatives = sampled
                elif self.passage_sampling_strategy == "middle_negatives":
                    positive = _first_positive(item)
                    negatives = _middle_k_negatives(item, num_negatives)
                else:
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


ID_LIKE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{1,159}$")


def _is_text_like(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    text = value.strip()
    if not text:
        return False
    if len(text) >= 80:
        return True
    if any(ch.isspace() for ch in text) and len(text) >= 20:
        return True
    return False


def _is_id_like(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    text = value.strip()
    if not text:
        return False
    if _is_text_like(text):
        return False
    return bool(ID_LIKE_RE.match(text))


def reasonir_query_to_text(value: Any) -> str:
    if isinstance(value, str):
        text = value.strip()
        if text:
            return text
        raise ValueError("query is an empty string.")

    if isinstance(value, (list, tuple)):
        parts = [str(part).strip() for part in value if isinstance(part, str) and str(part).strip()]
        if parts:
            return " ".join(parts)
        raise ValueError("query list contains no non-empty string parts.")

    if isinstance(value, dict):
        parts = []
        for key in ("instruction", "query", "question", "text"):
            item = value.get(key)
            if isinstance(item, str) and item.strip():
                parts.append(item.strip())
        if parts:
            return " ".join(parts)
        raise ValueError("query dict contains no instruction/query/question/text string fields.")

    raise ValueError(f"query must be a string, list, or dict, got {type(value).__name__}.")


def _join_instruction_and_text(instruction: str, text: str) -> str:
    instruction = str(instruction or "").strip()
    text = str(text or "").strip()
    if instruction and text and instruction not in text:
        return f"{instruction} {text}"
    return text or instruction


def reasonir_passage_to_text_or_id(value: Any, field_name: str) -> Tuple[Optional[str], Optional[str]]:
    """Return ``(text, doc_id)`` for a ReasonIR positive/negative passage cell.

    ReasonIR HQ rows store positives as [instruction, doc_id] and negatives as
    [instruction, text]. VL rows may already contain text on both sides. The
    parser accepts either shape, but it refuses ambiguous id-only values until a
    caller resolves them through the BRIGHT documents datastore.
    """

    if isinstance(value, str):
        text = value.strip()
        if _is_text_like(text):
            return text, None
        if _is_id_like(text):
            return None, text
        raise ValueError(f"{field_name} string is neither text-like nor id-like: {text[:120]!r}")

    if isinstance(value, dict):
        doc_id = value.get("id")
        text = value.get("text", value.get("passage", value.get("contents", value.get("content"))))
        title = str(value.get("title", "") or "").strip()
        if isinstance(text, str) and text.strip():
            body = f"{title}\n{text.strip()}" if title else text.strip()
            return body, str(doc_id).strip() if doc_id is not None else None
        if doc_id is not None and _is_id_like(str(doc_id)):
            return None, str(doc_id).strip()
        raise ValueError(f"{field_name} dict has no text/content field or id. Keys: {sorted(value.keys())}")

    if isinstance(value, (list, tuple)):
        strings = [str(part).strip() for part in value if isinstance(part, str) and str(part).strip()]
        text_candidates = [part for part in strings if _is_text_like(part)]
        id_candidates = [part for part in strings if _is_id_like(part)]
        if text_candidates:
            text = max(text_candidates, key=len)
            instruction = ""
            for part in strings:
                if part != text and not _is_id_like(part):
                    instruction = part
                    break
            doc_id = id_candidates[0] if id_candidates else None
            return _join_instruction_and_text(instruction, text), doc_id
        if id_candidates:
            return None, id_candidates[-1]
        raise ValueError(f"{field_name} list contains no text-like or id-like string parts: {strings[:4]!r}")

    raise ValueError(f"{field_name} must be a string, dict, or list, got {type(value).__name__}.")


def _first_passage_cell(value: Any, field_name: str) -> Any:
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError(f"{field_name} must contain at least one passage.")
        return value[0]
    return value


def _first_k_reasonir_negatives(value: Any, k: int) -> List[Any]:
    if not isinstance(value, (list, tuple)):
        raise ValueError("neg must be a list of passages.")
    if len(value) < int(k):
        raise ValueError(f"neg contains {len(value)} passages, but num_negatives={k}.")
    return list(value[: int(k)])


def _load_bright_doc_texts_for_ids(required_ids: Sequence[str]) -> Dict[str, str]:
    missing = {str(doc_id) for doc_id in required_ids if str(doc_id).strip()}
    id_to_text: Dict[str, str] = {}
    if not missing:
        return id_to_text

    bright_docs = load_dataset("xlangai/BRIGHT", "documents")
    for domain in bright_docs.keys():
        for doc in bright_docs[domain]:
            doc_id = str(doc.get("id", "")).strip()
            if doc_id in missing:
                content = str(doc.get("content", "") or "").strip()
                if not content:
                    raise RuntimeError(f"BRIGHT document {doc_id!r} in domain {domain!r} has empty content.")
                id_to_text[doc_id] = content
                missing.remove(doc_id)
                if not missing:
                    return id_to_text
    if missing:
        preview = ", ".join(sorted(missing)[:10])
        raise RuntimeError(f"Could not resolve {len(missing)} ReasonIR document id(s) in BRIGHT documents: {preview}")
    return id_to_text


class ReasonIRRetrievalDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_name: str = "reasonir/reasonir-data",
        dataset_config: str = "hq",
        split: str = "train",
        train_sample_size: Optional[int] = None,
        smoke_test_sample_size: Optional[int] = None,
        num_negatives: int = 1,
        seed: int = 42,
    ) -> None:
        super().__init__()
        if int(num_negatives) < 1:
            raise ValueError("num_negatives must be at least 1.")
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.split = split
        self.num_negatives = int(num_negatives)
        target_size = smoke_test_sample_size if smoke_test_sample_size is not None else train_sample_size

        raw = load_dataset(dataset_name, dataset_config, split=split)
        self.features = raw.features
        order = list(range(len(raw)))
        random.Random(seed).shuffle(order)

        staged: List[Dict[str, Any]] = []
        required_doc_ids: List[str] = []
        first_error: Optional[Exception] = None
        first_error_sample: Optional[Dict[str, Any]] = None

        for idx in order:
            item = raw[int(idx)]
            try:
                query = reasonir_query_to_text(item.get("query"))
                pos_text, pos_id = reasonir_passage_to_text_or_id(
                    _first_passage_cell(item.get("pos"), "pos"),
                    "pos[0]",
                )
                neg_entries = []
                for neg_idx, neg_cell in enumerate(_first_k_reasonir_negatives(item.get("neg"), self.num_negatives)):
                    neg_text, neg_id = reasonir_passage_to_text_or_id(neg_cell, f"neg[{neg_idx}]")
                    if neg_text is None and neg_id:
                        required_doc_ids.append(neg_id)
                    neg_entries.append({"text": neg_text, "doc_id": neg_id})
                if pos_text is None and pos_id:
                    required_doc_ids.append(pos_id)
            except Exception as exc:  # noqa: BLE001 - preview is included if no records can be staged.
                if first_error is None:
                    first_error = exc
                    first_error_sample = item
                continue

            staged.append(
                {
                    "query": query,
                    "positive": pos_text,
                    "positive_id": pos_id,
                    "negatives": neg_entries,
                    "source_index": int(idx),
                }
            )
            if target_size is not None and len(staged) >= int(target_size):
                break

        if not staged:
            detail = _preview_sample(first_error_sample) if first_error_sample is not None else "<no sample available>"
            raise RuntimeError(
                f"No usable samples were staged from {dataset_name}/{dataset_config}. "
                f"First parsing error: {first_error}. Sample structure: {detail}"
            )
        if target_size is not None and len(staged) < int(target_size):
            raise RuntimeError(
                f"Requested {target_size} usable samples from {dataset_name}/{dataset_config}, "
                f"but only staged {len(staged)}."
            )

        id_to_text = _load_bright_doc_texts_for_ids(sorted(set(required_doc_ids)))
        records: List[Dict[str, Any]] = []
        for staged_item in staged:
            positive = staged_item["positive"]
            if positive is None:
                positive = id_to_text.get(staged_item["positive_id"])
            negatives = []
            negative_ids = []
            for neg_entry in staged_item["negatives"]:
                text = neg_entry["text"]
                if text is None:
                    text = id_to_text.get(neg_entry["doc_id"])
                if not isinstance(text, str) or not text.strip():
                    raise RuntimeError(
                        "ReasonIR negative could not be resolved to text. "
                        f"source_index={staged_item['source_index']} doc_id={neg_entry['doc_id']!r}"
                    )
                negatives.append(text.strip())
                negative_ids.append(neg_entry["doc_id"])
            if not isinstance(positive, str) or not positive.strip():
                raise RuntimeError(
                    "ReasonIR positive could not be resolved to text. "
                    f"source_index={staged_item['source_index']} doc_id={staged_item['positive_id']!r}"
                )
            records.append(
                {
                    "query": staged_item["query"],
                    "positive": positive.strip(),
                    "positive_id": staged_item["positive_id"],
                    "negatives": negatives,
                    "negative_ids": negative_ids,
                    "source_index": staged_item["source_index"],
                }
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
