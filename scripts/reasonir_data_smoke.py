#!/usr/bin/env python3
"""Smoke-test ReasonIR HQ parsing into retrieval training triples."""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import ReasonIRRetrievalDataset  # noqa: E402


def preview(text: str, limit: int) -> str:
    text = " ".join(str(text or "").split())
    return text[:limit] + ("..." if len(text) > limit else "")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load and preview parsed ReasonIR retrieval samples.")
    parser.add_argument("--dataset-name", default="reasonir/reasonir-data")
    parser.add_argument("--dataset-config", default="hq")
    parser.add_argument("--split", default="train")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--num-negatives", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preview-chars", type=int, default=220)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = ReasonIRRetrievalDataset(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        split=args.split,
        smoke_test_sample_size=args.samples,
        num_negatives=args.num_negatives,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "dataset_name": args.dataset_name,
                "dataset_config": args.dataset_config,
                "split": args.split,
                "features": str(dataset.features),
                "parsed_samples": len(dataset),
                "num_negatives": args.num_negatives,
            },
            indent=2,
            sort_keys=True,
        )
    )
    for idx in range(len(dataset)):
        row = dataset[idx]
        print(f"--- sample {idx} source_index={row.get('source_index')} positive_id={row.get('positive_id')}")
        print("query:", preview(row["query"], args.preview_chars))
        print("positive:", preview(row["positive"], args.preview_chars))
        for neg_idx, negative in enumerate(row["negatives"]):
            print(f"negative[{neg_idx}]:", preview(negative, args.preview_chars))


if __name__ == "__main__":
    main()
