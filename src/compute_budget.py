import math
from typing import Any, Dict


def updates_per_epoch(config: Dict[str, Any]) -> int:
    train_sample_size = config.get("train_sample_size")
    if train_sample_size is None:
        raise ValueError("train_sample_size must be set to infer optimizer steps.")
    batch_size = int(config.get("batch_size", 1))
    gradient_accumulation_steps = int(config.get("gradient_accumulation_steps", 1))
    micro_batches = math.ceil(int(train_sample_size) / batch_size)
    return math.ceil(micro_batches / gradient_accumulation_steps)


def loop_compute_multiplier(config: Dict[str, Any]) -> float:
    tmax = int(config.get("tmax", 1))
    num_negatives = int(config.get("num_negatives", 0))
    return (tmax + num_negatives + 1) / (num_negatives + 2)


def more_steps_budget(config: Dict[str, Any]) -> Dict[str, Any]:
    base_epochs = int(config.get("epochs", 1))
    base_steps = base_epochs * updates_per_epoch(config)
    multiplier = loop_compute_multiplier(config)
    target_steps = math.ceil(base_steps * multiplier)
    train_epochs = math.ceil(target_steps / updates_per_epoch(config))
    return {
        "base_steps": base_steps,
        "compute_multiplier": multiplier,
        "target_steps": target_steps,
        "train_epochs": train_epochs,
    }
