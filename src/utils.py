import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

import numpy as np
import torch
import yaml


def str2bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    value = str(value).strip().lower()
    if value in {"true", "1", "yes", "y", "on"}:
        return True
    if value in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def ensure_dir(path: Union[os.PathLike, str]) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_yaml(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {path}")
    return data


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if torch.is_tensor(obj):
        if obj.numel() == 1:
            return obj.detach().cpu().item()
        return obj.detach().cpu().tolist()
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    return str(obj)


def write_json(path: Union[os.PathLike, str], obj: Any, indent: int = 2) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=indent, sort_keys=True, default=json_default)
        handle.write("\n")


def append_jsonl(path: Union[os.PathLike, str], row: Dict[str, Any]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, default=json_default, sort_keys=True) + "\n")


def tensor_to_float(value: Any) -> float:
    if value is None:
        return float("nan")
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


def current_lr(optimizer: torch.optim.Optimizer, group_idx: int) -> float:
    if group_idx >= len(optimizer.param_groups):
        return float("nan")
    return float(optimizer.param_groups[group_idx]["lr"])


def make_jsonable(obj: Any) -> Any:
    if hasattr(obj, "to_dict"):
        return make_jsonable(obj.to_dict())
    if isinstance(obj, dict):
        return {str(key): make_jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_jsonable(value) for value in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    return str(obj)


def acquire_file_lock(lock_path: Union[os.PathLike, str], timeout_s: int = 600, poll_s: float = 0.25) -> None:
    lock_path = Path(lock_path)
    ensure_dir(lock_path.parent)
    start = time.time()
    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            return
        except FileExistsError:
            if time.time() - start > timeout_s:
                raise TimeoutError(f"Timed out waiting for lock: {lock_path}")
            time.sleep(poll_s)


def release_file_lock(lock_path: Union[os.PathLike, str]) -> None:
    try:
        Path(lock_path).unlink()
    except FileNotFoundError:
        pass


def merge_cli_with_config(args: argparse.Namespace, defaults: Dict[str, Any], config: Dict[str, Any]) -> argparse.Namespace:
    merged = dict(defaults)
    merged.update(config)
    for key, value in vars(args).items():
        if key == "config":
            merged[key] = value
        elif value is not None:
            merged[key] = value
    return argparse.Namespace(**merged)


def add_bool_arg(parser: argparse.ArgumentParser, name: str, help_text: str) -> None:
    parser.add_argument(
        f"--{name}",
        default=None,
        action=argparse.BooleanOptionalAction,
        help=help_text,
    )


def non_null_rows(rows: Iterable[Dict[str, Any]], keys: List[str]) -> List[Dict[str, Any]]:
    return [{key: row.get(key) for key in keys} for row in rows]
