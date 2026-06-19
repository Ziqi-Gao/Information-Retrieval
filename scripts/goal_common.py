#!/usr/bin/env python3
"""Shared utilities for autonomous goal-control scripts.

This module intentionally avoids importing model, torch, MTEB, or other heavy
libraries so preflight and Slurm orchestration stay cheap and deterministic.
"""

import csv
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - exercised only in broken envs
    yaml = None


PRIMARY_METRIC = "ndcg_at_10"
DEFAULT_WIN_MARGIN = 0.001
FINAL_TASKS = ["SciFact", "NFCorpus", "SCIDOCS", "FiQA2018", "ArguAna", "Touche2020", "TRECCOVID"]
FAILURE_STATUSES = {
    "failed_train",
    "failed_eval",
    "missing_result",
    "invalid_metric",
    "partial_tasks",
    "timeout",
}
SAFE_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def fail(message: str) -> None:
    print("ERROR: {}".format(message), file=sys.stderr)
    raise SystemExit(1)


def load_yaml(path: os.PathLike) -> Dict[str, Any]:
    if yaml is None:
        fail("PyYAML is required. Use the project Python environment or install pyyaml.")
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        fail("YAML file must contain a mapping: {}".format(path))
    return data


def write_yaml(path: os.PathLike, data: Dict[str, Any]) -> None:
    if yaml is None:
        fail("PyYAML is required. Use the project Python environment or install pyyaml.")
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def load_json(path: os.PathLike) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        fail("JSON file must contain an object: {}".format(path))
    return data


def write_json(path: os.PathLike, data: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def atomic_write_json(path: os.PathLike, data: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    fd, tmp_name = tempfile.mkstemp(prefix=".tmp-", suffix=".json", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp_name, str(path))
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def sha256_file(path: os.PathLike) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_dir(path: os.PathLike) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_run_id(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    if not SAFE_RUN_ID_RE.match(value):
        return False
    if ".." in value or "/" in value or "\\" in value:
        return False
    return True


def _git_output(args: Sequence[str]) -> str:
    try:
        proc = subprocess.Popen(
            ["git"] + list(args),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        stdout, _ = proc.communicate()
    except OSError:
        return ""
    if proc.returncode != 0:
        return ""
    return stdout.strip()


def repo_status() -> Dict[str, Any]:
    branch = _git_output(["rev-parse", "--abbrev-ref", "HEAD"]) or None
    commit = _git_output(["rev-parse", "HEAD"]) or None
    dirty = bool(_git_output(["status", "--porcelain"]))
    return {"branch": branch, "commit": commit, "dirty": dirty}


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_csv_rows(path: os.PathLike) -> List[Dict[str, str]]:
    with open(path, "r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: os.PathLike, rows: Iterable[Dict[str, Any]], columns: Sequence[str]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    fd, tmp_name = tempfile.mkstemp(prefix=".tmp-", suffix=".csv", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({column: row.get(column, "") for column in columns})
        os.replace(tmp_name, str(path))
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def write_csv_rows_nonatomic(path: os.PathLike, rows: Iterable[Dict[str, Any]], columns: Sequence[str]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def parse_task_list(value: Any) -> List[str]:
    if value is None:
        return []
    raw_values: List[str] = []
    if isinstance(value, str):
        raw_values.append(value)
    elif isinstance(value, (list, tuple)):
        for item in value:
            raw_values.extend(parse_task_list(item))
    else:
        raw_values.append(str(value))

    tasks: List[str] = []
    seen = set()
    for raw in raw_values:
        for part in str(raw).replace(";", ",").split(","):
            task = part.strip()
            if task and task not in seen:
                seen.add(task)
                tasks.append(task)
    return tasks


def metric_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, str) and not value.strip():
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def maybe_run(command: Sequence[str], dry_run: bool) -> Dict[str, Any]:
    printable = " ".join(str(part) for part in command)
    if dry_run:
        return {"command": list(command), "dry_run": True, "returncode": None, "stdout": "", "stderr": ""}
    proc = subprocess.Popen(
        list(command),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    stdout, stderr = proc.communicate()
    return {
        "command": list(command),
        "dry_run": False,
        "returncode": proc.returncode,
        "stdout": stdout.strip(),
        "stderr": stderr.strip(),
        "printable": printable,
    }


def path_under(path: os.PathLike, root: os.PathLike) -> bool:
    path_obj = Path(path)
    root_obj = Path(root)
    try:
        path_obj.resolve().relative_to(root_obj.resolve())
        return True
    except ValueError:
        return False


def relative_path_under(path: os.PathLike, root: os.PathLike) -> bool:
    path_obj = Path(path)
    if path_obj.is_absolute():
        return False
    parts = path_obj.parts
    if any(part == ".." for part in parts):
        return False
    try:
        path_obj.relative_to(Path(root))
        return True
    except ValueError:
        return False


def validate_baseline_artifacts(summary_csv: os.PathLike, manifest_json: os.PathLike, metric: str = PRIMARY_METRIC) -> Dict[str, Any]:
    summary_path = Path(summary_csv)
    manifest_path = Path(manifest_json)
    if not summary_path.exists() or not manifest_path.exists():
        return {"valid": False, "reason": "baseline summary or manifest is missing"}
    try:
        manifest = load_json(manifest_path)
    except SystemExit:
        return {"valid": False, "reason": "baseline manifest is invalid JSON"}
    if manifest.get("metric") != metric:
        return {"valid": False, "reason": "baseline metric mismatch"}
    if manifest.get("tasks") != FINAL_TASKS:
        return {"valid": False, "reason": "baseline task list mismatch"}
    expected_hash = manifest.get("sha256")
    actual_hash = sha256_file(summary_path)
    if expected_hash != actual_hash:
        return {"valid": False, "reason": "baseline summary sha256 mismatch"}
    rows = read_csv_rows(summary_path)
    by_task: Dict[str, List[Dict[str, str]]] = {task: [] for task in FINAL_TASKS}
    for row in rows:
        if row.get("version") == "standard" and row.get("task") in by_task:
            by_task[row["task"]].append(row)
    per_task: Dict[str, float] = {}
    for task in FINAL_TASKS:
        task_rows = by_task[task]
        if len(task_rows) != 1:
            return {"valid": False, "reason": "{} has {} standard rows".format(task, len(task_rows))}
        value = metric_float(task_rows[0].get(metric))
        if value is None:
            return {"valid": False, "reason": "{} has invalid {}".format(task, metric)}
        per_task[task] = value
    manifest_values = manifest.get("per_task_baseline")
    if isinstance(manifest_values, dict):
        for task, value in per_task.items():
            manifest_value = metric_float(manifest_values.get(task))
            if manifest_value is None or abs(manifest_value - value) > 1e-12:
                return {"valid": False, "reason": "{} baseline value mismatch".format(task)}
    return {
        "valid": True,
        "reason": "",
        "summary": str(summary_path),
        "manifest": str(manifest_path),
        "sha256": actual_hash,
        "per_task_baseline": per_task,
    }


def recursive_fill_missing(base: Dict[str, Any], defaults: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in defaults.items():
        if key not in base:
            base[key] = value
        elif isinstance(base[key], dict) and isinstance(value, dict):
            recursive_fill_missing(base[key], value)
    return base


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Shared helpers for goal-control scripts; not a workflow entry point.")
    parser.add_argument("--print-repo-status", action="store_true", help="Print current git branch/commit/dirty status as JSON.")
    args = parser.parse_args()
    if args.print_repo_status:
        print(json.dumps(repo_status(), indent=2, sort_keys=True))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
