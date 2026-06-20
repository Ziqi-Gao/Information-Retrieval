#!/usr/bin/env python3
"""Inspect Slurm status for goal-control batches."""

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from goal_common import atomic_write_json, load_json, repo_status


SLURM_STATUS_MAP = {
    "PENDING": "pending",
    "CONFIGURING": "pending",
    "RUNNING": "running",
    "COMPLETING": "running",
    "COMPLETED": "completed",
    "FAILED": "failed",
    "CANCELLED": "cancelled",
    "CANCELLED+": "cancelled",
    "TIMEOUT": "timeout",
    "OUT_OF_MEMORY": "failed",
    "NODE_FAIL": "failed",
}
TERMINAL = {"completed", "failed", "cancelled", "timeout"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect Slurm state for a goal batch.")
    parser.add_argument("--state", default="outputs/goal/state.json")
    parser.add_argument("--batch-id", default=None)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--update-state", action="store_true")
    return parser.parse_args()


def run_command(command: List[str]) -> Optional[str]:
    try:
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        stdout, _ = proc.communicate()
    except OSError:
        return None
    if proc.returncode != 0:
        return None
    return stdout.strip()


def map_status(raw: str) -> str:
    if not raw:
        return "unknown"
    primary = raw.split()[0].split("+")[0]
    return SLURM_STATUS_MAP.get(primary, SLURM_STATUS_MAP.get(raw, "unknown"))


def query_squeue(job_id: str) -> Optional[Dict[str, Any]]:
    if not shutil.which("squeue"):
        return None
    output = run_command(["squeue", "-h", "-j", str(job_id), "-o", "%i|%T|%M|%R"])
    if not output:
        return None
    line = output.splitlines()[0]
    parts = line.split("|")
    raw = parts[1] if len(parts) > 1 else ""
    return {"source": "squeue", "raw_state": raw, "status": map_status(raw), "line": line}


def query_sacct(job_id: str) -> Optional[Dict[str, Any]]:
    if not shutil.which("sacct"):
        return None
    output = run_command(["sacct", "-n", "-P", "-j", str(job_id), "--format=JobIDRaw,State,Elapsed,ExitCode"])
    if not output:
        return None
    best_line = None
    for line in output.splitlines():
        if line.split("|", 1)[0] == str(job_id):
            best_line = line
            break
    best_line = best_line or output.splitlines()[0]
    parts = best_line.split("|")
    raw = parts[1] if len(parts) > 1 else ""
    return {"source": "sacct", "raw_state": raw, "status": map_status(raw), "line": best_line}


def job_status(job_id: Optional[str]) -> Dict[str, Any]:
    if not job_id:
        return {"job_id": job_id, "status": "unknown", "source": "none", "raw_state": None}
    live = query_squeue(job_id)
    if live:
        live["job_id"] = job_id
        return live
    historical = query_sacct(job_id)
    if historical:
        historical["job_id"] = job_id
        return historical
    return {"job_id": job_id, "status": "unknown", "source": "none", "raw_state": None}


def load_plan(batch_id: str) -> Dict[str, Any]:
    path = Path("outputs/goal/runs") / batch_id / "submission_plan.json"
    if not path.exists():
        raise SystemExit("Missing submission plan for batch {}: {}".format(batch_id, path))
    return load_json(path)


def main() -> None:
    args = parse_args()
    state_path = Path(args.state)
    state: Dict[str, Any] = load_json(state_path) if state_path.exists() else {}
    batch_id = args.batch_id or ((state.get("current_batch") or {}).get("batch_id"))
    if not batch_id:
        raise SystemExit("Provide --batch-id or initialize state with a current batch.")
    plan = load_plan(batch_id)

    rows: List[Dict[str, Any]] = []
    for job in plan.get("jobs", []):
        for job_type, key in [("train", "train_job_id"), ("eval", "eval_job_id")]:
            if job.get("eval_only") and job_type == "train":
                continue
            status = job_status(job.get(key))
            row = {
                "batch_id": batch_id,
                "run_id": job.get("run_id"),
                "type": job_type,
                "job_id": job.get(key),
                "status": "dry_run" if plan.get("dry_run") and not job.get(key) else status["status"],
                "source": status.get("source"),
                "raw_state": status.get("raw_state"),
            }
            rows.append(row)

    postprocess = plan.get("postprocess") or {}
    postprocess_job_id = plan.get("postprocess_job_id") or postprocess.get("job_id")
    if postprocess.get("enabled") or postprocess_job_id:
        status = job_status(postprocess_job_id)
        rows.append(
            {
                "batch_id": batch_id,
                "run_id": batch_id,
                "type": "postprocess",
                "job_id": postprocess_job_id,
                "status": "dry_run" if plan.get("dry_run") and not postprocess_job_id else status["status"],
                "source": status.get("source"),
                "raw_state": status.get("raw_state"),
            }
        )

    result = {"batch_id": batch_id, "repo": repo_status(), "jobs": rows}
    if args.update_state and state_path.exists():
        state["repo"] = repo_status()
        state["open_jobs"] = rows
        if rows and all(row["status"] in TERMINAL or row["status"] == "dry_run" for row in rows):
            state["next_required_action"] = "Collect results for completed jobs or inspect failures."
        atomic_write_json(state_path, state)

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print("Batch {}".format(batch_id))
        for row in rows:
            print("- {run_id} {type}: {status} job={job_id}".format(**row))


if __name__ == "__main__":
    main()
