#!/usr/bin/env python3
"""Watch a submitted goal batch and resume only after jobs are terminal."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

from goal_common import atomic_write_json, ensure_dir, load_json


TERMINAL_STATUSES = {
    "completed",
    "failed",
    "cancelled",
    "timeout",
    "missing_result",
    "invalid_metric",
    "partial_tasks",
    "failed_train",
    "failed_eval",
}
NON_TERMINAL_STATUSES = {"pending", "running", "unknown", "dry_run"}


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Poll an existing goal batch until Slurm jobs reach terminal state.")
    parser.add_argument("--state", default="outputs/goal/state.json", help="Goal state JSON path.")
    parser.add_argument("--batch-id", required=True, help="Submitted batch id to watch.")
    parser.add_argument("--interval-seconds", type=float, default=600, help="Polling interval in seconds.")
    parser.add_argument("--max-hours", type=float, default=12, help="Maximum wall-clock watch duration.")
    parser.add_argument("--mode", choices=["notify", "codex"], default="notify", help="Action after all jobs are terminal.")
    parser.add_argument(
        "--codex-output",
        default=None,
        help="Markdown output path for codex mode. Defaults under outputs/goal/runs/<batch_id>/.",
    )
    parser.add_argument("--json", action="store_true", help="In codex mode, invoke `codex exec --json` and write JSONL.")
    parser.add_argument("--force-codex", action="store_true", help="Allow codex mode even if the sentinel already exists.")
    parser.add_argument("--allow-inside-slurm", action="store_true", help="Allow running this watcher inside a Slurm job.")
    parser.add_argument("--status-timeout-seconds", type=float, default=10, help="Timeout for one goal_status.py poll.")
    return parser.parse_args()


def fail(message: str) -> None:
    raise SystemExit("ERROR: {}".format(message))


def append_log(log_path: Path, message: str) -> None:
    ensure_dir(log_path.parent)
    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write("[{}] {}\n".format(now_utc(), message))


def batch_dir(batch_id: str) -> Path:
    return Path("outputs/goal/runs") / batch_id


def load_submission_plan(batch_id: str) -> Dict[str, Any]:
    plan_path = batch_dir(batch_id) / "submission_plan.json"
    if not plan_path.exists():
        fail("Missing submission plan: {}".format(plan_path))
    plan = load_json(plan_path)
    jobs = plan.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        fail("Submission plan has no jobs: {}".format(plan_path))
    submitted_job_ids = []
    for job in jobs:
        for key in ["train_job_id", "eval_job_id"]:
            if job.get(key):
                submitted_job_ids.append(str(job[key]))
    if not submitted_job_ids:
        fail("Batch {} has no submitted job ids in {}".format(batch_id, plan_path))
    return plan


def validate_state(state_path: Path, batch_id: str) -> Dict[str, Any]:
    if not state_path.exists():
        fail("Missing state file: {}".format(state_path))
    state = load_json(state_path)
    baseline = state.get("baseline") or {}
    if baseline.get("status") != "frozen":
        fail("Refusing to watch without a frozen baseline in {}".format(state_path))
    if not baseline.get("path") or not Path(baseline["path"]).exists():
        fail("Frozen baseline summary is missing: {}".format(baseline.get("path")))
    if not baseline.get("manifest") or not Path(baseline["manifest"]).exists():
        fail("Frozen baseline manifest is missing: {}".format(baseline.get("manifest")))
    open_jobs = state.get("open_jobs")
    if not isinstance(open_jobs, list) or not open_jobs:
        fail("State has no open jobs; refusing to watch batch {}".format(batch_id))
    return state


def goal_status_command(state_path: Path, batch_id: str) -> List[str]:
    return [
        sys.executable,
        "scripts/goal_status.py",
        "--state",
        str(state_path),
        "--batch-id",
        batch_id,
        "--update-state",
        "--json",
    ]


def run_status(state_path: Path, batch_id: str, timeout_seconds: float) -> Dict[str, Any]:
    command = goal_status_command(state_path, batch_id)
    try:
        proc = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=max(1.0, float(timeout_seconds)),
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "timeout": True,
            "stdout": exc.stdout or "",
            "stderr": "goal_status.py timed out after {} seconds".format(timeout_seconds),
            "jobs": [],
        }
    if proc.returncode != 0:
        return {"ok": False, "timeout": False, "stdout": proc.stdout, "stderr": proc.stderr, "jobs": []}
    try:
        parsed = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {"ok": False, "timeout": False, "stdout": proc.stdout, "stderr": "goal_status.py emitted invalid JSON", "jobs": []}
    parsed["ok"] = True
    parsed["stdout"] = proc.stdout
    parsed["stderr"] = proc.stderr
    return parsed


def all_terminal(jobs: Sequence[Dict[str, Any]]) -> bool:
    if not jobs:
        return False
    return all(str(job.get("status", "")).lower() in TERMINAL_STATUSES for job in jobs)


def has_nonterminal(jobs: Sequence[Dict[str, Any]]) -> bool:
    return any(str(job.get("status", "")).lower() in NON_TERMINAL_STATUSES for job in jobs)


def next_commands(batch_id: str) -> List[str]:
    collected = "outputs/goal/runs/{}/collected_results.csv".format(batch_id)
    scoreboard_csv = "outputs/goal/runs/{}/scoreboard.csv".format(batch_id)
    scoreboard_json = "outputs/goal/runs/{}/scoreboard.json".format(batch_id)
    return [
        "python scripts/goal_status.py --state outputs/goal/state.json --batch-id {} --update-state".format(batch_id),
        "python scripts/goal_collect.py --batch-id {} --eval-root outputs/goal/eval --output {}".format(batch_id, collected),
        (
            "python scripts/goal_scoreboard.py --baseline outputs/baselines/standard_frozen/results_summary.csv "
            "--results {} --metric ndcg_at_10 --margin 0.001 --output-csv {} --output-json {}"
        ).format(collected, scoreboard_csv, scoreboard_json),
    ]


def resume_prompt(batch_id: str) -> str:
    return """The Slurm jobs for {batch_id} are now terminal.

Resume the autonomous goal workflow:
1. Read AGENTS.md.
2. Read docs/goal_protocol.md.
3. Read outputs/goal/state.json.
4. Run scripts/goal_status.py --state outputs/goal/state.json --update-state.
5. Collect results with scripts/goal_collect.py for {batch_id}.
6. Score results with scripts/goal_scoreboard.py against outputs/baselines/standard_frozen/results_summary.csv using ndcg_at_10 and margin 0.001.
7. Use result_analyst as a read-only subagent if available.
8. Update docs/agent_lab_notebook.md and outputs/goal/state.json.
9. Summarize per-task NDCG@10 deltas, invalid/missing results, and failure modes.
10. Decide whether to fix failures, submit a second dev batch, or promote a candidate to final validation.
11. Do not submit new jobs until you have summarized {batch_id} results and written the next manifest/dry-run plan.
""".format(batch_id=batch_id)


def extract_final_message_from_jsonl(text: str) -> str:
    candidates: List[str] = []
    for line in text.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        for key in ["message", "text", "content", "output", "final_message"]:
            value = event.get(key) if isinstance(event, dict) else None
            if isinstance(value, str) and value.strip():
                candidates.append(value.strip())
        if isinstance(event, dict):
            data = event.get("data")
            if isinstance(data, dict):
                for key in ["message", "text", "content", "output", "final_message"]:
                    value = data.get(key)
                    if isinstance(value, str) and value.strip():
                        candidates.append(value.strip())
    return candidates[-1] if candidates else "Codex JSONL output was written; no final text message was detected."


def launch_codex(args: argparse.Namespace, run_dir: Path, status: Dict[str, Any], log_path: Path) -> Dict[str, Any]:
    sentinel = run_dir / ".codex_resume_launched"
    if sentinel.exists() and not args.force_codex:
        append_log(log_path, "Codex sentinel exists; not launching again: {}".format(sentinel))
        return {"launched": False, "reason": "sentinel_exists", "sentinel": str(sentinel)}
    if not shutil.which("codex"):
        return {"launched": False, "reason": "codex_not_found"}

    markdown_path = Path(args.codex_output) if args.codex_output else run_dir / "codex_resume_after_terminal.md"
    jsonl_path = run_dir / "codex_resume_after_terminal.jsonl"
    prompt = resume_prompt(args.batch_id)
    command = ["codex", "exec", "--sandbox", "workspace-write"]
    if args.json:
        command.append("--json")
    command.append(prompt)

    ensure_dir(markdown_path.parent)
    sentinel.write_text(json.dumps({"launched_at": now_utc(), "batch_id": args.batch_id}) + "\n", encoding="utf-8")
    append_log(log_path, "Launching codex exec for terminal batch {}".format(args.batch_id))
    proc = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if args.json:
        jsonl_path.write_text(proc.stdout, encoding="utf-8")
        final_message = extract_final_message_from_jsonl(proc.stdout)
        markdown_path.write_text(final_message + "\n", encoding="utf-8")
    else:
        markdown_path.write_text(proc.stdout, encoding="utf-8")
    if proc.stderr:
        append_log(log_path, "codex stderr: {}".format(proc.stderr.strip()))
    result = {
        "launched": True,
        "returncode": proc.returncode,
        "sentinel": str(sentinel),
        "markdown": str(markdown_path),
        "jsonl": str(jsonl_path) if args.json else None,
    }
    if proc.returncode != 0:
        result["reason"] = "codex_exec_failed"
    return result


def write_watcher_status(run_dir: Path, data: Dict[str, Any]) -> None:
    atomic_write_json(run_dir / "watcher_status.json", data)


def mark_state_terminal(state_path: Path, batch_id: str, run_dir: Path, jobs: Sequence[Dict[str, Any]]) -> None:
    state = load_json(state_path)
    state["phase"] = "COLLECT_RESULTS"
    state["open_jobs"] = list(jobs)
    state["next_required_action"] = "Collect results for completed jobs or inspect terminal failures."
    state["last_watcher"] = {
        "batch_id": batch_id,
        "path": str(run_dir / "watcher_status.json"),
        "updated_at": now_utc(),
    }
    atomic_write_json(state_path, state)


def main() -> None:
    args = parse_args()
    if os.environ.get("SLURM_JOB_ID") and not args.allow_inside_slurm:
        fail("Refusing to run watcher inside Slurm. Pass --allow-inside-slurm only for an intentional debug case.")
    if args.interval_seconds <= 0:
        fail("--interval-seconds must be positive")
    if args.max_hours <= 0:
        fail("--max-hours must be positive")

    state_path = Path(args.state)
    validate_state(state_path, args.batch_id)
    plan = load_submission_plan(args.batch_id)
    run_dir = ensure_dir(batch_dir(args.batch_id))
    log_path = run_dir / "watcher.log"
    append_log(log_path, "Starting watcher mode={} batch={} jobs={}".format(args.mode, args.batch_id, len(plan.get("jobs", []))))

    deadline = time.time() + float(args.max_hours) * 3600.0
    poll_count = 0
    latest: Dict[str, Any] = {"jobs": []}

    while True:
        poll_count += 1
        latest = run_status(state_path, args.batch_id, args.status_timeout_seconds)
        jobs = latest.get("jobs", [])
        if latest.get("ok"):
            summary = ", ".join("{}:{}={}".format(job.get("run_id"), job.get("type"), job.get("status")) for job in jobs)
            append_log(log_path, "Poll {}: {}".format(poll_count, summary or "no jobs"))
        else:
            append_log(log_path, "Poll {} failed: {}".format(poll_count, latest.get("stderr", "unknown error")))

        if all_terminal(jobs):
            commands = next_commands(args.batch_id)
            status = {
                "batch_id": args.batch_id,
                "finished_at": now_utc(),
                "terminal": True,
                "timed_out": False,
                "poll_count": poll_count,
                "jobs": jobs,
                "next_commands": commands,
                "mode": args.mode,
            }
            if args.mode == "codex":
                status["codex"] = launch_codex(args, run_dir, status, log_path)
            write_watcher_status(run_dir, status)
            mark_state_terminal(state_path, args.batch_id, run_dir, jobs)
            print("Batch {} reached terminal state.".format(args.batch_id))
            for job in jobs:
                print("- {run_id} {type}: {status} job={job_id}".format(**job))
            print("Next commands:")
            for command in commands:
                print(command)
            return

        if latest.get("ok") and not has_nonterminal(jobs):
            append_log(log_path, "Poll {} returned non-terminal-unclassified statuses; continuing.".format(poll_count))

        remaining = deadline - time.time()
        if remaining <= 0:
            status = {
                "batch_id": args.batch_id,
                "finished_at": now_utc(),
                "terminal": False,
                "timed_out": True,
                "poll_count": poll_count,
                "jobs": jobs,
                "mode": args.mode,
                "last_error": None if latest.get("ok") else latest.get("stderr"),
            }
            write_watcher_status(run_dir, status)
            print("Watcher timed out before batch {} reached terminal state.".format(args.batch_id))
            for job in jobs:
                print("- {run_id} {type}: {status} job={job_id}".format(**job))
            raise SystemExit(2)

        time.sleep(min(float(args.interval_seconds), remaining))


if __name__ == "__main__":
    main()
