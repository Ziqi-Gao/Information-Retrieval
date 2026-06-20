#!/usr/bin/env python3
"""Safely dry-run or submit Slurm train/eval jobs from a batch manifest."""

import argparse
import os
import shlex
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from goal_common import (
    atomic_write_json,
    ensure_dir,
    load_json,
    load_yaml,
    now_utc,
    parse_task_list,
    repo_status,
    strict_bool,
    write_yaml,
)
from goal_validate_manifest import validate_manifest


SAFE_RUNTIME_ENV_KEYS = [
    "CONDA_ENV",
    "DEFAULT_CONDA_ENV",
    "PYTHON_BIN",
    "CONDA_SH",
    "HF_HOME",
    "TRANSFORMERS_CACHE",
    "HF_DATASETS_CACHE",
    "MPLCONFIGDIR",
    "TOKENIZERS_PARALLELISM",
    "LD_LIBRARY_PATH",
]

ALLOWED_SBATCH_ARGS = {
    "--account",
    "-A",
    "--partition",
    "-p",
    "--qos",
    "--time",
    "-t",
    "--cpus-per-task",
    "-c",
    "--mem",
    "--gres",
    "--constraint",
    "-C",
    "--reservation",
    "--nodes",
    "-N",
    "--ntasks",
    "-n",
    "--job-name",
    "-J",
}
BANNED_SBATCH_ARGS = {"--export", "--wrap", "--array", "--dependency", "-d"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dry-run or submit a validated goal batch through Slurm wrappers.")
    parser.add_argument("manifest", help="YAML batch manifest.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Write the plan without calling sbatch.")
    mode.add_argument("--submit", action="store_true", help="Actually call sbatch.")
    parser.add_argument("--resume", action="store_true", help="Allow existing per-run output directories.")
    parser.add_argument("--state", default="outputs/goal/state.json")
    return parser.parse_args()


def safe_runtime_exports() -> Dict[str, str]:
    exports: Dict[str, str] = {}
    for key in SAFE_RUNTIME_ENV_KEYS:
        value = os.environ.get(key)
        if value:
            exports[key] = value
    return exports


def format_export(base: Dict[str, Any]) -> str:
    parts = ["NONE"]
    for key in sorted(base):
        value = base[key]
        if value is None:
            continue
        text = str(value)
        if "," in text:
            raise ValueError("Slurm --export value for {} contains a comma; use semicolon-separated task lists".format(key))
        parts.append("{}={}".format(key, text))
    return ",".join(parts)


def sbatch_args_from_env() -> List[str]:
    raw = os.environ.get("SBATCH_ARGS", "")
    if not raw.strip():
        return []
    tokens = shlex.split(raw)
    validated: List[str] = []
    idx = 0
    while idx < len(tokens):
        token = tokens[idx]
        option = token.split("=", 1)[0]
        if option in BANNED_SBATCH_ARGS:
            raise SystemExit("SBATCH_ARGS may not include {}".format(option))
        if option not in ALLOWED_SBATCH_ARGS:
            raise SystemExit("SBATCH_ARGS contains unsupported option {}".format(option))
        validated.append(token)
        if "=" not in token and idx + 1 < len(tokens) and not tokens[idx + 1].startswith("-"):
            idx += 1
            validated.append(tokens[idx])
        idx += 1
    return validated


def run_sbatch(command: Sequence[str], dry_run: bool) -> Optional[str]:
    if dry_run:
        return None
    import subprocess

    proc = subprocess.Popen(
        list(command),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    stdout, stderr = proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError("sbatch failed:\ncommand={}\nstdout={}\nstderr={}".format(" ".join(command), stdout, stderr))
    first = stdout.strip().splitlines()[0] if stdout.strip() else ""
    return first.split(";")[0] if first else None


def task_names_for_experiment(manifest: Dict[str, Any], experiment: Dict[str, Any]) -> List[str]:
    exp_tasks = parse_task_list((experiment.get("eval") or {}).get("task_names"))
    if exp_tasks:
        return exp_tasks
    tasks = manifest.get("tasks") or {}
    if manifest.get("purpose") == "final":
        return parse_task_list(tasks.get("final"))
    return parse_task_list(tasks.get("dev")) or ["SciFact"]


def bool_text(value: Any) -> str:
    return "true" if bool(value) else "false"


def require_bool(value: Any, field_name: str, default: bool = False) -> bool:
    parsed = strict_bool(value, default=default)
    if parsed is None:
        raise SystemExit("{} must be a YAML boolean true/false".format(field_name))
    return parsed


def assert_output_dirs_available(plan_jobs: List[Dict[str, Any]], resume: bool) -> None:
    if resume:
        return
    collisions: List[str] = []
    for job in plan_jobs:
        for key in ["train_output_dir", "eval_output_dir"]:
            if job.get("eval_only") and key == "train_output_dir":
                continue
            path = Path(job[key])
            if path.exists() and any(path.iterdir()):
                collisions.append(str(path))
    if collisions:
        raise SystemExit("Refusing to reuse non-empty output directories without --resume: {}".format(", ".join(collisions)))


def build_plan(manifest_path: Path, manifest: Dict[str, Any]) -> Dict[str, Any]:
    batch_id = manifest["batch_id"]
    defaults = manifest.get("defaults") or {}
    purpose = manifest.get("purpose")
    run_root = Path(defaults.get("output_base", "outputs/goal/runs")) / batch_id
    eval_root = Path(defaults.get("eval_output_base", "outputs/goal/eval")) / batch_id
    scheduler_args = sbatch_args_from_env()
    runtime_exports = safe_runtime_exports()
    jobs: List[Dict[str, Any]] = []

    for experiment in manifest["experiments"]:
        run_id = experiment["run_id"]
        version = experiment.get("version")
        if not version:
            raise SystemExit("Experiment {} is missing version".format(run_id))
        config = experiment.get("config") or defaults.get("config")
        train_output_dir = run_root / run_id
        eval_output_dir = eval_root / run_id
        eval_tasks = task_names_for_experiment(manifest, experiment)
        train_settings = experiment.get("train") or {}
        eval_settings = experiment.get("eval") or {}
        eval_only = require_bool(experiment.get("eval_only"), "experiment {} eval_only".format(run_id), default=False)
        if any(key in eval_settings for key in ["checkpoint_dir", "fusion_standard_checkpoint_dir", "fusion_alpha"]) and not eval_only:
            raise SystemExit("Experiment {} uses eval checkpoint/fusion fields and must set eval_only: true".format(run_id))
        if eval_only:
            checkpoint_dir_value = eval_settings.get("checkpoint_dir")
            if not checkpoint_dir_value:
                raise SystemExit("Experiment {} has eval_only=true but eval.checkpoint_dir is missing".format(run_id))
            checkpoint_dir = Path(checkpoint_dir_value)
        else:
            checkpoint_dir = train_output_dir / "final"

        train_exports: Dict[str, Any] = dict(runtime_exports)
        train_exports.update(
            {
                "VERSION": version,
                "CONFIG": config,
                "OUTPUT_BASE": str(run_root),
                "RUN_NAME": run_id,
            }
        )
        if experiment.get("loop_memory_mode") is not None:
            train_exports["LOOP_MEMORY_MODE"] = experiment.get("loop_memory_mode")
        if experiment.get("loop_query_mode") is not None:
            train_exports["LOOP_QUERY_MODE"] = experiment.get("loop_query_mode")
        for manifest_key, export_key in [("epochs", "EPOCHS"), ("max_steps", "MAX_STEPS"), ("save_steps", "SAVE_STEPS")]:
            if train_settings.get(manifest_key) is not None:
                train_exports[export_key] = train_settings.get(manifest_key)

        eval_exports: Dict[str, Any] = dict(runtime_exports)
        eval_exports.update(
            {
                "VERSION": version,
                "CHECKPOINT_DIR": str(checkpoint_dir),
                "OUTPUT_DIR": str(eval_output_dir),
                "EVAL_ALL_LOOPS": bool_text(
                    require_bool(
                        eval_settings.get("eval_all_loops", defaults.get("eval_all_loops", False)),
                        "experiment {} eval.eval_all_loops".format(run_id),
                        default=False,
                    )
                ),
                "TASK_NAMES": ";".join(eval_tasks),
            }
        )

        train_command = None
        if not eval_only:
            train_command = ["sbatch", "--parsable"] + scheduler_args + ["--export={}".format(format_export(train_exports)), "scripts/slurm_train.sbatch"]
        eval_command_base = ["sbatch", "--parsable"] + scheduler_args
        if eval_settings.get("loop_idx") is not None:
            eval_exports["LOOP_IDX"] = eval_settings.get("loop_idx")
        if eval_settings.get("fusion_standard_checkpoint_dir") is not None:
            eval_exports["FUSION_STANDARD_CHECKPOINT_DIR"] = eval_settings.get("fusion_standard_checkpoint_dir")
        if eval_settings.get("fusion_alpha") is not None:
            eval_exports["FUSION_ALPHA"] = eval_settings.get("fusion_alpha")
        jobs.append(
            {
                "batch_id": batch_id,
                "run_id": run_id,
                "purpose": purpose,
                "eval_only": eval_only,
                "version": version,
                "config": config,
                "tasks": eval_tasks,
                "train_output_dir": str(train_output_dir),
                "eval_output_dir": str(eval_output_dir),
                "checkpoint_dir": str(checkpoint_dir),
                "train_exports": sorted(train_exports) if not eval_only else [],
                "eval_exports": sorted(eval_exports),
                "train_command": train_command,
                "eval_command_base": eval_command_base,
                "eval_export": format_export(eval_exports),
                "train_job_id": None,
                "eval_job_id": None,
            }
        )

    return {
        "created_at": now_utc(),
        "manifest": str(manifest_path),
        "batch_id": batch_id,
        "purpose": purpose,
        "repo": repo_status(),
        "jobs": jobs,
    }


def update_state(state_path: Path, plan: Dict[str, Any], dry_run: bool) -> None:
    state: Dict[str, Any] = {}
    if state_path.exists():
        state = load_json(state_path)
    state["phase"] = "DRY_RUN_SUBMIT" if dry_run else "SUBMIT_BATCH"
    state["repo"] = repo_status()
    state["current_batch"] = {
        "batch_id": plan["batch_id"],
        "manifest": plan["manifest"],
        "plan": "outputs/goal/runs/{}/submission_plan.json".format(plan["batch_id"]),
        "dry_run": dry_run,
    }
    open_jobs = []
    for job in plan["jobs"]:
        if not job.get("eval_only"):
            open_jobs.append(
                {
                    "batch_id": plan["batch_id"],
                    "run_id": job["run_id"],
                    "type": "train",
                    "job_id": job.get("train_job_id"),
                    "status": "dry_run" if dry_run else "submitted",
                }
            )
        open_jobs.append(
            {
                "batch_id": plan["batch_id"],
                "run_id": job["run_id"],
                "type": "eval",
                "job_id": job.get("eval_job_id"),
                "depends_on": job.get("train_job_id"),
                "status": "dry_run" if dry_run else "submitted",
            }
        )
    state["open_jobs"] = open_jobs
    if dry_run:
        baseline = state.get("baseline") or {}
        if baseline.get("status") != "frozen":
            state["next_required_action"] = "Freeze or validate baseline before real autonomous experiments."
        else:
            state["next_required_action"] = "Review dry-run plan, then submit only if the manifest allows it."
    else:
        state["next_required_action"] = "Wait for jobs, then collect results."
    atomic_write_json(state_path, state)


def main() -> None:
    args = parse_args()
    dry_run = not args.submit
    manifest_path = Path(args.manifest)
    validation = validate_manifest(manifest_path)
    if not validation["valid"]:
        for error in validation["errors"]:
            print("- {}".format(error))
        raise SystemExit("Manifest validation failed")
    manifest = load_yaml(manifest_path)
    budget = manifest.get("budget") or {}
    allow_submit = require_bool(budget.get("allow_submit"), "budget.allow_submit", default=False)
    if args.submit and not allow_submit:
        raise SystemExit("Refusing --submit because budget.allow_submit is false.")
    if args.submit and not validation["baseline_present"] and manifest.get("purpose") != "smoke":
        raise SystemExit("Refusing --submit without a frozen baseline.")

    plan = build_plan(manifest_path, manifest)
    assert_output_dirs_available(plan["jobs"], args.resume)

    batch_dir = ensure_dir(Path("outputs/goal/runs") / manifest["batch_id"])
    submitted_manifest = batch_dir / "batch_manifest.submitted.yaml"
    plan_path = batch_dir / "submission_plan.json"
    write_yaml(submitted_manifest, manifest)

    for job in plan["jobs"]:
        train_job_id = None
        if not job.get("eval_only"):
            train_job_id = run_sbatch(job["train_command"], dry_run=dry_run)
        job["train_job_id"] = train_job_id
        dependency_args: List[str] = []
        if train_job_id:
            dependency_args = ["--dependency=afterok:{}".format(train_job_id)]
        eval_command = list(job["eval_command_base"]) + dependency_args + [
            "--export={}".format(job["eval_export"]),
            "scripts/slurm_eval.sbatch",
        ]
        job["eval_command"] = eval_command
        eval_job_id = run_sbatch(eval_command, dry_run=dry_run)
        job["eval_job_id"] = eval_job_id

    plan["dry_run"] = dry_run
    atomic_write_json(plan_path, plan)
    update_state(Path(args.state), plan, dry_run=dry_run)

    mode = "DRY RUN" if dry_run else "SUBMITTED"
    print("{} batch {} with {} experiment(s).".format(mode, manifest["batch_id"], len(plan["jobs"])))
    print("Plan: {}".format(plan_path))
    for job in plan["jobs"]:
        mode = "eval_only" if job.get("eval_only") else "train_eval"
        print("- {} mode={} train_job={} eval_job={}".format(job["run_id"], mode, job["train_job_id"], job["eval_job_id"]))


if __name__ == "__main__":
    main()
