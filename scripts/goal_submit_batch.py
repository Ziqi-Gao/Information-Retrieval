#!/usr/bin/env python3
"""Safely dry-run or submit Slurm train/eval/postprocess jobs from a batch manifest."""

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
POSTPROCESS_RUNTIME_ENV_KEYS = [
    "CONDA_ENV",
    "DEFAULT_CONDA_ENV",
    "PYTHON_BIN",
    "CONDA_SH",
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
POSTPROCESS_EXCLUDED_SBATCH_ARGS = {
    "--time",
    "-t",
    "--cpus-per-task",
    "-c",
    "--mem",
    "--gres",
    "--nodes",
    "-N",
    "--ntasks",
    "-n",
    "--job-name",
    "-J",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dry-run or submit a validated goal batch through Slurm wrappers.")
    parser.add_argument("manifest", help="YAML batch manifest.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Write the plan without calling sbatch.")
    mode.add_argument("--submit", action="store_true", help="Actually call sbatch.")
    parser.add_argument("--resume", action="store_true", help="Allow existing per-run output directories.")
    parser.add_argument("--state", default="outputs/goal/state.json")
    parser.add_argument(
        "--submit-postprocess",
        action="store_true",
        help="Add one Slurm dependency job that collects and scores after all eval jobs reach terminal state.",
    )
    parser.add_argument(
        "--submit-postprocess-only",
        action="store_true",
        help="Submit only the Slurm dependency postprocess job for already-submitted eval jobs.",
    )
    parser.add_argument(
        "--eval-job-id",
        action="append",
        default=[],
        help="Existing eval job mapping for --submit-postprocess-only, formatted run_id=job_id.",
    )
    return parser.parse_args()


def safe_runtime_exports() -> Dict[str, str]:
    exports: Dict[str, str] = {}
    for key in SAFE_RUNTIME_ENV_KEYS:
        value = os.environ.get(key)
        if value:
            if key == "PYTHON_BIN" and value in {"python", "python3"}:
                continue
            exports[key] = value
    return exports


def safe_postprocess_runtime_exports() -> Dict[str, str]:
    exports: Dict[str, str] = {}
    for key in POSTPROCESS_RUNTIME_ENV_KEYS:
        value = os.environ.get(key)
        if value:
            if key == "PYTHON_BIN" and value in {"python", "python3"}:
                continue
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


def sbatch_args_from_env(excluded_options: Optional[Sequence[str]] = None, env_key: str = "SBATCH_ARGS") -> List[str]:
    raw = os.environ.get(env_key, "")
    if not raw.strip():
        return []
    tokens = shlex.split(raw)
    excluded = set(excluded_options or [])
    validated: List[str] = []
    idx = 0
    while idx < len(tokens):
        token = tokens[idx]
        option = token.split("=", 1)[0]
        if option in BANNED_SBATCH_ARGS:
            raise SystemExit("SBATCH_ARGS may not include {}".format(option))
        if option not in ALLOWED_SBATCH_ARGS:
            raise SystemExit("SBATCH_ARGS contains unsupported option {}".format(option))
        skip_option = option in excluded
        if not skip_option:
            validated.append(token)
        if "=" not in token and idx + 1 < len(tokens) and not tokens[idx + 1].startswith("-"):
            idx += 1
            if not skip_option:
                validated.append(tokens[idx])
        idx += 1
    return validated


def shell_join(command: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in command)


def parse_eval_job_ids(values: Sequence[str]) -> Dict[str, str]:
    parsed: Dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit("--eval-job-id must be formatted run_id=job_id")
        run_id, job_id = value.split("=", 1)
        run_id = run_id.strip()
        job_id = job_id.strip()
        if not run_id or not job_id:
            raise SystemExit("--eval-job-id must include non-empty run_id and job_id")
        parsed[run_id] = job_id
    return parsed


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


def assert_output_dirs_available(plan_jobs: List[Dict[str, Any]], resume: bool, dry_run: bool) -> None:
    if resume or dry_run:
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
        if any(key in eval_settings for key in ["checkpoint_dir", "fusion_standard_checkpoint_dir", "fusion_alpha", "fusion_scope"]) and not eval_only:
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
                "GOAL_SUBMIT_BATCH": "1",
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
                "GOAL_SUBMIT_BATCH": "1",
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
        if eval_settings.get("fusion_scope") is not None:
            eval_exports["FUSION_SCOPE"] = eval_settings.get("fusion_scope")
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


def build_postprocess_exports(args: argparse.Namespace, manifest: Dict[str, Any], batch_id: str) -> Dict[str, Any]:
    defaults = manifest.get("defaults") or {}
    baseline = manifest.get("baseline") or {}
    output_dir = Path("outputs/goal/runs") / batch_id
    auto_codex = os.environ.get("AUTO_CODEX", "false").strip().lower()
    if auto_codex not in {"true", "false"}:
        raise SystemExit("AUTO_CODEX must be true or false when submitting postprocess jobs.")
    exports: Dict[str, Any] = safe_postprocess_runtime_exports()
    exports.update(
        {
            "GOAL_SUBMIT_BATCH": "1",
            "BATCH_ID": batch_id,
            "STATE_PATH": args.state,
            "EVAL_ROOT": defaults.get("eval_output_base", "outputs/goal/eval"),
            "BASELINE_CSV": baseline.get("summary_csv", "outputs/baselines/standard_frozen/results_summary.csv"),
            "METRIC": manifest.get("primary_metric", "ndcg_at_10"),
            "MARGIN": manifest.get("win_margin", 0.001),
            "OUTPUT_DIR": str(output_dir),
            "AUTO_CODEX": auto_codex,
        }
    )
    return exports


def build_postprocess_command(plan: Dict[str, Any], exports: Dict[str, Any]) -> Dict[str, Any]:
    eval_job_ids = []
    for job in plan["jobs"]:
        eval_job_id = job.get("eval_job_id") or "<eval_job_id:{}>".format(job["run_id"])
        eval_job_ids.append(str(eval_job_id))
    if not eval_job_ids:
        raise SystemExit("Cannot build postprocess dependency without eval jobs.")
    dependency = "afterany:{}".format(":".join(eval_job_ids))
    postprocess_env_key = "POSTPROCESS_SBATCH_ARGS" if os.environ.get("POSTPROCESS_SBATCH_ARGS", "").strip() else "SBATCH_ARGS"
    scheduler_args = sbatch_args_from_env(excluded_options=POSTPROCESS_EXCLUDED_SBATCH_ARGS, env_key=postprocess_env_key)
    command = ["sbatch", "--parsable"] + scheduler_args + [
        "--dependency={}".format(dependency),
        "--export={}".format(format_export(exports)),
        "scripts/slurm_postprocess.sbatch",
    ]
    return {
        "enabled": True,
        "job_id": None,
        "dependency": dependency,
        "command": command,
        "exports": sorted(exports),
        "output_dir": str(exports["OUTPUT_DIR"]),
    }


def update_state(state_path: Path, plan: Dict[str, Any], dry_run: bool, plan_path: Path) -> None:
    state: Dict[str, Any] = {}
    if state_path.exists():
        state = load_json(state_path)
    state["phase"] = "DRY_RUN_SUBMIT" if dry_run else "SUBMIT_BATCH"
    state["repo"] = repo_status()
    state["current_batch"] = {
        "batch_id": plan["batch_id"],
        "manifest": plan["manifest"],
        "plan": str(plan_path),
        "dry_run": dry_run,
        "postprocess": bool((plan.get("postprocess") or {}).get("enabled")),
        "postprocess_job_id": plan.get("postprocess_job_id"),
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
    postprocess = plan.get("postprocess") or {}
    if postprocess.get("enabled"):
        open_jobs.append(
            {
                "batch_id": plan["batch_id"],
                "run_id": plan["batch_id"],
                "type": "postprocess",
                "job_id": postprocess.get("job_id"),
                "depends_on": postprocess.get("dependency"),
                "status": "dry_run" if dry_run else "submitted",
            }
        )
        state["postprocess_job_id"] = postprocess.get("job_id")
    else:
        state.pop("postprocess_job_id", None)
    state["open_jobs"] = open_jobs
    if dry_run:
        baseline = state.get("baseline") or {}
        if baseline.get("status") != "frozen":
            state["next_required_action"] = "Freeze or validate baseline before real autonomous experiments."
        else:
            state["next_required_action"] = "Review dry-run plan, then submit only if the manifest allows it."
    else:
        if postprocess.get("enabled"):
            state["next_required_action"] = "Wait for eval jobs and the Slurm postprocess job, then inspect the scoreboard."
        else:
            state["next_required_action"] = "Wait for jobs, then collect results."
    atomic_write_json(state_path, state)


def apply_existing_eval_job_ids(plan: Dict[str, Any], eval_job_ids: Dict[str, str]) -> None:
    missing = []
    for job in plan["jobs"]:
        run_id = job["run_id"]
        job_id = eval_job_ids.get(run_id) or job.get("eval_job_id")
        if not job_id:
            missing.append(run_id)
            continue
        job["train_job_id"] = job.get("train_job_id")
        job["eval_job_id"] = job_id
        job["eval_command"] = job.get("eval_command") or list(job["eval_command_base"]) + [
            "--export={}".format(job["eval_export"]),
            "scripts/slurm_eval.sbatch",
        ]
    if missing:
        raise SystemExit("Missing existing eval job IDs for postprocess-only submit: {}".format(", ".join(missing)))


def main() -> None:
    args = parse_args()
    dry_run = not args.submit
    if args.submit_postprocess_only and not args.submit_postprocess:
        args.submit_postprocess = True
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
    assert_output_dirs_available(plan["jobs"], args.resume or args.submit_postprocess_only, dry_run=dry_run)

    batch_dir = ensure_dir(Path("outputs/goal/runs") / manifest["batch_id"])
    manifest_copy = batch_dir / ("batch_manifest.dry_run.yaml" if dry_run else "batch_manifest.submitted.yaml")
    plan_path = batch_dir / ("dry_run_plan.json" if dry_run else "submission_plan.json")
    write_yaml(manifest_copy, manifest)

    if args.submit_postprocess_only:
        existing_eval_job_ids = parse_eval_job_ids(args.eval_job_id)
        if not existing_eval_job_ids:
            submitted_plan = batch_dir / "submission_plan.json"
            if submitted_plan.exists():
                prior_plan = load_json(submitted_plan)
                existing_eval_job_ids = {
                    job["run_id"]: str(job["eval_job_id"])
                    for job in prior_plan.get("jobs", [])
                    if job.get("run_id") and job.get("eval_job_id")
                }
        apply_existing_eval_job_ids(plan, existing_eval_job_ids)
    else:
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

    if args.submit_postprocess:
        if not dry_run:
            missing_eval_ids = [job["run_id"] for job in plan["jobs"] if not job.get("eval_job_id")]
            if missing_eval_ids:
                raise RuntimeError("Cannot submit postprocess job without eval job ids: {}".format(", ".join(missing_eval_ids)))
        postprocess_exports = build_postprocess_exports(args, manifest, manifest["batch_id"])
        postprocess = build_postprocess_command(plan, postprocess_exports)
        postprocess_job_id = run_sbatch(postprocess["command"], dry_run=dry_run)
        postprocess["job_id"] = postprocess_job_id
        plan["postprocess"] = postprocess
        plan["postprocess_job_id"] = postprocess_job_id
    else:
        plan["postprocess"] = {"enabled": False, "job_id": None}
        plan["postprocess_job_id"] = None

    plan["dry_run"] = dry_run
    atomic_write_json(plan_path, plan)
    update_state(Path(args.state), plan, dry_run=dry_run, plan_path=plan_path)

    mode = "DRY RUN" if dry_run else "SUBMITTED"
    print("{} batch {} with {} experiment(s).".format(mode, manifest["batch_id"], len(plan["jobs"])))
    print("Plan: {}".format(plan_path))
    for job in plan["jobs"]:
        mode = "eval_only" if job.get("eval_only") else "train_eval"
        print("- {} mode={} train_job={} eval_job={}".format(job["run_id"], mode, job["train_job_id"], job["eval_job_id"]))
    postprocess = plan.get("postprocess") or {}
    if postprocess.get("enabled"):
        print("- postprocess job={}".format(postprocess.get("job_id")))
        if dry_run:
            print("Postprocess sbatch: {}".format(shell_join(postprocess["command"])))


if __name__ == "__main__":
    main()
