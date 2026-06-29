#!/usr/bin/env python3
"""Dry-run or submit an isolated ReasonIR-BRIGHT Slurm batch."""

import argparse
from pathlib import Path
from typing import Any, Dict, List

from goal_common import atomic_write_json, ensure_dir, load_yaml, now_utc, parse_task_list, repo_status, strict_bool, write_yaml
from goal_submit_batch import (
    format_export,
    run_sbatch,
    safe_postprocess_runtime_exports,
    safe_runtime_exports,
    sbatch_args_from_env,
    shell_join,
)


EXPECTED_RUNS = [
    "standard",
    "loop_final_mean_pool",
    "loop_matryoshka_mean_pool",
    "loop_final_recurrent_mean_pool",
    "loop_matryoshka_recurrent_mean_pool",
    "loop_final_recurrent_no_memory",
    "loop_matryoshka_recurrent_no_memory",
]
FORBIDDEN_EVAL_KEYS = {
    "lexical_hash_dim",
    "lexical_weight",
    "fusion_standard_checkpoint_dir",
    "fusion_alpha",
    "fusion_scope",
    "self_query_alpha",
    "self_query_source_loop",
    "doc_chunk_words",
    "doc_chunk_stride",
    "doc_chunk_max_chunks",
    "bm25",
    "use_reasoning",
    "use_gold_answer",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submit a ReasonIR-BRIGHT batch.")
    parser.add_argument("manifest")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--submit", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Allow existing non-empty output directories.")
    parser.add_argument("--submit-postprocess", action="store_true")
    return parser.parse_args()


def bool_text(value: Any) -> str:
    return "true" if bool(value) else "false"


def require_bool(value: Any, field_name: str, default: bool = False) -> bool:
    parsed = strict_bool(value, default=default)
    if parsed is None:
        raise SystemExit(f"{field_name} must be a YAML boolean true/false")
    return parsed


def validate_manifest(path: Path, manifest: Dict[str, Any]) -> None:
    batch_id = manifest.get("batch_id")
    if not isinstance(batch_id, str) or not batch_id.startswith("reasonir_bright_"):
        raise SystemExit("batch_id must start with reasonir_bright_")
    if manifest.get("purpose") != "dev":
        raise SystemExit("ReasonIR-BRIGHT batch_001 is expected to be purpose: dev")
    if manifest.get("baseline"):
        raise SystemExit("ReasonIR-BRIGHT manifests must not declare the old frozen MTEB baseline.")
    defaults = manifest.get("defaults") or {}
    output_base = defaults.get("output_base", "outputs/reasonir_bright/runs")
    eval_output_base = defaults.get("eval_output_base", "outputs/reasonir_bright/eval")
    if not str(output_base).startswith("outputs/reasonir_bright/runs"):
        raise SystemExit("defaults.output_base must stay under outputs/reasonir_bright/runs")
    if not str(eval_output_base).startswith("outputs/reasonir_bright/eval"):
        raise SystemExit("defaults.eval_output_base must stay under outputs/reasonir_bright/eval")

    config_path = Path(defaults.get("config", ""))
    if not config_path.exists():
        raise SystemExit(f"Missing training config: {config_path}")
    config = load_yaml(config_path)
    if config.get("dataset_name") != "reasonir/reasonir-data":
        raise SystemExit("training config dataset_name must be reasonir/reasonir-data")
    if config.get("dataset_config") != "hq":
        raise SystemExit("training config dataset_config must be hq")
    if int(config.get("num_negatives", 0)) != 1:
        raise SystemExit("training config num_negatives must be 1 for ReasonIR HQ")

    domains = parse_task_list(((manifest.get("domains") or {}).get("dev")))
    if domains != ["biology", "economics", "psychology", "stackoverflow"]:
        raise SystemExit("domains.dev must be biology, economics, psychology, stackoverflow in that order")

    experiments = manifest.get("experiments") or []
    run_ids = [experiment.get("run_id") for experiment in experiments]
    if run_ids != EXPECTED_RUNS:
        raise SystemExit(f"experiments must contain the original seven runs in order: {EXPECTED_RUNS}")
    for experiment in experiments:
        if experiment.get("version") != experiment.get("run_id"):
            raise SystemExit(f"{experiment.get('run_id')} must use the matching version name for this track.")
        eval_config = experiment.get("eval") or {}
        forbidden = sorted(FORBIDDEN_EVAL_KEYS.intersection(eval_config))
        if forbidden:
            raise SystemExit(f"{experiment.get('run_id')} eval contains forbidden fields: {forbidden}")
        if require_bool(eval_config.get("use_long_documents"), f"{experiment.get('run_id')} eval.use_long_documents", default=False):
            raise SystemExit("First ReasonIR-BRIGHT dev batch must use short BRIGHT documents.")
        if experiment.get("run_id") != "standard":
            declared = eval_config.get("candidate_loop_indices")
            if declared != [10]:
                raise SystemExit(f"{experiment.get('run_id')} must predeclare candidate_loop_indices: [10]")


def assert_output_dirs_available(plan: Dict[str, Any], resume: bool, dry_run: bool) -> None:
    if resume or dry_run:
        return
    collisions = []
    for job in plan["jobs"]:
        for key in ("train_output_dir", "eval_output_dir"):
            path = Path(job[key])
            if path.exists() and any(path.iterdir()):
                collisions.append(str(path))
    if collisions:
        raise SystemExit("Refusing to reuse non-empty output directories without --resume: {}".format(", ".join(collisions)))


def build_plan(manifest_path: Path, manifest: Dict[str, Any]) -> Dict[str, Any]:
    batch_id = manifest["batch_id"]
    defaults = manifest.get("defaults") or {}
    run_root = Path(defaults.get("output_base", "outputs/reasonir_bright/runs")) / batch_id
    eval_root = Path(defaults.get("eval_output_base", "outputs/reasonir_bright/eval")) / batch_id
    config = defaults.get("config", "configs/reasonir_bright_dev.yaml")
    domains = parse_task_list(((manifest.get("domains") or {}).get("dev")))
    scheduler_args = sbatch_args_from_env()
    runtime_exports = safe_runtime_exports()
    jobs: List[Dict[str, Any]] = []
    for experiment in manifest["experiments"]:
        run_id = experiment["run_id"]
        version = experiment["version"]
        train_output_dir = run_root / run_id
        eval_output_dir = eval_root / run_id
        checkpoint_dir = train_output_dir / "final"
        train_exports = dict(runtime_exports)
        train_exports.update(
            {
                "REASONIR_BRIGHT_SUBMIT_BATCH": "1",
                "VERSION": version,
                "CONFIG": config,
                "OUTPUT_BASE": str(run_root),
                "RUN_NAME": run_id,
            }
        )
        train_settings = experiment.get("train") or {}
        if train_settings.get("max_steps") is not None:
            train_exports["MAX_STEPS"] = train_settings["max_steps"]
        if train_settings.get("save_steps") is not None:
            train_exports["SAVE_STEPS"] = train_settings["save_steps"]

        eval_config = experiment.get("eval") or {}
        eval_exports = dict(runtime_exports)
        eval_exports.update(
            {
                "REASONIR_BRIGHT_SUBMIT_BATCH": "1",
                "VERSION": version,
                "CHECKPOINT_DIR": str(checkpoint_dir),
                "OUTPUT_DIR": str(eval_output_dir),
                "DOMAINS": ";".join(domains),
                "USE_LONG_DOCUMENTS": bool_text(
                    require_bool(eval_config.get("use_long_documents"), f"{run_id} eval.use_long_documents", default=False)
                ),
                "EVAL_ALL_LOOPS": bool_text(
                    require_bool(eval_config.get("eval_all_loops"), f"{run_id} eval.eval_all_loops", default=False)
                ),
                "QUERY_BATCH_SIZE": eval_config.get("query_batch_size", defaults.get("query_batch_size", 32)),
                "CORPUS_BATCH_SIZE": eval_config.get("corpus_batch_size", defaults.get("corpus_batch_size", 64)),
                "SCORE_CHUNK_SIZE": eval_config.get("score_chunk_size", defaults.get("score_chunk_size", 8192)),
            }
        )
        if eval_config.get("loop_idx") is not None:
            eval_exports["LOOP_IDX"] = eval_config["loop_idx"]

        jobs.append(
            {
                "batch_id": batch_id,
                "run_id": run_id,
                "version": version,
                "train_output_dir": str(train_output_dir),
                "eval_output_dir": str(eval_output_dir),
                "checkpoint_dir": str(checkpoint_dir),
                "train_command": ["sbatch", "--parsable"] + scheduler_args + [
                    f"--export={format_export(train_exports)}",
                    "scripts/slurm_reasonir_bright_train.sbatch",
                ],
                "eval_command_base": ["sbatch", "--parsable"] + scheduler_args,
                "eval_export": format_export(eval_exports),
                "train_job_id": None,
                "eval_job_id": None,
            }
        )
    return {
        "created_at": now_utc(),
        "manifest": str(manifest_path),
        "batch_id": batch_id,
        "purpose": manifest.get("purpose"),
        "repo": repo_status(),
        "jobs": jobs,
    }


def build_postprocess(plan: Dict[str, Any], manifest: Dict[str, Any]) -> Dict[str, Any]:
    batch_id = plan["batch_id"]
    defaults = manifest.get("defaults") or {}
    run_root = Path(defaults.get("output_base", "outputs/reasonir_bright/runs"))
    eval_root = Path(defaults.get("eval_output_base", "outputs/reasonir_bright/eval"))
    output_dir = run_root / batch_id
    eval_job_ids = [str(job.get("eval_job_id") or f"<eval_job_id:{job['run_id']}>") for job in plan["jobs"]]
    dependency = "afterany:{}".format(":".join(eval_job_ids))
    exports = safe_postprocess_runtime_exports()
    exports.update(
        {
            "REASONIR_BRIGHT_SUBMIT_BATCH": "1",
            "BATCH_ID": batch_id,
            "RUN_ROOT": str(run_root),
            "EVAL_ROOT": str(eval_root),
            "OUTPUT_DIR": str(output_dir),
            "MANIFEST": str(output_dir / "batch_manifest.submitted.yaml"),
            "METRIC": manifest.get("primary_metric", "ndcg_at_10"),
        }
    )
    scheduler_args = sbatch_args_from_env(
        excluded_options={"--time", "-t", "--cpus-per-task", "-c", "--mem", "--gres", "--job-name", "-J"},
        env_key="POSTPROCESS_SBATCH_ARGS",
    )
    if not scheduler_args:
        scheduler_args = sbatch_args_from_env(
            excluded_options={"--time", "-t", "--cpus-per-task", "-c", "--mem", "--gres", "--job-name", "-J"}
        )
    command = ["sbatch", "--parsable"] + scheduler_args + [
        f"--dependency={dependency}",
        f"--export={format_export(exports)}",
        "scripts/slurm_reasonir_bright_postprocess.sbatch",
    ]
    return {"enabled": True, "job_id": None, "dependency": dependency, "command": command, "exports": sorted(exports)}


def main() -> None:
    args = parse_args()
    dry_run = not args.submit
    manifest_path = Path(args.manifest)
    manifest = load_yaml(manifest_path)
    validate_manifest(manifest_path, manifest)
    allow_submit = require_bool((manifest.get("budget") or {}).get("allow_submit"), "budget.allow_submit", default=False)
    if args.submit and not allow_submit:
        raise SystemExit("Refusing --submit because budget.allow_submit is false.")

    plan = build_plan(manifest_path, manifest)
    assert_output_dirs_available(plan, resume=args.resume, dry_run=dry_run)
    batch_dir = ensure_dir(Path((manifest.get("defaults") or {}).get("output_base", "outputs/reasonir_bright/runs")) / manifest["batch_id"])
    manifest_copy = batch_dir / ("batch_manifest.dry_run.yaml" if dry_run else "batch_manifest.submitted.yaml")
    plan_path = batch_dir / ("dry_run_plan.json" if dry_run else "submission_plan.json")
    write_yaml(manifest_copy, manifest)

    for job in plan["jobs"]:
        train_job_id = run_sbatch(job["train_command"], dry_run=dry_run)
        job["train_job_id"] = train_job_id
        dependency_args = [f"--dependency=afterok:{train_job_id}"] if train_job_id else []
        eval_command = list(job["eval_command_base"]) + dependency_args + [
            f"--export={job['eval_export']}",
            "scripts/slurm_reasonir_bright_eval.sbatch",
        ]
        job["eval_command"] = eval_command
        job["eval_job_id"] = run_sbatch(eval_command, dry_run=dry_run)

    if args.submit_postprocess:
        postprocess = build_postprocess(plan, manifest)
        postprocess["job_id"] = run_sbatch(postprocess["command"], dry_run=dry_run)
        plan["postprocess"] = postprocess
        plan["postprocess_job_id"] = postprocess["job_id"]
    else:
        plan["postprocess"] = {"enabled": False, "job_id": None}
        plan["postprocess_job_id"] = None
    plan["dry_run"] = dry_run
    atomic_write_json(plan_path, plan)

    mode = "DRY RUN" if dry_run else "SUBMITTED"
    print(f"{mode} ReasonIR-BRIGHT batch {manifest['batch_id']} with {len(plan['jobs'])} experiment(s).")
    print(f"Plan: {plan_path}")
    for job in plan["jobs"]:
        print(f"- {job['run_id']} train_job={job['train_job_id']} eval_job={job['eval_job_id']}")
    if plan["postprocess"].get("enabled"):
        print(f"- postprocess job={plan['postprocess'].get('job_id')}")
        if dry_run:
            print("Postprocess sbatch: {}".format(shell_join(plan["postprocess"]["command"])))


if __name__ == "__main__":
    main()
