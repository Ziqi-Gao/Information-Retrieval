#!/usr/bin/env python3
"""Validate autonomous experiment batch manifests before submission."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from goal_common import (
    DEFAULT_WIN_MARGIN,
    FINAL_TASKS,
    PRIMARY_METRIC,
    load_json,
    load_yaml,
    metric_float,
    parse_task_list,
    path_under,
    relative_path_under,
    safe_run_id,
    strict_bool,
    validate_baseline_artifacts,
)


VALID_PURPOSES = {"dev", "final", "smoke"}


def known_versions() -> List[str]:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        from src.experiments import version_names
    except Exception:
        return []
    return version_names()


def _add(errors: List[str], message: str) -> None:
    errors.append(message)


def _baseline_exists(manifest: Dict[str, Any]) -> bool:
    baseline = manifest.get("baseline") or {}
    summary = baseline.get("summary_csv")
    manifest_json = baseline.get("manifest_json")
    if not summary or not manifest_json:
        return False
    return bool(validate_baseline_artifacts(summary, manifest_json).get("valid"))


def _state_budget_limits() -> Dict[str, Optional[float]]:
    state_path = Path("outputs/goal/state.json")
    if not state_path.exists():
        return {"max_concurrent_gpu_jobs": None, "max_gpu_hours_per_batch": None}
    try:
        state = load_json(state_path)
    except SystemExit:
        return {"max_concurrent_gpu_jobs": None, "max_gpu_hours_per_batch": None}
    budget = state.get("budget") or {}
    return {
        "max_concurrent_gpu_jobs": budget.get("max_concurrent_gpu_jobs"),
        "max_gpu_hours_per_batch": budget.get("max_gpu_hours_per_batch"),
    }


def _validate_bool_field(errors: List[str], value: Any, field_name: str, default: bool = False) -> bool:
    parsed = strict_bool(value, default=default)
    if parsed is None:
        _add(errors, "{} must be a YAML boolean true/false".format(field_name))
        return default
    return parsed


def _validate_checkpoint_dir(
    errors: List[str],
    label: str,
    checkpoint_dir: Any,
    loop_idx: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    if not checkpoint_dir:
        _add(errors, "{} is required".format(label))
        return None
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        _add(errors, "{} does not exist: {}".format(label, checkpoint_dir))
        return None
    if path_under(checkpoint_path, "outputs/baselines"):
        _add(errors, "{} must not be under outputs/baselines".format(label))
        return None
    config_path = checkpoint_path / "loop_config.json"
    state_path = checkpoint_path / "model_state.pt"
    if not config_path.exists():
        _add(errors, "{} is missing loop_config.json: {}".format(label, config_path))
        return None
    if not state_path.exists():
        _add(errors, "{} is missing model_state.pt: {}".format(label, state_path))
        return None
    try:
        config = load_json(config_path)
    except SystemExit:
        _add(errors, "{} has invalid loop_config.json: {}".format(label, config_path))
        return None
    tmax = config.get("tmax")
    if loop_idx is not None and (not isinstance(tmax, int) or loop_idx > tmax):
        _add(errors, "{} loop_idx {} exceeds checkpoint tmax {}".format(label, loop_idx, tmax))
    return config


def validate_manifest_dict(manifest: Dict[str, Any], path: Optional[Path] = None) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []

    batch_id = manifest.get("batch_id")
    if not batch_id:
        _add(errors, "missing batch_id")
    elif not safe_run_id(batch_id):
        _add(errors, "batch_id contains unsafe characters: {}".format(batch_id))

    purpose = manifest.get("purpose")
    if purpose not in VALID_PURPOSES:
        _add(errors, "purpose must be one of {}".format(", ".join(sorted(VALID_PURPOSES))))

    metric = manifest.get("primary_metric")
    if metric != PRIMARY_METRIC:
        _add(errors, "unsupported metric {!r}; expected {}".format(metric, PRIMARY_METRIC))

    margin = metric_float(manifest.get("win_margin", DEFAULT_WIN_MARGIN))
    if margin is None or margin < 0:
        _add(errors, "win_margin must be a non-negative number")

    experiments = manifest.get("experiments")
    if not isinstance(experiments, list) or not experiments:
        _add(errors, "missing experiments")
        experiments = []

    tasks = manifest.get("tasks") or {}
    dev_tasks = parse_task_list(tasks.get("dev"))
    final_tasks = parse_task_list(tasks.get("final"))
    all_manifest_tasks = dev_tasks + [task for task in final_tasks if task not in dev_tasks]
    for task in all_manifest_tasks:
        if task not in FINAL_TASKS:
            _add(errors, "unknown final task in manifest tasks: {}".format(task))
    if final_tasks and final_tasks != FINAL_TASKS:
        if purpose == "final":
            _add(errors, "purpose=final requires exact protocol final task list")
        else:
            warnings.append("final task list differs from protocol order; final claims require exact protocol coverage")

    defaults = manifest.get("defaults") or {}
    for key in ["config", "output_base", "eval_output_base"]:
        if key not in defaults:
            _add(errors, "defaults.{} is required".format(key))
    if "eval_all_loops" in defaults:
        _validate_bool_field(errors, defaults.get("eval_all_loops"), "defaults.eval_all_loops", default=False)
    if defaults.get("config") and not Path(defaults["config"]).exists():
        _add(errors, "defaults.config does not exist: {}".format(defaults["config"]))
    path_requirements = {
        "output_base": "outputs/goal/runs",
        "eval_output_base": "outputs/goal/eval",
    }
    for key in ["output_base", "eval_output_base"]:
        value = defaults.get(key)
        if value and path_under(value, "outputs/baselines"):
            _add(errors, "defaults.{} must not be under outputs/baselines".format(key))
        if value and not relative_path_under(value, path_requirements[key]):
            _add(errors, "defaults.{} must be a relative path under {}".format(key, path_requirements[key]))

    budget = manifest.get("budget") or {}
    max_jobs = budget.get("max_concurrent_gpu_jobs")
    max_hours = budget.get("max_gpu_hours_estimate")
    allow_submit = _validate_bool_field(errors, budget.get("allow_submit"), "budget.allow_submit", default=False)
    allow_over_budget = _validate_bool_field(errors, budget.get("allow_over_budget"), "budget.allow_over_budget", default=False)
    if not isinstance(max_jobs, int) or max_jobs <= 0:
        _add(errors, "budget.max_concurrent_gpu_jobs must be a positive integer")
    parsed_hours = metric_float(max_hours)
    if parsed_hours is None or parsed_hours <= 0:
        _add(errors, "budget.max_gpu_hours_estimate must be a positive number")

    limits = _state_budget_limits()
    if not allow_over_budget:
        limit_jobs = limits.get("max_concurrent_gpu_jobs")
        if limit_jobs is not None and isinstance(max_jobs, int) and max_jobs > int(limit_jobs):
            _add(errors, "budget.max_concurrent_gpu_jobs exceeds state limit {}".format(limit_jobs))
        limit_hours = limits.get("max_gpu_hours_per_batch")
        if limit_hours is not None and parsed_hours is not None and parsed_hours > float(limit_hours):
            _add(errors, "budget.max_gpu_hours_estimate exceeds state limit {}".format(limit_hours))
    if isinstance(max_jobs, int) and len(experiments) > max_jobs:
        _add(errors, "manifest has {} experiments but max_concurrent_gpu_jobs is {}".format(len(experiments), max_jobs))

    baseline_present = _baseline_exists(manifest)
    if allow_submit and not baseline_present and purpose != "smoke":
        _add(errors, "allow_submit=true requires a frozen baseline unless purpose=smoke")
    if not baseline_present:
        warnings.append("frozen baseline is missing; only dry-run or smoke work is valid")

    seen_run_ids = set()
    versions = known_versions()
    for idx, experiment in enumerate(experiments):
        if not isinstance(experiment, dict):
            _add(errors, "experiments[{}] must be a mapping".format(idx))
            continue
        run_id = experiment.get("run_id")
        if not run_id:
            _add(errors, "experiments[{}] missing run_id".format(idx))
        elif not safe_run_id(run_id):
            _add(errors, "run_id contains unsafe characters: {}".format(run_id))
        elif run_id in seen_run_ids:
            _add(errors, "duplicate run_id: {}".format(run_id))
        else:
            seen_run_ids.add(run_id)

        if not experiment.get("hypothesis"):
            _add(errors, "experiment {} missing hypothesis".format(run_id or idx))
        version = experiment.get("version")
        if not version:
            _add(errors, "experiment {} missing version".format(run_id or idx))
        elif versions and version not in versions:
            _add(errors, "experiment {} has unknown version {}".format(run_id or idx, version))
        risk = experiment.get("risk") or {}
        if not risk.get("reason"):
            _add(errors, "experiment {} missing risk.reason".format(run_id or idx))

        exp_config = experiment.get("config") or defaults.get("config")
        if exp_config and not Path(exp_config).exists():
            _add(errors, "experiment {} config does not exist: {}".format(run_id or idx, exp_config))

        for key in ["output_base", "eval_output_base"]:
            value = experiment.get(key) or defaults.get(key)
            if value and path_under(value, "outputs/baselines"):
                _add(errors, "experiment {} {} must not be under outputs/baselines".format(run_id or idx, key))
            if value and not relative_path_under(value, path_requirements[key]):
                _add(errors, "experiment {} {} must be a relative path under {}".format(run_id or idx, key, path_requirements[key]))

        eval_config = experiment.get("eval") or {}
        exp_tasks = parse_task_list(eval_config.get("task_names"))
        eval_only = _validate_bool_field(errors, experiment.get("eval_only"), "experiment {} eval_only".format(run_id or idx), default=False)
        if "eval_all_loops" in eval_config:
            _validate_bool_field(errors, eval_config.get("eval_all_loops"), "experiment {} eval.eval_all_loops".format(run_id or idx), default=False)
        has_external_eval_checkpoint = any(
            key in eval_config for key in ["checkpoint_dir", "fusion_standard_checkpoint_dir", "fusion_alpha"]
        )
        if has_external_eval_checkpoint and not eval_only:
            _add(errors, "experiment {} uses eval checkpoint/fusion fields and must set eval_only: true".format(run_id or idx))
        loop_idx = eval_config.get("loop_idx")
        if loop_idx is not None and (not isinstance(loop_idx, int) or loop_idx <= 0):
            _add(errors, "experiment {} eval.loop_idx must be a positive integer".format(run_id or idx))
            loop_idx = None
        loop_config: Optional[Dict[str, Any]] = None
        if eval_only:
            loop_config = _validate_checkpoint_dir(
                errors,
                "experiment {} eval.checkpoint_dir".format(run_id or idx),
                eval_config.get("checkpoint_dir"),
                loop_idx=loop_idx,
            )
        fusion_checkpoint = eval_config.get("fusion_standard_checkpoint_dir")
        fusion_alpha_value = eval_config.get("fusion_alpha")
        if bool(fusion_checkpoint) != (fusion_alpha_value is not None):
            _add(errors, "experiment {} fusion_standard_checkpoint_dir and fusion_alpha must be provided together".format(run_id or idx))
        fusion_config: Optional[Dict[str, Any]] = None
        if fusion_checkpoint:
            fusion_config = _validate_checkpoint_dir(
                errors,
                "experiment {} fusion_standard_checkpoint_dir".format(run_id or idx),
                fusion_checkpoint,
                loop_idx=1,
            )
        if fusion_alpha_value is not None:
            fusion_alpha = metric_float(fusion_alpha_value)
            if fusion_alpha is None or fusion_alpha < 0.0 or fusion_alpha > 1.0:
                _add(errors, "experiment {} eval.fusion_alpha must be a number in [0, 1]".format(run_id or idx))
        if loop_config and fusion_config:
            loop_dim = loop_config.get("embedding_dim")
            fusion_dim = fusion_config.get("embedding_dim")
            if loop_dim is not None and fusion_dim is not None and loop_dim != fusion_dim:
                _add(errors, "experiment {} fusion embedding_dim mismatch: {} vs {}".format(run_id or idx, fusion_dim, loop_dim))
        for task in exp_tasks:
            if task not in FINAL_TASKS:
                _add(errors, "experiment {} has unknown eval task {}".format(run_id or idx, task))
        if purpose == "final":
            if exp_tasks != FINAL_TASKS:
                _add(errors, "final experiment {} must evaluate exactly the protocol final tasks".format(run_id or idx))
            loop_indices = eval_config.get("candidate_loop_indices")
            if not isinstance(loop_indices, list) or not loop_indices:
                _add(errors, "final experiment {} must predeclare eval.candidate_loop_indices".format(run_id or idx))
            elif not all(isinstance(loop_idx, int) and loop_idx > 0 for loop_idx in loop_indices):
                _add(errors, "final experiment {} has invalid candidate_loop_indices".format(run_id or idx))

    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "path": str(path) if path else None,
        "batch_id": batch_id,
        "purpose": purpose,
        "baseline_present": baseline_present,
        "experiments": len(experiments),
    }


def validate_manifest(path: Path) -> Dict[str, Any]:
    return validate_manifest_dict(load_yaml(path), path=path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a goal experiment batch manifest.")
    parser.add_argument("manifest", help="Path to a YAML batch manifest.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable validation result.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = validate_manifest(Path(args.manifest))
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        if result["valid"]:
            print("Manifest valid: {}".format(args.manifest))
        else:
            print("Manifest invalid: {}".format(args.manifest))
            for error in result["errors"]:
                print("- {}".format(error))
        for warning in result["warnings"]:
            print("WARNING: {}".format(warning))
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
