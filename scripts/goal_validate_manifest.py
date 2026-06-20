#!/usr/bin/env python3
"""Validate autonomous experiment batch manifests before submission."""

import argparse
import copy
import json
import sys
import tempfile
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
VALID_CLAIM_TRACKS = {"standalone_main", "fusion_diagnostic", "diagnostic"}
FUSION_EVAL_KEYS = {"fusion_standard_checkpoint_dir", "fusion_alpha", "fusion_scope"}
FUSION_TEXT_MARKERS = [
    "standard+loop",
    "standard + loop",
    "weighted concat",
    "weighted concatenation",
    "standard embedding",
    "standard embeddings",
    "standard score",
    "standard scores",
    "frozen standard plus",
    "ensemble with the frozen standard",
    "explicit ensemble",
    "score fusion",
    "fusion_scope",
    "fusion_alpha",
]
STANDARD_SCORING_KEYS = {
    "standard_checkpoint_dir",
    "standard_score",
    "standard_scores",
    "standard_embedding",
    "standard_embeddings",
    "fusion_standard_checkpoint_dir",
}


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


def _present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, dict, set)):
        return bool(value)
    return True


def _string_values(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        values: List[str] = []
        for key, item in value.items():
            values.append(str(key))
            values.extend(_string_values(item))
        return values
    if isinstance(value, (list, tuple, set)):
        values = []
        for item in value:
            values.extend(_string_values(item))
        return values
    return []


def fusion_diagnostic_evidence(experiment: Dict[str, Any]) -> List[str]:
    """Return evidence that an experiment uses frozen-standard fusion/ensemble scoring."""
    evidence: List[str] = []
    eval_config = experiment.get("eval") or {}
    for key in sorted(FUSION_EVAL_KEYS):
        if _present(eval_config.get(key)):
            evidence.append("eval.{}".format(key))

    text = " ".join(_string_values({key: experiment.get(key) for key in [
        "hypothesis",
        "mechanism",
        "candidate_rule",
        "expected_effect",
        "fallback",
    ]})).lower()
    for marker in FUSION_TEXT_MARKERS:
        if marker in text:
            evidence.append("text:{}".format(marker))
    return sorted(set(evidence))


def standard_scoring_evidence(experiment: Dict[str, Any]) -> List[str]:
    eval_config = experiment.get("eval") or {}
    evidence = []
    for key, value in eval_config.items():
        if key in STANDARD_SCORING_KEYS and _present(value):
            evidence.append("eval.{}".format(key))
        if isinstance(value, str) and Path(value).name == "standard" and "checkpoint" in key:
            evidence.append("eval.{}={}".format(key, value))
    return sorted(set(evidence))


def explicit_claim_track(experiment: Dict[str, Any]) -> Optional[str]:
    claim_track = experiment.get("claim_track")
    candidate_track = experiment.get("candidate_track")
    if claim_track is not None and candidate_track is not None and claim_track != candidate_track:
        return "__conflicting__"
    return claim_track if claim_track is not None else candidate_track


def infer_claim_track(experiment: Dict[str, Any], purpose: Optional[str]) -> str:
    explicit = explicit_claim_track(experiment)
    if explicit in VALID_CLAIM_TRACKS:
        return explicit
    if fusion_diagnostic_evidence(experiment):
        return "fusion_diagnostic"
    if purpose == "final":
        return "standalone_main"
    return "diagnostic"


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
    experiment_tracks: List[Dict[str, Any]] = []

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
        explicit_track = explicit_claim_track(experiment)
        if explicit_track == "__conflicting__":
            _add(errors, "experiment {} claim_track and candidate_track conflict".format(run_id or idx))
        elif explicit_track is not None and explicit_track not in VALID_CLAIM_TRACKS:
            _add(errors, "experiment {} claim_track must be one of {}".format(run_id or idx, ", ".join(sorted(VALID_CLAIM_TRACKS))))
        fusion_evidence = fusion_diagnostic_evidence(experiment)
        standard_evidence = standard_scoring_evidence(experiment)
        candidate_track = infer_claim_track(experiment, purpose)
        experiment_tracks.append(
            {
                "run_id": run_id or str(idx),
                "candidate_track": candidate_track,
                "fusion_diagnostic_evidence": fusion_evidence,
            }
        )
        if fusion_evidence and explicit_track is None:
            warnings.append(
                "experiment {} inferred candidate_track=fusion_diagnostic from {}; it cannot trigger main goal success".format(
                    run_id or idx, ", ".join(fusion_evidence)
                )
            )
        if candidate_track == "standalone_main":
            if purpose != "final":
                warnings.append(
                    "experiment {} is standalone_main on purpose={}; it is valid for standalone exploration but cannot trigger main goal success until final validation".format(
                        run_id or idx, purpose
                    )
                )
            if fusion_evidence:
                _add(
                    errors,
                    "experiment {} cannot be standalone_main because it uses frozen-standard fusion/ensemble evidence: {}".format(
                        run_id or idx, ", ".join(fusion_evidence)
                    ),
                )
            if standard_evidence:
                _add(
                    errors,
                    "experiment {} standalone_main must not use the frozen standard checkpoint or standard score in candidate scoring: {}".format(
                        run_id or idx, ", ".join(standard_evidence)
                    ),
                )
            if version == "standard":
                _add(errors, "experiment {} standalone_main cannot use version=standard".format(run_id or idx))
        elif purpose == "final":
            warnings.append(
                "experiment {} is candidate_track={}; final results may be reported only as diagnostic, not main goal success".format(
                    run_id or idx, candidate_track
                )
            )
        if "eval_all_loops" in eval_config:
            _validate_bool_field(errors, eval_config.get("eval_all_loops"), "experiment {} eval.eval_all_loops".format(run_id or idx), default=False)
        has_external_eval_checkpoint = any(
            key in eval_config for key in ["checkpoint_dir", "fusion_standard_checkpoint_dir", "fusion_alpha", "fusion_scope"]
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
        fusion_scope = eval_config.get("fusion_scope", "both")
        if fusion_scope not in {"both", "query_only", "doc_only"}:
            _add(errors, "experiment {} eval.fusion_scope must be one of both, query_only, doc_only".format(run_id or idx))
        if "fusion_scope" in eval_config and not fusion_checkpoint:
            _add(errors, "experiment {} eval.fusion_scope requires fusion_standard_checkpoint_dir and fusion_alpha".format(run_id or idx))
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
        "candidate_tracks": experiment_tracks,
    }


def validate_manifest(path: Path) -> Dict[str, Any]:
    return validate_manifest_dict(load_yaml(path), path=path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a goal experiment batch manifest.")
    parser.add_argument("manifest", nargs="?", help="Path to a YAML batch manifest.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable validation result.")
    parser.add_argument("--self-test", action="store_true", help="Run cheap validator self-tests.")
    return parser.parse_args()


def _fake_checkpoint(root: Path, name: str) -> str:
    path = root / name
    path.mkdir(parents=True)
    (path / "loop_config.json").write_text(json.dumps({"tmax": 3, "embedding_dim": 768}), encoding="utf-8")
    (path / "model_state.pt").write_bytes(b"placeholder")
    return str(path)


def self_test() -> None:
    base_experiment = {
        "run_id": "standalone_candidate",
        "hypothesis": "Self-test standalone final candidate.",
        "version": "loop_matryoshka",
        "eval": {
            "task_names": list(FINAL_TASKS),
            "candidate_loop_indices": [3],
        },
        "risk": {"reason": "self-test"},
    }
    base_manifest = {
        "batch_id": "validator_selftest",
        "purpose": "final",
        "primary_metric": PRIMARY_METRIC,
        "win_margin": DEFAULT_WIN_MARGIN,
        "baseline": {
            "summary_csv": "outputs/baselines/standard_frozen/results_summary.csv",
            "manifest_json": "outputs/baselines/standard_frozen/baseline_manifest.json",
        },
        "budget": {
            "max_concurrent_gpu_jobs": 1,
            "max_gpu_hours_estimate": 1,
            "allow_submit": False,
        },
        "tasks": {"dev": ["SciFact"], "final": list(FINAL_TASKS)},
        "defaults": {
            "config": "configs/smoke.yaml",
            "output_base": "outputs/goal/runs",
            "eval_output_base": "outputs/goal/eval",
        },
        "experiments": [base_experiment],
    }
    standalone = validate_manifest_dict(copy.deepcopy(base_manifest))
    assert standalone["valid"], standalone
    assert standalone["candidate_tracks"][0]["candidate_track"] == "standalone_main"

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        loop_ckpt = _fake_checkpoint(tmpdir, "loop")
        standard_ckpt = _fake_checkpoint(tmpdir, "standard")
        fusion_manifest = copy.deepcopy(base_manifest)
        fusion_exp = fusion_manifest["experiments"][0]
        fusion_exp["run_id"] = "fusion_candidate"
        fusion_exp["eval_only"] = True
        fusion_exp["claim_track"] = "standalone_main"
        fusion_exp["mechanism"] = "standard+loop weighted concat"
        fusion_exp["eval"] = {
            "checkpoint_dir": loop_ckpt,
            "task_names": list(FINAL_TASKS),
            "eval_all_loops": False,
            "loop_idx": 3,
            "candidate_loop_indices": [3],
            "fusion_standard_checkpoint_dir": standard_ckpt,
            "fusion_alpha": 0.2,
            "fusion_scope": "query_only",
        }
        fusion_invalid = validate_manifest_dict(fusion_manifest)
        assert not fusion_invalid["valid"], fusion_invalid
        assert any("cannot be standalone_main" in error for error in fusion_invalid["errors"])

        fusion_manifest["experiments"][0].pop("claim_track")
        fusion_diagnostic = validate_manifest_dict(fusion_manifest)
        assert fusion_diagnostic["valid"], fusion_diagnostic
        assert fusion_diagnostic["candidate_tracks"][0]["candidate_track"] == "fusion_diagnostic"

        dev_manifest = copy.deepcopy(fusion_manifest)
        dev_manifest["batch_id"] = "validator_selftest_dev"
        dev_manifest["purpose"] = "dev"
        dev_manifest["budget"]["max_concurrent_gpu_jobs"] = 1
        dev_manifest["experiments"][0]["eval"]["task_names"] = ["SciFact", "NFCorpus"]
        dev_manifest["experiments"][0]["eval"]["candidate_loop_indices"] = [3]
        dev_result = validate_manifest_dict(dev_manifest)
        assert dev_result["valid"], dev_result
        assert dev_result["candidate_tracks"][0]["candidate_track"] == "fusion_diagnostic"

    standalone_dev = copy.deepcopy(base_manifest)
    standalone_dev["batch_id"] = "validator_selftest_standalone_dev"
    standalone_dev["purpose"] = "dev"
    standalone_dev["experiments"][0]["claim_track"] = "standalone_main"
    standalone_dev["experiments"][0]["eval"]["task_names"] = ["SciFact", "NFCorpus"]
    standalone_dev["experiments"][0]["eval"]["candidate_loop_indices"] = [3]
    standalone_dev_result = validate_manifest_dict(standalone_dev)
    assert standalone_dev_result["valid"], standalone_dev_result
    assert standalone_dev_result["candidate_tracks"][0]["candidate_track"] == "standalone_main"
    print("goal_validate_manifest self-test passed")


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    if not args.manifest:
        raise SystemExit("manifest is required unless --self-test is used")
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
