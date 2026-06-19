#!/usr/bin/env python3
"""Run cheap safety checks before any autonomous batch submission."""

import argparse
import copy
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from goal_common import FINAL_TASKS, PRIMARY_METRIC, validate_baseline_artifacts, write_csv_rows
from goal_validate_manifest import validate_manifest_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run cheap goal-control preflight checks.")
    parser.add_argument("--manifest", default="experiments/batches/batch_template.yaml")
    return parser.parse_args()


def run_step(name: str, command: Sequence[str]) -> Tuple[str, int, str, str]:
    print("==> {}".format(name))
    print("    {}".format(" ".join(command)))
    proc = subprocess.Popen(
        list(command),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    stdout, stderr = proc.communicate()
    if stdout.strip():
        print(stdout.strip())
    if stderr.strip():
        print(stderr.strip(), file=sys.stderr)
    status = "PASS" if proc.returncode == 0 else "FAIL"
    print("{}: {}".format(status, name))
    return name, proc.returncode, stdout, stderr


def shell_scripts() -> List[str]:
    paths: List[str] = []
    for pattern in ["*.sh", "*.sbatch"]:
        paths.extend(str(path) for path in sorted(Path("scripts").glob(pattern)))
    return paths


def base_manifest() -> Dict[str, Any]:
    return {
        "batch_id": "selftest_batch",
        "description": "self-test",
        "created_by": "codex",
        "purpose": "smoke",
        "primary_metric": PRIMARY_METRIC,
        "win_margin": 0.001,
        "baseline": {
            "summary_csv": "outputs/baselines/standard_frozen/results_summary.csv",
            "manifest_json": "outputs/baselines/standard_frozen/baseline_manifest.json",
        },
        "budget": {
            "max_concurrent_gpu_jobs": 2,
            "max_gpu_hours_estimate": 1,
            "allow_submit": False,
        },
        "tasks": {"dev": ["SciFact"], "final": list(FINAL_TASKS)},
        "defaults": {
            "config": "configs/smoke.yaml",
            "output_base": "outputs/goal/runs",
            "eval_output_base": "outputs/goal/eval",
            "eval_all_loops": False,
        },
        "experiments": [
            {
                "run_id": "selftest_run",
                "hypothesis": "self-test",
                "version": "standard",
                "config": "configs/smoke.yaml",
                "train": {"epochs": 1, "max_steps": 1, "save_steps": 0},
                "eval": {"task_names": ["SciFact"], "eval_all_loops": False},
                "risk": {"level": "low", "reason": "self-test"},
            }
        ],
    }


def assert_invalid(manifest: Dict[str, Any], expected: str) -> None:
    result = validate_manifest_dict(manifest)
    if result["valid"]:
        raise AssertionError("manifest unexpectedly valid for {}".format(expected))


def guardrail_self_tests() -> None:
    manifest = base_manifest()
    path_escape = dict(manifest)
    path_escape["defaults"] = dict(manifest["defaults"])
    path_escape["defaults"]["output_base"] = "../outside"
    assert_invalid(path_escape, "path traversal")

    too_many = base_manifest()
    too_many["experiments"] = [copy.deepcopy(too_many["experiments"][0]) for _ in range(3)]
    for idx, experiment in enumerate(too_many["experiments"]):
        experiment["run_id"] = "selftest_run_{}".format(idx)
    assert_invalid(too_many, "max_concurrent_gpu_jobs")

    final_subset = base_manifest()
    final_subset["purpose"] = "final"
    final_subset["experiments"][0]["eval"] = {
        "task_names": ["SciFact"],
        "eval_all_loops": False,
        "candidate_loop_indices": [1],
    }
    assert_invalid(final_subset, "final task subset")

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        summary = tmpdir / "results_summary.csv"
        manifest_path = tmpdir / "baseline_manifest.json"
        rows = []
        per_task = {}
        for idx, task in enumerate(FINAL_TASKS):
            value = 0.1 + idx * 0.01
            rows.append({"version": "standard", "task": task, PRIMARY_METRIC: value})
            per_task[task] = value
        write_csv_rows(summary, rows, ["version", "task", PRIMARY_METRIC])
        bad_manifest = {
            "metric": PRIMARY_METRIC,
            "tasks": list(FINAL_TASKS),
            "sha256": "bad",
            "per_task_baseline": per_task,
        }
        manifest_path.write_text(json.dumps(bad_manifest), encoding="utf-8")
        validation = validate_baseline_artifacts(summary, manifest_path)
        if validation["valid"]:
            raise AssertionError("tampered baseline unexpectedly validated")
    print("Guardrail self-tests passed")


def main() -> None:
    args = parse_args()
    py = sys.executable
    steps: List[Tuple[str, List[str]]] = [
        ("compile src and scripts", [py, "-m", "compileall", "-q", "src", "scripts"]),
    ]
    shell_paths = shell_scripts()
    if shell_paths:
        steps.append(("shell syntax", ["bash", "-n"] + shell_paths))
    steps.extend(
        [
            ("manifest validation", [py, "scripts/goal_validate_manifest.py", args.manifest]),
            ("submission dry-run", [py, "scripts/goal_submit_batch.py", args.manifest, "--dry-run"]),
            ("scoreboard self-test", [py, "scripts/goal_scoreboard.py", "--self-test"]),
        ]
    )

    print("==> guardrail self-tests")
    guardrail_self_tests()
    print("PASS: guardrail self-tests")

    results = [run_step(name, command) for name, command in steps]
    failed = [name for name, code, _, _ in results if code != 0]
    if failed:
        print("Preflight failed: {}".format(", ".join(failed)))
        raise SystemExit(1)
    print("Preflight passed.")


if __name__ == "__main__":
    main()
