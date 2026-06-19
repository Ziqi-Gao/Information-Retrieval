# repo_auditor report

## Current execution path

当前 autonomous framework 的主路径是：

1. `scripts/goal_init.py`
   初始化 `outputs/goal/state.json`，记录 protocol task、metric、budget、baseline 状态。
2. `scripts/goal_freeze_baseline.py`
   从已有 `standard` summary 冻结 baseline 到 `outputs/baselines/standard_frozen/`。
3. `scripts/goal_validate_manifest.py`
   校验 `experiments/batches/*.yaml`。
4. `scripts/goal_submit_batch.py --dry-run|--submit`
   生成 `outputs/goal/runs/<batch_id>/submission_plan.json` 和 `batch_manifest.submitted.yaml`；真实提交时调用 `sbatch`。
5. `scripts/goal_status.py --update-state`
   按 plan 里的 job id 查询一次 `squeue`/`sacct`，更新 state。
6. `scripts/goal_collect.py`
   从 `outputs/goal/eval/<batch_id>/<run_id>/results_summary.csv` 收集结果。
7. `scripts/goal_scoreboard.py`
   把 collected CSV 与 frozen baseline 比较，写 scoreboard 并更新 state。

当前状态：
- `outputs/goal/state.json` 处于 `DRY_RUN_SUBMIT`。
- `baseline.status` 是 `missing`。
- `current_batch.batch_id` 是 `batch_001`。
- `outputs/goal/runs/batch_001/submission_plan.json` 是 dry-run plan，没有真实 job id。
- `outputs/goal/runs/batch_001/collected_results.csv` 只有 dry-run 的 `missing_result` 占位行，不是实验结果。

## Important files/functions

- `scripts/goal_common.py`
  - `FINAL_TASKS`, `PRIMARY_METRIC`, `DEFAULT_WIN_MARGIN`
  - `repo_status()`
  - `safe_run_id()`
  - `metric_float()`
  - `parse_task_list()`
  - `atomic_write_json()`

- `scripts/goal_validate_manifest.py`
  - `validate_manifest_dict()`
  - `_baseline_exists()`
  - `_state_budget_limits()`
  - `known_versions()`

- `scripts/goal_submit_batch.py`
  - `SAFE_RUNTIME_ENV_KEYS`
  - `safe_runtime_exports()`
  - `format_export()`
  - `sbatch_args_from_env()`
  - `build_plan()`
  - `assert_output_dirs_available()`
  - `run_sbatch()`
  - `update_state()`

- `scripts/goal_status.py`
  - `query_squeue()`
  - `query_sacct()`
  - `job_status()`
  - `map_status()`

- `scripts/goal_collect.py`
  - `expected_tasks()`
  - `row_candidate_id()`
  - `collect_experiment()`

- `scripts/goal_scoreboard.py`
  - `load_baseline()`
  - `load_candidates()`
  - `evaluate_candidate()`
  - `compare()`
  - `update_state()`

- `src/eval_mteb.py`
  - `evaluate_one_loop()`
  - `append_summary_rows()`
  - `main()`

- `src/train.py`
  - `parse_args()`
  - `assert_encoder_only_trainable()`
  - `build_optimizer()`
  - `main()`

- `src/experiments.py`
  - `VERSION_SPECS`
  - `get_version_spec()`
  - `version_names()`

## Slurm flow

`goal_submit_batch.py` builds one train job and one eval job per manifest experiment.

Train command:
- `sbatch --parsable ... --export=NONE,... scripts/slurm_train.sbatch`

Eval command:
- `sbatch --parsable ... --dependency=afterok:<train_job_id> --export=NONE,... scripts/slurm_eval.sbatch`

Wrappers:
- `scripts/slurm_train.sbatch` runs `python -m src.train`.
- `scripts/slurm_eval.sbatch` runs `python -m src.eval_mteb`.
- `scripts/slurm_env.sh` activates conda or uses `PYTHON_BIN`, then sets HF/cache env defaults.

Important behavior:
- Eval depends on train with `afterok`.
- `goal_submit_batch.py` submits all manifest train jobs sequentially without enforcing `budget.max_concurrent_gpu_jobs`.
- `SBATCH_ARGS` is accepted from the environment and inserted into every `sbatch` call.
- Older scripts still directly call `sbatch`: `scripts/slurm_run_smoke.sh`, `scripts/slurm_run_preexp.sh`, `scripts/slurm_run_eval_all.sh`. They use `--export=ALL`, which can leak environment variables into jobs.

## Result flow

`src.eval_mteb.main()` evaluates one or more tasks and loop indices, then appends rows to:

- `outputs/goal/eval/<batch_id>/<run_id>/results_summary.csv`

Summary columns include:
- `version`
- `task`
- `loop_idx`
- `ndcg_at_10`
- `checkpoint_dir`
- `raw_result_path`

`goal_collect.py` reads each expected run summary:
- Missing summary becomes `missing_result`.
- Missing task becomes `missing_result`.
- Duplicate `(task, loop_idx)` becomes `invalid_metric`.
- Invalid or NaN metric becomes `invalid_metric`.
- Candidate id is synthesized as `<run_id>__loop<loop_idx>`.

`goal_scoreboard.py`:
- Loads frozen baseline from `outputs/baselines/standard_frozen/results_summary.csv`.
- Requires exactly one valid `standard` row per final task.
- Scores every candidate id against all hardcoded `FINAL_TASKS`.
- Candidate passes only if all 7 tasks are valid and each delta is `>= margin`.

## Risks

- `budget.max_concurrent_gpu_jobs` is validated but not enforced. `goal_submit_batch.py` submits every train job in the manifest; concurrency is left to Slurm/account limits.
- `SBATCH_ARGS` is unrestricted. A caller can inject scheduler options that bypass intended resource shape, walltime, partition/account policy, or add arrays/dependencies.
- Existing legacy Slurm launchers still provide direct `sbatch` bypasses: `scripts/slurm_run_smoke.sh`, `scripts/slurm_run_preexp.sh`, `scripts/slurm_run_eval_all.sh`. They also use `--export=ALL`, which conflicts with the framework's safe-export model.
- Manifest final task validation is too soft. `goal_validate_manifest.py` only warns when `tasks.final` differs from protocol order, even for `purpose: final`.
- Baseline presence check only verifies that two paths exist. `_baseline_exists()` does not validate hash consistency, manifest contents, metric, task list, or state agreement.
- `known_versions()` silently returns `[]` if importing `src.experiments` fails. That means manifest validation can skip unknown-version checks when the experiment registry is broken.
- Partial submit failure can leave orphaned jobs. `goal_submit_batch.py` writes the final plan after submitting all jobs; if `sbatch` fails midway, already-submitted job ids may not be persisted.
- `goal_collect.py` does not consult Slurm status or the submission plan. If called before eval completion, it records `missing_result`; if train failed, the result classification is still usually `missing_result`, not `failed_train`.
- `goal_collect.py` overwrites per-run validation status to `invalid_metric` whenever any row is non-completed, including missing-task cases. Row-level status is still useful, but run-level status loses precision.
- Candidate IDs are generated from observed loop outputs rather than being declared in the manifest. With `eval_all_loops=true`, every emitted loop becomes scoreable, which weakens the pre-defined candidate ID guardrail.
- `src/eval_mteb.append_summary_rows()` dedupes by `(version, task, loop_idx, checkpoint_dir)`. Reusing an eval output directory with a different checkpoint can preserve old rows and later produce duplicate candidate/task failures.
- Current repo state is dirty and much of the autonomous framework appears untracked. Until committed or otherwise captured, future sessions may not reproduce the same infrastructure.

## Recommended guardrails

- Make `purpose: final` require exact `tasks.final == FINAL_TASKS`; warning is not enough.
- Enforce `max_concurrent_gpu_jobs` in `goal_submit_batch.py`, or explicitly encode Slurm dependencies / throttling in the generated plan.
- Restrict `SBATCH_ARGS` to an allowlist such as account, partition, qos, time, cpus, mem, gres; reject arrays and conflicting `--export`.
- Add a pre-submit check that blocks direct goal-batch use of `scripts/slurm_run_*.sh`, or move legacy launchers under a clearly deprecated path.
- Replace legacy `--export=ALL` with safe explicit export if those scripts remain.
- Validate frozen baseline manifest in `goal_validate_manifest.py`: metric, task list, summary hash, and exact standard row coverage.
- Persist plan incrementally during submission, especially after each successful train/eval `sbatch`, so partial failures leave recoverable job ids.
- Make `goal_collect.py` consume status data from `goal_status.py` or `submission_plan.json` so train/eval failures are classified distinctly from missing files.
- Add manifest-declared candidate IDs / permitted loop indices, then have `goal_collect.py` reject unexpected loops.
- Add a cheap checker that ensures all goal batch output roots stay under `outputs/goal/` and baseline paths stay under `outputs/baselines/standard_frozen/`.

## Uncertainty

- I did not run validation, preflight, training, MTEB evaluation, `sbatch`, `squeue`, or `sacct`.
- I did not inspect secrets, `.env`, SSH files, token caches, or credential caches.
- I treated current untracked files as part of the active working tree because they are present in the repository checkout.
- I did not verify runtime behavior on the actual Slurm cluster; this is a static/read-only execution-path audit.
