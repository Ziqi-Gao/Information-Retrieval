# Agent Lab Notebook

This notebook records factual preparation steps for the autonomous retrieval goal. Keep it concise and update it when state changes.

## 2026-06-19 Preparation Pass

- Branch audited: `codex-bert-only-loop-memory`.
- Current goal: prepare safe autonomous experiment-control infrastructure only.
- No new retrieval model was implemented.
- No expensive training was run.
- No real Slurm batch was submitted.
- Baseline status at start: no frozen baseline found under `outputs/baselines/standard_frozen/`.
- Existing standard final-grid summary observed at `outputs/final_grid_experiment/eval/summaries/standard_results_summary.csv`; it is not considered frozen until processed by `scripts/goal_freeze_baseline.py`.
- Real Codex subagents were used for read-only repo audit, literature scouting, experiment planning, and code-risk review.
- Subagent reports were saved under `docs/subagent_reports/`.
- Parent applied blocker/high risk fixes from the code-risk review: legacy Slurm wrappers and local training wrappers now fail safe by default, manifest validation is stricter, baseline validation checks hash/task coverage, `SBATCH_ARGS` is allowlisted, and final candidates must be predeclared.

## Operating Notes

- Use the project Python environment if plain `python` is unavailable on the login shell.
- Run `scripts/goal_preflight.py` before any real autonomous batch.
- Treat missing, failed, duplicate, partial, or NaN results as failures.
- Keep personal paths, account names, and tokens out of tracked files.

## 2026-06-19 Round 001 Dev Batch Design

- Baseline validated before Round 001:
  - `outputs/baselines/standard_frozen/results_summary.csv`
  - `outputs/baselines/standard_frozen/baseline_manifest.json`
  - `baseline.status = frozen`
  - primary metric: `ndcg_at_10`
- Preflight passed on `experiments/batches/batch_template.yaml`.
- Read-only subagents used:
  - `repo_auditor`: confirmed current wrapper always used train+eval and identified minimal hooks for `LOOP_IDX`, eval-only, and retrieval-time fusion.
  - `literature_scout`: recommended standard-preserving evaluation-only fusion before training changes.
  - `experiment_planner`: recommended a compact dev batch within the 4-job and 24-GPU-hour limits.
- Parent decision: implement a minimal evaluation-only standard+loop weighted-concat fusion path, not new training, not a new base model, and not final-task validation.
- Round 001 manifest: `experiments/batches/batch_001_dev.yaml`.
- Dev tasks: `SciFact`, `NFCorpus`, `FiQA2018`, `SCIDOCS`.
- Pre-registered dev candidates:
  - `r001_fuse_loop_matryoshka_t2_a10__loop2`
  - `r001_fuse_loop_matryoshka_t3_a10__loop3`
  - `r001_fuse_loop_final_t2_a10__loop2`
- These are exploratory dev candidates only. They cannot support a final claim.
- Round 001 validation:
  - `python scripts/goal_validate_manifest.py experiments/batches/batch_001_dev.yaml` passed.
  - `python scripts/goal_submit_batch.py experiments/batches/batch_001_dev.yaml --dry-run` passed with `mode=eval_only` and no train jobs.
  - `python scripts/goal_preflight.py --manifest experiments/batches/batch_001_dev.yaml` passed.
- Code-risk reviewer findings:
  - Fixed strict YAML boolean handling for submit-related fields.
  - Fixed eval-only fail-closed validation.
  - Added checkpoint/tmax validation for eval-only manifests.
  - Added fusion metadata columns to result summaries and collection.
  - Fixed `goal_status.py` so eval-only plans do not synthesize train status rows.
- Round 001 submitted through `scripts/goal_submit_batch.py --submit` only.
- Slurm job IDs:
  - `r001_fuse_loop_matryoshka_t2_a10`: eval job `4956041`
  - `r001_fuse_loop_matryoshka_t3_a10`: eval job `4956042`
  - `r001_fuse_loop_final_t2_a10`: eval job `4956043`
- Immediate `scripts/goal_status.py --update-state` showed all three jobs `running`.
- Current state phase after submission: `SUBMIT_BATCH`.
- Next action: resume later with `scripts/goal_status.py`, then collect and score only after jobs are terminal.

## 2026-06-20 Watcher Infrastructure

- Added `scripts/goal_watch_batch.py` to poll submitted goal batches through `scripts/goal_status.py`.
- Added docs in `docs/goal_watch_batch.md` and updated `docs/goal_protocol.md`.
- Watcher modes:
  - `notify`: print next status, collect, and scoreboard commands after all jobs are terminal.
  - `codex`: after terminal state only, launch `codex exec --sandbox workspace-write` with a resume prompt.
- Watcher safety:
  - refuses to run inside Slurm unless explicitly allowed
  - requires frozen baseline, state file, submitted job IDs, and submission plan
  - writes `watcher.log` and `watcher_status.json`
  - uses `.codex_resume_launched` to avoid repeated Codex resumes
  - does not submit jobs, train, evaluate, collect, score, or change metrics by itself
- Watcher notify test:
  - normal sandbox polling timed out because Slurm status access was restricted
  - elevated local polling succeeded and found all three `batch_001_dev` eval jobs terminal with `completed` status
  - no collect or scoreboard command was run by the watcher test

## 2026-06-20 Round 001 Dev Results

- Status refresh through `scripts/goal_status.py --update-state` found all three `batch_001_dev` eval jobs completed.
- Collection through `scripts/goal_collect.py` wrote 12 valid rows to `outputs/goal/runs/batch_001_dev/collected_results.csv`.
- Scoreboard through `scripts/goal_scoreboard.py` wrote `outputs/goal/runs/batch_001_dev/scoreboard.csv` and `.json`.
- No invalid dev-task rows were found; all three candidates covered `SciFact`, `NFCorpus`, `FiQA2018`, and `SCIDOCS`.
- Scoreboard correctly marks all candidates `pass_all_tasks = false` because this was a 4-task dev batch and does not cover `ArguAna`, `Touche2020`, or `TRECCOVID`.
- Dev-only deltas against frozen standard:
  - `r001_fuse_loop_matryoshka_t3_a10__loop3`: wins 3/4 dev tasks, min delta `+0.00070`, mean delta `+0.00165`; misses the margin on `SCIDOCS`.
  - `r001_fuse_loop_matryoshka_t2_a10__loop2`: wins 2/4 dev tasks, min delta `+0.00017`, mean delta `+0.00129`; misses the margin on `NFCorpus` and `SCIDOCS`.
  - `r001_fuse_loop_final_t2_a10__loop2`: wins 1/4 dev tasks, min delta `-0.00023`, mean delta `+0.00067`; only `SCIDOCS` clears the margin.
- Read-only result analyst agreed that `loop_matryoshka_t3_a10` is the best dev signal but not sufficient for final validation.
- Parent decision: do not promote Round 001 to final validation. Design another dev batch that preserves the matryoshka loop-3 gains while addressing the `SCIDOCS` margin gap without per-task cherry-picking.

## 2026-06-20 Round 002 Dev Batch Design

- Round 002 manifest: `experiments/batches/batch_002_dev.yaml`.
- Dev tasks remain `SciFact`, `NFCorpus`, `FiQA2018`, and `SCIDOCS`.
- Parent decision: keep the method standard-preserving and evaluation-only. Do not train, change the base model, or make a final claim.
- Batch goal: test whether the Round 001 matryoshka loop-3 signal can clear the `SCIDOCS` dev margin without losing the other three dev tasks.
- Pre-registered dev candidates:
  - `r002_both_loop_matryoshka_t3_a15__loop3`: both-side fusion, alpha `0.15`.
  - `r002_both_loop_matryoshka_t3_a20__loop3`: both-side fusion, alpha `0.20`.
  - `r002_query_loop_matryoshka_t3_a10__loop3`: query-only fusion, alpha `0.10`.
  - `r002_query_loop_matryoshka_t3_a20__loop3`: query-only fusion, alpha `0.20`.
- Candidate rule is global for each run: same loop index, fusion scope, and alpha for every task. No per-task loop, alpha, or scope selection is allowed.
- Added `fusion_scope` with default `both`; `query_only` uses standard document embeddings on both concatenated document sides and loop-enhanced embeddings only on the query side.
- Code-risk reviewer was used before submission. Blocker/high fixes applied:
  - `scripts/slurm_train.sbatch` and `scripts/slurm_eval.sbatch` now refuse direct `sbatch` unless launched by `scripts/goal_submit_batch.py` or explicitly overridden with `ALLOW_LEGACY_DIRECT_SBATCH=1`.
  - `scripts/goal_submit_batch.py` now writes dry-run artifacts to `dry_run_plan.json` and `batch_manifest.dry_run.yaml` instead of overwriting submitted plans.
  - `scripts/goal_preflight.py` uses a temporary state file for dry-run checks.
  - Fusion raw artifact directories now include fusion scope and alpha to reduce accidental raw JSON overwrites.
- Validation before planned submission:
  - `python -m compileall src scripts` passed.
  - `bash -n scripts/*.sh scripts/*.sbatch` passed.
  - `python scripts/goal_validate_manifest.py experiments/batches/batch_002_dev.yaml` passed.
  - `python scripts/goal_submit_batch.py experiments/batches/batch_002_dev.yaml --dry-run --resume` passed and wrote `outputs/goal/runs/batch_002_dev/dry_run_plan.json`.
  - `python scripts/goal_preflight.py --manifest experiments/batches/batch_002_dev.yaml` passed.
- Round 002 submitted through `scripts/goal_submit_batch.py --submit` only.
- Slurm job IDs:
  - `r002_both_loop_matryoshka_t3_a15`: eval job `4957408`
  - `r002_both_loop_matryoshka_t3_a20`: eval job `4957409`
  - `r002_query_loop_matryoshka_t3_a10`: eval job `4957410`
  - `r002_query_loop_matryoshka_t3_a20`: eval job `4957411`
- Scheduler options were provided through `SBATCH_ARGS` for this local cluster; they were not written into tracked manifests.
- Post-submit watcher fix: `scripts/goal_watch_batch.py` now marks state terminal before launching Codex so codex-mode resume cannot be overwritten by the watcher after Codex returns.
- Long-running codex-mode watcher is running in tmux session `goal_watch_batch_002` after plain `nohup` was not persistent in this execution environment.
- Watcher files:
  - `outputs/goal/runs/batch_002_dev/watcher.log`
  - `outputs/goal/runs/batch_002_dev/watcher.tmux.log`
  - `outputs/goal/runs/batch_002_dev/watcher_status.json`
  - `outputs/goal/runs/batch_002_dev/codex_resume_after_terminal.md`
- First codex-mode watcher poll found all four eval jobs running.

## 2026-06-20 Round 002 Dev Results

- Watcher check after resume:
  - `tmux` session `goal_watch_batch_002` was no longer present.
  - No `goal_watch_batch.py` or `codex exec` watcher process was found.
  - `watcher_status.json` remained from an earlier short notify-mode timeout, so it was not used as proof of current watcher activity.
- Status refresh through `scripts/goal_status.py --update-state` found all four `batch_002_dev` eval jobs completed.
- Collection through `scripts/goal_collect.py` wrote 16 valid rows to `outputs/goal/runs/batch_002_dev/collected_results.csv`.
- Scoreboard through `scripts/goal_scoreboard.py` wrote `outputs/goal/runs/batch_002_dev/scoreboard.csv` and `.json`.
- No invalid dev-task rows were found; all four candidates covered `SciFact`, `NFCorpus`, `FiQA2018`, and `SCIDOCS`.
- Scoreboard correctly marks all candidates `pass_all_tasks = false` because this was a 4-task dev batch and does not cover `ArguAna`, `Touche2020`, or `TRECCOVID`.
- Dev-only deltas against frozen standard:
  - `r002_query_loop_matryoshka_t3_a20__loop3`: wins 4/4 dev tasks, min delta `+0.00150`, mean delta `+0.00245`.
  - `r002_both_loop_matryoshka_t3_a15__loop3`: wins 4/4 dev tasks, min delta `+0.00140`, mean delta `+0.00193`.
  - `r002_both_loop_matryoshka_t3_a20__loop3`: wins 4/4 dev tasks, min delta `+0.00129`, mean delta `+0.00181`.
  - `r002_query_loop_matryoshka_t3_a10__loop3`: wins 3/4 dev tasks, min delta `+0.00070`, mean delta `+0.00165`; misses the margin on `SCIDOCS`.
- Parent decision: Round 002 gives a stronger dev-only signal than Round 001, with `r002_query_loop_matryoshka_t3_a20__loop3` as the leading pre-registered candidate. It still cannot support a final claim without full final-task coverage.
- Resume note:
  - Plain `python` was unavailable in the interactive shell; local collection and scoring were rerun with `python3`.
  - A direct `scripts/goal_status.py --state outputs/goal/state.json --update-state` retry timed out because Slurm socket access returned `Operation not permitted` in the current sandbox. Existing state already showed the four `batch_002_dev` eval jobs terminal, and local result files were collected through the goal scripts.
  - A read-only result analyst independently confirmed the same Round 002 deltas, missing final tasks, and final-validation recommendation.

## 2026-06-20 Round 003 Final Validation Dry-Run Plan

- Parent decision: promote only the leading Round 002 dev candidate to a final-validation dry-run plan. Do not run another dev batch unless final validation fails or the user asks for more dev exploration.
- Round 003 manifest: `experiments/batches/batch_003_final.yaml`.
- Predeclared final candidate:
  - `r003_final_query_loop_matryoshka_t3_a20__loop3`: query-only fusion, alpha `0.20`, loop index `3`, all seven protocol final tasks.
- Candidate rule is global for every final task: same fusion scope, alpha, and loop index. No per-task loop, alpha, or scope selection is allowed.
- Manifest safety:
  - `purpose: final`
  - exact protocol final task order
  - `candidate_loop_indices: [3]`
  - `budget.allow_submit: false`
- Validation:
  - `scripts/goal_validate_manifest.py experiments/batches/batch_003_final.yaml --json` passed.
  - System `python3` could validate but failed dry-run writing because its PyYAML is too old for `sort_keys=False`.
  - Dry-run was rerun through the project environment loaded by `scripts/slurm_env.sh` and passed.
- Dry-run output:
  - `outputs/goal/runs/batch_003_final/dry_run_plan.json`
  - `outputs/goal/runs/batch_003_final/batch_manifest.dry_run.yaml`
- No new Slurm jobs were submitted.
- Next action: review the final dry-run plan. Submit final validation only after explicit user approval and after enabling `allow_submit` or otherwise providing an approved submit path.

## 2026-06-20 Slurm-Native Postprocess Framework

- Motivation: this HPC may kill `tmux`, `nohup`, and other login-node watcher processes when the SSH or VSCode session closes.
- Parent decision: do not rely on long-running login-node watchers for unattended autonomous workflow progress.
- Added `scripts/slurm_postprocess.sbatch`:
  - CPU-only Slurm job with 1 CPU, 8G memory, and 1 hour wall time.
  - Runs deterministic status refresh, collection, and scoreboard commands after eval jobs are terminal.
  - Writes `postprocess_done.json` on success and `postprocess_failed.json` on failure under `outputs/goal/runs/<batch_id>/`.
  - Does not train, evaluate MTEB, submit new jobs, overwrite baselines, or change metric semantics.
- Updated `scripts/goal_submit_batch.py`:
  - Added `--submit-postprocess`.
  - Dry-run records and prints the postprocess `sbatch` command.
  - Real submit will submit one postprocess dependency job with `afterany:<all eval job ids>`.
  - Postprocess export uses `--export=NONE` plus a narrow allowlist; token/API/SSH/cloud credential variables are not exported.
- Updated `scripts/goal_status.py` so train, eval, and postprocess jobs are all displayed and written to state.
- Updated watcher docs and runtime warning:
  - `scripts/goal_watch_batch.py` remains available as an optional local helper.
  - Slurm dependency postprocess is the preferred automation path because Slurm jobs survive VSCode logout.
- No training, evaluation, real Slurm submission, baseline overwrite, or GitHub push was performed for this framework change.

## 2026-06-20 Autonomous Loop Resume Check

- Current state points to `batch_003_final`.
- `batch_003_final` is dry-run only:
  - plan: `outputs/goal/runs/batch_003_final/dry_run_plan.json`
  - manifest: `experiments/batches/batch_003_final.yaml`
  - no `outputs/goal/runs/batch_003_final/submission_plan.json`
  - no Slurm eval or postprocess job IDs
- Required status command outcome:
  - `python scripts/goal_status.py --state outputs/goal/state.json --update-state` could not run because plain `python` is unavailable in this shell.
  - The project Python fallback reached `scripts/goal_status.py`, which correctly failed because `batch_003_final` has no submitted plan.
- Postprocess search:
  - no `postprocess_done.json` was found under `outputs/goal/runs/`
  - no `postprocess_failed.json` was found under `outputs/goal/runs/`
  - no completed Slurm-native postprocess batch is available to analyze
- Latest submitted batch remains `batch_002_dev`; `scripts/goal_status.py --batch-id batch_002_dev --json` shows its four eval jobs completed, but that batch predates Slurm-native postprocess submission and has no postprocess job.
- Stop condition: no new batch was created or submitted. Next action remains review of the `batch_003_final` dry-run plan and explicit approval before any final validation submission.

## 2026-06-20 Round 003 Final Validation Submission

- User explicitly approved final validation for `batch_003_final`.
- Guardrails before submission:
  - Candidate rule was not changed: `fusion_scope=query_only`, `fusion_alpha=0.20`, `loop_idx=3`.
  - Manifest remains `purpose: final` with all seven protocol final tasks.
  - Final candidate is predeclared through `candidate_loop_indices: [3]`.
  - Frozen baseline paths remain unchanged.
  - No metric semantics were changed.
- Manifest update:
  - `experiments/batches/batch_003_final.yaml` changed only `budget.allow_submit` from `false` to `true`.
- Validation:
  - `scripts/goal_validate_manifest.py experiments/batches/batch_003_final.yaml` passed.
  - `scripts/goal_submit_batch.py experiments/batches/batch_003_final.yaml --dry-run --submit-postprocess` passed.
  - `scripts/goal_preflight.py --manifest experiments/batches/batch_003_final.yaml` passed.
- Submission:
  - Eval job was submitted through `scripts/goal_submit_batch.py --submit --submit-postprocess`.
  - Eval job ID: `4958307`.
  - Initial postprocess submit failed because the GPU partition rejected CPU-only postprocess jobs.
  - Added a controlled `--submit-postprocess-only` repair path to `scripts/goal_submit_batch.py` so postprocess can be submitted through the project wrapper without repeating eval submission.
  - Postprocess job was submitted through `scripts/goal_submit_batch.py --submit --submit-postprocess-only`.
  - Postprocess job ID: `4958309`.
  - Postprocess dependency: `afterany:4958307`.
- Immediate queue state:
  - eval `4958307`: `PENDING`
  - postprocess `4958309`: `PENDING` with dependency
- Immediate historical status after submission:
  - eval `4958307`: `FAILED`, exit `127:0`, elapsed `00:00:06`
  - postprocess `4958309`: `PENDING`; because dependency is `afterany`, it should still run and record deterministic failure outputs
- No final claim is made. Final validation can be interpreted only after the postprocess scoreboard exists and passes all seven final tasks by the `+0.001` margin.

## 2026-06-20 Round 003 Final Validation Repair

- Resume check found both original jobs terminal:
  - eval `4958307`: `FAILED`, exit `127:0`
  - postprocess `4958309`: `FAILED`, exit `127:0`
- Postprocess marker:
  - `outputs/goal/runs/batch_003_final/postprocess_failed.json`
  - failure line: `74`
- Root cause:
  - `scripts/goal_submit_batch.py` exported `PYTHON_BIN=python` because the submit shell had sourced `scripts/slurm_env.sh`.
  - With `--export=NONE`, compute jobs inherited `PYTHON_BIN=python` but did not inherit an activated conda environment.
  - `scripts/slurm_env.sh` skipped conda activation because `PYTHON_BIN` was already set.
  - Compute nodes did not have `python` on `PATH`, so both eval and postprocess failed with `python: command not found`.
- Infrastructure-only fix:
  - `scripts/goal_submit_batch.py` now filters generic `PYTHON_BIN=python` and `PYTHON_BIN=python3` out of Slurm exports.
  - Absolute or explicitly configured Python paths can still be exported.
  - Candidate rule, metric, frozen baseline, and final task set were not changed.
- Repair manifest:
  - `experiments/batches/batch_003_final_repair.yaml`
  - Same run ID and candidate rule as `batch_003_final`.
  - Same seven final tasks and `candidate_loop_indices: [3]`.
- Validation before repair submit:
  - `source scripts/slurm_env.sh && "$PYTHON_BIN" -m compileall src scripts` passed.
  - `bash -n scripts/*.sh scripts/*.sbatch` passed.
  - `source scripts/slurm_env.sh && "$PYTHON_BIN" scripts/goal_validate_manifest.py experiments/batches/batch_003_final_repair.yaml` passed.
  - `source scripts/slurm_env.sh && "$PYTHON_BIN" scripts/goal_submit_batch.py experiments/batches/batch_003_final_repair.yaml --dry-run --submit-postprocess` passed and did not export generic `PYTHON_BIN`.
  - `source scripts/slurm_env.sh && "$PYTHON_BIN" scripts/goal_preflight.py --manifest experiments/batches/batch_003_final_repair.yaml` passed.
- Repair submission through `scripts/goal_submit_batch.py --submit --submit-postprocess` only:
  - eval job `4958436`
  - postprocess job `4958437`
  - postprocess dependency `afterany:4958436`
- Immediate queue state:
  - eval `4958436`: `PENDING`
  - postprocess `4958437`: `PENDING` with dependency
- No final claim is made. The repair batch must wait for Slurm-native postprocess and then be scored against all seven final tasks.

## 2026-06-20 Acceptance Policy Tightening

- Protocol-only change; no retrieval method, model architecture, loss, evaluation metric semantics, experiment candidate, training, eval, Slurm submission, or GitHub push was performed.
- Candidate claims are now split into `standalone_main`, `fusion_diagnostic`, and `diagnostic`.
- Existing weighted standard+loop fusion batches, including `batch_003_final` and `batch_003_final_repair`, remain valid historical diagnostics. They cannot trigger `main_goal_success`; if their numeric deltas pass, they are labeled `fusion_diagnostic_pass`.
- Main goal success now requires a `standalone_main` final candidate, all seven final tasks valid, every final-task delta at least `+0.002`, macro mean delta at least `+0.005`, and no task regression.
- `+0.001` all-task clearance is retained only as `minimal_positive_signal`; it is not goal achieved.
- `publishable_score_candidate` requires `standalone_main`, every final-task delta at least `+0.002`, and macro mean delta at least `+0.008`; without query-level significance evidence it remains `score-only, not statistically certified`.
