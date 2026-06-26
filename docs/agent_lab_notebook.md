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

## 2026-06-20 Batch 003 Final Repair Result

- `batch_003_final_repair` completed Slurm-native postprocess:
  - eval job `4958436`
  - postprocess job `4958437`
  - marker: `outputs/goal/runs/batch_003_final_repair/postprocess_done.json`
- Collection validated all seven protocol final tasks for `r003_final_query_loop_matryoshka_t3_a20__loop3`; no missing, duplicate, NaN, failed, timeout, or partial result rows were found.
- The scoreboard was regenerated under the tightened acceptance policy.
- Candidate track: `fusion_diagnostic`, because the run uses `fusion_standard_checkpoint_dir`, `fusion_alpha`, `fusion_scope`, and frozen-standard weighted fusion.
- Per-task `ndcg_at_10` deltas vs frozen standard:
  - `SciFact`: `+0.00181`
  - `NFCorpus`: `+0.00227`
  - `SCIDOCS`: `+0.00186`
  - `FiQA2018`: `+0.00129`
  - `ArguAna`: `+0.00019`
  - `Touche2020`: `-0.00045`
  - `TRECCOVID`: `+0.01152`
- Aggregate:
  - valid tasks: `7/7`
  - diagnostic tasks won at `+0.001`: `5/7`
  - tasks at standalone main `+0.002` margin: `2/7`
  - min delta: `-0.00045`
  - mean delta: `+0.00264`
  - `minimal_positive_signal=false`
  - `fusion_diagnostic_pass=false`
  - `main_goal_success=false`
  - `publishable_score_candidate=false`
- Failure modes:
  - `ArguAna` did not clear the weak diagnostic `+0.001` threshold.
  - `Touche2020` regressed below the frozen standard baseline.
  - The run is a frozen-standard fusion diagnostic and cannot count as `standalone_main` even if it had passed numerically.
- Decision: no goal-success claim and no new batch submission in this step.

## 2026-06-20 Batch 004 Dev Standalone Exploration

- Parent decision: after the tightened acceptance policy, resume with dev-only standalone exploration rather than another fusion diagnostic.
- `batch_003_final_repair` final-task deltas were treated as diagnostic bookkeeping only and were not used for task-specific tuning.
- Created `experiments/batches/batch_004_dev.yaml`.
- Batch purpose: `dev`.
- Candidate track: `standalone_main` exploration on dev tasks only. These dev results cannot trigger `main_goal_success`.
- Dev tasks:
  - `SciFact`
  - `NFCorpus`
  - `FiQA2018`
  - `SCIDOCS`
- Predeclared standalone candidates:
  - `r004_standalone_loop_matryoshka_t2`: existing `loop_matryoshka_mean_pool` checkpoint, fixed `loop_idx=2`.
  - `r004_standalone_loop_matryoshka_t3`: existing `loop_matryoshka_mean_pool` checkpoint, fixed `loop_idx=3`.
  - `r004_standalone_loop_matryoshka_t4`: existing `loop_matryoshka_mean_pool` checkpoint, fixed `loop_idx=4`.
  - `r004_standalone_recurrent_matryoshka_t3`: existing `loop_matryoshka_recurrent_mean_pool` checkpoint, fixed `loop_idx=3`.
- Guardrails:
  - No training.
  - No new retrieval method or architecture change.
  - No frozen-standard checkpoint, frozen-standard score, `fusion_standard_checkpoint_dir`, `fusion_alpha`, `fusion_scope`, weighted standard+loop concatenation, or score interpolation in candidate scoring.
  - Candidate rule is global per run across all dev tasks.
- Minor validator/protocol adjustment:
  - Dev manifests may use `claim_track: standalone_main` for standalone-only exploration.
  - `main_goal_success` still requires `purpose: final`.
- Checks before submission:
  - `source scripts/slurm_env.sh && "$PYTHON_BIN" -m compileall src scripts` passed.
  - `bash -n scripts/*.sh scripts/*.sbatch` passed.
  - `source scripts/slurm_env.sh && "$PYTHON_BIN" scripts/goal_validate_manifest.py experiments/batches/batch_004_dev.yaml` passed with expected dev-only standalone warnings.
  - `source scripts/slurm_env.sh && "$PYTHON_BIN" scripts/goal_submit_batch.py experiments/batches/batch_004_dev.yaml --dry-run --submit-postprocess` passed.
  - `source scripts/slurm_env.sh && "$PYTHON_BIN" scripts/goal_preflight.py --manifest experiments/batches/batch_004_dev.yaml` passed.
- Submission:
  - Submitted only through `scripts/goal_submit_batch.py --submit --submit-postprocess`.
  - Eval job IDs:
    - `r004_standalone_loop_matryoshka_t2`: `4960330`
    - `r004_standalone_loop_matryoshka_t3`: `4960331`
    - `r004_standalone_loop_matryoshka_t4`: `4960332`
    - `r004_standalone_recurrent_matryoshka_t3`: `4960333`
  - Postprocess job ID: `4960334`
  - Postprocess dependency: `afterany:4960330:4960331:4960332:4960333`
- Next action: wait for Slurm-native postprocess, then inspect `outputs/goal/runs/batch_004_dev/scoreboard.json`.

## 2026-06-20 Batch 004 Dev Standalone Result

- `batch_004_dev` completed Slurm-native postprocess:
  - eval jobs: `4960330`, `4960331`, `4960332`, `4960333`
  - postprocess job: `4960334`
  - marker: `outputs/goal/runs/batch_004_dev/postprocess_done.json`
  - scoreboard: `outputs/goal/runs/batch_004_dev/scoreboard.json`
- A local `goal_status.py --batch-id batch_004_dev --update-state` refresh attempt hung in Slurm status querying and was terminated. The postprocess marker and scoreboard are the terminal evidence.
- Batch purpose: `dev`; all candidates are `standalone_main` exploration only, so this batch cannot trigger `main_goal_success`.
- Scoreboard interpretation under the current claim-track policy:
  - `r004_standalone_loop_matryoshka_t3__loop3`: track `standalone_main`, `minimal_positive_signal=false`, `fusion_diagnostic_pass=false`, `research_grade_threshold_pass=false`, `main_goal_success=false`, `publishable_score_candidate=false`, min delta `-0.00732`, mean delta `-0.00253`, tasks won/lost `1/6`, valid tasks `4/7`.
    - Dev deltas: `SciFact +0.00011`, `NFCorpus -0.00408`, `SCIDOCS +0.00118`, `FiQA2018 -0.00732`; `ArguAna`, `Touche2020`, and `TRECCOVID` were not evaluated by design.
  - `r004_standalone_loop_matryoshka_t4__loop4`: track `standalone_main`, `minimal_positive_signal=false`, `fusion_diagnostic_pass=false`, `research_grade_threshold_pass=false`, `main_goal_success=false`, `publishable_score_candidate=false`, min delta `-0.00770`, mean delta `-0.00278`, tasks won/lost `0/7`, valid tasks `4/7`.
    - Dev deltas: `SciFact +0.00008`, `NFCorpus -0.00419`, `SCIDOCS +0.00068`, `FiQA2018 -0.00770`; three final-only tasks were not evaluated by design.
  - `r004_standalone_loop_matryoshka_t2__loop2`: track `standalone_main`, `minimal_positive_signal=false`, `fusion_diagnostic_pass=false`, `research_grade_threshold_pass=false`, `main_goal_success=false`, `publishable_score_candidate=false`, min delta `-0.00853`, mean delta `-0.00383`, tasks won/lost `1/6`, valid tasks `4/7`.
    - Dev deltas: `SciFact -0.00344`, `NFCorpus -0.00477`, `SCIDOCS +0.00140`, `FiQA2018 -0.00853`; three final-only tasks were not evaluated by design.
  - `r004_standalone_recurrent_matryoshka_t3__loop3`: track `standalone_main`, `minimal_positive_signal=false`, `fusion_diagnostic_pass=false`, `research_grade_threshold_pass=false`, `main_goal_success=false`, `publishable_score_candidate=false`, min delta `-0.38973`, mean delta `-0.21850`, tasks won/lost `0/7`, valid tasks `4/7`.
    - Dev deltas: `SciFact -0.38973`, `NFCorpus -0.15960`, `SCIDOCS -0.11732`, `FiQA2018 -0.20735`; three final-only tasks were not evaluated by design.
- Decision: `main_goal_success=false`. The standalone mean-pool loopwise checkpoint did not show a viable dev signal as an independent scorer, and recurrent mean-pool loopwise scoring was much worse on dev tasks.

## 2026-06-20 Batch 005 Dev Standalone Exploration

- Parent decision: continue with exactly one dev-only standalone batch using existing checkpoints that were not covered by `batch_004_dev`; do not use final-task results to tune the next candidate.
- Created `experiments/batches/batch_005_dev.yaml`.
- Batch purpose: `dev`.
- Candidate tracks: two `standalone_main` exploration candidates plus one `diagnostic` candidate. These dev results cannot trigger `main_goal_success`.
- Dev tasks:
  - `SciFact`
  - `NFCorpus`
  - `FiQA2018`
  - `SCIDOCS`
- Predeclared standalone candidates:
  - `r005_standalone_loop_final_t10`: existing `loop_final_mean_pool` checkpoint, fixed `loop_idx=10`.
  - `r005_standalone_loop_final_recurrent_t10`: existing `loop_final_recurrent_mean_pool` checkpoint, fixed `loop_idx=10`.
  - `r005_standalone_loop_final_recurrent_no_memory_t10`: existing `loop_final_recurrent_no_memory` checkpoint, fixed `loop_idx=10`.
  - `r005_standalone_matryoshka_recurrent_no_memory_t3`: existing `loop_matryoshka_recurrent_no_memory` checkpoint, fixed `loop_idx=3`.
- Guardrails:
  - No training.
  - No new retrieval method or architecture change.
  - No frozen-standard checkpoint, frozen-standard score, `fusion_standard_checkpoint_dir`, `fusion_alpha`, `fusion_scope`, weighted standard+loop concatenation, or score interpolation in candidate scoring.
  - Candidate rule is global per run across all dev tasks.
- Checks before submission:
  - `source scripts/slurm_env.sh && "$PYTHON_BIN" -m compileall src scripts` passed.
  - `bash -n scripts/*.sh scripts/*.sbatch` passed.
  - `source scripts/slurm_env.sh && "$PYTHON_BIN" scripts/goal_validate_manifest.py experiments/batches/batch_005_dev.yaml` passed with expected dev-only standalone warnings.
  - `source scripts/slurm_env.sh && "$PYTHON_BIN" scripts/goal_submit_batch.py experiments/batches/batch_005_dev.yaml --dry-run --submit-postprocess` passed.
  - `source scripts/slurm_env.sh && "$PYTHON_BIN" scripts/goal_preflight.py --manifest experiments/batches/batch_005_dev.yaml` passed.
  - A first real submit without scheduler env failed before any job was submitted because Slurm required a partition option.
  - A second real submit inside the sandbox reached Slurm but failed before any job was submitted with `Operation not permitted` on the Slurm stream socket.
  - The final submit was rerun outside the sandbox through the same `scripts/goal_submit_batch.py --submit --submit-postprocess` wrapper.
- Submission:
  - Submitted only through `scripts/goal_submit_batch.py --submit --submit-postprocess`.
  - Eval job IDs:
    - `r005_standalone_loop_final_t10`: `4989172`
    - `r005_standalone_loop_final_recurrent_t10`: `4989173`
    - `r005_standalone_loop_final_recurrent_no_memory_t10`: `4989174`
    - `r005_standalone_matryoshka_recurrent_no_memory_t3`: `4989175`
  - Postprocess job ID: `4989176`
  - Postprocess dependency: `afterany:4989172:4989173:4989174:4989175`
- Next action: wait for Slurm-native postprocess, then inspect `outputs/goal/runs/batch_005_dev/scoreboard.json`.

## 2026-06-20 Batch 005 Dev Standalone Result

- `batch_005_dev` completed Slurm-native postprocess:
  - eval jobs: `4989172`, `4989173`, `4989174`, `4989175`
  - postprocess job: `4989176`
  - marker: `outputs/goal/runs/batch_005_dev/postprocess_done.json`
  - scoreboard: `outputs/goal/runs/batch_005_dev/scoreboard.json`
- A local `goal_status.py --batch-id batch_005_dev --update-state` refresh attempt hung in Slurm status querying. The postprocess marker and scoreboard are the terminal evidence.
- Batch purpose: `dev`; all candidates are `standalone_main` exploration only, so this batch cannot trigger `main_goal_success`.
- Scoreboard interpretation under the current claim-track policy:
  - `r005_standalone_loop_final_t10__loop10`: track `standalone_main`, `minimal_positive_signal=false`, `fusion_diagnostic_pass=false`, `research_grade_threshold_pass=false`, `main_goal_success=false`, `publishable_score_candidate=false`, min delta `-0.00598`, mean delta `-0.00306`, tasks won/lost `0/7`, valid tasks `4/7`.
    - Dev deltas: `SciFact -0.00560`, `NFCorpus -0.00598`, `SCIDOCS -0.00130`, `FiQA2018 +0.00063`; `ArguAna`, `Touche2020`, and `TRECCOVID` were not evaluated by design.
  - `r005_standalone_matryoshka_recurrent_no_memory_t3__loop3`: track `standalone_main`, `minimal_positive_signal=false`, `fusion_diagnostic_pass=false`, `research_grade_threshold_pass=false`, `main_goal_success=false`, `publishable_score_candidate=false`, min delta `-0.39494`, mean delta `-0.21950`, tasks won/lost `0/7`, valid tasks `4/7`.
    - Dev deltas: `SciFact -0.39494`, `NFCorpus -0.15812`, `SCIDOCS -0.11502`, `FiQA2018 -0.20991`; three final-only tasks were not evaluated by design.
  - `r005_standalone_loop_final_recurrent_no_memory_t10__loop10`: track `standalone_main`, `minimal_positive_signal=false`, `fusion_diagnostic_pass=false`, `research_grade_threshold_pass=false`, `main_goal_success=false`, `publishable_score_candidate=false`, min delta `-0.53827`, mean delta `-0.29136`, tasks won/lost `0/7`, valid tasks `4/7`.
    - Dev deltas: `SciFact -0.53827`, `NFCorpus -0.23201`, `SCIDOCS -0.13661`, `FiQA2018 -0.25855`; three final-only tasks were not evaluated by design.
  - `r005_standalone_loop_final_recurrent_t10__loop10`: track `standalone_main`, `minimal_positive_signal=false`, `fusion_diagnostic_pass=false`, `research_grade_threshold_pass=false`, `main_goal_success=false`, `publishable_score_candidate=false`, min delta `-0.53932`, mean delta `-0.29072`, tasks won/lost `0/7`, valid tasks `4/7`.
    - Dev deltas: `SciFact -0.53932`, `NFCorpus -0.22870`, `SCIDOCS -0.13630`, `FiQA2018 -0.25855`; three final-only tasks were not evaluated by design.
- Decision: `main_goal_success=false`. Existing final-loop and recurrent no-memory standalone probes did not show a viable dev signal.

## 2026-06-20 Batch 006 Dev Standalone Exploration

- Parent decision: continue with exactly one dev-only standalone batch by training two already-supported, newly registered memory-mode variants rather than changing metric semantics or using frozen-standard fusion.
- Dev-only evidence used for the next direction:
  - Existing standalone checkpoint summaries on the four dev tasks showed all evaluated mean-pool/recurrent variants below the frozen standard on average.
  - The fixed evaluation loop `7` was chosen from dev-only historical loopwise mean-pool summary as the best global standalone loop depth on the dev tasks; no final-task deltas were used for this choice.
- Code/config updates:
  - Registered `loop_matryoshka_first_token` in `src/experiments.py`.
  - Registered `loop_matryoshka_token_concat` in `src/experiments.py`.
  - Created `experiments/batches/batch_006_dev.yaml`.
- Batch purpose: `dev`.
- Candidate track: `standalone_main` exploration only. These dev results cannot trigger `main_goal_success`.
- Predeclared candidates:
  - `r006_loop_matryoshka_first_token_t7`: train `loop_matryoshka_first_token`, evaluate fixed `loop_idx=7` on all dev tasks.
  - `r006_loop_matryoshka_token_concat_t7`: train `loop_matryoshka_token_concat`, evaluate fixed `loop_idx=7` on all dev tasks.
- Guardrails:
  - Existing loop-memory modes only; no projection heads, gates, learned scaling, metric change, baseline change, final-task set change, or frozen-standard scoring input.
  - Candidate rule is global per run across all dev tasks.
- Checks before submission:
  - `source scripts/slurm_env.sh && "$PYTHON_BIN" -m compileall src scripts` passed.
  - `bash -n scripts/*.sh scripts/*.sbatch` passed.
  - `source scripts/slurm_env.sh && "$PYTHON_BIN" scripts/goal_validate_manifest.py experiments/batches/batch_006_dev.yaml` passed with expected dev-only standalone warnings.
  - `source scripts/slurm_env.sh && "$PYTHON_BIN" scripts/goal_submit_batch.py experiments/batches/batch_006_dev.yaml --dry-run --submit-postprocess` passed.
  - `source scripts/slurm_env.sh && "$PYTHON_BIN" scripts/goal_preflight.py --manifest experiments/batches/batch_006_dev.yaml` passed.
  - A scheduler-argument dry-run with `SBATCH_ARGS='--account=p32737 --partition=gengpu'` and `POSTPROCESS_SBATCH_ARGS='--account=p32737 --partition=short'` passed.
- Submission:
  - Submitted only through `scripts/goal_submit_batch.py --submit --submit-postprocess`.
  - Train/eval job IDs:
    - `r006_loop_matryoshka_first_token_t7`: train `4991224`, eval `4991226`
    - `r006_loop_matryoshka_token_concat_t7`: train `4991227`, eval `4991228`
  - Postprocess job ID: `4991230`
  - Postprocess dependency: `afterany:4991226:4991228`
- Next action: wait for Slurm-native postprocess, then inspect `outputs/goal/runs/batch_006_dev/scoreboard.json`.

## 2026-06-20 Batch 006 Dev Standalone Result

- `batch_006_dev` completed Slurm-native postprocess:
  - train/eval jobs:
    - `r006_loop_matryoshka_first_token_t7`: train `4991224`, eval `4991226`
    - `r006_loop_matryoshka_token_concat_t7`: train `4991227`, eval `4991228`
  - postprocess job: `4991230`
  - marker: `outputs/goal/runs/batch_006_dev/postprocess_done.json`
  - scoreboard: `outputs/goal/runs/batch_006_dev/scoreboard.json`
- A local `goal_status.py --batch-id batch_006_dev --update-state` refresh attempt hung in Slurm `squeue` status querying and was terminated. The postprocess marker and scoreboard are the terminal evidence.
- Batch purpose: `dev`; all candidates are `standalone_main` exploration only, so this batch cannot trigger `main_goal_success`.
- Scoreboard interpretation under the current claim-track policy:
  - `r006_loop_matryoshka_first_token_t7__loop7`: track `standalone_main`, `minimal_positive_signal=false`, `fusion_diagnostic_pass=false`, `research_grade_threshold_pass=false`, `main_goal_success=false`, `publishable_score_candidate=false`, min delta `-0.01126`, mean delta `-0.00112`, tasks won/lost `2/5`, valid tasks `4/7`.
    - Dev deltas: `SciFact +0.00419`, `NFCorpus +0.00359`, `SCIDOCS -0.00102`, `FiQA2018 -0.01126`; `ArguAna`, `Touche2020`, and `TRECCOVID` were not evaluated by design.
  - `r006_loop_matryoshka_token_concat_t7__loop7`: track `standalone_main`, `minimal_positive_signal=false`, `fusion_diagnostic_pass=false`, `research_grade_threshold_pass=false`, `main_goal_success=false`, `publishable_score_candidate=false`, min delta `-0.03403`, mean delta `-0.02160`, tasks won/lost `0/7`, valid tasks `4/7`.
    - Dev deltas: `SciFact -0.03403`, `NFCorpus -0.01153`, `SCIDOCS -0.00831`, `FiQA2018 -0.03253`; three final-only tasks were not evaluated by design.
- Decision: `main_goal_success=false`. The first-token memory variant showed a narrow positive dev signal on `SciFact` and `NFCorpus`, but it regressed on `SCIDOCS` and `FiQA2018`; token-concat regressed on all four dev tasks. Under the current guardrails, the already-supported standalone checkpoint/evaluation probes and the allowed parameter-free memory-mode training probes have not produced a valid standalone dev path to final validation without a new research method or protocol change, so no new batch was submitted.

## 2026-06-20 Batch 007 Dev Standalone Exploration

- Parent decision: after re-auditing the repo, continue with exactly one low-cost dev-only standalone batch rather than changing retrieval semantics or opening a final validation.
- `batch_006_dev` was used only as dev evidence. No final-task deltas were used to tune this batch.
- Created `experiments/batches/batch_007_dev.yaml`.
- Batch purpose: `dev`.
- Candidate track: `standalone_main` exploration only. These dev results cannot trigger `main_goal_success`.
- Predeclared candidates:
  - `r007_first_token_t5`: eval-only scoring from the `batch_006_dev` first-token checkpoint at fixed `loop_idx=5`.
  - `r007_first_token_t6`: eval-only scoring from the same checkpoint at fixed `loop_idx=6`.
  - `r007_first_token_t8`: eval-only scoring from the same checkpoint at fixed `loop_idx=8`.
  - `r007_first_token_t9`: eval-only scoring from the same checkpoint at fixed `loop_idx=9`.
- Guardrails:
  - Dev tasks only: `SciFact`, `NFCorpus`, `FiQA2018`, `SCIDOCS`.
  - No training.
  - No frozen-standard checkpoint, frozen-standard score, `fusion_standard_checkpoint_dir`, `fusion_alpha`, `fusion_scope`, weighted standard+loop concatenation, or standard score interpolation in candidate scoring.
  - Candidate rule is global per run across all dev tasks.
- Checks before submission:
  - `source scripts/slurm_env.sh && "$PYTHON_BIN" -m compileall src scripts` passed.
  - `bash -n scripts/*.sh scripts/*.sbatch` passed.
  - `source scripts/slurm_env.sh && "$PYTHON_BIN" scripts/goal_validate_manifest.py experiments/batches/batch_007_dev.yaml` passed with expected dev-only standalone warnings.
  - `source scripts/slurm_env.sh && "$PYTHON_BIN" scripts/goal_submit_batch.py experiments/batches/batch_007_dev.yaml --dry-run --submit-postprocess` passed.
  - `source scripts/slurm_env.sh && "$PYTHON_BIN" scripts/goal_preflight.py --manifest experiments/batches/batch_007_dev.yaml` passed.
- Submission:
  - Submitted only through `scripts/goal_submit_batch.py --submit --submit-postprocess`.
  - Eval job IDs:
    - `r007_first_token_t5`: `5034376`
    - `r007_first_token_t6`: `5034377`
    - `r007_first_token_t8`: `5034378`
    - `r007_first_token_t9`: `5034379`
  - Postprocess job ID: `5034380`
  - Postprocess dependency: `afterany:5034376:5034377:5034378:5034379`
- Next action: wait for Slurm-native postprocess, then inspect `outputs/goal/runs/batch_007_dev/scoreboard.json`.

## 2026-06-20 Research Design Discipline Protocol Update

- Protocol-only documentation change made while `batch_007_dev` may still be pending or running.
- No Slurm jobs were submitted, cancelled, restarted, or resubmitted.
- No training, evaluation, collection, scoring, or postprocess command was run.
- No batch manifest was created or modified, including `experiments/batches/batch_007_dev.yaml`.
- No results, scoreboards, raw MTEB outputs, checkpoints, Slurm logs, frozen baselines, or `outputs/goal/state.json` were modified by this protocol update.
- Updated `AGENTS.md` and `docs/goal_protocol.md` to require local-search exhaustion detection, `RESEARCH_DESIGN_MODE`, and portfolio-style dev batch design after repeated `standalone_main` failures.
- Updated `README.md` with a short pointer to the same autonomous exploration discipline.
- Next action remains unchanged: wait for `batch_007_dev` Slurm-native postprocess, analyze its scoreboard, and only then decide whether the result triggers `RESEARCH_DESIGN_MODE`.

## 2026-06-20 Batch 007 Dev Standalone Result

- `batch_007_dev` completed Slurm-native postprocess:
  - eval jobs: `5034376`, `5034377`, `5034378`, `5034379`
  - postprocess job: `5034380`
  - marker: `outputs/goal/runs/batch_007_dev/postprocess_done.json`
  - scoreboard: `outputs/goal/runs/batch_007_dev/scoreboard.json`
- A local `goal_status.py --batch-id batch_007_dev --update-state` refresh attempt hung in Slurm status querying and was terminated. The postprocess marker and scoreboard are the terminal evidence.
- Batch purpose: `dev`; all candidates are `standalone_main` exploration only, so this batch cannot trigger `main_goal_success`.
- Scoreboard interpretation under the current claim-track policy:
  - `r007_first_token_t8__loop8`: track `standalone_main`, `minimal_positive_signal=false`, `fusion_diagnostic_pass=false`, `research_grade_threshold_pass=false`, `main_goal_success=false`, `publishable_score_candidate=false`, dev min delta `-0.00710`, dev mean delta `+0.00056`, dev tasks won/lost `2/2`.
    - Dev deltas: `SciFact +0.00664`, `NFCorpus +0.00310`, `FiQA2018 -0.00710`, `SCIDOCS -0.00040`.
  - `r007_first_token_t6__loop6`: track `standalone_main`, all success flags false, dev min delta `-0.00791`, dev mean delta `+0.00047`, dev tasks won/lost `2/2`.
    - Dev deltas: `SciFact +0.00663`, `NFCorpus +0.00329`, `FiQA2018 -0.00791`, `SCIDOCS -0.00013`.
  - `r007_first_token_t9__loop9`: track `standalone_main`, all success flags false, dev min delta `-0.00910`, dev mean delta `-0.00032`, dev tasks won/lost `2/2`.
    - Dev deltas: `SciFact +0.00621`, `NFCorpus +0.00317`, `FiQA2018 -0.00910`, `SCIDOCS -0.00156`.
  - `r007_first_token_t5__loop5`: track `standalone_main`, all success flags false, dev min delta `-0.00958`, dev mean delta `-0.00062`, dev tasks won/lost `2/2`.
    - Dev deltas: `SciFact +0.00442`, `NFCorpus +0.00338`, `FiQA2018 -0.00958`, `SCIDOCS -0.00071`.
- Decision: `main_goal_success=false`. `batch_007_dev` was a local-neighborhood loop-depth sweep and did not produce a viable global dev signal.

## 2026-06-20 Research Design Round 001 And Batch 008 Portfolio

- Local search is exhausted under the current protocol:
  - recent standalone_main dev batches `batch_004_dev`, `batch_005_dev`, `batch_006_dev`, and `batch_007_dev` did not produce a viable global dev signal;
  - recent work mainly evaluated existing checkpoints, memory modes, and nearby loop depths;
  - the recurring standalone pattern is SciFact/NFCorpus gains with FiQA2018 and SCIDOCS regressions.
- Entered `RESEARCH_DESIGN_MODE` before creating a new batch.
- Wrote the research plan at `docs/research_design_round_001.md`.
- Broad standalone directions considered:
  - final-loop objective with first-token memory;
  - detached first-token memory training;
  - short-horizon first-token loopwise training;
  - single-loop standalone control;
  - document-side loop symmetry;
  - hard-negative curriculum or sampling change;
  - conservative evaluation-time loop normalization.
- Selected one portfolio-style dev batch: `experiments/batches/batch_008_dev.yaml`.
- Batch purpose: `dev`.
- Candidate track: `standalone_main` exploration only. These dev results cannot trigger `main_goal_success`.
- Portfolio candidates:
  - `r008_final_first_token_t10`: final-loop first-token objective, fixed `loop_idx=10`.
  - `r008_detached_first_token_t10`: detached first-token loopwise training, fixed `loop_idx=10`.
  - `r008_short_first_token_t4`: first-token loopwise training with `tmax=4`, fixed `loop_idx=4`.
  - `r008_single_loop_t1`: one-loop standalone control, fixed `loop_idx=1`.
- Portfolio rationale:
  - tests objective, recurrence-gradient behavior, training horizon, and recurrence control;
  - does not reuse the same failed checkpoint at neighboring loop depths;
  - uses only dev tasks: `SciFact`, `NFCorpus`, `FiQA2018`, and `SCIDOCS`;
  - keeps one global candidate rule per run across all dev tasks;
  - uses no frozen-baseline checkpoint, baseline ensemble, baseline concatenation, or interpolation in candidate scoring.
- Estimated GPU budget: 24 GPU hours against the configured 24 GPU-hour limit, with at most 4 concurrent GPU jobs.
- Checks before submission:
  - `source scripts/slurm_env.sh && "$PYTHON_BIN" -m compileall src scripts` passed.
  - `bash -n scripts/*.sh scripts/*.sbatch` passed.
  - `git diff --check` passed.
  - `source scripts/slurm_env.sh && "$PYTHON_BIN" scripts/goal_validate_manifest.py experiments/batches/batch_008_dev.yaml` passed with expected dev-only standalone warnings.
  - `source scripts/slurm_env.sh && "$PYTHON_BIN" scripts/goal_submit_batch.py experiments/batches/batch_008_dev.yaml --dry-run --submit-postprocess` passed.
  - `source scripts/slurm_env.sh && "$PYTHON_BIN" scripts/goal_preflight.py --manifest experiments/batches/batch_008_dev.yaml` passed.
- Submission:
  - Submitted only through `scripts/goal_submit_batch.py --submit --submit-postprocess`.
  - Train/eval job IDs:
    - `r008_final_first_token_t10`: train `5035132`, eval `5035133`
    - `r008_detached_first_token_t10`: train `5035134`, eval `5035135`
    - `r008_short_first_token_t4`: train `5035136`, eval `5035137`
    - `r008_single_loop_t1`: train `5035138`, eval `5035139`
  - Postprocess job ID: `5035140`
  - Postprocess dependency: `afterany:5035133:5035135:5035137:5035139`
- Next action: wait for Slurm-native postprocess, then inspect `outputs/goal/runs/batch_008_dev/scoreboard.json`.

## 2026-06-21 Batch 008 Dev Standalone Result

- `batch_008_dev` completed Slurm-native postprocess:
  - train/eval jobs:
    - `r008_final_first_token_t10`: train `5035132`, eval `5035133`
    - `r008_detached_first_token_t10`: train `5035134`, eval `5035135`
    - `r008_short_first_token_t4`: train `5035136`, eval `5035137`
    - `r008_single_loop_t1`: train `5035138`, eval `5035139`
  - postprocess job: `5035140`
  - marker: `outputs/goal/runs/batch_008_dev/postprocess_done.json`
  - scoreboard: `outputs/goal/runs/batch_008_dev/scoreboard.json`
- A local `goal_status.py --batch-id batch_008_dev --update-state` refresh attempt hung in Slurm `squeue` querying and was terminated. The postprocess marker and scoreboard are the terminal evidence.
- Batch purpose: `dev`; all candidates are `standalone_main` exploration only, so this batch cannot trigger `main_goal_success`.
- Scoreboard interpretation under the current claim-track policy:
  - `r008_detached_first_token_t10__loop10`: track `standalone_main`, `minimal_positive_signal=false`, `fusion_diagnostic_pass=false`, `research_grade_threshold_pass=false`, `main_goal_success=false`, `publishable_score_candidate=false`, dev min delta `-0.03572`, dev mean delta `-0.02551`, dev tasks won/lost `0/4`.
    - Dev deltas: `SciFact -0.03503`, `NFCorpus -0.03572`, `FiQA2018 -0.02194`, `SCIDOCS -0.00935`.
  - `r008_final_first_token_t10__loop10`: track `standalone_main`, all success flags false, dev min delta `-0.04031`, dev mean delta `-0.03250`, dev tasks won/lost `0/4`.
    - Dev deltas: `SciFact -0.03109`, `NFCorpus -0.03838`, `FiQA2018 -0.04031`, `SCIDOCS -0.02023`.
  - `r008_short_first_token_t4__loop4`: track `standalone_main`, all success flags false, dev min delta `-0.04657`, dev mean delta `-0.03780`, dev tasks won/lost `0/4`.
    - Dev deltas: `SciFact -0.04371`, `NFCorpus -0.04338`, `FiQA2018 -0.04657`, `SCIDOCS -0.01753`.
  - `r008_single_loop_t1__loop1`: track `standalone_main`, all success flags false, dev min delta `-0.05210`, dev mean delta `-0.03594`, dev tasks won/lost `0/4`.
    - Dev deltas: `SciFact -0.05210`, `NFCorpus -0.03617`, `FiQA2018 -0.03969`, `SCIDOCS -0.01579`.
- Decision: `main_goal_success=false`. `batch_008_dev` did not produce a viable global dev signal. Training logs show these candidates stopped at about `0.32` epoch, so the negative result also tests a capped-budget variant rather than a full-epoch replacement for `batch_006_dev`.

## 2026-06-21 Research Design Round 002 And Batch 009 Portfolio

- Local search remains exhausted:
  - `batch_004_dev` through `batch_008_dev` did not produce a viable standalone global dev signal;
  - local first-token loop-depth sweeps already failed;
  - the latest portfolio failed broadly on all dev tasks.
- Entered `RESEARCH_DESIGN_MODE` again before creating a new batch.
- Wrote the research plan at `docs/research_design_round_002.md`.
- Broad standalone directions considered:
  - document-side loop symmetry;
  - full-epoch detached first-token training;
  - lower hard-negative pressure;
  - loop-depth dropout during training;
  - query-side residual mixing without frozen baseline;
  - conservative corpus normalization;
  - data curriculum.
- Code/config updates:
  - Added an opt-in document-loop evaluation path. Default document encoding remains one-pass.
  - Added `eval.loop_docs` and `eval.doc_loop_idx` manifest/export support.
  - Created `configs/goal_batch_009_detached_first_token_full.yaml`.
  - Created `configs/goal_batch_009_first_token_neg3.yaml`.
  - Created `experiments/batches/batch_009_dev.yaml`.
- Batch purpose: `dev`.
- Candidate track: `standalone_main` exploration only. These dev results cannot trigger `main_goal_success`.
- Portfolio candidates:
  - `r009_docloop_first_token_t7`: eval-only document-loop symmetry using the `batch_006_dev` first-token checkpoint, fixed query/document loop index `7`.
  - `r009_detached_first_token_full_t10`: full-epoch detached first-token loopwise training, fixed `loop_idx=10`.
  - `r009_first_token_neg3_t10`: full-epoch first-token loopwise training with `num_negatives=3`, fixed `loop_idx=10`.
- Portfolio rationale:
  - tests query/document representation symmetry, full-budget detached-memory training, and hard-negative pressure;
  - does not retest neighboring loop depths of the same checkpoint;
  - uses only dev tasks: `SciFact`, `NFCorpus`, `FiQA2018`, and `SCIDOCS`;
  - keeps one global candidate rule per run across all dev tasks;
  - uses no frozen-baseline checkpoint, baseline ensemble, baseline concatenation, or interpolation in candidate scoring.
- Estimated GPU budget: 24 GPU hours against the configured 24 GPU-hour limit, with at most 3 concurrent GPU jobs.

## 2026-06-21 Batch 011 Dev Standalone Result

- `batch_011_dev` completed Slurm-native postprocess:
  - train/eval jobs:
    - `r011_inbatch_hybrid_first_token_t10`: train `5092527`, eval `5092528`
    - `r011_pairwise_rank_first_token_t10`: train `5092529`, eval `5092530`
  - postprocess job: `5092531`
  - marker: `outputs/goal/runs/batch_011_dev/postprocess_done.json`
  - scoreboard: `outputs/goal/runs/batch_011_dev/scoreboard.json`
- There is no `postprocess_failed.json`.
- A local `goal_status.py --batch-id batch_011_dev --update-state` refresh attempt hung in Slurm status querying. The postprocess marker, scoreboard, and subagent workflow audit were used as terminal evidence.
- Batch purpose: `dev`; both candidates are `standalone_main` exploration only, so this batch cannot trigger `main_goal_success`.
- Scoreboard interpretation under the current claim-track policy:
  - `r011_inbatch_hybrid_first_token_t10__loop10`: track `standalone_main`, `minimal_positive_signal=false`, `fusion_diagnostic_pass=false`, `research_grade_threshold_pass=false`, `main_goal_success=false`, `publishable_score_candidate=false`, dev min delta `-0.01287`, dev mean delta `-0.006275`, dev tasks won/lost `0/4`.
    - Dev deltas: `SciFact -0.00511`, `NFCorpus -0.00648`, `FiQA2018 -0.01287`, `SCIDOCS -0.00064`.
  - `r011_pairwise_rank_first_token_t10__loop10`: track `standalone_main`, all success flags false, dev min delta `-0.03389`, dev mean delta `-0.02554`, dev tasks won/lost `0/4`.
    - Dev deltas: `SciFact -0.03389`, `NFCorpus -0.02759`, `FiQA2018 -0.02739`, `SCIDOCS -0.01329`.
- Decision: `main_goal_success=false`. Neither standalone objective candidate produced a viable global dev signal; both regressed on every completed dev task.
- Local search remains exhausted. Recent evidence has now falsified first-token loop-depth tuning, memory-mode variants, detached-memory training, lower negative count, document-loop symmetry, loop-loss tail weighting, adjacent-loop consistency, candidate-internal self-residual stabilization, in-batch hybrid contrastive loss, and pairwise ranking loss.

## 2026-06-21 Batch 010 Dev Mixed Portfolio Result

- `batch_010_dev` completed Slurm-native postprocess:
  - train/eval jobs:
    - `r010_tail_weighted_first_token_t10`: train `5076613`, eval `5076614`
    - `r010_consistency_first_token_t10`: train `5076615`, eval `5076616`
    - `r010_self_residual_first_token_t7_a50`: eval `5076617`
  - postprocess job: `5076618`
  - marker: `outputs/goal/runs/batch_010_dev/postprocess_done.json`
  - scoreboard: `outputs/goal/runs/batch_010_dev/scoreboard.json`
- There is no `postprocess_failed.json`.
- A local `goal_status.py --batch-id batch_010_dev --update-state` refresh attempt hung without output in Slurm status querying. The postprocess marker, scoreboard, and subagent workflow audit were used as terminal evidence.
- Batch purpose: `dev`; these results cannot trigger `main_goal_success`.
- Scoreboard interpretation under the current claim-track policy:
  - `r010_tail_weighted_first_token_t10__loop10`: track `standalone_main`, `minimal_positive_signal=false`, `fusion_diagnostic_pass=false`, `research_grade_threshold_pass=false`, `main_goal_success=false`, `publishable_score_candidate=false`, dev min delta `-0.01743`, dev mean delta `-0.01095`, dev tasks won/lost `0/4`.
    - Dev deltas: `SciFact -0.01743`, `NFCorpus -0.00848`, `FiQA2018 -0.01491`, `SCIDOCS -0.00298`.
  - `r010_consistency_first_token_t10__loop10`: track `standalone_main`, all success flags false, dev min delta `-0.03741`, dev mean delta `-0.0209075`, dev tasks won/lost `0/4`.
    - Dev deltas: `SciFact -0.03741`, `NFCorpus -0.01396`, `FiQA2018 -0.02648`, `SCIDOCS -0.00578`.
  - `r010_self_residual_first_token_t7_a50__loop7`: track `diagnostic`, all success flags false, dev min delta `-0.01115`, dev mean delta `-0.001125`, dev tasks won/lost `2/2`.
    - Dev deltas: `SciFact +0.00396`, `NFCorpus +0.00371`, `FiQA2018 -0.01115`, `SCIDOCS -0.00102`.
- Decision: `main_goal_success=false`. The two `standalone_main` candidates did not produce a viable global dev signal; the diagnostic self-residual probe remains non-main and non-global.
- Local search remains exhausted. Recent evidence has now falsified local loop-depth tuning, memory-mode variants, detached-memory training, lower negative count, document-loop symmetry, loop-loss tail weighting, adjacent-loop consistency, and candidate-internal self-residual stabilization.

## 2026-06-21 Research Design Round 004 And Batch 011 Portfolio

- Entered `RESEARCH_DESIGN_MODE` again before validating, dry-running, preflighting, or submitting any new batch.
- Real subagents were used for parallel read-only gates:
  - one analyzed `batch_010_dev` under claim-track policy;
  - one audited workflow state, postprocess completion, and next batch id;
  - one assessed local-search exhaustion and proposed broader standalone directions.
- Wrote the research plan at `docs/research_design_round_004.md`.
- Broad standalone directions considered:
  - true in-batch hybrid contrastive loss;
  - pairwise/listwise ranking loss;
  - seeded random positive/negative sampling;
  - query/document loop co-training;
  - loop-depth dropout or sparse loop supervision;
  - hardness curriculum;
  - parameter-free pooling alternative.
- Code/config updates prepared before validation:
  - Added registered versions `loop_inbatch_hybrid_first_token` and `loop_pairwise_first_token`.
  - Added candidate-only in-batch hybrid loss support and loopwise pairwise ranking loss support.
  - Added Slurm manifest/export support for `inbatch_weight` and `pairwise_margin`.
  - Added `configs/goal_batch_011_inbatch_hybrid_first_token.yaml`.
  - Added `configs/goal_batch_011_pairwise_first_token.yaml`.
  - Created `experiments/batches/batch_011_dev.yaml`.
- Batch purpose: `dev`.
- Candidate track: `standalone_main` exploration only. These dev results cannot trigger `main_goal_success`.
- Portfolio candidates:
  - `r011_inbatch_hybrid_first_token_t10`: first-token loopwise training with hard negatives plus candidate-only in-batch contrastive loss, `inbatch_weight=0.25`, fixed `loop_idx=10`.
  - `r011_pairwise_rank_first_token_t10`: first-token loopwise training with pairwise softplus ranking loss, `pairwise_margin=0.0`, fixed `loop_idx=10`.
- Portfolio rationale:
  - tests two broader retrieval training objectives after loop-local search failure;
  - does not retest neighboring loop depths, memory modes, or the same failed checkpoint;
  - uses only dev tasks: `SciFact`, `NFCorpus`, `FiQA2018`, and `SCIDOCS`;
  - keeps one global candidate rule per run across all dev tasks;
  - uses no frozen-baseline checkpoint, baseline ensemble, baseline concatenation, or interpolation in candidate scoring.
- Estimated GPU budget: 20 GPU hours against the configured 24 GPU-hour limit, with at most 2 concurrent GPU jobs.
- Checks before submission:
  - Research-design subagent gate passed.
  - Code/protocol subagent gate passed and confirmed no existing no-code mechanism was suitable for a non-local `batch_014_dev`.
  - `source scripts/slurm_env.sh && "$PYTHON_BIN" -m compileall src scripts` passed.
  - `bash -n scripts/*.sh scripts/*.sbatch` passed.
  - `git diff --check` passed.
  - `source scripts/slurm_env.sh && "$PYTHON_BIN" scripts/goal_validate_manifest.py experiments/batches/batch_014_dev.yaml` passed with expected dev-only standalone warnings.
  - `source scripts/slurm_env.sh && "$PYTHON_BIN" scripts/goal_submit_batch.py experiments/batches/batch_014_dev.yaml --dry-run --submit-postprocess` passed.
  - `source scripts/slurm_env.sh && "$PYTHON_BIN" scripts/goal_preflight.py --manifest experiments/batches/batch_014_dev.yaml` passed.
- Submission:
  - Submitted only through `scripts/goal_submit_batch.py --submit --submit-postprocess`.
  - The sandboxed submit attempt with site scheduler options failed before job creation because Slurm controller socket access is blocked inside the sandbox.
  - The approved external retry through the same `goal_submit_batch.py` command succeeded with site scheduler options provided only through temporary environment variables.
  - Train/eval job IDs:
    - `r014_sparse_late_first_token_t10`: train `5196833`, eval `5196834`
    - `r014_label_smooth_first_token_t10`: train `5196835`, eval `5196836`
  - Postprocess job ID: `5196837`
  - Postprocess dependency: `afterany:5196834:5196836`
- Next action: wait for Slurm-native postprocess, then inspect `outputs/goal/runs/batch_014_dev/scoreboard.json`.
- Checks before submission:
  - Code-risk subagent gate completed; it found one misplaced notebook block, which was corrected before validation.
  - `source scripts/slurm_env.sh && "$PYTHON_BIN" -m compileall src scripts` passed.
  - `bash -n scripts/*.sh scripts/*.sbatch` passed.
  - `git diff --check` passed.
  - Initial manifest validation failed only because a negative prose phrase contained a forbidden standalone audit token; the manifest wording was corrected without changing candidate rules.
  - `source scripts/slurm_env.sh && "$PYTHON_BIN" scripts/goal_validate_manifest.py experiments/batches/batch_011_dev.yaml` passed with expected dev-only standalone warnings.
  - `source scripts/slurm_env.sh && "$PYTHON_BIN" scripts/goal_submit_batch.py experiments/batches/batch_011_dev.yaml --dry-run --submit-postprocess` passed.
  - `source scripts/slurm_env.sh && "$PYTHON_BIN" scripts/goal_preflight.py --manifest experiments/batches/batch_011_dev.yaml` passed.
- Submission:
  - Submitted only through `scripts/goal_submit_batch.py --submit --submit-postprocess`.
  - The first submit attempt failed before job creation because the cluster required a scheduler partition.
  - The second submit attempt failed before job creation because the cluster also required a scheduler account.
  - The approved external retry through the same `goal_submit_batch.py` command succeeded with site scheduler arguments provided only through temporary environment variables.
  - Train/eval job IDs:
    - `r011_inbatch_hybrid_first_token_t10`: train `5092527`, eval `5092528`
    - `r011_pairwise_rank_first_token_t10`: train `5092529`, eval `5092530`
  - Postprocess job ID: `5092531`
  - Postprocess dependency: `afterany:5092528:5092530`
- Next action: wait for Slurm-native postprocess, then inspect `outputs/goal/runs/batch_011_dev/scoreboard.json`.

## 2026-06-21 Batch 009 Dev Standalone Result And Repair

- `batch_009_dev` completed Slurm-native postprocess:
  - eval/train jobs:
    - `r009_docloop_first_token_t7`: eval `5040674`
    - `r009_detached_first_token_full_t10`: train `5040675`, eval `5040676`
    - `r009_first_token_neg3_t10`: train `5040677`, eval `5040678`
  - postprocess job: `5040679`
  - marker: `outputs/goal/runs/batch_009_dev/postprocess_done.json`
  - scoreboard: `outputs/goal/runs/batch_009_dev/scoreboard.json`
- There is no `postprocess_failed.json`. A local `goal_status.py --batch-id batch_009_dev --update-state` refresh hung while querying `squeue` for old eval job `5040674` and was terminated; the postprocess marker and scoreboard are the terminal evidence.
- Batch purpose: `dev`; all candidates are `standalone_main` exploration only, so this batch cannot trigger `main_goal_success`.
- Scoreboard interpretation under the current claim-track policy:
  - `r009_detached_first_token_full_t10__loop10`: track `standalone_main`, `minimal_positive_signal=false`, `fusion_diagnostic_pass=false`, `research_grade_threshold_pass=false`, `main_goal_success=false`, `publishable_score_candidate=false`, dev min delta `-0.01024`, dev mean delta `-0.00458`, dev tasks won/lost `0/4`.
    - Dev deltas: `SciFact -0.01024`, `NFCorpus -0.00269`, `FiQA2018 -0.00570`, `SCIDOCS +0.00030`.
  - `r009_first_token_neg3_t10__loop10`: track `standalone_main`, all success flags false, dev min delta `-0.02528`, dev mean delta `-0.01549`, dev tasks won/lost `0/4`.
    - Dev deltas: `SciFact -0.01282`, `NFCorpus -0.01521`, `FiQA2018 -0.02528`, `SCIDOCS -0.00866`.
  - `r009_docloop_first_token_t7__loop7`: track `standalone_main`, all success flags false, no valid dev rows, and all four dev tasks are `missing_result`.
    - Eval job `5040674` failed at `NFCorpus` with a Hugging Face datasets cache loader error: multiple cached `mteb/nfcorpus` configurations were present and the loader asked for an explicit configuration.
- Decision: `main_goal_success=false`. The two completed full-budget standalone candidates did not produce a viable global dev signal. The document-loop candidate is not interpretable as a mechanism result because its eval failed before writing `results_summary.csv`.
- Local search remains exhausted. Batch 004 through batch 009 have not produced a standalone global dev signal, and another local loop-depth or memory-mode sweep would be low-value.
- Repair action:
  - Created `experiments/batches/batch_009_dev_repair.yaml`.
  - Purpose: `dev`.
  - Candidate track: `standalone_main`.
  - Candidate: `r009_docloop_first_token_t7_repair`, preserving the original `batch_009_dev` document-loop rule: query loop index `7`, document loop index `7`, same batch_006 first-token checkpoint, same four dev tasks, no frozen-standard scoring input or interpolation.
  - This is a pure eval-only infrastructure repair for a missing result, not a new research batch and not a local-neighborhood sweep.
- Repair budget: estimated 4 GPU hours against the configured 24 GPU-hour batch budget, with one concurrent GPU job.
- Checks before repair submission:
  - `source scripts/slurm_env.sh && "$PYTHON_BIN" -m compileall src scripts` passed.
  - `bash -n scripts/*.sh scripts/*.sbatch` passed.
  - `source scripts/slurm_env.sh && "$PYTHON_BIN" scripts/goal_validate_manifest.py experiments/batches/batch_009_dev_repair.yaml` passed with the expected dev-only standalone warning.
  - `source scripts/slurm_env.sh && "$PYTHON_BIN" scripts/goal_submit_batch.py experiments/batches/batch_009_dev_repair.yaml --dry-run --submit-postprocess` passed.
  - `source scripts/slurm_env.sh && "$PYTHON_BIN" scripts/goal_preflight.py --manifest experiments/batches/batch_009_dev_repair.yaml` passed.
- Submission:
  - Submitted only through `scripts/goal_submit_batch.py --submit --submit-postprocess`.
  - The first submit attempt without site scheduler args failed before job creation because the cluster required `--partition`.
  - A sandboxed retry with scheduler args failed before job creation because Slurm controller socket access is blocked inside the sandbox.
  - The approved external retry through the same `goal_submit_batch.py` command succeeded.
  - Eval job ID: `5043318`
  - Postprocess job ID: `5043319`
  - Postprocess dependency: `afterany:5043318`
- Next action: wait for the repair postprocess, then inspect `outputs/goal/runs/batch_009_dev_repair/scoreboard.json`.

## 2026-06-21 Batch 009 Dev Repair Result

- `batch_009_dev_repair` completed Slurm-native postprocess:
  - eval job: `5043318`
  - postprocess job: `5043319`
  - marker: `outputs/goal/runs/batch_009_dev_repair/postprocess_done.json`
  - scoreboard: `outputs/goal/runs/batch_009_dev_repair/scoreboard.json`
- There is no `postprocess_failed.json`.
- A local `goal_status.py --batch-id batch_009_dev_repair --update-state` refresh hung in Slurm status querying and was terminated. The postprocess marker and scoreboard are the terminal evidence.
- Batch purpose: `dev`; the candidate is `standalone_main` exploration only, so this batch cannot trigger `main_goal_success`.
- Scoreboard interpretation under the current claim-track policy:
  - `r009_docloop_first_token_t7_repair__loop7`: track `standalone_main`, `minimal_positive_signal=false`, `fusion_diagnostic_pass=false`, `research_grade_threshold_pass=false`, `main_goal_success=false`, `publishable_score_candidate=false`, dev min delta `-0.01126`, dev mean delta `-0.001125`, dev tasks won/lost `2/2`.
    - Dev deltas: `SciFact +0.00419`, `NFCorpus +0.00359`, `FiQA2018 -0.01126`, `SCIDOCS -0.00102`; `ArguAna`, `Touche2020`, and `TRECCOVID` were not evaluated by design.
- Decision: `main_goal_success=false`. The repaired document-loop candidate exactly matched the earlier `batch_006_dev` first-token loop-7 dev pattern, so document-side looping did not recover the recurring `FiQA2018` and `SCIDOCS` regressions.
- Local search remains exhausted. Recent standalone dev evidence has now falsified local loop-depth tuning, token-concat memory, capped final-only/detached/short-horizon variants, full-epoch detached memory, lower hard-negative count, and document-loop symmetry.

## 2026-06-21 Research Design Round 003 And Batch 010 Portfolio

- Entered `RESEARCH_DESIGN_MODE` again before creating a new batch.
- Real subagents were used for parallel read-only analysis:
  - one analyzed `batch_009_dev_repair` and recent standalone dev failure patterns;
  - one inspected code for broader standalone directions;
  - one checked manifest/state constraints and validation requirements.
- Wrote the research plan at `docs/research_design_round_003.md`.
- Broad standalone directions considered:
  - loop-loss tail weighting;
  - candidate-internal loop consistency;
  - candidate self-residual query stabilization;
  - true in-batch negatives;
  - loop-depth dropout during training;
  - random positive/negative sampling;
  - pairwise ranking loss.
- Code/config updates:
  - Added registered versions `loop_tail_weighted_first_token` and `loop_consistency_first_token`.
  - Added loop-loss tail weighting and adjacent-loop consistency loss paths.
  - Added evaluation-only `self_query_alpha` and `self_query_source_loop` support for candidate-internal query residual scoring.
  - Added `configs/goal_batch_010_tail_weighted_first_token.yaml`.
  - Added `configs/goal_batch_010_consistency_first_token.yaml`.
  - Created `experiments/batches/batch_010_dev.yaml`.
- Batch purpose: `dev`.
- Candidate track: `standalone_main` exploration only. These dev results cannot trigger `main_goal_success`.
- Portfolio candidates:
  - `r010_tail_weighted_first_token_t10`: full-epoch first-token training with deeper loop losses weighted by `loop_loss_gamma=1.25`, fixed `loop_idx=10`.
  - `r010_consistency_first_token_t10`: full-epoch first-token training with adjacent-loop consistency penalty `loop_consistency_lambda=0.05`, fixed `loop_idx=10`.
  - `r010_self_residual_first_token_t7_a50`: diagnostic eval-only candidate self-residual query scoring from the `batch_006_dev` first-token checkpoint, fixed `loop_idx=7`, `self_query_source_loop=1`, `self_query_alpha=0.50`.
- Portfolio rationale:
  - tests loopwise supervision weighting, explicit loop-drift regularization, and diagnostic candidate-internal query stabilization;
  - does not retest neighboring loop depths of the same checkpoint;
  - uses only dev tasks: `SciFact`, `NFCorpus`, `FiQA2018`, and `SCIDOCS`;
  - keeps one global candidate rule per run across all dev tasks;
  - uses no frozen-baseline checkpoint, baseline ensemble, baseline concatenation, or interpolation in candidate scoring.
- Estimated GPU budget: 24 GPU hours against the configured 24 GPU-hour limit, with at most 3 concurrent GPU jobs.

## 2026-06-21 Research Design Round 005 And Batch 012 Portfolio

- Trigger: `batch_011_dev` completed Slurm-native postprocess with `postprocess_done.json` and no `postprocess_failed.json`.
- Scoreboard conclusion for `batch_011_dev` under the current claim-track policy:
  - `r011_inbatch_hybrid_first_token_t10__loop10`: `standalone_main`, purpose `dev`, dev deltas `SciFact -0.00511`, `NFCorpus -0.00648`, `FiQA2018 -0.01287`, `SCIDOCS -0.00064`, min delta `-0.01287`, mean delta `-0.006275`, dev tasks won/lost `0/4`, all success flags false.
  - `r011_pairwise_rank_first_token_t10__loop10`: `standalone_main`, purpose `dev`, dev deltas `SciFact -0.03389`, `NFCorpus -0.02759`, `FiQA2018 -0.02739`, `SCIDOCS -0.01329`, min delta `-0.03389`, mean delta `-0.02554`, dev tasks won/lost `0/4`, all success flags false.
- Decision: `main_goal_success=false`. This was a dev batch, and both standalone candidates regressed on all four dev tasks.
- Local search remains exhausted. Recent standalone evidence has now falsified first-token loop-depth tuning, memory-mode variants, detached-memory training, lower negative count, document-loop evaluation, loop-loss tail weighting, adjacent-loop consistency, candidate-internal self-residual stabilization, in-batch hybrid contrastive loss, and pairwise ranking loss.
- Real subagents were used for workflow/research gates before new batch validation:
  - one analyzed `batch_011_dev` results under claim-track policy;
  - one audited workflow state and confirmed postprocess completion/no queued jobs;
  - one assessed local-search exhaustion and proposed broader standalone directions.
- Entered `RESEARCH_DESIGN_MODE` before creating the next batch.
- Wrote the research plan at `docs/research_design_round_005.md`.
- Broad standalone directions considered:
  - seeded positive/negative sampling;
  - parameter-free pooling alternative;
  - query/document loop co-training;
  - hardness curriculum;
  - sparse loop supervision;
  - two-stage standard-to-loop training;
  - label-smoothed listwise ranking.
- Code/config updates prepared before validation:
  - Added deterministic `passage_sampling_strategy` support with default `first` and opt-in `seeded_random`.
  - Added `embedding_pooling_mode` support with default `mean_pool` and opt-in `first_token`.
  - Added registered versions `loop_matryoshka_first_token_seeded_sampling` and `loop_matryoshka_first_token_first_pool`.
  - Added `configs/goal_batch_012_seeded_sampling_first_token.yaml`.
  - Added `configs/goal_batch_012_first_pool_first_token.yaml`.
  - Created `experiments/batches/batch_012_dev.yaml`.
- Batch purpose: `dev`.
- Candidate track: `standalone_main` exploration only. These dev results cannot trigger `main_goal_success`.
- Portfolio candidates:
  - `r012_seeded_sampling_first_token_t10`: deterministic seeded random positive/negative sampling, full-epoch first-token loopwise training, fixed `loop_idx=10`.
  - `r012_first_pool_first_token_t10`: first-token retrieval embedding pooling for queries and documents, full-epoch first-token loopwise training, fixed `loop_idx=10`.
- Portfolio rationale:
  - tests two distinct mechanisms after objective-local search failure: training data construction and parameter-free embedding extraction;
  - does not retest neighboring loop depths, memory modes, or the same failed checkpoint;
  - uses only dev tasks: `SciFact`, `NFCorpus`, `FiQA2018`, and `SCIDOCS`;
  - keeps one global candidate rule per run across all dev tasks;
  - uses no frozen-baseline checkpoint, baseline ensemble, baseline concatenation, or interpolation in candidate scoring.
- Estimated GPU budget: 20 GPU hours against the configured 24 GPU-hour limit, with at most 2 concurrent GPU jobs.
- Checks before submission:
  - Subagent code/protocol review gate passed after `outputs/goal/state.json` research-design bookkeeping was updated from round 004 to round 005.
  - `source scripts/slurm_env.sh && "$PYTHON_BIN" -m compileall src scripts` passed.
  - `bash -n scripts/*.sh scripts/*.sbatch` passed.
  - `source scripts/slurm_env.sh && "$PYTHON_BIN" scripts/goal_validate_manifest.py experiments/batches/batch_012_dev.yaml` passed with expected dev-only standalone warnings.
  - `source scripts/slurm_env.sh && "$PYTHON_BIN" scripts/goal_submit_batch.py experiments/batches/batch_012_dev.yaml --dry-run --submit-postprocess` passed.
  - `source scripts/slurm_env.sh && "$PYTHON_BIN" scripts/goal_preflight.py --manifest experiments/batches/batch_012_dev.yaml` passed.
  - `git diff --check` passed.
- Submission:
  - Submitted only through `scripts/goal_submit_batch.py --submit --submit-postprocess`.
  - The first submit attempt failed before job creation because the cluster required site scheduler options.
  - A sandboxed retry with site scheduler options failed before job creation because Slurm controller socket access is blocked inside the sandbox.
  - The approved external retry through the same `goal_submit_batch.py` command succeeded with site scheduler options provided only through temporary environment variables.
  - Train/eval job IDs:
    - `r012_seeded_sampling_first_token_t10`: train `5130092`, eval `5130093`
    - `r012_first_pool_first_token_t10`: train `5130094`, eval `5130095`
  - Postprocess job ID: `5130096`
  - Postprocess dependency: `afterany:5130093:5130095`
- Next action: wait for Slurm-native postprocess, then inspect `outputs/goal/runs/batch_012_dev/scoreboard.json`.

## 2026-06-21 Batch 012 Dev Standalone Result

- `batch_012_dev` completed Slurm-native postprocess:
  - train/eval jobs:
    - `r012_seeded_sampling_first_token_t10`: train `5130092`, eval `5130093`
    - `r012_first_pool_first_token_t10`: train `5130094`, eval `5130095`
  - postprocess job: `5130096`
  - marker: `outputs/goal/runs/batch_012_dev/postprocess_done.json`
  - scoreboard: `outputs/goal/runs/batch_012_dev/scoreboard.json`
- There is no `postprocess_failed.json`.
- Batch purpose: `dev`; both candidates are `standalone_main` exploration only, so this batch cannot trigger `main_goal_success`.
- Scoreboard interpretation under the current claim-track policy:
  - `r012_seeded_sampling_first_token_t10__loop10`: track `standalone_main`, `minimal_positive_signal=false`, `fusion_diagnostic_pass=false`, `research_grade_threshold_pass=false`, `main_goal_success=false`, `publishable_score_candidate=false`, dev min delta `-0.01053`, dev mean delta `-0.00197`, dev tasks won/lost `2/2`.
    - Dev deltas: `SciFact +0.00572`, `NFCorpus -0.00746`, `FiQA2018 -0.01053`, `SCIDOCS +0.00439`.
  - `r012_first_pool_first_token_t10__loop10`: track `standalone_main`, all success flags false, dev min delta `-0.02756`, dev mean delta `-0.0095725`, dev tasks won/lost `0/4`.
    - Dev deltas: `SciFact -0.02756`, `NFCorpus -0.00595`, `FiQA2018 -0.00474`, `SCIDOCS -0.00004`.
- Decision: `main_goal_success=false`. Neither candidate produced a viable global dev signal. Seeded sampling has meaningful local positives on `SciFact` and `SCIDOCS`, but still regresses `NFCorpus` and `FiQA2018` with negative macro mean. First-token retrieval pooling regressed all four dev tasks.
- Local search remains exhausted. Recent standalone dev evidence has now also falsified seeded random passage sampling as a global fix and first-token retrieval pooling as a global fix.

## 2026-06-22 Research Design Round 006 And Batch 013 Portfolio

- Trigger: `batch_012_dev` completed Slurm-native postprocess with `postprocess_done.json`, no `postprocess_failed.json`, and no strong viable standalone dev signal.
- Scoreboard conclusion for `batch_012_dev` under the current claim-track policy:
  - `r012_seeded_sampling_first_token_t10__loop10`: `standalone_main`, purpose `dev`, dev deltas `SciFact +0.00572`, `NFCorpus -0.00746`, `FiQA2018 -0.01053`, `SCIDOCS +0.00439`, min delta `-0.01053`, mean delta `-0.00197`, dev tasks won/lost `2/2`, all success flags false.
  - `r012_first_pool_first_token_t10__loop10`: `standalone_main`, purpose `dev`, dev deltas `SciFact -0.02756`, `NFCorpus -0.00595`, `FiQA2018 -0.00474`, `SCIDOCS -0.00004`, min delta `-0.02756`, mean delta `-0.0095725`, dev tasks won/lost `0/4`, all success flags false.
- Decision: `main_goal_success=false`. Batch 012 was dev-only, and neither standalone candidate was globally viable.
- Local search remains exhausted. Batch 012 further showed that seeded sampling gives split task behavior and first-token retrieval pooling is broadly harmful on the dev set.
- Entered `RESEARCH_DESIGN_MODE` before creating a new batch.
- Real subagents were used for the required workflow gates:
  - one research-design subagent summarized broader standalone directions and portfolio options;
  - one code/protocol subagent reviewed `batch_013_dev` before validation and reported no blockers.
- Wrote the research plan at `docs/research_design_round_006.md`.
- Broad standalone directions considered:
  - query/document loop co-training;
  - two-stage standard-to-loop training;
  - medium-hard negative window;
  - sparse or late loop supervision;
  - label-smoothed listwise loss;
  - candidate-internal multi-loop score smoothing;
  - length/truncation-aware document encoding.
- Code/config updates prepared before validation:
  - Added deterministic `middle_negatives` passage sampling.
  - Added registered versions `loop_two_stage_first_token` and `loop_matryoshka_first_token_middle_negatives`.
  - Added `two_stage_loopwise` training support with a standard warmup stage followed by loopwise training.
  - Added `configs/goal_batch_013_two_stage_first_token.yaml`.
  - Added `configs/goal_batch_013_middle_negatives_first_token.yaml`.
  - Created `experiments/batches/batch_013_dev.yaml`.
- Batch purpose: `dev`.
- Candidate track: `standalone_main` exploration only. These dev results cannot trigger `main_goal_success`.
- Portfolio candidates:
  - `r013_two_stage_warmup_first_token_t10`: 25% standard hard-negative warmup followed by loopwise first-token training, fixed `loop_idx=10`.
  - `r013_middle_negatives_first_token_t10`: first-token loopwise training with deterministic middle-window negatives, fixed `loop_idx=10`.
- Portfolio rationale:
  - tests two distinct mechanisms after local-search exhaustion: optimization path and negative-hardness pressure;
  - does not retest neighboring loop depths, memory modes, or the same failed checkpoint;
  - uses only dev tasks: `SciFact`, `NFCorpus`, `FiQA2018`, and `SCIDOCS`;
  - keeps one global candidate rule per run across all dev tasks;
  - uses candidate-only scoring with no frozen-standard scoring input or interpolation.
- Estimated GPU budget: 20 GPU hours against the configured 24 GPU-hour limit, with at most 2 concurrent GPU jobs.
- Checks before submission:
  - Research-design subagent gate passed.
  - Code/protocol subagent gate passed with no blockers.
  - `source scripts/slurm_env.sh && "$PYTHON_BIN" -m compileall src scripts` passed.
  - `bash -n scripts/*.sh scripts/*.sbatch` passed.
  - `git diff --check` passed.
  - `source scripts/slurm_env.sh && "$PYTHON_BIN" scripts/goal_validate_manifest.py experiments/batches/batch_013_dev.yaml` passed with expected dev-only standalone warnings.
  - `source scripts/slurm_env.sh && "$PYTHON_BIN" scripts/goal_submit_batch.py experiments/batches/batch_013_dev.yaml --dry-run --submit-postprocess` passed.
  - `source scripts/slurm_env.sh && "$PYTHON_BIN" scripts/goal_preflight.py --manifest experiments/batches/batch_013_dev.yaml` passed.
- Submission:
  - Submitted only through `scripts/goal_submit_batch.py --submit --submit-postprocess`.
  - The first submit attempt failed before job creation because the cluster required site scheduler options.
  - A sandboxed retry with site scheduler options failed before job creation because Slurm controller socket access is blocked inside the sandbox.
  - The approved external retry through the same `goal_submit_batch.py` command succeeded with site scheduler options provided only through temporary environment variables.
  - Train/eval job IDs:
    - `r013_two_stage_warmup_first_token_t10`: train `5156217`, eval `5156218`
    - `r013_middle_negatives_first_token_t10`: train `5156219`, eval `5156220`
  - Postprocess job ID: `5156221`
  - Postprocess dependency: `afterany:5156218:5156220`
- Next action: wait for Slurm-native postprocess, then inspect `outputs/goal/runs/batch_013_dev/scoreboard.json`.

## 2026-06-26 Batch 013 Dev Standalone Result

- `batch_013_dev` completed Slurm-native postprocess:
  - train/eval jobs:
    - `r013_two_stage_warmup_first_token_t10`: train `5156217`, eval `5156218`
    - `r013_middle_negatives_first_token_t10`: train `5156219`, eval `5156220`
  - postprocess job: `5156221`
  - marker: `outputs/goal/runs/batch_013_dev/postprocess_done.json`
  - scoreboard: `outputs/goal/runs/batch_013_dev/scoreboard.json`
- There is no `postprocess_failed.json`.
- Status refresh through `scripts/goal_status.py --batch-id batch_013_dev --update-state` confirmed all train, eval, and postprocess jobs completed. The sandboxed status call hung in `squeue`, so the successful refresh was run externally through the same repository status command.
- Batch purpose: `dev`; both candidates are `standalone_main` exploration only, so this batch cannot trigger `main_goal_success`.
- Scoreboard interpretation under the current claim-track policy:
  - `r013_two_stage_warmup_first_token_t10__loop10`: `standalone_main`, purpose `dev`, dev deltas `SciFact +0.00248`, `NFCorpus +0.00030`, `FiQA2018 -0.00603`, `SCIDOCS -0.00224`, min delta `-0.00603`, mean delta `-0.0013725`, dev tasks won/lost `1/3`, all success flags false.
  - `r013_middle_negatives_first_token_t10__loop10`: `standalone_main`, purpose `dev`, dev deltas `SciFact +0.00619`, `NFCorpus +0.00164`, `FiQA2018 -0.01183`, `SCIDOCS -0.00188`, min delta `-0.01183`, mean delta `-0.00147`, dev tasks won/lost `2/2`, all success flags false.
- Decision: `main_goal_success=false`. Batch 013 was dev-only, and neither standalone candidate was globally viable.
- Local search remains exhausted. Batch 013 further falsified standard-to-loop warmup and deterministic middle-window negatives as global fixes; FiQA2018 and SCIDOCS remain recurring regressions.

## 2026-06-26 Research Design Round 007 And Batch 014 Portfolio

- Trigger: `batch_013_dev` completed Slurm-native postprocess with `postprocess_done.json`, no `postprocess_failed.json`, and no strong viable standalone dev signal.
- Entered `RESEARCH_DESIGN_MODE` before creating the next batch.
- Real subagents were used for required workflow gates:
  - one research gate analyzed `batch_013_dev`, recent standalone failure patterns, broad directions, and portfolio options;
  - one code/protocol gate reviewed available mechanisms, code-change requirements, state blockers, and manifest constraints.
- Wrote the research plan at `docs/research_design_round_007.md`.
- Broad standalone directions considered:
  - query/document loop co-training;
  - label-smoothed listwise loss;
  - sparse or late loop supervision;
  - deterministic easy-to-hard negative curriculum;
  - length/truncation-aware document encoding;
  - candidate-only multi-loop score aggregation;
  - non-fusion candidate-internal calibration.
- Code/config updates prepared before validation:
  - Added `loopwise_label_smoothed` and `loopwise_sparse` loss paths.
  - Added registered versions `loop_label_smooth_first_token` and `loop_sparse_first_token`.
  - Added `configs/goal_batch_014_label_smooth_first_token.yaml`.
  - Added `configs/goal_batch_014_sparse_first_token.yaml`.
  - Created `experiments/batches/batch_014_dev.yaml`.
- Batch purpose: `dev`.
- Candidate track: `standalone_main` exploration only. These dev results cannot trigger `main_goal_success`.
- Portfolio candidates:
  - `r014_sparse_late_first_token_t10`: first-token loopwise training supervised only at loops `4`, `7`, and `10`, fixed `loop_idx=10`.
  - `r014_label_smooth_first_token_t10`: first-token loopwise training with `label_smoothing=0.05`, fixed `loop_idx=10`.
- Portfolio rationale:
  - tests two distinct mechanisms after local-search exhaustion: supervision topology and hard-target calibration;
  - does not retest neighboring loop depths, memory modes, failed checkpoint depths, or nearby negative-window variants;
  - uses only dev tasks: `SciFact`, `NFCorpus`, `FiQA2018`, and `SCIDOCS`;
  - keeps one global candidate rule per run across all dev tasks;
  - uses candidate-only scoring with no frozen-standard scoring input or interpolation.
- Estimated GPU budget: 20 GPU hours against the configured 24 GPU-hour limit, with at most 2 concurrent GPU jobs.
