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
