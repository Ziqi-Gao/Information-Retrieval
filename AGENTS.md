# Project Maintenance Rules

This file is for future Codex sessions working in this repository. Read it before making changes.

## Project Scope

- This repository maintains the Loop-Wise Matryoshka Retrieval experiments.
- Keep the public repository reproducible with code, configs, scripts, docs, and dependency declarations only.
- Generated artifacts are local-only unless the user explicitly asks otherwise.
- The later research goal is to develop a retrieval pipeline that robustly beats the frozen `standard` baseline on every final retrieval dataset by NDCG@10.
- The current preparation scope is infrastructure only: safe batch design, Slurm orchestration, result collection, baseline comparison, and handoff documentation.
- Do not implement new retrieval models, new scoring semantics, or new final-claim logic during preparation work.

## Protocol Metric

- Primary metric: `ndcg_at_10`.
- Weak diagnostic margin: `0.001`.
- Standalone research-grade threshold: every final-task delta must be at least `+0.002`, macro mean delta must be at least `+0.005`, and no task may regress.
- Publishable score-candidate threshold: every final-task delta must be at least `+0.002`, macro mean delta must be at least `+0.008`, and the result must be marked as score-only unless bootstrap/significance evidence is available.
- Final tasks, in protocol order:
  - `SciFact`
  - `NFCorpus`
  - `SCIDOCS`
  - `FiQA2018`
  - `ArguAna`
  - `Touche2020`
  - `TRECCOVID`
- Candidate claims are split into tracks:
  - `standalone_main`: candidate pipeline scoring only, no standard+candidate ensemble or frozen-standard score/embedding in candidate scoring.
  - `fusion_diagnostic`: any candidate using `fusion_standard_checkpoint_dir`, `fusion_alpha`, `fusion_scope`, standard+loop weighted concatenation, standard embeddings/scores, or an explicit ensemble with the frozen standard model.
  - `diagnostic`: exploratory or non-final analyses.
- `fusion_diagnostic` candidates may be evaluated and reported, but they must never trigger main goal success.
- Main goal success requires `candidate_track == standalone_main`, `purpose: final`, exact coverage of all seven final tasks, no missing/failed/timed-out/duplicate/NaN results, every final-task delta at least `+0.002`, macro mean delta at least `+0.005`, and no task regression.
- `+0.001` on every final task is only `minimal_positive_signal`; it must not be called goal achieved.
- Missing, failed, timed-out, NaN, duplicate, or partial-task results are failures.
- Dev subsets are allowed for iteration, but final claims require full final-task coverage against the frozen baseline.

## Required Workflow

Future autonomous sessions must follow this sequence:

```text
BOOTSTRAP
-> AUDIT_REPO
-> INIT_GOAL_STATE
-> FREEZE_OR_VALIDATE_BASELINE
-> DESIGN_BATCH
-> VALIDATE_BATCH
-> DRY_RUN_SUBMIT
-> SUBMIT_BATCH
-> WAIT_FOR_JOBS
-> COLLECT_RESULTS
-> SCORE_RESULTS
-> DECIDE_NEXT
-> FINAL_VALIDATE
-> REPORT
```

For preparation-only tasks, stop before real large-scale `SUBMIT_BATCH`.

## Research Exploration Discipline

- If repeated `standalone_main` dev batches fail, or recent work is mainly local sweeps over nearby `loop_idx`, memory modes, failed checkpoint depths, or small hyperparameter perturbations, treat local search as exhausted.
- After local search exhaustion, do not submit another local-neighborhood sweep. Enter `RESEARCH_DESIGN_MODE` first and write a research plan in `docs/agent_lab_notebook.md` or a dedicated research-design note.
- `RESEARCH_DESIGN_MODE` must summarize prior standalone and fusion-diagnostic evidence, identify repeated regression patterns, explain why more local sweeps are low-value, compare 5-8 substantially different `standalone_main` directions, rank them by validity/novelty/risk/cost/falsifiability, and select either a portfolio for one next dev batch or a stop decision.
- New dev batches should normally be efficient portfolios of 2-4 informative, diverse, protocol-valid candidates when budget allows. Prefer information gain and independent failure hypotheses over redundant local variants.
- Do not create or submit a new batch while a current batch is pending or running. Wait for postprocess and analyze the scoreboard first; if it confirms continued local-search failure, enter `RESEARCH_DESIGN_MODE` before designing the next batch.

## Experiment Logic

- Register experiment variants in `src/experiments.py` first. Training, evaluation, plotting, README, and scripts should derive from those version names.
- Keep version naming consistent across `src/experiments.py`, `src/train.py`, scripts, docs, and plots.
- Evaluation datasets must be MTEB text retrieval tasks. Keep `task_name` for single-task compatibility and use `task_names` for multi-task evaluation.
- Current experiments should update only ModernBERT encoder parameters. Do not add trainable projection heads, memory projections, memory-state embeddings, gates, or learned scaling unless the user explicitly asks for that ablation.
- Current parameter-free loop-memory modes mean:
  - `first_token`: prepend the previous loop's first query-token hidden state to the next loop.
  - `mean_pool`: prepend the previous loop's mean-pooled query-token hidden state to the next loop.
  - `token_concat`: prepend all previous-loop query-token hidden states to the next loop.
  - `none`: prepend no memory token; pass only the selected query-token inputs.
- Current loop-loss modes mean:
  - `loop_final`: train only from the final loop output.
  - `loop_matryoshka`: train from loopwise losses across loop depths.
  - `standard`: single-pass no-loop baseline.
- Do not reintroduce the old ambiguous `*_no_history` meaning or old `full`/`last`/`none` memory-history variants.

## Baseline Rules

- Do not compare against a moving or re-trained baseline.
- Freeze baseline summaries only through `scripts/goal_freeze_baseline.py`.
- Frozen baseline files belong under `outputs/baselines/standard_frozen/`.
- Do not overwrite `outputs/baselines/**` unless the user explicitly requests it and the command uses the script's `--force`.
- Do not claim progress until `outputs/baselines/standard_frozen/results_summary.csv` and `baseline_manifest.json` exist and validate.

## Slurm Rules

- Never run full training directly on the login node.
- Never call `sbatch` manually for goal batches after this framework exists.
- Slurm submission must go through `scripts/goal_submit_batch.py`.
- Slurm status must go through `scripts/goal_status.py`.
- Result collection must go through `scripts/goal_collect.py`.
- Scoring must go through `scripts/goal_scoreboard.py`.
- Legacy direct Slurm wrappers refuse to run unless `ALLOW_LEGACY_DIRECT_SBATCH=1` is explicitly set. They are outside the autonomous goal workflow.
- Local training wrappers and `src.run_all` refuse login-node training unless running inside Slurm or `ALLOW_LOGIN_NODE_TRAINING=1` is explicitly set for a deliberate small debug run.
- Use `scripts/slurm_env.sh` for batch jobs. Keep scheduler-specific options in environment variables such as `SBATCH_ARGS`, `CONDA_ENV`, `DEFAULT_CONDA_ENV`, and `PYTHON_BIN`.
- Slurm jobs should inherit only safe experiment/runtime variables. Never export API keys, SSH keys, cloud credentials, or token variables into jobs.
- Respect manifest budget controls: `max_concurrent_gpu_jobs`, `max_gpu_hours_estimate`, and `allow_submit`.

## State Management

- Goal state lives at `outputs/goal/state.json`.
- Batch manifests live under `experiments/batches/`.
- Submitted batch plans live under `outputs/goal/runs/<batch_id>/`.
- Evaluation outputs should use `outputs/goal/eval/<batch_id>/<run_id>/`.
- Do not overwrite existing run directories unless using an explicit resume path.
- Update state after initialization, dry-run submission, real submission, status updates, and scoreboards.
- If resuming after queue delays, read `outputs/goal/state.json` and the batch `submission_plan.json` before acting.

## Result Interpretation

- `pending`, `running`, and `unknown` jobs are not results.
- `failed_train`, `failed_eval`, `missing_result`, `invalid_metric`, `partial_tasks`, and `timeout` are failures.
- Do not interpret missing rows as zero or as ties.
- Do not modify `src/eval_mteb.py` metric extraction to make results look better.
- Do not label fusion or frozen-standard ensemble candidates as `standalone_main`; even if their deltas pass numerically, label them only as diagnostic/fusion results.
- Do not call `minimal_positive_signal` or legacy `pass_all_tasks` main goal success.
- Do not select the best loop per final task after seeing final-task results unless the manifest explicitly marks the batch as exploratory and excludes it from final claims.
- Loop-depth candidates should be compared as pre-defined candidate IDs, not per-task cherry-picks.

## Privacy And Repository Hygiene

- Never commit personal paths, usernames, cluster account names, scheduler partition names, absolute home paths, API keys, tokens, SSH keys, or machine-specific environment paths.
- This restriction applies to tracked, committed, or pushed repository content.
- Local untracked or ignored files may contain machine-specific runtime details, but they must remain untracked.
- Use relative paths in README, docs, scripts, and configs.
- Keep site-specific Slurm choices outside the repo through environment variables.
- Do not commit these generated/local directories:
  - `outputs/`
  - `.hf_cache/`
  - `.mplconfig/`
  - `slurm_logs/*.out`
  - `slurm_logs/*.err`
  - `__pycache__/`
- Before pushing, check tracked content with commands like:

```bash
git status --short
git ls-files
git ls-files -ci --exclude-standard
git grep -n -I -E '<local-user>|<cluster-project>|<absolute-path-prefix>|SBATCH --account|SBATCH --partition|HF_TOKEN|WANDB_API_KEY'
```

Replace placeholder patterns with real local identifiers before running the check locally. The final `git grep` should return no repository secrets or personal machine paths in tracked content.

## Output And Plot Boundaries

- `outputs/` is ignored and may contain checkpoints, eval JSON, summaries, plots, and goal-control state.
- Do not delete or overwrite `outputs/plots/`, `outputs/final_grid_experiment/`, or any raw MTEB result files unless the user explicitly asks.
- When removing stale experiment outputs, delete only specific method directories that are obsolete or explicitly requested.
- Plotting should read combined summaries and write back under `outputs/plots/` by default.
- Multi-task evaluation must keep raw/parsed MTEB outputs in task-specific directories and plot each task separately. Do not mix different `task` values into one loop-depth plot.

## Forbidden Actions

- Do not push to GitHub or open PRs unless explicitly asked.
- Do not run expensive training or large Slurm batches during preparation.
- Do not delete `outputs/**`, `slurm_logs/**`, checkpoints, or MTEB raw result files.
- Do not read `.env`, private keys, token caches, or credential files unless explicitly necessary and approved.
- Do not fabricate, infer, or backfill results.
- Do not make final-task claims from dev-only tasks.
- Final manifests must predeclare candidate loop indices or candidate IDs before evaluation.

## Validation Checklist

Run relevant cheap checks before saying the project is healthy:

```bash
python -m compileall -q src scripts
bash -n scripts/*.sh scripts/*.sbatch
python scripts/goal_validate_manifest.py experiments/batches/batch_template.yaml
python scripts/goal_submit_batch.py experiments/batches/batch_template.yaml --dry-run
python scripts/goal_scoreboard.py --self-test
python scripts/goal_preflight.py --manifest experiments/batches/batch_template.yaml
```

For model or experiment-registry changes, also verify that each version resolves to the expected loss mode and that training exposes only `encoder.*` trainable parameters.

## Collaboration Notes

- The user prefers concise Chinese explanations.
- Inspect the existing code and outputs before making behavioral claims.
- Do not make destructive file changes unless the user clearly requested them.
- If a request concerns experiment conclusions, distinguish between what the code tests, what current outputs show, and what would require new completed runs.
