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
