# repo_auditor report

## Current execution path

- Current branch: `codex-bert-only-loop-memory`, HEAD `831a8f7`.
- `batch_014_dev` completed postprocess: `outputs/goal/runs/batch_014_dev/postprocess_done.json`.
- Scoreboard exists. Both `standalone_main` dev candidates regressed all four dev tasks:
  - `r014_sparse_late_first_token_t10__loop10`: mean `-0.009035`, min `-0.01268`, dev wins `0/4`.
  - `r014_label_smooth_first_token_t10__loop10`: mean `-0.0132775`, min `-0.02339`, dev wins `0/4`.

## Important Files/Functions

- Experiment registry: `src/experiments.py`.
- Training path: `src/train.py` enforces encoder-only trainable params and dispatches sparse/label-smoothed loop losses.
- Metric parsing/writing: `src/eval_mteb.py` parses MTEB metrics and appends `results_summary.csv`.
- Collection/scoring: `scripts/goal_collect.py` and `scripts/goal_scoreboard.py`.
- Manifest safety: `scripts/goal_validate_manifest.py`.

## Slurm Flow

- Submission path remains `scripts/goal_submit_batch.py` only; direct `sbatch` remains guarded by Slurm wrappers.
- `batch_014_dev` used train/eval jobs plus Slurm-native postprocess in `outputs/goal/runs/batch_014_dev/submission_plan.json`.
- No `batch_015_dev` manifest or `r015_` entries existed at audit time.

## Result Flow

- `collected_results.csv` has 8 completed rows, four dev tasks per candidate, no fusion fields populated.
- Scoreboard final-task `missing_result` rows for `ArguAna`, `Touche2020`, and `TRECCOVID` are expected because the manifest is dev-only.
- Frozen baseline exists and covers all seven protocol tasks.

## Risks

- Blocker before `batch_015_dev` design/submission: `outputs/goal/state.json` is stale/inconsistent. `last_dev_result` and `last_postprocess_check` still point to `batch_013_dev`, while `last_scoreboard` points to `batch_014_dev`.
- State still has `phase: SUBMIT_BATCH` and `open_jobs` as `unknown` for `batch_014_dev` despite completed postprocess evidence.
- `outputs/goal/state.json` is the only dirty tracked file. Parent should reconcile it intentionally; subagents should not.
- `docs/agent_lab_notebook.md` records `batch_014_dev` design/submission preparation but not the completed result.
- Scientifically, `batch_014_dev` strengthens local-search exhaustion. Another nearby loop-depth/objective tweak would violate the protocol unless a new research-design note justifies it.

## Recommended Guardrails

- Parent should record `batch_014_dev` as terminal using `postprocess_done.json`, `collected_results.csv`, and `scoreboard.json`; update `last_dev_result`, `last_postprocess_check`, `phase`, and `open_jobs` coherently.
- Before creating `batch_015_dev`, enter `RESEARCH_DESIGN_MODE` again and write `docs/research_design_round_008.md` or update the lab notebook.
- `batch_015_dev` should be a non-local, predeclared `standalone_main` portfolio or an explicit stop decision.
- Keep real submission through `scripts/goal_submit_batch.py --submit --submit-postprocess`.
- Treat any fusion/frozen-standard score or embedding/standard+candidate concatenation as `fusion_diagnostic` only.

## Uncertainty

The audit did not query Slurm. Terminal evidence comes from `postprocess_done.json` and scoreboard files.
