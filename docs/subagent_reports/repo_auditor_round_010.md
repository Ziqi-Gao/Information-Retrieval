# Repo Auditor Round 010

Date: 2026-06-28

Read-only audit completed. No files were edited by the subagent, no Slurm jobs were submitted or cancelled, no training or evaluation was run, and no secrets were read.

## Files Used

- `AGENTS.md`
- `docs/goal_protocol.md`
- `docs/codex_subagents.md`
- `docs/codex_subagent_drafts/repo-auditor.toml`
- `outputs/goal/state.json`
- `experiments/batches/batch_016_dev.yaml`
- `outputs/goal/runs/batch_016_dev/postprocess_done.json`
- `outputs/goal/runs/batch_016_dev/scoreboard.json`
- `outputs/goal/runs/batch_016_dev/scoreboard.csv`
- `outputs/goal/runs/batch_016_dev/collected_results.csv`
- `outputs/goal/runs/batch_016_dev/submission_plan.json`
- `outputs/baselines/standard_frozen/results_summary.csv`
- `outputs/baselines/standard_frozen/baseline_manifest.json`

## Findings

`batch_016_dev` completed Slurm-native postprocess. `postprocess_done.json` records `status: completed` at `2026-06-28T11:08:10Z`, and the batch has `collected_results.csv`, `scoreboard.csv`, and `scoreboard.json`.

The result does not support a final-validation plan:

- `r016_standard_seeded_sampling__loop1`: best candidate, but weak. Mean delta `+0.00338`, min delta `+0.00045`; `NFCorpus` and `FiQA2018` are below the weak `+0.001` diagnostic margin.
- `r016_standard_inbatch_hybrid__loop1`: mean delta `-0.00215`, min delta `-0.00686`.
- `r016_qdoc_final_mean_pool_t3_neg3__loop3`: mean delta `-0.04931`, min delta `-0.07807`.

All candidates are `standalone_main`, but the batch purpose is `dev` and only four dev tasks were evaluated. Current protocol requires a `standalone_main` final candidate with seven final tasks, every final-task delta at least `+0.002`, macro mean at least `+0.005`, and no invalid or regressing task.

## Blockers And High-Risk Findings

- Do not write a final-validation plan from `batch_016_dev`; the dev signal is too weak.
- `outputs/goal/state.json` is stale: `last_dev_result` and `last_postprocess_check` still point at `batch_015_dev`, `phase` remains `SUBMIT_BATCH`, and postprocess job `5340275` is still marked running even though `postprocess_done.json` shows completion.
- If a new dev batch is created, it must not be a simple local sweep. It must first update factual state and continue `RESEARCH_DESIGN_MODE`.
- `submission_plan.json` contains site-specific scheduler details and must remain local/generated.
