# repo_auditor_round_013

## Verdict

`batch_018_final` failure is a pure infrastructure/cache/config/data-loading issue, not a failed candidate score. The eval reached `ArguAna` and failed while MTEB/HF datasets tried to load `mteb/arguana` corpus data, before a valid seven-task `results_summary.csv` could be written.

A `batch_018_final_repair` manifest is protocol-valid if it preserves the exact `r017_seeded_lexical_hash__loop1` candidate rule and only changes repair metadata, batch id, output paths, and cache configuration needed to make data loading succeed.

## Evidence

- `batch_018_final.yaml` is `purpose: final`, `standalone_main`, all seven tasks, `loop_idx=1`, `lexical_hash_dim=1024`, `lexical_weight=0.15`.
- `submission_plan.json` submitted the same exact rule and seven tasks.
- `slurm_logs/loopmat_eval_5386592.out` shows eval completed through task startup for SciFact, NFCorpus, SCIDOCS, FiQA2018, then entered ArguAna.
- `slurm_logs/loopmat_eval_5386592.err` shows `FileNotFoundError` for `hf://datasets/mteb/arguana.../corpus.jsonl` during `task.load_data`, not model scoring.
- `src/eval_mteb.py` writes `results_summary.csv` only after all task loops finish, so ArguAna failure makes the whole run collect as missing.
- `per_run_validation.json`, `collected_results.csv`, and `scoreboard.json` correctly mark all seven rows `missing_result`; this cannot support a final claim.

## Required Repair Manifest Shape

- `batch_id: batch_018_final_repair`
- `purpose: final`
- `repair: true`, `repair_of: batch_018_final`
- `primary_metric: ndcg_at_10`
- frozen baseline paths unchanged
- `claim_track: standalone_main`
- `run_id: r017_seeded_lexical_hash`
- `version: standard_seeded_sampling`
- `eval_only: true`
- checkpoint unchanged: `outputs/goal/runs/batch_016_dev/r016_standard_seeded_sampling/final`
- `task_names`: exactly `SciFact`, `NFCorpus`, `SCIDOCS`, `FiQA2018`, `ArguAna`, `Touche2020`, `TRECCOVID`
- `eval_all_loops: false`
- `loop_idx: 1`
- `candidate_loop_indices: [1]`
- `lexical_hash_dim: 1024`
- `lexical_weight: 0.15`
- no fusion fields, no doc chunking, no self-query changes, no per-task rule changes.

## Blockers / High Risks

- Existing `batch_018_final` is not a valid final result; do not backfill from partial raw JSON.
- Repair must fix ArguAna data availability/cache/config first, or it will repeat the same failure.
- Use a new batch id/output path; do not overwrite `batch_018_final` outputs.
- Submit only through `scripts/goal_submit_batch.py`; no manual `sbatch`.
- Any missing/failed/NaN/duplicate/partial final-task row invalidates the final claim.
