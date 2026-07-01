# Experiment Planner Round 013

Read-only planning review for `batch_018_final` repair. No files edited, no jobs submitted, no `sbatch`, no training/eval run.

## Evidence

`batch_018_final` used the correct final manifest scope, but eval job `5386592` failed during `ArguAna` loading. `scoreboard.json` marks all seven tasks as `missing results_summary.csv` because no top-level summary was written. Partial raw outputs exist for the first four tasks, but they must not be backfilled or interpreted as final results.

Failure point: `slurm_logs/loopmat_eval_5386592.err` reports `FileNotFoundError` for `mteb/arguana` corpus loading from Hugging Face.

## Recommended Repair Manifest

- `batch_id`: `batch_018_final_repair`
- `purpose`: `final`
- `repair: true`
- `repair_of`: `batch_018_final`
- `primary_metric`: `ndcg_at_10`
- `win_margin`: `0.001`
- budget:
  - `max_concurrent_gpu_jobs: 1`
  - `max_gpu_hours_estimate: 12`
  - `allow_submit: true` because the user approved repair submission
- final tasks, exact order:
  - `SciFact`
  - `NFCorpus`
  - `SCIDOCS`
  - `FiQA2018`
  - `ArguAna`
  - `Touche2020`
  - `TRECCOVID`

One experiment only:

- `run_id`: `r017_seeded_lexical_hash`
- `claim_track`: `standalone_main`
- `version`: `standard_seeded_sampling`
- `eval_only`: `true`
- checkpoint: `outputs/goal/runs/batch_016_dev/r016_standard_seeded_sampling/final`
- exact rule preserved: `loop_idx=1`, `candidate_loop_indices: [1]`, `lexical_hash_dim=1024`, `lexical_weight=0.15`
- candidate-only scoring: dense candidate embeddings plus deterministic lexical hash features
- no `fusion_standard_checkpoint_dir`, `fusion_alpha`, `fusion_scope`, frozen-standard embedding/score input, or interpolation

## Expected Postprocess

Plan should contain one eval job, no train job, and one Slurm-native postprocess job with `afterany:<eval_job_id>`. If eval succeeds, postprocess should collect seven final rows and score `main_goal_success`; if eval fails again, all missing/partial rows remain failures.

## Validation Commands To Run Before Submission

```bash
python -m compileall -q src scripts
bash -n scripts/*.sh scripts/*.sbatch
python scripts/goal_validate_manifest.py experiments/batches/batch_018_final_repair.yaml
python scripts/goal_submit_batch.py experiments/batches/batch_018_final_repair.yaml --dry-run --submit-postprocess
python scripts/goal_scoreboard.py --self-test
python scripts/goal_preflight.py --manifest experiments/batches/batch_018_final_repair.yaml
```

## Blockers / High Risks

Blocker before repair submission: fix or verify the `ArguAna` dataset loading path/cache issue; otherwise the repair is likely to repeat the same failure.

High risk: final held-out tasks may still regress. Any missing, failed, duplicate, NaN, partial, or task delta below `+0.002` invalidates `main_goal_success`.

Do not reuse partial `batch_018_final` raw outputs; repair must rerun the full seven-task final candidate.
