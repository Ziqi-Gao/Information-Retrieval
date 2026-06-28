# Experiment Planner Round 012

Read-only planning review for final validation after `batch_017_dev_repair`. No files edited, no jobs submitted, no `sbatch`, no training/eval run.

## Basis

`batch_017_dev_repair` completed with strong dev evidence for `r017_seeded_lexical_hash__loop1`: all four dev tasks exceeded `+0.002`, min delta `+0.00665`, mean delta `+0.0162525`. `main_goal_success=false` only because the batch was `purpose: dev` and covered 4/7 final tasks.

## Recommended Final Manifest Scope

- `batch_id`: `batch_018_final`
- `purpose`: `final`
- `primary_metric`: `ndcg_at_10`
- `win_margin`: `0.001`
- baseline:
  - `outputs/baselines/standard_frozen/results_summary.csv`
  - `outputs/baselines/standard_frozen/baseline_manifest.json`
- budget:
  - `max_concurrent_gpu_jobs: 1`
  - `max_gpu_hours_estimate: 12`
  - `allow_submit: false` for draft; set `true` only with explicit final-submit approval
- final tasks, exact protocol/eval order:
  - `SciFact`
  - `NFCorpus`
  - `SCIDOCS`
  - `FiQA2018`
  - `ArguAna`
  - `Touche2020`
  - `TRECCOVID`

One experiment only:

- recommended `run_id`: `r018_final_seeded_lexical_hash`
- `claim_track`: `standalone_main`
- `version`: `standard_seeded_sampling`
- `eval_only`: `true`
- checkpoint: `outputs/goal/runs/batch_016_dev/r016_standard_seeded_sampling/final`
- exact rule: `loop_idx=1`, `candidate_loop_indices: [1]`, `lexical_hash_dim=1024`, `lexical_weight=0.15`
- scoring: candidate dense embeddings plus deterministic lexical hash features only
- no `fusion_standard_checkpoint_dir`, `fusion_alpha`, `fusion_scope`, frozen-standard embeddings, frozen-standard scores, or ensemble/interpolation

## Expected Postprocess

Plan should contain one eval job, no train job, and one Slurm-native postprocess job depending on the eval job. If later submitted, use only `scripts/goal_submit_batch.py --submit --submit-postprocess`.

## Blockers / High Risks

No design-level protocol blocker found for preparing the final manifest.

Submission blocker: final validation still needs explicit user approval plus normal validation/dry-run/preflight gates.

High risk: the three held-out final tasks may regress, especially `TRECCOVID` where the frozen baseline is already high. Any missing, failed, duplicate, NaN, partial, or below-`+0.002` task invalidates main success.

Medium risk: lexical hashing may overemphasize term overlap on `ArguAna`, `Touche2020`, or `TRECCOVID`; no task-specific retuning is allowed.
