# Experiment Planner Round 011

Read-only review for proposed `batch_017_dev_repair`. No files edited, no jobs submitted, no training/eval run.

## Evidence

`batch_017_dev` had two eval-only `standalone_main` candidates. `r017_seeded_chunked_docs` completed. `r017_seeded_lexical_hash` failed with `missing_result` because no top-level `results_summary.csv` was written.

The failed eval produced partial raw metrics for `SciFact` and `NFCorpus`, then stopped on `FiQA2018` with an MTEB/datasets cache ambiguity. These partial metrics must not be used as a scoreboard claim.

## Recommended Repair Scope

Create `experiments/batches/batch_017_dev_repair.yaml` with only one experiment:

- `batch_id`: `batch_017_dev_repair`
- `purpose`: `dev`
- `primary_metric`: `ndcg_at_10`
- `win_margin`: `0.001`
- same frozen baseline paths
- dev tasks: exactly `SciFact`, `NFCorpus`, `FiQA2018`, `SCIDOCS`
- final task list: unchanged protocol list
- budget: `max_concurrent_gpu_jobs: 1`, `max_gpu_hours_estimate: 4`, `allow_submit: true`
- experiment: only `run_id: r017_seeded_lexical_hash`
- `claim_track`: `standalone_main`
- `eval_only`: `true`
- `version`: `standard_seeded_sampling`
- checkpoint: `outputs/goal/runs/batch_016_dev/r016_standard_seeded_sampling/final`
- eval rule unchanged: `loop_idx=1`, `candidate_loop_indices=[1]`, `lexical_hash_dim=1024`, `lexical_weight=0.15`
- no `fusion_standard_checkpoint_dir`, `fusion_alpha`, `fusion_scope`, standard-score interpolation, or ensemble input

Expected plan shape: one eval job, zero train jobs, one postprocess job depending only on that eval job. Submission must go through `scripts/goal_submit_batch.py --submit --submit-postprocess`.

## Blockers And Risks

No design-level protocol blocker found. The unresolved `mteb/fiqa` cache ambiguity must be fixed before repair submission. Do not include `r017_seeded_chunked_docs`; it already completed and is not part of the failed-candidate repair.
