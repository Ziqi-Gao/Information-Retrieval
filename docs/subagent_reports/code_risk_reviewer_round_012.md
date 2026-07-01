# Code Risk Reviewer Round 012

Read-only review only. No files edited, no jobs submitted, no `sbatch`, no training, no MTEB eval.

## Verdict

No blocker or high-severity protocol risk found for `experiments/batches/batch_018_final.yaml`.

`batch_018_final` preserves the `batch_017_dev_repair` candidate rule and is shaped as a valid final-validation manifest: `purpose: final`, `claim_track: standalone_main`, all seven protocol final tasks, eval-only candidate scoring, and no frozen-standard fusion/interpolation fields.

## Compliance Check

- Candidate rule preserved: same `standard_seeded_sampling` checkpoint, `loop_idx=1`, `candidate_loop_indices=[1]`, `lexical_hash_dim=1024`, `lexical_weight=0.15`.
- Final coverage: eval task list is exactly `SciFact`, `NFCorpus`, `SCIDOCS`, `FiQA2018`, `ArguAna`, `Touche2020`, `TRECCOVID`.
- Standalone track: no `fusion_standard_checkpoint_dir`, `fusion_alpha`, `fusion_scope`, `self_query_alpha`, doc chunking, frozen-standard score, or ensemble input.
- Metric/thresholds unchanged: manifest uses `primary_metric: ndcg_at_10`; no tracked script/code diff is present in this review.
- Budget: `max_concurrent_gpu_jobs=1` and `max_gpu_hours_estimate=12` are within state limits `4` and `24.0`.
- `allow_submit: true` is acceptable given explicit user approval recorded in round 012 context.

## Blockers

None found.

## High Risks

None found.

## Medium / Residual Risks

- Scientific risk: held-out `ArguAna`, `Touche2020`, and `TRECCOVID` may regress. Any missing, failed, duplicate, partial, NaN, or delta below `+0.002` invalidates main-goal success.
- Dependency risk: FiQA still relies on the narrow MTEB qrels loader patch; dependency/cache changes could require revalidation.
- Procedural risk: this reviewer did not run validation/dry-run/preflight by instruction. Parent should run the normal required gates before submission.

## Remediation

No blocker/high remediation required. Before submit, run the standard cheap gates only: manifest validation, dry-run with postprocess, and preflight, then submit only through `scripts/goal_submit_batch.py --submit --submit-postprocess`.
