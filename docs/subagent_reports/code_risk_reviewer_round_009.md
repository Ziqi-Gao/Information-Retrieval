# Code Risk Reviewer Round 009

## Blockers

- None found for protocol compliance: `batch_016_dev` is dev-only, `standalone_main`, candidate-only scoring, no frozen-standard fusion fields, no baseline/metric/threshold code changes. Manifest validation passes with only expected dev-only warnings.

## High Severity

- q/doc loop candidate has serious timeout risk. `loop_final_qdoc_mean_pool` loops queries, positives, and all negatives at `tmax=3`; with `train_sample_size=25000`, `batch_size=2`, and doc length 512, this may exceed the default 4h train walltime in `scripts/slurm_train.sbatch`. Required fix: reduce q/doc workload further or ensure external `SBATCH_ARGS --time` is sufficient before submission.

## Medium/Low Risks

- `encode_looped_doc_training_batch` encodes flattened negatives with `batch_size=batch_size`, not `batch_size * num_negatives`, in `src/train.py`. Shape is valid, but it increases chunk count and runtime. This is acceptable as memory protection for the q/doc candidate.
- `results_summary.csv` does not record `loop_docs` / `doc_loop_idx`; doc-loop results are distinguishable by artifact path and manifest in this unique batch.
- State is in a transition state until dry-run/submit advances `current_batch` and job rows.
- `round_009_summary.md` needed this code-risk reviewer report added before declaring workflow gate readiness.

## Required Fixes

1. Address q/doc runtime risk before real submit.
2. Save this report as `docs/subagent_reports/code_risk_reviewer_round_009.md`.
3. Update round summary/state after the report is saved.
