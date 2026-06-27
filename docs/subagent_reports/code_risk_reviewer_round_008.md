# code_risk_reviewer report

## Blockers

None.

## High Severity

None.

## Medium Severity

- `outputs/goal/state.json`: state still had `current_batch`/`postprocess_job_id`/`open_jobs` pointing to completed `batch_014_dev`, while `phase=DESIGN_BATCH` and `next_batch_design` pointed to `batch_015_dev`.
  - Impact: resume sessions or `scripts/goal_status.py` calls without `--batch-id` could inspect batch014 rather than batch015; `last_validation` still described batch014.
  - Required fix: move state to batch015 dry-run/preflight state before submission, or otherwise disambiguate completed batch014 from active batch015.

## Low Severity

- `batch_015` configs/manifest/docs were untracked at review time.
  - Impact: `git diff` did not include them; future review/commit could miss the manifest/config.
  - Required fix: include the new files in the final tracked/committed scope.

- `src/train.py` logged `standard_dim_mrl` dimensions by using `loss_dict["matryoshka_dims"].max()` to infer embedding dimension.
  - Impact: safe for batch015 `[768,384,192]`, but future smaller-only dimension sets could be logged awkwardly.
  - Required fix: record dimensions directly from `loss_dict["matryoshka_dims"]`.

## Positive Checks

- No direct `sbatch` bypass was found in batch015 changes.
- Manifest is `purpose: dev`, fixed to four dev tasks, and predeclares `candidate_loop_indices: [1]`.
- No fusion/frozen-standard scoring fields were found in the new batch.
- Baseline paths are read-only references to `outputs/baselines/standard_frozen/*`.
- New versions are not exact `version=standard`; frozen baseline identification remains unchanged.
- Query/doc prefixes are saved in `loop_config.json` and restored by `load_model()`.
- Optimizer and `assert_encoder_only_trainable()` still restrict training to `encoder.*`.
- Metric, NaN, missing, partial, and scoreboard handling were not loosened.

## Required Fixes Before Preflight

1. Advance state to batch015 through dry-run before preflight/submission.
2. Include the untracked batch015 files in the final review/commit scope.
3. Simplify `standard_dim_mrl` logging so dimensions are recorded directly.

The reviewer did not run validation, preflight, training, MTEB, `sbatch`, `squeue`, or `sacct`.
