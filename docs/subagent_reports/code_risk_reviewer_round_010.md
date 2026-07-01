# Code Risk Reviewer Round 010

Date: 2026-06-28

Read-only review completed after the initial `batch_017_dev` diff. No files were edited by the subagent, no Slurm jobs were submitted or cancelled, no training or evaluation was run, and no secrets were read.

## Blocker

- `experiments/batches/batch_017_dev.yaml` used text containing `standard embeddings`, which triggered the manifest validator's frozen-standard fusion text guard for a `standalone_main` candidate. `goal_submit_batch.py` would have rejected submission.

Resolution: parent changed the manifest wording to avoid the false-positive phrase. `scripts/goal_validate_manifest.py experiments/batches/batch_017_dev.yaml` now passes with only expected dev-only standalone warnings.

## High

- No real frozen-standard/fusion misuse was found. Both candidates use checkpoint `outputs/goal/runs/batch_016_dev/r016_standard_seeded_sampling/final` and no `fusion_standard_checkpoint_dir`, `fusion_alpha`, frozen-standard score, or frozen-standard embedding input.

## Medium

- `src/eval_mteb.py` initially did not include new `doc_chunk_*` and `lexical_*` fields in the `results_summary.csv` dedupe key.
  - Resolution: parent added these fields to the dedupe key.
- Fusion wrapper evaluation would have ignored `doc_chunk_*` and `lexical_*` parameters while still recording them in summary rows.
  - Resolution: parent added evaluator and manifest-validation guards forbidding `doc_chunk_*` or `lexical_*` with frozen-standard fusion.
- Chunked document evaluation may increase runtime and memory use.
  - Resolution: candidate is eval-only, dev-only, capped at `doc_chunk_max_chunks=8`, and budget/risk text records the runtime risk.

## Low

- `outputs/goal/state.json` had stale `current_batch` and `postprocess_job_id` values from `batch_016_dev`.
  - Resolution: state was factually updated for `batch_016_dev`; subsequent dry-run/submission will advance `current_batch` to `batch_017_dev`.
- `last_status_attempt.command` provenance was malformed after shell expansion.
  - Resolution: parent corrected it in state before submission.
