# literature_scout_round_013

## Scope

Read-only local cache-loader scout. No edits, no `sbatch`, no training/eval, no web.

## Finding

`batch_018_final` failed during `ArguAna` data loading, before metric/scoring output was written.

Evidence:

- `slurm_logs/loopmat_eval_5386592.out`: eval reached `ArguAna` after `SciFact`, `NFCorpus`, `SCIDOCS`, `FiQA2018`.
- `slurm_logs/loopmat_eval_5386592.err`: failure in `HFDataLoader._load_corpus()`: `FileNotFoundError` for `hf://datasets/mteb/arguana.../corpus.jsonl`.
- `.hf_cache/datasets/mteb___arguana/` contains local configs:
  - `corpus`: `_id`, `title`, `text`, 8674 docs
  - `queries`: `_id`, `text`, 1406 queries
  - `default`: qrels `query-id`, `corpus-id`, `score`, test 1406 rows
- `src/eval_mteb.py` currently patches only FiQA qrels. That patch does not affect ArguAna corpus loading.
- `outputs/goal/runs/batch_018_final/scoreboard.csv`: 0 valid tasks; all seven marked `missing results_summary.csv`.
- Partial raw JSON exists for early tasks, but no top-level `results_summary.csv`; do not backfill.

## Likely Root Cause

This is an MTEB/HF datasets cache-loading infrastructure failure, not a candidate-scoring failure. MTEB generic retrieval loading calls `load_dataset(hf_repo, "corpus")`, `load_dataset(hf_repo, "queries")`, and config-less qrels loading. For ArguAna, local Arrow cache exists, but datasets attempted to resolve the remote `corpus.jsonl` path and failed.

## Safe Fix Scope

Safe infrastructure-only repair may extend the FiQA-style loader patch into a guarded cached-BEIR loader fallback:

- use existing local cache configs: `corpus`, `queries`, `default`
- preserve MTEB transformations: `_id` cast to string, `_id -> id`, qrels cast to `query-id/corpus-id/score`
- do not touch `LoopRetrieverMTEBWrapper`, lexical hash rule, embeddings, rankings, MTEB evaluator, metric parsing, thresholds, baseline, or candidate manifest
- write repair outputs under a new `batch_018_final_repair` path

Not safe: changing final task list, changing lexical params, editing metric extraction, using old partial raw rows, patching site-packages, or broad unguarded defaults for every dataset.

## Blockers / High Risks

- Current `batch_018_final` is invalid for final claims: no valid final-task rows.
- FiQA-only qrels patch is insufficient for ArguAna.
- A generic cache fallback could silently affect other datasets if not guarded and schema-checked.
- Hardcoded absolute cache paths would violate repo hygiene; derive from `HF_DATASETS_CACHE` / `HF_HOME`.
- Even after loader repair, final validation can still fail on `ArguAna`, `Touche2020`, or `TRECCOVID` numerically.

## Minimal Regression Check Idea

After an approved code fix, run a CPU-only loader seam test, not MTEB eval:

- load patched `HFDataLoader(hf_repo="mteb/arguana").load("test")`
- assert corpus count `8674`, qrels count `1406`, query IDs filtered to qrels, qrels schema valid
- repeat FiQA qrels seam to ensure prior patch still works
- run compile/syntax checks before any repair submission
