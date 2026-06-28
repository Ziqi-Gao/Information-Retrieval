# Repo Auditor Round 011

## Verdict

`r017_seeded_lexical_hash__loop1` failed from an infrastructure/cache/config issue, not from the candidate retrieval rule. A repair rerun is valid if it preserves the exact candidate rule and remains dev-only; the original `batch_017_dev` remains failed and partial raw outputs must not be backfilled.

## Evidence

- `submission_plan.json` fixed the original rule: `standard_seeded_sampling`, r016 checkpoint, `LOOP_IDX=1`, `LEXICAL_HASH_DIM=1024`, `LEXICAL_WEIGHT=0.15`, dev tasks `SciFact;NFCorpus;FiQA2018;SCIDOCS`.
- `batch_manifest.submitted.yaml` labels the candidate as candidate-only dense plus deterministic lexical hash features, with no frozen-standard scoring input and `claim_track: standalone_main`.
- `slurm_logs/loopmat_eval_5383288.out` shows eval entered SciFact, NFCorpus, and then FiQA2018.
- `slurm_logs/loopmat_eval_5383288.err` shows the actual failure in HF datasets cache loading: `mteb/fiqa` has multiple cached configs `queries`, `corpus`, and `default`, and the qrels load did not specify a config.
- The stack is in `task.load_data` / `load_dataset`, not lexical hash encoding or metric extraction.
- `collected_results.csv`, `per_run_validation.json`, and `scoreboard.json` correctly mark the candidate as `missing_result` with `tasks_valid=0`.

## Repair Validity

`batch_017_dev_repair` is valid if it preserves the same checkpoint, version, task list, global `loop_idx=1`, `lexical_hash_dim=1024`, `lexical_weight=0.15`, and no fusion or standard-score inputs. Changing only cache/data-loading behavior, batch id, and output directory is infrastructure repair rather than candidate tuning.

## Blockers And High Risks

- The FiQA cache ambiguity must be fixed before rerun.
- The repair must write to `batch_017_dev_repair` output paths, not overwrite `batch_017_dev`.
- Do not backfill `results_summary.csv` from partial raw SciFact/NFCorpus outputs.
- The repaired run needs complete SciFact, NFCorpus, FiQA2018, and SCIDOCS rows before scoring.
- Any claim remains dev-only and cannot trigger final success.
