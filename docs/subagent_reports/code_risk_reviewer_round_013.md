# Code Risk Reviewer Round 013

Read-only review only. No files edited, no jobs submitted, no `sbatch`, no training/eval run.

## Verdict

No blocker or high-severity protocol risk found.

The `src/eval_mteb.py` patch is infrastructure-only: it changes MTEB/HF retrieval dataset loading to use explicit `mteb/*` configs for `corpus`, `queries`, and qrels `default`, while preserving schema normalization. It does not change metric parsing, NDCG semantics, ranking logic, embeddings, lexical hashing, thresholds, baseline comparison, or candidate scoring.

## Protocol Check

- `batch_018_final_repair` preserves exact candidate rule from `batch_018_final`: same checkpoint, `standard_seeded_sampling`, `loop_idx=1`, `candidate_loop_indices=[1]`, `lexical_hash_dim=1024`, `lexical_weight=0.15`.
- `purpose: final`, `claim_track: standalone_main`, all seven final tasks in protocol order.
- No forbidden fields: no `fusion_standard_checkpoint_dir`, `fusion_alpha`, `fusion_scope`, self-query, doc chunking, standard scores, or interpolation.
- Metric/thresholds unchanged: `primary_metric: ndcg_at_10`, `win_margin: 0.001`.
- Budget valid against state limits: `1` GPU job and `12` GPU hours are below `4` and `24`; `allow_submit: true` matches user-approved repair context.
- New batch id/output path avoids overwriting `batch_018_final`.

## Risks

- Medium: loader patch monkey-patches MTEB private `HFDataLoader` methods for all `mteb/*` retrieval repos. It is guarded and schema-preserving, but should be rechecked if MTEB/datasets versions change.
- Medium: `_load_mteb_config` falls back to normal datasets resolution if local-only cache loading fails. That preserves explicit config semantics, but can still depend on remote/cache behavior.
- Medium: `outputs/goal/state.json` still shows old `batch_018_final` postprocess as running, although `postprocess_done.json` and scoreboard exist. This is state hygiene, not a manifest blocker.
- Scientific high risk remains: final held-out tasks may still regress; any missing/failed/duplicate/NaN/partial row or any final delta below `+0.002` invalidates main success.

## Remediation

No blocker/high remediation required before validation. Recommended before submit: run a cheap loader seam test for ArguAna and FiQA, compile/syntax checks, manifest validation, dry-run with postprocess, scoreboard self-test, preflight, and refresh/update state to mark `batch_018_final` postprocess completed.
