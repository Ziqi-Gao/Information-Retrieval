# Code Risk Reviewer Round 011

Read-only review only. No files edited, no jobs submitted, no `sbatch`, no training/eval run.

## Verdict

No blocker or high-severity protocol risks found for `batch_017_dev_repair`.

The diff is consistent with a narrow infrastructure repair: FiQA qrels loading is disambiguated to the `default` qrels config, and the repair manifest reruns only the failed `r017_seeded_lexical_hash` candidate with the original candidate rule.

## Protocol Compliance

- Metric semantics unchanged: no NDCG parsing/scoring changes; only qrels dataset config selection for FiQA.
- Candidate rule preserved: same r016 checkpoint, `loop_idx=1`, `lexical_hash_dim=1024`, `lexical_weight=0.15`.
- No frozen-standard fusion or interpolation added: repair manifest has no `fusion_standard_checkpoint_dir`, `fusion_alpha`, `fusion_scope`, ensemble, or standard-score input.
- Same dev tasks: `SciFact`, `NFCorpus`, `FiQA2018`, `SCIDOCS`.
- Repair scope is correct: only failed `r017_seeded_lexical_hash` is included; completed `r017_seeded_chunked_docs` is not rerun.
- Output isolation is correct by batch id: `batch_017_dev_repair` should avoid overwriting `batch_017_dev`.

## Risks

- Medium: `src/eval_mteb.py` monkey-patches MTEB private API `HFDataLoader._load_qrels`. This is acceptable for the documented local `mteb==1.39.7` repair, but could break if MTEB changes that method signature or attributes.
- Medium: The repair remains dev-only evidence. Even if it succeeds, it cannot support final success without separate final validation.
- Medium: Partial raw SciFact/NFCorpus outputs from the failed original eval must remain excluded from scoring/backfill; only a complete repaired four-task summary should be interpreted.

No additional blockers found.
