# literature_scout_round_012

## Assessment

Read-only local review only. No edits, no jobs, no training/eval, no web.

Final validation is protocol-consistent if it uses the same predeclared candidate rule from `batch_017_dev_repair`:

- checkpoint: `outputs/goal/runs/batch_016_dev/r016_standard_seeded_sampling/final`
- version: `standard_seeded_sampling`
- loop: `loop_idx=1`
- lexical rule: `lexical_hash_dim=1024`, `lexical_weight=0.15`
- no `fusion_standard_checkpoint_dir`, no `fusion_alpha`, no frozen-standard score/embedding input
- only change: `purpose: final`, all seven final tasks, new output paths

This does not change metric semantics. `src/eval_mteb.py` still calls MTEB retrieval evaluation and parses `ndcg_at_10` from raw MTEB output. The FiQA patch only disambiguates qrels loading to the cached `"default"` qrels config for `mteb/fiqa`; it does not alter rankings, metrics, embeddings, or score aggregation.

The lexical hash rule is candidate scoring by design, but it is the same candidate-only scoring rule already tested in `batch_017_dev_repair`. It remains `standalone_main` because candidate scores come from the candidate checkpoint plus deterministic lexical features, not from frozen-standard fusion or baseline scores.

## Local Evidence

- Protocol: `AGENTS.md`, `docs/goal_protocol.md`
- Lexical rule and FiQA patch: `src/eval_mteb.py`
- Repair manifest: `experiments/batches/batch_017_dev_repair.yaml`
- Repair scoreboard: `outputs/goal/runs/batch_017_dev_repair/scoreboard.json`
- Repair results: `outputs/goal/runs/batch_017_dev_repair/collected_results.csv`
- State note: `outputs/goal/state.json`
- Frozen baseline: `outputs/baselines/standard_frozen/results_summary.csv`

Repair result: `r017_seeded_lexical_hash__loop1` completed all four dev tasks with deltas `SciFact +0.03522`, `NFCorpus +0.00968`, `SCIDOCS +0.01346`, `FiQA2018 +0.00665`; mean `+0.01625`, min `+0.00665`. It is still not a final claim because `ArguAna`, `Touche2020`, and `TRECCOVID` were not evaluated.

## Risks

- Final tasks may regress, especially unseen `ArguAna`, `Touche2020`, `TRECCOVID`.
- Any final manifest must predeclare the same global rule; no per-task tuning or loop/task cherry-picking.
- Adding frozen-standard fusion, baseline scores, or standard embeddings would change the track to diagnostic/fusion, not `standalone_main`.
- FiQA loader patch is a narrow monkey patch around MTEB internals; dependency upgrades could need revalidation.
- Lexical hashing may inflate term-overlap behavior and hurt semantic tasks; final validation is necessary, not optional.
- Publishable certification still needs significance/bootstrap evidence if claimed beyond score-only.
