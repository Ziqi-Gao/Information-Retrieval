# Subagent Round 011 Summary

Date: 2026-06-28

Required repository workflow gates from `docs/codex_subagents.md` were used before repair implementation/validation/submission for `batch_017_dev_repair`.

Real read-only subagents were used:

- `repo_auditor`: `docs/subagent_reports/repo_auditor_round_011.md`
- `literature_scout`: `docs/subagent_reports/literature_scout_round_011.md`
- `experiment_planner`: `docs/subagent_reports/experiment_planner_round_011.md`
- `code_risk_reviewer`: `docs/subagent_reports/code_risk_reviewer_round_011.md`

## Gate Findings

- The `r017_seeded_lexical_hash__loop1` failure is infrastructure/cache/config, not retrieval scoring.
- The root error is MTEB FiQA qrels loading with no config while the local cache contains `queries`, `corpus`, and `default`.
- A repair is valid if it reruns only the failed lexical candidate with the exact original candidate rule.
- Partial raw SciFact/NFCorpus outputs must not be backfilled into `results_summary.csv`.

## Required Resolution

- Apply a narrow repo-local FiQA qrels loader fix that selects the cached qrels `default` config.
- Create `batch_017_dev_repair` with only `r017_seeded_lexical_hash`, same checkpoint, same dev tasks, same `loop_idx=1`, same `lexical_hash_dim=1024`, same `lexical_weight=0.15`, and no frozen-standard fusion or score interpolation.
- Code-risk review found no blocker or high-severity protocol risks. Medium residual risks are limited to the private MTEB loader monkey-patch, dev-only interpretation, and ensuring partial original raw outputs are not backfilled.
