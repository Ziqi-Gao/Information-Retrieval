# Result Analyst Round 011: batch_017_dev_repair

Batch `batch_017_dev_repair` completed postprocess successfully at `2026-06-28T19:20:38Z`. The repair run has one eval-only dev candidate.

## Candidate Summary

- `candidate_id`: `r017_seeded_lexical_hash__loop1`
- `run_id`: `r017_seeded_lexical_hash`
- `candidate_track`: `standalone_main`
- `purpose`: `dev`
- `version`: `standard_seeded_sampling`
- Candidate rule: reuse `r016_standard_seeded_sampling` checkpoint, fixed `loop_idx=1`, `lexical_hash_dim=1024`, `lexical_weight=0.15`, candidate dense embeddings plus deterministic lexical hash features.
- Evaluated dev tasks: `SciFact`, `NFCorpus`, `FiQA2018`, `SCIDOCS`

## Dev Deltas vs Frozen Standard

| Task | Baseline | Candidate | Delta | Main Margin |
|---|---:|---:|---:|---|
| SciFact | 0.53932 | 0.57454 | +0.03522 | pass |
| NFCorpus | 0.23849 | 0.24817 | +0.00968 | pass |
| SCIDOCS | 0.13671 | 0.15017 | +0.01346 | pass |
| FiQA2018 | 0.25855 | 0.26520 | +0.00665 | pass |

## Scoreboard Fields

- `tasks_total`: 7
- `tasks_valid`: 4
- `tasks_won`: 4
- `tasks_lost`: 3, due missing final-only tasks `ArguAna`, `Touche2020`, `TRECCOVID`
- Evaluated dev won/lost: 4/0
- `tasks_at_main_margin`: 4
- `min_delta`: `+0.00665`
- `mean_delta`: `+0.0162525`
- `minimal_positive_signal`: `false`
- `research_grade_threshold_pass`: `false`
- `fusion_diagnostic_pass`: `false`
- `main_goal_success`: `false`
- `publishable_score_candidate`: `false`

## Protocol Interpretation

This is a strong viable global dev signal: all four evaluated dev tasks are valid, non-regressing, and exceed the `+0.002` final-task margin, with dev macro mean above `+0.005`.

`main_goal_success` is not true. The batch is `purpose: dev` and covers only 4/7 protocol final tasks; `ArguAna`, `Touche2020`, and `TRECCOVID` are missing by design. Final claims require a separate predeclared `purpose: final` run over all seven final tasks.
