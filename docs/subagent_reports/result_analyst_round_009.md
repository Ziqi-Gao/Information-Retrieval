# Result Analyst Round 009: batch_015_dev

Based on the current claim-track protocol, `batch_015_dev` only has dev subset results: `SciFact`, `NFCorpus`, `SCIDOCS`, and `FiQA2018`. `ArguAna`, `Touche2020`, and `TRECCOVID` were not evaluated, so no final, main, or publishable claim is valid. `min_delta` and `mean_delta` below are over the 4 valid dev tasks in the scoreboard.

| run_id | candidate_id | track | purpose | dev deltas | min / mean | won / lost | minimal_positive_signal | research_grade_threshold_pass | fusion_diagnostic_pass | main_goal_success | publishable_score_candidate |
|---|---|---|---|---|---:|---:|---|---|---|---|---|
| `r015_role_dim_mrl_standard` | `r015_role_dim_mrl_standard__loop1` | `standalone_main` | `dev` | SciFact `-0.00320`; NFCorpus `+0.00862`; SCIDOCS `+0.00562`; FiQA2018 `-0.00255` | `-0.00320` / `+0.00212` | dev `2/2`; protocol `2/5` | false | false | false | false | false |
| `r015_role_prompt_standard` | `r015_role_prompt_standard__loop1` | `standalone_main` | `dev` | SciFact `-0.00730`; NFCorpus `+0.00910`; SCIDOCS `+0.00185`; FiQA2018 `-0.00714` | `-0.00730` / `-0.00087` | dev `2/2`; protocol `2/5` | false | false | false | false | false |
| `r015_dim_mrl_standard` | `r015_dim_mrl_standard__loop1` | `standalone_main` | `dev` | SciFact `-0.00966`; NFCorpus `+0.00040`; SCIDOCS `+0.00132`; FiQA2018 `-0.00183` | `-0.00966` / `-0.00244` | dev `1/3`; protocol `1/6` | false | false | false | false | false |

## Decision

No strong viable global dev signal.

The best candidate is `r015_role_dim_mrl_standard__loop1`, but its positive macro dev mean is task-selective, not global: it wins `NFCorpus` and `SCIDOCS` while regressing on both `SciFact` and `FiQA2018`. The other two candidates are weaker, with negative mean deltas.

## Failure Pattern

All three `standalone_main` candidates regress on `SciFact` and `FiQA2018`. Gains concentrate on `NFCorpus` and, more weakly, `SCIDOCS`. The combined role+dim MRL variant improves the mean relative to either component alone, but it still fails the non-regression requirement and does not justify final validation planning under the current protocol.
