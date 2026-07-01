# Result Analyst Round 010

Date: 2026-06-28

Read-only analysis of `outputs/goal/runs/batch_016_dev/scoreboard.json`, `scoreboard.csv`, `collected_results.csv`, and `experiments/batches/batch_016_dev.yaml`.

`batch_016_dev` is `purpose: dev`, candidate track `standalone_main`, and evaluated only `SciFact`, `NFCorpus`, `FiQA2018`, and `SCIDOCS`.

## Candidate Results

| Candidate | Dev deltas, SciFact / NFCorpus / SCIDOCS / FiQA2018 | Min | Mean | Flags |
| --- | --- | ---: | ---: | --- |
| `r016_standard_seeded_sampling__loop1` | `+0.00834 / +0.00045 / +0.00386 / +0.00086` | `+0.00045` | `+0.00338` | all success flags false |
| `r016_standard_inbatch_hybrid__loop1` | `-0.00686 / -0.00132 / +0.00101 / -0.00142` | `-0.00686` | `-0.00215` | all success flags false |
| `r016_qdoc_final_mean_pool_t3_neg3__loop3` | `-0.07807 / -0.04792 / -0.02024 / -0.05100` | `-0.07807` | `-0.04931` | all success flags false |

`main_goal_success=false` for the batch and every candidate. Reasons: the purpose is not `final`, only four of seven final tasks are valid, `ArguAna`, `Touche2020`, and `TRECCOVID` are missing by design, and no candidate satisfies the final thresholds.

## Dev Signal Assessment

No candidate gives a protocol-viable global dev signal. `r016_standard_seeded_sampling__loop1` is weakly interesting because all four observed deltas are positive, but two tasks are below `+0.001`, two are below `+0.002`, and mean delta is below `+0.005`.
