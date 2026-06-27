# result_analyst report

## Candidate Summary

`batch_014_dev` is `purpose=dev`. Both candidates are `candidate_track=standalone_main`, and no fusion fields are populated. Slurm-native postprocess completed, and all expected dev-task rows completed. The batch covers only 4/7 protocol final tasks by design.

| candidate | version | valid/final tasks | wins | mean delta | min delta | verdict |
|---|---|---:|---:|---:|---:|---|
| `r014_sparse_late_first_token_t10__loop10` | `loop_sparse_first_token` | 4/7 | 0 | -0.009035 | -0.01268 | dev failed |
| `r014_label_smooth_first_token_t10__loop10` | `loop_label_smooth_first_token` | 4/7 | 0 | -0.013278 | -0.02339 | dev failed |

## Per-Task Deltas

| task | baseline | sparse | delta | label_smooth | delta |
|---|---:|---:|---:|---:|---:|
| SciFact | 0.53932 | 0.53348 | -0.00584 | 0.51593 | -0.02339 |
| NFCorpus | 0.23849 | 0.22819 | -0.01030 | 0.22474 | -0.01375 |
| SCIDOCS | 0.13671 | 0.12939 | -0.00732 | 0.13161 | -0.00510 |
| FiQA2018 | 0.25855 | 0.24587 | -0.01268 | 0.24768 | -0.01087 |
| ArguAna | 0.50874 | missing | n/a | missing | n/a |
| Touche2020 | 0.19917 | missing | n/a | missing | n/a |
| TRECCOVID | 0.67118 | missing | n/a | missing | n/a |

## Invalid Or Missing Results

Missing final-task results for both candidates: `ArguAna`, `Touche2020`, and `TRECCOVID`.

This is expected from the dev manifest/submission plan, which evaluated only `SciFact`, `NFCorpus`, `FiQA2018`, and `SCIDOCS`. Per-run validation reports both runs `completed` for those expected dev tasks, so there are no hidden failed eval rows in the collected results.

## Failure Patterns

Both standalone candidates regress on every completed dev task. No task reaches even the weak diagnostic `+0.001` margin, and none reaches the main `+0.002` task margin. Sparse late-loop supervision is less bad than label smoothing overall, but still uniformly negative.

The strongest regressions are:

- sparse: `FiQA2018` -0.01268, `NFCorpus` -0.01030
- label smoothing: `SciFact` -0.02339, `NFCorpus` -0.01375

## Recommendation

There is no viable global dev signal. This batch argues against both mechanisms as currently configured: sparse loop supervision and label-smoothed listwise loss both lose globally on the evaluated dev set.

`main_goal_success` cannot be true. Reasons: `purpose=dev`, only 4/7 final tasks valid, three final tasks missing, every observed final-task delta is negative, macro mean delta is negative, and `no_task_regression=false`.
