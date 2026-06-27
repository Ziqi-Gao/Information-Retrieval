# experiment_planner report

## Gate Status

The planner read existing artifacts only. `batch_014_dev` has `postprocess_done.json` and `scoreboard.json`, and both candidates failed on all four dev tasks. `outputs/goal/state.json` remained stale with `phase=SUBMIT_BATCH` and postprocess/open-job state not reconciled, so the parent must update state and write a new research design note before creating/submitting a new manifest.

## Conditional batch_015_dev Proposal

The planner recommended a two-candidate dev-only portfolio under `max_concurrent_gpu_jobs: 2`, `max_gpu_hours_estimate: 22`, with both candidates outside the first-token loop-memory family:

1. `r015_qdoc_mean_pool_t4_neg3`
   - Track: `standalone_main` dev-only.
   - Hypothesis: persistent FiQA2018/SCIDOCS regressions may come from looped-query vs one-pass-document representation asymmetry.
   - Mechanism: q/doc loop co-training with `loop_memory_mode=mean_pool`, `loop_query_mode=initial_embedding`, `tmax=4`, `num_negatives=3`, reduced dev training sample; train/eval query loop 4 and doc loop 4.
   - Risk: high; doc-loop backprop may be slow or OOM.
   - Estimated GPU: about 12h.
   - Required changes: registered version/loss path for q/doc loopwise training; extend train batch encoding to loop positives/negatives; config; use existing eval `loop_docs: true`, `doc_loop_idx: 4`.

2. `r015_seeded_mean_pool_t10`
   - Track: `standalone_main` dev-only.
   - Hypothesis: seeded sampling helped SCIDOCS in `batch_012`, but first-token memory kept FiQA2018 negative; mean-pool memory may retain sampling benefit with less first-token drift.
   - Mechanism: mean-pool loopwise training with `passage_sampling_strategy=seeded_random`, fixed `loop_idx=10`.
   - Risk: medium; may reproduce old mean-pool FiQA regression or seeded-sampling NFCorpus regression.
   - Estimated GPU: about 10h.
   - Required changes: register `loop_matryoshka_seeded_mean_pool`; add config.

## Parent Decision

The parent did not select q/doc loop co-training for this resume cycle because of high OOM/timeout risk and the user's one-batch limit. The selected portfolio instead follows the lower-risk literature-scout recommendation: role prompting, dimensional MRL, and their combination as no-loop single-vector candidates outside the first-token loop-memory family.

## Uncertainty

The planner proposal was conditional and read-only. It did not validate code or manifests.
