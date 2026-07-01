# Research Design Round 004

Date: 2026-06-21

This plan was written after `batch_010_dev` completed and before validating, dry-running, preflighting, or submitting the next batch. It uses only dev-task evidence and prior protocol history. It does not use final-task deltas to tune dev candidates.

## Trigger

`batch_010_dev` completed Slurm-native postprocess with `postprocess_done.json`, `collected_results.csv`, and `scoreboard.json`. It was dev-only and cannot trigger `main_goal_success`.

Local search remains exhausted. The two `standalone_main` candidates in `batch_010_dev` both regressed on all four dev tasks, while the diagnostic self-residual probe retained only the recurring SciFact/NFCorpus local signal and still regressed on FiQA2018 and SCIDOCS.

## What Has Been Tried

- Fusion diagnostic candidates in early batches showed useful loop signal only with frozen-standard scoring input. These remain diagnostic and cannot support `standalone_main`.
- Standalone mean-pool, recurrent mean-pool, no-memory, token-concat, final-loop, detached-memory, shorter-horizon, lower-negative-count, and document-loop variants did not produce a viable global dev signal.
- First-token loopwise training repeatedly improved SciFact and NFCorpus while regressing FiQA2018 and SCIDOCS.
- `batch_010_dev` falsified two broader first-token objective fixes:
  - tail-weighted loopwise loss: all four dev deltas were negative;
  - adjacent-loop consistency loss: all four dev deltas were negative.
- The `batch_010_dev` self-residual diagnostic remained non-global and non-main: SciFact and NFCorpus improved, FiQA2018 and SCIDOCS regressed, and candidate_track was `diagnostic`.

## Track Distinction

- `fusion_diagnostic` candidates may be useful for mechanism discovery but cannot trigger `main_goal_success`.
- `diagnostic` candidates, including candidate-internal probes such as self-residual query scoring, cannot trigger `main_goal_success`.
- `standalone_main` candidates must score with candidate-only inputs and no frozen-standard checkpoint, embedding, score, concatenation, ensemble, or interpolation.
- The next batch is dev-only, so even a strong result would require separate user-approved final validation before any main claim.

## Failure Pattern

- FiQA2018 is the most consistent blocker across recent standalone dev batches.
- SCIDOCS is the second recurring blocker; it occasionally moves near zero but has not become a reliable positive task.
- SciFact and NFCorpus are the only dev tasks with repeated local gains from first-token loop checkpoints.
- Batch 010 widened the objective search within the first-token family and still failed globally, which makes another loop-depth, memory-mode, detached-memory, doc-loop, or tiny hyperparameter neighborhood sweep low-value.

## Research Directions Considered

### Direction 1: True In-Batch Hybrid Contrastive Loss

- Core mechanism hypothesis: per-sample hard-negative CE is too local; adding a conservative in-batch positive classification term supplies broader discrimination without frozen-standard scoring.
- Code components likely affected: `src/losses.py`, `src/train.py`, `src/experiments.py`, Slurm train/export scripts, config, manifest.
- Why it may address the failure pattern: it changes the retrieval training signal rather than loop depth or memory state, directly testing whether SciFact/NFCorpus-only gains come from a narrow hard-negative objective.
- Expected risk: medium. Small batch size can create noisy in-batch negatives or false negatives.
- Estimated GPU cost: about 10 GPU hours.
- Smallest dev-only falsification: one first-token in-batch hybrid candidate, fixed loop 10, four dev tasks.
- Requires: training and evaluation.

### Direction 2: Pairwise Ranking Loss

- Core mechanism hypothesis: hard-negative softmax is too sharp and overfits RLHN negative ordering; pairwise softplus ranking may transfer better.
- Code components likely affected: `src/losses.py`, `src/train.py`, `src/experiments.py`, config, manifest.
- Why it may address the failure pattern: it tests a different ranking objective after lower negative count and loop-loss weighting failed.
- Expected risk: medium. Pairwise loss may weaken absolute discrimination.
- Estimated GPU cost: about 10 GPU hours.
- Smallest dev-only falsification: one first-token pairwise candidate, fixed loop 10, four dev tasks.
- Requires: training and evaluation.

### Direction 3: Seeded Random Positive/Negative Sampling

- Core mechanism hypothesis: using the first positive and first `k` negatives may encode ordering artifacts; seeded sampling could improve transfer.
- Code components likely affected: `src/data.py`, `src/train.py`, config, manifest.
- Why it may address the failure pattern: it changes the data distribution independently of loop mechanics and loss shape.
- Expected risk: medium. Sampling noise may reduce reproducibility if not carefully seeded.
- Estimated GPU cost: about 10 GPU hours.
- Smallest dev-only falsification: one seeded random-sampling first-token candidate on the four dev tasks.
- Requires: code changes, training, and evaluation.

### Direction 4: Query/Document Loop Co-Training

- Core mechanism hypothesis: training query loops against one-pass documents creates representation mismatch; training looped documents may reduce mismatch.
- Code components likely affected: `src/model.py`, `src/train.py`, `src/losses.py`, `src/eval_mteb.py`.
- Why it may address the failure pattern: document-loop evaluation alone did not help, but training-time symmetry is a distinct mechanism.
- Expected risk: medium to high. It changes encoding cost and may amplify document noise.
- Estimated GPU cost: 12 to 14 GPU hours.
- Smallest dev-only falsification: one fixed query/document loop training candidate and matching dev evaluation.
- Requires: training and evaluation.

### Direction 5: Loop-Depth Dropout Or Sparse Loop Supervision

- Core mechanism hypothesis: supervising every loop every batch makes loop states brittle; sparse loop supervision may regularize without changing scoring inputs.
- Code components likely affected: `src/losses.py`, `src/train.py`, config, manifest.
- Why it may address the failure pattern: it changes training pressure rather than probing nearby evaluation depths.
- Expected risk: medium. Stochastic loop selection needs strict seeding and auditability.
- Estimated GPU cost: about 10 GPU hours.
- Smallest dev-only falsification: one deterministic sparse loop schedule and fixed loop 10 evaluation.
- Requires: training and evaluation.

### Direction 6: Hardness Curriculum

- Core mechanism hypothesis: immediate hard-negative pressure harms cross-domain generalization; a deterministic curriculum from easier to harder negatives may reduce FiQA2018/SCIDOCS regressions.
- Code components likely affected: `src/data.py`, `src/train.py`, config, manifest.
- Why it may address the failure pattern: it attacks the training distribution rather than loop recurrence.
- Expected risk: medium to high. It assumes RLHN negative ordering correlates with hardness.
- Estimated GPU cost: 12 to 16 GPU hours.
- Smallest dev-only falsification: one predeclared curriculum schedule on four dev tasks.
- Requires: training and evaluation.

### Direction 7: Parameter-Free Pooling Alternative

- Core mechanism hypothesis: the pooled embedding choice may favor SciFact/NFCorpus and hurt FiQA2018/SCIDOCS; a candidate-only pooling alternative may shift transfer behavior without a new head.
- Code components likely affected: `src/model.py`, `src/eval_mteb.py`, config, manifest.
- Why it may address the failure pattern: it tests representation extraction rather than loop depth.
- Expected risk: medium. Pooling changes may degrade all tasks and complicate comparison.
- Estimated GPU cost: 8 to 10 GPU hours.
- Smallest dev-only falsification: one fixed pooling rule with first-token loop training and four dev tasks.
- Requires: training and evaluation.

## Ranking

1. True in-batch hybrid contrastive loss: high standalone validity, high novelty relative to local loop sweeps, direct training-signal test, moderate risk, clear falsification.
2. Pairwise ranking loss: high standalone validity, strong novelty, directly tests CE over-sharpness, moderate risk, clear falsification.
3. Seeded random positive/negative sampling: high validity and useful novelty, but data-path changes need extra audit and would exceed the current 24 GPU-hour budget if paired with two full-epoch objective tests.
4. Query/document loop co-training: valid and novel, but higher implementation risk and higher GPU cost.
5. Loop-depth dropout: plausible but closer to loop-local training pressure than the top two.
6. Hardness curriculum: potentially useful but higher-risk because hardness semantics are inferred from RLHN ordering.
7. Parameter-free pooling alternative: valid but less directly tied to the observed training-objective failure pattern.

## Selected Portfolio: batch_011_dev

The next batch should include two `standalone_main` dev candidates:

- `r011_inbatch_hybrid_first_token_t10`: tests Direction 1. It is included because it changes the candidate retrieval objective's negative structure without using frozen-standard scoring input. It falsifies whether broader contrastive signal can recover FiQA2018/SCIDOCS while preserving any SciFact/NFCorpus signal. Estimated cost: 10 GPU hours. Expected information gain: high.
- `r011_pairwise_rank_first_token_t10`: tests Direction 2. It is included because it changes the ranking loss shape independently of in-batch negatives. It falsifies whether hard-negative CE sharpness is the driver of the recurring transfer failure. Estimated cost: 10 GPU hours. Expected information gain: high.

Portfolio size: 2 candidates, 20 estimated GPU hours against a 24 GPU-hour batch budget, with at most 2 concurrent GPU jobs.

This is not an uninformative local sweep. It does not change only `loop_idx`, evaluate neighboring depths, adjust memory mode, or retest the same checkpoint. Both candidates train new standalone models with candidate-only scoring and predeclared global loop 10 evaluation across the dev tasks.

The seeded random-sampling direction is intentionally not included in this batch because adding it as a full-epoch candidate would exceed the configured 24 GPU-hour budget, while shrinking all candidates into short pilots would make negative results harder to interpret after prior capped runs were already ambiguous.
