# Research Design Round 005

Date: 2026-06-21

This plan was written after `batch_011_dev` completed Slurm-native postprocess and before validating, dry-running, preflighting, or submitting the next batch. It uses only dev-task evidence and prior protocol history. It does not use final-task deltas to tune dev candidates.

## Trigger

`batch_011_dev` completed with `postprocess_done.json`, `collected_results.csv`, and `scoreboard.json`. It was dev-only and cannot trigger `main_goal_success`.

Both `standalone_main` candidates regressed on all four dev tasks:

- `r011_inbatch_hybrid_first_token_t10__loop10`: mean delta `-0.006275`, min delta `-0.01287`.
- `r011_pairwise_rank_first_token_t10__loop10`: mean delta `-0.02554`, min delta `-0.03389`.

Local search is exhausted. Recent standalone batches have already tested checkpoint-independent scoring, loop index, memory modes, first-token memory, detached memory, shorter horizon, lower negative count, document-loop evaluation, tail-weighted loop loss, adjacent-loop consistency, in-batch hybrid loss, and pairwise ranking loss without a viable global dev signal.

## Track Distinction

- `fusion_diagnostic` candidates helped reveal loop signal in early batches, but they use frozen-standard scoring input or standard+candidate fusion and can never trigger `main_goal_success`.
- `diagnostic` candidates, including candidate-internal probes, may be reported but cannot trigger the main goal.
- `standalone_main` candidates must score only with candidate pipeline outputs and no frozen-standard checkpoint, embedding, score, ensemble, weighted standard+loop concatenation, or interpolation.
- `batch_012_dev` remains dev-only, so even a strong global dev signal would require separate user-approved final validation.

## Failure Pattern

- FiQA2018 is the most consistent regression across recent standalone dev batches.
- SCIDOCS is the second recurring blocker; some diagnostic or repaired candidates came close to zero but did not become reliably positive.
- SciFact and NFCorpus are the only tasks that repeatedly showed local gains under first-token loop variants.
- `batch_010_dev` and `batch_011_dev` widened the loss/objective search and still produced all-negative standalone dev candidates.
- Another loop index sweep, memory-mode tweak, neighboring checkpoint evaluation, or tiny loss hyperparameter perturbation is low-value because those neighborhoods have already failed to change the global pattern.

## Research Directions Considered

### Direction 1: Seeded Positive/Negative Sampling

- Core mechanism hypothesis: fixed first-positive and first-k-negative selection in RLHN may encode ordering artifacts and overfit the candidate to a narrow training distribution.
- Code components likely affected: `src/data.py`, `src/train.py`, `src/experiments.py`, config, manifest.
- Why it may address the failure pattern: changes the training examples seen by the same standalone first-token loop family without changing final scoring semantics.
- Expected risk: medium. Sampling can add noise, so the rule must be deterministic and checkpoint-auditable.
- Estimated GPU cost: about 10 GPU hours.
- Smallest dev-only falsification: one full-epoch first-token candidate using deterministic seeded random positive/negative sampling, fixed loop 10, four dev tasks.
- Requires: training and evaluation.

### Direction 2: Parameter-Free Pooling Alternative

- Core mechanism hypothesis: mean-pooled loop embeddings may be misaligned with first-token memory states; first-token retrieval pooling may reduce representation drift without adding trainable heads.
- Code components likely affected: `src/model.py`, `src/train.py`, `src/experiments.py`, config, manifest.
- Why it may address the failure pattern: tests representation extraction rather than another loop-depth or loss-family neighborhood.
- Expected risk: medium. First-token pooling may underperform mean pooling on retrieval passages.
- Estimated GPU cost: about 10 GPU hours.
- Smallest dev-only falsification: one full-epoch first-token memory candidate with first-token query/doc embedding pooling, fixed loop 10, four dev tasks.
- Requires: training and evaluation.

### Direction 3: Query/Document Loop Co-Training

- Core mechanism hypothesis: query loops trained against one-pass documents create representation mismatch; training query and document loop states together may reduce transfer regressions.
- Code components likely affected: `src/model.py`, `src/train.py`, `src/losses.py`, `src/eval_mteb.py`.
- Why it may address the failure pattern: document-loop evaluation alone was not sufficient, but training-time symmetry is a distinct mechanism.
- Expected risk: medium to high. It increases training cost and may amplify document-side noise.
- Estimated GPU cost: 14 to 18 GPU hours.
- Smallest dev-only falsification: one fixed q/doc loop co-training candidate and matching dev evaluation.
- Requires: training and evaluation.

### Direction 4: Hardness Curriculum

- Core mechanism hypothesis: immediate exposure to the hardest RLHN negatives may damage cross-domain generalization; a deterministic negative-window curriculum may reduce FiQA2018/SCIDOCS regressions.
- Code components likely affected: `src/data.py`, `src/train.py`, config, manifest.
- Why it may address the failure pattern: attacks training distribution pressure instead of recurrence mechanics.
- Expected risk: medium to high. It assumes RLHN negative order correlates with hardness.
- Estimated GPU cost: 12 to 16 GPU hours.
- Smallest dev-only falsification: one predeclared curriculum schedule, fixed loop 10, four dev tasks.
- Requires: training and evaluation.

### Direction 5: Sparse Loop Supervision

- Core mechanism hypothesis: supervising every loop every batch over-constrains the trajectory; deterministic sparse loop supervision may regularize loop drift.
- Code components likely affected: `src/losses.py`, `src/train.py`, `src/experiments.py`, config, manifest.
- Why it may address the failure pattern: changes supervision topology while preserving standalone scoring.
- Expected risk: medium. It is closer to prior loop-loss local search than the top two directions.
- Estimated GPU cost: about 10 GPU hours.
- Smallest dev-only falsification: one deterministic sparse schedule such as loops `{1, 4, 7, 10}`, fixed loop 10 dev evaluation.
- Requires: training and evaluation.

### Direction 6: Two-Stage Standard-To-Loop Training

- Core mechanism hypothesis: direct full loopwise optimization may destabilize the baseline representation; a short standard warmup followed by loopwise training may preserve broad retrieval ability.
- Code components likely affected: `src/train.py`, `src/experiments.py`, config, manifest.
- Why it may address the failure pattern: tests optimization schedule rather than another loop architecture tweak.
- Expected risk: medium. It increases codepath complexity and may simply reproduce the weak standard/loop compromise.
- Estimated GPU cost: 12 to 16 GPU hours.
- Smallest dev-only falsification: one predeclared warmup fraction followed by first-token loopwise training.
- Requires: training and evaluation.

### Direction 7: Label-Smoothed Listwise Ranking

- Core mechanism hypothesis: hard CE and pairwise softplus both failed; a listwise objective with conservative label smoothing may reduce overconfidence without discarding the positive-vs-negative list structure.
- Code components likely affected: `src/losses.py`, `src/train.py`, `src/experiments.py`, config, manifest.
- Why it may address the failure pattern: tests loss calibration after sharp CE and pairwise objectives both failed.
- Expected risk: medium. It remains a loss-family change and could be another local objective variant.
- Estimated GPU cost: about 10 GPU hours.
- Smallest dev-only falsification: one listwise smoothed candidate, fixed loop 10, four dev tasks.
- Requires: training and evaluation.

## Ranking

1. Seeded positive/negative sampling: high standalone validity, strongest novelty versus recent objective-local failures, moderate implementation risk, clear falsification, 10 GPU hours.
2. Parameter-free pooling alternative: high standalone validity, orthogonal to data sampling and loss changes, moderate risk, clear falsification, 10 GPU hours.
3. Query/document loop co-training: valid and novel, but higher implementation risk and cost.
4. Hardness curriculum: valid and data-focused, but relies on inferred RLHN hardness order.
5. Sparse loop supervision: valid, but closer to previous loop-loss local search.
6. Two-stage standard-to-loop training: valid, but schedule complexity is higher and falsification is less clean.
7. Label-smoothed listwise ranking: valid, but less novel after `batch_011_dev` objective failures.

## Selected Portfolio: batch_012_dev

The next batch should include two `standalone_main` dev candidates:

- `r012_seeded_sampling_first_token_t10`
  - Tests Direction 1.
  - Included because it attacks a still-untested data-construction failure mode rather than loop depth, memory mode, or loss family.
  - Failure mode falsified: fixed RLHN first-positive/first-k-negative sampling is not the main cause of global dev regressions.
  - Estimated cost: 10 GPU hours.
  - Expected information gain: high.
  - Candidate track: `standalone_main`.

- `r012_first_pool_first_token_t10`
  - Tests Direction 2.
  - Included because it is orthogonal to data sampling and tests whether retrieval embedding extraction is the bottleneck.
  - Failure mode falsified: mean pooling is not the main cause of the recurring FiQA2018/SCIDOCS regressions.
  - Estimated cost: 10 GPU hours.
  - Expected information gain: high.
  - Candidate track: `standalone_main`.

Portfolio size: 2 candidates, 20 estimated GPU hours against the configured 24 GPU-hour batch limit, with at most 2 concurrent GPU jobs.

This is not an uninformative local sweep. It does not change only `loop_idx`, evaluate neighboring depths of a failed checkpoint, or adjust memory mode around the same failed result. The two candidates test distinct mechanisms: data sampling and parameter-free embedding extraction.

Sparse loop supervision is intentionally not included in this resume cycle because adding it would push the batch beyond the 24 GPU-hour budget, and it is less mechanistically distinct from the recent loop-objective failures.
