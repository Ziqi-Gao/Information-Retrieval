# Research Design Round 001

Date: 2026-06-20

This plan was written after `batch_007_dev` completed and before creating any new batch. It follows the Local Search Exhaustion and RESEARCH_DESIGN_MODE rules. It uses only dev-task evidence for next dev planning.

## Trigger

Local search is exhausted.

Recent standalone_main dev batches have mostly tested existing checkpoints, nearby loop depths, and memory-mode variants. They have not produced a viable global dev signal. `batch_007_dev` was a local-neighborhood sweep around the `batch_006_dev` first-token checkpoint and did not remove the recurring FiQA2018 and SCIDOCS regressions.

## What Has Been Tried

- `batch_001_dev` and `batch_002_dev` tested frozen-baseline fusion diagnostics. These results were useful for diagnosing whether a loop signal exists, but they are not standalone_main and cannot trigger main goal success.
- `batch_003_final` and `batch_003_final_repair` were fusion_diagnostic final validation attempts. They remain diagnostic only and are not used to tune the next dev candidate.
- `batch_004_dev` evaluated standalone mean-pool and recurrent mean-pool checkpoints at fixed loop depths on dev tasks. No candidate showed a viable global dev signal.
- `batch_005_dev` evaluated existing final-loop and recurrent no-memory standalone checkpoints. These probes were broadly negative on dev tasks.
- `batch_006_dev` trained first-token and token-concat memory variants. First-token improved SciFact and NFCorpus but regressed on SCIDOCS and FiQA2018; token-concat regressed across all dev tasks.
- `batch_007_dev` evaluated nearby loop depths 5, 6, 8, and 9 from the `batch_006_dev` first-token checkpoint. All candidates kept the same mixed pattern: SciFact and NFCorpus improved, while FiQA2018 and SCIDOCS regressed.

## Track Distinction

- Fusion diagnostics may reveal that a loop representation contains useful signal, but they use frozen-baseline scoring input and cannot satisfy `standalone_main`.
- The next dev batch must be `standalone_main` or `diagnostic`, must avoid frozen-baseline scoring input, and must not use final-task deltas for tuning.
- `main_goal_success` remains impossible from a dev batch, even if every dev task improves.

## Standalone Failure Pattern

Recent standalone_main dev results show a consistent split:

- SciFact and NFCorpus can improve under first-token loop memory.
- FiQA2018 is the most consistent and largest dev regression.
- SCIDOCS is often slightly negative or below margin.
- Recurrent hidden-state variants are broadly unstable and much worse than the frozen baseline.
- Token-concat memory is higher cost and produced no useful dev signal.

Another local loop-depth or memory-mode sweep is low-value because `batch_007_dev` already tested nearby depths around the best first-token checkpoint and preserved the same failure shape. Further local probing would mostly estimate the same tradeoff more finely rather than test a new mechanism.

## Research Directions Considered

### Direction 1: Final-Loop Objective With First-Token Memory

- Core mechanism hypothesis: the loopwise loss may over-constrain intermediate loops and amplify task-specific drift; supervising only the final loop may preserve first-token gains while reducing FiQA2018 and SCIDOCS regressions.
- Code components likely affected: `src/experiments.py` registry only, using existing `final_loop_loss` and first-token memory in `src/model.py`.
- Why it may address the failure pattern: it changes the optimization target while keeping the strongest dev memory mode from `batch_006_dev`.
- Expected risk: medium. The final-loop objective previously underperformed for mean-pool memory, but it has not been tested with first-token memory.
- Estimated GPU cost: about 7 GPU hours for one dev train/eval candidate with capped steps.
- Smallest falsifying dev test: train one first-token final-loop candidate and evaluate one fixed global loop depth on SciFact, NFCorpus, FiQA2018, and SCIDOCS.
- Requires: training and evaluation.

### Direction 2: Detached First-Token Memory Training

- Core mechanism hypothesis: gradients through the recurrent memory path may let task-specific query drift accumulate; detaching memory may regularize the loop while preserving the first-token signal.
- Code components likely affected: config only, using existing `detach_memory` support in `src/model.py` and `src/train.py`.
- Why it may address the failure pattern: it targets instability without changing scoring semantics or adding parameters.
- Expected risk: medium. Detachment may also remove useful refinement and reduce SciFact/NFCorpus gains.
- Estimated GPU cost: about 7 GPU hours for one dev train/eval candidate with capped steps.
- Smallest falsifying dev test: train first-token loopwise with `detach_memory: true` and evaluate one fixed global loop depth on the dev tasks.
- Requires: training and evaluation.

### Direction 3: Short-Horizon Loopwise Training

- Core mechanism hypothesis: training with `tmax=10` may reward long-horizon behavior that helps some factoid tasks but harms broader retrieval; a shorter training horizon may reduce over-iteration drift.
- Code components likely affected: config only.
- Why it may address the failure pattern: it changes the training problem, not just the evaluation loop depth, and may produce a representation whose final loop is less specialized.
- Expected risk: low to medium. It may simply remove the positive loop signal.
- Estimated GPU cost: about 5 GPU hours for one shorter dev train/eval candidate.
- Smallest falsifying dev test: train first-token loopwise with `tmax=4` and evaluate loop 4 globally on dev tasks.
- Requires: training and evaluation.

### Direction 4: Single-Loop Standalone Control

- Core mechanism hypothesis: the observed regressions may come from the loop mechanism itself rather than the training recipe; a one-loop candidate tests whether the same data and objective can match the frozen baseline without recurrence.
- Code components likely affected: config only.
- Why it may address the failure pattern: it provides a standalone control that separates loop recurrence risk from training/data risk.
- Expected risk: low. It is unlikely to beat all dev tasks, but it has high diagnostic value.
- Estimated GPU cost: about 5 GPU hours for one dev train/eval candidate.
- Smallest falsifying dev test: train `tmax=1` and evaluate loop 1 globally on the dev tasks.
- Requires: training and evaluation.

### Direction 5: Document-Side Loop Symmetry

- Core mechanism hypothesis: query-only looping may create query/document distribution mismatch; looping documents under the same standalone scorer might reduce mismatch.
- Code components likely affected: `src/model.py`, `src/eval_mteb.py`, training path, and checkpoint config.
- Why it may address the failure pattern: it directly targets asymmetric query refinement.
- Expected risk: high. It changes more code, increases compute, and could change the retrieval pipeline substantially.
- Estimated GPU cost: at least 14 GPU hours for the smallest useful dev test.
- Smallest falsifying dev test: implement document loop scoring with a fixed loop depth and evaluate one dev candidate.
- Requires: code changes, training, and evaluation.

### Direction 6: Hard-Negative Curriculum or Sampling Change

- Core mechanism hypothesis: loop training may be overfitting to RLHN hard negatives in a way that hurts FiQA2018 and SCIDOCS; a different negative mix may improve global retrieval.
- Code components likely affected: `src/data.py`, configs, and training path.
- Why it may address the failure pattern: it changes the supervision distribution instead of local loop mechanics.
- Expected risk: medium to high. Dataset changes can be noisy and must remain auditable.
- Estimated GPU cost: about 8 to 12 GPU hours for one dev train/eval candidate.
- Smallest falsifying dev test: one predeclared sampling variant on the four dev tasks.
- Requires: code/config changes, training, and evaluation.

### Direction 7: Conservative Evaluation-Time Loop Normalization

- Core mechanism hypothesis: loop outputs may have task-dependent scale or anisotropy even after L2 normalization; a parameter-free normalization computed inside the candidate pipeline might reduce drift.
- Code components likely affected: `src/eval_mteb.py` and possibly `src/model.py`.
- Why it may address the failure pattern: it targets representation calibration without frozen-baseline input.
- Expected risk: medium. It is easy to accidentally change metric semantics or introduce dataset-specific adaptation, so it needs a stricter audit before use.
- Estimated GPU cost: about 4 GPU hours for one eval-only dev test after implementation.
- Smallest falsifying dev test: one global normalization rule applied identically to all dev tasks.
- Requires: evaluation-only code changes and evaluation.

## Ranking

1. Final-loop first-token objective: strongest combination of standalone validity, novelty, limited implementation risk, and direct connection to observed first-token gains.
2. Detached first-token memory: strong standalone validity and low code risk; directly targets recurrence instability.
3. Short-horizon first-token training: useful mechanism change at lower cost; not just an evaluation-depth sweep because it changes training horizon.
4. Single-loop standalone control: high falsification clarity and low risk; lower chance of winning but valuable for diagnosing whether recurrence is the source of regressions.
5. Hard-negative curriculum or sampling change: potentially relevant, but needs a careful data audit before submission.
6. Conservative evaluation-time loop normalization: potentially useful, but higher metric-semantics risk and should not be rushed into a batch.
7. Document-side loop symmetry: interesting but too broad and expensive for this resume cycle.

## Selected Portfolio: batch_008_dev

The next batch should include four standalone_main candidates:

- `r008_final_first_token_t10`: tests Direction 1. Included because it changes the objective while keeping the strongest memory mode. It falsifies the idea that loopwise intermediate supervision is the main cause of FiQA2018/SCIDOCS regressions. Estimated cost: 7 GPU hours. Expected information gain: tells whether first-token memory needs final-only supervision.
- `r008_detached_first_token_t10`: tests Direction 2. Included because it changes recurrence gradient flow without changing scoring semantics. It falsifies the idea that memory-gradient accumulation drives the mixed dev pattern. Estimated cost: 7 GPU hours. Expected information gain: separates useful memory content from unstable recurrent credit assignment.
- `r008_short_first_token_t4`: tests Direction 3. Included because it changes training horizon, not just evaluation depth. It falsifies the idea that long-horizon training is the source of over-iteration drift. Estimated cost: 5 GPU hours. Expected information gain: tells whether the first-token mechanism is viable at lower loop budget.
- `r008_single_loop_t1`: tests Direction 4. Included as a control. It falsifies the idea that the current training recipe alone can recover a global standalone dev signal without recurrence. Estimated cost: 5 GPU hours. Expected information gain: distinguishes loop-specific failure from general training failure.

Portfolio size: 4 candidates, 24 estimated GPU hours against a 24 GPU-hour batch budget, with at most 4 concurrent GPU jobs. This uses the available batch capacity without adding near-duplicate local probes.

This is not an uninformative local sweep because the candidates differ by objective, memory-gradient behavior, training horizon, and recurrence control. They do not reuse the same failed checkpoint at neighboring loop depths, and none uses frozen-baseline scoring input.

If all four candidates fail, the next research state is still informative: first-token local-depth search, final-only supervision, memory detachment, shorter-horizon training, and single-loop training will all be falsified on the same dev task set under comparable conditions.

Expected postprocess behavior: one Slurm-native postprocess dependency job should run after all eval jobs, collect all dev-task rows, write `collected_results.csv`, write `scoreboard.csv` and `scoreboard.json`, and update goal state. The batch remains dev-only and cannot trigger `main_goal_success`.
