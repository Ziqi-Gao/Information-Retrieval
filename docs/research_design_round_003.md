# Research Design Round 003

Date: 2026-06-21

This plan was written after `batch_009_dev_repair` completed and before submitting any new batch. It uses only dev-task evidence and prior protocol history. It does not use final-task deltas to tune dev candidates.

## Trigger

`batch_009_dev_repair` completed the missing document-loop repair from `batch_009_dev`. The repaired candidate is dev-only `standalone_main`, covers only `SciFact`, `NFCorpus`, `FiQA2018`, and `SCIDOCS`, and cannot trigger `main_goal_success`.

Local search remains exhausted. `batch_004_dev` through `batch_009_dev_repair` have not produced a viable standalone global dev signal. Another nearby loop-depth, memory-mode, detached-memory, or small negative-count sweep would be low-value.

## What Has Been Tried

- Fusion diagnostic batches (`batch_001_dev`, `batch_002_dev`, `batch_003_final_repair`) showed useful loop signal only when frozen-standard scoring input was included. These remain diagnostic and cannot support `standalone_main`.
- Existing standalone mean-pool, recurrent mean-pool, final-loop, recurrent no-memory, and no-history style checkpoints failed to produce a viable dev signal.
- First-token loopwise training in `batch_006_dev` and its local loop-depth sweep in `batch_007_dev` consistently improved `SciFact` and `NFCorpus`, while regressing `FiQA2018` and usually missing the margin on `SCIDOCS`.
- `batch_008_dev` tested first-token final-loop supervision, detached memory, shorter training horizon, and a one-loop control under capped steps; all four candidates regressed on all four dev tasks.
- `batch_009_dev` tested full-epoch detached first-token training and lower hard-negative pressure; both failed globally.
- `batch_009_dev_repair` tested document-side loop symmetry for the `batch_006_dev` first-token checkpoint. It exactly recovered the same dev pattern as the original query-only loop-7 candidate: `SciFact +0.00419`, `NFCorpus +0.00359`, `SCIDOCS -0.00102`, `FiQA2018 -0.01126`.

## Track Distinction

- `fusion_diagnostic` results are excluded from main success even if they pass numerically.
- `standalone_main` candidates must score with candidate pipeline inputs only. They may use candidate-internal loop states, but not frozen-standard checkpoints, embeddings, scores, concatenation, or score interpolation.
- The next batch is dev-only. Its results may guide research but cannot trigger `main_goal_success`.

## Failure Pattern

- `FiQA2018` is the most stable blocker: nearly every recent standalone candidate regresses, often by much more than the diagnostic margin.
- `SCIDOCS` is the second blocker: it occasionally improves slightly, but has not cleared the `+0.002` main-task margin in recent standalone batches.
- `SciFact` and `NFCorpus` are the only dev tasks where the first-token loop family repeatedly shows positive deltas.
- Document-loop symmetry did not change the first-token loop-7 result, so query/document asymmetry alone is not enough under the current checkpoint and evaluation path.
- Full-epoch detached memory and lower negative count did not rescue the first-token family.

Further local sweeps are low-value because the first-token neighborhood has already been probed through loop depth, objective, memory detachment, training horizon, negative count, and document-loop evaluation. The next batch must test broader training objective and drift-control mechanisms.

## Research Directions Considered

### Direction 1: Loop-Loss Tail Weighting

- Core mechanism hypothesis: uniform loopwise supervision over-constrains shallow loops and may force unstable intermediate representations; weighting deeper loops more heavily could preserve late-loop utility without the brittleness of pure final-loop supervision.
- Code components likely affected: `src/losses.py`, `src/train.py`, `src/experiments.py`, config, manifest.
- Why it may address the failure pattern: it changes the training objective, not local loop depth, and targets the difference between uniform loopwise and failed final-only supervision.
- Expected risk: medium. Late-loop weighting may still amplify FiQA2018 drift.
- Estimated GPU cost: about 10 GPU hours.
- Smallest dev-only falsification: one full-epoch first-token tail-weighted candidate, fixed global loop 10, four dev tasks.
- Requires: training and evaluation.

### Direction 2: Candidate-Internal Loop Consistency

- Core mechanism hypothesis: adjacent loop query embeddings drift too far; a small consistency penalty can reduce this drift while preserving candidate-only scoring.
- Code components likely affected: `src/losses.py`, `src/train.py`, `src/experiments.py`, config, manifest.
- Why it may address the failure pattern: it directly targets the recurring first-token pattern of task-specific gains plus FiQA2018/SCIDOCS regressions.
- Expected risk: medium. Too much consistency may suppress useful refinement.
- Estimated GPU cost: about 10 GPU hours.
- Smallest dev-only falsification: one full-epoch first-token consistency candidate, fixed global loop 10, four dev tasks.
- Requires: training and evaluation.

### Direction 3: Candidate Self-Residual Query Stabilization

- Core mechanism hypothesis: the batch_006 checkpoint has useful loop-7 signal but needs anchoring to its own loop-1 query embedding, not to the frozen standard model.
- Code components likely affected: `src/eval_mteb.py`, `scripts/slurm_eval.sbatch`, `scripts/goal_submit_batch.py`, collector/validator metadata.
- Why it may address the failure pattern: it tests candidate-only drift control cheaply before spending another full training budget.
- Expected risk: medium to high. It is close to fusion-style thinking, so it must remain candidate-internal, globally predeclared, and reported as `diagnostic`, not `standalone_main`.
- Estimated GPU cost: about 4 GPU hours.
- Smallest dev-only falsification: one fixed self-residual alpha on the four dev tasks.
- Requires: evaluation-only code changes and evaluation.

### Direction 4: True In-Batch Negatives

- Core mechanism hypothesis: current training uses only per-sample hard negatives, so it may lack global contrastive signal; in-batch positives could provide broader discrimination.
- Code components likely affected: `src/losses.py`, `src/train.py`, config, manifest.
- Why it may address the failure pattern: it changes the retrieval training signal more fundamentally than loop depth or memory mode.
- Expected risk: medium. In-batch false negatives could harm transfer, and current code intentionally forces this path off.
- Estimated GPU cost: 8 to 10 GPU hours.
- Smallest dev-only falsification: one first-token candidate with in-batch loss restored, fixed global loop depth, four dev tasks.
- Requires: code changes, training, and evaluation.

### Direction 5: Loop-Depth Dropout During Training

- Core mechanism hypothesis: supervising every loop every batch makes the representation brittle; randomly or deterministically dropping loop losses could regularize loop behavior.
- Code components likely affected: `src/losses.py`, `src/train.py`, config, manifest.
- Why it may address the failure pattern: it changes which loop states are optimized instead of probing nearby evaluation depths.
- Expected risk: medium. Stochastic training behavior needs careful reproducibility controls.
- Estimated GPU cost: 8 to 10 GPU hours.
- Smallest dev-only falsification: one predeclared dropout schedule and one fixed evaluation loop.
- Requires: code changes, training, and evaluation.

### Direction 6: Random Positive/Negative Sampling

- Core mechanism hypothesis: fixed first-positive and first-k-negative sampling may overfit RLHN ordering artifacts; randomized sampling could improve transfer.
- Code components likely affected: `src/data.py`, config, manifest.
- Why it may address the failure pattern: it changes the data distribution rather than loop mechanics.
- Expected risk: medium. It introduces more data noise and may reduce reproducibility if not seeded carefully.
- Estimated GPU cost: 8 to 10 GPU hours.
- Smallest dev-only falsification: one seeded random-sampling first-token candidate on four dev tasks.
- Requires: code changes, training, and evaluation.

### Direction 7: Pairwise Ranking Loss

- Core mechanism hypothesis: softmax hard-negative CE may be too sharp; pairwise softplus or margin loss may reduce overfitting to hard negatives.
- Code components likely affected: `src/losses.py`, `src/train.py`, `src/experiments.py`, config, manifest.
- Why it may address the failure pattern: it tests a different ranking objective after lower negative count failed.
- Expected risk: medium. It may weaken absolute discrimination.
- Estimated GPU cost: 8 to 10 GPU hours.
- Smallest dev-only falsification: one first-token pairwise candidate, fixed global loop depth, four dev tasks.
- Requires: code changes, training, and evaluation.

## Ranking

1. Candidate-internal loop consistency: high relevance to observed drift, standalone-valid, clear falsification, moderate code risk.
2. Loop-loss tail weighting: objective-level change, standalone-valid, relatively low implementation risk.
3. Candidate self-residual query stabilization: cheap and directly tests candidate-only drift control, but is diagnostic-only because of claim-track audit risk.
4. True in-batch negatives: high information gain but broader training-semantics change.
5. Loop-depth dropout: plausible but stochastic and less auditable than fixed tail weighting in this cycle.
6. Pairwise ranking loss: useful objective change but less directly tied to the first-token drift evidence.
7. Random positive/negative sampling: potentially important, but data-path changes should follow a tighter audit.

## Selected Portfolio: batch_010_dev

The next batch should include two `standalone_main` dev candidates and one diagnostic candidate:

- `r010_tail_weighted_first_token_t10`: tests Direction 1. It is included because it changes loopwise supervision pressure without retesting neighboring depths or using any frozen-baseline scoring input. It falsifies whether uniform loopwise weighting is the source of the first-token tradeoff. Estimated cost: 10 GPU hours. Expected information gain: medium to high.
- `r010_consistency_first_token_t10`: tests Direction 2. It is included because it directly regularizes loop drift, the clearest observed failure mechanism. It falsifies whether a small adjacent-loop consistency term can recover FiQA2018/SCIDOCS while preserving SciFact/NFCorpus signal. Estimated cost: 10 GPU hours. Expected information gain: high.
- `r010_self_residual_first_token_t7_a50`: tests Direction 3 as `diagnostic`. It is included because it cheaply tests candidate-only query stabilization using the best partially positive standalone checkpoint, but it is not allowed to trigger `main_goal_success` or be promoted unchanged to a final standalone claim. It falsifies whether the loop-7 result can be stabilized by anchoring to the candidate's own loop-1 embedding. Estimated cost: 4 GPU hours. Expected information gain: high relative to cost.

Portfolio size: 3 candidates, 24 estimated GPU hours against a 24 GPU-hour batch budget, with at most 3 concurrent GPU jobs.

This is not an uninformative local sweep: the candidates test a loop-loss weighting mechanism, an explicit loop-drift regularizer, and an evaluation-only candidate-internal self-residual diagnostic. They do not change only `loop_idx`, do not evaluate neighboring depths of the same failed checkpoint, do not reuse frozen-standard scoring input, and do not tune on final-task deltas.

If all three candidates fail, the next research state is informative: fixed loop-depth/local memory search, detached-memory training, hard-negative count reduction, document-loop symmetry, loop-loss weighting, explicit loop consistency, and candidate-internal self-residual stabilization will all be falsified on the same dev task set.
