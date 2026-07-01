# Research Design Round 002

Date: 2026-06-21

This plan was written after `batch_008_dev` completed and before creating any new batch. It uses only dev-task evidence and prior protocol history. It does not use final-task deltas to tune dev candidates.

## Trigger

`batch_008_dev` completed with four valid dev-task rows per candidate, but every candidate regressed on every dev task. The batch remains dev-only and cannot trigger `main_goal_success`.

Local search remains exhausted. `batch_004_dev` through `batch_008_dev` did not produce a viable global standalone signal, and repeated local loop-depth or memory-mode probing is now low-value.

## What Has Been Tried

- Fusion diagnostic batches found some loop signal when combined with frozen-baseline scoring input. These results remain diagnostic only and cannot support `standalone_main`.
- Existing standalone mean-pool, recurrent, final-loop, and no-memory checkpoints did not produce viable dev results.
- First-token loopwise training in `batch_006_dev` improved SciFact and NFCorpus but regressed on FiQA2018 and SCIDOCS.
- Nearby first-token loop depths in `batch_007_dev` preserved the same mixed pattern.
- `batch_008_dev` tested a portfolio of first-token final-loop supervision, detached memory, short-horizon training, and a one-loop control. All four candidates regressed on all four dev tasks.
- Training logs show `batch_008_dev` candidates stopped at `max_steps=1000`, about `0.32` epoch, while `batch_006_dev` first-token training ran close to a full epoch. This means the `batch_008_dev` mechanisms were tested under a capped training budget and the broad negative result may include an undertraining component.

## Track Distinction

- `fusion_diagnostic` results are useful for evidence that a non-standard loop signal exists, but any frozen-baseline scoring input excludes them from `standalone_main`.
- `standalone_main` candidates must score with their own candidate pipeline only.
- The next batch remains dev-only. It can guide research but cannot trigger `main_goal_success`.

## Failure Pattern

The recurrent standalone failure modes are now clearer:

- First-token loopwise can sometimes improve SciFact and NFCorpus after full training.
- FiQA2018 and SCIDOCS consistently regress in first-token local-depth sweeps.
- Recurrent hidden-state and token-concat variants are unstable.
- Undertrained first-token variants regress across all dev tasks.
- The current query-only loop may create a query/document representation mismatch because documents are encoded once while queries loop.

Further local loop-depth or memory-mode sweeps are low-value because they would keep probing the same query-only first-token neighborhood. The next dev batch should test broader causes: query/document asymmetry, adequate training budget, and negative-sampling pressure.

## Research Directions Considered

### Direction 1: Document-Side Loop Symmetry

- Core mechanism hypothesis: query-only loops shift query representations away from document embeddings; looping documents with the same candidate checkpoint may reduce that asymmetry.
- Code components likely affected: `src/model.py`, `src/eval_mteb.py`, `scripts/slurm_eval.sbatch`, `scripts/goal_submit_batch.py`, and manifest validation.
- Why it may address the failure pattern: it targets the query/document mismatch without frozen-baseline input or new trainable parameters.
- Expected risk: medium. Evaluation cost rises because corpora are looped, and training still used non-looped documents.
- Estimated GPU cost: about 4 GPU hours for one eval-only dev candidate.
- Smallest dev-only falsification: evaluate the `batch_006_dev` first-token checkpoint with fixed query and document loop indices on the four dev tasks.
- Requires: evaluation-only code changes and evaluation.

### Direction 2: Full-Epoch Detached First-Token Training

- Core mechanism hypothesis: detached first-token memory may still be useful, but `batch_008_dev` undertrained it by stopping at about one-third epoch.
- Code components likely affected: config and manifest only.
- Why it may address the failure pattern: it retests memory-gradient stabilization under a comparable training budget to the only first-token run that showed partial positive dev signal.
- Expected risk: medium. It may remain below the frozen baseline even with full training.
- Estimated GPU cost: about 10 GPU hours.
- Smallest dev-only falsification: train detached first-token loopwise for one full epoch and evaluate one fixed global loop depth on dev tasks.
- Requires: training and evaluation.

### Direction 3: Lower Hard-Negative Pressure

- Core mechanism hypothesis: the seven-hard-negative RLHN setup may overfit loop representations to hard-negative discrimination and hurt transfer to FiQA2018 and SCIDOCS.
- Code components likely affected: config and manifest only.
- Why it may address the failure pattern: it changes the supervision pressure rather than the loop depth, memory mode, or frozen-baseline scoring.
- Expected risk: medium. Fewer negatives may weaken retrieval training generally.
- Estimated GPU cost: about 10 GPU hours.
- Smallest dev-only falsification: train first-token loopwise with fewer negatives for one full epoch and evaluate one fixed global loop depth on dev tasks.
- Requires: training and evaluation.

### Direction 4: Loop-Depth Dropout During Training

- Core mechanism hypothesis: forcing all loops to optimize every batch may make the model brittle; randomly supervising loop depths could regularize the iterative representation.
- Code components likely affected: `src/train.py` and `src/losses.py`.
- Why it may address the failure pattern: it changes the objective's pressure across loop depths without using final-task feedback.
- Expected risk: medium to high. It changes the loss path and needs careful implementation.
- Estimated GPU cost: about 10 GPU hours.
- Smallest dev-only falsification: one first-token loopwise candidate with a global loop-depth dropout schedule.
- Requires: code changes, training, and evaluation.

### Direction 5: Query-Side Residual Mixing Without Frozen Baseline

- Core mechanism hypothesis: loop output may drift too far from the initial query encoding; mixing the candidate's own first-loop embedding with the later loop could stabilize retrieval.
- Code components likely affected: `src/eval_mteb.py` and manifest validation.
- Why it may address the failure pattern: it uses only candidate checkpoint embeddings and may reduce FiQA2018/SCIDOCS drift without frozen-baseline input.
- Expected risk: high. It is close to score/embedding fusion and must be audited carefully to avoid claim-track confusion.
- Estimated GPU cost: about 4 GPU hours.
- Smallest dev-only falsification: one fixed self-residual rule applied globally to all dev tasks.
- Requires: evaluation-only code changes and evaluation.

### Direction 6: Conservative Corpus Normalization

- Core mechanism hypothesis: loop embeddings may have corpus-dependent anisotropy; a predeclared parameter-free normalization could improve ranking.
- Code components likely affected: `src/eval_mteb.py`.
- Why it may address the failure pattern: it targets representation calibration rather than training dynamics.
- Expected risk: high. It may be considered transductive evaluation adaptation and should not be rushed into `standalone_main`.
- Estimated GPU cost: about 4 GPU hours.
- Smallest dev-only falsification: one global normalization rule on dev tasks, likely as `diagnostic`.
- Requires: evaluation-only code changes and evaluation.

### Direction 7: Data Curriculum

- Core mechanism hypothesis: starting with easier negatives or shorter passages could stabilize loop training before hard-negative pressure.
- Code components likely affected: `src/data.py`, configs, and training orchestration.
- Why it may address the failure pattern: it changes training dynamics more fundamentally than local loop parameters.
- Expected risk: high. It is broader, less auditable in one resume cycle, and may need new data bookkeeping.
- Estimated GPU cost: 12 to 20 GPU hours.
- Smallest dev-only falsification: one predeclared curriculum candidate on dev tasks.
- Requires: code changes, training, and evaluation.

## Ranking

1. Document-side loop symmetry: high novelty, direct asymmetry hypothesis, eval-only, clear falsification.
2. Lower hard-negative pressure: standalone-valid training change, targets transfer failures, low code risk.
3. Full-epoch detached first-token training: useful correction for an undertrained mechanism, but less novel than Direction 1 or 3.
4. Loop-depth dropout: plausible objective change but more implementation risk.
5. Data curriculum: plausible but broad and harder to audit in one cycle.
6. Query-side residual mixing without frozen baseline: potentially useful but claim-track risk is high.
7. Conservative corpus normalization: useful diagnostic idea but too much transductive-evaluation risk for the next standalone_main batch.

## Selected Portfolio: batch_009_dev

The next batch should include three standalone_main dev candidates:

- `r009_docloop_first_token_t7`: tests Direction 1. It is included because it directly probes query/document asymmetry with an existing first-token candidate checkpoint. It falsifies whether document-side looping can recover the dev tasks that query-only looping regressed. Estimated cost: 4 GPU hours. Expected information gain: high, because it isolates scoring-side asymmetry without new training.
- `r009_detached_first_token_full_t10`: tests Direction 2. It is included because `batch_008_dev` detached memory was undertrained and was the least negative candidate in that batch. It falsifies whether detached memory remains poor after an adequate training budget. Estimated cost: 10 GPU hours. Expected information gain: medium to high.
- `r009_first_token_neg3_t10`: tests Direction 3. It is included because it changes hard-negative supervision pressure while keeping first-token loop mechanics fixed. It falsifies whether seven hard negatives are contributing to poor transfer. Estimated cost: 10 GPU hours. Expected information gain: high if all tasks still regress.

Portfolio size: 3 candidates, 24 estimated GPU hours against a 24 GPU-hour batch budget, with at most 3 concurrent GPU jobs. This uses the available budget without adding low-value near-duplicates.

This is not an uninformative local sweep: the candidates test document-side scoring symmetry, training-budget adequacy for detached memory, and negative-sampling pressure. They do not retest neighboring loop depths of the same checkpoint and do not use frozen-baseline scoring input.

Expected postprocess behavior: one Slurm-native postprocess dependency job runs after all eval jobs, collects all dev-task rows, writes `collected_results.csv`, writes `scoreboard.csv` and `scoreboard.json`, and updates goal state. The batch remains dev-only and cannot trigger `main_goal_success`.
