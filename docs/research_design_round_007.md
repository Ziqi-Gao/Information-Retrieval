# Research Design Round 007

Date: 2026-06-26

This plan was written after `batch_013_dev` completed Slurm-native postprocess and before validating, dry-running, preflighting, or submitting the next batch. It uses only dev-task evidence and prior protocol history. It does not use final-task deltas to tune dev candidates.

## Trigger

`batch_013_dev` completed with `postprocess_done.json`, `collected_results.csv`, and `scoreboard.json`. It was dev-only and cannot trigger `main_goal_success`.

Both candidates were `standalone_main`, but neither produced a viable global dev signal:

- `r013_two_stage_warmup_first_token_t10__loop10`: SciFact `+0.00248`, NFCorpus `+0.00030`, SCIDOCS `-0.00224`, FiQA2018 `-0.00603`, mean delta `-0.00137`, min delta `-0.00603`.
- `r013_middle_negatives_first_token_t10__loop10`: SciFact `+0.00619`, NFCorpus `+0.00164`, SCIDOCS `-0.00188`, FiQA2018 `-0.01183`, mean delta `-0.00147`, min delta `-0.01183`.

Local search remains exhausted. Batch 013 added evidence that standard-to-loop warmup and a deterministic middle negative window do not remove the recurring FiQA2018 and SCIDOCS regressions.

## What Has Been Tried

- Early retrieval-time standard+loop fusion produced positive diagnostic evidence only when using frozen-standard checkpoint/embedding inputs. Those results remain `fusion_diagnostic` and cannot trigger `main_goal_success`.
- Standalone dev batches have tested fixed loop depths, mean-pool loops, recurrent/no-memory variants, first-token memory, token-concat memory, final-loop and loopwise losses, detached memory, shorter horizons, lower negative count, document-loop evaluation, tail-weighted loop loss, adjacent-loop consistency, candidate-internal self-residual scoring, in-batch hybrid loss, pairwise ranking loss, seeded random passage sampling, first-token retrieval pooling, standard-to-loop warmup, and deterministic middle-window negatives.
- No recent `standalone_main` dev batch produced a viable global dev signal.

## Track Distinction

- `fusion_diagnostic` candidates can guide mechanism discovery but cannot be promoted to `main_goal_success`.
- `diagnostic` candidates can be reported but cannot trigger the main goal.
- `standalone_main` candidates must score with candidate-only outputs and no frozen-standard checkpoint, embedding, score, weighted standard+candidate concatenation, or interpolation.
- The next batch is dev-only, so even a strong result would require separate user-approved final validation.

## Failure Pattern

- FiQA2018 is the most persistent standalone regression.
- SCIDOCS is the second recurring blocker.
- First-token loop candidates can improve SciFact and sometimes NFCorpus, but the same candidates usually regress FiQA2018 and SCIDOCS.
- Recent data-selection changes can also turn NFCorpus negative, so more negative-window or seeded-sampling local variants are low-value.
- Another loop-depth, memory-mode, checkpoint-depth, or tiny objective-weight sweep would likely repeat the same split-positive/split-negative pattern.

## Research Directions Considered

### Direction 1: Query/Document Loop Co-Training

- Core mechanism hypothesis: training query loops against one-pass documents creates representation mismatch; training looped query and document states together may reduce transfer regressions.
- Code components likely affected: `src/model.py`, `src/train.py`, `src/losses.py`, `src/eval_mteb.py`, configs, manifest.
- Why it may address the failure pattern: it directly tests whether query/document training asymmetry is causing FiQA2018 and SCIDOCS regressions.
- Expected risk: high. Backpropagating document loops over long documents and hard negatives is likely to be slow or memory-heavy under the current Slurm budget.
- Estimated GPU cost: 14 to 20 GPU hours for one reduced candidate, with timeout/OOM risk.
- Smallest dev-only falsification: one fixed q/doc loop co-training candidate with matching candidate-only dev evaluation.
- Requires: training and evaluation changes.

### Direction 2: Label-Smoothed Listwise Hard-Negative Loss

- Core mechanism hypothesis: hard positive-vs-negative CE is overconfident on RLHN negatives; conservative label smoothing may reduce cross-domain overfitting while retaining listwise structure.
- Code components likely affected: `src/losses.py`, `src/train.py`, `src/experiments.py`, config, manifest.
- Why it may address the failure pattern: it tests loss calibration after hard CE, pairwise softplus, and in-batch hybrid objectives failed.
- Expected risk: medium. It may underfit or simply reproduce prior objective failures.
- Estimated GPU cost: about 10 GPU hours.
- Smallest dev-only falsification: one full-epoch first-token candidate with fixed label smoothing and fixed loop 10 dev evaluation.
- Requires: training and evaluation.

### Direction 3: Sparse Or Late Loop Supervision

- Core mechanism hypothesis: supervising every loop over-constrains the representation trajectory; supervising only a predeclared late sparse set may reduce loop drift.
- Code components likely affected: `src/losses.py`, `src/train.py`, `src/experiments.py`, config, manifest.
- Why it may address the failure pattern: it changes supervision topology rather than loop depth, memory mode, or checkpoint selection.
- Expected risk: medium. It is closer to previous objective work than q/doc co-training, but it has not been tested directly.
- Estimated GPU cost: about 10 GPU hours.
- Smallest dev-only falsification: one first-token candidate supervised only at loops `{4, 7, 10}`, fixed loop 10 dev evaluation.
- Requires: training and evaluation.

### Direction 4: Deterministic Easy-To-Hard Negative Curriculum

- Core mechanism hypothesis: static first-k, middle-window, or random negatives are all too blunt; staged negative hardness may reduce transfer regressions.
- Code components likely affected: `src/data.py`, `src/train.py`, config, manifest.
- Why it may address the failure pattern: it attacks training pressure over time rather than a single fixed negative window.
- Expected risk: medium to high. It assumes RLHN order tracks hardness well enough to define a curriculum.
- Estimated GPU cost: 12 to 16 GPU hours.
- Smallest dev-only falsification: one predeclared curriculum schedule across one full epoch and fixed loop 10 dev evaluation.
- Requires: training and evaluation.

### Direction 5: Length/Truncation-Aware Document Encoding

- Core mechanism hypothesis: a task-agnostic document text policy may reduce failures caused by fixed 512-token truncation and title/text construction.
- Code components likely affected: `src/eval_mteb.py`, text helpers, config, manifest.
- Why it may address the failure pattern: FiQA2018 and SCIDOCS may be sensitive to document construction and truncation.
- Expected risk: high. Interpretation is difficult, and the change can look dataset-preprocessing-specific even if applied globally.
- Estimated GPU cost: 6 to 10 GPU hours.
- Smallest dev-only falsification: one global candidate-only text policy applied to all dev tasks.
- Requires: evaluation changes, possibly training changes.

### Direction 6: Candidate-Only Multi-Loop Score Aggregation

- Core mechanism hypothesis: predeclared candidate-only loop score averaging may reduce instability across loop depths without frozen-standard inputs.
- Code components likely affected: `src/eval_mteb.py`, manifest validation/collection if explicit metadata is added.
- Why it may address the failure pattern: it can test whether a single loop depth is too brittle.
- Expected risk: medium to high for scientific value. It is cheap but close to local loop-depth postprocessing.
- Estimated GPU cost: 4 to 6 GPU hours.
- Smallest dev-only falsification: one fixed global score rule over predeclared candidate loops.
- Requires: evaluation changes.

### Direction 7: Non-Fusion Candidate-Internal Calibration

- Core mechanism hypothesis: a candidate-only calibration rule may stabilize scores without using frozen-standard scores or embeddings.
- Code components likely affected: evaluation and possibly calibration utilities.
- Why it may address the failure pattern: it targets score scale/geometry instead of training objective.
- Expected risk: high. It can easily become diagnostic or transductive if not carefully constrained.
- Estimated GPU cost: 4 to 8 GPU hours.
- Smallest dev-only falsification: one fixed, unsupervised, candidate-only normalization rule on all dev tasks.
- Requires: evaluation changes.

## Ranking

1. Label-smoothed listwise hard-negative loss: high standalone validity, moderate novelty after pairwise/in-batch failures, low implementation risk, clear falsification, about 10 GPU hours.
2. Sparse or late loop supervision: high standalone validity, distinct from loop-depth and memory sweeps, low implementation risk, clear falsification, about 10 GPU hours.
3. Query/document loop co-training: strongest mechanistic novelty, but too risky for this single resume cycle because of document-loop backpropagation cost.
4. Deterministic easy-to-hard negative curriculum: valid and relevant, but follows several already failed negative-selection probes.
5. Length/truncation-aware document encoding: potentially useful but hard to interpret.
6. Candidate-only multi-loop score aggregation: cheap but close to local evaluation postprocessing.
7. Non-fusion candidate-internal calibration: valid only under careful constraints and not the best next use of compute.

## Selected Portfolio: batch_014_dev

The next batch should include two `standalone_main` dev candidates:

- `r014_sparse_late_first_token_t10`
  - Tests Direction 3.
  - Included because it attacks over-constrained loop trajectory supervision rather than loop depth, memory mode, or checkpoint selection.
  - Failure mode falsified: supervising every loop is not the main cause of the recurring FiQA2018/SCIDOCS regressions.
  - Estimated cost: 10 GPU hours.
  - Expected information gain: medium to high.
  - Candidate track: `standalone_main`.

- `r014_label_smooth_first_token_t10`
  - Tests Direction 2.
  - Included because it attacks hard-negative CE overconfidence while preserving candidate-only scoring and listwise structure.
  - Failure mode falsified: hard target overconfidence is not the main cause of transfer regressions.
  - Estimated cost: 10 GPU hours.
  - Expected information gain: high.
  - Candidate track: `standalone_main`.

Portfolio size: 2 candidates, 20 estimated GPU hours against the configured 24 GPU-hour batch limit, with at most 2 concurrent GPU jobs.

This is not an uninformative local sweep. It does not change only `loop_idx`, evaluate neighboring depths of a failed checkpoint, adjust memory mode, or retest a nearby negative window. The two candidates test distinct training mechanisms: supervision topology and target calibration.

Query/document loop co-training is deferred despite high novelty because the current one-batch cycle should avoid a high probability of Slurm timeout or GPU OOM before lower-risk standalone objective mechanisms are falsified.
