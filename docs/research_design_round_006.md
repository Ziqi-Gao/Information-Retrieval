# Research Design Round 006

Date: 2026-06-21

This plan was written after `batch_012_dev` completed Slurm-native postprocess and before validating, dry-running, preflighting, or submitting the next batch. It uses only dev-task evidence and prior protocol history. It does not use final-task deltas to tune dev candidates.

## Trigger

`batch_012_dev` completed with `postprocess_done.json`, `collected_results.csv`, and `scoreboard.json`. It was dev-only and cannot trigger `main_goal_success`.

Both candidates were `standalone_main`, but neither produced a viable global dev signal:

- `r012_seeded_sampling_first_token_t10__loop10`: SciFact `+0.00572`, NFCorpus `-0.00746`, FiQA2018 `-0.01053`, SCIDOCS `+0.00439`, mean delta `-0.00197`, min delta `-0.01053`.
- `r012_first_pool_first_token_t10__loop10`: SciFact `-0.02756`, NFCorpus `-0.00595`, FiQA2018 `-0.00474`, SCIDOCS `-0.00004`, mean delta `-0.0095725`, min delta `-0.02756`.

Local search remains exhausted. Batch 012 added evidence against seeded random passage sampling as a global fix and against first-token retrieval pooling as a global fix.

## What Has Been Tried

- Early fusion experiments produced positive diagnostic evidence only when using frozen-standard checkpoint/embedding/score inputs. Those remain `fusion_diagnostic` and cannot trigger `main_goal_success`.
- Standalone dev batches have tested mean-pool loop variants, recurrent/no-memory variants, first-token memory, token-concat memory, final-loop versus loopwise losses, loop index choices, detached memory, shorter horizon, lower negative count, document-loop evaluation, tail-weighted loop loss, adjacent-loop consistency, candidate-internal self-residual scoring, in-batch hybrid loss, pairwise ranking loss, seeded random passage sampling, and first-token retrieval pooling.
- No recent `standalone_main` dev batch produced a viable global dev signal.

## Track Distinction

- `fusion_diagnostic` candidates can guide mechanism discovery but cannot be promoted to `main_goal_success`.
- `diagnostic` candidates can be reported but cannot trigger the main objective.
- `standalone_main` candidates must score with candidate-only outputs and no frozen-standard checkpoint, embedding, score, weighted standard+candidate concatenation, or interpolation.
- The next batch is dev-only, so even a strong result would require separate user-approved final validation.

## Failure Pattern

- FiQA2018 is the most persistent standalone regression.
- SCIDOCS was historically the second blocker, although `batch_012_dev` seeded sampling made SCIDOCS positive while NFCorpus became negative.
- Recent objective/data/pooling tests show that NFCorpus can also regress when the training distribution is disturbed.
- Continuing to sweep nearby loop depths, memory modes, loss weights, or the same failed checkpoint would be low-value.

## Research Directions Considered

### Direction 1: Query/Document Loop Co-Training

- Core mechanism hypothesis: training query loops against one-pass documents creates representation mismatch; training looped document states may reduce transfer regressions.
- Code components likely affected: `src/model.py`, `src/train.py`, `src/eval_mteb.py`, config, manifest.
- Why it may address the failure pattern: it directly tests whether query/doc symmetry is missing from standalone training.
- Expected risk: high for this resume cycle. Backpropagating through document loops at 512-token length and multiple hard negatives is likely to be slow or memory-heavy under the current Slurm train walltime.
- Estimated GPU cost: 14 to 18 GPU hours if reduced carefully; higher if full-epoch.
- Smallest dev-only falsification: one fixed q/doc loop co-training candidate with matching document-loop evaluation.
- Requires: training and evaluation changes.

### Direction 2: Two-Stage Standard-To-Loop Training

- Core mechanism hypothesis: directly optimizing loopwise first-token training from the base checkpoint may disrupt general retrieval geometry; a standard warmup before loopwise training may preserve broader transfer.
- Code components likely affected: `src/train.py`, `src/experiments.py`, config, manifest.
- Why it may address the failure pattern: tests optimization path rather than another loop index, memory mode, or loss-weight neighborhood.
- Expected risk: medium. It may simply reproduce the weak compromise between standard and first-token loop behavior.
- Estimated GPU cost: about 10 GPU hours.
- Smallest dev-only falsification: one full-epoch first-token candidate with a predeclared standard-warmup fraction followed by loopwise training, fixed loop 10 dev evaluation.
- Requires: training and evaluation.

### Direction 3: Medium-Hard Negative Window

- Core mechanism hypothesis: first-k RLHN negatives may be too hard or too ordering-biased for cross-domain transfer; using a deterministic middle negative window may reduce overfitting while remaining more structured than random sampling.
- Code components likely affected: `src/data.py`, `src/train.py`, `src/experiments.py`, config, manifest.
- Why it may address the failure pattern: it tests negative hardness pressure separately from `batch_012_dev` random sampling.
- Expected risk: medium. If RLHN ordering does not correlate with hardness, the window may only inject a different bias.
- Estimated GPU cost: about 10 GPU hours.
- Smallest dev-only falsification: one full-epoch first-token candidate using first positive plus middle-window negatives, fixed loop 10 dev evaluation.
- Requires: training and evaluation.

### Direction 4: Sparse Or Late Loop Supervision

- Core mechanism hypothesis: supervising every loop over-constrains representation trajectories; sparse or late-only loop supervision may reduce loop drift.
- Code components likely affected: `src/losses.py`, `src/train.py`, `src/experiments.py`, config, manifest.
- Why it may address the failure pattern: changes supervision topology while preserving standalone scoring.
- Expected risk: medium. It is closer to prior loop-objective local search than the top data/optimization directions.
- Estimated GPU cost: 8 to 10 GPU hours.
- Smallest dev-only falsification: one deterministic sparse loop schedule, fixed loop 10 dev evaluation.
- Requires: training and evaluation.

### Direction 5: Label-Smoothed Listwise Loss

- Core mechanism hypothesis: hard CE and pairwise softplus both failed; label smoothing over positive plus negatives may reduce overconfidence without discarding list structure.
- Code components likely affected: `src/losses.py`, `src/train.py`, `src/experiments.py`, config, manifest.
- Why it may address the failure pattern: tests loss calibration rather than a local loop setting.
- Expected risk: medium. It remains another objective variant after several objective failures.
- Estimated GPU cost: about 10 GPU hours.
- Smallest dev-only falsification: one label-smoothed first-token candidate, fixed loop 10 dev evaluation.
- Requires: training and evaluation.

### Direction 6: Candidate-Internal Multi-Loop Score Smoothing

- Core mechanism hypothesis: different candidate loop depths contain complementary signal; fixed candidate-only score averaging across predeclared loops may reduce instability.
- Code components likely affected: `src/eval_mteb.py`, manifest.
- Why it may address the failure pattern: it uses only candidate embeddings and no frozen-standard inputs.
- Expected risk: medium to high for scientific value. It is evaluation-only and close to local loop-depth postprocessing.
- Estimated GPU cost: 4 to 6 GPU hours.
- Smallest dev-only falsification: one fixed average over predeclared candidate loop scores across all dev tasks.
- Requires: evaluation-only changes.

### Direction 7: Length/Truncation-Aware Document Encoding

- Core mechanism hypothesis: dataset-specific document length distributions may interact poorly with fixed 512-token truncation; a deterministic document text policy may improve transfer.
- Code components likely affected: `src/eval_mteb.py`, data/text normalization helpers, config, manifest.
- Why it may address the failure pattern: FiQA2018 and SCIDOCS may be sensitive to passage construction and truncation.
- Expected risk: high. It can be hard to interpret and may edge toward dataset-specific preprocessing.
- Estimated GPU cost: 8 to 12 GPU hours.
- Smallest dev-only falsification: one fixed, task-agnostic truncation/text-construction policy on all dev tasks.
- Requires: evaluation and possibly training changes.

## Ranking

1. Two-stage standard-to-loop training: high standalone validity, good novelty after batch 012, moderate implementation risk, clear falsification, about 10 GPU hours.
2. Medium-hard negative window: high standalone validity, distinct from random sampling, low-to-moderate implementation risk, clear falsification, about 10 GPU hours.
3. Query/document loop co-training: high novelty and plausible mechanism, but too risky for the current one-batch resume cycle because of document-loop training cost and memory pressure.
4. Label-smoothed listwise loss: valid but less novel after in-batch and pairwise objective failures.
5. Sparse or late loop supervision: valid but closer to prior loop-objective local search.
6. Candidate-internal multi-loop score smoothing: cheap but close to local evaluation postprocessing.
7. Length/truncation-aware document encoding: potentially useful but high interpretation risk.

## Selected Portfolio: batch_013_dev

The next batch should include two `standalone_main` dev candidates:

- `r013_two_stage_warmup_first_token_t10`
  - Tests Direction 2.
  - Included because it attacks optimization-path instability rather than loop depth or memory mode.
  - Failure mode falsified: direct loopwise optimization is not the main cause of FiQA2018/SCIDOCS/NFCorpus regressions.
  - Estimated cost: 10 GPU hours.
  - Expected information gain: high.
  - Candidate track: `standalone_main`.

- `r013_middle_negatives_first_token_t10`
  - Tests Direction 3.
  - Included because it attacks hard-negative pressure with a deterministic structured data rule, distinct from `batch_012_dev` random sampling.
  - Failure mode falsified: first-k negative hardness/order is not the main cause of recurring transfer regressions.
  - Estimated cost: 10 GPU hours.
  - Expected information gain: medium to high.
  - Candidate track: `standalone_main`.

Portfolio size: 2 candidates, 20 estimated GPU hours against the configured 24 GPU-hour batch limit, with at most 2 concurrent GPU jobs.

This is not an uninformative local sweep. It does not change only `loop_idx`, evaluate neighboring depths of a failed checkpoint, or adjust memory mode. The two candidates test optimization schedule and deterministic negative hardness pressure.

Query/document loop co-training is deferred despite high novelty because the current training wrapper and GPU budget make document-loop backpropagation a timeout/OOM risk in this single resume cycle.
