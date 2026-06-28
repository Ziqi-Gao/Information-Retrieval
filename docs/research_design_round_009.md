# Research Design Round 009

Date: 2026-06-27

This plan was written after `batch_015_dev` completed Slurm-native postprocess and before creating, validating, dry-running, preflighting, or submitting `batch_016_dev`. It uses dev-task evidence only.

## Trigger

`batch_015_dev` completed with `postprocess_done.json`, `collected_results.csv`, and `scoreboard.json`. It was dev-only and cannot trigger `main_goal_success`.

All three candidates were `standalone_main`, but none produced a viable global dev signal:

- `r015_role_prompt_standard__loop1`: SciFact `-0.00730`, NFCorpus `+0.00910`, SCIDOCS `+0.00185`, FiQA2018 `-0.00714`, mean delta `-0.0008725`, min delta `-0.00730`.
- `r015_dim_mrl_standard__loop1`: SciFact `-0.00966`, NFCorpus `+0.00040`, SCIDOCS `+0.00132`, FiQA2018 `-0.00183`, mean delta `-0.0024425`, min delta `-0.00966`.
- `r015_role_dim_mrl_standard__loop1`: SciFact `-0.00320`, NFCorpus `+0.00862`, SCIDOCS `+0.00562`, FiQA2018 `-0.00255`, mean delta `+0.0021225`, min delta `-0.00320`.

`minimal_positive_signal=false`, `research_grade_threshold_pass=false`, `fusion_diagnostic_pass=false`, `main_goal_success=false`, and `publishable_score_candidate=false` for all candidates.

## What Has Been Tried

- Early standard+loop retrieval-time fusion produced diagnostic evidence only when using frozen-standard checkpoint or embedding inputs. Those results remain `fusion_diagnostic` and cannot trigger `main_goal_success`.
- Standalone first-token loop-memory batches tested loop-depth tuning, memory-mode variants, detached training, shorter horizons, lower hard-negative count, document-loop evaluation, tail-weighted loop loss, adjacent-loop consistency, self-residual scoring, in-batch hybrid contrastive loss, pairwise loss, seeded random passage sampling, first-token retrieval pooling, standard-to-loop warmup, middle-window negatives, sparse late-loop supervision, and label-smoothed listwise loss.
- `batch_015_dev` pivoted outside first-token loop-memory and tested no-loop role prompting, no-loop dimensional Matryoshka supervision, and their combination.

## Track Distinction

- `fusion_diagnostic` candidates may guide mechanism discovery but cannot trigger the main goal.
- `standalone_main` candidates must use candidate-only scores with no frozen-standard checkpoint, embedding, score, weighted standard+candidate concatenation, or score interpolation.
- `batch_015_dev` and the selected next batch are dev-only; even a strong dev result would require separate user-approved final validation.

## Failure Pattern

- `FiQA2018` remains the most persistent standalone regression.
- `SciFact` became a recurring blocker in `batch_015_dev`; all no-loop role/MRL candidates regressed it.
- `SCIDOCS` and `NFCorpus` are the recurring positives from `batch_015_dev`, especially for the combined role+MRL candidate.
- The failure looks less like pure loop-memory instability and more like a combination of query/document representation mismatch, hard-negative/training-objective mismatch, and task-format sensitivity.
- First-token loop-memory is exhausted for now. Batch 015 also falsifies simple no-loop role prompting and simple dimensional MRL as sufficient fixes.
- Further local sweeps over prefix wording, nearby MRL dimension lists, MRL weights, or first-token loop/loss variants are low-value.

## Research Directions Considered

### Direction 1: Query/Document Loop Co-Training

- Core mechanism hypothesis: training looped queries against one-pass documents creates representation mismatch; looped documents during training and evaluation may reduce FiQA2018/SCIDOCS regressions.
- Code components likely affected: `src/train.py`, `src/experiments.py`, configs, manifest. `src/model.py` and `src/eval_mteb.py` already expose looped document encoding/evaluation paths.
- Why it may address the failure pattern: directly tests q/doc asymmetry instead of another first-token or role/MRL local variant.
- Outside exhausted families: yes.
- Expected risk: high runtime/OOM risk, reduced by `tmax=3`, `num_negatives=3`, and `train_sample_size=12000`.
- Estimated GPU cost: about 7 GPU hours for the reduced dev test.
- Smallest dev-only falsification: one `tmax=3`, `num_negatives=3`, final-loop q/doc candidate evaluated with `loop_docs=true`; fail if FiQA2018/SCIDOCS remain negative or macro mean is non-positive.
- Requires: training and evaluation.

### Direction 2: No-Loop Seeded Passage Sampling

- Core mechanism hypothesis: RLHN's first positive/first negatives may bias transfer; deterministic seeded positive/negative sampling can test data-order sensitivity without loop memory.
- Code components likely affected: `src/experiments.py`, configs, manifest; sampling support already exists in `src/data.py`.
- Why it may address the failure pattern: isolates sampling from first-token loop drift, unlike earlier seeded first-token results.
- Outside exhausted families: outside first-token and role/MRL; overlaps the broader negative-sampling family but changes the base mechanism to no-loop.
- Expected risk: medium; it may reproduce split behavior or underperform the frozen standard.
- Estimated GPU cost: about 7 GPU hours.
- Smallest dev-only falsification: one no-loop seeded-sampling candidate; fail if SciFact/FiQA2018 remain negative and macro mean is not clearly positive.
- Requires: training and evaluation.

### Direction 3: No-Loop Candidate-Only In-Batch Hybrid Objective

- Core mechanism hypothesis: current hard-negative-only no-loop training gives weak global geometry; adding in-batch positive classification may improve cross-query separation without frozen-standard inputs.
- Code components likely affected: `src/experiments.py`, `src/train.py`, configs, manifest.
- Why it may address the failure pattern: attacks representation geometry with a different objective, not role wording, MRL dimensions, or loop depth.
- Outside exhausted families: yes.
- Expected risk: medium; in-batch positives may over-separate domains or hurt NFCorpus/SCIDOCS.
- Estimated GPU cost: about 7 GPU hours.
- Smallest dev-only falsification: one no-loop candidate with fixed `inbatch_weight=0.25`; fail if SciFact/FiQA2018 remain negative or NFCorpus/SCIDOCS collapse.
- Requires: training and evaluation.

### Direction 4: Length/Truncation-Aware Document Encoding

- Core mechanism hypothesis: SciFact/FiQA evidence may be truncated or diluted; chunked document scoring can recover missed evidence.
- Code components likely affected: `src/eval_mteb.py`, evaluator/index code, manifest exports.
- Why it may address the failure pattern: targets document construction and truncation rather than training loss.
- Outside exhausted families: yes.
- Expected risk: medium; MTEB wrapper changes can be subtle and runtime may increase.
- Estimated GPU cost: about 4-8 GPU hours for eval-only.
- Smallest dev-only falsification: one fixed chunk aggregation rule on all dev tasks; fail if FiQA/SciFact do not lift.
- Requires: diagnostics and evaluation-only changes first.

### Direction 5: Embedding Norm/Isotropy Regularization

- Core mechanism hypothesis: role+MRL may leave anisotropy or norm bias that harms SciFact/FiQA; a candidate-only uniformity regularizer may improve geometry.
- Code components likely affected: `src/losses.py`, `src/train.py`, configs, manifest.
- Why it may address the failure pattern: regularizes embedding geometry without another MRL dimension sweep.
- Outside exhausted families: mostly yes, though adjacent to embedding-geometry work.
- Expected risk: medium over-regularization risk.
- Estimated GPU cost: about 7-12 GPU hours.
- Smallest dev-only falsification: one fixed-weight regularizer candidate; fail if SciFact/FiQA remain negative or NFCorpus/SCIDOCS regress.
- Requires: training and evaluation.

### Direction 6: Candidate-Only Sparse+Dense Hybrid

- Core mechanism hypothesis: FiQA/SciFact may need exact term/entity matching missing from single-vector dense scoring.
- Code components likely affected: sparse retrieval path, `src/eval_mteb.py`, manifest validation.
- Why it may address the failure pattern: adds a non-frozen, candidate-internal lexical signal.
- Outside exhausted families: yes, but standalone-track eligibility must be reviewed carefully.
- Expected risk: medium-high; protocol classification and score normalization must be explicit.
- Estimated GPU cost: about 4-8 GPU hours.
- Smallest dev-only falsification: one predeclared global hybrid rule; fail if FiQA/SciFact do not improve or NFCorpus/SCIDOCS collapse.
- Requires: evaluation-only changes and protocol review.

### Direction 7: Audited Multi-Source Retrieval Data Mixture

- Core mechanism hypothesis: RLHN is under-diverse for finance/scientific claim retrieval; balanced multi-source supervision may improve transfer.
- Code components likely affected: dataset registry, sampler, `src/data.py`, configs.
- Why it may address the failure pattern: addresses training-distribution mismatch directly.
- Outside exhausted families: yes.
- Expected risk: high leakage/provenance risk.
- Estimated GPU cost: 20-40 GPU hours.
- Smallest dev-only falsification: one fixed-ratio audited mixture candidate; fail if FiQA/SciFact remain negative.
- Requires: data audit, training, and evaluation.

### Direction 8: Standalone Late Interaction

- Core mechanism hypothesis: token-level MaxSim can recover term/evidence matches lost by single-vector pooling.
- Code components likely affected: model, evaluator, indexing/scoring, loss.
- Why it may address the failure pattern: higher ceiling for entity-heavy tasks such as SciFact and FiQA2018.
- Outside exhausted families: yes.
- Expected risk: very high implementation and runtime risk.
- Estimated GPU cost: 30+ GPU hours.
- Smallest dev-only falsification: separate tiny late-interaction prototype, not this batch.
- Requires: training, evaluation, and diagnostics.

## Ranking

1. No-loop candidate-only in-batch hybrid objective: high standalone validity, clear implementation, low cost, and direct geometry test.
2. Query/document loop co-training: strongest mechanistic novelty for q/doc mismatch, but higher runtime risk.
3. No-loop seeded passage sampling: cheap isolation of data-order effects outside first-token loops.
4. Length/truncation-aware document encoding: promising and cheap, but evaluator/index changes need separate care.
5. Embedding norm/isotropy regularization: plausible, but adjacent to already-failed geometry regularization.
6. Candidate-only sparse+dense hybrid: useful, but protocol and score normalization risk are higher.
7. Audited multi-source data mixture: high potential, but too much provenance and budget risk for this cycle.
8. Standalone late interaction: high ceiling but out of scope for one efficient batch.

## Selected Portfolio: batch_016_dev

The next batch should include three `standalone_main` dev candidates:

- `r016_qdoc_final_mean_pool_t3_neg3`
  - Tests Direction 1.
  - Outside exhausted mechanism families: yes.
  - Included because it directly diagnoses query/document representation asymmetry and addresses recurring FiQA2018/SCIDOCS regressions.
  - Failure mode falsified: looped-doc symmetry is not sufficient to recover dev transfer.
  - Estimated cost: 7 GPU hours.
  - Expected information gain: high.
  - Candidate track: `standalone_main`.

- `r016_standard_seeded_sampling`
  - Tests Direction 2.
  - Outside exhausted mechanism families: outside first-token loop-memory and no-loop role/MRL; overlaps negative sampling but isolates it in a no-loop candidate.
  - Included because it tests whether sampling order, rather than loop memory, drives some regressions.
  - Failure mode falsified: deterministic random positive/negative sampling alone is not sufficient.
  - Estimated cost: 7 GPU hours.
  - Expected information gain: medium.
  - Candidate track: `standalone_main`.

- `r016_standard_inbatch_hybrid`
  - Tests Direction 3.
  - Outside exhausted mechanism families: yes.
  - Included because it tests candidate-only cross-query contrastive geometry, not prefix wording, MRL dimensions, or loop depth.
  - Failure mode falsified: no-loop in-batch positives are not sufficient to improve SciFact/FiQA2018 without losing NFCorpus/SCIDOCS.
  - Estimated cost: 7 GPU hours.
  - Expected information gain: high.
  - Candidate track: `standalone_main`.

Portfolio size: 3 candidates, 21 estimated GPU hours against the configured 24 GPU-hour batch limit, with at most 3 concurrent GPU jobs.

This is not an uninformative local sweep. It does not vary first-token loop indices, first-token loss schedules, prefix wording, MRL dimension lists, label smoothing, or frozen-standard scoring. It tests three different mechanisms: q/doc symmetry, no-loop sampling-order sensitivity, and candidate-only in-batch geometry.

If all three candidates fail, the result would imply that cheap standalone pivots after role/MRL are not enough. The next cycle should either stop or invest in higher-risk mechanisms with stronger prior justification, such as length-aware document encoding, audited multi-source data, or late interaction.

Postprocess behavior must use `scripts/goal_submit_batch.py --submit --submit-postprocess`, producing one Slurm-native postprocess job after all eval jobs with deterministic status, collection, and scoreboard files under `outputs/goal/runs/batch_016_dev/`.
