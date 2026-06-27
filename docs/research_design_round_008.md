# Research Design Round 008

Date: 2026-06-27

This plan was written after `batch_014_dev` completed Slurm-native postprocess and before validating, dry-running, preflighting, or submitting the next batch. It uses dev-task evidence only and does not use final-task deltas to tune dev candidates.

## Trigger

`batch_014_dev` completed with `postprocess_done.json`, `collected_results.csv`, and `scoreboard.json`. It was dev-only and cannot trigger `main_goal_success`.

Both candidates were `standalone_main`, but neither produced a viable global dev signal:

- `r014_sparse_late_first_token_t10__loop10`: SciFact `-0.00584`, NFCorpus `-0.01030`, SCIDOCS `-0.00732`, FiQA2018 `-0.01268`, mean delta `-0.009035`, min delta `-0.01268`.
- `r014_label_smooth_first_token_t10__loop10`: SciFact `-0.02339`, NFCorpus `-0.01375`, SCIDOCS `-0.00510`, FiQA2018 `-0.01087`, mean delta `-0.0132775`, min delta `-0.02339`.

No task won. `minimal_positive_signal=false`, `research_grade_threshold_pass=false`, `fusion_diagnostic_pass=false`, `main_goal_success=false`, and `publishable_score_candidate=false` for both candidates.

## What Has Been Tried

- Early standard+loop retrieval-time fusion produced diagnostic evidence only when using frozen-standard checkpoint/embedding inputs. Those results remain `fusion_diagnostic` and cannot trigger `main_goal_success`.
- Standalone dev batches have tested fixed loop depths, mean-pool loops, recurrent/no-memory variants, first-token memory, token-concat memory, final-loop and loopwise losses, detached memory, shorter horizons, lower hard-negative count, document-loop evaluation, tail-weighted loop loss, adjacent-loop consistency, candidate-internal self-residual scoring, in-batch hybrid loss, pairwise ranking loss, seeded random passage sampling, first-token retrieval pooling, standard-to-loop warmup, deterministic middle-window negatives, sparse late-loop supervision, and label-smoothed listwise loss.
- No recent `standalone_main` dev batch produced a viable global dev signal.

## Track Distinction

- `fusion_diagnostic` candidates can guide mechanism discovery but cannot be promoted to `main_goal_success`.
- `diagnostic` candidates can be reported but cannot trigger the main goal.
- `standalone_main` candidates must score with candidate-only outputs and no frozen-standard checkpoint, embedding, score, weighted standard+candidate concatenation, or interpolation.
- The next batch is dev-only, so even a strong result would require separate user-approved final validation.

## Failure Pattern

- `FiQA2018` is the most persistent standalone regression.
- `SCIDOCS` is the second recurring blocker.
- First-token loop candidates sometimes improved `SciFact` and `NFCorpus`, but `batch_014_dev` removed even that split-positive pattern: both candidates regressed all four dev tasks.
- Recent local first-token changes to supervision topology and target calibration did not help. Further first-token loop-depth, first-token loss, first-token supervision schedule, first-token negative-window, or first-token label-smoothing sweeps are low-value.
- The first-token loop-memory standalone family is exhausted for now as a primary path.

## Research Directions Considered

### Direction 1: Query/Document Role Prompting

- Core mechanism hypothesis: retrieval input formatting is under-specified; global `query:` and `passage:` prefixes may stabilize query/document geometry across domains.
- Code components likely affected: `src/model.py`, `src/train.py`, configs, manifest.
- Why it may address the failure pattern: FiQA2018 and SCIDOCS have short or domain-specific queries and specialized document text; role prefixes may reduce query/document ambiguity without loop memory.
- Outside first-token loop-memory family: yes.
- Expected risk: medium; ModernBERT may not benefit from literal prefixes.
- Estimated GPU cost: about 7 GPU hours.
- Smallest dev-only falsification: one no-loop candidate with prefixes applied globally to training and dev evaluation; fail if FiQA2018 or SCIDOCS still regress and macro mean is non-positive.
- Requires: training and evaluation through existing single-vector path with model/config support for prefixes.

### Direction 2: Dimensional Matryoshka Embedding Supervision

- Core mechanism hypothesis: nested embedding-prefix losses regularize the representation and reduce cross-domain geometry drift without adding heads.
- Code components likely affected: `src/losses.py`, `src/train.py`, `src/experiments.py`, configs, manifest.
- Why it may address the failure pattern: regularizing full and lower-dimensional prefixes may reduce overfitting to RLHN hard-negative ordering while preserving single-vector retrieval.
- Outside first-token loop-memory family: yes.
- Expected risk: medium; it may be too conservative or hurt full-dimension ranking.
- Estimated GPU cost: about 7 GPU hours.
- Smallest dev-only falsification: one no-loop candidate with dimensions `[768, 384, 192]`, full embedding evaluation on all dev tasks.
- Requires: training changes; evaluation uses existing full-embedding path.

### Direction 3: Role Prompting Plus Dimensional Matryoshka Supervision

- Core mechanism hypothesis: role prefixes address input ambiguity while dimensional MRL regularizes geometry; together they may fix regressions neither mechanism fixes alone.
- Code components likely affected: same as Directions 1 and 2.
- Why it may address the failure pattern: combines the two lower-risk, non-loop mechanisms most directly aimed at FiQA2018/SCIDOCS transfer.
- Outside first-token loop-memory family: yes.
- Expected risk: medium; attribution is less clean if only the combination works.
- Estimated GPU cost: about 7 GPU hours.
- Smallest dev-only falsification: one no-loop candidate with both role prefixes and dimensions `[768, 384, 192]`.
- Requires: training changes; evaluation uses existing full-embedding path.

### Direction 4: Query/Document Loop Co-Training

- Core mechanism hypothesis: training looped query states against one-pass documents creates representation mismatch; looped docs in training and evaluation may reduce this mismatch.
- Code components likely affected: `src/model.py`, `src/train.py`, `src/losses.py`, `src/eval_mteb.py`, configs, manifest.
- Why it may address the failure pattern: directly tests the q/doc asymmetry hypothesis for FiQA2018 and SCIDOCS.
- Outside first-token loop-memory family: yes if implemented with mean-pool memory.
- Expected risk: high; document-loop backprop over positives and negatives can OOM or time out.
- Estimated GPU cost: 12 to 24 GPU hours for a reduced candidate.
- Smallest dev-only falsification: reduced `tmax=4`, fewer negatives, looped docs in train/eval.
- Requires: training and evaluation changes.

### Direction 5: Candidate-Only Teacher Soft Labels

- Core mechanism hypothesis: RLHN hard labels may be too brittle; non-standard teacher soft labels can preserve fine-grained negative ordering.
- Code components likely affected: data caching, `src/data.py`, `src/losses.py`, `src/train.py`, configs, manifest.
- Why it may address the failure pattern: FiQA2018/SCIDOCS may need softer relevance ordering than the current hard positive-vs-negative CE.
- Outside first-token loop-memory family: yes.
- Expected risk: medium-high; teacher selection, licensing, and cache reproducibility must be audited, and the frozen standard cannot be used.
- Estimated GPU cost: 16 to 30 GPU hours plus teacher scoring.
- Smallest dev-only falsification: one no-loop candidate using a predeclared non-standard teacher score cache.
- Requires: data generation and training changes.

### Direction 6: Multi-Source Retrieval Data Mixture

- Core mechanism hypothesis: single-source RLHN training is under-diverse for FiQA2018/SCIDOCS; balanced multi-source retrieval supervision may improve transfer.
- Code components likely affected: dataset registry, sampler, `src/data.py`, configs, manifest.
- Why it may address the failure pattern: adds finance/scientific/citation-like signal rather than another local loop objective.
- Outside first-token loop-memory family: yes.
- Expected risk: high; data leakage and reproducibility need careful audit.
- Estimated GPU cost: 20 to 40 GPU hours.
- Smallest dev-only falsification: one small balanced mixture candidate with documented dataset provenance.
- Requires: data and training changes.

### Direction 7: Candidate-Internal Hard-Negative Mining Refresh

- Core mechanism hypothesis: static first/middle/random negatives are too blunt; candidate-mined hard negatives better shape local ranking boundaries.
- Code components likely affected: mining script, cached dataset format, `src/data.py`, `src/train.py`, configs, manifest.
- Why it may address the failure pattern: attacks hard-negative quality rather than loop memory.
- Outside first-token loop-memory family: yes.
- Expected risk: medium-high; mining can overfit the miner and adds operational complexity.
- Estimated GPU cost: 18 to 30 GPU hours.
- Smallest dev-only falsification: one candidate-mined cache from a predeclared checkpoint, no frozen-standard miner.
- Requires: mining and training changes.

### Direction 8: Dense Late Interaction

- Core mechanism hypothesis: single-vector pooling loses term-level evidence needed by FiQA2018 and SCIDOCS; token-level MaxSim may recover it.
- Code components likely affected: model/evaluator/index path and possibly loss code.
- Why it may address the failure pattern: term/entity evidence may matter more on SCIDOCS and FiQA2018 than on the current single-vector loop candidates.
- Outside first-token loop-memory family: yes.
- Expected risk: very high; changes scoring architecture and cost profile substantially.
- Estimated GPU cost: 30+ GPU hours.
- Smallest dev-only falsification: one tiny dev-only late-interaction candidate after separate design approval.
- Requires: training and evaluation changes.

## Ranking

1. Query/document role prompting: high standalone validity, high novelty relative to first-token local search, medium FiQA/SCIDOCS relevance, low implementation risk, low cost, clear falsification.
2. Dimensional Matryoshka embedding supervision: high standalone validity, high novelty, medium FiQA/SCIDOCS relevance, low-to-medium implementation risk, low cost, clear falsification.
3. Role prompting plus dimensional MRL: high standalone validity, high novelty, highest low-risk chance of addressing both ambiguity and geometry, medium attribution risk, low cost.
4. Query/document loop co-training: strong mechanistic novelty and direct mismatch test, but high OOM/timeout risk in this one-batch cycle.
5. Candidate-only teacher soft labels: valid and potentially powerful, but teacher/caching work is too large for this resume cycle.
6. Multi-source retrieval data mixture: likely important, but leakage/provenance/budget risk is high.
7. Candidate-internal hard-negative mining refresh: valid, but needs an additional mining pipeline and is less clean than the no-loop candidates.
8. Dense late interaction: potentially high ceiling but outside the current efficient one-batch scope.

## Selected Portfolio: batch_015_dev

The next batch should include three `standalone_main` dev candidates:

- `r015_role_prompt_standard`
  - Tests Direction 1.
  - Outside first-token loop-memory family: yes.
  - Included because it directly tests whether explicit query/document roles reduce the recurring FiQA2018/SCIDOCS mismatch.
  - Failure mode falsified: input-role ambiguity is not sufficient to explain the dev regressions.
  - Estimated cost: 7 GPU hours.
  - Expected information gain: high.
  - Candidate track: `standalone_main`.

- `r015_dim_mrl_standard`
  - Tests Direction 2.
  - Outside first-token loop-memory family: yes.
  - Included because it tests parameter-free embedding-geometry regularization without loop memory.
  - Failure mode falsified: dimensional embedding regularization is not sufficient to recover cross-domain dev transfer.
  - Estimated cost: 7 GPU hours.
  - Expected information gain: high.
  - Candidate track: `standalone_main`.

- `r015_role_dim_mrl_standard`
  - Tests Direction 3.
  - Outside first-token loop-memory family: yes.
  - Included because the combination is still low-cost and tests whether the two non-loop mechanisms interact constructively.
  - Failure mode falsified: combining input-role semantics and embedding-prefix regularization is not sufficient to recover dev transfer.
  - Estimated cost: 7 GPU hours.
  - Expected information gain: medium to high.
  - Candidate track: `standalone_main`.

Portfolio size: 3 candidates, 21 estimated GPU hours against the configured 24 GPU-hour batch limit, with at most 3 concurrent GPU jobs.

This is not an uninformative local sweep. It does not change `loop_idx`, evaluate neighboring depths, adjust first-token loop supervision, vary first-token negative windows, tune label smoothing, or reuse frozen-standard scoring. All candidates are no-loop single-vector candidates with global rules across the four dev tasks.

If all three candidates fail, the result would imply that low-risk no-loop input-format and embedding-regularization pivots are also insufficient, and the next cycle should either stop or consider higher-risk mechanisms such as q/doc loop co-training, non-standard teacher soft labels, or audited multi-source training.

Postprocess behavior must use `scripts/goal_submit_batch.py --submit --submit-postprocess`, producing one Slurm-native postprocess job after all eval jobs with deterministic status, collection, and scoreboard files under `outputs/goal/runs/batch_015_dev/`.
