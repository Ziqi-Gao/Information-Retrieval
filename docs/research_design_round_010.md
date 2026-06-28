# Research Design Round 010

Date: 2026-06-28

This plan was written after `batch_016_dev` completed Slurm-native postprocess and before creating, validating, dry-running, preflighting, or submitting `batch_017_dev`. It uses dev-task evidence only.

## Trigger

`batch_016_dev` completed with `postprocess_done.json`, `collected_results.csv`, and `scoreboard.json`. The local `goal_status.py` refresh hung inside Slurm querying, so the parent used deterministic file evidence.

All three candidates were `standalone_main`, but none produced a strong viable global dev signal:

- `r016_standard_seeded_sampling__loop1`: SciFact `+0.00834`, NFCorpus `+0.00045`, SCIDOCS `+0.00386`, FiQA2018 `+0.00086`, mean delta `+0.0033775`, min delta `+0.00045`.
- `r016_standard_inbatch_hybrid__loop1`: SciFact `-0.00686`, NFCorpus `-0.00132`, SCIDOCS `+0.00101`, FiQA2018 `-0.00142`, mean delta `-0.0021475`, min delta `-0.00686`.
- `r016_qdoc_final_mean_pool_t3_neg3__loop3`: SciFact `-0.07807`, NFCorpus `-0.04792`, SCIDOCS `-0.02024`, FiQA2018 `-0.05100`, mean delta `-0.0493075`, min delta `-0.07807`.

`minimal_positive_signal=false`, `research_grade_threshold_pass=false`, `fusion_diagnostic_pass=false`, `main_goal_success=false`, and `publishable_score_candidate=false` for all candidates.

## What Has Been Tried

- Early frozen-standard retrieval-time fusion produced diagnostic evidence only. It remains `fusion_diagnostic` and cannot trigger `main_goal_success`.
- Standalone first-token loop-memory batches tested loop-depth tuning, memory modes, detached memory, shorter horizons, lower hard-negative count, document-loop evaluation, tail-weighted loop loss, adjacent-loop consistency, self-residual scoring, in-batch hybrid loss, pairwise loss, seeded random passage sampling, first-token pooling, standard-to-loop warmup, middle-window negatives, sparse late-loop supervision, and label-smoothed listwise loss.
- No-loop pivots tested role prompting, dimensional Matryoshka supervision, role+MRL, deterministic seeded passage sampling, and candidate-only in-batch geometry.
- Batch 016 tested q/doc loop symmetry and broadly falsified it at the reduced `tmax=3` setting.

## Track Distinction

- `fusion_diagnostic` can inform hypotheses but cannot be promoted into `main_goal_success`.
- `standalone_main` must use candidate-only scoring: no frozen-standard checkpoint, frozen-standard embedding, frozen-standard score, weighted standard+candidate concatenation, or standard-score interpolation.
- The next batch is dev-only. Even a strong result would require separate user approval before final validation.

## Failure Pattern

- `FiQA2018` remains the recurring weakest dev task.
- `SCIDOCS` sometimes improves, especially for role+MRL and seeded sampling, but it is not consistently robust.
- `NFCorpus` can swing positive or negative depending on mechanism and remains below even the weak margin in the best batch 016 candidate.
- `SciFact` is sensitive: seeded sampling helps it, while q/doc loop symmetry and in-batch geometry hurt it.
- The latest failures look like a mix of document construction/truncation sensitivity, exact term or entity matching limits, and training-distribution mismatch. Q/doc loop symmetry and no-loop in-batch objectives are not enough.

## Exhausted Families For Now

- First-token loop-memory standalone search is exhausted.
- Local loop-loss and loop-depth variants are exhausted.
- No-loop role prompting and dimensional-MRL local variants are exhausted.
- Q/doc loop symmetry is exhausted for now as a primary fix because the reduced candidate regressed all four dev tasks.
- No-loop in-batch hybrid is exhausted for now because it regressed three of four dev tasks.
- No-loop seeded sampling is weakly positive but not a sufficient final candidate; do not follow it with seed, negative-count, or sample-size micro-sweeps.

Further local sweeps are low-value because recent batches repeatedly shift gains among one or two dev tasks while failing the global no-regression and margin requirements.

## Research Directions Considered

### Direction 1: Length/Truncation-Aware Document Encoding

- Core mechanism hypothesis: FiQA2018, SciFact, and SCIDOCS may lose evidence when documents are truncated or represented by only the leading text.
- Code components likely affected: `src/eval_mteb.py`, `scripts/slurm_eval.sbatch`, `scripts/goal_submit_batch.py`, manifest.
- Why it may address the pattern: directly changes document construction at evaluation without altering NDCG semantics or using frozen-standard inputs.
- Outside exhausted families: yes.
- Expected risk: medium; chunk averaging may dilute strong head evidence.
- Estimated GPU cost: about 4 GPU hours for four dev tasks.
- Smallest dev-only falsification: evaluate one candidate checkpoint with fixed global chunked-doc aggregation; fail if FiQA2018/SCIDOCS do not improve without hurting SciFact.
- Requires: evaluation-only changes.

### Direction 2: Candidate-Only Dense+Lexical Hash Hybrid

- Core mechanism hypothesis: dense single-vector retrieval misses exact term/entity matches important to finance, scientific claims, and citation-style retrieval.
- Code components likely affected: `src/eval_mteb.py`, `scripts/slurm_eval.sbatch`, `scripts/goal_submit_batch.py`, manifest.
- Why it may address the pattern: adds a candidate-internal lexical signal without frozen-standard embeddings or scores.
- Outside exhausted families: yes.
- Expected risk: medium; fixed lexical weight can hurt semantic tasks.
- Estimated GPU cost: about 4 GPU hours for four dev tasks.
- Smallest dev-only falsification: evaluate one fixed hashed lexical+dense rule on all dev tasks; fail if FiQA2018 and SCIDOCS remain below margin or SciFact regresses.
- Requires: evaluation-only changes.

### Direction 3: Query/Document Text Construction Audit

- Core mechanism hypothesis: title/body concatenation and missing metadata handling differ across MTEB tasks and may distort document representations.
- Code components likely affected: `src/eval_mteb.py`, task-specific text builders, manifest.
- Why it may address the pattern: targets task-format sensitivity rather than training objective.
- Outside exhausted families: yes.
- Expected risk: medium-high; task-specific construction can become hard to audit.
- Estimated GPU cost: 2-6 GPU hours.
- Smallest dev-only falsification: one global title/body construction rule; fail if all weak tasks remain weak.
- Requires: diagnostics and evaluation-only changes.

### Direction 4: Embedding Isotropy Or Uniformity Regularization

- Core mechanism hypothesis: candidate embeddings may be anisotropic despite L2 normalization, hurting transfer.
- Code components likely affected: `src/losses.py`, `src/train.py`, `src/experiments.py`, configs, manifest.
- Why it may address the pattern: regularizes geometry without another MRL dimension-list sweep.
- Outside exhausted families: partly; adjacent to failed geometry work but mechanistically distinct.
- Expected risk: medium over-regularization risk.
- Estimated GPU cost: about 7 GPU hours.
- Smallest dev-only falsification: one fixed regularizer weight candidate.
- Requires: training and evaluation.

### Direction 5: Audited Multi-Source Retrieval Data Mixture

- Core mechanism hypothesis: RLHN alone under-covers finance/scientific/citation retrieval styles.
- Code components likely affected: dataset registry, data audit docs, `src/data.py`, configs, manifest.
- Why it may address the pattern: addresses training-distribution mismatch directly.
- Outside exhausted families: yes.
- Expected risk: high leakage/provenance risk.
- Estimated GPU cost: 20-40 GPU hours.
- Smallest dev-only falsification: one small audited mixture candidate.
- Requires: data audit, training, and evaluation.

### Direction 6: Candidate-Mined Hard-Negative Refresh

- Core mechanism hypothesis: static RLHN negative order is not the right boundary for transfer tasks.
- Code components likely affected: mining script, cached dataset format, `src/data.py`, `src/train.py`, configs, manifest.
- Why it may address the pattern: attacks negative quality rather than loop memory.
- Outside exhausted families: yes.
- Expected risk: medium-high; can overfit the miner and adds operational complexity.
- Estimated GPU cost: 18-30 GPU hours.
- Smallest dev-only falsification: one candidate-mined cache with no frozen-standard miner.
- Requires: mining, training, and evaluation.

### Direction 7: Standalone Late Interaction

- Core mechanism hypothesis: token-level MaxSim can recover evidence lost by single-vector pooling.
- Code components likely affected: model, evaluator, indexing/scoring, loss.
- Why it may address the pattern: entity and term evidence may matter on SciFact and FiQA2018.
- Outside exhausted families: yes.
- Expected risk: very high implementation and runtime risk.
- Estimated GPU cost: 30+ GPU hours.
- Smallest dev-only falsification: separate tiny prototype before goal-batch submission.
- Requires: training, evaluation, and diagnostics.

## Ranking

1. Length/truncation-aware document encoding: high relevance to weak tasks, low cost, clear falsification.
2. Candidate-only dense+lexical hash hybrid: high novelty relative to previous dense-only work, low cost, clear standalone validity.
3. Query/document text construction audit: relevant but needs more diagnostics to avoid task-specific tuning.
4. Embedding isotropy/uniformity regularization: plausible but adjacent to failed MRL/in-batch geometry work.
5. Audited multi-source retrieval data mixture: likely important but high provenance and budget risk.
6. Candidate-mined hard-negative refresh: valid but operationally heavier than this cycle allows.
7. Standalone late interaction: high ceiling but too large for one efficient batch.

## Selected Portfolio: batch_017_dev

The next batch should include two eval-only `standalone_main` dev candidates based on the weakly non-regressing `r016_standard_seeded_sampling` checkpoint. The checkpoint is used only as a candidate carrier; the tested mechanisms are global evaluation rules, not seeded-sampling micro-tuning.

- `r017_seeded_chunked_docs`
  - Tests Direction 1.
  - Outside exhausted mechanism families: yes.
  - Candidate rule: evaluate `r016_standard_seeded_sampling` with fixed chunked document aggregation, `doc_chunk_words=180`, `doc_chunk_stride=120`, `doc_chunk_max_chunks=8`, `loop_idx=1`, on all dev tasks.
  - Failure mode falsified: document truncation or late-document evidence is the main blocker.
  - Estimated cost: 4 GPU hours.
  - Expected information gain: high.
  - Candidate track: `standalone_main`.

- `r017_seeded_lexical_hash`
  - Tests Direction 2.
  - Outside exhausted mechanism families: yes.
  - Candidate rule: evaluate `r016_standard_seeded_sampling` with fixed candidate-only lexical hashing, `lexical_hash_dim=1024`, `lexical_weight=0.15`, `loop_idx=1`, on all dev tasks.
  - Failure mode falsified: exact lexical/entity mismatch is the main blocker.
  - Estimated cost: 4 GPU hours.
  - Expected information gain: high.
  - Candidate track: `standalone_main`.

Portfolio size: 2 candidates, 8 estimated GPU hours against the configured 24 GPU-hour batch limit, with at most 2 concurrent GPU jobs.

This is not an uninformative local sweep. It does not vary seed, negatives, loop depth, prefix wording, MRL dimensions, in-batch weight, q/doc loop depth, or frozen-standard scoring. It tests two new evaluator-level standalone mechanisms directly aimed at recurring weak tasks.

If both candidates fail, the result will support the conclusion that cheap standalone dense-candidate fixes and evaluator-level text/lexical adjustments are insufficient, making higher-risk data-mixture or late-interaction directions the next meaningful options.

Postprocess must use `scripts/goal_submit_batch.py --submit --submit-postprocess`, producing one Slurm-native postprocess job after all eval jobs.
