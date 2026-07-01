# Literature Scout Round 009

## Scope

Read-only pass. Reviewed `docs/goal_protocol.md`, `docs/research_design_round_008.md`, `docs/agent_lab_notebook.md`, `experiments/batches/batch_015_dev.yaml`, and `outputs/goal/runs/batch_015_dev/scoreboard.json`. No files edited, no Slurm/training/eval actions run.

## Batch 015 Signal

Best dev signal is `r015_role_dim_mrl_standard__loop1`: SciFact `-0.00320`, FiQA2018 `-0.00255`, NFCorpus `+0.00862`, SCIDOCS `+0.00562`, mean `+0.00212`. This is still not viable because SciFact and FiQA regress, but it gives a useful anchor: preserve role+MRL's NFCorpus/SCIDOCS gains while adding a non-local mechanism for finance/scientific claim robustness.

Avoid for `batch_016`: frozen-standard fusion, first-token loop-memory variants, role-prefix wording tweaks, and nearby MRL dimension-list sweeps.

## Broad Standalone Directions

1. **Audited Domain-Balanced Role+MRL Training**
   - Mechanism: add a small audited finance/scientific retrieval mixture while retaining role+MRL, so FiQA/SciFact see closer supervision without sacrificing NFCorpus/SCIDOCS geometry.
   - Components: dataset registry, sampler, configs, manifest; maybe `src/data.py`, `src/train.py`.
   - Risk/cost: high leakage/provenance risk; about 20-40 GPU hours.
   - Falsification: one fixed-ratio dev candidate; fail if FiQA/SciFact remain negative or NFCorpus/SCIDOCS lose diagnostic positives.
   - Needs: diagnostics, training, eval.

2. **Non-Standard Teacher Soft Labels**
   - Mechanism: use a predeclared external/non-standard teacher for graded negative ordering; no frozen standard checkpoint, embeddings, or scores.
   - Components: teacher score cache, loss path, config/manifest metadata.
   - Risk/cost: medium-high reproducibility/licensing risk; about 16-30 GPU hours plus teacher scoring.
   - Falsification: one role+MRL-distilled candidate on the four dev tasks; fail if FiQA/SciFact do not improve while positives collapse.
   - Needs: diagnostics/cache generation, training, eval.

3. **Candidate-Only Hard-Negative Mining Refresh**
   - Mechanism: static negatives may shape poor FiQA/SciFact boundaries; mine harder negatives with a candidate-only model, then retrain role+MRL.
   - Components: mining script/cache, sampler, `src/data.py`, `src/train.py`.
   - Risk/cost: medium-high miner bias and operational complexity; about 18-30 GPU hours.
   - Falsification: one mined-cache candidate, fixed mining rule; fail on unchanged FiQA/SciFact regression or lost NFCorpus/SCIDOCS.
   - Needs: mining diagnostics, training, eval.

4. **Length/Truncation-Aware Document Encoding**
   - Mechanism: SciFact/FiQA evidence may be truncated or diluted; chunk documents and aggregate candidate-only chunk scores using the role+MRL checkpoint.
   - Components: `src/eval_mteb.py`, document encoder/index path, manifest exports.
   - Risk/cost: medium index/runtime risk; about 4-8 GPU hours eval-only, about 8-14 if later trained.
   - Falsification: one fixed chunking/aggregation rule on dev tasks; fail if FiQA/SciFact do not lift or positives disappear.
   - Needs: length diagnostics and eval-only first.

5. **Embedding Norm/Isotropy Regularization**
   - Mechanism: role+MRL may preserve broad structure but leave task-specific norm/anisotropy bias; add candidate-only norm/uniformity regularization, not another MRL dimension sweep.
   - Components: `src/losses.py`, `src/train.py`, configs.
   - Risk/cost: medium over-regularization risk; about 7-12 GPU hours.
   - Falsification: one role+MRL+regularizer candidate; fail if SciFact/FiQA stay negative or NFCorpus/SCIDOCS regress.
   - Needs: embedding diagnostics, training, eval.

6. **Candidate-Only Sparse+Dense Hybrid**
   - Mechanism: FiQA/SciFact may need exact entity/term matching; combine BM25 with role+MRL dense scores using one predeclared global weight.
   - Components: sparse index/scoring path, `src/eval_mteb.py`, manifest validation.
   - Risk/cost: medium-high; confirm protocol accepts this as `standalone_main` before use. About 4-8 GPU hours.
   - Falsification: one global hybrid rule; fail if FiQA/SciFact do not improve or SCIDOCS/NFCorpus gains vanish.
   - Needs: eval-only plus protocol/diagnostic check.

7. **Candidate-Only Query Expansion / PRF**
   - Mechanism: FiQA questions and SciFact claims are short; candidate-only pseudo-relevance feedback may add missing terms without frozen-standard fusion.
   - Components: eval-time query rewrite/feedback path, configs, manifest metadata.
   - Risk/cost: medium-high query-drift risk; about 4-8 GPU hours.
   - Falsification: one fixed PRF rule on all dev tasks; fail if FiQA/SciFact remain below baseline or positives collapse.
   - Needs: eval-only diagnostics first.

8. **Standalone Late Interaction**
   - Mechanism: token-level MaxSim can recover term/evidence matches that single-vector role+MRL misses.
   - Components: model scoring/index path, loss, evaluator.
   - Risk/cost: very high implementation and runtime risk; 30+ GPU hours.
   - Falsification: tiny dev-only prototype after separate design approval; fail if no FiQA/SciFact lift.
   - Needs: training, eval, diagnostics.

## Suggested Batch 016 Shortlist

Highest information per cost: direction 4, direction 5, and direction 6 if standalone-track validation is acceptable. Defer directions 1-3 until provenance/cache design is audited; defer direction 8 unless the next cycle explicitly accepts a larger scoring-architecture change.
