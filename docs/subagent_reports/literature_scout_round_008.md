# literature_scout report

This read-only pass used local/recalled literature reasoning and did not use the network, edit files, train, evaluate, submit, or inspect secrets. Key local evidence: recent standalone work covered many first-token local repairs, but FiQA2018/SCIDOCS regressions persisted; `batch_014_dev` produced 0/4 dev-task wins for both candidates.

## High-priority low-risk ideas

| direction | motivation | why it may help FiQA/SCIDOCS | risk | likely files | cost |
|---|---|---|---|---|---|
| Query/document role prompting | E5/BGE/GTR-style `query:` / `passage:` prefixes often stabilize cross-domain retrieval. | FiQA/SCIDOCS have short queries and different document registers; explicit role text may reduce query/doc ambiguity while preserving SciFact/NFCorpus semantic matching. | medium | `src/data.py`, `src/eval_mteb.py`, `src/train.py`, `src/experiments.py`, configs/manifests | 8-10 GPUh |
| Dimensional Matryoshka embedding loss | Matryoshka Representation Learning trains nested embedding prefixes without adding heads. | Multi-resolution embedding supervision may regularize geometry and reduce first-token loop drift while staying single-vector dense standalone. | medium | `src/losses.py`, `src/model.py`, `src/train.py`, `src/eval_mteb.py`, `src/experiments.py` | 10-12 GPUh |
| Candidate-only role prefix plus dimensional MRL | Combines input-role stability with embedding regularization. | If FiQA/SCIDOCS failure is role ambiguity plus overfit geometry, the combination is more informative than another first-token local sweep. | medium | same as above | 12-14 GPUh |

## Medium-risk ideas

| direction | motivation | why it may help FiQA/SCIDOCS | risk | likely files | cost |
|---|---|---|---|---|---|
| Teacher soft-label distillation, no standard teacher | RocketQA/TAS-B/GPL-style soft labels from a non-frozen-standard teacher. | RLHN hard labels/negative order may be too brittle; soft labels can preserve close negative ordering. | medium-high | `src/data.py`, cached teacher-score reader, `src/losses.py`, `src/train.py`, configs | 16-30 GPUh plus teacher scoring |
| Multi-source retrieval data mixture | Contriever/E5/BGE experience suggests cross-domain retrieval often needs multi-source supervision. | Finance/scientific/citation-like data may cover FiQA/SCIDOCS while balanced sampling protects SciFact/NFCorpus. | high | `src/data.py`, dataset registry/sampler, configs/manifests | 20-40 GPUh |
| Candidate-internal hard-negative mining refresh | ANCE/DPR/RocketQA-style iterative mining. | Candidate-mined negatives may better shape target retrieval boundaries than fixed first/middle/random negatives. | medium-high | mining script, `src/data.py`, `src/train.py`, configs | 18-30 GPUh |

## High-risk ideas

| direction | motivation | why it may help FiQA/SCIDOCS | risk | likely files | cost |
|---|---|---|---|---|---|
| Query/document loop co-training | Current training loops queries against one-pass docs; q/doc loop symmetry tests representation mismatch directly. | SCIDOCS/FiQA long/specialized docs may need doc-side iterative encoding. | high OOM/timeout risk | `src/model.py`, `src/train.py`, `src/losses.py`, `src/eval_mteb.py`, configs | 14-24 GPUh reduced |
| Dense late-interaction candidate | ColBERT-style MaxSim keeps token-level evidence. | FiQA/SCIDOCS may need exact entity/term matching. | very high; larger scoring semantic shift | major model/evaluator/index work | 30+ GPUh |

## Recommended First Batch

Avoid first-token loop loss/idx work. Use a dev-only batch with three complementary `standalone_main` candidates:

1. `role_prompt_standard`: query/passage role prompting, single-vector, candidate-only.
2. `dim_mrl_standard`: dimensional Matryoshka loss, single-vector, candidate-only.
3. `role_prompt_dim_mrl`: combined low-risk mechanisms.

Do not include teacher distillation or q/doc loop co-training in the same batch because they have higher engineering and budget risk.

## Sources Or Rationale

Reasoning is based on common retrieval literature patterns: DPR/ANCE/RocketQA/TAS-B/GPL for hard-negative mining and distillation; Contriever/E5/BGE/GTR for multi-source data and instruction/prefixing; Matryoshka Representation Learning for multi-resolution embedding regularization; ColBERT for late interaction.

## Uncertainty

These are research directions, not result claims. The frozen standard may already be strong enough that low-risk single-vector changes only reduce regressions without clearing protocol thresholds.
