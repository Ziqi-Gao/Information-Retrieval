# Future Research Seed

This document is a preparation artifact only. Do not implement these ideas in the preparation task.

Light literature seeds used:

- ANCE: https://arxiv.org/abs/2007.00808
- ColBERT: https://arxiv.org/abs/2004.12832
- Dense retrieval PRF: https://arxiv.org/abs/2108.13454
- Multiple-representation PRF: https://arxiv.org/abs/2106.11251
- BEIR benchmark: https://arxiv.org/abs/2104.08663

## Candidate Experiment Families

### 1. Standard-Preserving Embedding Fusion

- Hypothesis: a fixed weighted fusion of standard and loop query embeddings may keep the standard baseline's strengths while adding loop-specific gains.
- Why it might improve all-task NDCG@10: conservative fusion can reduce regressions on tasks where loops drift.
- Risk: fusion weights may overfit dev tasks or dilute real loop gains.
- Code areas: evaluation wrapper, experiment registry, manifest protocol.
- Expected GPU cost: low for evaluation-only fusion, medium if trained.
- Risk level: medium.

### 2. Standard/Loop Score Fusion

- Hypothesis: combining standard and loop similarity scores with a pre-registered global weight may outperform either score alone.
- Why it might improve all-task NDCG@10: score fusion can preserve standard rankings while promoting loop-supported documents.
- Risk: task-specific score distributions may make one global weight fragile.
- Code areas: scoring/evaluation pipeline, result collection, pre-registered fusion config.
- Expected GPU cost: low if reusing embeddings, medium if re-evaluating all tasks.
- Risk level: medium.

### 3. Loop-Depth Weighted Supervision

- Hypothesis: a monotonic or curriculum weight over loop losses can improve deeper loops without harming early-loop stability.
- Why it might improve all-task NDCG@10: it may avoid over-penalizing shallow loops while still training final representations.
- Risk: extra training objective can hurt some datasets and needs careful ablation.
- Code areas: `src/losses.py`, `src/experiments.py`, configs.
- Expected GPU cost: medium.
- Risk level: medium.

### 4. Loop Consistency Regularization

- Hypothesis: regularizing adjacent loop embeddings to remain semantically close may reduce query drift.
- Why it might improve all-task NDCG@10: drift control is useful when PRF-style or recurrent updates over-specialize.
- Risk: too much consistency may prevent useful refinement.
- Code areas: `src/losses.py`, `src/train.py`, experiment registry.
- Expected GPU cost: medium.
- Risk level: medium-high.

### 5. Shallow-Loop Early Fusion

- Hypothesis: using only the first few loop depths with pre-registered fusion can capture refinement benefits at lower cost.
- Why it might improve all-task NDCG@10: shallow loops may be less prone to late-loop degradation.
- Risk: some tasks may require deeper context or show no shallow-loop benefit.
- Code areas: evaluation candidate definition, collection, scoreboard candidate IDs.
- Expected GPU cost: low to medium.
- Risk level: low-medium.

### 6. Retrieval-Time Union Reranking

- Hypothesis: unioning top-k documents from standard and loop candidates before a cheap rerank can improve recall without retraining.
- Why it might improve all-task NDCG@10: complementary candidate pools may recover task-specific misses.
- Risk: reranking may require a new scoring pipeline and could break the preparation constraint if not scoped later.
- Code areas: future retrieval pipeline, result validation, manifest schema.
- Expected GPU cost: medium for all-task evaluation.
- Risk level: high.

### 7. Hard-Negative Weighting Variants

- Hypothesis: weighting hard negatives by difficulty may improve robustness over the current fixed hard-negative CE.
- Why it might improve all-task NDCG@10: ANCE-style lessons suggest negative quality strongly affects dense retrieval.
- Risk: aggressive weighting can overfit RLHN negatives and hurt BEIR-style transfer tasks.
- Code areas: `src/losses.py`, `src/data.py`, configs.
- Expected GPU cost: medium-high.
- Risk level: medium-high.

### 8. Recurrent Query Feedback With Drift Control

- Hypothesis: recurrent query feedback can help if paired with explicit drift limits or standard anchoring.
- Why it might improve all-task NDCG@10: PRF-style work suggests query refinement can help dense retrieval when noise is controlled.
- Risk: query drift can selectively hurt out-of-domain datasets.
- Code areas: `src/model.py`, `src/experiments.py`, `src/losses.py`.
- Expected GPU cost: high.
- Risk level: high.
