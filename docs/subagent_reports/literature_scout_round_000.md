# literature_scout report

## High-priority low-risk ideas

- **Standard/loop score fusion**: mechanism: pre-register one global weight `score = (1-a)*standard + a*loop`, with `a=0` as fallback; why improve NDCG@10: keeps standard ranking pressure while allowing loop gains near top-10; risk of hurting datasets: score-scale mismatch across SciFact/NFCorpus/FiQA can demote fragile standard wins; files likely affected: `src/eval_mteb.py`, `src/experiments.py`, `scripts/goal_scoreboard.py`, `experiments/batches/*.yaml`; estimated GPU cost: low if reusing checkpoints, medium if re-evaluating all tasks; risk level: low; preserves standard baseline fallback: yes.

- **Rank-level fusion / RRF over standard + loop depths**: mechanism: fuse ranked lists instead of raw scores, e.g. reciprocal-rank-style union from standard and fixed loop candidate; why improve NDCG@10: avoids score calibration and can recover documents missed by one retriever; risk of hurting datasets: union noise can move weak false positives into top-10; files likely affected: `src/eval_mteb.py` or a future retrieval-run combiner, `scripts/goal_scoreboard.py`, manifests; estimated GPU cost: low after embeddings/runs exist; risk level: low; preserves standard baseline fallback: yes.

- **Standard-anchored query embedding interpolation**: mechanism: query vector `normalize((1-a)*q_standard + a*q_loop)` with small pre-registered global `a`; why improve NDCG@10: constrains query drift while letting looped query compute add semantic refinement; risk of hurting datasets: interpolation can dilute useful exact/domain cues, especially ArguAna/Touche-style argumentative retrieval; files likely affected: `src/model.py`, `src/eval_mteb.py`, `src/experiments.py`; estimated GPU cost: low-medium evaluation only; risk level: low; preserves standard baseline fallback: yes via `a=0`.

- **Shallow-loop candidate policy**: mechanism: evaluate only pre-registered shallow loops or shallow-loop fusion, not per-task best loop selection; why improve NDCG@10: early refinement may capture benefits before late-loop drift; risk of hurting datasets: some tasks may need deeper refinement or no loop at all; files likely affected: `src/experiments.py`, `src/eval_mteb.py`, batch manifests, scoreboard candidate IDs; estimated GPU cost: low-medium; risk level: low; preserves standard baseline fallback: yes.

## Medium-risk ideas

- **Loop-loss weighting curriculum**: mechanism: replace uniform loopwise loss with fixed monotonic or warmup weights over loop depths; why improve NDCG@10: trains deeper representations without forcing every shallow loop to satisfy the same objective; risk of hurting datasets: over-optimizing final loops may weaken stable early-loop behavior; files likely affected: `src/losses.py`, `src/train.py`, `src/experiments.py`, configs; estimated GPU cost: medium, at least new training runs; risk level: medium; preserves standard baseline fallback: yes if registered as separate variants.

- **Denoised or ambiguous hard negatives**: mechanism: filter or weight negatives so training avoids too-easy negatives and likely false negatives; why improve NDCG@10: ANCE/RocketQA/SimANS-style work argues negative quality is central to dense retrieval transfer; risk of hurting datasets: RLHN-specific negative policy may overfit and reduce BEIR/MTEB transfer; files likely affected: `src/data.py`, `src/losses.py`, `src/train.py`, configs; estimated GPU cost: medium-high; risk level: medium; preserves standard baseline fallback: yes.

- **Conservative dense PRF query refinement**: mechanism: use top-k first-pass documents to form a second query embedding, but mix it back with the original/standard query; why improve NDCG@10: PRF can reduce short-query ambiguity and retrieve complementary relevant docs; risk of hurting datasets: classic query drift when top-k contains non-relevant docs; files likely affected: `src/eval_mteb.py`, `src/model.py`, possible cached corpus embedding utilities; estimated GPU cost: medium for full MTEB reruns; risk level: medium; preserves standard baseline fallback: yes if PRF is a separate candidate.

- **Two-stage reranking of standard+loop union**: mechanism: retrieve top-k union, rerank with a fixed cross-encoder/monoT5/late-interaction model; why improve NDCG@10: BEIR reports reranking/late interaction as strong zero-shot families, especially near top ranks; risk of hurting datasets: reranker domain bias and latency; files likely affected: new rerank module, `src/eval_mteb.py`, manifests, result collection; estimated GPU cost: medium-high; risk level: medium; preserves standard baseline fallback: yes if rerank candidate is opt-in.

## High-risk ideas

- **LLM/HyDE query expansion**: mechanism: generate hypothetical documents or rewritten queries, embed them, and fuse with standard retrieval; why improve NDCG@10: HyDE reports strong zero-shot dense retrieval gains by moving through a document-like embedding; risk of hurting datasets: hallucinated or over-specific expansions can damage exact/scientific tasks and add nondeterminism; files likely affected: new query expansion pipeline, `src/eval_mteb.py`, manifests, dependency/config policy; estimated GPU cost: high if local LLM, low-medium if API but not repo-reproducible; risk level: high; preserves standard baseline fallback: yes only with strict separate candidate and deterministic cache.

- **Test-time adaptive reranking/training**: mechanism: per-query adapt a small scoring function from pseudo-positive top docs and pseudo-negative bottom docs; why improve NDCG@10: recent DART-style work reports zero-resource dense reranking gains; risk of hurting datasets: pseudo-label errors can amplify initial mistakes, and protocol reproducibility becomes harder; files likely affected: new test-time adaptation module, `src/eval_mteb.py`, scoreboard validation; estimated GPU cost: medium-high; risk level: high; preserves standard baseline fallback: yes if adaptation is separate and can no-op.

- **Latent/iterative reasoning loop with generated rationales**: mechanism: use an LLM or learned controller to decompose/rewrite/refine queries across retrieval rounds; why improve NDCG@10: query-refinement/RAG literature suggests complex queries benefit from rewrite/decomposition; risk of hurting datasets: high variance, leakage-like prompt tuning risk, expensive and less aligned with current encoder-only protocol; files likely affected: new pipeline outside current `LoopMatryoshkaRetriever`, eval orchestration, manifests; estimated GPU cost: high; risk level: high; preserves standard baseline fallback: yes only as a separate non-final exploratory family.

- **Late-interaction retriever replacement**: mechanism: move from single-vector cosine to ColBERT-style token-level matching; why improve NDCG@10: late interaction keeps fine-grained term evidence and is strong on BEIR-style retrieval; risk of hurting datasets: major index/storage/runtime change, no longer directly tests current loop idea; files likely affected: `src/model.py`, `src/eval_mteb.py`, data/indexing, Slurm configs, dependencies; estimated GPU cost: high; risk level: high; preserves standard baseline fallback: yes only if kept as separate pipeline.

## Recommended first batch

1. **Evaluation-only standard-preserving fusion batch**: standard/loop score fusion, rank fusion, and shallow-loop fusion. Pre-register one small global grid on dev tasks only, then freeze candidate IDs before final tasks. No training.

2. **Anchored embedding interpolation**: test small `a` values against frozen standard and existing loop checkpoints. This is closest to the current code and has clean fallback.

3. **Only after fusion works**: one training batch for loop-loss weighting or denoised negatives. Keep it separate from final claims until it beats frozen standard on all final tasks.

## Sources or rationale

- Repo rationale: current implementation uses ModernBERT dual-encoder embeddings, query-only loop compute, hard-negative CE, MTEB retrieval evaluation, and `ndcg_at_10` as primary metric in `src/model.py`, `src/losses.py`, `src/eval_mteb.py`, and `experiments/batches/batch_template.yaml`.
- BEIR: heterogeneous retrieval benchmarks show dense retrievers can underperform robust baselines, while reranking and late-interaction methods are strong but costly. Source: https://arxiv.org/abs/2104.08663
- MTEB: no embedding method dominates every task, supporting standard-preserving and all-task validation rather than single-task tuning. Source: https://arxiv.org/abs/2210.07316
- Matryoshka Representation Learning: nested supervision can provide flexible representations with little inference overhead, motivating loop/depth supervision variants. Source: https://arxiv.org/abs/2205.13147
- ANCE/RocketQA/SimANS/GISTEmbed: negative selection quality strongly affects dense retrieval. Sources: https://arxiv.org/abs/2007.00808, https://arxiv.org/abs/2010.08191, https://arxiv.org/abs/2210.11773, https://arxiv.org/abs/2402.16829
- Dense PRF and ColBERT-PRF: PRF can improve dense query representations, but drift/noise is the key risk. Sources: https://arxiv.org/abs/2108.13454, https://arxiv.org/abs/2106.11251
- ColBERT and monoT5: late interaction and neural reranking improve top-rank quality at higher cost. Sources: https://arxiv.org/abs/2004.12832, https://arxiv.org/abs/2003.06713
- HyDE/query rewriting/test-time adaptation: useful for query-side compute but higher variance and protocol risk. Sources: https://arxiv.org/abs/2212.10496, https://arxiv.org/abs/2305.14283, https://arxiv.org/abs/2606.01070

## Uncertainty

- I did not inspect raw result values or run evaluation, so all priorities are research-direction estimates, not empirical claims about this repository.
- GPU costs are relative estimates for the current ModernBERT/MTEB setup, not measured cluster budgets.
- The safest near-term bet is evaluation-only fusion because it can preserve `standard` exactly; training-objective changes need new runs before any conclusion.
