# mteb-result-analyst

Use this skill when interpreting MTEB retrieval results in this repository.

Rules:

- Primary metric is `ndcg_at_10`.
- Final tasks are `SciFact`, `NFCorpus`, `SCIDOCS`, `FiQA2018`, `ArguAna`, `Touche2020`, and `TRECCOVID`.
- Compare only against the frozen `standard` baseline under `outputs/baselines/standard_frozen/`.
- Missing, failed, timed-out, duplicate, NaN, or partial-task results are failures.
- Do not modify NDCG@10 extraction semantics.
- Do not select the best loop per final task after seeing final-task results.
- Use `scripts/goal_scoreboard.py` for pass/fail claims.
