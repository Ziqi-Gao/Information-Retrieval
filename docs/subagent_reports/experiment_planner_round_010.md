# Experiment Planner Round 010

Date: 2026-06-28

Read-only planning review after `batch_016_dev`.

## Recommendation

Do not create a final-validation dry-run plan. The best candidate, `r016_standard_seeded_sampling__loop1`, has dev deltas:

- `SciFact`: `+0.00834`
- `NFCorpus`: `+0.00045`
- `SCIDOCS`: `+0.00386`
- `FiQA2018`: `+0.00086`

It is non-regressing on the four observed dev tasks, but only two of four tasks reach the final single-task threshold `+0.002`, the mean delta is `+0.00338`, and `NFCorpus` plus `FiQA2018` are below the weak `+0.001` diagnostic margin. It is not strong enough for final validation.

## Next Protocol-Valid Action

First record `batch_016_dev` and synchronize state. Then either stop or enter a new `RESEARCH_DESIGN_MODE`. Any new dev batch must be broader than local variants of q/doc loops, seeded sampling, or in-batch weighting.

## Blockers

- `outputs/goal/state.json` is stale and must be updated before further workflow claims.
- `r016_qdoc_final_mean_pool_t3_neg3` and `r016_standard_inbatch_hybrid` cannot be final candidates.
- Any final submission would require explicit user approval and must go through `scripts/goal_submit_batch.py`, but current evidence does not justify that step.
