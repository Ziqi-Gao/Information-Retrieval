# Subagent Round 010 Summary

Date: 2026-06-28

Required workflow gate from `docs/codex_subagents.md` was used before any new batch design. Real read-only subagents were invoked through the available `spawn_agent` tool:

- `repo_auditor`: `docs/subagent_reports/repo_auditor_round_010.md`
- `literature_scout`: `docs/subagent_reports/literature_scout_round_010.md`
- `experiment_planner`: `docs/subagent_reports/experiment_planner_round_010.md`
- `result_analyst`: `docs/subagent_reports/result_analyst_round_010.md`
- `code_risk_reviewer`: `docs/subagent_reports/code_risk_reviewer_round_010.md`

## Gate Findings

- `batch_016_dev` completed postprocess and has deterministic scoreboard evidence.
- `main_goal_success=false`.
- `r016_standard_seeded_sampling__loop1` is the only weak positive observed-dev result, but it is below the weak diagnostic margin on `NFCorpus` and `FiQA2018`, below final single-task threshold on those same tasks, and below final macro threshold.
- No final-validation dry-run plan is justified.
- `outputs/goal/state.json` was stale and needed factual `batch_016_dev` updates.
- Any further batch must enter `RESEARCH_DESIGN_MODE` and avoid local sweeps of the failed q/doc loop, seeded-sampling, or in-batch objective mechanisms.
- Code risk review initially found one manifest-validation blocker caused by false-positive standalone text, plus medium risks around result-row dedupe and fusion+new-eval parameter semantics.

## Resolution Plan

The parent will update the notebook and goal state with `batch_016_dev` facts, enter `RESEARCH_DESIGN_MODE`, and only create a new dev batch if it tests broader standalone mechanisms outside the exhausted families.

## Gate Resolution

- The manifest text blocker was resolved; `goal_validate_manifest.py` now passes.
- `results_summary.csv` dedupe now includes `doc_chunk_*` and `lexical_*`.
- The evaluator and manifest validator now forbid combining `doc_chunk_*` or `lexical_*` with frozen-standard fusion.
- The runtime risk is recorded in manifest risk text and constrained to a two-candidate eval-only dev batch.
