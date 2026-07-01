# Round 012 Subagent Summary

## Context

User approval was received to prepare and submit final validation for the `batch_017_dev_repair` strong dev candidate. The target rule is the exact standalone candidate `r017_seeded_lexical_hash__loop1`.

## Required Gates

`docs/codex_subagents.md` requires real subagents before designing, validating, dry-running, preflighting, or submitting a new batch. Phase A reports were completed before creating the final manifest:

- `repo_auditor`: `docs/subagent_reports/repo_auditor_round_012.md`
- `literature_scout`: `docs/subagent_reports/literature_scout_round_012.md`
- `experiment_planner`: `docs/subagent_reports/experiment_planner_round_012.md`

## Findings

- Final validation is protocol-valid only after explicit user approval, which was provided.
- `batch_017_dev_repair` is a strong dev signal, not a final claim.
- The final manifest must preserve the exact candidate rule: `standard_seeded_sampling`, checkpoint `outputs/goal/runs/batch_016_dev/r016_standard_seeded_sampling/final`, `loop_idx=1`, `candidate_loop_indices=[1]`, `lexical_hash_dim=1024`, `lexical_weight=0.15`.
- The final manifest must cover all seven protocol final tasks and remain `candidate_track: standalone_main`.
- No frozen-standard fusion, baseline score input, or standard embedding input is allowed.

## Blockers / High Risks

- No Phase A blocker was found.
- Phase C code-risk review found no blocker or high-severity protocol risk.
- Main residual risk is scientific: held-out final tasks `ArguAna`, `Touche2020`, and `TRECCOVID` may regress.
- Any missing, failed, NaN, duplicate, partial, or below-threshold final-task result invalidates main goal success.

## Phase C Report

- `code_risk_reviewer`: `docs/subagent_reports/code_risk_reviewer_round_012.md`

## Gate Status

Workflow gates passed before validation/dry-run/preflight/submission.
