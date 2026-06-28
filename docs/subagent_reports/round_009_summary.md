# Round 009 Subagent Summary

## Required Gate

`docs/codex_subagents.md` requires real read-only subagents before a new batch design. The parent remains the only writer.

## Subagents Used

- `repo_auditor`: `docs/subagent_reports/repo_auditor_round_009.md`
- `result_analyst`: `docs/subagent_reports/result_analyst_round_009.md`
- `literature_scout`: `docs/subagent_reports/literature_scout_round_009.md`
- `experiment_planner`: `docs/subagent_reports/experiment_planner_round_009.md`
- `code_risk_reviewer`: `docs/subagent_reports/code_risk_reviewer_round_009.md`

## High-Risk Findings

- `outputs/goal/state.json` and `docs/agent_lab_notebook.md` still needed factual `batch_015_dev` completion updates before any next design.
- `batch_015_dev` gave no viable global dev signal: all three standalone candidates regressed `SciFact` and `FiQA2018`; gains were limited to `NFCorpus` and `SCIDOCS`.
- First-token loop-memory remains exhausted, and narrow no-loop role-prefix or MRL-dimension sweeps should also be avoided.
- q/doc loop co-training is informative but higher risk for runtime/OOM; it should be reduced (`tmax=3`, `num_negatives=3`) if selected.
- Post-implementation code-risk review found no protocol blocker, but flagged q/doc runtime as high risk under the default Slurm train walltime.

## Resolution Plan

- Parent will update state and notebook with factual batch 015 results.
- Parent will enter `RESEARCH_DESIGN_MODE` in `docs/research_design_round_009.md`.
- Parent reduced q/doc training to `train_sample_size=12000` before validation/submission.
- Any new batch must be dev-only, standalone_main, candidate-only scoring, global rules across dev tasks, under 24 estimated GPU hours, and submitted only through `scripts/goal_submit_batch.py --submit --submit-postprocess`.
