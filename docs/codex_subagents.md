# Codex Subagents For This Repository

This repository currently does not have an obvious project-local Codex subagent configuration directory.

The preferred path from the requested design was:

```text
.codex/agents/*.toml
```

In this checkout, `.codex` is a regular file, not a directory. Replacing it with a directory would be a repo-structure change with unclear Codex support, so the subagent definitions are stored as drafts instead:

```text
docs/codex_subagent_drafts/
```

Use these drafts as the source of truth if a future Codex installation adds a documented project-local subagent mechanism.

## Draft Agents

- `docs/codex_subagent_drafts/repo-auditor.toml`
- `docs/codex_subagent_drafts/literature-scout.toml`
- `docs/codex_subagent_drafts/experiment-planner.toml`
- `docs/codex_subagent_drafts/result-analyst.toml`
- `docs/codex_subagent_drafts/code-risk-reviewer.toml`

## Global Rules

All subagents are read-only by default.

They must not:

- edit files
- call `sbatch`
- submit, cancel, or modify Slurm jobs
- wait on queues
- run training
- run MTEB evaluation
- overwrite outputs
- read secrets, `.env`, token files, SSH keys, or credential caches
- export credentials
- fabricate results
- make final pass/fail claims from partial data

The parent agent is the only writer and the only agent allowed to update manifests, scripts, docs, `AGENTS.md`, or `outputs/goal/state.json`.

## Workflow

Phase A, pre-implementation reading:

1. Spawn `repo_auditor`, `literature_scout`, and `experiment_planner` in parallel.
2. Save their reports under `docs/subagent_reports/`.
3. Parent writes `docs/subagent_reports/round_000_summary.md`.

Phase B, parent implementation:

1. Parent implements or updates the goal-control framework.
2. Parent runs cheap checks only.
3. No real Slurm submission is allowed during preparation.

Phase C, post-implementation review:

1. Spawn `code_risk_reviewer` after the diff is stable.
2. Spawn `result_analyst` only after real results and a frozen baseline exist.
3. Parent fixes all blocker/high findings before declaring readiness.

## Current Session Note

Real Codex subagents were used for report generation in this session through the available `spawn_agent` tool. The draft TOML files remain documentation because this checkout does not expose a safe `.codex/agents/` directory.
