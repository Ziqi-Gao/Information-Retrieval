# Repo Auditor Round 009

## Scope

Read-only audit before any `batch_016` design. No files edited, no Slurm actions, no training/eval, no secrets read.

## Latest Batch Readiness

`batch_015_dev` is terminal and scoreable.

Evidence:
- `postprocess_done.json`: completed `2026-06-27T21:56:51Z`
- `submission_plan.json`: 3 dev-only `standalone_main` candidates submitted through goal framework
- `scoreboard.json`: present and complete for the 4 dev tasks

Batch 015 result summary:
- `r015_role_prompt_standard__loop1`: mean `-0.0008725`, min `-0.00730`; wins NFCorpus, weak SCIDOCS; regresses SciFact and FiQA2018.
- `r015_dim_mrl_standard__loop1`: mean `-0.0024425`, min `-0.00966`; weak SCIDOCS only; regresses or misses margin elsewhere.
- `r015_role_dim_mrl_standard__loop1`: mean `+0.0021225`, min `-0.00320`; wins NFCorpus and SCIDOCS; regresses SciFact and FiQA2018.

No candidate has a viable global dev signal. All are `purpose: dev`, so none can trigger `main_goal_success`.

## Protocol Blockers Before Batch 016

- `outputs/goal/state.json` is stale for terminal interpretation: `last_dev_result` and `last_postprocess_check` still describe `batch_014_dev`, while `last_scoreboard` points to `batch_015_dev`.
- `phase` remains `SUBMIT_BATCH`, and `next_required_action` still says to collect/inspect results, although batch 015 postprocess and scoreboard already exist.
- `docs/agent_lab_notebook.md` ends with batch 015 submission and "wait for postprocess"; it needs a factual batch 015 result entry.
- Because batch 015 did not produce a strong global dev signal, a new `RESEARCH_DESIGN_MODE` note or notebook section is required before creating `batch_016`.

## State / Notebook Updates Needed

Update state facts before design:
- record batch 015 postprocess completion
- record batch 015 candidate deltas and failure pattern
- advance decision state from collection to decision/research design
- include `batch_015_dev` in local-search evidence if continuing

Update notebook:
- append "Batch 015 Dev Standalone Result"
- note that role prompting and dimensional MRL were dev-only standalone tests
- record that combined role+MRL helped NFCorpus/SCIDOCS but still regressed SciFact/FiQA2018

## Mechanism-Family Exhaustion

First-token loop-memory standalone search was already marked exhausted. Batch 015 tested mechanisms outside that family and still failed globally.

New exhaustion consideration:
- no-loop role prompting alone is not sufficient
- no-loop dimensional MRL alone is not sufficient
- no-loop role prompting plus dimensional MRL gives split-positive evidence, but not a coherent global signal

A batch 016 that only tweaks prefix strings, MRL dimensions, MRL weights, or nearby no-loop variants should be treated as another local-neighborhood sweep unless the new research design explicitly justifies it.

## Constraints For Protocol-Valid Batch 016

- Must be predeclared under `experiments/batches/` with unique `batch_id`.
- Use `primary_metric: ndcg_at_10` and frozen baseline paths.
- Keep `purpose: dev` unless explicitly planning final validation.
- Use explicit `candidate_track`; `standalone_main` must not use frozen-standard checkpoint, embeddings, scores, interpolation, weighted concat, or ensemble scoring.
- Predeclare global candidate rules and loop/candidate IDs. No post-hoc per-task selection.
- Prefer a diverse 2-4 candidate portfolio after research design, not a narrow local sweep.
- Stay within budget controls, currently max 24 GPU-hours per batch and safe concurrency.
- Submit only through `scripts/goal_submit_batch.py`; postprocess only through the goal framework.
- Keep generated outputs and site-specific Slurm/env details out of tracked docs/manifests.
