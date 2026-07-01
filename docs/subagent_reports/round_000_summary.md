# Round 000 Subagent Summary

Real Codex subagents were used in parallel for this round:

- `repo_auditor`
- `literature_scout`
- `experiment_planner`

All subagents were instructed to be read-only. They did not edit files, submit Slurm jobs, run training, run MTEB evaluation, overwrite outputs, or make final pass/fail claims.

## repo_auditor Key Findings

- The goal-control flow is organized around `goal_init`, `goal_freeze_baseline`, `goal_validate_manifest`, `goal_submit_batch`, `goal_status`, `goal_collect`, and `goal_scoreboard`.
- `baseline.status` is currently `missing`; the dry-run batch has no real Slurm job IDs and no real results.
- NDCG@10 is parsed and written in `src/eval_mteb.py`, then collected and scored by the goal scripts.
- Main risks identified:
  - `budget.max_concurrent_gpu_jobs` is validated but not enforced.
  - `SBATCH_ARGS` is unrestricted.
  - legacy Slurm launchers still call `sbatch` directly and use `--export=ALL`.
  - final task mismatch is only a warning.
  - baseline existence check does not validate manifest/hash/task coverage.
  - partial submit failure could lose already-submitted job IDs.
  - candidate IDs are generated from observed loop outputs rather than manifest-declared loop candidates.

## literature_scout Top Future Ideas

High-priority low-risk families:

- Standard/loop score fusion.
- Rank-level fusion or reciprocal-rank-style fusion across standard and fixed loop candidates.
- Standard-anchored query embedding interpolation.
- Shallow-loop candidate policy with pre-registered loop IDs.

Medium-risk families:

- Loop-loss weighting curriculum.
- Denoised or ambiguity-aware hard-negative weighting.
- Conservative dense PRF query refinement.
- Two-stage reranking of standard plus loop union.

Parent decision: keep these ideas as future research seeds only. No model, scoring, or evaluation pipeline changes are part of this preparation step.

## experiment_planner Recommended Framework

- Manifest must declare batch ID, purpose, metric, margin, baseline paths, budget, tasks, defaults, and experiments.
- State must remain the resume entry point, with current batch, open jobs, baseline status, budget, scoreboard, and next required action.
- Dev tasks are for screening only; final claims require all final tasks.
- Candidate identity must be pre-registered enough to prevent per-task final-result cherry-picking.
- First future dev batch should remain conservative, standard-preserving, low concurrency, and dry-run first.

## Conflicts Between Reports

- The planner accepts manifest-level `max_concurrent_gpu_jobs` as a budget guardrail; the auditor correctly notes it is not currently enforced as a submission throttle. Parent decision: document this as a known limitation and tighten implementation where cheap, but avoid adding complex Slurm throttling in this preparation pass unless required by code-risk review.
- The literature scout recommends evaluation-only fusion as first future work; the current preparation framework intentionally does not implement fusion. Parent decision: leave fusion in `docs/future_research_seed.md`.
- The auditor recommends deprecating or changing legacy Slurm launchers. Parent decision: do not destructively change legacy scripts in this pass; document that future goal batches must use `scripts/goal_submit_batch.py`.

## Parent Decision

Keep the existing preparation framework, then harden it before final handoff by focusing on:

- stricter final-task validation for `purpose: final`
- better baseline validation in manifest checks
- safer `SBATCH_ARGS` handling
- clearer documentation that legacy launchers are outside the autonomous-goal path
- post-implementation risk review by `code_risk_reviewer`

## Implementation Plan

1. Save subagent drafts under `docs/codex_subagent_drafts/`.
2. Save reports under `docs/subagent_reports/`.
3. Run `code_risk_reviewer` after this stable parent-written state.
4. Fix blocker/high findings.
5. Re-run cheap checks:
   - `python -m compileall src scripts`
   - `bash -n scripts/*.sh scripts/*.sbatch`
   - `python scripts/goal_validate_manifest.py experiments/batches/batch_template.yaml`
   - `python scripts/goal_submit_batch.py experiments/batches/batch_template.yaml --dry-run`
   - `python scripts/goal_scoreboard.py --self-test`
   - `python scripts/goal_preflight.py --manifest experiments/batches/batch_template.yaml`
6. Update `docs/goal_prep_handoff.md` with the subagent outcome.

## Post-Review Parent Actions

`code_risk_reviewer` ran after Phase A and reported two blockers plus high-severity issues. Parent fixes applied:

- Legacy direct Slurm launchers now refuse by default unless `ALLOW_LEGACY_DIRECT_SBATCH=1` is set.
- Legacy Slurm launchers no longer use `--export=ALL`; remaining direct legacy operation uses explicit `--export=NONE,...`.
- Local training launchers and `src.run_all` now refuse login-node training unless inside Slurm or `ALLOW_LOGIN_NODE_TRAINING=1` is explicitly set.
- `goal_validate_manifest.py` now requires goal output roots under `outputs/goal/runs` and `outputs/goal/eval`.
- `goal_validate_manifest.py` now validates frozen baseline manifest/hash/task coverage rather than checking only file existence.
- `goal_validate_manifest.py` now rejects manifests whose experiment count exceeds `max_concurrent_gpu_jobs`.
- `goal_submit_batch.py` now validates `SBATCH_ARGS` against an allowlist and rejects `--export`, `--wrap`, `--array`, and `--dependency`.
- Final manifests now require exact final-task coverage and predeclared `eval.candidate_loop_indices`.
- `goal_collect.py` now uses declared loop candidates for final batches and rejects unexpected task/loop candidates.
- `goal_preflight.py` now includes guardrail self-tests for path traversal, concurrency, final task subset rejection, and baseline hash validation.

Deferred medium/low findings:

- `src/eval_mteb.py` recursive metric parsing was not changed because this preparation task must not modify NDCG@10 metric semantics unless necessary.
- More comprehensive collection/status integration is deferred; current collector still marks absent summaries explicitly as `missing_result`.
- Legacy shell scripts do not have full `--help`; they now fail safely by default.
