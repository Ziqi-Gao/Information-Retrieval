# Goal Preparation Handoff

Date: 2026-06-19

Branch: `codex-bert-only-loop-memory`

Scope: preparation framework only. No new retrieval model, scoring pipeline, expensive training, real Slurm submission, Git push, or PR was performed.

Preparation framework commit: `10bc97ca5ba7b046b5359853dc5ea43e9531ad34`

Frozen baseline commit: `c436b6d9dec2db81666eff414f91a84852996455`

`AGENTS.md` skip-worktree status was cleared before committing the framework.

Real Codex subagents were used in this preparation pass:

- `repo_auditor`
- `literature_scout`
- `experiment_planner`
- `code_risk_reviewer`

Subagent drafts are in `docs/codex_subagent_drafts/`, usage notes are in `docs/codex_subagents.md`, and reports are in `docs/subagent_reports/`.

## Implemented

- Added a goal-control protocol in `docs/goal_protocol.md`.
- Rewrote `AGENTS.md` with the later research goal, preparation scope, metric, final tasks, win rule, safety rules, Slurm rules, baseline rules, state rules, failure rules, and required workflow.
- Added `outputs/goal/state.json` with repo, metric, final tasks, frozen baseline status, budget, and next action.
- Added batch manifest docs and a harmless smoke/dry-run template under `experiments/batches/`.
- Added baseline freeze instructions under `outputs/baselines/README.md`.
- Added safe wrapper scripts:
  - `scripts/goal_common.py`
  - `scripts/goal_init.py`
  - `scripts/goal_freeze_baseline.py`
  - `scripts/goal_validate_manifest.py`
  - `scripts/goal_submit_batch.py`
  - `scripts/goal_status.py`
  - `scripts/goal_collect.py`
  - `scripts/goal_scoreboard.py`
  - `scripts/goal_preflight.py`
- Added local draft guardrails:
  - `docs/codex_hooks_draft.md`
  - `docs/codex_skill_drafts/hpc-slurm-orchestrator.md`
  - `docs/codex_skill_drafts/mteb-result-analyst.md`
  - `docs/codex_skill_drafts/ir-experiment-designer.md`
- Added project-specific subagent drafts:
  - `docs/codex_subagent_drafts/repo-auditor.toml`
  - `docs/codex_subagent_drafts/literature-scout.toml`
  - `docs/codex_subagent_drafts/experiment-planner.toml`
  - `docs/codex_subagent_drafts/result-analyst.toml`
  - `docs/codex_subagent_drafts/code-risk-reviewer.toml`
- Added subagent reports under `docs/subagent_reports/`.
- Added `docs/future_research_seed.md` with future experiment families only. Nothing from that document was implemented.
- Added `docs/agent_lab_notebook.md` as the running factual notebook.
- Hardened legacy wrappers so direct Slurm/local training paths fail safe by default.
- Hardened manifest validation, baseline validation, output path validation, `SBATCH_ARGS` validation, and final candidate pre-registration.
- Frozen the standard baseline from `outputs/final_grid_experiment/eval/summaries/standard_results_summary.csv`.

## Files Changed

Tracked or source-controlled paths created/updated:

- `AGENTS.md`
- `docs/goal_protocol.md`
- `docs/agent_lab_notebook.md`
- `docs/goal_prep_handoff.md`
- `docs/future_research_seed.md`
- `docs/codex_hooks_draft.md`
- `docs/codex_skill_drafts/hpc-slurm-orchestrator.md`
- `docs/codex_skill_drafts/mteb-result-analyst.md`
- `docs/codex_skill_drafts/ir-experiment-designer.md`
- `docs/codex_subagents.md`
- `docs/codex_subagent_drafts/repo-auditor.toml`
- `docs/codex_subagent_drafts/literature-scout.toml`
- `docs/codex_subagent_drafts/experiment-planner.toml`
- `docs/codex_subagent_drafts/result-analyst.toml`
- `docs/codex_subagent_drafts/code-risk-reviewer.toml`
- `docs/subagent_reports/repo_auditor_round_000.md`
- `docs/subagent_reports/literature_scout_round_000.md`
- `docs/subagent_reports/experiment_planner_round_000.md`
- `docs/subagent_reports/code_risk_reviewer_round_000.md`
- `docs/subagent_reports/round_000_summary.md`
- `experiments/batches/README.md`
- `experiments/batches/batch_template.yaml`
- `scripts/goal_common.py`
- `scripts/goal_init.py`
- `scripts/goal_freeze_baseline.py`
- `scripts/goal_validate_manifest.py`
- `scripts/goal_submit_batch.py`
- `scripts/goal_status.py`
- `scripts/goal_collect.py`
- `scripts/goal_scoreboard.py`
- `scripts/goal_preflight.py`
- `scripts/run_preexp.sh`
- `scripts/run_smoke.sh`
- `scripts/slurm_run_preexp.sh`
- `scripts/slurm_run_smoke.sh`
- `scripts/slurm_run_eval_all.sh`
- `src/run_all.py`
- `outputs/goal/state.json`
- `outputs/baselines/README.md`
- `outputs/baselines/standard_frozen/results_summary.csv`
- `outputs/baselines/standard_frozen/baseline_manifest.json`

Local ignored state/output paths created but not committed:

- `outputs/goal/runs/batch_001/batch_manifest.submitted.yaml`
- `outputs/goal/runs/batch_001/submission_plan.json`
- `outputs/goal/runs/batch_001/collected_results.csv`
- `outputs/goal/runs/batch_001/collected_results.json`
- `outputs/goal/runs/batch_001/per_run_validation.json`

## Validation Results

The login shell did not have a default `python` command, so the project Python environment was put first on `PATH` for the required `python ...` commands.

Commands run successfully before the framework commit:

```bash
python -m compileall src scripts
bash -n scripts/*.sh scripts/*.sbatch
python scripts/goal_init.py --state outputs/goal/state.json --max-concurrent-gpu-jobs 4 --max-gpu-hours-per-batch 24
python scripts/goal_validate_manifest.py experiments/batches/batch_template.yaml
python scripts/goal_submit_batch.py experiments/batches/batch_template.yaml --dry-run
python scripts/goal_scoreboard.py --self-test
python scripts/goal_preflight.py --manifest experiments/batches/batch_template.yaml
```

Important outputs:

- Guardrail self-tests passed.
- Manifest validation passed.
- Dry-run submission wrote `outputs/goal/runs/batch_001/submission_plan.json` and did not call `sbatch`.
- Scoreboard self-test passed.
- Preflight passed.
- `goal_status.py --json` reported dry-run train/eval entries with no Slurm job IDs.
- `goal_collect.py` on the dry-run batch produced one explicit `missing_result` row, as expected.
- `goal_freeze_baseline.py` was smoke-tested only against a temporary output directory, not the real baseline directory.
- `scripts/slurm_run_smoke.sh` refused direct legacy sbatch by default.
- `scripts/run_smoke.sh` refused login-node training by default.

Additional commands run successfully while freezing and validating the baseline:

```bash
python scripts/goal_freeze_baseline.py \
  --source-summary outputs/final_grid_experiment/eval/summaries/standard_results_summary.csv \
  --output-dir outputs/baselines/standard_frozen \
  --tasks SciFact,NFCorpus,SCIDOCS,FiQA2018,ArguAna,Touche2020,TRECCOVID \
  --metric ndcg_at_10
python scripts/goal_validate_manifest.py experiments/batches/batch_template.yaml --json
python scripts/goal_preflight.py --manifest experiments/batches/batch_template.yaml
```

The frozen CSV line endings were normalized from CRLF to LF before commit so `git diff --cached --check` would pass. Metric values were not edited.

## Baseline Status

The official frozen baseline is present and committed:

- Frozen summary: `outputs/baselines/standard_frozen/results_summary.csv`
- Manifest: `outputs/baselines/standard_frozen/baseline_manifest.json`
- SHA256: `41984b40e07ef7160444172fcaf895bcf74b42e44de5c8a700c3ac942cdbdee5`
- Primary metric: `ndcg_at_10`
- Win margin: `0.001`
- State phase: `READY_FOR_AUTONOMOUS_GOAL`

Final tasks:

- `SciFact`
- `NFCorpus`
- `SCIDOCS`
- `FiQA2018`
- `ArguAna`
- `Touche2020`
- `TRECCOVID`

Frozen standard NDCG@10 values:

- `SciFact`: `0.53932`
- `NFCorpus`: `0.23849`
- `SCIDOCS`: `0.13671`
- `FiQA2018`: `0.25855`
- `ArguAna`: `0.50874`
- `Touche2020`: `0.19917`
- `TRECCOVID`: `0.67118`

Future autonomous experiments can start only through validated manifests and `scripts/goal_submit_batch.py`.

## How To Run The Framework

Preflight:

```bash
python scripts/goal_preflight.py --manifest experiments/batches/batch_template.yaml
```

Create a new batch:

```bash
cp experiments/batches/batch_template.yaml experiments/batches/batch_002.yaml
```

Edit `batch_id`, `description`, `purpose`, `budget`, and `experiments`. Every experiment needs `run_id`, `hypothesis`, `version`, and `risk.reason`.

Validate:

```bash
python scripts/goal_validate_manifest.py experiments/batches/batch_002.yaml
```

Dry-run submit:

```bash
python scripts/goal_submit_batch.py experiments/batches/batch_002.yaml --dry-run
```

Real submit later:

```bash
python scripts/goal_submit_batch.py experiments/batches/batch_002.yaml --submit
```

Only do this after `budget.allow_submit: true`, and the dry-run plan is reviewed.

Check status:

```bash
python scripts/goal_status.py --state outputs/goal/state.json --update-state
```

Collect results:

```bash
python scripts/goal_collect.py \
  --batch-id batch_002 \
  --eval-root outputs/goal/eval \
  --output outputs/goal/runs/batch_002/collected_results.csv
```

Score results:

```bash
python scripts/goal_scoreboard.py \
  --baseline outputs/baselines/standard_frozen/results_summary.csv \
  --results outputs/goal/runs/batch_002/collected_results.csv \
  --metric ndcg_at_10 \
  --margin 0.001 \
  --output-csv outputs/goal/scoreboard.csv \
  --output-json outputs/goal/scoreboard.json
```

## Guardrail Stress Test

Common mistake checks:

- Direct `sbatch`: documented as forbidden; goal batches route through `goal_submit_batch.py`.
- Baseline missing: validator warns, submit wrapper refuses real non-smoke submit without a frozen baseline.
- Output overwrite: submit wrapper refuses non-empty run/eval directories unless `--resume`.
- Credential leak: submit wrapper constructs `--export=NONE,...` with a small safe variable whitelist.
- Missing results as wins: collector writes explicit `missing_result`; scoreboard requires valid rows for every final task.
- NaN or invalid metric: `metric_float` rejects NaN/inf/non-numeric values.
- Partial task coverage: collector and scoreboard mark missing task rows as failure.
- Final-task per-task best loop: collector uses loop-specific candidate IDs; scoreboard compares each candidate ID across all final tasks.
- Budget drift: manifest validator checks GPU concurrency and GPU-hour estimates against state limits unless explicitly over-budget.

Known limitations:

- Hooks are drafts only; no repo-local Codex hook convention was obvious.
- Most `outputs/goal/**` files are ignored local runtime state; only the small state JSON was force-added.
- `outputs/baselines/**` is ignored by default; only the small frozen baseline CSV and manifest were force-added.
- `goal_collect.py` validates CSV summaries; it does not inspect raw MTEB JSON.
- Slurm status depends on `squeue`/`sacct` availability and scheduler retention.
- `src/eval_mteb.py` metric parser was not changed in this pass to avoid altering NDCG@10 semantics.

## Next Recommended Prompt

Use this prompt when ready for the real autonomous research phase:

```text
Use the goal-control framework in this repository. Validate the frozen standard baseline, then design a small dev-only batch that cannot make final claims. Validate it, dry-run it, and ask before real Slurm submission. Do not change model code until the batch manifest and safety checks pass.
```
