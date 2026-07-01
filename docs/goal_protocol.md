# Autonomous Retrieval Goal Protocol

This protocol prepares future Codex sessions to run retrieval experiments safely. It does not authorize new model work by itself.

## Goal

The later research goal is to develop a retrieval pipeline that robustly beats the frozen `standard` baseline on every final retrieval dataset by `ndcg_at_10`.

Primary metric: `ndcg_at_10`

Weak diagnostic margin: `0.001`

Standalone research-grade threshold:

- every final task delta must be at least `+0.002`
- macro mean delta must be at least `+0.005`
- no final task may regress
- candidate track must be `standalone_main`

Publishable score-candidate threshold:

- every final task delta must be at least `+0.002`
- macro mean delta must be at least `+0.008`
- candidate track must be `standalone_main`
- bootstrap/significance evidence is required when query-level data is available
- if significance cannot be computed, label it `score-only, not statistically certified`

Final tasks:

- `SciFact`
- `NFCorpus`
- `SCIDOCS`
- `FiQA2018`
- `ArguAna`
- `Touche2020`
- `TRECCOVID`

Candidate claims are split into tracks:

- `standalone_main`: the candidate pipeline itself produces the candidate scores. It uses no frozen-standard checkpoint, standard embedding, standard score, weighted standard+candidate concatenation, or explicit ensemble with the frozen standard model except for baseline comparison.
- `fusion_diagnostic`: any candidate using `fusion_standard_checkpoint_dir`, `fusion_alpha`, `fusion_scope`, standard+loop weighted concat, standard embeddings/scores, or any explicit ensemble with the frozen standard model.
- `diagnostic`: non-final, exploratory, smoke, or otherwise non-main analyses.

`fusion_diagnostic` candidates may be evaluated and reported, but they must never trigger main goal success. Existing fusion batches, including `batch_003_final` and `batch_003_final_repair`, remain valid historical/diagnostic runs; if they pass numerically, label them `fusion_diagnostic_pass`, not `main_goal_success`.

`minimal_positive_signal` means every final task satisfies `candidate_ndcg_at_10 >= frozen_standard_ndcg_at_10 + 0.001`. It is a weak diagnostic label only and must not be called goal achieved.

## Allowed Actions

Future autonomous agents may:

- Audit repository state, current branch, dirty files, Slurm command availability, and Python environment.
- Initialize or resume `outputs/goal/state.json`.
- Freeze an existing standard baseline summary with `scripts/goal_freeze_baseline.py`.
- Design YAML experiment batches under `experiments/batches/`.
- Validate manifests with `scripts/goal_validate_manifest.py`.
- Dry-run Slurm submission plans with `scripts/goal_submit_batch.py --dry-run`.
- Submit Slurm jobs only through `scripts/goal_submit_batch.py --submit`.
- Submit deterministic Slurm-native postprocessing only through `scripts/goal_submit_batch.py --submit-postprocess`.
- Check queue and historical job state through `scripts/goal_status.py`.
- Collect evaluation summaries through `scripts/goal_collect.py`.
- Compare against a frozen baseline through `scripts/goal_scoreboard.py`.
- Update `docs/agent_lab_notebook.md` and handoff docs with factual status.

## Forbidden Actions

Future autonomous agents must not:

- Run full training directly on the login node.
- Call `sbatch` manually outside `scripts/goal_submit_batch.py`.
- Delete or overwrite `outputs/**`, `outputs/baselines/**`, `slurm_logs/**`, checkpoints, or raw MTEB files.
- Export API keys, tokens, SSH keys, or cloud credentials into Slurm jobs.
- Read `.env`, private keys, token caches, or credential files without explicit approval.
- Change `src/eval_mteb.py` metric semantics to improve reported results.
- Treat missing, failed, timed-out, duplicate, partial, or NaN results as wins.
- Select the best loop per final task after seeing final-task results unless the manifest marks the run as exploratory and excludes it from final claims.
- Push to GitHub or open pull requests unless explicitly asked.

## Baseline Freezing

The frozen baseline must be produced with:

```bash
python scripts/goal_freeze_baseline.py \
  --source-summary <existing_standard_results_summary.csv> \
  --output-dir outputs/baselines/standard_frozen \
  --tasks SciFact,NFCorpus,SCIDOCS,FiQA2018,ArguAna,Touche2020,TRECCOVID \
  --metric ndcg_at_10
```

The script validates:

- `version == standard`
- exactly one valid row per final task
- numeric, non-NaN `ndcg_at_10`
- final task list matches the protocol

It writes:

- `outputs/baselines/standard_frozen/results_summary.csv`
- `outputs/baselines/standard_frozen/baseline_manifest.json`

If a frozen baseline is missing, real autonomous experiments must not start. Smoke or dry-run checks may still run.

## Manifest Format

Batch manifests live under `experiments/batches/`. The template is `experiments/batches/batch_template.yaml`.

Important fields:

- `batch_id`: safe identifier for the batch.
- `purpose`: `smoke`, `dev`, or `final`.
- `primary_metric`: must be `ndcg_at_10`.
- `win_margin`: legacy/diagnostic margin, default `0.001`; main success thresholds are fixed by the protocol and are not relaxed by this field.
- `baseline`: paths to frozen baseline summary and manifest.
- `budget`: GPU concurrency, GPU-hour estimate, and `allow_submit`.
- `tasks.dev`: fast iteration tasks.
- `tasks.final`: full protocol tasks.
- `defaults`: config and output roots.
- `experiments`: pre-registered run entries with hypothesis and risk.
- optional experiment-level `claim_track` or `candidate_track`: one of `standalone_main`, `fusion_diagnostic`, or `diagnostic`. If omitted, the validator infers the track.

The validator rejects missing IDs, duplicate or unsafe run IDs, missing hypotheses, missing risk reasons, unknown tasks, unsupported metrics, output paths under `outputs/baselines`, over-budget manifests, unsafe submit conditions, and any attempt to mark a frozen-standard fusion/ensemble candidate as `standalone_main`.

Dev manifests may use `claim_track: standalone_main` for standalone-only exploration when candidate scoring avoids frozen-standard fusion. Such dev results still cannot trigger `main_goal_success`; that label requires `purpose: final`.

For final manifests, a `standalone_main` experiment must evaluate exactly the seven protocol final tasks, predeclare candidate loop indices or candidate IDs, and avoid frozen-standard scoring inputs except for the baseline comparison. A final fusion experiment remains valid as `fusion_diagnostic`, but it cannot support the main goal claim.

## Research Design Discipline

Autonomous agents must avoid getting trapped in narrow local searches. The goal is to maximize useful evidence per user resume cycle while preserving the existing acceptance thresholds, metric semantics, baseline, and claim-track separation.

### Local Search Exhaustion

A local search is exhausted when at least two recent `standalone_main` dev batches fail to produce a viable global dev signal, or when recent batches mainly change only local knobs such as:

- `loop_idx`
- neighboring eval depth around the same checkpoint
- memory mode variants without changing the underlying standalone mechanism
- re-evaluating the same failed checkpoint at nearby depths
- small hyperparameter perturbations around a failed candidate
- repeated narrow sweeps around the same failure pattern

A viable global dev signal means:

- all dev tasks are non-regressing, or nearly all are non-regressing with a clearly positive macro mean
- the candidate does not depend on frozen-standard fusion or standard-score interpolation
- the result is strong enough to justify either another focused dev test or final-validation planning under this protocol

If local search is exhausted, the agent must not submit another local-neighborhood sweep.

Examples of local-neighborhood sweeps include:

- only changing `loop_idx` around the same checkpoint
- only testing `t5/t6/t8/t9` after `t7` failed
- only changing one memory mode while keeping the same mechanism and failure pattern
- only retesting nearby depths of a checkpoint that already showed broad dev regressions

This rule does not ban focused follow-up. A local follow-up is allowed only when the latest dev result shows a strong, coherent global signal and the next local probe is clearly justified.

### RESEARCH_DESIGN_MODE

When local search is exhausted, the agent must enter `RESEARCH_DESIGN_MODE` before creating any new batch.

In `RESEARCH_DESIGN_MODE`, the agent must write a short research plan into `docs/agent_lab_notebook.md` or a dedicated `docs/research_design_round_<N>.md` file before creating or submitting a new manifest.

The plan must include:

- summary of what has been tried so far
- distinction between `fusion_diagnostic` results and `standalone_main` results
- failure pattern across recent `standalone_main` dev batches
- which dev tasks consistently regress
- why further local loop-depth or memory-mode sweeps are likely low-value
- 5 to 8 substantially different `standalone_main` research directions
- for each direction:
  - core mechanism hypothesis
  - code components likely affected
  - why it may address the observed failure pattern
  - expected risk
  - estimated GPU cost
  - smallest dev-only test that would falsify it
  - whether it requires training, evaluation-only changes, or both
- ranking of the directions by:
  - standalone-main validity
  - novelty relative to previous batches
  - likelihood of addressing regressions
  - implementation risk
  - GPU cost
  - clarity of falsification
- one chosen portfolio for the next dev batch, or a stop decision if no valid direction is available

`RESEARCH_DESIGN_MODE` must not:

- use final-task deltas to tune dev candidates
- lower acceptance thresholds
- convert `fusion_diagnostic` results into `main_goal_success`
- use frozen-standard embeddings, frozen-standard scores, weighted standard+candidate concatenation, or standard-score interpolation for `standalone_main` candidates
- submit jobs before the research plan is written
- create more than one next dev batch in a single resume cycle
- continue local loop-depth sweeps unless the plan explicitly justifies why local search is not exhausted

### Batch Efficiency And Portfolio Design

The user wants fewer manual resume cycles. Whenever the agent creates a new dev batch, it should maximize useful information per batch under the existing GPU, concurrency, and safety budgets.

Prefer portfolio batches over tiny single-candidate batches. A new dev batch should normally include multiple valuable experiments, not just one local candidate, unless:

- the experiment is unusually expensive
- a repair batch is needed
- the protocol requires a narrow validation step
- only one valid candidate can be designed safely
- adding more candidates would reduce scientific clarity

Use the available batch budget efficiently. When designing a dev batch, the agent should consider:

- `max_concurrent_gpu_jobs`
- `max_gpu_hours_per_batch`
- expected GPU hours per candidate
- whether jobs can run independently
- whether postprocess can score all candidates together

The agent should try to use the available batch capacity for informative candidates while staying safely within budget. Do not leave most of the batch budget unused if additional high-value, protocol-valid candidates are available.

Prefer diversity over redundant local sweeps. A batch should maximize expected information gain. Prefer candidates that test meaningfully different mechanisms or failure hypotheses.

Avoid filling a batch with many near-duplicates such as:

- only neighboring `loop_idx` values
- only tiny hyperparameter changes
- only variants of the same failed checkpoint
- only one memory mode with several adjacent depths

Such local sweeps are allowed only when explicitly justified by strong dev evidence and are not the dominant pattern after local search has been exhausted.

### Portfolio Design After RESEARCH_DESIGN_MODE

When `RESEARCH_DESIGN_MODE` is triggered, the agent should propose 5 to 8 substantially different `standalone_main` directions, then select a portfolio of candidates for the next dev batch.

The selected portfolio should usually include 2 to 4 candidates, depending on cost and risk. Candidates may come from different research directions if they can be tested safely in the same batch.

For each selected candidate, the agent must record:

- which research direction it tests
- why it is included in the same batch
- what failure mode it is meant to falsify
- estimated GPU cost
- expected information gain
- whether it is `standalone_main`, `diagnostic`, or `fusion_diagnostic`

Do not add low-value experiments just to fill the batch.

Every candidate in a portfolio batch must still:

- be predeclared
- have a clear hypothesis
- have a global candidate rule across dev tasks
- avoid final-task tuning
- avoid frozen-standard fusion if marked `standalone_main`
- stay within the configured GPU budget
- be auditable and reproducible

A resume cycle may submit at most one new batch, but that batch should be designed as an efficient portfolio when possible.

Before submitting a new dev batch, the agent must report:

- why this batch is the right size
- how many candidates are included
- why each candidate is valuable
- why the batch is not an uninformative local sweep
- estimated GPU hours versus allowed budget
- expected information gained if all candidates fail
- exact postprocess behavior

Do not submit a new batch until this portfolio rationale is written.

### Interaction With Running Batches

If a batch is currently pending or running, protocol updates and planning notes must not disturb it.

Future Codex sessions should first wait for the running batch's postprocess to complete, then analyze its scoreboard. If the result confirms continued local-search failure, `RESEARCH_DESIGN_MODE` should activate before any new batch is created.

Do not modify, resubmit, restart, cancel, or replace a running batch while making protocol-only updates.

## Slurm Submission

All Slurm submission must go through:

```bash
python scripts/goal_submit_batch.py experiments/batches/<batch>.yaml --dry-run
python scripts/goal_submit_batch.py experiments/batches/<batch>.yaml --submit
python scripts/goal_submit_batch.py experiments/batches/<batch>.yaml --submit --submit-postprocess
```

The wrapper:

- runs manifest validation first
- defaults to dry-run
- writes dry-run plans to `outputs/goal/runs/<batch_id>/dry_run_plan.json`
- refuses submit when `budget.allow_submit: false`
- refuses submit without a frozen baseline unless `purpose: smoke`
- refuses non-empty duplicate output directories unless `--resume` is passed
- writes submitted manifests to `outputs/goal/runs/<batch_id>/batch_manifest.submitted.yaml`
- writes submitted plans to `outputs/goal/runs/<batch_id>/submission_plan.json`
- calls existing `scripts/slurm_train.sbatch`
- calls existing `scripts/slurm_eval.sbatch` with `afterok` dependency
- optionally calls `scripts/slurm_postprocess.sbatch` with `afterany` dependency on all eval jobs
- exports only safe runtime and experiment variables

Base train/eval sbatch scripts and legacy Slurm launchers are guarded and refuse direct submission by default. They are not part of the autonomous goal workflow. Manual override requires an explicit `ALLOW_LEGACY_DIRECT_SBATCH=1` environment variable and must not be used for autonomous goal batches.

GPU budget controls are manifest-level guardrails. They are not a scheduler guarantee. The future agent must still review queue state and actual job outcomes.

## Slurm-Native Postprocessing

Long-running login-node processes are not reliable on every HPC. Some sites clean up `tmux`, `nohup`, and child processes when the SSH or VSCode session closes. Slurm jobs survive logout, so dependency-based postprocessing is the preferred automation path.

When `--submit-postprocess` is passed, `scripts/goal_submit_batch.py` submits one CPU-only postprocess job after all eval jobs are submitted:

```text
sbatch --parsable --dependency=afterany:<eval1>:<eval2>:... --export=NONE,<safe vars> scripts/slurm_postprocess.sbatch
```

The postprocess job:

- starts only after all eval jobs are terminal according to Slurm dependency handling
- runs `scripts/goal_status.py --update-state`
- runs `scripts/goal_collect.py`
- runs `scripts/goal_scoreboard.py`
- writes `outputs/goal/runs/<batch_id>/postprocess_done.json` on success
- writes `outputs/goal/runs/<batch_id>/postprocess_failed.json` and exits nonzero on collection or scoring failure
- does not train, evaluate MTEB, submit new jobs, overwrite baselines, or change metric semantics

The postprocess job uses `--export=NONE` plus a narrow allowlist for `BATCH_ID`, state/result paths, metric settings, `AUTO_CODEX`, and the Python/conda variables needed by `scripts/slurm_env.sh`. It must not export API keys, SSH keys, cloud credentials, or token variables.

`AUTO_CODEX` defaults to `false`. If set to `true`, the postprocess job may try `codex exec` after deterministic scoring and uses `outputs/goal/runs/<batch_id>/.codex_resume_launched` as a sentinel. If Codex auth or the `codex` command is unavailable on compute nodes, deterministic collection/scoring still stands; resume manually from the scoreboard and state files.

If eval submission succeeds but postprocess submission is rejected by the scheduler, do not resubmit eval. Use the project wrapper's postprocess-only repair path with the existing eval job id:

```bash
python scripts/goal_submit_batch.py experiments/batches/<batch>.yaml \
  --submit --submit-postprocess-only \
  --eval-job-id <run_id>=<eval_job_id>
```

Use scheduler options through `POSTPROCESS_SBATCH_ARGS` when the CPU postprocess job must use a different partition than GPU eval jobs.

## Resume After Queue Delays

After a pause or queue delay:

1. Read `outputs/goal/state.json`.
2. Read the active batch `submission_plan.json`.
3. Run `scripts/goal_status.py --state outputs/goal/state.json --update-state`.
4. If jobs are still pending or running, wait.
5. If a postprocess job was submitted, inspect its status and `postprocess_done.json` or `postprocess_failed.json`.
6. If no postprocess job was submitted and eval jobs are terminal, collect results manually through the goal scripts.

Do not infer completion from elapsed wall time alone.

## Batch Watcher

For long queue delays, a lightweight local watcher may poll the current batch without invoking an LLM while jobs are still pending or running. This is optional and less reliable than Slurm-native postprocessing on HPCs that clean up login-node processes after SSH or VSCode logout:

```bash
python scripts/goal_watch_batch.py \
  --state outputs/goal/state.json \
  --batch-id batch_001_dev \
  --interval-seconds 600 \
  --max-hours 12 \
  --mode notify
```

The watcher:

- refuses to run inside Slurm unless `--allow-inside-slurm` is explicit
- refuses to run without a frozen baseline, state file, submission plan, and submitted job IDs
- polls only through `scripts/goal_status.py --update-state --json`
- treats `completed`, `failed`, `cancelled`, `timeout`, `missing_result`, `invalid_metric`, `partial_tasks`, `failed_train`, and `failed_eval` as terminal
- treats `pending`, `running`, `unknown`, and `dry_run` as non-terminal
- writes `outputs/goal/runs/<batch_id>/watcher.log`
- writes `outputs/goal/runs/<batch_id>/watcher_status.json`
- never submits jobs, trains models, evaluates MTEB, collects results, scores results, or changes metric semantics by itself

Prefer `scripts/goal_submit_batch.py --submit --submit-postprocess` for deterministic collect/score automation that survives logout.

In `--mode notify`, the watcher prints the exact next status, collect, and scoreboard commands once all jobs are terminal.

In `--mode codex`, the watcher may invoke:

```bash
codex exec --sandbox workspace-write
```

only after all jobs are terminal. It uses a sentinel file, `outputs/goal/runs/<batch_id>/.codex_resume_launched`, so Codex is not launched repeatedly unless `--force-codex` is explicitly passed. The resumed Codex session must collect and score through the goal scripts and must not submit new jobs before summarizing the completed batch and preparing a new dry-run plan.

## Failure Classes

Use these classifications:

- `pending`: queued and not started.
- `running`: active job.
- `completed`: Slurm job completed and required result rows validate.
- `failed_train`: training failed.
- `failed_eval`: evaluation failed.
- `missing_result`: expected summary or task row is absent.
- `invalid_metric`: metric is missing, NaN, non-numeric, duplicate, or ambiguous.
- `partial_tasks`: not all expected tasks are present.
- `timeout`: scheduler reported timeout.

Only `completed` rows with valid metrics can be scored.

## Result Collection

Use:

```bash
python scripts/goal_collect.py \
  --batch-id <batch_id> \
  --eval-root outputs/goal/eval \
  --output outputs/goal/runs/<batch_id>/collected_results.csv
```

Collection writes:

- `collected_results.csv`
- `collected_results.json`
- `per_run_validation.json`

Collection does not compare to baseline. Missing or invalid results remain explicit failure rows.

## Scoreboard

Use:

```bash
python scripts/goal_scoreboard.py \
  --baseline outputs/baselines/standard_frozen/results_summary.csv \
  --results outputs/goal/runs/<batch_id>/collected_results.csv \
  --metric ndcg_at_10 \
  --margin 0.001 \
  --output-csv outputs/goal/scoreboard.csv \
  --output-json outputs/goal/scoreboard.json
```

The scoreboard validates one frozen standard row per final task and then scores each candidate ID. It reports multiple labels:

- `minimal_positive_signal`: every final task is valid and every delta is at least `+0.001`.
- `fusion_diagnostic_pass`: `minimal_positive_signal` is true for a `fusion_diagnostic` candidate.
- `research_grade_threshold_pass`: every final task is valid, every delta is at least `+0.002`, macro mean delta is at least `+0.005`, and there is no regression.
- `main_goal_success`: `research_grade_threshold_pass` is true and the candidate is a `standalone_main` final candidate.
- `publishable_score_candidate`: `standalone_main`, every final-task delta at least `+0.002`, and macro mean delta at least `+0.008`; certification remains `score-only, not statistically certified` unless significance evidence is available.

`pass_all_tasks` is retained only as a legacy compatibility alias for `minimal_positive_signal`. It must not be interpreted as main goal success.

Candidates are sorted by:

1. `main_goal_success` descending
2. `publishable_score_candidate` descending
3. `research_grade_threshold_pass` descending
4. `fusion_diagnostic_pass` descending
5. `minimal_positive_signal` descending
6. `min_delta` descending
7. `mean_delta` descending

## Dev Tasks Vs Final Tasks

Dev subsets are for fast iteration and debugging. They may guide hypotheses, but they cannot support a final claim.

Final claims require:

- a frozen standard baseline
- full final task coverage
- pre-registered candidate identity
- predeclared final loop indices or candidate IDs in the manifest
- no missing or invalid metrics
- `candidate_track == standalone_main`
- no frozen-standard checkpoint, standard embedding, standard score, standard+candidate weighted concatenation, or explicit frozen-standard ensemble in candidate scoring
- every final task delta at least `+0.002`
- macro mean delta at least `+0.005`
- no task regression
- no post-hoc per-task loop selection

## Why Per-Task Best Loop Is Invalid

Loop-depth curves are useful for analysis, but choosing the best loop separately for each final task after seeing final results leaks test feedback into the method definition. That creates an exploratory result, not a valid final claim. A final claim must score a pre-defined candidate such as `run_id__loop3` across all final tasks, or another explicitly pre-registered aggregation rule.
