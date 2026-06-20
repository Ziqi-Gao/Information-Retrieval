# Goal Batch Watcher

`scripts/goal_watch_batch.py` is a lightweight local watcher for submitted goal batches. It polls Slurm through `scripts/goal_status.py` and stays silent from an LLM perspective until the batch reaches terminal state.

It is infrastructure only. It does not submit Slurm jobs, run training, run MTEB evaluation, collect results, score results, modify the frozen baseline, or change metric semantics.

## Notify Mode

Use notify mode when you want the watcher to print the next commands after all jobs are terminal:

```bash
python scripts/goal_watch_batch.py \
  --state outputs/goal/state.json \
  --batch-id batch_001_dev \
  --interval-seconds 600 \
  --max-hours 12 \
  --mode notify
```

When the batch is terminal, run the printed commands to refresh status, collect results, and score against the frozen baseline.

## Codex Mode

Use codex mode when you want the watcher to start a non-interactive Codex resume after the batch is terminal:

```bash
python scripts/goal_watch_batch.py \
  --state outputs/goal/state.json \
  --batch-id batch_001_dev \
  --interval-seconds 600 \
  --max-hours 12 \
  --mode codex \
  --codex-output outputs/goal/runs/batch_001_dev/codex_resume_after_terminal.md
```

With `--json`, Codex JSONL is written to:

```text
outputs/goal/runs/<batch_id>/codex_resume_after_terminal.jsonl
```

The final Markdown message is written to the `--codex-output` path, or to:

```text
outputs/goal/runs/<batch_id>/codex_resume_after_terminal.md
```

The watcher uses this sentinel to avoid launching Codex more than once:

```text
outputs/goal/runs/<batch_id>/.codex_resume_launched
```

Pass `--force-codex` only when you intentionally want to launch Codex again after inspecting the prior output.

## Safety Checks

The watcher refuses to run when:

- it is inside a Slurm job and `--allow-inside-slurm` is not passed
- `outputs/goal/state.json` is missing
- the baseline is not frozen
- the batch submission plan is missing
- the submission plan has no submitted job IDs
- the state file has no open jobs

Terminal statuses are:

```text
completed, failed, cancelled, timeout, missing_result, invalid_metric, partial_tasks, failed_train, failed_eval
```

Non-terminal statuses are:

```text
pending, running, unknown, dry_run
```

Failed, cancelled, and timed-out jobs are terminal, but they are not wins.

## Outputs

The watcher appends a local log:

```text
outputs/goal/runs/<batch_id>/watcher.log
```

It writes machine-readable final watcher state:

```text
outputs/goal/runs/<batch_id>/watcher_status.json
```

These files are local runtime artifacts unless the user explicitly asks to commit them.
