# Goal Batch Manifests

Batch manifests describe autonomous retrieval experiments before any Slurm submission.

Use the template:

```bash
cp experiments/batches/batch_template.yaml experiments/batches/batch_002.yaml
```

Then edit:

- `batch_id`
- `description`
- `purpose`
- `budget`
- `experiments`

Validate before submission:

```bash
python scripts/goal_validate_manifest.py experiments/batches/batch_002.yaml
```

Dry-run submission:

```bash
python scripts/goal_submit_batch.py experiments/batches/batch_002.yaml --dry-run
```

Real submission is allowed only when:

- the baseline is frozen
- `budget.allow_submit: true`
- the batch passed validation
- output directories are new or `--resume` is intentional

```bash
python scripts/goal_submit_batch.py experiments/batches/batch_002.yaml --submit
```

Do not put batch outputs under `outputs/baselines/`. Do not use final-task results for post-hoc per-task loop selection.
