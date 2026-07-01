# hpc-slurm-orchestrator

Use this skill for Slurm submission, monitoring, and collection in this repository.

Rules:

- Submit only through `scripts/goal_submit_batch.py`.
- Check status only through `scripts/goal_status.py`.
- Collect results only through `scripts/goal_collect.py`.
- Never call `sbatch` manually.
- Never export tokens or broad environment dumps into Slurm jobs.
- Respect manifest `max_concurrent_gpu_jobs`, `max_gpu_hours_estimate`, and `allow_submit`.
- Treat timeout, cancellation, missing summaries, and partial task coverage as failures.
- Update `outputs/goal/state.json` and `docs/agent_lab_notebook.md` after material state changes.
