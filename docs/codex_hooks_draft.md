# Codex Hook Drafts

No obvious repo-local Codex hook convention was found in this workspace. These are draft guardrails to implement only if the local Codex installation supports hooks.

Desired hook behavior:

- Block direct `sbatch` unless the command is launched by `scripts/goal_submit_batch.py`.
- Block `rm -rf outputs/baselines`.
- Block `git reset --hard`.
- Block `git clean -fdx`.
- Block direct full training on the login node, especially `python -m src.train --config configs/preexp.yaml`.
- Warn when ending a long experiment-control session without updating `outputs/goal/state.json` or `docs/agent_lab_notebook.md`.

Hooks are only defense in depth. The Python scripts must remain the primary enforcement layer.
