# ir-experiment-designer

Use this skill when proposing future retrieval experiments for the later research goal.

Required fields for every proposed experiment:

- hypothesis
- mechanism
- code areas likely affected
- expected GPU cost
- risk level
- risk reason
- dev tasks
- final-claim eligibility

Design rules:

- Preserve the frozen `standard` baseline.
- Prefer controlled ablations and standard-preserving changes first.
- Separate dev-task iteration from final-task claims.
- Do not add trainable projection heads, memory projections, gates, or learned scaling unless explicitly requested.
- Register versions in `src/experiments.py` before training.
- Validate manifests before any Slurm submission.
