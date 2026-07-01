# Literature Scout Round 010

Date: 2026-06-28

Read-only local-document review. No browsing, no file edits, no Slurm actions, and no secrets access.

## Findings

Local docs before this parent update had not yet recorded completed `batch_016_dev` results, so `r016_standard_seeded_sampling` should not be promoted based on notebook evidence alone.

Mechanism families that have been tried or are exhausted for now:

- first-token loop memory: loop depth, memory mode, detached memory, short horizons, single-loop, and token-concat variants;
- loop objective local variants: final-only, tail weighting, consistency, sparse/late supervision, and label smoothing;
- hard-negative/data-order variants inside first-token loops: lower negatives, middle-window negatives, and first-token seeded sampling;
- candidate-only stabilization/objective variants: self-residual, in-batch hybrid, and pairwise ranking;
- no-loop low-risk pivots: role prompting, dimensional MRL, and role+MRL;
- fusion diagnostics: frozen-standard fusion remains diagnostic only and cannot trigger `standalone_main`.

`batch_016_dev` should be treated as insufficient for final validation unless the parent records scoreboard evidence and separately justifies a stronger signal. The completed scoreboard shows the signal remains weak, so final validation is not recommended.
