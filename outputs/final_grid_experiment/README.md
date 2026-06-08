# Final Grid Experiment Outputs

This directory contains the cleaned final grid experiment outputs only. Training logs remain in their original `outputs/preexp/<method>/train_log.jsonl` locations, and Slurm logs remain in `slurm_logs/`.

## Structure

- `checkpoints/<method>/`: final checkpoint for each of the 7 registered methods.
- `eval/raw/<method>/<task>/`: raw and parsed MTEB loop outputs.
- `eval/summaries/`: canonical CSV summaries generated from `eval/raw`.
- `plots/`: aggregate nDCG@10 heatmaps.

## Coverage

- Available metric rows: 426
- Expected metric rows: 427
- Missing rows: 1
  - `loop_final_recurrent_no_memory` / `TRECCOVID` / loop `10`
