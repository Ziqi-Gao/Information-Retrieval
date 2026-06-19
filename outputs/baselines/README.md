# Frozen Baselines

This directory is for local frozen baseline artifacts. Do not edit baseline files by hand.

Freeze an existing standard summary with:

```bash
python scripts/goal_freeze_baseline.py \
  --source-summary outputs/final_grid_experiment/eval/summaries/standard_results_summary.csv \
  --output-dir outputs/baselines/standard_frozen \
  --tasks SciFact,NFCorpus,SCIDOCS,FiQA2018,ArguAna,Touche2020,TRECCOVID \
  --metric ndcg_at_10
```

Expected outputs:

- `outputs/baselines/standard_frozen/results_summary.csv`
- `outputs/baselines/standard_frozen/baseline_manifest.json`

Future autonomous experiments should not start until the frozen baseline exists and validates.
