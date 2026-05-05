# Adding Ablations

Keep new ablations small and explicit. The experiment code is organized around a central version registry:

```text
src/experiments.py
```

Add a new `VersionSpec` there first. Choose:

- `family="standard"` for no-loop models that should use the standard query path and one SciFact evaluation row.
- `family="loop"` for memory-token loop models that should unroll query loops and evaluate all loop depths.
- `plot_kind="baseline"` for horizontal baselines.
- `plot_kind="curve"` for loop-depth curves.

Training behavior is selected in `src/train.py` from the version family. Evaluation behavior is selected in
`src/eval_mteb.py` from the same registry. Plot colors and baseline/curve behavior are also read from the registry by
`src/plot_results.py`.

For a one-off ablation script, follow `scripts/run_standard_more_steps.sh`:

1. Train into `outputs/<method>/`.
2. Rename the final checkpoint to `outputs/<method>/model/`.
3. Evaluate into `outputs/<method>/eval_tmp/`.
4. Run `python -m src.finalize_ablation --method <method> --run_dir outputs/<method>`.
5. Run `python -m src.plot_results --summary_csv outputs/plots/results_summary_all.csv --output_dir outputs/plots`.

Do not add online tuning, exit gates, dimension slicing, pseudo labels, or in-batch loss unless the ablation is explicitly
about one of those features. Keep each ablation isolated under its own `outputs/<method>/` directory.

The memory-history range ablations add last-state-only and no-feedback variants:

- `loop_final_last`
- `loop_final_none`
- `loop_matryoshka_last`
- `loop_matryoshka_none`

Run them with:

```bash
bash scripts/run_no_history_ablation.sh
```

On SLURM:

```bash
bash scripts/slurm_run_no_history_ablation.sh
```

They keep the loop training objectives unchanged and vary only `memory_history_mode`:

- `full`: `h_t` receives the full prefix `h_1...h_{t-1}`.
- `last`: `h_t` receives only the immediately preceding state `h_{t-1}`.
- `none`: `h_t` receives no recurrent memory-state feedback.
