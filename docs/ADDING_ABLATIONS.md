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

The current final registry intentionally keeps only:

- `standard`
- `loop_final`
- `loop_matryoshka`

All trainable parameters must live under `model.encoder`. Do not add trainable projection heads, memory adapters, state
embeddings, gates, or learned scaling unless the ablation is explicitly about adding new trainable parameters.

For a one-off ablation:

1. Train into `outputs/<method>/`.
2. Rename the final checkpoint to `outputs/<method>/model/`.
3. Evaluate into `outputs/<method>/eval_tmp/`.
4. Run `python -m src.finalize_ablation --method <method> --run_dir outputs/<method>`.
5. Run `python -m src.plot_results --summary_csv outputs/plots/results_summary_all.csv --output_dir outputs/plots`.

Do not add online tuning, exit gates, dimension slicing, pseudo labels, or in-batch loss unless the ablation is explicitly
about one of those features. Keep each ablation isolated under its own `outputs/<method>/` directory.

Parameter-free loop memory modes are configured with `loop_memory_mode`:

- `first_token`: prepend the previous loop's first query-token hidden state.
- `mean_pool`: prepend the previous loop's mean-pooled query hidden state.
- `token_concat`: prepend all previous-loop query-token hidden states.
