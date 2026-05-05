# Loop-Wise Matryoshka Retrieval

This repository contains a retrieval ablation suite for testing whether repeated query-side loops improve dense retrieval beyond a standard single-pass retriever.

The current experimental pipeline is:

```text
ModernBERT-base
-> RLHN-680K hard-negative training subset
-> SciFact retrieval evaluation with MTEB
```

The project intentionally does not use online tuning, exit gates, adaptive stopping, dimension slicing, pseudo labels, or in-batch negatives. All reported variants use normalized dot product similarity, equivalent to cosine similarity, and hard-negative cross entropy with `tau = 0.05`.

## Core Idea

The loop models keep the embedding dimension fixed and use query loop depth as the budget axis:

```text
h_1 = Pool_q(ModernBERT(query tokens))

for t = 2..Tmax:
    m_i = W_m h_i + e_i, for i = 1..t-1
    input_t = [m_1, ..., m_{t-1}, query tokens]
    h_t = Pool_q(ModernBERT(input_t))
```

Pooling excludes memory tokens. Documents are encoded once and do not loop. Every query state and document embedding is full-dimensional and L2-normalized.

## Experiment Versions

Experiment versions are registered in [src/experiments.py](src/experiments.py). Add new ablations there first so training, evaluation, and plotting stay consistent.

| Version | Family | What It Tests |
| --- | --- | --- |
| `standard` | baseline | Single-pass no-loop retriever. |
| `standard_more_steps` | baseline | Same as `standard`, trained for more optimizer steps as a stronger compute-budget baseline. |
| `loop_final` | loop curve | Memory-history loop model supervised only at `h_10`. |
| `loop_matryoshka` | loop curve | Memory-history loop model supervised at every `h_1..h_10`. |
| `loop_final_no_history` | loop curve | Same loss as `loop_final`, but previous loop states are not prepended as memory tokens. |
| `loop_matryoshka_no_history` | loop curve | Same loss as `loop_matryoshka`, but previous loop states are not prepended as memory tokens. |

## Repository Layout

```text
configs/                  YAML configs for smoke and pre-experiment runs
docs/                     Notes for adding future ablations
scripts/                  Local and Slurm experiment entry points
src/                      Training, model, evaluation, plotting, and registry code
requirements.txt          Python dependencies
```

Generated artifacts are intentionally ignored by git:

```text
outputs/                  checkpoints, MTEB results, plots, summaries
.hf_cache/                Hugging Face datasets/models cache
slurm_logs/               cluster stdout/stderr logs
```

## Installation

```bash
conda create -n loopmat python=3.10 -y
conda activate loopmat
pip install -r requirements.txt
```

On the cluster used for these experiments, the Slurm scripts default to:

```text
CONDA_ENV=/gpfs/projects/p32737/del6500_home/.conda/envs/ragen_ttt
partition=gengpu
account=p32737
```

Edit the `#SBATCH` fields and environment path in `scripts/*.sbatch` for a different cluster.

## Quick Smoke Test

From this directory:

```bash
bash scripts/run_smoke.sh
```

This trains the three main variants on the smoke config and evaluates SciFact. The Python orchestrator exposes the same workflow:

```bash
python -m src.run_all --config configs/smoke.yaml --stage all
```

## Main Pre-Experiment

Local:

```bash
bash scripts/run_preexp.sh
```

Slurm:

```bash
bash scripts/slurm_run_preexp.sh
```

The pre-experiment config uses `train_sample_size = 50000`, `Tmax = 10`, and `num_negatives = 7`.

## Ablation Entrypoints

Longer standard baseline:

```bash
bash scripts/run_standard_more_steps.sh
```

No-history loop ablation:

```bash
bash scripts/run_no_history_ablation.sh
```

Slurm no-history workflow:

```bash
bash scripts/slurm_run_no_history_ablation.sh
```

The longer standard baseline uses:

```text
r = (Tmax + K + 1) / (K + 2)
```

With `Tmax=10` and `K=7`, `r=2.0`, so `standard_more_steps` trains for two epochs over the same 50K subset. This is a stronger longer-training baseline, not an exact FLOPs match.

## Evaluation And Plots

Evaluation is parameter-free inference on SciFact through MTEB. It writes raw MTEB JSON, parsed metrics, and summary rows. The primary metrics are:

- `nDCG@10`
- `Recall@10`
- `Recall@100`
- `MRR@10`
- `MAP@10`

To replot from a combined summary:

```bash
python -m src.plot_results \
  --summary_csv outputs/plots/results_summary_all.csv \
  --output_dir outputs/plots
```

Plotting is implemented with the Python standard library plus Matplotlib; it does not depend on pandas.

## Output Convention

Organized runs use one directory per method:

```text
outputs/<method>/
  model/                 final checkpoint
  eval/                  raw and parsed MTEB outputs
  train_log.jsonl
  results_summary.csv

outputs/plots/
  results_summary_all.csv
  loop_depth_vs_*.png
  best_loop_summary.json
```

`outputs/` is ignored by git because checkpoints are large. Keep reproducible code/configs in the repository and archive generated artifacts separately.

## Adding New Ablations

See [docs/ADDING_ABLATIONS.md](docs/ADDING_ABLATIONS.md).

The short version:

1. Add a `VersionSpec` in [src/experiments.py](src/experiments.py).
2. Reuse the existing `standard` or `loop` family when possible.
3. Add a small script under `scripts/` only for orchestration.
4. Finalize generated outputs with `python -m src.finalize_ablation`.
5. Replot with `python -m src.plot_results`.
