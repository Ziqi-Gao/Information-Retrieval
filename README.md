# Loop-Wise Matryoshka Retrieval

This repository contains a retrieval ablation suite for testing whether repeated query-side loops improve dense retrieval beyond a standard single-pass retriever.

The current experimental pipeline is:

```text
ModernBERT-base
-> RLHN-680K hard-negative training subset
-> SciFact retrieval evaluation with MTEB
```

The project intentionally does not use online tuning, exit gates, adaptive stopping, dimension slicing, pseudo labels, in-batch negatives, trainable projection heads, or trainable memory adapters. All reported variants update only the ModernBERT encoder parameters and use normalized dot product similarity, equivalent to cosine similarity, with hard-negative cross entropy and `tau = 0.05`.

## Core Idea

The loop models keep the embedding dimension fixed and use query loop depth as the budget axis:

```text
x_1 = Emb(query tokens)

for t = 1..Tmax:
    y_t = ModernBERT(x_t)
    h_t = normalize(mean_pool(query-token positions in y_t))
    m_t = memory(y_t)
    x_{t+1} = [m_t, Emb(query tokens)]
```

```mermaid
flowchart TD
    TrainData["RLHN-680K hard-negative training subset"] --> Versions{"Experiment version"}

    Versions --> Standard["standard"]
    Versions --> LoopFinal["loop_final"]
    Versions --> LoopMat["loop_matryoshka"]

    TrainData --> Docs["Positive + hard-negative doc tokens"]
    Docs --> DocEncoder["ModernBERT encoder<br/>shared trainable weights"]
    DocEncoder --> DocPool["mean_pool + L2 normalize<br/>doc embeddings"]

    Standard --> StdQuery["Query tokens"]
    StdQuery --> StdEncoder["ModernBERT encoder<br/>trainable"]
    StdEncoder --> StdPool["mean_pool + L2 normalize"]
    StdPool --> StdLoss["Hard-negative CE loss"]
    DocPool --> StdLoss

    LoopFinal --> LoopInput["Query token embeddings"]
    LoopMat --> LoopInput
    LoopInput --> LoopEncoder["ModernBERT encoder loop<br/>shared trainable weights"]
    LoopEncoder --> QueryOnly["Drop memory-token positions<br/>keep query-token hidden states"]
    QueryOnly --> LoopPool["mean_pool + L2 normalize -> h_t"]
    QueryOnly --> MemoryMode{"parameter-free<br/>loop_memory_mode"}
    MemoryMode --> FirstToken["first_token: [B,1,H]"]
    MemoryMode --> MeanPool["mean_pool: [B,1,H]"]
    MemoryMode --> TokenConcat["token_concat: [B,L,H]"]
    FirstToken --> NextLoop["prepend memory to original query embeddings"]
    MeanPool --> NextLoop
    TokenConcat --> NextLoop
    NextLoop --> LoopEncoder

    LoopPool --> LoopLoss{"Loop objective"}
    LoopLoss --> FinalOnly["loop_final:<br/>loss only on h_T"]
    LoopLoss --> EveryLoop["loop_matryoshka:<br/>average loss over h_1..h_T"]
    DocPool --> LoopLoss

    FinalOnly --> Eval["SciFact MTEB retrieval evaluation"]
    EveryLoop --> Eval
    StdLoss --> Eval
```

Pooling excludes memory tokens. Documents are encoded once and do not loop:

```text
doc_embedding = normalize(mean_pool(ModernBERT(doc tokens)))
```

The loop memory construction is parameter-free and selected by `loop_memory_mode`:

| Mode | Memory tensor passed to the next loop |
| --- | --- |
| `first_token` | First query-token hidden state, shape `[B, 1, H]`. |
| `mean_pool` | Mean-pooled query-token hidden state, shape `[B, 1, H]`. This is the default. |
| `token_concat` | All query-token hidden states, shape `[B, L, H]`; the next loop input has shape `[B, 2L, H]`. |

Every query state and document embedding is full-dimensional and L2-normalized. With ModernBERT-base, `H = 768`.

## Experiment Versions

Experiment versions are registered in [src/experiments.py](src/experiments.py). Add new ablations there first so training, evaluation, and plotting stay consistent.

| Version | Family | What It Tests |
| --- | --- | --- |
| `standard` | baseline | Single-pass no-loop retriever; only ModernBERT encoder parameters are trainable. |
| `loop_final` | loop curve | Parameter-free memory loop retriever supervised only at the final loop. |
| `loop_matryoshka` | loop curve | Parameter-free memory loop retriever supervised at every loop. |

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

For Slurm runs, keep site-specific scheduler and environment choices outside the repository:

```bash
export CONDA_ENV=loopmat
export SBATCH_ARGS="--partition=<partition> --account=<account>"
bash scripts/slurm_run_preexp.sh
```

If `CONDA_ENV` is unset, batch scripts use `PYTHON_BIN` or the default `python` already available in the job shell.
Generated caches default to relative ignored directories such as `.hf_cache/` and `.mplconfig/`.

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

## Loop Memory Modes

The final experiment config uses `loop_memory_mode: mean_pool`. To run the same three registered versions with another parameter-free memory construction, override the config:

```bash
python -m src.run_all --config configs/smoke.yaml --stage train
python -m src.train --config configs/preexp.yaml --version loop_final --loop_memory_mode first_token
python -m src.train --config configs/preexp.yaml --version loop_matryoshka --loop_memory_mode token_concat
```

The version names stay fixed. `loop_memory_mode` changes only how the previous loop's query hidden states are converted into memory tokens for the next loop.

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

Smoke and pre-experiment runs use one directory per method under the run group:

```text
outputs/preexp/<method>/
  final/                 final checkpoint
  train_log.jsonl

outputs/preexp_eval/
  <method>/              raw and parsed MTEB outputs
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
