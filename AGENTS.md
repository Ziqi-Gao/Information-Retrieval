# Project Maintenance Rules

This file is for future Codex sessions working in this repository. Read it before making changes.

## Project Scope

- This repository maintains the Loop-Wise Matryoshka Retrieval experiments.
- Keep the public repository reproducible with code, configs, scripts, docs, and dependency declarations only.
- Generated artifacts are local-only unless the user explicitly asks otherwise.

## Privacy And Repository Hygiene

- Never commit personal paths, usernames, cluster account names, scheduler partition names, absolute home paths, API keys, tokens, or machine-specific environment paths.
- This restriction applies to tracked, committed, or pushed repository content.
- Local untracked or ignored files may contain machine-specific paths, runtime logs, cache paths, and environment details, but they must remain untracked.
- Use relative paths in README, docs, scripts, and configs.
- Keep site-specific Slurm choices outside the repo through environment variables such as `SBATCH_ARGS`, `CONDA_ENV`, and `PYTHON_BIN`.
- Do not commit these generated/local directories:
  - `outputs/`
  - `.hf_cache/`
  - `.mplconfig/`
  - `slurm_logs/*.out`
  - `slurm_logs/*.err`
  - `__pycache__/`
- Before pushing, check tracked content with commands like:

```bash
git status --short
git ls-files
git ls-files -ci --exclude-standard
git grep -n -I -E '<local-user>|<cluster-project>|<absolute-path-prefix>|SBATCH --account|SBATCH --partition|HF_TOKEN|WANDB_API_KEY'
```

Replace the placeholder patterns with the actual local username, cluster project/account names, and absolute path prefixes before running the check locally. The final `git grep` should return no repository secrets or personal machine paths in tracked content. Local runtime files may contain machine-specific information, but they must remain ignored or untracked.

## Experiment Logic

- Register experiment variants in `src/experiments.py` first. Training, evaluation, plotting, README, and scripts should derive from those version names.
- Keep version naming consistent across `src/experiments.py`, `src/train.py`, scripts, docs, and plots.
- Current experiments should update only ModernBERT encoder parameters. Do not add trainable projection heads, memory projections, memory-state embeddings, gates, or learned scaling unless the user explicitly asks for that ablation.
- Current parameter-free loop-memory modes mean:
  - `first_token`: prepend the previous loop's first query-token hidden state to the next loop.
  - `mean_pool`: prepend the previous loop's mean-pooled query-token hidden state to the next loop.
  - `token_concat`: prepend all previous-loop query-token hidden states to the next loop.
- Current loop-loss modes mean:
  - `loop_final`: train only from the final loop output.
  - `loop_matryoshka`: train from loopwise losses across loop depths.
  - `standard`: single-pass no-loop baseline.
- Do not reintroduce the old ambiguous `*_no_history` meaning or old `full`/`last`/`none` memory-history variants.

## Output And Plot Boundaries

- `outputs/` is ignored and may contain checkpoints, eval JSON, summaries, and plots.
- Do not delete or overwrite `outputs/plots/` unless the user explicitly asks. The user previously asked to preserve plot files.
- When removing stale experiment outputs, delete only the specific method directories that are obsolete or explicitly requested.
- Plotting should read combined summaries and write back under `outputs/plots/` by default.

## Slurm And Local Runtime

- Use `scripts/slurm_env.sh` for batch jobs. It should remain free of hardcoded local paths.
- Slurm wrapper scripts should accept scheduler-specific options through `SBATCH_ARGS`.
- If `CONDA_ENV` is unset, scripts should still work with `PYTHON_BIN` or the shell's default `python`.
- Hugging Face and Matplotlib caches should default to ignored relative directories such as `.hf_cache/` and `.mplconfig/`.

## Validation Checklist

Run relevant checks before saying the project is healthy:

```bash
python -m compileall -q src
bash -n scripts/*.sh scripts/*.sbatch
python -m src.train --help
python -m src.eval_mteb --help
python -m src.plot_results --help
```

For model or experiment-registry changes, also verify that each version resolves to the expected loss mode and that training exposes only `encoder.*` trainable parameters.

For broader changes, prefer a smoke run:

```bash
bash scripts/run_smoke.sh
```

## Git Workflow

- Keep commits focused on reproducible project files.
- Do not push generated outputs, caches, or large checkpoints.
- Do not rewrite remote history unless removing sensitive or clearly mistaken pushed content, and explain why.
- Before pushing, confirm `git status` is clean except for intentional ignored local artifacts.

## Collaboration Notes

- The user prefers concise Chinese explanations.
- Inspect the existing code and outputs before making behavioral claims.
- Do not make destructive file changes unless the user clearly requested them.
- If a request concerns experiment conclusions, distinguish between what the code tests, what current outputs show, and what would require new completed runs.
