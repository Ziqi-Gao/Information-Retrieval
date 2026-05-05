#!/bin/bash
set -euo pipefail

if [ -n "${CONDA_ENV:-}" ]; then
  if ! command -v conda >/dev/null 2>&1; then
    echo "CONDA_ENV is set, but conda is not available in this batch shell." >&2
    echo "Load conda before submitting, or set PYTHON_BIN to an environment-specific Python." >&2
    exit 1
  fi
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV}"
fi

PYTHON_BIN=${PYTHON_BIN:-python}
export PYTHON_BIN

export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export HF_HOME=${HF_HOME:-.hf_cache}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-${HF_HOME}/datasets}
