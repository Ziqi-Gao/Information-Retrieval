#!/bin/bash
set -euo pipefail

DEFAULT_CONDA_ENV=${DEFAULT_CONDA_ENV:-ragen_ttt}

load_conda() {
  if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    return 0
  fi

  for conda_sh in \
    "${CONDA_SH:-}" \
    "/software/miniconda3/4.10.3/etc/profile.d/conda.sh" \
    "/software/miniconda3/etc/profile.d/conda.sh"; do
    if [ -n "${conda_sh}" ] && [ -r "${conda_sh}" ]; then
      # shellcheck disable=SC1090
      source "${conda_sh}"
      return 0
    fi
  done

  return 1
}

find_conda_env_prefix() {
  local env_name="$1"
  conda env list | awk -v env_name="${env_name}" '
    $0 !~ /^#/ {
      path=$NF
      n=split(path, parts, "/")
      if (parts[n] == env_name) {
        print path
        exit
      }
    }
  '
}

if [ -z "${PYTHON_BIN:-}" ]; then
  requested_env="${CONDA_ENV:-${DEFAULT_CONDA_ENV}}"
  if [ -n "${requested_env}" ]; then
    if ! load_conda; then
      echo "Conda is not available in this batch shell." >&2
      echo "Set PYTHON_BIN to an environment-specific Python." >&2
      exit 1
    fi

    if ! conda activate "${requested_env}" 2>/dev/null; then
      env_prefix=$(find_conda_env_prefix "${requested_env}")
      if [ -z "${env_prefix}" ]; then
        echo "Could not find conda environment '${requested_env}'." >&2
        echo "Set CONDA_ENV to a valid conda environment/prefix, or set PYTHON_BIN directly." >&2
        exit 1
      fi
      conda activate "${env_prefix}"
    fi
  fi
fi

if [ -z "${PYTHON_BIN:-}" ]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN=python
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN=python3
  else
    echo "Could not find python or python3 in PATH." >&2
    echo "Set PYTHON_BIN to an environment-specific Python." >&2
    exit 1
  fi
fi
export PYTHON_BIN

if [ -n "${CONDA_PREFIX:-}" ] && [ -d "${CONDA_PREFIX}/lib" ]; then
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi

export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export HF_HOME=${HF_HOME:-.hf_cache}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-${HF_HOME}/datasets}
