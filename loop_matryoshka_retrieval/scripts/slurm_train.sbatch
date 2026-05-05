#!/bin/bash
#SBATCH --job-name=loopmat_train
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --partition=gengpu
#SBATCH --account=p32737

# Edit account, partition, time, memory, and GPU request above for your HPC.

set -euo pipefail

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Start time: $(date)"

# Edit these module lines for your HPC.
module purge || true
# module load cuda/12.1
# module load python/3.10

# Activate your environment.
# Edit this path as needed.
set +u
source ~/.bashrc
set -u
CONDA_ENV=${CONDA_ENV:-/gpfs/projects/p32737/del6500_home/.conda/envs/ragen_ttt}
conda activate "${CONDA_ENV}"
PYTHON_BIN=${PYTHON_BIN:-python}

echo "Conda env: ${CONDA_ENV}"
echo "Python: $(which "${PYTHON_BIN}")"
"${PYTHON_BIN}" --version

export TOKENIZERS_PARALLELISM=false
export HF_HOME=${HF_HOME:-$PWD/.hf_cache}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HF_HOME/transformers}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HF_HOME/datasets}

VERSION=${VERSION:-standard}
CONFIG=${CONFIG:-configs/preexp.yaml}
OUTPUT_BASE=${OUTPUT_BASE:-outputs/preexp}

OUTDIR="${OUTPUT_BASE}/${VERSION}"

mkdir -p "${OUTDIR}" slurm_logs

echo "Training version: ${VERSION}"
echo "Config: ${CONFIG}"
echo "Output: ${OUTDIR}"

TRAIN_EXTRA_ARGS=()
if [ -n "${EPOCHS:-}" ]; then
  TRAIN_EXTRA_ARGS+=(--epochs "${EPOCHS}")
fi
if [ -n "${MAX_STEPS:-}" ]; then
  TRAIN_EXTRA_ARGS+=(--max_steps "${MAX_STEPS}")
fi
if [ -n "${SAVE_STEPS:-}" ]; then
  TRAIN_EXTRA_ARGS+=(--save_steps "${SAVE_STEPS}")
fi
if [ "${#TRAIN_EXTRA_ARGS[@]}" -gt 0 ]; then
  echo "Extra train args: ${TRAIN_EXTRA_ARGS[*]}"
fi

"${PYTHON_BIN}" -m src.train \
  --config "${CONFIG}" \
  --version "${VERSION}" \
  --output_dir "${OUTDIR}" \
  "${TRAIN_EXTRA_ARGS[@]}"

echo "End time: $(date)"
