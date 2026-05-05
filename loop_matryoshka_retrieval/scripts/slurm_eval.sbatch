#!/bin/bash
#SBATCH --job-name=loopmat_eval
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --partition=gengpu
#SBATCH --account=p32737

# Edit account, partition, time, memory, and GPU request above for your HPC.

set -euo pipefail

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Start time: $(date)"

module purge || true
# module load cuda/12.1
# module load python/3.10

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
CHECKPOINT_DIR=${CHECKPOINT_DIR:-outputs/preexp/standard/final}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/preexp_eval}
EVAL_ALL_LOOPS=${EVAL_ALL_LOOPS:-true}

mkdir -p "${OUTPUT_DIR}" slurm_logs

"${PYTHON_BIN}" -m src.eval_mteb \
  --checkpoint_dir "${CHECKPOINT_DIR}" \
  --version "${VERSION}" \
  --task_name SciFact \
  --eval_all_loops "${EVAL_ALL_LOOPS}" \
  --batch_size 32 \
  --output_dir "${OUTPUT_DIR}"

echo "End time: $(date)"
