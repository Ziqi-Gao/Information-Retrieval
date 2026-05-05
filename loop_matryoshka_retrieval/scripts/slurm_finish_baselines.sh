#!/bin/bash
set -euo pipefail

mkdir -p slurm_logs outputs/preexp outputs/preexp_eval

CONFIG=${CONFIG:-configs/preexp.yaml}
OUTPUT_BASE=${OUTPUT_BASE:-outputs/preexp}
EVAL_OUTPUT_DIR=${EVAL_OUTPUT_DIR:-outputs/preexp_eval}
CONDA_ENV=${CONDA_ENV:-/gpfs/projects/p32737/del6500_home/.conda/envs/ragen_ttt}

jid_standard=$(sbatch --parsable \
  --export=ALL,CONDA_ENV="${CONDA_ENV}",VERSION=standard,CONFIG="${CONFIG}",OUTPUT_BASE="${OUTPUT_BASE}" \
  scripts/slurm_train.sbatch)

jid_loop_final=$(sbatch --parsable \
  --dependency=afterok:${jid_standard} \
  --export=ALL,CONDA_ENV="${CONDA_ENV}",VERSION=loop_final,CONFIG="${CONFIG}",OUTPUT_BASE="${OUTPUT_BASE}" \
  scripts/slurm_train.sbatch)

jid_eval_standard=$(sbatch --parsable \
  --dependency=afterok:${jid_standard} \
  --export=ALL,CONDA_ENV="${CONDA_ENV}",VERSION=standard,CHECKPOINT_DIR="${OUTPUT_BASE}/standard/final",OUTPUT_DIR="${EVAL_OUTPUT_DIR}",EVAL_ALL_LOOPS=false \
  scripts/slurm_eval.sbatch)

jid_eval_loop_final=$(sbatch --parsable \
  --dependency=afterok:${jid_loop_final} \
  --export=ALL,CONDA_ENV="${CONDA_ENV}",VERSION=loop_final,CHECKPOINT_DIR="${OUTPUT_BASE}/loop_final/final",OUTPUT_DIR="${EVAL_OUTPUT_DIR}",EVAL_ALL_LOOPS=true \
  scripts/slurm_eval.sbatch)

jid_plot=$(sbatch --parsable \
  --dependency=afterok:${jid_eval_standard}:${jid_eval_loop_final} \
  --export=ALL,CONDA_ENV="${CONDA_ENV}",SUMMARY_CSV="${EVAL_OUTPUT_DIR}/results_summary.csv",PLOT_DIR="${EVAL_OUTPUT_DIR}/plots" \
  scripts/slurm_plot.sbatch)

echo "Submitted missing baseline pipeline:"
echo "${jid_standard} standard train"
echo "${jid_loop_final} loop_final train"
echo "${jid_eval_standard} standard eval"
echo "${jid_eval_loop_final} loop_final eval"
echo "${jid_plot} plot"
