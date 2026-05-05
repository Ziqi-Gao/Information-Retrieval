#!/bin/bash
set -euo pipefail

mkdir -p slurm_logs outputs

CONFIG=${CONFIG:-configs/preexp.yaml}
OUTPUT_BASE=${OUTPUT_BASE:-outputs}
METHODS=(loop_final_no_history loop_matryoshka_no_history)

for METHOD in "${METHODS[@]}"; do
  RUN_DIR="${OUTPUT_BASE}/${METHOD}"
  if [ -e "${RUN_DIR}" ]; then
    echo "Refusing to overwrite existing run directory: ${RUN_DIR}" >&2
    echo "Move or remove it before rerunning this ablation." >&2
    exit 1
  fi
done

jid_loop_final_no_history=$(sbatch --parsable --export=ALL,VERSION=loop_final_no_history,CONFIG=${CONFIG},OUTPUT_BASE=${OUTPUT_BASE} scripts/slurm_train.sbatch)
jid_loop_matryoshka_no_history=$(sbatch --parsable --export=ALL,VERSION=loop_matryoshka_no_history,CONFIG=${CONFIG},OUTPUT_BASE=${OUTPUT_BASE} scripts/slurm_train.sbatch)

echo "Submitted no-history train jobs:"
echo "${jid_loop_final_no_history} loop_final_no_history"
echo "${jid_loop_matryoshka_no_history} loop_matryoshka_no_history"

jid_eval_loop_final_no_history=$(sbatch --parsable --dependency=afterok:${jid_loop_final_no_history} --export=ALL,VERSION=loop_final_no_history,CHECKPOINT_DIR=${OUTPUT_BASE}/loop_final_no_history/final,OUTPUT_DIR=${OUTPUT_BASE}/loop_final_no_history/eval_tmp,EVAL_ALL_LOOPS=true scripts/slurm_eval.sbatch)
jid_eval_loop_matryoshka_no_history=$(sbatch --parsable --dependency=afterok:${jid_loop_matryoshka_no_history} --export=ALL,VERSION=loop_matryoshka_no_history,CHECKPOINT_DIR=${OUTPUT_BASE}/loop_matryoshka_no_history/final,OUTPUT_DIR=${OUTPUT_BASE}/loop_matryoshka_no_history/eval_tmp,EVAL_ALL_LOOPS=true scripts/slurm_eval.sbatch)

echo "Submitted no-history eval jobs:"
echo "${jid_eval_loop_final_no_history} loop_final_no_history"
echo "${jid_eval_loop_matryoshka_no_history} loop_matryoshka_no_history"

jid_finalize=$(sbatch --parsable --dependency=afterok:${jid_eval_loop_final_no_history}:${jid_eval_loop_matryoshka_no_history} --export=ALL,METHODS="loop_final_no_history loop_matryoshka_no_history",PLOTS_DIR=${OUTPUT_BASE}/plots scripts/slurm_finalize_ablation.sbatch)

echo "Submitted finalize job:"
echo "${jid_finalize} finalize_no_history"
