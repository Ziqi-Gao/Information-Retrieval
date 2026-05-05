#!/bin/bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-python}
CONFIG=${CONFIG:-configs/preexp.yaml}
PLOTS_DIR=${PLOTS_DIR:-outputs/plots}
TASK_NAME=${TASK_NAME:-SciFact}
METHODS=(loop_final_last loop_final_none loop_matryoshka_last loop_matryoshka_none)

for METHOD in "${METHODS[@]}"; do
  RUN_DIR="outputs/${METHOD}"
  if [ -e "${RUN_DIR}" ]; then
    echo "Refusing to overwrite existing run directory: ${RUN_DIR}" >&2
    echo "Move or remove it before rerunning this ablation." >&2
    exit 1
  fi
done

mkdir -p "${PLOTS_DIR}"

for METHOD in "${METHODS[@]}"; do
  RUN_DIR="outputs/${METHOD}"
  mkdir -p "${RUN_DIR}"

  echo "Training ${METHOD}"
  echo "Config: ${CONFIG}"
  echo "Output: ${RUN_DIR}"

  "${PYTHON_BIN}" -m src.train \
    --config "${CONFIG}" \
    --version "${METHOD}" \
    --output_dir "${RUN_DIR}"

  mv "${RUN_DIR}/final" "${RUN_DIR}/model"

  EVAL_TMP="${RUN_DIR}/eval_tmp"
  "${PYTHON_BIN}" -m src.eval_mteb \
    --checkpoint_dir "${RUN_DIR}/model" \
    --version "${METHOD}" \
    --task_name "${TASK_NAME}" \
    --eval_all_loops true \
    --batch_size 32 \
    --output_dir "${EVAL_TMP}"

  "${PYTHON_BIN}" -m src.finalize_ablation \
    --method "${METHOD}" \
    --run_dir "${RUN_DIR}" \
    --plots_dir "${PLOTS_DIR}"
done

"${PYTHON_BIN}" -m src.plot_results \
  --summary_csv "${PLOTS_DIR}/results_summary_all.csv" \
  --output_dir "${PLOTS_DIR}"

echo "Finished memory-history range ablations."
echo "Run directories:"
for METHOD in "${METHODS[@]}"; do
  echo "  outputs/${METHOD}"
done
echo "Combined summary: ${PLOTS_DIR}/results_summary_all.csv"
echo "Plots: ${PLOTS_DIR}"
