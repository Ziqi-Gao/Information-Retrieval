#!/bin/bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-python}
CONFIG=${CONFIG:-configs/preexp.yaml}
METHOD=standard_more_steps
RUN_DIR=${RUN_DIR:-outputs/${METHOD}}
PLOTS_DIR=${PLOTS_DIR:-outputs/plots}
TASK_NAME=${TASK_NAME:-SciFact}

if [ -e "${RUN_DIR}" ]; then
  echo "Refusing to overwrite existing run directory: ${RUN_DIR}" >&2
  echo "Move or remove it, or set RUN_DIR to a new path." >&2
  exit 1
fi

read -r BASE_STEPS COMPUTE_MULTIPLIER TARGET_STEPS TRAIN_EPOCHS < <("${PYTHON_BIN}" - "${CONFIG}" <<'PY'
import sys

from src.compute_budget import more_steps_budget
from src.utils import load_yaml

config = load_yaml(sys.argv[1])
budget = more_steps_budget(config)
print(
    budget["base_steps"],
    f'{budget["compute_multiplier"]:.6f}',
    budget["target_steps"],
    budget["train_epochs"],
)
PY
)

mkdir -p "${RUN_DIR}" "${PLOTS_DIR}"

echo "Training ${METHOD}"
echo "Config: ${CONFIG}"
echo "Base optimizer steps: ${BASE_STEPS}"
echo "Approx compute multiplier: ${COMPUTE_MULTIPLIER}"
echo "Target optimizer steps: ${TARGET_STEPS}"
echo "Training epochs for target steps: ${TRAIN_EPOCHS}"

"${PYTHON_BIN}" -m src.train \
  --config "${CONFIG}" \
  --version "${METHOD}" \
  --output_dir "${RUN_DIR}" \
  --epochs "${TRAIN_EPOCHS}" \
  --max_steps "${TARGET_STEPS}" \
  --save_steps 0

mv "${RUN_DIR}/final" "${RUN_DIR}/model"

EVAL_TMP="${RUN_DIR}/eval_tmp"
"${PYTHON_BIN}" -m src.eval_mteb \
  --checkpoint_dir "${RUN_DIR}/model" \
  --version "${METHOD}" \
  --task_name "${TASK_NAME}" \
  --eval_all_loops false \
  --batch_size 32 \
  --output_dir "${EVAL_TMP}"

"${PYTHON_BIN}" -m src.finalize_ablation \
  --method "${METHOD}" \
  --run_dir "${RUN_DIR}" \
  --plots_dir "${PLOTS_DIR}"

"${PYTHON_BIN}" -m src.plot_results \
  --summary_csv "${PLOTS_DIR}/results_summary_all.csv" \
  --output_dir "${PLOTS_DIR}"

echo "Finished ${METHOD}."
echo "Run directory: ${RUN_DIR}"
echo "Combined summary: ${PLOTS_DIR}/results_summary_all.csv"
echo "Plots: ${PLOTS_DIR}"
