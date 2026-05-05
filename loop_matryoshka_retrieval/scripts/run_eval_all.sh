#!/bin/bash
set -euo pipefail

OUTPUT_BASE=${OUTPUT_BASE:-outputs/preexp}
EVAL_OUTPUT=${EVAL_OUTPUT:-outputs/preexp_eval}
PYTHON_BIN=${PYTHON_BIN:-python}

mkdir -p "${EVAL_OUTPUT}"

"${PYTHON_BIN}" -m src.eval_mteb --checkpoint_dir "${OUTPUT_BASE}/standard/final" --version standard --task_name SciFact --eval_all_loops false --output_dir "${EVAL_OUTPUT}"
"${PYTHON_BIN}" -m src.eval_mteb --checkpoint_dir "${OUTPUT_BASE}/loop_final/final" --version loop_final --task_name SciFact --eval_all_loops true --output_dir "${EVAL_OUTPUT}"
"${PYTHON_BIN}" -m src.eval_mteb --checkpoint_dir "${OUTPUT_BASE}/loop_matryoshka/final" --version loop_matryoshka --task_name SciFact --eval_all_loops true --output_dir "${EVAL_OUTPUT}"

"${PYTHON_BIN}" -m src.plot_results --summary_csv "${EVAL_OUTPUT}/results_summary.csv" --output_dir "${EVAL_OUTPUT}/plots"
