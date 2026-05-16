#!/bin/bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-python}

"${PYTHON_BIN}" -m src.train --config configs/smoke.yaml --version standard --output_dir outputs/smoke/standard
"${PYTHON_BIN}" -m src.train --config configs/smoke.yaml --version loop_final --output_dir outputs/smoke/loop_final
"${PYTHON_BIN}" -m src.train --config configs/smoke.yaml --version loop_matryoshka --output_dir outputs/smoke/loop_matryoshka
"${PYTHON_BIN}" -m src.train --config configs/smoke.yaml --version loop_final_recurrent_mean_pool --output_dir outputs/smoke/loop_final_recurrent_mean_pool
"${PYTHON_BIN}" -m src.train --config configs/smoke.yaml --version loop_matryoshka_recurrent_mean_pool --output_dir outputs/smoke/loop_matryoshka_recurrent_mean_pool
"${PYTHON_BIN}" -m src.train --config configs/smoke.yaml --version loop_final_recurrent_no_memory --output_dir outputs/smoke/loop_final_recurrent_no_memory
"${PYTHON_BIN}" -m src.train --config configs/smoke.yaml --version loop_matryoshka_recurrent_no_memory --output_dir outputs/smoke/loop_matryoshka_recurrent_no_memory

"${PYTHON_BIN}" -m src.eval_mteb --checkpoint_dir outputs/smoke/standard/final --version standard --task_name SciFact --eval_all_loops false --output_dir outputs/smoke_eval
"${PYTHON_BIN}" -m src.eval_mteb --checkpoint_dir outputs/smoke/loop_final/final --version loop_final --task_name SciFact --eval_all_loops true --output_dir outputs/smoke_eval
"${PYTHON_BIN}" -m src.eval_mteb --checkpoint_dir outputs/smoke/loop_matryoshka/final --version loop_matryoshka --task_name SciFact --eval_all_loops true --output_dir outputs/smoke_eval
"${PYTHON_BIN}" -m src.eval_mteb --checkpoint_dir outputs/smoke/loop_final_recurrent_mean_pool/final --version loop_final_recurrent_mean_pool --task_name SciFact --eval_all_loops true --output_dir outputs/smoke_eval
"${PYTHON_BIN}" -m src.eval_mteb --checkpoint_dir outputs/smoke/loop_matryoshka_recurrent_mean_pool/final --version loop_matryoshka_recurrent_mean_pool --task_name SciFact --eval_all_loops true --output_dir outputs/smoke_eval
"${PYTHON_BIN}" -m src.eval_mteb --checkpoint_dir outputs/smoke/loop_final_recurrent_no_memory/final --version loop_final_recurrent_no_memory --task_name SciFact --eval_all_loops true --output_dir outputs/smoke_eval
"${PYTHON_BIN}" -m src.eval_mteb --checkpoint_dir outputs/smoke/loop_matryoshka_recurrent_no_memory/final --version loop_matryoshka_recurrent_no_memory --task_name SciFact --eval_all_loops true --output_dir outputs/smoke_eval

"${PYTHON_BIN}" -m src.plot_results --summary_csv outputs/smoke_eval/results_summary.csv --output_dir outputs/smoke_eval/plots
