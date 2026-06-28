# Experiment Planner Round 009

Read-only recommendation; no files edited, no Slurm action run.

## Evidence Used

`batch_015_dev` was dev-only `standalone_main`. Best candidate was `r015_role_dim_mrl_standard__loop1`: mean delta `+0.00212`, wins on `NFCorpus` and `SCIDOCS`, but still regressed `SciFact` `-0.00320` and `FiQA2018` `-0.00255`. Role/MRL no-loop variants therefore produced useful split evidence, not a protocol-viable global signal.

## Recommended `batch_016_dev` Portfolio

Use 3 dev-only `standalone_main` candidates, estimated total `24` GPU hours, `max_concurrent_gpu_jobs: 3`.

| run_id | mechanism | estimate | why include |
|---|---|---:|---|
| `r016_qdoc_final_mean_pool_t3_neg3` | Train/eval both queries and docs with mean-pool loops, `tmax=3`, `num_negatives=3`, final-loop loss, `loop_docs: true`, `doc_loop_idx: 3`. | 10h | Directly tests q/doc representation asymmetry, aimed at persistent `FiQA2018/SCIDOCS` regressions. Outside first-token, role-prompt, and dim-MRL families. |
| `r016_standard_seeded_sampling` | No-loop standard retriever with deterministic `seeded_random` positive/negative passage sampling, eval `loop_idx=1`. | 7h | Tests whether RLHN first-positive/first-negatives ordering is hurting transfer. Outside loop-memory and role/MRL families. |
| `r016_standard_inbatch_hybrid` | No-loop standard hard-negative loss plus candidate-only in-batch positive classification, e.g. `inbatch_weight=0.25`, eval `loop_idx=1`. | 7h | Tests broader contrastive geometry after batch_015 hurt `SciFact/FiQA2018`; avoids frozen-standard inputs and avoids loop-memory drift. |

## Protocol Notes

All candidates should evaluate only dev tasks: `SciFact`, `NFCorpus`, `FiQA2018`, `SCIDOCS`.

All candidates must use candidate-only embeddings/scores. Do not set `fusion_standard_checkpoint_dir`, `fusion_alpha`, `fusion_scope`, standard embeddings, standard scores, or score interpolation.

Do not tune on final-only tasks. `ArguAna`, `Touche2020`, and `TRECCOVID` should remain absent from this dev run by design.

## Likely Files To Change

- `src/experiments.py`: register `loop_final_qdoc_mean_pool`, `standard_seeded_sampling`, `standard_inbatch_hybrid`.
- `src/train.py`: add q/doc looped training batch path; allow/select no-loop in-batch standard loss.
- `src/losses.py`: likely no change if q/doc candidate reuses `final_loop_loss`; only change if adding per-depth q/doc losses.
- `configs/goal_batch_016_*.yaml`: one config per candidate.
- `experiments/batches/batch_016_dev.yaml`: predeclare candidates, dev tasks, budget, and candidate loop indices.

`src/model.py` and `src/eval_mteb.py` likely need no change: `encode_docs_looped`, `loop_docs`, and `doc_loop_idx` already exist.

## Postprocess Expectation

Submit later only through:

```bash
python scripts/goal_submit_batch.py experiments/batches/batch_016_dev.yaml --submit --submit-postprocess
```

Expected artifacts under `outputs/goal/runs/batch_016_dev/`: `submission_plan.json`, Slurm-native `postprocess_done.json`, `collected_results.csv`, and `scoreboard.json`.

Expected scoreboard shape: each candidate has 4 completed dev rows, `tasks_valid=4`, `candidate_track=standalone_main`, no fusion fields, and `main_goal_success=false` because `purpose=dev` and only 4/7 final tasks are evaluated.
