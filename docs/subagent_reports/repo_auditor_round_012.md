# repo_auditor_round_012

## Verdict

是，**在明确用户批准后**，为 exact `r017_seeded_lexical_hash__loop1` 规则创建并提交 `purpose: final`、七个 final tasks 全覆盖的验证批次是 protocol-valid。当前 `batch_017_dev_repair` 本身仍不是 final claim：它是 `purpose: dev`，只评估 4/7 final tasks。

## Evidence

- 协议要求 final claim 必须 `purpose: final`、`standalone_main`、七任务全覆盖、预声明 loop/candidate、无 frozen-standard scoring input：`docs/goal_protocol.md:13-18,38-46,125-127,465-481`。
- Repair manifest 保留 exact rule：`standard_seeded_sampling`、r016 checkpoint、`loop_idx=1`、`lexical_hash_dim=1024`、`lexical_weight=0.15`、candidate-only scoring：`experiments/batches/batch_017_dev_repair.yaml:41-66`。
- Repair 结果是强 dev signal：4 个 dev tasks 全部 completed 且 delta 都超过 `+0.002`，min delta `+0.00665`，mean delta `+0.0162525`：`outputs/goal/runs/batch_017_dev_repair/scoreboard.json:4-24,30-76`。
- Repair 不能当 final：scoreboard 明确缺 `ArguAna/Touche2020/TRECCOVID`，`purpose=dev`，`main_goal_success=false`：`scoreboard.json:6-18,78-112`。
- State 也记录下一步只能在 explicit user approval 后做 final validation：`outputs/goal/state.json:42-48,215`。

## Required Final Manifest Shape

Final validation must preserve the same candidate rule exactly:

- `purpose: final`
- all seven protocol tasks: `SciFact,NFCorpus,SCIDOCS,FiQA2018,ArguAna,Touche2020,TRECCOVID`
- `claim_track: standalone_main`
- `eval_only: true`
- same checkpoint: `outputs/goal/runs/batch_016_dev/r016_standard_seeded_sampling/final`
- `version: standard_seeded_sampling`
- `loop_idx: 1`
- `candidate_loop_indices: [1]`
- `lexical_hash_dim: 1024`
- `lexical_weight: 0.15`
- no `fusion_standard_checkpoint_dir`, `fusion_alpha`, `fusion_scope`, doc chunking, or self-query changes.

## Blockers / High Risks

- No final batch has been validated here; current approved/completed artifact is dev-only.
- Final validation must use `scripts/goal_submit_batch.py`; no manual `sbatch`.
- Any missing/failed/NaN/duplicate/partial final-task result invalidates the final claim.
- The three untested final tasks may regress; dev success is not final evidence.
- Keep generated `outputs/` and Slurm/site-specific paths untracked.
