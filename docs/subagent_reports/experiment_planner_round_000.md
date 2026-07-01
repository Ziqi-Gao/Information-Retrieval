# experiment_planner report

## Manifest requirements

- 必填：`batch_id`, `description`, `purpose: dev|final|smoke`, `primary_metric: ndcg_at_10`, `win_margin: 0.001`, `baseline.summary_csv`, `baseline.manifest_json`, `budget`, `tasks`, `defaults`, `experiments`.
- `tasks.final` 固定为：`SciFact`, `NFCorpus`, `SCIDOCS`, `FiQA2018`, `ArguAna`, `Touche2020`, `TRECCOVID`，顺序也应固定。
- 每个 experiment 必须有：`run_id`, `hypothesis`, `version`, `config`, `train`, `eval.task_names`, `eval.eval_all_loops`, `risk.level`, `risk.reason`。
- `version` 只能来自 `src/experiments.py`；第一批优先用现有 registered variants，不新增模型语义。
- `run_id` 必须安全、唯一、可恢复；候选身份应预注册为 `run_id__loop{k}`，不能 final 后按 task 选最佳 loop。
- 输出根目录必须在 `outputs/goal/runs/<batch_id>/` 和 `outputs/goal/eval/<batch_id>/`，不得写入 `outputs/baselines/`。

## State/resume requirements

- `outputs/goal/state.json` 是唯一 resume 入口；恢复时先读 state，再读 `outputs/goal/runs/<batch_id>/submission_plan.json`。
- state 至少记录：`phase`, `baseline.status/path/manifest/sha256`, `current_batch`, `open_jobs`, `budget`, `primary_metric`, `win_margin`, `final_tasks`, `last_scoreboard`, `best_candidate`, `next_required_action`。
- 阶段流转保持：baseline frozen/validated -> manifest validated -> dry-run plan -> submit -> status -> collect -> score -> decide。
- queue delay 后只通过 `scripts/goal_status.py --update-state` 判断 job 状态；不能按时间推断完成。
- 已存在非空输出目录时默认拒绝，只有明确 `--resume` 才允许继续。

## Budget controls

- manifest 必须声明：`max_concurrent_gpu_jobs`, `max_gpu_hours_estimate`, `allow_submit`。
- 当前 state 限制应作为上限：并发 GPU jobs 不超过 4，单 batch 估算不超过 24 GPU-hours，除非显式 `allow_over_budget` 且人工确认。
- `allow_submit: false` 是默认；dev/final 真实提交前必须先 baseline frozen、manifest valid、dry-run plan reviewed。
- Slurm 只能经 `scripts/goal_submit_batch.py`；环境变量只允许安全白名单，不能导出 token/key。
- 第一批 dev 建议低并发：`max_concurrent_gpu_jobs: 2`，`max_gpu_hours_estimate: 12`，先 dry-run，不自动 submit。

## Suggested dev/final protocol

- Dev：只用于筛选假设和发现失败模式；建议先跑 `SciFact`, `NFCorpus`，必要时加 `SCIDOCS`，但 dev 结果不能支撑 final claim。
- Final：只允许预注册候选进入，必须覆盖全部 7 个 final tasks。
- Scoring：对每个预注册 `candidate_id`，每个 final task 必须满足 `candidate_ndcg_at_10 >= frozen_standard_ndcg_at_10 + 0.001`。
- 缺失、NaN、重复、failed、timeout、partial task、invalid metric 都是失败；不能当 0、tie 或可忽略项。
- loop 曲线可分析，但 final claim 只能用预定义 loop candidate，例如 `run_id__loop3` 横跨所有 final tasks；不能 final 后逐 task 挑最佳 loop。
- 当前只读观察：官方 frozen baseline 仍缺失，`outputs/baselines/standard_frozen/results_summary.csv` 和 manifest 尚未存在；真实实验前必须先 freeze 或 validate。

## Candidate first batch

建议未来创建 `batch_002_dev_standard_preserving.yaml`，purpose 为 `dev`，baseline 指向 frozen standard，`allow_submit: false` 初始只 dry-run。

候选：
- `dev002_loop_final_recurrent_no_memory`
  - `version: loop_final_recurrent_no_memory`
  - 假设：仅用 recurrent query hidden state、不加 memory token，检验是否保留 standard-like 简洁性同时改善 loop refinement。
  - `eval.task_names: [SciFact, NFCorpus]`, `eval_all_loops: true`
- `dev002_loop_matryoshka_recurrent_no_memory`
  - `version: loop_matryoshka_recurrent_no_memory`
  - 假设：无 memory token 的 loopwise supervision 是否比 final-only 更稳定。
  - 同样只 dev tasks，所有 loop 作为独立 candidate IDs。
- `dev002_loop_final_recurrent_mean_pool`
  - `version: loop_final_recurrent_mean_pool`
  - 假设：最小 parameter-free mean-pool memory 是否优于 no-memory recurrent。
  - 同样只 dev tasks。

训练建议先用 `configs/preexp.yaml` 的保守规模，或显式 `max_steps` 上限作为 dev 探针；不重训 `standard` 作为比较基线，避免 moving baseline 混淆。

## Risks

- baseline 尚未 frozen；没有它就不能启动真实 dev/final 自主实验。
- 当前 worktree dirty 且多处 goal 框架文件未提交；未来 session 需先确认这些改动是预期状态。
- `goal_collect.py` 读 summary CSV，不深查 raw MTEB JSON；raw/summary 不一致仍需人工抽查。
- 小 dev tasks 可能和 full final 排名不一致；dev 只能淘汰明显失败项，不能证明胜出。
- `eval_all_loops: true` 会产生多个 loop candidate，必须在 scoring/report 中避免 post-hoc task-wise cherry-pick。

## Uncertainty

- 没有运行训练、评测、Slurm 或 baseline freeze；以上是控制流程规划，不是实验结果。
- 未验证 `outputs/final_grid_experiment/eval/summaries/standard_results_summary.csv` 是否满足 freeze 脚本要求。
- 第一批 dev 的 GPU-hour 估算需要根据实际 cluster 运行时间校准；建议首次真实 submit 前保留人工确认。
