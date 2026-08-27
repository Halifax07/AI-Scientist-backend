# 项目协作说明

## 仓库关系

- 当前仓库 `F:\AI-Scientist-backend` 是 FastAPI 后端、科研工作流与实验执行服务。
- 配套前端位于独立的同级仓库 `F:\AI-Scientist-frontend`，技术栈为 React、TypeScript 与 Vite。
- 涉及页面、交互、实验可视化或前后端契约的任务，必须同时检查前端的 `src/types.ts`、`src/api.ts`、`src/ExperimentCampaignPanel.tsx` 和相关样式。
- 修改后端响应模型时，要同步评估前端类型和展示是否需要更新；不要假设两个仓库会自动同步。

## 实验页面产品目标

实验执行页面必须让不了解内部实现的用户快速回答以下问题：

1. 当前实验要验证什么假设，比较哪两个方法，主指标是什么？
2. 整个实验进行到哪个阶段、哪一轮、哪个实验单元和哪个 Run？
3. 当前正在处理什么数据，数据规模、类别、K-shot、seed 和支持集策略是什么？
4. 已完成多少、还剩多少、预计还需多久，是否受预算或停止条件限制？
5. 当前结果说明了什么，证据是否足够，系统为什么扩展、复现、诊断或停止？
6. 如果失败，失败发生在哪一步，日志、错误和可复现产物在哪里？

前端展示应优先使用后端已经持久化的真实状态，不得用定时动画或虚构百分比模拟进度。

## 实验可视化优先级

后续扩展实验界面时，按以下顺序实现：

1. **实验总览**：假设、treatment/control、主指标、campaign 状态、当前轮次、`next_action`、终止原因。
2. **总体进度与预算**：已完成/失败/排队/运行 Run 数，`max_runs`、`max_rounds`、有效配对数、`minimum_pairs`、穷举运行数与节省运行数。
3. **当前 Run 卡片**：phase、category、detector、selection strategy、K-shot、seed、状态、开始时间、耗时、指标、错误。
4. **执行流水线**：数据审计、DINOv2 特征、支持集选择、数据视图、检测器执行、结果解析、结果入账、成对统计、反馈规划。
5. **轮次与实验树**：round objective/rationale、节点优先级、信息增益、证伪价值、预计成本、父子关系和节点内配对 Run。
6. **结果解释**：treatment/control 原始值、成对差值、累计效应、置信区间、p 值、样本量、是否达到最小证据门槛。
7. **反馈决策**：advisor、decision、rationale、observed patterns、expected information gain、推荐实验单元和受保护约束。
8. **数据与产物**：数据审计计数、支持集样本/几何指标、artifact paths、stdout/stderr、execution record、环境摘要和代码版本。
9. **Research Ledger 时间线**：用户指导、AI 解释、选择的 Run、状态变化、失败与重试、轮次总结和停止决策。

## 前后端数据契约

- 当前可直接使用的数据主要来自 `ResearchProject` 中的 `experiment_campaign`、`runs`、`guidance_records`、`events`、`findings`、`dataset_audits` 和 `artifacts`。
- `ExperimentRun` 已包含 `round_id`、`node_id`、`phase`、`status`、`metrics`、`artifact_paths`、`started_at`、`finished_at`、`duration_seconds`、`error`、`result_source`、`preparation_path` 和 `execution_record_path`；前端类型不得无故丢弃这些字段。
- `ExperimentNodeRecord` 的 `information_gain`、`falsification_value`、`estimated_cost`、`novelty`、`config` 和 `result_summary` 应作为解释实验优先级的核心数据。
- `ExperimentRound.result_summary` 与 `efficiency` 不应长期保持 `Record<string, unknown>`；新增展示前优先定义稳定、明确的后端模型与 TypeScript 类型。
- 若要显示单个 Run 内部的实时子步骤、百分比、GPU/显存、当前 epoch/batch 或 ETA，后端必须先增加可持久化的结构化进度事件或查询接口；不要从日志文本猜测关键状态。
- 状态标签要提供中文解释，并明确区分 `planned`、`queued`、`running`、`succeeded`、`failed`、`ready_for_feedback`、`completed` 和 `verified`，避免只展示内部英文枚举。

## 实现原则

- 先做信息层级和解释，再增加图表；每张图都要说明指标、方向和结论。
- 同时展示“配置”“过程”“结果”“决策依据”，不能只展示 AUROC 数字。
- treatment 与 control 必须成对展示，避免用户把单次结果误认为科学结论。
- 失败、证据不足和未核验状态必须显式展示，不能只突出成功结果。
- 保持科研可追溯性：页面上的关键数字应能追溯到 Run ID、轮次、统计摘要或产物路径。
