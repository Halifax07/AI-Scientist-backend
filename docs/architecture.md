# 系统架构

## 产品定义

FSAD Scientist 是科研工作台，不是聊天机器人。用户只给研究领域、数据、现实约束和预算；系统负责发现研究空白、提出可证伪假设、设计并执行实验、分析结果，并在证据充分时输出创新候选。

## 分层结构

```text
React research workbench
        │ HTTP + SSE/WebSocket
FastAPI application service
        │
Durable Research Workflow / Research Ledger
        ├── Human Guidance Agent / guarded action selector
        ├── Supervisor
        ├── Evidence Agent
        ├── Data Scientist Agent
        ├── Hypothesis Agent
        ├── Skeptic / Meta-review Agent
        ├── Experiment Planner / Qwen Feedback Advisor
        └── Result & Innovation Agent
        │ registered tool calls only
Scientific tool boundary
        ├── arXiv/Crossref search + PDF claim anchoring
        ├── dataset hash audit + DINOv2 profiler
        ├── adaptive experiment campaign + progressive experiment tree
        ├── immutable support-set views + argv-only local GPU executor
        ├── detector-specific result normalization
        ├── paired bootstrap + sign permutation statistics
        └── Qwen-VL visual failure reviewer
        │
JSON artifacts now; MLflow + PostgreSQL/pgvector + object storage migration later
```

## 为什么自建状态机而不把状态全部交给 AgentScope

AgentScope 用于智能体推理、消息与工具调用；科研阶段、预算、人工审批、证据状态和实验 Run ID 则由本项目的持久化状态机管理。这样即使模型调用失败、上下文被压缩或更换模型，已完成的科研证据不会丢失，也不会越过审批闸门。

## 当前状态机

```text
created
→ scope_formalized
→ evidence_ready
→ gaps_discovered
→ hypotheses_proposed
→ hypotheses_reviewed
→ awaiting_experiment_approval
→ experiments_queued
→ results_ready
→ results_analyzed
→ [若证伪/不确定且仍有循环预算：修正假设并返回 hypotheses_reviewed]
→ innovation_reviewed
→ report_ready
```

关键规则：

- `awaiting_experiment_approval` 必须由人批准预注册计划。
- 每个真实 run 执行前必须接收一条用户指导；LLM 只能据此在本轮冻结队列中重排，服务端会校验最终 Run ID，原文、解释、处置结果和保护边界全部入账。
- `experiments_queued` 不能直接进入分析，必须导入真实且终态的 Run。
- `experiment_campaign` 把执行阶段拆为多轮：`planned → running → ready_for_feedback → completed`。
- 每轮结束后，Qwen 只从服务端生成的 `allowed_cells` 中推荐最多两个类别/K/seed 单元；每个单元由系统强制生成 random/k-center 成对运行。
- 提前停止只有在达到预注册 `minimum_pairs` 后才会被接受；轮次、run 预算或候选空间耗尽时可确定性停止。
- 只有 `verified=true` 的实验运行可以支持统计发现。
- 只有文献和实验均完成校验，结果才可标记为 `evidence_supported_candidate`。
- 假设修正默认最多 3 个研究循环，每一轮都保留父假设、旧计划和旧发现，不能覆盖失败证据。
- 启动否定/不确定结论后的下一研究循环时，界面要求用户先提交建议；建议进入 Hypothesis Revision Agent，新循环仍回到假设辩论和实验预注册闸门。
- 已完成的 `ExperimentCampaign` 会转入历史集合，新循环创建独立 campaign；旧 Run 和负结果不会被新队列覆盖。

## 代码边界

- `src/fsad_scientist/domain`：科研对象和状态，不依赖 Agent 框架。
- `src/fsad_scientist/workflow.py`：状态转换与人工闸门。
- `src/fsad_scientist/agents`：Mock、Qwen/AgentScope 等认知运行时。
- `src/fsad_scientist/experiments`：异常检测方法适配器与执行命令。
- `src/fsad_scientist/experiments/loop.py`：轮次、节点优先级、允许动作、预算验证、真实结果摘要和下一轮构造。
- `src/fsad_scientist/evidence`：真实检索、PDF 提取和声明锚定。
- `src/fsad_scientist/datasets`：数据哈希审计和不可变少样本视图。
- `src/fsad_scientist/features`：仅对 `train/good` 运行的 DINOv2 特征画像。
- `src/fsad_scientist/science`：假设竞技、实验树和确定性统计。
- `src/fsad_scientist/tools`：允许智能体调用的工具能力清单。
- `tests`：后端状态机、API、数据和实验闭环测试。
- 配套前端是独立 React/Vite 仓库，仅通过 FastAPI JSON API 访问本服务。
- `third_party`：只读官方参考代码，不在其中开发本项目逻辑。

## 信任边界

LLM 可以：提出假设、解释证据、设计实验、提出下一步。

LLM 不可以：

- 自己填写 AUROC、AUPRO 或置信区间；
- 把未校验论文标记为真实证据；
- 使用测试标签选择正常参考集；
- 修改已批准实验计划而不生成新版本和摘要；
- 在没有 Run ID 的情况下宣布创新得到验证。
- 提交任意 shell 命令、绕过允许的实验单元或扩大预注册预算。
- 把用户的自然语言建议直接翻译为未登记参数；单次实验建议只能改变已冻结候选 Run 的执行顺序。

证据状态被明确拆为三级：`unverified`、`metadata_verified`、`verified`。Crossref/arXiv 返回结果最多晋级到书目级；只有 Qwen 返回的短引文能够在指定 PDF 页逐字定位时，相关 claim 才能晋级到声明级。
