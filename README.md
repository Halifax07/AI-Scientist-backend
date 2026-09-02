# AI Scientist Backend

面向少样本工业视觉异常检测的自主科研后端。系统使用 Qwen/AgentScope 完成研究问题形式化、
证据整合、可证伪假设生成与反馈规划，并通过受约束的本地工具链执行 MVTec 真实实验、
配对统计和 Research Ledger 留痕。

## 主要能力

- FastAPI JSON API 与持久化科研状态机
- arXiv/Crossref 检索及 PDF 声明级核验
- MVTec AD 数据审计与 DINOv2 正常样本画像
- random/k-center 少样本支持集选择
- AnomalyDINO、PatchCore、SubspaceAD 命令适配
- 多创新点并行实验树：用户排名后每个创新点对应一个 Round；首轮完成后每个 Round 接收一次用户指导，再自动完成第 2、3 次迭代
- SSE 实验流与 Research Ledger 回放：Run/Round/汇总状态实时推送，断线后可按序号恢复
- Qwen 结果规划、方法边界校验和 Research Ledger 留痕
- 配对 bootstrap、符号置换检验和创新审查

## 快速启动

```powershell
Copy-Item .env.example .env
./scripts/bootstrap.ps1 -WithAgentRuntime -WithExperimentTools
./scripts/dev-api.ps1
```

- API：`http://127.0.0.1:8000`
- OpenAPI：`http://127.0.0.1:8000/docs`
- 健康检查：`http://127.0.0.1:8000/health`

默认使用 JSON Research Ledger。数据集、模型权重、实验产物、日志和 `.env` 不进入 Git。
第三方方法源码请按 `third_party/manifest.lock.json` 使用 `scripts/sync-third-party.ps1` 获取，
并遵守各上游项目许可证。

## 验证

```powershell
./scripts/validate.ps1
```

项目当前测试覆盖自主科研状态机、证据核验、数据协议、自适应实验闭环、人机指导和 API。

## 自动排名与并行实验 API

新的主流程只有一个候选审核闸门：

```text
POST /projects/{id}/automation/ideation
  → 自动完成问题形式化、文献/证据、空白和可证伪假设生成
POST /projects/{id}/hypotheses/rank
  → 用户一次性提交 selected、priority、score；服务端自动生成缺失的策略适配器并生成预注册计划
POST /projects/{id}/experiment-campaign/auto-start
  → 数据清单审计通过后自动批准并为每个已选创新点建立一个 Round
POST /projects/{id}/experiment-campaign/execute-stream
  → 按 max_parallel_runs 并行执行，返回 text/event-stream
GET  /projects/{id}/experiment-campaign/events?after=N
  → 回放已持久化的结构化进度事件
```

并行 Round 预注册三次迭代（每次迭代为同类别、同 K、同 seed 的 random/k-center 成对运行），
不会穷举所有候选组合。系统先并行完成所有 Round 的第 1 次迭代，并发出
`round_guidance_required`；用户可在每个 Round 卡片中提交一次指导，系统只重排该 Round
已预注册的第 2、3 次迭代，不能改变方法、指标或数据边界。每个 Run 的排队、启动、结束、
Round 指导、Round 汇总和批次完成事件都会写入 `ResearchProject.experiment_progress`；全部
Round 完成时才发出 `campaign_completed`，随后继续流式发出结果锁定、统计分析、创新审查和
报告就绪事件。实验指标仍只能来自本地真实执行器或明确标记的 Mock。
