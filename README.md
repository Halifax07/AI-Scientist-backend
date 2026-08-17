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
- 自适应实验树、Qwen 结果反馈和人类指导闸门
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
