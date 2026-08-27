# 自主科研流程与页面映射

| 阶段 | 评委看到的页面 | 智能体行为 | 必须保存的产物 |
|---|---|---|---|
| 研究范围 | 领域、数据、约束、预算 | Supervisor 将宽泛目标转成研究边界 | research scope |
| 证据构建 | 文献矩阵、数据画像、真实性状态 | Evidence/Data Agent 检索和分析 | evidence records、dataset card |
| 空白发现 | 研究空白及其依据 | Gap Agent 比较论文结论和数据特征 | research gaps |
| 假设竞技场 | 候选假设、反方意见、Elo 排名 | Hypothesis/Skeptic 生成、反驳、演化 | hypothesis versions、debate log |
| 逐创新点实验计划 | 每个创新点的实验组、对照组、指标、最小配对数和实现状态 | Planner 为每个创新点建立独立可证伪契约；没有适配器的创新明确阻塞 | preregistered plan + hypothesis contract + SHA-256 |
| 逐创新点实验树 | 一个 Campaign 按顺序覆盖多个创新点；每个 Round 绑定一个创新点，固定 3 次内部迭代 | 第 1 次迭代完成后暂停一次；Guidance Agent 解释用户指导并规划第 2、3 次；Executor 自动执行剩余迭代 | innovation-bound rounds、one midpoint guidance、iteration/run manifests |
| 反馈迭代 | 当前 Round 三次成对效应、失败、Qwen 决策与下一个创新点 | Planner 只在同一 Round 内调整后两次迭代；完成后自动汇总并切换下一创新点 | round summary、feedback proposal、next-round seed |
| 结果审查 | 置信区间、失败图像、边界 | Statistics + Qwen-VL 分工分析 | findings、visual observations |
| 创新审查 | 创新卡、最近工作、证据链 | Meta-review 判断支持/部分支持/证伪 | innovation candidates |
| 报告导出 | 赛题研究计划与复现命令 | Reporter 只引用已登记产物 | PDF、JSON、代码与环境摘要 |

## 当前已完成的可执行边界

已实现：

- 研究对象、假设、计划、运行、发现和创新卡的数据模型；
- 从创建项目到报告清单的持久化状态机；
- 实验前人工审批；
- “一个 Campaign、多创新点、每个创新点一个 Round、每个 Round 三次内部迭代”的验证队列；
- Round 只绑定当前创新点，创新点切换发生在当前 Round 三次迭代汇总之后；
- 当前工具链不支持的创新点标记为“需要先实现方法”，实现适配器后才能进入真实执行；
- 每个 Round 第 1 次迭代后的单次用户指导输入、Qwen/确定性解释、后两次迭代自动排程和完整留痕；
- Mock 自主发现流程；
- Qwen/AgentScope 结构化调用适配器；
- 可复用的假设 Elo 锦标赛和假设演化请求模型；
- 按信息增益、证伪价值和成本选择节点的渐进实验树；每个 Round 的迭代目标固定为 3；
- 持久化 `ExperimentCampaign → ExperimentRound → ExperimentNode → ExperimentRun` 层级；
- 结果驱动的 Qwen 中途指导规划、允许动作校验、三迭代 Round 和 24-run 默认预算；
- arXiv/Crossref 在线检索、书目身份复核和部分失败降级；
- arXiv PDF 下载、逐页文本提取、Qwen 声明判断及逐字锚定拒绝器；
- MVTec AD 内容哈希清单、掩码完整性和 train/test 重复泄漏检查；
- DINOv2 CLS 正常图像特征缓存（严格拒绝 test 输入）；
- strict-K 与 pool-compression 协议、random/k-center 选择及几何指标；
- 只暴露 K 张 `train/good` 的硬链接/复制数据视图；
- PatchCore、单类别 AnomalyDINO、SubspaceAD 命令适配器；
- argv-only 执行器、超时/日志/环境摘要和三种结果解析器；
- 配对 bootstrap、精确/Monte-Carlo 符号置换检验和假设判定；
- 否定或不确定结论触发的有界假设修正循环（保留父子版本、计划与发现历史）；
- 下一研究循环启动前的用户建议闸门；建议会进入假设修订，但新假设仍需重新辩论、预注册和批准；
- FastAPI 数据审计、闭环初始化、执行下一 run、反馈规划和结果锁定接口；
- 面向方向 B 的闭环实验工作台，可视化轮次、成对证据、计算节省和 Research Ledger；
- Windows 中文路径下通过 CUDA 完成的 AnomalyDINO 合成与真实 MVTec 运行；
- 官方参考仓库与版本清单。

下一阶段扩展：

- detector × strategy 的混合效应/层级模型；
- Docker/Celery 多 GPU 调度和 MLflow 持久化；
- Qwen-VL 热力图审查；
- PDF 报告渲染。
