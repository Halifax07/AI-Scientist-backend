# 少样本工业异常检测研究协议

## 两条不能混淆的协议

### Strict K-shot

系统只观察到 K 张正常图像，没有额外候选池，因此不能声称在同一批图像中“选择最具代表性的 K 张”。该协议研究不同随机支持集带来的敏感度和稳定性。

### Pool compression

系统先获得 M 张正常候选图像，再从中选择 K 张进入检测器。该协议研究参考库压缩、部署成本和代表性选择，应明确报告 M 和候选池生成方式。

## 初始方法矩阵

- K：1、2、4、8。
- M：30。
- 策略：random、k-center、k-medoids、DPP。
- 检测器：PatchCore、AnomalyDINO、SubspaceAD。
- 扩展：FastRef，待官方仓库发布代码后接入。
- 主数据集：MVTec AD。
- 验证数据集：VisA。
- 初始类别：bottle、carpet、capsule、cable、transistor。

## 预注册指标

- Image AUROC、Image AP。
- Pixel AUROC、AUPRO。
- 支持集重采样标准差和 95% CI。
- 最坏十分位性能。
- 正常特征覆盖半径。
- 特征协方差有效秩或 PCA 有效维度。

正式结论使用至少 10 次支持集重采样；3 次重复只用于可行性检查。

## 自适应执行顺序

完整因素矩阵是候选空间，不是一次性队列。默认预算为 3 轮、24 个 run：

1. 可行性轮：`bottle / K=2 / seeds 0,1`，random 与 k-center 成对，共 4 个 run；
2. 反馈轮：根据真实主指标、像素指标、失败运行和现有配对数，选择最多两个新单元；
3. 复现或诊断轮：优先补足跨类别证据、K 边界或独立 seed；
4. 达到 `minimum_pairs` 才允许证据充分停止，预算/轮次耗尽时只能标记为证据不足并进入统计。

实验节点优先级为：

```text
(information_gain × falsification_value) / estimated_cost + 0.05 × novelty
```

Qwen 输出的是结构化 `ExperimentFeedbackProposal`。服务端会拒绝重复单元、未批准类别/K/seed、非 random/k-center 策略、超预算建议和过早停止。

## 泄漏防护

- 支持样本选择只能访问候选正常训练池。
- 选择策略不得访问测试图像、异常标签、掩膜或最终指标。
- 每个支持集保存完整文件清单和 SHA-256。
- 同一候选池和 seed 下对策略进行配对比较。
- 参数选择、主要检验和停止条件在实验前冻结。
