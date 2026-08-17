# 真实 MVTec 自适应实验闭环验收

> 本文记录系统验收，不把当前统计不确定的结果表述为“创新已验证”。

## 验收对象

- 项目：`project_a9dde7c91426`
- 活动：`campaign_8b5dd742937d`
- 数据：MVTec AD，15 类、6,612 文件
- 数据摘要：`ed7f275e1c0035f7ff40063bc7fd190ee742f106955c1280a561501d6dc7ebc6`
- 本机：NVIDIA GeForce RTX 4060 Laptop GPU，8 GB
- 检测器：AnomalyDINO，CUDA 视觉骨干、CPU FAISS
- 支持集画像：DINOv2-S CLS，只读取 `train/good`
- 协议：`pool_compression_m30`，random 对照 k-center
- 预注册主指标：Image AUROC

## 反馈迭代轨迹

1. 第 1 轮用 bottle、K=2、seeds 0/1 完成 2 个配对。Image AUROC 全部饱和为 1.0，Qwen 判定不能停止，改为跨类别验证。
2. 第 2 轮执行 cable/capsule、K=2、seed=0。两个类别的主效应方向相反，Qwen 继续选择新类别。
3. Qwen 的原建议含已执行 bottle；动作校验器拒绝重复单元，保留 transistor/K=2，并从允许空间补入 carpet/K=1。
4. 第 3 轮完成后累计达到 6 对；Qwen 建议继续扩大，但确定性规划器按三轮上限停止。

## 六个成对单元

| 类别 | K | seed | random Image | k-center Image | Image Δ | random AUPRO | k-center AUPRO | AUPRO Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bottle | 2 | 0 | 1.000000 | 1.000000 | 0.000000 | 0.962221 | 0.963103 | +0.000883 |
| bottle | 2 | 1 | 1.000000 | 1.000000 | 0.000000 | 0.967070 | 0.963828 | -0.003243 |
| cable | 2 | 0 | 0.935720 | 0.931222 | -0.004498 | 0.878458 | 0.893524 | +0.015067 |
| capsule | 2 | 0 | 0.896290 | 0.902274 | +0.005983 | 0.970566 | 0.968636 | -0.001930 |
| carpet | 1 | 0 | 1.000000 | 1.000000 | 0.000000 | 0.979900 | 0.978530 | -0.001370 |
| transistor | 2 | 0 | 0.940417 | 0.906667 | -0.033750 | 0.675698 | 0.660397 | -0.015302 |

## 正式主指标判定

- 配对数：6/6
- k-center − random 的平均 Image AUROC：`-0.0053774`
- bootstrap 95% CI：`[-0.0173770, 0.0022420]`
- 配对符号置换检验：`p=0.75`
- 判定：`inconclusive`

置信区间跨零且置换检验不显著，因此既不能支持“k-center 普遍优于 random”，也不能在当前小样本活动中给出稳定的总体负效应结论。transistor 的明显反向结果和 Image/AUPRO 方向不一致，是下一研究循环最值得解释的边界。

## 可追溯产物

- Research Ledger：`storage/projects/project_a9dde7c91426/project.json`
- 数据清单：`artifacts/datasets/ed7f275e1c0035f7ff40063bc7fd190ee742f106955c1280a561501d6dc7ebc6.json`
- DINOv2 缓存：`artifacts/features/ed7f275e1c0035f7/`
- 每个 run：`artifacts/runs/project_a9dde7c91426/<run_id>/execution.json`
- 每个 run 同目录保存 stdout、stderr、指标 JSON、预测图和环境摘要。

## 复现入口

```powershell
.\.venv\Scripts\python.exe scripts\run-real-loop-validation.py `
  'E:\揭榜挂帅\AI Scientist\data\mvtec_anomaly_detection' --runs 1
```

当前重大决策点：是冻结这套三轮闭环作为赛题演示，还是批准研究循环 2，围绕“代表性选择的收益依赖检测器、类别和 K”修订假设并追加预算。
