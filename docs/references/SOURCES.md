# 本地参考资料来源

## AI Scientist 系统

- Google Co-Scientist：<https://www.nature.com/articles/s41586-026-10644-y>。官方说明完整源码目前不公开；本地仅保存论文 `papers/google-co-scientist-nature-2026.pdf`。
- Sakana AI Scientist-v2：<https://github.com/SakanaAI/AI-Scientist-v2>，本地源码位于 `third_party/ai_scientists/ai-scientist-v2`。
- Sakana AI Scientist v1：<https://github.com/SakanaAI/AI-Scientist>。
- Robin：<https://github.com/Future-House/robin>。
- Finch：<https://github.com/Future-House/finch>。

## 智能体框架

- AgentScope：<https://github.com/agentscope-ai/agentscope>。
- Qwen-Agent：<https://github.com/QwenLM/Qwen-Agent>。

本项目当前选择 AgentScope 作为智能体调用适配层；Qwen-Agent 仅作为 Qwen 工具调用、RAG 和代码解释器的官方参考，不同时叠加两套主编排框架。

## 异常检测方法

- PatchCore：<https://github.com/amazon-science/patchcore-inspection>。
- AnomalyDINO：<https://github.com/dammsi/AnomalyDINO>。
- SubspaceAD：<https://github.com/CLendering/SubspaceAD>。
- FastRef：论文声明的官方仓库为 <https://github.com/liyufei25/FastRef>，但在本次同步时仍为空；本地保存 CVPR 论文 `papers/fastref-cvpr-2026.pdf`。

具体提交版本和文件校验值见 `third_party/manifest.lock.json`。

