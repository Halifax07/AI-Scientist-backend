# Third-party references

该目录存放浅克隆的官方参考仓库，目的是研究架构、核对调用接口和复现实验。项目自身代码必须写在 `src` 或 `scripts`，不得直接修改这里的源码。

```text
third_party/
├── ai_scientists/       # AI Scientist v1/v2、Robin、Finch
├── frameworks/          # AgentScope、Qwen-Agent
└── anomaly_detection/   # PatchCore、AnomalyDINO、SubspaceAD
```

更新第三方源码时运行 `scripts/sync-third-party.ps1`，并检查 `manifest.lock.json`。各项目仍适用其自身 LICENSE；浅克隆不表示其许可证与本项目许可证相同。
