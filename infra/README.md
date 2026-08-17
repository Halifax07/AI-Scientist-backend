# Infrastructure

当前 compose 只启动 API、PostgreSQL 和 Redis。第一周 Research Ledger 默认仍使用本地 JSON，便于流程开发和测试；完成数据库迁移后再启用 PostgreSQL。MLflow、MinIO 和 GPU Worker 将在实验执行阶段加入，避免本周引入未使用的服务。

```powershell
Set-Location 'E:\揭榜挂帅\AI Scientist\infra'
docker compose up --build
```

