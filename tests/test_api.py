import asyncio

from fastapi.testclient import TestClient

from fsad_scientist.agents.agentscope_client import AgentScopeUnavailableError
from fsad_scientist.agents.mock_runtime import MockScientistRuntime
from fsad_scientist.api.app import create_app
from fsad_scientist.config import Settings
from fsad_scientist.domain.enums import ResearchStage
from fsad_scientist.domain.models import ComputeBudget, ProjectSpec


def run(coro):
    return asyncio.run(coro)


def test_health_and_project_creation(tmp_path):
    app = create_app(
        settings=Settings(runtime="mock"),
        storage_path=tmp_path / "api-ledger",
        runtime=MockScientistRuntime(),
    )
    client = TestClient(app)

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["runtime"] == "mock-scientist-runtime"

    created = client.post("/api/v1/projects/demo")
    assert created.status_code == 201
    project_id = created.json()["id"]

    advanced = client.post(f"/api/v1/projects/{project_id}/advance")
    assert advanced.status_code == 200
    assert advanced.json()["stage"] == "scope_formalized"


def test_cors_allows_both_loopback_frontend_origins(tmp_path):
    app = create_app(
        settings=Settings(runtime="mock"),
        storage_path=tmp_path / "cors-ledger",
        runtime=MockScientistRuntime(),
    )
    client = TestClient(app)

    for origin in ("http://localhost:5173", "http://127.0.0.1:5173"):
        response = client.options(
            "/health",
            headers={
                "Origin": origin,
                "Access-Control-Request-Method": "GET",
            },
        )

        assert response.status_code == 200
        assert response.headers["access-control-allow-origin"] == origin


def test_agent_runtime_failure_returns_readable_cors_error(tmp_path):
    class FailingRuntime(MockScientistRuntime):
        async def formalize_scope(self, project):
            raise AgentScopeUnavailableError("DashScope 账户欠费或余额不足，请充值后重试。")

    app = create_app(
        settings=Settings(runtime="mock"),
        storage_path=tmp_path / "runtime-error-ledger",
        runtime=FailingRuntime(),
    )
    client = TestClient(app)
    created = client.post("/api/v1/projects/demo").json()
    response = client.post(
        f"/api/v1/projects/{created['id']}/advance",
        headers={"Origin": "http://127.0.0.1:5173"},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "DashScope 账户欠费或余额不足，请充值后重试。"
    assert response.headers["access-control-allow-origin"] == "http://127.0.0.1:5173"


def test_next_research_cycle_endpoint_requires_and_records_guidance(tmp_path):
    app = create_app(
        settings=Settings(runtime="mock"),
        storage_path=tmp_path / "cycle-ledger",
        runtime=MockScientistRuntime(),
    )
    workflow = app.state.workflow
    project = workflow.create_project(
        ProjectSpec(budget=ComputeBudget(max_experiments=2))
    )
    while project.stage != ResearchStage.AWAITING_EXPERIMENT_APPROVAL:
        project = run(workflow.advance(project.id))
    project = workflow.approve_experiment_plan(project.id, approved_by="api-test")
    first, second = project.runs
    workflow.record_run_result(
        project.id,
        run_id=first.id,
        metrics={"image_auroc": 0.8},
        success=True,
        verified=True,
        result_source="synthetic_test",
    )
    workflow.record_run_result(
        project.id,
        run_id=second.id,
        metrics={},
        success=False,
        verified=False,
        result_source="synthetic_test",
    )
    project = workflow.finalize_results(project.id)
    project = run(workflow.advance(project.id))
    assert project.stage == ResearchStage.RESULTS_ANALYZED

    client = TestClient(app)
    missing = client.post(f"/api/v1/projects/{project.id}/research-cycles/next", json={})
    assert missing.status_code == 422

    response = client.post(
        f"/api/v1/projects/{project.id}/research-cycles/next",
        json={"user_guidance": "聚焦失败类别并缩小假设范围。"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["research_cycle"] == 2
    assert payload["guidance_records"][-1]["disposition"] == "applied"
    assert payload["guidance_records"][-1]["affected_ids"]
