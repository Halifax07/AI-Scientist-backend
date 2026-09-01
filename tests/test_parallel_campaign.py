import asyncio
import json

from fastapi.testclient import TestClient

from fsad_scientist.agents.mock_runtime import MockScientistRuntime
from fsad_scientist.api.app import create_app
from fsad_scientist.config import Settings
from fsad_scientist.datasets.models import DatasetManifest
from fsad_scientist.domain.enums import ResearchStage, RunStatus
from fsad_scientist.domain.models import ComputeBudget, HypothesisRanking, ProjectSpec
from fsad_scientist.repository import JsonProjectRepository
from fsad_scientist.workflow import ResearchWorkflow


def fixture_manifest() -> DatasetManifest:
    return DatasetManifest(
        dataset="MVTec AD",
        root="C:/fixture/mvtec",
        categories=["bottle", "carpet", "capsule", "cable", "transistor"],
        files=[],
        counts={"files": 0},
        digest="b" * 64,
    )


def test_ranking_is_the_only_gate_before_parallel_preregistration(tmp_path):
    workflow = ResearchWorkflow(
        repository=JsonProjectRepository(tmp_path / "ledger"),
        runtime=MockScientistRuntime(),
    )
    project = workflow.create_project(
        ProjectSpec(budget=ComputeBudget(max_experiments=240))
    )
    project = asyncio.run(workflow.advance_to_hypothesis_ranking(project.id))
    assert project.stage == ResearchStage.HYPOTHESES_PROPOSED

    selected = project.hypotheses[:2]
    rankings = [
        HypothesisRanking(
            hypothesis_id=hypothesis.id,
            selected=hypothesis in selected,
            priority=index + 1,
            score=90 - index,
        )
        for index, hypothesis in enumerate(project.hypotheses)
    ]
    project = asyncio.run(workflow.rank_hypotheses(project.id, rankings=rankings))

    assert project.stage == ResearchStage.AWAITING_EXPERIMENT_APPROVAL
    assert project.experiment_plan is not None
    assert project.experiment_plan.hypothesis_ids == [item.id for item in selected]
    assert [item.user_selected for item in project.hypotheses[:2]] == [True, True]
    assert all(item.user_selected is False for item in project.hypotheses[2:])


def test_ranking_automatically_registers_custom_selected_strategy(tmp_path):
    workflow = ResearchWorkflow(
        repository=JsonProjectRepository(tmp_path / "ledger"),
        runtime=MockScientistRuntime(),
    )
    project = workflow.create_project(ProjectSpec())
    project = asyncio.run(workflow.advance_to_hypothesis_ranking(project.id))
    selected = project.hypotheses[-1]
    rankings = [
        HypothesisRanking(
            hypothesis_id=hypothesis.id,
            selected=hypothesis.id == selected.id,
            priority=1 if hypothesis.id == selected.id else 99,
            score=95 if hypothesis.id == selected.id else 10,
        )
        for hypothesis in project.hypotheses
    ]

    project = asyncio.run(workflow.rank_hypotheses(project.id, rankings=rankings))

    assert project.stage == ResearchStage.AWAITING_EXPERIMENT_APPROVAL
    assert project.experiment_plan is not None
    assert project.experiment_plan.hypothesis_ids == [selected.id]
    generated = [
        item
        for item in project.method_implementations
        if item.hypothesis_id == selected.id and item.kind == "selection_strategy"
    ]
    assert generated
    assert all(item.status == "validated" for item in generated)


def test_parallel_stream_executes_selected_rounds_and_persists_events(tmp_path):
    artifact_root = tmp_path / "artifacts"
    storage_root = tmp_path / "storage"
    artifact_root.mkdir()
    settings = Settings(
        runtime="mock",
        live_evidence=False,
        artifact_root=str(artifact_root),
        storage_root=str(storage_root),
    )
    app = create_app(
        settings=settings,
        storage_path=storage_root,
        runtime=MockScientistRuntime(),
    )
    workflow = app.state.workflow
    project = workflow.create_project(
        ProjectSpec(budget=ComputeBudget(max_experiments=240))
    )
    project = asyncio.run(workflow.advance_to_hypothesis_ranking(project.id))
    rankings = [
        HypothesisRanking(
            hypothesis_id=hypothesis.id,
            selected=index < 2,
            priority=index + 1,
            score=80 - index,
        )
        for index, hypothesis in enumerate(project.hypotheses)
    ]
    project = asyncio.run(workflow.rank_hypotheses(project.id, rankings=rankings))
    manifest = fixture_manifest()
    manifest_path = artifact_root / "datasets" / "fixture.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(manifest.model_dump_json(), encoding="utf-8")
    project = workflow.attach_dataset_audit(
        project.id,
        manifest=manifest,
        manifest_path=str(manifest_path.resolve()),
    )

    client = TestClient(app)
    started = client.post(
        f"/api/v1/projects/{project.id}/experiment-campaign/auto-start",
        json={
            "dataset_manifest_path": str(manifest_path.resolve()),
            "hypothesis_id": project.experiment_plan.hypothesis_ids[0],
            "selected_hypothesis_ids": project.experiment_plan.hypothesis_ids,
            "device": "cpu",
        },
    )
    assert started.status_code == 200
    queued = started.json()
    assert queued["experiment_campaign"]["execution_mode"] == "parallel"
    assert len(queued["experiment_campaign"]["rounds"]) == 2
    assert len(queued["runs"]) == 12

    first_round_ids = queued["experiment_campaign"]["rounds"][0]["run_ids"]
    streamed = client.post(
        f"/api/v1/projects/{project.id}/experiment-campaign/execute-stream",
        json={"max_parallel_runs": 2, "run_ids": first_round_ids},
    )
    assert streamed.status_code == 200
    frames = [
        json.loads(line[6:])
        for line in streamed.text.splitlines()
        if line.startswith("data: ")
    ]
    event_types = [frame["event_type"] for frame in frames]
    assert event_types[0] == "campaign_started"
    assert "run_started" in event_types
    assert "round_ready" in event_types
    assert "batch_completed" in event_types
    assert event_types[-1] == "stream_completed"

    partial = workflow.repository.get(project.id)
    assert partial.experiment_campaign is not None
    assert partial.experiment_campaign.status == "active"
    assert partial.experiment_campaign.rounds[0].result_summary["terminal_runs"] == 6
    assert partial.experiment_campaign.rounds[0].result_summary["round_pair_count"] == 3

    streamed_remaining = client.post(
        f"/api/v1/projects/{project.id}/experiment-campaign/execute-stream",
        json={"max_parallel_runs": 2},
    )
    assert streamed_remaining.status_code == 200
    remaining_frames = [
        json.loads(line[6:])
        for line in streamed_remaining.text.splitlines()
        if line.startswith("data: ")
    ]
    remaining_event_types = [frame["event_type"] for frame in remaining_frames]
    assert "round_completed" in remaining_event_types
    assert "campaign_completed" in remaining_event_types
    assert "results_locked" in remaining_event_types
    assert "statistics_completed" in remaining_event_types
    assert remaining_event_types[-1] == "stream_completed"

    final = workflow.repository.get(project.id)
    assert final.experiment_campaign is None
    assert final.experiment_campaign_history
    assert final.experiment_campaign_history[-1].status == "completed"
    assert all(
        run.status in {RunStatus.SUCCEEDED, RunStatus.FAILED}
        for run in final.runs
        if run.round_id is not None
    )
    assert len(final.experiment_progress) == len(frames) + len(remaining_frames)
