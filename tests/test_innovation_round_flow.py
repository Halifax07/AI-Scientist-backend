import asyncio

import pytest

from fsad_scientist.agents.mock_runtime import MockScientistRuntime
from fsad_scientist.datasets.models import DatasetManifest
from fsad_scientist.domain.enums import ResearchStage
from fsad_scientist.domain.models import ComputeBudget, ProjectSpec
from fsad_scientist.repository import JsonProjectRepository
from fsad_scientist.workflow import InvalidTransitionError, ResearchWorkflow


def run(coro):
    return asyncio.run(coro)


def build_project(tmp_path):
    workflow = ResearchWorkflow(
        repository=JsonProjectRepository(tmp_path / "ledger"),
        runtime=MockScientistRuntime(),
    )
    project = workflow.create_project(
        ProjectSpec(budget=ComputeBudget(max_experiments=24))
    )
    while project.stage != ResearchStage.AWAITING_EXPERIMENT_APPROVAL:
        project = run(workflow.advance(project.id))
    project = workflow.approve_experiment_plan(project.id, approved_by="test")
    dataset = DatasetManifest(
        dataset="MVTec AD",
        root="C:/fixture/mvtec",
        categories=["bottle", "carpet", "capsule"],
        files=[],
        counts={"files": 0},
        digest="a" * 64,
    )
    project = workflow.attach_dataset_audit(
        project.id,
        manifest=dataset,
        manifest_path=str(tmp_path / "dataset.json"),
    )
    hypothesis_id = project.experiment_plan.hypothesis_ids[0]  # type: ignore[union-attr]
    return workflow, workflow.initialize_experiment_campaign(
        project.id,
        dataset=dataset,
        hypothesis_id=hypothesis_id,
        max_runs=24,
    )


def finish_queued(workflow, project):
    campaign = project.experiment_campaign
    assert campaign is not None
    current = campaign.rounds[-1]
    for run_record in list(project.runs):
        if run_record.id not in current.run_ids or run_record.status.value != "queued":
            continue
        project = workflow.record_run_result(
            project.id,
            run_id=run_record.id,
            metrics={"image_auroc": 0.8},
            success=True,
            verified=True,
            result_source="synthetic_test",
        )
    return project


def test_round_has_one_innovation_and_three_internal_iterations(tmp_path):
    workflow, project = build_project(tmp_path)
    campaign = project.experiment_campaign
    assert campaign is not None
    first = campaign.rounds[-1]
    assert first.hypothesis_id == campaign.hypothesis_id
    assert first.iteration_target == 3
    assert len(first.run_ids) == 2

    project = finish_queued(workflow, project)
    assert project.experiment_campaign.status == "awaiting_guidance"
    with pytest.raises(InvalidTransitionError):
        run(workflow.review_experiment_round(project.id))

    project = run(
        workflow.review_experiment_round(project.id, user_guidance="扩大类别覆盖")
    )
    current = project.experiment_campaign.rounds[-1]
    assert current.hypothesis_id == first.hypothesis_id
    assert current.guidance_received is True
    assert len(current.run_ids) == 6
    assert sum(item.scope == "round_iteration" for item in project.guidance_records) == 1

    project = finish_queued(workflow, project)
    assert project.experiment_campaign.status == "awaiting_feedback"
    project = run(workflow.review_experiment_round(project.id))
    campaign = project.experiment_campaign
    assert campaign is not None
    if campaign.status != "completed":
        assert campaign.current_round == 2
        assert campaign.rounds[-1].hypothesis_id != first.hypothesis_id
