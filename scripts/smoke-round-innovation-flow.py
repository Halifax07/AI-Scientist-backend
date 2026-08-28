"""Small mock-runtime smoke test for the innovation-per-round loop."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from fsad_scientist.agents.mock_runtime import MockScientistRuntime
from fsad_scientist.datasets.models import DatasetManifest
from fsad_scientist.domain.enums import ResearchStage
from fsad_scientist.domain.models import ComputeBudget, ProjectSpec
from fsad_scientist.repository import JsonProjectRepository
from fsad_scientist.workflow import ResearchWorkflow


def manifest() -> DatasetManifest:
    return DatasetManifest(
        dataset="MVTec AD",
        root="C:/fixture/mvtec",
        categories=["bottle", "carpet", "capsule"],
        files=[],
        counts={"files": 0},
        digest="a" * 64,
    )


def main() -> None:
    root = Path(tempfile.mkdtemp())
    workflow = ResearchWorkflow(
        repository=JsonProjectRepository(root / "ledger"),
        runtime=MockScientistRuntime(),
    )
    project = workflow.create_project(
        ProjectSpec(budget=ComputeBudget(max_experiments=24))
    )
    while project.stage != ResearchStage.AWAITING_EXPERIMENT_APPROVAL:
        project = asyncio.run(workflow.advance(project.id))
    project = workflow.approve_experiment_plan(project.id, approved_by="smoke")
    project = workflow.attach_dataset_audit(
        project.id, manifest=manifest(), manifest_path=str(root / "dataset.json")
    )
    hypothesis_id = project.experiment_plan.hypothesis_ids[0]  # type: ignore[union-attr]
    project = workflow.initialize_experiment_campaign(
        project.id, dataset=manifest(), hypothesis_id=hypothesis_id, max_runs=24
    )
    campaign = project.experiment_campaign
    assert campaign is not None
    assert len(campaign.hypothesis_ids) >= 1
    assert len(campaign.rounds[-1].run_ids) == 2
    for run in list(project.runs):
        if run.round_id:
            project = workflow.record_run_result(
                project.id,
                run_id=run.id,
                metrics={"image_auroc": 0.8},
                success=True,
                verified=True,
                result_source="synthetic_test",
            )
    assert project.experiment_campaign.status == "awaiting_guidance"
    project = asyncio.run(
        workflow.review_experiment_round(project.id, user_guidance="扩大类别覆盖")
    )
    assert len(project.experiment_campaign.rounds[-1].run_ids) == 6  # type: ignore[union-attr]
    print("innovation-round smoke test passed")


if __name__ == "__main__":
    main()
