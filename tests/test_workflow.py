import asyncio

import pytest

from fsad_scientist.agents.mock_runtime import MockScientistRuntime
from fsad_scientist.domain.enums import ResearchStage
from fsad_scientist.domain.models import ComputeBudget, ProjectSpec
from fsad_scientist.repository import JsonProjectRepository
from fsad_scientist.workflow import ApprovalRequiredError, ResearchWorkflow


def run(coro):
    return asyncio.run(coro)


def build_workflow(tmp_path) -> ResearchWorkflow:
    return ResearchWorkflow(
        repository=JsonProjectRepository(tmp_path / "ledger"),
        runtime=MockScientistRuntime(),
    )


def advance_to_approval(workflow: ResearchWorkflow):
    project = workflow.create_project(ProjectSpec())
    while project.stage != ResearchStage.AWAITING_EXPERIMENT_APPROVAL:
        project = run(workflow.advance(project.id))
    return project


def test_autonomous_discovery_reaches_human_gate(tmp_path):
    workflow = build_workflow(tmp_path)
    project = advance_to_approval(workflow)

    assert project.gaps
    assert project.hypotheses
    assert project.experiment_plan is not None
    assert project.experiment_plan.approved is False
    assert project.next_action == "human_approve_preregistered_plan"
    assert any(hypothesis.null_hypothesis for hypothesis in project.hypotheses)

    with pytest.raises(ApprovalRequiredError):
        run(workflow.advance(project.id))


def test_approval_queues_a_bounded_feasibility_batch(tmp_path):
    workflow = build_workflow(tmp_path)
    project = workflow.create_project(
        ProjectSpec(budget=ComputeBudget(max_experiments=200))
    )
    while project.stage != ResearchStage.AWAITING_EXPERIMENT_APPROVAL:
        project = run(workflow.advance(project.id))

    approved = workflow.approve_experiment_plan(project.id, approved_by="test-reviewer")

    assert approved.stage == ResearchStage.EXPERIMENTS_QUEUED
    assert 0 < len(approved.runs) <= 200
    assert all(
        run.selection_strategy == "random"
        for run in approved.runs
        if run.protocol == "strict_k_shot"
    )
    pool_strategies = {
        run.selection_strategy for run in approved.runs if run.protocol.startswith("pool_")
    }
    assert pool_strategies <= {
        "random",
        "k_center",
    }


def test_inconclusive_real_cycle_revises_hypothesis_without_losing_history(tmp_path):
    workflow = build_workflow(tmp_path)
    project = workflow.create_project(ProjectSpec(budget=ComputeBudget(max_experiments=2)))
    while project.stage != ResearchStage.AWAITING_EXPERIMENT_APPROVAL:
        project = run(workflow.advance(project.id))
    project = workflow.approve_experiment_plan(project.id, approved_by="test-reviewer")

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
    assert any(item.claim_verdict == "inconclusive" for item in project.findings)

    project = run(workflow.advance(project.id))

    assert project.stage == ResearchStage.HYPOTHESES_PROPOSED
    assert project.research_cycle == 2
    assert project.hypothesis_history
    assert project.finding_history
    assert project.hypotheses[0].parent_hypothesis_id is not None
    assert project.experiment_plan is None
