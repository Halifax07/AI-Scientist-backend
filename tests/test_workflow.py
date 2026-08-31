import asyncio

import pytest

from fsad_scientist.agents.mock_runtime import MockScientistRuntime
from fsad_scientist.domain.enums import HypothesisStatus, ResearchStage
from fsad_scientist.domain.models import ComputeBudget, ProjectSpec
from fsad_scientist.repository import JsonProjectRepository
from fsad_scientist.workflow import (
    ApprovalRequiredError,
    InvalidTransitionError,
    ResearchWorkflow,
)


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


def test_review_registers_an_image_metric_core_before_method_generation(tmp_path):
    workflow = ResearchWorkflow(
        repository=JsonProjectRepository(tmp_path / "ledger"),
        runtime=MockScientistRuntime(),
    )
    project = workflow.create_project(ProjectSpec())
    while project.stage != ResearchStage.HYPOTHESES_REVIEWED:
        project = run(workflow.advance(project.id))

    assert any(
        hypothesis.analysis_contract is not None
        and hypothesis.analysis_contract.kind == "selection_main_effect"
        and hypothesis.analysis_contract.metric == "image_auroc"
        for hypothesis in project.hypotheses
    )


def test_autonomous_discovery_reaches_human_gate(tmp_path):
    workflow = build_workflow(tmp_path)
    project = advance_to_approval(workflow)

    assert project.gaps
    assert project.hypotheses
    assert project.experiment_plan is not None
    assert len(project.experiment_plan.hypothesis_ids) >= 1
    assert set(project.experiment_plan.hypothesis_contracts) == set(
        project.experiment_plan.hypothesis_ids
    )
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
    assert {run.hypothesis_id for run in approved.runs} == set(
        approved.experiment_plan.hypothesis_ids
    )
    assert all(
        run.selection_strategy in {"random", "k_center"}
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


def test_legacy_multi_hypothesis_plan_requires_review_after_scoping(tmp_path):
    workflow = build_workflow(tmp_path)
    project = advance_to_approval(workflow)
    assert project.experiment_plan is not None
    primary_hypothesis_id = project.experiment_plan.hypothesis_ids[0]
    secondary = next(
        hypothesis
        for hypothesis in project.hypotheses
        if hypothesis.id != primary_hypothesis_id
        and hypothesis.analysis_contract is not None
    )
    secondary.status = HypothesisStatus.SHORTLISTED
    secondary.analysis_contract = secondary.analysis_contract.model_copy(
        update={
            "kind": "selection_main_effect",
            "treatment": "High Compression Ratio (Pool Size < 50)",
            "control": "Low Compression Ratio (Pool Size > 200)",
        }
    )
    project.experiment_plan.hypothesis_ids.append(secondary.id)
    project.experiment_plan.hypothesis_contracts[secondary.id] = (
        secondary.analysis_contract.model_copy(deep=True)
    )
    workflow.repository.save(project)

    with pytest.raises(InvalidTransitionError, match="没有已注册实现"):
        workflow.approve_experiment_plan(project.id, approved_by="test-reviewer")

    migrated = workflow.repository.get(project.id)
    assert migrated.stage == ResearchStage.AWAITING_EXPERIMENT_APPROVAL
    assert migrated.experiment_plan is not None
    assert primary_hypothesis_id in migrated.experiment_plan.hypothesis_ids
    assert secondary.id in migrated.experiment_plan.hypothesis_ids
    assert set(migrated.experiment_plan.hypothesis_contracts) == set(
        migrated.experiment_plan.hypothesis_ids
    )
    assert not migrated.runs


def test_scoping_skips_unregistered_dynamic_strategy_for_builtin_core(tmp_path):
    workflow = build_workflow(tmp_path)
    project = workflow.create_project(ProjectSpec())
    while project.stage != ResearchStage.HYPOTHESES_REVIEWED:
        project = run(workflow.advance(project.id))

    query = next(
        hypothesis
        for hypothesis in project.hypotheses
        if hypothesis.analysis_contract is not None
        and hypothesis.analysis_contract.kind == "query_adaptation"
    )
    core = next(
        hypothesis
        for hypothesis in project.hypotheses
        if hypothesis.analysis_contract is not None
        and hypothesis.analysis_contract.kind == "selection_main_effect"
    )
    assert query.analysis_contract is not None
    query.analysis_contract = query.analysis_contract.model_copy(
        update={
            "treatment": "Dynamic Density-based Denoising",
            "control": "random",
        }
    )
    query.status = HypothesisStatus.SHORTLISTED
    core.status = HypothesisStatus.SHORTLISTED
    project.hypotheses = [
        query,
        core,
        *[
            hypothesis
            for hypothesis in project.hypotheses
            if hypothesis.id not in {query.id, core.id}
        ],
    ]
    workflow.repository.save(project)

    preregistered = run(workflow.advance(project.id))
    assert preregistered.experiment_plan is not None
    assert core.id in preregistered.experiment_plan.hypothesis_ids
    assert query.id not in preregistered.experiment_plan.hypothesis_ids

    approved = workflow.approve_experiment_plan(
        project.id, approved_by="test-reviewer"
    )
    assert core.id in {run.hypothesis_id for run in approved.runs}
    assert query.id not in {run.hypothesis_id for run in approved.runs}


def test_scoping_prefers_generated_dynamic_strategy_over_builtin_core(tmp_path):
    workflow = build_workflow(tmp_path)
    project = workflow.create_project(ProjectSpec())
    while project.stage != ResearchStage.HYPOTHESES_REVIEWED:
        project = run(workflow.advance(project.id))

    query = next(
        hypothesis
        for hypothesis in project.hypotheses
        if hypothesis.analysis_contract is not None
        and hypothesis.analysis_contract.kind == "query_adaptation"
    )
    core = next(
        hypothesis
        for hypothesis in project.hypotheses
        if hypothesis.analysis_contract is not None
        and hypothesis.analysis_contract.kind == "selection_main_effect"
    )
    assert query.analysis_contract is not None
    query.analysis_contract = query.analysis_contract.model_copy(
        update={
            "treatment": "Dynamic Density-based Denoising",
            "control": "random",
        }
    )
    query.status = HypothesisStatus.SHORTLISTED
    core.status = HypothesisStatus.SHORTLISTED
    project.hypotheses = [
        query,
        core,
        *[
            hypothesis
            for hypothesis in project.hypotheses
            if hypothesis.id not in {query.id, core.id}
        ],
    ]
    workflow.repository.save(project)

    generated = run(
        workflow.implement_experiment_method(project.id, hypothesis_id=query.id)
    )
    implementation = next(
        item
        for item in generated.method_implementations
        if item.hypothesis_id == query.id and item.kind == "selection_strategy"
    )
    assert implementation.name == "dynamic_density_based_denoising"
    preregistered = run(workflow.advance(project.id))
    assert preregistered.experiment_plan is not None
    assert query.id in preregistered.experiment_plan.hypothesis_ids
    assert core.id in preregistered.experiment_plan.hypothesis_ids

    approved = workflow.approve_experiment_plan(
        project.id, approved_by="test-reviewer"
    )
    assert query.id in {run.hypothesis_id for run in approved.runs}
    assert core.id in {run.hypothesis_id for run in approved.runs}


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
    # 配对数不足时发现被标记为 not_tested 而非 inconclusive，不会触发修订循环
    assert any(item.claim_verdict == "not_tested" for item in project.findings)

    # 当所有发现都是 not_tested 时，_should_revise() 返回 False，流程进入创新审查阶段
    project = run(workflow.advance(project.id))
    assert project.stage == ResearchStage.INNOVATION_REVIEWED
    assert project.research_cycle == 1  # research_cycle 不会增加，因为没有修订
    assert project.experiment_plan is not None  # 实验计划保留
    assert project.experiment_campaign is None  # campaign 已完成
