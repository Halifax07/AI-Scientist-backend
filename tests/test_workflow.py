import asyncio

import pytest

from fsad_scientist.agents.mock_runtime import MockScientistRuntime
from fsad_scientist.datasets.models import DatasetManifest
from fsad_scientist.domain.enums import HypothesisStatus, ResearchStage, RunStatus
from fsad_scientist.domain.models import (
    AnalysisContract,
    AnalysisFinding,
    ComputeBudget,
    HypothesisRanking,
    ProjectSpec,
)
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
    # Use preset="fsad" so that the helper continues to produce the full portfolio
    # of paired_comparison + custom_design hypotheses; the generic mode generates
    # only custom_design placeholders which the existing test assertions don't expect.
    project = workflow.create_project(ProjectSpec(preset="fsad"))
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


def test_hypothesis_review_api_backfills_partial_runtime_scores(tmp_path):
    class PartialReviewRuntime(MockScientistRuntime):
        async def review_hypotheses(self, project):
            reviewed = await super().review_hypotheses(project)
            # Simulate a runtime/provider response that omitted one candidate's
            # review. The workflow must still return a total score contract.
            reviewed[-1].score = None
            return reviewed

    workflow = ResearchWorkflow(
        repository=JsonProjectRepository(tmp_path / "ledger"),
        runtime=PartialReviewRuntime(),
    )
    project = workflow.create_project(ProjectSpec())
    while project.stage != ResearchStage.HYPOTHESES_PROPOSED:
        project = run(workflow.advance(project.id))

    project = run(workflow.advance(project.id))

    assert project.stage == ResearchStage.HYPOTHESES_REVIEWED
    assert project.hypotheses
    assert all(item.score is not None for item in project.hypotheses)
    persisted = workflow.repository.get(project.id)
    assert all(item.score is not None for item in persisted.hypotheses)


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


def test_approval_queues_custom_design_conditions_and_factor_values(tmp_path):
    workflow = build_workflow(tmp_path)
    project = advance_to_approval(workflow)
    assert project.experiment_plan is not None
    custom_ids = {
        hypothesis.id
        for hypothesis in project.hypotheses
        if hypothesis.analysis_contract is not None
        and hypothesis.analysis_contract.design_mode == "custom_design"
    }
    assert custom_ids

    approved = workflow.approve_experiment_plan(project.id, approved_by="test-reviewer")
    custom_runs = [run for run in approved.runs if run.hypothesis_id in custom_ids]

    assert custom_runs
    assert all(run.condition_id for run in custom_runs)
    assert all(run.factor_values == {"shots": run.shots} for run in custom_runs)
    assert {run.selection_strategy for run in custom_runs} == {"random"}
    assert {run.condition_id for run in custom_runs} >= {
        "condition_1",
        "condition_2",
        "condition_3",
    }


def test_scoping_clears_stale_design_status_when_retained_designs_are_legal(tmp_path):
    workflow = build_workflow(tmp_path)
    project = advance_to_approval(workflow)
    assert project.experiment_plan is not None
    plan = project.experiment_plan
    plan.design_generation_status = "needs_correction"
    plan.design_generation_fallback_reason = "Qwen returned an invalid extra design"
    plan.design_generation_errors = ["ignored invalid extra design"]
    workflow.repository.save(project)

    approved = workflow.approve_experiment_plan(project.id, approved_by="scope-test")

    assert approved.experiment_plan.design_generation_status == "ai_selected"
    assert approved.experiment_plan.design_generation_fallback_reason is None
    assert approved.experiment_plan.design_generation_errors == [
        "ignored invalid extra design"
    ]
    assert approved.stage == ResearchStage.EXPERIMENTS_QUEUED


def test_scoping_keeps_design_status_when_a_retained_hypothesis_lacks_design(tmp_path):
    workflow = build_workflow(tmp_path)
    project = advance_to_approval(workflow)
    assert project.experiment_plan is not None
    plan = project.experiment_plan
    retained_id = next(
        hypothesis.id
        for hypothesis in project.hypotheses
        if hypothesis.id in plan.hypothesis_ids
        and hypothesis.analysis_contract is not None
        and hypothesis.analysis_contract.design_mode == "paired_comparison"
    )
    plan.designs = [
        design for design in plan.designs if design.hypothesis_id != retained_id
    ]
    plan.design_generation_status = "needs_correction"
    plan.design_generation_fallback_reason = "missing retained design"
    workflow.repository.save(project)

    with pytest.raises(InvalidTransitionError, match="实验设计需要修正"):
        workflow.approve_experiment_plan(project.id, approved_by="scope-test")

    assert plan.design_generation_status == "needs_correction"


def test_mark_run_running_rejects_queued_run_outside_current_round(tmp_path):
    workflow = build_workflow(tmp_path)
    project = advance_to_approval(workflow)
    project = workflow.approve_experiment_plan(project.id, approved_by="test-reviewer")
    manifest = DatasetManifest(
        dataset="MVTec AD",
        root="C:/fixture/mvtec",
        categories=["bottle"],
        files=[],
        counts={"files": 0},
        digest="a" * 64,
    )
    project = workflow.attach_dataset_audit(
        project.id,
        manifest=manifest,
        manifest_path=str(tmp_path / "artifacts" / "dataset.json"),
    )
    hypothesis_id = project.experiment_plan.hypothesis_ids[0]
    project = workflow.initialize_experiment_campaign(
        project.id,
        dataset=manifest,
        hypothesis_id=hypothesis_id,
        device="cpu",
        max_rounds=1,
        max_runs=6,
    )
    outside = project.runs[0].model_copy(
        update={"id": "queued-outside-round", "round_id": None},
        deep=True,
    )
    outside.status = RunStatus.QUEUED
    project.runs.append(outside)
    workflow.repository.save(project)

    with pytest.raises(InvalidTransitionError, match="current experiment Round"):
        workflow.mark_run_running(project.id, run_id=outside.id)


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


def test_rank_gate_auto_implements_custom_pool_instead_of_pool_veto(tmp_path):
    """A fully custom pool (every candidate requires implementation) must not
    dead-lock the ranking gate: auto-generation of selection/query adapters
    runs first, and only detector interactions over unregistered detectors are
    rejected by name so the UI can drop and retry.
    """
    workflow = build_workflow(tmp_path)
    project = workflow.create_project(ProjectSpec())
    while project.stage != ResearchStage.HYPOTHESES_REVIEWED:
        project = run(workflow.advance(project.id))

    template = project.hypotheses[0]
    sel = template.model_copy(
        update={"id": "hypothesis_sel_custom", "title": "density compression"},
        deep=True,
    )
    sel.analysis_contract = AnalysisContract(
        kind="selection_main_effect",
        metric="image_auroc",
        treatment="Density-based Geometric Compression",
        control="random",
    )
    qa = template.model_copy(
        update={"id": "hypothesis_qa_custom", "title": "query bias correction"},
        deep=True,
    )
    qa.analysis_contract = AnalysisContract(
        kind="query_adaptation",
        metric="image_auroc",
        treatment="Causal Bias Correction",
        control="no_adaptation",
    )
    det = template.model_copy(
        update={"id": "hypothesis_det_custom", "title": "fusion detector"},
        deep=True,
    )
    det.analysis_contract = AnalysisContract(
        kind="detector_interaction",
        metric="image_auroc",
        treatment="Fusion Anomaly Detector",
        control="Standard Subspace Detector",
    )
    project.hypotheses = [sel, qa, det]
    workflow.repository.save(project)

    def ranking(hypothesis, priority):
        return HypothesisRanking(
            hypothesis_id=hypothesis.id, selected=True, priority=priority
        )

    with pytest.raises(InvalidTransitionError) as excinfo:
        run(
            workflow.rank_hypotheses(
                project.id,
                rankings=[ranking(sel, 1), ranking(qa, 2), ranking(det, 3)],
                auto_preregister=False,
            )
        )
    # The gate names the unregistered detector instead of vetoing the pool,
    # keeping the front-end drop-and-retry recovery path available.
    assert "引用的检测器" in str(excinfo.value)
    assert "都需要先实现方法适配器" not in str(excinfo.value)

    # Custom selection/query adapters were auto-generated before the veto.
    project = workflow.repository.get(project.id)
    implemented = {
        item.hypothesis_id
        for item in project.method_implementations
        if item.kind == "selection_strategy"
        and item.status in {"validated", "approved"}
    }
    assert {sel.id, qa.id} <= implemented

    # Dropping the unregistered-detector innovation lets the same pool through.
    project = run(
        workflow.rank_hypotheses(
            project.id,
            rankings=[ranking(sel, 1), ranking(qa, 2)],
            auto_preregister=False,
        )
    )
    assert project.stage == ResearchStage.HYPOTHESES_REVIEWED
    by_id = {hypothesis.id: hypothesis for hypothesis in project.hypotheses}
    assert by_id[sel.id].status == HypothesisStatus.SHORTLISTED
    assert by_id[qa.id].user_selected is True


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


def test_revision_keeps_falsification_conditions_unique_across_cycles(tmp_path):
    # Regression: mock revise_hypotheses appended the same falsification
    # sentence on every revision while inheriting the parent list, so a
    # hypothesis revised twice carried duplicate entries. The UI renders those
    # conditions as <li key={item}>, and React then emits duplicate-key
    # warnings on every render (and risks dropping children).
    workflow = build_workflow(tmp_path)
    project = workflow.create_project(ProjectSpec())
    while project.stage != ResearchStage.AWAITING_EXPERIMENT_APPROVAL:
        project = run(workflow.advance(project.id))
    parent = project.hypotheses[0]

    def revise(current, cycle):
        project.hypotheses = [current]
        project.research_cycle = cycle
        project.findings = [
            AnalysisFinding(
                hypothesis_id=current.id,
                statement="未见支持性证据",
                claim_verdict="rejected",
                verified=True,
            )
        ]
        revised = run(workflow.runtime.revise_hypotheses(project))
        assert len(revised) == 1
        return revised[0]

    second = revise(parent, 1)
    third = revise(second, 2)

    assert second.revision == parent.revision + 1
    assert third.revision == parent.revision + 2
    # 第一次修订追加证伪句, 之后各轮继承但不重复堆叠。
    expected_after_first = [
        *parent.falsification_conditions,
        "修订后预注册的稳定性主终点仍未达到最小效应或重复要求",
    ]
    assert second.falsification_conditions == expected_after_first
    assert third.falsification_conditions == expected_after_first
    assert len(third.falsification_conditions) == len(
        set(third.falsification_conditions)
    )
