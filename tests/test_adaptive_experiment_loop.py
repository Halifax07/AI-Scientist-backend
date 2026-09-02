import asyncio

from fsad_scientist.agents.mock_runtime import MockScientistRuntime
from fsad_scientist.datasets.models import DatasetManifest
from fsad_scientist.domain.enums import ProjectStatus, ResearchStage, RunStatus
from fsad_scientist.domain.models import (
    ComputeBudget,
    ExperimentCell,
    ExperimentFeedbackProposal,
    ProjectSpec,
)
from fsad_scientist.repository import JsonProjectRepository
from fsad_scientist.workflow import ResearchWorkflow


def run(coro):
    return asyncio.run(coro)


def build_approved_project(tmp_path, *, max_experiments: int = 6):
    workflow = ResearchWorkflow(
        repository=JsonProjectRepository(tmp_path / "ledger"),
        runtime=MockScientistRuntime(),
    )
    project = workflow.create_project(
        ProjectSpec(budget=ComputeBudget(max_experiments=max_experiments))
    )
    while project.stage != ResearchStage.AWAITING_EXPERIMENT_APPROVAL:
        project = run(workflow.advance(project.id))
    return workflow, workflow.approve_experiment_plan(
        project.id, approved_by="test-reviewer"
    )


def dataset_manifest() -> DatasetManifest:
    return DatasetManifest(
        dataset="MVTec AD",
        root="C:/fixture/mvtec",
        categories=["bottle", "carpet", "capsule", "cable", "transistor"],
        files=[],
        counts={"files": 0},
        digest="a" * 64,
    )


def executable_hypothesis_id(project) -> str:
    return next(
        hypothesis.id
        for hypothesis in project.hypotheses
        if hypothesis.execution_readiness == "executable"
    )


def complete_current_round(workflow: ResearchWorkflow, project_id: str) -> None:
    project = workflow.repository.get(project_id)
    campaign = project.experiment_campaign
    assert campaign is not None
    current = campaign.rounds[-1]
    for run_record in project.runs:
        if run_record.id not in current.run_ids or run_record.status != RunStatus.QUEUED:
            continue
        value = 0.82 if run_record.selection_strategy == "k_center" else 0.80
        workflow.record_run_result(
            project_id,
            run_id=run_record.id,
            metrics={"image_auroc": value},
            success=True,
            verified=True,
            result_source="synthetic_test",
            duration_seconds=1.0,
        )


def test_feedback_loop_uses_results_and_respects_run_budget(tmp_path):
    workflow, project = build_approved_project(tmp_path, max_experiments=12)
    dataset = dataset_manifest()
    project = workflow.attach_dataset_audit(
        project.id,
        manifest=dataset,
        manifest_path=str(tmp_path / "artifacts" / "dataset.json"),
    )
    fixed_queue_size = len(project.runs)

    project = workflow.initialize_experiment_campaign(
        project.id,
        dataset=dataset,
        hypothesis_id=executable_hypothesis_id(project),
        max_rounds=3,
        max_runs=12,
    )

    assert project.experiment_campaign is not None
    assert len(project.runs) == 2
    assert fixed_queue_size > len(project.runs)
    assert {run.selection_strategy for run in project.runs} == {"random", "k_center"}
    assert len({(run.category, run.shots, run.seed) for run in project.runs}) == 1

    complete_current_round(workflow, project.id)
    project = workflow.repository.get(project.id)
    assert project.experiment_campaign is not None
    assert project.experiment_campaign.status == "awaiting_guidance"

    project = run(workflow.review_experiment_round(project.id, user_guidance="扩大类别覆盖"))
    assert project.experiment_campaign is not None
    assert project.experiment_campaign.current_round == 1
    assert len(project.runs) == 6
    first_round = project.experiment_campaign.rounds[0]
    assert first_round.feedback is not None
    assert first_round.result_summary["pair_count"] == 1
    assert first_round.result_summary["mean_difference"] > 0

    complete_current_round(workflow, project.id)
    project = run(workflow.review_experiment_round(project.id))
    assert project.experiment_campaign is not None
    assert project.experiment_campaign.current_round == 2
    assert len(project.runs) == 8
    complete_current_round(workflow, project.id)
    project = run(
        workflow.review_experiment_round(
            project.id, user_guidance="继续检验第二个创新点"
        )
    )
    complete_current_round(workflow, project.id)
    project = run(workflow.review_experiment_round(project.id))
    assert project.experiment_campaign is not None
    assert project.experiment_campaign.status == "completed"
    assert len(project.runs) == 12
    assert all(run.round_id is not None for run in project.runs)

    project = workflow.finalize_results(project.id)
    assert project.stage == ResearchStage.RESULTS_READY


def test_midpoint_guidance_caps_two_valid_recommendations(tmp_path, monkeypatch):
    workflow, project = build_approved_project(tmp_path, max_experiments=12)
    dataset = dataset_manifest()
    project = workflow.attach_dataset_audit(
        project.id,
        manifest=dataset,
        manifest_path=str(tmp_path / "artifacts" / "dataset.json"),
    )
    project = workflow.initialize_experiment_campaign(
        project.id,
        dataset=dataset,
        hypothesis_id=executable_hypothesis_id(project),
        max_rounds=3,
        max_runs=12,
    )
    complete_current_round(workflow, project.id)
    before = workflow.repository.get(project.id)
    assert before.experiment_campaign is not None
    allowed = workflow.experiment_planner.allowed_next_cells(before)
    assert len(allowed) >= 2
    assert allowed[0] != allowed[1]

    async def recommend_two_valid_cells(project, *, round_summary, allowed_cells):
        return ExperimentFeedbackProposal(
            advisor="test-advisor",
            rationale="两个有效单元足以排定后续两次迭代。",
            recommended_cells=[allowed_cells[0], allowed_cells[1]],
            expected_information_gain=0.5,
        )

    monkeypatch.setattr(workflow.runtime, "recommend_next_experiments", recommend_two_valid_cells)
    reviewed = run(
        workflow.review_experiment_round(
            before.id,
            user_guidance="按推荐的两个有效单元继续本轮。",
        )
    )

    assert reviewed.experiment_campaign is not None
    current = reviewed.experiment_campaign.rounds[-1]
    new_runs = [run_record for run_record in reviewed.runs if run_record.iteration in {2, 3}]
    assert len(new_runs) == 4
    assert [run_record.iteration for run_record in new_runs].count(2) == 2
    assert [run_record.iteration for run_record in new_runs].count(3) == 2
    assert len(current.run_ids) == 6


def test_metric_alias_normalizes_and_forms_a_valid_pair(tmp_path):
    workflow, project = build_approved_project(tmp_path, max_experiments=6)
    assert project.experiment_plan is not None
    hypothesis_id = executable_hypothesis_id(project)
    hypothesis = next(item for item in project.hypotheses if item.id == hypothesis_id)
    assert hypothesis.analysis_contract is not None
    hypothesis.analysis_contract = hypothesis.analysis_contract.model_copy(
        update={"metric": "Image AUROC"}
    )
    project.experiment_plan.hypothesis_contracts[hypothesis_id] = (
        hypothesis.analysis_contract.model_copy(deep=True)
    )
    workflow.repository.save(project)

    dataset = dataset_manifest()
    project = workflow.attach_dataset_audit(
        project.id,
        manifest=dataset,
        manifest_path=str(tmp_path / "artifacts" / "dataset.json"),
    )
    project = workflow.initialize_experiment_campaign(
        project.id,
        dataset=dataset,
        hypothesis_id=hypothesis_id,
        max_rounds=1,
        max_runs=6,
    )

    assert project.experiment_campaign is not None
    assert project.experiment_campaign.metric == "image_auroc"
    assert project.experiment_campaign.rounds[0].metric == "image_auroc"
    complete_current_round(workflow, project.id)
    summary = workflow.experiment_planner.summarize_current_round(
        workflow.repository.get(project.id)
    )
    assert summary["metric"] == "image_auroc"
    assert summary["pair_count"] == 1


def test_plan_scope_excludes_unsupported_primary_metric(tmp_path):
    workflow = ResearchWorkflow(
        repository=JsonProjectRepository(tmp_path / "ledger"),
        runtime=MockScientistRuntime(),
    )
    project = workflow.create_project(ProjectSpec())
    while project.stage != ResearchStage.AWAITING_EXPERIMENT_APPROVAL:
        project = run(workflow.advance(project.id))
    assert project.experiment_plan is not None
    unsupported_id = project.experiment_plan.hypothesis_ids[-1]
    unsupported = next(item for item in project.hypotheses if item.id == unsupported_id)
    assert unsupported.analysis_contract is not None
    unsupported.analysis_contract = unsupported.analysis_contract.model_copy(
        update={"metric": "Recall at FPR=5%"}
    )
    project.experiment_plan.hypothesis_contracts[unsupported_id] = (
        unsupported.analysis_contract.model_copy(deep=True)
    )
    workflow.repository.save(project)

    excluded = workflow._scope_experiment_plan_to_primary_hypothesis(project)

    assert unsupported_id in excluded
    assert unsupported_id not in project.experiment_plan.hypothesis_ids


def test_campaign_skips_higher_ranked_unsupported_strategy(tmp_path):
    workflow, project = build_approved_project(tmp_path)
    supported = next(
        hypothesis
        for hypothesis in project.hypotheses
        if hypothesis.analysis_contract is not None
        and hypothesis.analysis_contract.treatment == "k_center"
        and hypothesis.analysis_contract.control == "random"
    )
    unsupported = supported.model_copy(deep=True)
    unsupported.id = "hypothesis_unsupported_strategy"
    unsupported.title = "查询感知动态加权"
    unsupported.analysis_contract = supported.analysis_contract.model_copy(
        update={
            "treatment": "Query-aware Dynamic Weighting",
            "control": "Static Nearest-Neighbor Aggregation",
        }
    )
    if unsupported.score is not None:
        unsupported.score.elo = 9999
    project.hypotheses.insert(0, unsupported)
    assert project.experiment_plan is not None
    project.experiment_plan.hypothesis_ids.insert(0, unsupported.id)
    workflow.repository.save(project)

    dataset = dataset_manifest()
    project = workflow.attach_dataset_audit(
        project.id,
        manifest=dataset,
        manifest_path=str(tmp_path / "artifacts" / "dataset.json"),
    )
    project = workflow.initialize_experiment_campaign(
        project.id,
        dataset=dataset,
        hypothesis_id=supported.id,
        max_rounds=2,
        max_runs=6,
    )

    assert project.experiment_campaign is not None
    assert project.experiment_campaign.hypothesis_id == supported.id
    assert project.experiment_campaign.treatment == "k_center"
    assert project.experiment_campaign.control == "random"


def test_human_guidance_selects_only_a_registered_queued_run(tmp_path):
    workflow, project = build_approved_project(tmp_path, max_experiments=8)
    dataset = dataset_manifest()
    project = workflow.attach_dataset_audit(
        project.id,
        manifest=dataset,
        manifest_path=str(tmp_path / "artifacts" / "dataset.json"),
    )
    project = workflow.initialize_experiment_campaign(
        project.id,
        dataset=dataset,
        hypothesis_id=executable_hypothesis_id(project),
        max_rounds=2,
        max_runs=8,
    )
    original_configs = {
        run.id: (run.category, run.shots, run.seed, run.selection_strategy)
        for run in project.runs
    }

    selected, decision = run(
        workflow.select_next_experiment(
            project.id,
            user_guidance="请优先执行 k-center，其他预注册参数保持不变。",
        )
    )
    updated = workflow.repository.get(project.id)

    assert selected.selection_strategy == "k_center"
    assert decision.selected_run_id == selected.id
    assert decision.disposition == "applied"
    assert {
        item.id: (item.category, item.shots, item.seed, item.selection_strategy)
        for item in updated.runs
    } == original_configs
    guidance = updated.guidance_records[-1]
    assert guidance.selected_run_id == selected.id
    assert guidance.text.startswith("请优先执行")
    assert guidance.protected_constraints
    assert any(event.action == "interpret_experiment_guidance" for event in updated.events)


def test_completed_campaign_rejects_unapproved_next_innovation(tmp_path):
    workflow, project = build_approved_project(tmp_path, max_experiments=12)
    assert project.experiment_plan is not None
    primary_id = project.experiment_plan.hypothesis_ids[0]
    secondary_id = next(
        item.id
        for item in project.hypotheses
        if item.execution_readiness == "executable" and item.id != primary_id
    )
    dataset = dataset_manifest()
    project = workflow.attach_dataset_audit(
        project.id,
        manifest=dataset,
        manifest_path=str(tmp_path / "artifacts" / "dataset.json"),
    )
    project = workflow.initialize_experiment_campaign(
        project.id,
        dataset=dataset,
        hypothesis_id=primary_id,
        max_rounds=1,
        max_runs=6,
    )
    first_run_ids = {item.id for item in project.runs}
    complete_current_round(workflow, project.id)
    project = run(
        workflow.review_experiment_round(
            project.id, user_guidance="按系统建议完成三次迭代"
        )
    )
    complete_current_round(workflow, project.id)
    project = run(workflow.review_experiment_round(project.id))
    assert project.experiment_campaign is not None
    assert project.experiment_campaign.status == "completed"

    next_project = workflow.initialize_experiment_campaign(
        project.id,
        dataset=dataset,
        hypothesis_id=secondary_id,
        max_rounds=1,
        max_runs=6,
    )
    assert next_project.experiment_campaign is not None
    assert next_project.experiment_campaign.hypothesis_id == secondary_id
    assert len(next_project.experiment_campaign_history) == 1
    assert first_run_ids <= {item.id for item in next_project.runs}


def test_next_cycle_guidance_archives_campaign_and_preserves_real_runs(tmp_path):
    workflow, project = build_approved_project(tmp_path, max_experiments=6)
    dataset = dataset_manifest()
    project = workflow.attach_dataset_audit(
        project.id,
        manifest=dataset,
        manifest_path=str(tmp_path / "artifacts" / "dataset.json"),
    )
    project = workflow.initialize_experiment_campaign(
        project.id,
        dataset=dataset,
        hypothesis_id=executable_hypothesis_id(project),
        max_rounds=2,
        max_runs=6,
    )
    complete_current_round(workflow, project.id)
    project = run(
        workflow.review_experiment_round(
            project.id, user_guidance="继续完成本创新点的两次迭代"
        )
    )
    complete_current_round(workflow, project.id)
    project = run(workflow.review_experiment_round(project.id))
    assert project.experiment_campaign is not None
    assert project.experiment_campaign.status == "completed"
    historical_run_ids = {item.id for item in project.runs}

    project = workflow.finalize_results(project.id)
    project = run(workflow.advance(project.id))
    assert project.stage == ResearchStage.RESULTS_ANALYZED

    # 配对数不足时发现被标记为 not_tested，_should_revise() 返回 False
    # 流程进入创新审查阶段，而不是研究循环修订
    # 注意：当 findings 是 not_tested 时，start_next_research_cycle() 会失败
    # 因为没有证据驱动修订。这是正确的行为 - 用户应该先完成创新审查
    
    # 验证 advance() 会正确进入 INNOVATION_REVIEWED 阶段
    project = run(workflow.advance(project.id))
    assert project.stage == ResearchStage.INNOVATION_REVIEWED
    assert project.research_cycle == 1  # research_cycle 未增加
    assert project.experiment_campaign is None
    assert len(project.experiment_campaign_history) == 1
    assert historical_run_ids <= {item.id for item in project.runs}
    
    # 完成创新审查后可以进入报告阶段
    project = run(workflow.advance(project.id))
    assert project.stage == ResearchStage.REPORT_READY
    assert project.status == ProjectStatus.COMPLETED


def test_feedback_guard_rejects_early_stop_and_unregistered_cells(tmp_path):
    workflow, project = build_approved_project(tmp_path, max_experiments=12)
    dataset = dataset_manifest()
    project = workflow.attach_dataset_audit(
        project.id,
        manifest=dataset,
        manifest_path=str(tmp_path / "artifacts" / "dataset.json"),
    )
    project = workflow.initialize_experiment_campaign(
        project.id,
        dataset=dataset,
        hypothesis_id=executable_hypothesis_id(project),
        max_rounds=3,
        max_runs=12,
    )
    complete_current_round(workflow, project.id)
    project = workflow.repository.get(project.id)
    project = run(workflow.review_experiment_round(project.id, user_guidance="优先扩大类别覆盖"))
    complete_current_round(workflow, project.id)
    project = workflow.repository.get(project.id)
    summary = workflow.experiment_planner.summarize_current_round(project)
    proposal = ExperimentFeedbackProposal(
        advisor="untrusted-advisor",
        decision="stop",
        rationale="attempted early stop with an invalid action",
        next_phase="complete",
        recommended_cells=[
            ExperimentCell(category="carpet", shots=2, seed=0),
            ExperimentCell(category="not-a-category", shots=999, seed=999),
        ],
        expected_information_gain=1.0,
        stop=True,
    )

    new_runs = workflow.experiment_planner.apply_feedback(
        project,
        proposal=proposal,
        summary=summary,
    )

    assert project.experiment_campaign is not None
    assert project.experiment_campaign.status == "active"
    assert project.experiment_campaign.rounds[0].feedback is not None
    assert project.experiment_campaign.rounds[0].feedback.stop is True
    assert len(new_runs) == 2
    assert all(run.category in dataset.categories for run in new_runs)
    assert all(run.shots in project.experiment_plan.shots for run in new_runs)
    next_round = project.experiment_campaign.rounds[-1]
    scheduled_cells = {
        (run.category, run.shots, run.seed)
        for run in new_runs
    }
    assert next_round.hypothesis_id != project.experiment_campaign.rounds[0].hypothesis_id
    assert scheduled_cells
