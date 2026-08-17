import asyncio

from fsad_scientist.agents.mock_runtime import MockScientistRuntime
from fsad_scientist.datasets.models import DatasetManifest
from fsad_scientist.domain.enums import ResearchStage, RunStatus
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
    workflow, project = build_approved_project(tmp_path)
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
        max_rounds=3,
        max_runs=6,
    )

    assert project.experiment_campaign is not None
    assert len(project.runs) == 4
    assert fixed_queue_size > len(project.runs)
    assert {run.selection_strategy for run in project.runs} == {"random", "k_center"}
    assert len({(run.category, run.shots, run.seed) for run in project.runs}) == 2

    complete_current_round(workflow, project.id)
    project = workflow.repository.get(project.id)
    assert project.experiment_campaign is not None
    assert project.experiment_campaign.status == "awaiting_feedback"

    project = run(workflow.review_experiment_round(project.id))
    assert project.experiment_campaign is not None
    assert project.experiment_campaign.current_round == 2
    assert len(project.runs) == 6
    first_round = project.experiment_campaign.rounds[0]
    assert first_round.feedback is not None
    assert first_round.result_summary["pair_count"] == 2
    assert first_round.result_summary["mean_difference"] > 0

    complete_current_round(workflow, project.id)
    project = run(workflow.review_experiment_round(project.id))
    assert project.experiment_campaign is not None
    assert project.experiment_campaign.status == "completed"
    assert len(project.runs) == 6
    assert all(run.round_id is not None for run in project.runs)

    project = workflow.finalize_results(project.id)
    assert project.stage == ResearchStage.RESULTS_READY


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
        max_rounds=2,
        max_runs=6,
    )
    complete_current_round(workflow, project.id)
    project = run(workflow.review_experiment_round(project.id))
    complete_current_round(workflow, project.id)
    project = run(workflow.review_experiment_round(project.id))
    assert project.experiment_campaign is not None
    assert project.experiment_campaign.status == "completed"
    historical_run_ids = {item.id for item in project.runs}

    project = workflow.finalize_results(project.id)
    project = run(workflow.advance(project.id))
    assert project.stage == ResearchStage.RESULTS_ANALYZED

    guidance_text = "聚焦 transistor 的反向效应，并检验类别与 K 的交互。"
    project = run(
        workflow.start_next_research_cycle(
            project.id,
            user_guidance=guidance_text,
        )
    )

    assert project.stage == ResearchStage.HYPOTHESES_PROPOSED
    assert project.research_cycle == 2
    assert project.experiment_campaign is None
    assert len(project.experiment_campaign_history) == 1
    assert historical_run_ids <= {item.id for item in project.runs}
    assert project.guidance_records[-1].disposition == "applied"
    assert project.guidance_records[-1].affected_ids
    assert guidance_text in project.hypotheses[0].claim

    while project.stage != ResearchStage.AWAITING_EXPERIMENT_APPROVAL:
        project = run(workflow.advance(project.id))
    project = workflow.approve_experiment_plan(project.id, approved_by="cycle-2-reviewer")
    project = workflow.initialize_experiment_campaign(
        project.id,
        dataset=dataset,
        max_rounds=2,
        max_runs=6,
    )
    assert project.experiment_campaign is not None
    assert project.experiment_campaign.id != project.experiment_campaign_history[0].id
    assert historical_run_ids <= {item.id for item in project.runs}


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
        max_rounds=3,
        max_runs=12,
    )
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
    assert project.experiment_campaign.rounds[0].feedback.stop is False
    assert project.experiment_campaign.rounds[0].feedback.recommended_cells
    assert len(new_runs) == 4
    assert all(run.category in dataset.categories for run in new_runs)
    assert all(run.shots in project.experiment_plan.shots for run in new_runs)
