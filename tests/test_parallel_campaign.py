import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from fsad_scientist.agents.mock_runtime import MockScientistRuntime
from fsad_scientist.api.app import create_app
from fsad_scientist.config import Settings
from fsad_scientist.datasets.models import DatasetManifest
from fsad_scientist.domain.enums import ResearchStage, RunStatus
from fsad_scientist.domain.models import (
    AnalysisContract,
    ComputeBudget,
    HypothesisRanking,
    ProjectSpec,
)
from fsad_scientist.repository import JsonProjectRepository
from fsad_scientist.workflow import (
    InvalidTransitionError,
    ResearchWorkflow,
    ResultsRequiredError,
)


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


def reviewed_project(tmp_path):
    workflow = ResearchWorkflow(
        repository=JsonProjectRepository(tmp_path / "ledger"),
        runtime=MockScientistRuntime(),
    )
    project = workflow.create_project(
        ProjectSpec(budget=ComputeBudget(max_experiments=240))
    )
    while project.stage != ResearchStage.HYPOTHESES_REVIEWED:
        project = asyncio.run(workflow.advance(project.id))
    return workflow, project


def test_ranking_skips_custom_design_and_builtin_detector_interaction(tmp_path):
    # The ranking gate auto-implements custom selection strategies, but must not
    # treat custom_design arms (bound later by the design) or detector arms over
    # approved built-in detectors as missing strategy code to generate.
    workflow, project = reviewed_project(tmp_path)
    ids = [hypothesis.id for hypothesis in project.hypotheses]
    assert len(ids) >= 3
    reshaped = [
        hypothesis.model_copy(
            update={
                "analysis_contract": AnalysisContract(
                    kind=kind,
                    metric="image_auroc",
                    design_mode=design_mode,
                    treatment=treatment,
                    control=control,
                )
            },
            deep=True,
        )
        for hypothesis, (kind, design_mode, treatment, control) in zip(
            project.hypotheses,
            [
                ("selection_main_effect", "paired_comparison", "k_center", "random"),
                ("query_adaptation", "custom_design", None, None),
                ("detector_interaction", "paired_comparison", "subspacead", "patchcore"),
            ],
        )
    ]
    project.hypotheses = reshaped + project.hypotheses[len(reshaped):]
    project = workflow.repository.save(project)

    rankings = [
        HypothesisRanking(
            hypothesis_id=hypothesis.id,
            selected=index < 3,
            priority=index + 1,
            score=90 - index,
        )
        for index, hypothesis in enumerate(project.hypotheses)
    ]
    project = asyncio.run(workflow.rank_hypotheses(project.id, rankings=rankings))

    assert project.stage == ResearchStage.AWAITING_EXPERIMENT_APPROVAL
    assert project.experiment_plan is not None
    assert project.experiment_plan.hypothesis_ids
    assert set(project.experiment_plan.hypothesis_ids) <= set(ids[:3])


def test_parallel_campaign_custom_design_first_keeps_arm_fields_loadable(tmp_path):
    # Regression: a parallel campaign whose first selected hypothesis is a
    # custom_design contract (no treatment/control arms) used to persist
    # top-level treatment=None/control=None, which made every later
    # ResearchProject reload fail validation (projects list -> 500).
    workflow, project = reviewed_project(tmp_path)
    custom = next(
        hypothesis
        for hypothesis in project.hypotheses
        if (hypothesis.analysis_contract or None) is not None
        and hypothesis.analysis_contract.design_mode == "custom_design"
    )
    rankings = [
        HypothesisRanking(
            hypothesis_id=hypothesis.id,
            selected=hypothesis.id == custom.id,
            priority=1 if hypothesis.id == custom.id else 99,
            score=95 if hypothesis.id == custom.id else 10,
        )
        for hypothesis in project.hypotheses
    ]
    project = asyncio.run(workflow.rank_hypotheses(project.id, rankings=rankings))
    assert project.stage == ResearchStage.AWAITING_EXPERIMENT_APPROVAL
    assert project.experiment_plan.hypothesis_ids == [custom.id]

    manifest = fixture_manifest()
    project = workflow.attach_dataset_audit(
        project.id,
        manifest=manifest,
        manifest_path=str((tmp_path / "fixture.json").resolve()),
    )
    project = asyncio.run(
        workflow.auto_start_parallel_campaign(
            project.id,
            dataset=manifest,
            hypothesis_id=custom.id,
            selected_hypothesis_ids=[custom.id],
            device="cpu",
        )
    )
    campaign = project.experiment_campaign
    assert campaign is not None
    assert campaign.treatment == ""
    assert campaign.control == ""
    assert campaign.rounds[0].treatment == ""
    assert campaign.rounds[0].control == ""

    # The repository round-trip (json save + pydantic revalidation) must survive.
    reloaded = workflow.repository.get(project.id)
    assert reloaded.experiment_campaign is not None
    assert reloaded.experiment_campaign.treatment == ""


def test_ranking_reports_unregistered_custom_detector_arms(tmp_path):
    workflow, project = reviewed_project(tmp_path)
    hypothesis = project.hypotheses[0]
    project.hypotheses = [
        hypothesis.model_copy(
            update={
                "analysis_contract": AnalysisContract(
                    kind="detector_interaction",
                    metric="image_auroc",
                    treatment="custom_detector_x",
                    control="patchcore",
                )
            },
            deep=True,
        ),
        *project.hypotheses[1:],
    ]
    project = workflow.repository.save(project)
    rankings = [
        HypothesisRanking(
            hypothesis_id=hypothesis.id,
            selected=True,
            priority=1,
            score=90,
        ),
        *[
            HypothesisRanking(
                hypothesis_id=item.id,
                selected=False,
                priority=index + 2,
                score=10,
            )
            for index, item in enumerate(project.hypotheses[1:])
        ],
    ]

    with pytest.raises(InvalidTransitionError, match="尚未注册实现"):
        asyncio.run(workflow.rank_hypotheses(project.id, rankings=rankings))


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

    streamed = client.post(
        f"/api/v1/projects/{project.id}/experiment-campaign/execute-stream",
        json={"max_parallel_runs": 2},
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
    assert "round_guidance_required" in event_types
    assert "batch_completed" in event_types
    assert event_types[-1] == "stream_completed"

    partial = workflow.repository.get(project.id)
    assert partial.experiment_campaign is not None
    assert partial.experiment_campaign.status == "awaiting_guidance"
    assert all(
        item.status == "awaiting_guidance"
        for item in partial.experiment_campaign.rounds
    )
    assert partial.experiment_campaign.rounds[0].result_summary["terminal_runs"] == 2
    assert partial.experiment_campaign.rounds[0].result_summary["round_pair_count"] == 1

    first_round_id = partial.experiment_campaign.rounds[0].id
    reviewed = client.post(
        f"/api/v1/projects/{project.id}/experiment-campaign/review",
        json={
            "round_id": first_round_id,
            "user_guidance": "优先检验首轮趋势最可能失效的类别。",
        },
    )
    assert reviewed.status_code == 200
    after_first_guidance = reviewed.json()
    assert after_first_guidance["experiment_campaign"]["status"] == "active"

    continued = client.post(
        f"/api/v1/projects/{project.id}/experiment-campaign/execute-stream",
        json={"max_parallel_runs": 2},
    )
    assert continued.status_code == 200
    continued_frames = [
        json.loads(line[6:])
        for line in continued.text.splitlines()
        if line.startswith("data: ")
    ]
    continued_event_types = [frame["event_type"] for frame in continued_frames]
    assert "run_started" in continued_event_types
    assert "batch_completed" in continued_event_types
    assert continued_event_types[-1] == "stream_completed"

    waiting = workflow.repository.get(project.id)
    assert waiting.experiment_campaign is not None
    assert waiting.experiment_campaign.status == "awaiting_guidance"
    assert waiting.experiment_campaign.rounds[0].status == "ready_for_feedback"
    assert waiting.experiment_campaign.rounds[1].status == "awaiting_guidance"

    second_round_id = waiting.experiment_campaign.rounds[1].id
    reviewed_second = client.post(
        f"/api/v1/projects/{project.id}/experiment-campaign/review",
        json={
            "round_id": second_round_id,
            "user_guidance": "保持对照条件不变，并补齐第二轮的敏感性验证。",
        },
    )
    assert reviewed_second.status_code == 200

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
    assert len(final.experiment_progress) == (
        len(frames) + len(continued_frames) + len(remaining_frames)
    )


def test_auto_start_rebuilds_completed_parallel_campaign_but_stays_idempotent_while_active(tmp_path):
    # Regression: after every Run failed (e.g. broken runtime environment) the
    # parallel campaign ends "completed" with zero verified runs.  finalize
    # refuses to lock such a batch ("at least one verified successful run"),
    # so the only recovery is to rebuild the campaign and re-execute.  auto-start
    # used to return the project unchanged for ANY existing parallel campaign,
    # leaving the UI permanently stuck on "重新执行失败实验"-less dead end.
    workflow, project = reviewed_project(tmp_path)
    primary = project.hypotheses[0]
    rankings = [
        HypothesisRanking(
            hypothesis_id=hypothesis.id,
            selected=hypothesis.id == primary.id,
            priority=1 if hypothesis.id == primary.id else 99,
            score=95 if hypothesis.id == primary.id else 10,
        )
        for hypothesis in project.hypotheses
    ]
    project = asyncio.run(workflow.rank_hypotheses(project.id, rankings=rankings))
    manifest = fixture_manifest()
    project = workflow.attach_dataset_audit(
        project.id,
        manifest=manifest,
        manifest_path=str((tmp_path / "fixture.json").resolve()),
    )
    common = dict(
        dataset=manifest,
        hypothesis_id=primary.id,
        selected_hypothesis_ids=[primary.id],
        device="cpu",
        max_rounds=1,
        max_runs=8,
    )
    project = asyncio.run(workflow.auto_start_parallel_campaign(project.id, **common))
    assert project.experiment_campaign is not None
    assert project.experiment_campaign.status == "active"
    original_campaign_id = project.experiment_campaign.id
    original_run_ids = {item.id for item in project.runs}

    # While the campaign is still active a second call stays idempotent:
    # it must neither rebuild nor raise.
    again = asyncio.run(workflow.auto_start_parallel_campaign(project.id, **common))
    assert again.experiment_campaign is not None
    assert again.experiment_campaign.id == original_campaign_id
    assert again.experiment_campaign_history == []

    # Mark the batch completed without any verified success (all-failed case),
    # then auto-start again: it must rebuild a fresh active campaign and keep
    # the finished one in history for the audit trail.
    project = workflow.repository.get(project.id)
    assert project.experiment_campaign is not None
    project.experiment_campaign.status = "completed"
    project = workflow.repository.save(project)

    rebuilt = asyncio.run(workflow.auto_start_parallel_campaign(project.id, **common))
    campaign = rebuilt.experiment_campaign
    assert campaign is not None
    assert campaign.status == "active"
    assert campaign.id != original_campaign_id
    assert len(rebuilt.experiment_campaign_history) == 1
    assert rebuilt.experiment_campaign_history[0].id == original_campaign_id
    assert rebuilt.experiment_campaign_history[0].status == "completed"
    assert original_run_ids <= {item.id for item in rebuilt.runs}
    assert len(rebuilt.runs) > len(original_run_ids)


def test_recover_stale_parallel_runs_unblocks_campaign_without_queue(tmp_path):
    # Regression: when an execute-stream client disconnects mid-flight its
    # RUNNING Run never receives a result record.  refresh_after_run then keeps
    # seeing an unfinished Round: the campaign stays "active" forever while
    # select_parallel_runs finds nothing queued and raises, so every retry of
    # execute-stream dies silently after the SSE headers were already sent.
    # recover_stale_parallel_runs must fail the orphaned Run and let the
    # normal per-result refresh move the campaign to its real next state.
    workflow, project = reviewed_project(tmp_path)
    primary = project.hypotheses[0]
    rankings = [
        HypothesisRanking(
            hypothesis_id=hypothesis.id,
            selected=hypothesis.id == primary.id,
            priority=1 if hypothesis.id == primary.id else 99,
            score=95 if hypothesis.id == primary.id else 10,
        )
        for hypothesis in project.hypotheses
    ]
    project = asyncio.run(workflow.rank_hypotheses(project.id, rankings=rankings))
    manifest = fixture_manifest()
    project = workflow.attach_dataset_audit(
        project.id,
        manifest=manifest,
        manifest_path=str((tmp_path / "fixture.json").resolve()),
    )
    project = asyncio.run(
        workflow.auto_start_parallel_campaign(
            project.id,
            dataset=manifest,
            hypothesis_id=primary.id,
            selected_hypothesis_ids=[primary.id],
            device="cpu",
            max_rounds=1,
            max_runs=8,
        )
    )
    campaign = project.experiment_campaign
    assert campaign is not None
    assert campaign.status == "active"
    all_runs = list(project.runs)
    assert len(all_runs) >= 2
    orphan = all_runs[0]

    # Simulate a disconnected stream: one Run left RUNNING mid-flight, every
    # other Run of the Round already terminal.  No Run is QUEUED, so the queue
    # is empty while the campaign still looks "active".
    workflow.mark_run_running(project.id, run_id=orphan.id)
    for run in all_runs[1:]:
        workflow.record_run_result(
            project.id,
            run_id=run.id,
            metrics={},
            success=False,
            verified=False,
            result_source="real_executor",
            error="fixture failure",
        )
    stuck = workflow.repository.get(project.id)
    assert stuck.experiment_campaign is not None
    assert stuck.experiment_campaign.status == "active"
    with pytest.raises(ResultsRequiredError):
        workflow.select_parallel_runs(project.id)

    # Recovery fails the orphaned Run; the refreshed campaign can now move on
    # (all runs terminal, midpoint guidance not yet received -> awaiting_guidance).
    recovered = workflow.recover_stale_parallel_runs(project.id)
    orphan_after = next(item for item in recovered.runs if item.id == orphan.id)
    assert orphan_after.status == RunStatus.FAILED
    assert "execution_stream_interrupted" in (orphan_after.error or "")
    assert not any(
        item.status == RunStatus.RUNNING for item in recovered.runs
    )
    assert recovered.experiment_campaign is not None
    assert recovered.experiment_campaign.status == "awaiting_guidance"

    # A second pass over a healthy campaign is a harmless no-op.
    again = workflow.recover_stale_parallel_runs(project.id)
    assert again.experiment_campaign is not None
    assert again.experiment_campaign.id == recovered.experiment_campaign.id
    assert again.experiment_campaign.status == "awaiting_guidance"
