import asyncio
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from fsad_scientist.agents.mock_runtime import MockScientistRuntime
from fsad_scientist.api.app import create_app
from fsad_scientist.config import Settings
from fsad_scientist.datasets.models import DatasetManifest
from fsad_scientist.datasets.synthetic import build_synthetic_mvtec_smoke_dataset
from fsad_scientist.domain.enums import HypothesisStatus, ResearchStage
from fsad_scientist.domain.models import ComputeBudget, MethodImplementation, ProjectSpec
from fsad_scientist.repository import JsonProjectRepository
from fsad_scientist.workflow import InvalidTransitionError, ResearchWorkflow


def dataset_manifest() -> DatasetManifest:
    return DatasetManifest(
        dataset="MVTec AD",
        root="C:/fixture/mvtec",
        categories=["bottle", "carpet", "capsule", "cable", "transistor"],
        files=[],
        counts={"files": 0},
        digest="a" * 64,
    )


def run(coro):
    return asyncio.run(coro)


def build_awaiting_project(tmp_path: Path):
    workflow = ResearchWorkflow(
        repository=JsonProjectRepository(tmp_path / "ledger"),
        runtime=MockScientistRuntime(),
        artifact_root=tmp_path / "artifacts",
    )
    project = workflow.create_project(
        ProjectSpec(budget=ComputeBudget(max_experiments=6))
    )
    while project.stage != ResearchStage.AWAITING_EXPERIMENT_APPROVAL:
        project = run(workflow.advance(project.id))
    return workflow, project


def query_adaptation_hypothesis(project):
    return next(
        item
        for item in project.hypotheses
        if item.analysis_contract is not None
        and item.analysis_contract.kind == "query_adaptation"
    )


def inject_query_adaptation_into_plan(workflow: ResearchWorkflow, project_id: str):
    project = workflow.repository.get(project_id)
    hypothesis = query_adaptation_hypothesis(project)
    hypothesis.status = HypothesisStatus.SHORTLISTED
    assert project.experiment_plan is not None
    if hypothesis.id not in project.experiment_plan.hypothesis_ids:
        project.experiment_plan.hypothesis_ids.append(hypothesis.id)
    return workflow.repository.save(project), hypothesis


def test_operationalizer_normalizes_no_adaptation_control(tmp_path: Path) -> None:
    workflow, project = build_awaiting_project(tmp_path)
    hypothesis = query_adaptation_hypothesis(project)
    assert hypothesis.analysis_contract is not None
    assert hypothesis.analysis_contract.control == "random"
    assert hypothesis.analysis_contract.treatment == "query_adaptive"


def test_generate_validates_and_smokes_mock_strategy(tmp_path: Path) -> None:
    workflow, project = build_awaiting_project(tmp_path)
    hypothesis = query_adaptation_hypothesis(project)
    updated = run(
        workflow.implement_experiment_method(project.id, hypothesis_id=hypothesis.id)
    )
    implementation = next(
        item
        for item in updated.method_implementations
        if item.hypothesis_id == hypothesis.id
    )
    assert implementation.status == "validated"
    assert implementation.static_validation.passed is True
    assert implementation.smoke_result is not None
    assert implementation.smoke_result.passed is True
    assert implementation.name == "query_adaptive"

    assert implementation.artifact_path is not None
    artifact_path = Path(implementation.artifact_path)
    assert artifact_path.is_file()
    assert "def select(" in artifact_path.read_text(encoding="utf-8")

    artifact = next(item for item in updated.artifacts if item.kind == "generated_strategy")
    assert artifact.payload["code_digest"] == implementation.code_digest
    assert any(item.action == "implement_experiment_method" for item in updated.events)


def test_generate_is_idempotent_for_validated(tmp_path: Path) -> None:
    workflow, project = build_awaiting_project(tmp_path)
    hypothesis = query_adaptation_hypothesis(project)
    run(workflow.implement_experiment_method(project.id, hypothesis_id=hypothesis.id))
    second = run(
        workflow.implement_experiment_method(project.id, hypothesis_id=hypothesis.id)
    )
    implementations = [
        item
        for item in second.method_implementations
        if item.hypothesis_id == hypothesis.id
    ]
    assert len(implementations) == 1
    assert implementations[0].status == "validated"
    assert any("复用" in item.summary for item in second.events)


def test_generate_marks_rejected_on_bad_static_code(tmp_path: Path, monkeypatch) -> None:
    workflow, project = build_awaiting_project(tmp_path)
    hypothesis = query_adaptation_hypothesis(project)

    async def bad_implementation(
        self, project, *, hypothesis, strategy_name, control_name
    ):
        return MethodImplementation(
            kind="selection_strategy",
            name=strategy_name,
            hypothesis_id=hypothesis.id,
            source_code=(
                "def select(candidate_ids, embeddings, k, seed):\n"
                "    import os\n"
                "    return candidate_ids[:k]\n"
            ),
            code_digest="d" * 64,
        )

    monkeypatch.setattr(
        MockScientistRuntime, "implement_selection_strategy", bad_implementation
    )
    updated = run(
        workflow.implement_experiment_method(project.id, hypothesis_id=hypothesis.id)
    )
    implementation = next(
        item
        for item in updated.method_implementations
        if item.hypothesis_id == hypothesis.id
    )
    assert implementation.status == "rejected"
    assert implementation.static_validation.passed is False
    assert not any(item.kind == "generated_strategy" for item in updated.artifacts)


def test_approval_gate_blocks_unimplemented_strategy(tmp_path: Path) -> None:
    workflow, project = build_awaiting_project(tmp_path)
    project, _ = inject_query_adaptation_into_plan(workflow, project.id)
    with pytest.raises(InvalidTransitionError, match="没有已注册实现"):
        workflow.approve_experiment_plan(project.id, approved_by="test-reviewer")


def test_approval_approves_validated_implementation(tmp_path: Path) -> None:
    workflow, project = build_awaiting_project(tmp_path)
    hypothesis = query_adaptation_hypothesis(project)
    run(workflow.implement_experiment_method(project.id, hypothesis_id=hypothesis.id))
    project, _ = inject_query_adaptation_into_plan(workflow, project.id)
    approved = workflow.approve_experiment_plan(project.id, approved_by="test-reviewer")
    implementation = next(
        item
        for item in approved.method_implementations
        if item.hypothesis_id == hypothesis.id
    )
    assert implementation.status == "approved"


def test_approval_gate_blocks_unvalidated_implementation(tmp_path: Path) -> None:
    workflow, project = build_awaiting_project(tmp_path)
    project, hypothesis = inject_query_adaptation_into_plan(workflow, project.id)
    project.method_implementations.append(
        MethodImplementation(
            kind="selection_strategy",
            name="query_adaptive",
            hypothesis_id=hypothesis.id,
            source_code=(
                "def select(candidate_ids, embeddings, k, seed):\n"
                "    return candidate_ids[:k]\n"
            ),
            code_digest="e" * 64,
            status="rejected",
        )
    )
    workflow.repository.save(project)
    with pytest.raises(InvalidTransitionError, match="未通过静态校验"):
        workflow.approve_experiment_plan(project.id, approved_by="test-reviewer")


def inject_generated_detector_into_plan(
    workflow: ResearchWorkflow,
    project_id: str,
    name: str,
):
    project = workflow.repository.get(project_id)
    assert project.experiment_plan is not None
    if name not in project.experiment_plan.detectors:
        project.experiment_plan.detectors.append(name)
    return workflow.repository.save(project)


def test_generate_detector_validates_and_smokes_mock(tmp_path: Path) -> None:
    workflow, project = build_awaiting_project(tmp_path)
    hypothesis = query_adaptation_hypothesis(project)
    updated = run(
        workflow.implement_experiment_detector(
            project.id,
            name_stem="nearest_prototype",
            hypothesis_id=hypothesis.id,
            reference_description="PatchCore 式最近邻记忆库",
        )
    )
    implementation = next(
        item for item in updated.method_implementations if item.kind == "detector"
    )
    assert implementation.status == "validated"
    assert implementation.static_validation.passed is True
    assert implementation.smoke_result is not None
    assert implementation.smoke_result.passed is True
    assert implementation.smoke_result.deterministic is True
    assert implementation.name.startswith("generated_det_nearest_prototype_")

    assert implementation.artifact_path is not None
    artifact_path = Path(implementation.artifact_path)
    assert artifact_path.is_file()
    assert "def anomaly_score(" in artifact_path.read_text(encoding="utf-8")

    artifact = next(item for item in updated.artifacts if item.kind == "generated_detector")
    assert artifact.payload["code_digest"] == implementation.code_digest
    assert any(item.action == "implement_experiment_detector" for item in updated.events)


def test_generate_detector_is_idempotent_for_validated(tmp_path: Path) -> None:
    workflow, project = build_awaiting_project(tmp_path)
    hypothesis = query_adaptation_hypothesis(project)
    run(
        workflow.implement_experiment_detector(
            project.id,
            name_stem="nearest_prototype",
            hypothesis_id=hypothesis.id,
        )
    )
    second = run(
        workflow.implement_experiment_detector(
            project.id,
            name_stem="nearest_prototype",
            hypothesis_id=hypothesis.id,
        )
    )
    implementations = [item for item in second.method_implementations if item.kind == "detector"]
    assert len(implementations) == 1
    assert implementations[0].status == "validated"
    assert any("复用" in item.summary for item in second.events)


def test_generate_detector_marks_rejected_on_bad_static_code(tmp_path: Path, monkeypatch) -> None:
    workflow, project = build_awaiting_project(tmp_path)
    hypothesis = query_adaptation_hypothesis(project)

    async def bad_detector(
        self, project, *, hypothesis, name_stem, reference_description
    ):
        return MethodImplementation(
            kind="detector",
            name="generated_det_bad_ffffffff",
            hypothesis_id=hypothesis.id,
            source_code=(
                "import os\n"
                "def anomaly_score(image, support_images, seed):\n"
                "    return 1.0\n"
            ),
            code_digest="d" * 64,
        )

    monkeypatch.setattr(MockScientistRuntime, "implement_detector", bad_detector)
    updated = run(
        workflow.implement_experiment_detector(
            project.id,
            name_stem="bad",
            hypothesis_id=hypothesis.id,
        )
    )
    implementation = next(
        item for item in updated.method_implementations if item.kind == "detector"
    )
    assert implementation.status == "rejected"
    assert implementation.static_validation.passed is False
    assert not any(item.kind == "generated_detector" for item in updated.artifacts)


def test_approval_gate_blocks_unimplemented_detector(tmp_path: Path) -> None:
    workflow, project = build_awaiting_project(tmp_path)
    project = inject_generated_detector_into_plan(
        workflow, project.id, "generated_det_ghost_ffffffff"
    )
    with pytest.raises(InvalidTransitionError, match="没有已注册实现"):
        workflow.approve_experiment_plan(project.id, approved_by="test-reviewer")


def test_approval_gate_blocks_unvalidated_detector(tmp_path: Path) -> None:
    workflow, project = build_awaiting_project(tmp_path)
    hypothesis = query_adaptation_hypothesis(project)
    project.method_implementations.append(
        MethodImplementation(
            kind="detector",
            name="generated_det_broken_ffffffff",
            hypothesis_id=hypothesis.id,
            source_code=(
                "import numpy as np\n"
                "def anomaly_score(image, support_images, seed):\n"
                "    return 1.0\n"
            ),
            code_digest="e" * 64,
            status="rejected",
        )
    )
    project = workflow.repository.save(project)
    project = inject_generated_detector_into_plan(
        workflow, project.id, "generated_det_broken_ffffffff"
    )
    with pytest.raises(InvalidTransitionError, match="未通过静态校验"):
        workflow.approve_experiment_plan(project.id, approved_by="test-reviewer")


def test_approval_approves_validated_detector(tmp_path: Path) -> None:
    workflow, project = build_awaiting_project(tmp_path)
    hypothesis = query_adaptation_hypothesis(project)
    run(
        workflow.implement_experiment_detector(
            project.id,
            name_stem="nearest_prototype",
            hypothesis_id=hypothesis.id,
        )
    )
    project = workflow.repository.get(project.id)
    implementation = next(
        item for item in project.method_implementations if item.kind == "detector"
    )
    project = inject_generated_detector_into_plan(workflow, project.id, implementation.name)
    approved = workflow.approve_experiment_plan(project.id, approved_by="test-reviewer")
    implementation = next(
        item for item in approved.method_implementations if item.kind == "detector"
    )
    assert implementation.status == "approved"


def test_campaign_initializes_with_generated_detector(tmp_path: Path) -> None:
    workflow, project = build_awaiting_project(tmp_path)
    hypothesis = query_adaptation_hypothesis(project)
    run(
        workflow.implement_experiment_detector(
            project.id,
            name_stem="nearest_prototype",
            hypothesis_id=hypothesis.id,
        )
    )
    project = workflow.repository.get(project.id)
    implementation = next(
        item for item in project.method_implementations if item.kind == "detector"
    )
    project = inject_generated_detector_into_plan(workflow, project.id, implementation.name)
    approved = workflow.approve_experiment_plan(project.id, approved_by="test-reviewer")
    dataset = dataset_manifest()
    project = workflow.attach_dataset_audit(
        approved.id,
        manifest=dataset,
        manifest_path=str(tmp_path / "artifacts" / "dataset.json"),
    )
    project = workflow.initialize_experiment_campaign(
        project.id,
        dataset=dataset,
        detector=implementation.name,
        max_rounds=3,
        max_runs=6,
    )
    campaign = project.experiment_campaign
    assert campaign is not None
    assert campaign.detector == implementation.name
    assert {run.detector for run in project.runs} == {implementation.name}


def test_campaign_rejects_unapproved_generated_detector(tmp_path: Path) -> None:
    workflow, project = build_awaiting_project(tmp_path)
    hypothesis = query_adaptation_hypothesis(project)
    run(
        workflow.implement_experiment_detector(
            project.id,
            name_stem="nearest_prototype",
            hypothesis_id=hypothesis.id,
        )
    )
    approved = workflow.approve_experiment_plan(project.id, approved_by="test-reviewer")
    dataset = dataset_manifest()
    project = workflow.attach_dataset_audit(
        approved.id,
        manifest=dataset,
        manifest_path=str(tmp_path / "artifacts" / "dataset.json"),
    )
    implementation = next(
        item for item in project.method_implementations if item.kind == "detector"
    )
    with pytest.raises(InvalidTransitionError, match="not approved and executable"):
        workflow.initialize_experiment_campaign(
            project.id,
            dataset=dataset,
            detector=implementation.name,
            max_rounds=3,
            max_runs=6,
        )


def test_campaign_uses_approved_custom_strategy(tmp_path: Path) -> None:
    workflow, project = build_awaiting_project(tmp_path)
    hypothesis = query_adaptation_hypothesis(project)
    run(workflow.implement_experiment_method(project.id, hypothesis_id=hypothesis.id))
    project = workflow.repository.get(project.id)
    hypothesis = query_adaptation_hypothesis(project)
    hypothesis.status = HypothesisStatus.SHORTLISTED
    assert project.experiment_plan is not None
    project.experiment_plan.hypothesis_ids = [hypothesis.id]
    project = workflow.repository.save(project)

    approved = workflow.approve_experiment_plan(project.id, approved_by="test-reviewer")
    dataset = dataset_manifest()
    project = workflow.attach_dataset_audit(
        approved.id,
        manifest=dataset,
        manifest_path=str(tmp_path / "artifacts" / "dataset.json"),
    )
    project = workflow.initialize_experiment_campaign(
        project.id,
        dataset=dataset,
        max_rounds=3,
        max_runs=6,
    )
    campaign = project.experiment_campaign
    assert campaign is not None
    assert campaign.treatment == "query_adaptive"
    assert campaign.control == "random"
    assert {run.selection_strategy for run in project.runs} == {"query_adaptive", "random"}


def test_campaign_skips_validated_but_unapproved_strategy(tmp_path: Path) -> None:
    workflow, project = build_awaiting_project(tmp_path)
    hypothesis = query_adaptation_hypothesis(project)
    run(workflow.implement_experiment_method(project.id, hypothesis_id=hypothesis.id))
    approved = workflow.approve_experiment_plan(project.id, approved_by="test-reviewer")
    dataset = dataset_manifest()
    project = workflow.attach_dataset_audit(
        approved.id,
        manifest=dataset,
        manifest_path=str(tmp_path / "artifacts" / "dataset.json"),
    )
    project = workflow.initialize_experiment_campaign(
        project.id,
        dataset=dataset,
        max_rounds=3,
        max_runs=6,
    )
    campaign = project.experiment_campaign
    assert campaign is not None
    assert campaign.treatment == "k_center"
    implementation = next(
        item
        for item in project.method_implementations
        if item.hypothesis_id == hypothesis.id
    )
    assert implementation.status == "validated"


def test_api_generate_detector_endpoint(tmp_path: Path) -> None:
    app = create_app(
        settings=Settings(runtime="mock", artifact_root=str(tmp_path / "artifacts")),
        storage_path=tmp_path / "api-ledger",
        runtime=MockScientistRuntime(),
    )
    client = TestClient(app)
    created = client.post("/api/v1/projects/demo").json()
    project_id = created["id"]
    project = created
    while project["stage"] != "awaiting_experiment_approval":
        response = client.post(f"/api/v1/projects/{project_id}/advance")
        assert response.status_code == 200
        project = response.json()
    hypothesis = next(
        item
        for item in project["hypotheses"]
        if item.get("analysis_contract")
        and item["analysis_contract"]["kind"] == "query_adaptation"
    )
    response = client.post(
        f"/api/v1/projects/{project_id}/experiment-methods/generate-detector",
        json={
            "hypothesis_id": hypothesis["id"],
            "name_stem": "nearest_proto",
            "reference_description": "最近邻记忆库",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert any(
        item.get("kind") == "detector"
        and item.get("status") == "validated"
        and item.get("name", "").startswith("generated_det_nearest_proto_")
        for item in payload["method_implementations"]
    )


def test_api_campaign_initialize_accepts_generated_detector_name(tmp_path: Path) -> None:
    storage = tmp_path / "api-ledger"
    app = create_app(
        settings=Settings(runtime="mock", artifact_root=str(tmp_path / "artifacts")),
        storage_path=storage,
        runtime=MockScientistRuntime(),
    )
    client = TestClient(app)
    created = client.post("/api/v1/projects/demo").json()
    project_id = created["id"]
    project = created
    while project["stage"] != "awaiting_experiment_approval":
        project = client.post(f"/api/v1/projects/{project_id}/advance").json()
    hypothesis = next(
        item
        for item in project["hypotheses"]
        if item.get("analysis_contract")
        and item["analysis_contract"]["kind"] == "query_adaptation"
    )
    generated = client.post(
        f"/api/v1/projects/{project_id}/experiment-methods/generate-detector",
        json={"hypothesis_id": hypothesis["id"], "name_stem": "nearest_proto"},
    ).json()
    implementation = next(
        item
        for item in generated["method_implementations"]
        if item.get("kind") == "detector"
    )

    repository = JsonProjectRepository(storage)
    stored = repository.get(project_id)
    assert stored.experiment_plan is not None
    stored.experiment_plan.detectors.append(implementation["name"])
    repository.save(stored)

    approved = client.post(
        f"/api/v1/projects/{project_id}/approve",
        json={"approved_by": "api-tester"},
    )
    assert approved.status_code == 200

    data_root = tmp_path / "data"
    build_synthetic_mvtec_smoke_dataset(data_root)
    audited = client.post(
        f"/api/v1/projects/{project_id}/dataset/audit",
        json={"root": str(data_root), "dataset_name": "MVTec AD"},
    )
    assert audited.status_code == 200
    manifest_path = audited.json()["dataset_audits"][-1]["manifest_path"]

    initialized = client.post(
        f"/api/v1/projects/{project_id}/experiment-campaign/initialize",
        json={
            "dataset_manifest_path": manifest_path,
            "detector": implementation["name"],
            "device": "cpu",
            "max_rounds": 3,
            "max_runs": 6,
        },
    )
    assert initialized.status_code == 200
    campaign = initialized.json()["experiment_campaign"]
    assert campaign["detector"] == implementation["name"]


def test_api_generate_endpoint(tmp_path: Path) -> None:
    app = create_app(
        settings=Settings(runtime="mock", artifact_root=str(tmp_path / "artifacts")),
        storage_path=tmp_path / "api-ledger",
        runtime=MockScientistRuntime(),
    )
    client = TestClient(app)
    created = client.post("/api/v1/projects/demo").json()
    project_id = created["id"]
    project = created
    while project["stage"] != "awaiting_experiment_approval":
        response = client.post(f"/api/v1/projects/{project_id}/advance")
        assert response.status_code == 200
        project = response.json()
    hypothesis = next(
        item
        for item in project["hypotheses"]
        if item.get("analysis_contract")
        and item["analysis_contract"]["kind"] == "query_adaptation"
    )
    response = client.post(
        f"/api/v1/projects/{project_id}/experiment-methods/generate",
        json={"hypothesis_id": hypothesis["id"]},
    )
    assert response.status_code == 200
    payload = response.json()
    assert any(
        item.get("status") == "validated" and item.get("name") == "query_adaptive"
        for item in payload["method_implementations"]
    )
