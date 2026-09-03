import asyncio
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from fsad_scientist.agents.mock_runtime import MockScientistRuntime
from fsad_scientist.agents.qwen_runtime import QwenScientistRuntime
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


class _FailingAgentScopeClient:
    async def complete(self, **kwargs):
        raise RuntimeError("simulated AgentScope failure")


def use_failing_qwen_runtime(workflow: ResearchWorkflow) -> None:
    runtime = QwenScientistRuntime()
    runtime.client = _FailingAgentScopeClient()  # type: ignore[assignment]
    workflow.runtime = runtime


def build_reviewed_project(tmp_path: Path):
    workflow = ResearchWorkflow(
        repository=JsonProjectRepository(tmp_path / "ledger"),
        runtime=MockScientistRuntime(),
        artifact_root=tmp_path / "artifacts",
    )
    project = workflow.create_project(
        ProjectSpec(budget=ComputeBudget(max_experiments=6))
    )
    while project.stage != ResearchStage.HYPOTHESES_REVIEWED:
        project = run(workflow.advance(project.id))
    return workflow, project


def test_generated_detector_scopes_plan_to_compatible_bound_hypothesis(
    tmp_path: Path,
) -> None:
    workflow, project = build_reviewed_project(tmp_path)
    hypothesis = next(
        item
        for item in project.hypotheses
        if item.analysis_contract is not None
        and item.analysis_contract.metric in {"image_auroc", "image_ap"}
    )
    generated = run(
        workflow.implement_experiment_detector(
            project.id,
            hypothesis_id=hypothesis.id,
            name_stem="image_only",
        )
    )
    detector = next(
        item
        for item in generated.method_implementations
        if item.kind == "detector" and item.hypothesis_id == hypothesis.id
    )

    planned = run(workflow.advance(project.id))

    assert planned.experiment_plan is not None
    assert hypothesis.id in planned.experiment_plan.hypothesis_ids
    assert any(
        design.hypothesis_id != hypothesis.id
        and design.design_mode == "custom_design"
        and all(
            (factor.field or factor.run_field) != "detector"
            for factor in design.factors
        )
        for design in planned.experiment_plan.designs
    )
    assert planned.experiment_plan.detectors == [detector.name]
    assert planned.experiment_plan.hypothesis_contracts[hypothesis.id].metric in {
        "image_auroc",
        "image_ap",
    }


def query_adaptation_hypothesis(project):
    return next(
        item
        for item in project.hypotheses
        if item.analysis_contract is not None
        and item.analysis_contract.kind == "query_adaptation"
    )


def selection_main_effect_hypothesis(project):
    return next(
        item
        for item in project.hypotheses
        if item.analysis_contract is not None
        and item.analysis_contract.kind == "selection_main_effect"
    )


def inject_query_adaptation_into_plan(workflow: ResearchWorkflow, project_id: str):
    project = workflow.repository.get(project_id)
    hypothesis = query_adaptation_hypothesis(project)
    hypothesis.status = HypothesisStatus.SHORTLISTED
    assert project.experiment_plan is not None
    project.experiment_plan.hypothesis_ids = [hypothesis.id]
    assert hypothesis.analysis_contract is not None
    project.experiment_plan.hypothesis_contracts = {
        hypothesis.id: hypothesis.analysis_contract.model_copy(deep=True)
    }
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


def test_generate_replaces_builtin_strategy_and_syncs_plan(tmp_path: Path) -> None:
    workflow, project = build_awaiting_project(tmp_path)
    hypothesis = selection_main_effect_hypothesis(project)
    assert hypothesis.analysis_contract is not None
    assert hypothesis.analysis_contract.treatment == "k_center"
    assert hypothesis.analysis_contract.control == "random"
    assert project.experiment_plan is not None
    old_digest = project.experiment_plan.preregistration_digest

    updated = run(
        workflow.implement_experiment_method(project.id, hypothesis_id=hypothesis.id)
    )
    implementation = next(
        item
        for item in updated.method_implementations
        if item.kind == "selection_strategy" and item.hypothesis_id == hypothesis.id
    )
    updated_hypothesis = next(item for item in updated.hypotheses if item.id == hypothesis.id)
    plan = updated.experiment_plan

    assert implementation.status == "validated"
    assert implementation.name == f"ai_strategy_{hypothesis.id}"
    assert updated_hypothesis.analysis_contract is not None
    assert updated_hypothesis.analysis_contract.treatment == implementation.name
    assert updated_hypothesis.analysis_contract.control == "random"
    assert plan is not None
    assert plan.hypothesis_contracts[hypothesis.id].treatment == implementation.name
    assert implementation.name in plan.selection_strategies
    assert plan.method_implementation_digests[implementation.name] == implementation.code_digest
    assert plan.preregistration_digest != old_digest

    approved = workflow.approve_experiment_plan(project.id, approved_by="test-reviewer")
    dataset = dataset_manifest()
    approved = workflow.attach_dataset_audit(
        approved.id,
        manifest=dataset,
        manifest_path=str(tmp_path / "artifacts" / "dataset.json"),
    )
    campaign_project = workflow.initialize_experiment_campaign(
        approved.id,
        dataset=dataset,
        hypothesis_id=hypothesis.id,
        max_rounds=3,
        max_runs=6,
    )
    campaign = campaign_project.experiment_campaign
    assert campaign is not None
    assert campaign.treatment == implementation.name
    assert campaign.control == "random"
    assert {run.selection_strategy for run in campaign_project.runs} == {
        implementation.name,
        "random",
    }
    assert implementation.name in {
        run.selection_strategy
        for run in approved.runs
    }


def test_builtin_replacement_precedes_preregistration_generation(tmp_path: Path) -> None:
    workflow, project = build_reviewed_project(tmp_path)
    hypothesis = selection_main_effect_hypothesis(project)

    generated = run(
        workflow.implement_experiment_method(project.id, hypothesis_id=hypothesis.id)
    )
    assert generated.experiment_plan is None
    implementation = next(
        item
        for item in generated.method_implementations
        if item.kind == "selection_strategy" and item.hypothesis_id == hypothesis.id
    )

    preregistered = run(workflow.advance(project.id))
    assert preregistered.experiment_plan is not None
    assert implementation.name in preregistered.experiment_plan.selection_strategies
    assert (
        preregistered.experiment_plan.hypothesis_contracts[hypothesis.id].treatment
        == implementation.name
    )
    assert preregistered.experiment_plan.method_implementation_digests[implementation.name]
    assert preregistered.experiment_plan.preregistration_digest


def test_builtin_strategy_replacement_is_idempotent(tmp_path: Path) -> None:
    workflow, project = build_awaiting_project(tmp_path)
    hypothesis = selection_main_effect_hypothesis(project)
    first = run(
        workflow.implement_experiment_method(project.id, hypothesis_id=hypothesis.id)
    )
    first_impl = next(
        item
        for item in first.method_implementations
        if item.kind == "selection_strategy" and item.hypothesis_id == hypothesis.id
    )
    second = run(
        workflow.implement_experiment_method(project.id, hypothesis_id=hypothesis.id)
    )
    implementations = [
        item
        for item in second.method_implementations
        if item.kind == "selection_strategy" and item.hypothesis_id == hypothesis.id
    ]

    assert len(implementations) == 1
    assert implementations[0].name == first_impl.name
    assert implementations[0].status == "validated"
    assert any("复用" in item.summary for item in second.events)


def test_builtin_strategy_replacement_normalizes_reversed_pair(tmp_path: Path) -> None:
    workflow, project = build_awaiting_project(tmp_path)
    hypothesis = selection_main_effect_hypothesis(project)
    assert hypothesis.analysis_contract is not None
    hypothesis.analysis_contract = hypothesis.analysis_contract.model_copy(
        update={"treatment": "random", "control": "k_center"}
    )
    workflow.repository.save(project)

    updated = run(
        workflow.implement_experiment_method(project.id, hypothesis_id=hypothesis.id)
    )
    updated_hypothesis = next(item for item in updated.hypotheses if item.id == hypothesis.id)
    implementation = next(
        item
        for item in updated.method_implementations
        if item.kind == "selection_strategy" and item.hypothesis_id == hypothesis.id
    )

    assert updated_hypothesis.analysis_contract is not None
    assert updated_hypothesis.analysis_contract.treatment == implementation.name
    assert updated_hypothesis.analysis_contract.control == "random"


def test_generate_accepts_builtin_strategy_aliases(tmp_path: Path) -> None:
    workflow, project = build_awaiting_project(tmp_path)
    hypothesis = selection_main_effect_hypothesis(project)
    assert hypothesis.analysis_contract is not None
    hypothesis.analysis_contract = hypothesis.analysis_contract.model_copy(
        update={
            "treatment": "dynamic_geometric_consistency_pool_compression",
            "control": "random_sampling_baseline",
        }
    )
    workflow.repository.save(project)

    updated = run(
        workflow.implement_experiment_method(project.id, hypothesis_id=hypothesis.id)
    )
    updated_hypothesis = next(item for item in updated.hypotheses if item.id == hypothesis.id)
    implementation = next(
        item
        for item in updated.method_implementations
        if item.kind == "selection_strategy" and item.hypothesis_id == hypothesis.id
    )

    assert implementation.status == "validated"
    assert implementation.name == "dynamic_geometric_consistency_pool_compression"
    assert updated_hypothesis.analysis_contract is not None
    assert updated_hypothesis.analysis_contract.treatment == implementation.name
    assert updated_hypothesis.analysis_contract.control == "random"


def test_generate_accepts_static_prototype_baseline_alias(tmp_path: Path) -> None:
    workflow, project = build_awaiting_project(tmp_path)
    hypothesis = selection_main_effect_hypothesis(project)
    assert hypothesis.analysis_contract is not None
    hypothesis.analysis_contract = hypothesis.analysis_contract.model_copy(
        update={
            "treatment": "Query-density adaptive prototype refinement",
            "control": "Static single-sample prototype",
        }
    )
    workflow.repository.save(project)

    updated = run(
        workflow.implement_experiment_method(project.id, hypothesis_id=hypothesis.id)
    )
    updated_hypothesis = next(item for item in updated.hypotheses if item.id == hypothesis.id)
    implementation = next(
        item
        for item in updated.method_implementations
        if item.kind == "selection_strategy" and item.hypothesis_id == hypothesis.id
    )

    assert implementation.status == "validated"
    assert implementation.name == "query_density_adaptive_prototype_refinement"
    assert updated_hypothesis.analysis_contract is not None
    assert updated_hypothesis.analysis_contract.treatment == implementation.name
    assert updated_hypothesis.analysis_contract.control == "random"


def test_generate_supports_two_non_builtin_strategies(tmp_path: Path) -> None:
    workflow, project = build_awaiting_project(tmp_path)
    hypothesis = query_adaptation_hypothesis(project)
    assert hypothesis.analysis_contract is not None
    hypothesis.analysis_contract = hypothesis.analysis_contract.model_copy(
        update={
            "treatment": "Dynamic Refinement with Static Memory Constraint",
            "control": "Unconstrained Dynamic Refinement (FastRef)",
        }
    )
    hypothesis.status = HypothesisStatus.SHORTLISTED
    assert project.experiment_plan is not None
    project.experiment_plan.hypothesis_ids = [hypothesis.id]
    project.experiment_plan.hypothesis_contracts[hypothesis.id] = (
        hypothesis.analysis_contract.model_copy(deep=True)
    )
    old_digest = project.experiment_plan.preregistration_digest
    workflow.repository.save(project)

    updated = run(
        workflow.implement_experiment_method(project.id, hypothesis_id=hypothesis.id)
    )
    expected_names = {
        "dynamic_refinement_with_static_memory_constraint",
        "unconstrained_dynamic_refinement_fastref",
    }
    implementations = [
        item
        for item in updated.method_implementations
        if item.kind == "selection_strategy" and item.hypothesis_id == hypothesis.id
    ]
    updated_hypothesis = next(item for item in updated.hypotheses if item.id == hypothesis.id)
    plan = updated.experiment_plan

    assert {item.name for item in implementations} == expected_names
    assert {item.status for item in implementations} == {"validated"}
    assert len({item.code_digest for item in implementations}) == 2
    assert updated_hypothesis.analysis_contract is not None
    assert updated_hypothesis.analysis_contract.treatment == (
        "dynamic_refinement_with_static_memory_constraint"
    )
    assert updated_hypothesis.analysis_contract.control == (
        "unconstrained_dynamic_refinement_fastref"
    )
    assert plan is not None
    assert plan.hypothesis_contracts[hypothesis.id] == (
        updated_hypothesis.analysis_contract
    )
    assert expected_names <= set(plan.selection_strategies)
    assert set(plan.method_implementation_digests) >= expected_names
    assert plan.preregistration_digest != old_digest

    second = run(
        workflow.implement_experiment_method(project.id, hypothesis_id=hypothesis.id)
    )
    second_implementations = [
        item
        for item in second.method_implementations
        if item.kind == "selection_strategy" and item.hypothesis_id == hypothesis.id
    ]
    assert len(second_implementations) == 2
    assert {item.name for item in second_implementations} == expected_names
    assert sum("复用" in item.summary for item in second.events) >= 2

    approved = workflow.approve_experiment_plan(project.id, approved_by="test-reviewer")
    assert {
        item.name
        for item in approved.method_implementations
        if item.hypothesis_id == hypothesis.id and item.status == "approved"
    } == expected_names


def test_two_non_builtin_generation_keeps_contract_atomic_on_failure(
    tmp_path: Path, monkeypatch
) -> None:
    workflow, project = build_awaiting_project(tmp_path)
    hypothesis = query_adaptation_hypothesis(project)
    assert hypothesis.analysis_contract is not None
    hypothesis.analysis_contract = hypothesis.analysis_contract.model_copy(
        update={
            "treatment": "Dynamic Refinement with Static Memory Constraint",
            "control": "Unconstrained Dynamic Refinement (FastRef)",
        }
    )
    hypothesis.status = HypothesisStatus.SHORTLISTED
    assert project.experiment_plan is not None
    project.experiment_plan.hypothesis_ids = [hypothesis.id]
    project.experiment_plan.hypothesis_contracts[hypothesis.id] = (
        hypothesis.analysis_contract.model_copy(deep=True)
    )
    original_contract = hypothesis.analysis_contract.model_copy(deep=True)
    original_digest = project.experiment_plan.preregistration_digest
    workflow.repository.save(project)
    calls = 0

    original_implementation = MockScientistRuntime.implement_selection_strategy

    async def patched_implementation(
        self, project, *, hypothesis, strategy_name, control_name
    ):
        nonlocal calls
        calls += 1
        if calls == 1:
            return await original_implementation(
                self,
                project,
                hypothesis=hypothesis,
                strategy_name=strategy_name,
                control_name=control_name,
            )
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
        MockScientistRuntime,
        "implement_selection_strategy",
        patched_implementation,
    )
    updated = run(
        workflow.implement_experiment_method(project.id, hypothesis_id=hypothesis.id)
    )
    updated_hypothesis = next(item for item in updated.hypotheses if item.id == hypothesis.id)

    assert updated_hypothesis.analysis_contract == original_contract
    assert updated.experiment_plan is not None
    assert updated.experiment_plan.hypothesis_contracts[hypothesis.id] == original_contract
    assert updated.experiment_plan.preregistration_digest == original_digest
    assert {
        item.status
        for item in updated.method_implementations
        if item.kind == "selection_strategy" and item.hypothesis_id == hypothesis.id
    } == {"validated", "rejected"}


def test_approval_rejects_two_custom_arms_with_same_digest(tmp_path: Path) -> None:
    workflow, project = build_awaiting_project(tmp_path)
    hypothesis = query_adaptation_hypothesis(project)
    assert hypothesis.analysis_contract is not None
    hypothesis.analysis_contract = hypothesis.analysis_contract.model_copy(
        update={
            "treatment": "Dynamic Refinement with Static Memory Constraint",
            "control": "Unconstrained Dynamic Refinement (FastRef)",
        }
    )
    hypothesis.status = HypothesisStatus.SHORTLISTED
    assert project.experiment_plan is not None
    project.experiment_plan.hypothesis_ids = [hypothesis.id]
    project.experiment_plan.hypothesis_contracts[hypothesis.id] = (
        hypothesis.analysis_contract.model_copy(deep=True)
    )
    workflow.repository.save(project)
    generated = run(
        workflow.implement_experiment_method(project.id, hypothesis_id=hypothesis.id)
    )
    implementations = [
        item
        for item in generated.method_implementations
        if item.kind == "selection_strategy" and item.hypothesis_id == hypothesis.id
    ]
    assert len(implementations) == 2
    implementations[1].code_digest = implementations[0].code_digest
    assert generated.experiment_plan is not None
    generated.experiment_plan.method_implementation_digests[
        implementations[1].name
    ] = implementations[0].code_digest
    workflow.repository.save(generated)

    with pytest.raises(InvalidTransitionError, match="相同策略实现"):
        workflow.approve_experiment_plan(project.id, approved_by="test-reviewer")

    stored = workflow.repository.get(project.id)
    assert {
        item.status
        for item in stored.method_implementations
        if item.kind == "selection_strategy" and item.hypothesis_id == hypothesis.id
    } == {"validated"}


def test_approval_rejects_plan_contract_drift(tmp_path: Path) -> None:
    workflow, project = build_awaiting_project(tmp_path)
    hypothesis = selection_main_effect_hypothesis(project)
    assert project.experiment_plan is not None
    hypothesis.analysis_contract = hypothesis.analysis_contract.model_copy(
        update={"treatment": "ai_strategy_external_drift"}
    )
    workflow.repository.save(project)

    with pytest.raises(InvalidTransitionError, match="偏离预注册计划"):
        workflow.approve_experiment_plan(project.id, approved_by="test-reviewer")


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


def test_non_mock_selection_fallback_is_rejected_before_persistence(tmp_path: Path) -> None:
    workflow, project = build_awaiting_project(tmp_path)
    hypothesis = query_adaptation_hypothesis(project)
    use_failing_qwen_runtime(workflow)
    before = workflow.repository.get(project.id)

    with pytest.raises(InvalidTransitionError, match="deterministic-fallback"):
        run(workflow.implement_experiment_method(project.id, hypothesis_id=hypothesis.id))

    stored = workflow.repository.get(project.id)
    assert stored.method_implementations == before.method_implementations
    assert stored.artifacts == before.artifacts


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


def test_non_mock_detector_fallback_is_rejected_before_persistence(tmp_path: Path) -> None:
    workflow, project = build_awaiting_project(tmp_path)
    hypothesis = query_adaptation_hypothesis(project)
    use_failing_qwen_runtime(workflow)
    before = workflow.repository.get(project.id)

    with pytest.raises(InvalidTransitionError, match="deterministic-fallback"):
        run(
            workflow.implement_experiment_detector(
                project.id,
                name_stem="nearest_prototype",
                hypothesis_id=hypothesis.id,
            )
        )

    stored = workflow.repository.get(project.id)
    assert stored.method_implementations == before.method_implementations
    assert stored.artifacts == before.artifacts


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
    hypothesis = selection_main_effect_hypothesis(project)
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
    with pytest.raises(InvalidTransitionError, match="请刷新页面复核后再次批准"):
        workflow.approve_experiment_plan(project.id, approved_by="test-reviewer")
    approved = workflow.approve_experiment_plan(project.id, approved_by="test-reviewer")
    implementation = next(
        item for item in approved.method_implementations if item.kind == "detector"
    )
    assert implementation.status == "approved"


def test_campaign_initializes_with_generated_detector(tmp_path: Path) -> None:
    workflow, project = build_awaiting_project(tmp_path)
    hypothesis = selection_main_effect_hypothesis(project)
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
    with pytest.raises(InvalidTransitionError, match="请刷新页面复核后再次批准"):
        workflow.approve_experiment_plan(project.id, approved_by="test-reviewer")
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
        hypothesis_id=next(
            item.id
            for item in project.hypotheses
            if item.id
            in (
                project.experiment_plan.hypothesis_ids
                if project.experiment_plan
                else []
            )
            and item.execution_readiness == "executable"
        ),
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
            hypothesis_id=hypothesis.id,
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
        hypothesis_id=hypothesis.id,
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
        hypothesis_id=next(
            item.id
            for item in project.hypotheses
            if item.id
            in (
                project.experiment_plan.hypothesis_ids
                if project.experiment_plan
                else []
            )
            and item.execution_readiness == "executable"
        ),
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
        and item["analysis_contract"]["metric"] in {"image_auroc", "image_ap"}
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


def test_api_generate_replaces_builtin_strategy_endpoint(tmp_path: Path) -> None:
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
        if item.get("analysis_contract", {}).get("kind") == "selection_main_effect"
    )
    response = client.post(
        f"/api/v1/projects/{project_id}/experiment-methods/generate",
        json={"hypothesis_id": hypothesis["id"]},
    )
    assert response.status_code == 200
    payload = response.json()
    implementation = next(
        item
        for item in payload["method_implementations"]
        if item.get("kind") == "selection_strategy"
        and item.get("hypothesis_id") == hypothesis["id"]
    )
    updated_hypothesis = next(
        item for item in payload["hypotheses"] if item["id"] == hypothesis["id"]
    )

    assert implementation["status"] == "validated"
    assert implementation["name"].startswith("ai_strategy_")
    assert updated_hypothesis["analysis_contract"]["treatment"] == implementation["name"]
    assert updated_hypothesis["analysis_contract"]["control"] == "random"


def test_custom_design_method_endpoint_returns_structured_conflict(tmp_path: Path) -> None:
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
        project = client.post(f"/api/v1/projects/{project_id}/advance").json()
    hypothesis = next(
        item
        for item in project["hypotheses"]
        if item.get("analysis_contract", {}).get("design_mode") == "custom_design"
    )

    response = client.post(
        f"/api/v1/projects/{project_id}/experiment-methods/generate",
        json={"hypothesis_id": hypothesis["id"]},
    )

    assert response.status_code == 409
    assert "custom_design" in response.json()["detail"]
    assert "treatment/control" in response.json()["detail"]


def test_api_generate_supports_two_non_builtin_strategies(tmp_path: Path) -> None:
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
        response = client.post(f"/api/v1/projects/{project_id}/advance")
        assert response.status_code == 200
        project = response.json()

    repository = JsonProjectRepository(storage)
    stored = repository.get(project_id)
    hypothesis = query_adaptation_hypothesis(stored)
    assert hypothesis.analysis_contract is not None
    hypothesis.analysis_contract = hypothesis.analysis_contract.model_copy(
        update={
            "treatment": "Dynamic Refinement with Static Memory Constraint",
            "control": "Unconstrained Dynamic Refinement (FastRef)",
        }
    )
    hypothesis.status = HypothesisStatus.SHORTLISTED
    assert stored.experiment_plan is not None
    stored.experiment_plan.hypothesis_ids = [hypothesis.id]
    stored.experiment_plan.hypothesis_contracts[hypothesis.id] = (
        hypothesis.analysis_contract.model_copy(deep=True)
    )
    repository.save(stored)

    response = client.post(
        f"/api/v1/projects/{project_id}/experiment-methods/generate",
        json={"hypothesis_id": hypothesis.id},
    )

    assert response.status_code == 200
    payload = response.json()
    implementations = [
        item
        for item in payload["method_implementations"]
        if item.get("kind") == "selection_strategy"
        and item.get("hypothesis_id") == hypothesis.id
    ]
    updated_hypothesis = next(
        item for item in payload["hypotheses"] if item["id"] == hypothesis.id
    )
    assert {item["name"] for item in implementations} == {
        "dynamic_refinement_with_static_memory_constraint",
        "unconstrained_dynamic_refinement_fastref",
    }
    assert {item["status"] for item in implementations} == {"validated"}
    assert updated_hypothesis["analysis_contract"]["treatment"] == (
        "dynamic_refinement_with_static_memory_constraint"
    )
    assert updated_hypothesis["analysis_contract"]["control"] == (
        "unconstrained_dynamic_refinement_fastref"
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
        and item["analysis_contract"]["metric"] in {"image_auroc", "image_ap"}
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

    scoped = client.post(
        f"/api/v1/projects/{project_id}/approve",
        json={"approved_by": "api-tester"},
    )
    assert scoped.status_code == 409
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
            "hypothesis_id": approved.json()["experiment_plan"]["hypothesis_ids"][0],
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
