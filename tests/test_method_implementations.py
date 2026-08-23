from pathlib import Path

import pytest

from fsad_scientist.domain.models import (
    ExperimentPlan,
    MethodImplementation,
    ResearchProject,
)
from fsad_scientist.repository import JsonProjectRepository

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VALID_DIGEST = "a" * 64


def _minimal_plan() -> ExperimentPlan:
    return ExperimentPlan(
        hypothesis_ids=["hypothesis_x"],
        protocols=["pool_compression_m30"],
        detectors=["anomalydino"],
        selection_strategies=["random", "k_center"],
        datasets=["MVTec AD"],
        categories=["bottle"],
        shots=[2],
        seeds=[11],
        metrics=["image_auroc"],
        analysis_methods=["paired_test"],
        stages=["feasibility"],
        stopping_conditions=["minimum_pairs"],
        estimated_gpu_hours=1.0,
        preregistration_digest="prereg",
    )


class TestMethodImplementationModel:
    def test_defaults(self) -> None:
        source = "def select(candidate_ids, embeddings, k, seed):\n    return candidate_ids[:k]\n"
        impl = MethodImplementation(
            name="query_adaptive_abc12345",
            hypothesis_id="hypothesis_x",
            source_code=source,
            code_digest=VALID_DIGEST,
        )
        assert impl.kind == "selection_strategy"
        assert impl.status == "draft"
        assert impl.static_validation.passed is False
        assert impl.smoke_result is None
        assert impl.artifact_path is None
        assert impl.provenance == []


class TestBackwardCompatibility:
    def test_minimal_project_json_loads_without_new_fields(self) -> None:
        project = ResearchProject.model_validate_json('{"spec": {}}')
        assert project.method_implementations == []
        assert project.experiment_campaign is None

    def test_plan_digest_map_defaults(self) -> None:
        plan = _minimal_plan()
        assert plan.method_implementation_digests == {}

    def test_stored_projects_load_with_new_defaults(self) -> None:
        storage = PROJECT_ROOT / "storage" / "projects"
        if not storage.is_dir():
            pytest.skip("storage directory not present in this checkout")
        projects = JsonProjectRepository(storage).list()
        if not projects:
            pytest.skip("storage directory holds no projects")
        for project in projects:
            assert project.method_implementations == []
            for plan in [project.experiment_plan, *project.experiment_plan_history]:
                if plan is not None:
                    assert plan.method_implementation_digests == {}
