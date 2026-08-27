import json
from pathlib import Path

from fsad_scientist.domain.models import (
    ExperimentPlan,
    MethodImplementation,
    ResearchProject,
)
from fsad_scientist.repository import JsonProjectRepository

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

    def test_stored_projects_load_with_new_defaults(self, tmp_path: Path) -> None:
        legacy = ResearchProject(spec={}, experiment_plan=_minimal_plan())
        payload = legacy.model_dump(mode="json")
        payload.pop("method_implementations")
        payload["experiment_plan"].pop("method_implementation_digests")

        project_dir = tmp_path / legacy.id
        project_dir.mkdir()
        (project_dir / "project.json").write_text(
            json.dumps(payload),
            encoding="utf-8",
        )

        projects = JsonProjectRepository(tmp_path).list()
        assert len(projects) == 1
        assert projects[0].method_implementations == []
        assert projects[0].experiment_plan is not None
        assert projects[0].experiment_plan.method_implementation_digests == {}
