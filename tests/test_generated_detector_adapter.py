from pathlib import Path

import pytest

from fsad_scientist.domain.models import (
    ExperimentRun,
    MethodImplementation,
    ProjectSpec,
    ResearchProject,
)
from fsad_scientist.experiments.adapters import (
    MethodRegistry,
    build_generated_detector_command,
    resolve_detector_command,
)


def _run(detector: str = "generated_det_nearest_ab12cd34") -> ExperimentRun:
    return ExperimentRun(
        plan_id="plan_x",
        hypothesis_id="hypothesis_x",
        protocol="pool_compression_m30",
        dataset="MVTec AD",
        category="bottle",
        detector=detector,
        selection_strategy="random",
        shots=2,
        seed=11,
    )


def _implementation(
    status: str = "approved",
    artifact_path: Path | None = None,
) -> MethodImplementation:
    return MethodImplementation(
        kind="detector",
        name="generated_det_nearest_ab12cd34",
        hypothesis_id="hypothesis_x",
        source_code="def anomaly_score(image, support_images, seed):\n    return 1.0\n",
        code_digest="f" * 64,
        status=status,  # type: ignore[arg-type]
        artifact_path=str(artifact_path) if artifact_path else None,
    )


def _project(implementation: MethodImplementation) -> ResearchProject:
    project = ResearchProject(spec=ProjectSpec())
    project.method_implementations.append(implementation)
    return project


class TestBuildGeneratedDetectorCommand:
    def test_command_shape(self, tmp_path: Path) -> None:
        script = tmp_path / "detector.py"
        script.write_text("def anomaly_score(image, support_images, seed):\n    return 1.0\n")
        command = build_generated_detector_command(
            _run(),
            _implementation(artifact_path=script),
            dataset_view=tmp_path / "view",
            output_dir=tmp_path / "output",
            device="cuda:1",
        )
        assert command.method == "generated_det_nearest_ab12cd34"
        assert command.executable == "python"
        assert command.cwd == tmp_path / "output"
        assert "--data_root" in command.args
        assert "--category" in command.args and "bottle" in command.args
        assert "--shots" in command.args and "2" in command.args
        assert "--seed" in command.args and "11" in command.args
        assert "--output" in command.args
        assert "--device" in command.args and "cuda:1" in command.args
        assert command.environment["HF_HUB_OFFLINE"] == "1"
        assert command.environment["TRANSFORMERS_OFFLINE"] == "1"
        assert command.environment["CUDA_VISIBLE_DEVICES"] == "1"
        assert "metrics.json" in command.expected_outputs

    def test_cpu_device_clears_visible_devices(self, tmp_path: Path) -> None:
        script = tmp_path / "detector.py"
        script.write_text("pass\n")
        command = build_generated_detector_command(
            _run(),
            _implementation(artifact_path=script),
            dataset_view=tmp_path / "view",
            output_dir=tmp_path / "output",
            device="cpu",
        )
        assert command.environment["CUDA_VISIBLE_DEVICES"] == ""

    def test_missing_artifact_path_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="没有已写入的代码文件"):
            build_generated_detector_command(
                _run(),
                _implementation(),
                dataset_view=tmp_path / "view",
                output_dir=tmp_path / "output",
                device="cpu",
            )


class TestResolveDetectorCommand:
    def test_builtin_delegates_to_registry(self, tmp_path: Path) -> None:
        project = ResearchProject(spec=ProjectSpec())
        registry = MethodRegistry(tmp_path)
        command = resolve_detector_command(
            project,
            _run(detector="anomalydino"),
            registry,
            dataset_view=tmp_path / "view",
            output_dir=tmp_path / "output",
            device="cpu",
        )
        assert command.method == "anomalydino"

    def test_generated_uses_approved_implementation(self, tmp_path: Path) -> None:
        script = tmp_path / "detector.py"
        script.write_text("pass\n")
        project = _project(_implementation(artifact_path=script))
        command = resolve_detector_command(
            project,
            _run(),
            MethodRegistry(tmp_path),
            dataset_view=tmp_path / "view",
            output_dir=tmp_path / "output",
            device="cpu",
        )
        assert command.method == "generated_det_nearest_ab12cd34"

    def test_generated_validated_only_rejected(self, tmp_path: Path) -> None:
        script = tmp_path / "detector.py"
        script.write_text("pass\n")
        project = _project(_implementation(status="validated", artifact_path=script))
        with pytest.raises(ValueError, match="没有已批准的实现"):
            resolve_detector_command(
                project,
                _run(),
                MethodRegistry(tmp_path),
                dataset_view=tmp_path / "view",
                output_dir=tmp_path / "output",
                device="cpu",
            )

    def test_generated_missing_implementation_rejected(self, tmp_path: Path) -> None:
        project = ResearchProject(spec=ProjectSpec())
        with pytest.raises(ValueError, match="没有已批准的实现"):
            resolve_detector_command(
                project,
                _run(),
                MethodRegistry(tmp_path),
                dataset_view=tmp_path / "view",
                output_dir=tmp_path / "output",
                device="cpu",
            )
