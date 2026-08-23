import hashlib
import json
from pathlib import Path

import pytest

from fsad_scientist.datasets.models import DatasetFileRecord, DatasetManifest
from fsad_scientist.domain.models import ExperimentRun, MethodImplementation
from fsad_scientist.experiments.preparation import ExperimentPreparationService
from fsad_scientist.experiments.strategy_runner import (
    GeneratedStrategyRunner,
    assemble_strategy_file,
)
from fsad_scientist.experiments.support_selection import plan_support_set
from fsad_scientist.features.dinov2 import DinoEmbeddingManifest

CUSTOM_NAME = "query_adaptive_a1b2c3d4"

CUSTOM_SOURCE = (
    "def select(candidate_ids, embeddings, k, seed):\n"
    "    generator = random.Random(seed)\n"
    "    return sorted(generator.sample(candidate_ids, k))\n"
)


def _implementation(status: str = "approved") -> MethodImplementation:
    assembled = assemble_strategy_file(CUSTOM_SOURCE)
    digest = hashlib.sha256(assembled.encode("utf-8")).hexdigest()
    return MethodImplementation(
        name=CUSTOM_NAME,
        hypothesis_id="hypothesis_x",
        source_code=CUSTOM_SOURCE,
        code_digest=digest,
        status=status,  # type: ignore[arg-type]
    )


def _dataset(root: str, count: int = 10) -> DatasetManifest:
    return DatasetManifest(
        dataset="MVTec AD",
        root=root,
        categories=["bottle"],
        files=[
            DatasetFileRecord(
                relative_path=f"bottle/train/good/{index:03d}.png",
                category="bottle",
                split="train",
                anomaly_type="good",
                kind="image",
                byte_size=10,
                sha256="a" * 64,
            )
            for index in range(count)
        ],
        counts={"bottle": count},
        digest="b" * 64,
    )


def _embeddings(count: int = 10) -> dict[str, list[float]]:
    return {
        f"bottle/train/good/{index:03d}.png": [0.1 + 0.01 * index] * 4
        for index in range(count)
    }


def _plan_args(runner, *, strategy=CUSTOM_NAME, **overrides):
    arguments = dict(
        category="bottle",
        protocol="pool_compression_m30",
        strategy=strategy,
        shots=2,
        seed=11,
        candidate_pool_size=8,
        embeddings=_embeddings(),
        feature_extractor="dinov2@test",
        custom_strategies={CUSTOM_NAME: _implementation()},
        strategy_runner=runner,
    )
    arguments.update(overrides)
    return arguments


class TestPlanSupportSetDispatch:
    def test_dispatches_approved_custom_strategy(self, tmp_path: Path) -> None:
        runner = GeneratedStrategyRunner(tmp_path / "artifacts")
        manifest = plan_support_set(_dataset("dataset_root"), **_plan_args(runner))
        assert manifest.strategy == CUSTOM_NAME
        assert len(manifest.selected_files) == 2
        assert set(manifest.selected_files) <= set(manifest.candidate_pool_files)

    def test_custom_strategy_is_deterministic_across_builds(self, tmp_path: Path) -> None:
        runner = GeneratedStrategyRunner(tmp_path / "artifacts")
        first = plan_support_set(_dataset("dataset_root"), **_plan_args(runner))
        second = plan_support_set(_dataset("dataset_root"), **_plan_args(runner))
        assert first.digest == second.digest

    def test_rejects_unapproved_custom_strategy(self, tmp_path: Path) -> None:
        runner = GeneratedStrategyRunner(tmp_path / "artifacts")
        arguments = _plan_args(runner)
        arguments["custom_strategies"] = {CUSTOM_NAME: _implementation(status="validated")}
        with pytest.raises(ValueError, match="not approved"):
            plan_support_set(_dataset("dataset_root"), **arguments)

    def test_rejects_missing_runner(self, tmp_path: Path) -> None:
        arguments = _plan_args(None, strategy_runner=None)
        with pytest.raises(ValueError, match="runner is required"):
            plan_support_set(_dataset("dataset_root"), **arguments)

    def test_rejects_unknown_strategy(self, tmp_path: Path) -> None:
        runner = GeneratedStrategyRunner(tmp_path / "artifacts")
        arguments = _plan_args(runner, strategy="missing_xyz")
        with pytest.raises(ValueError, match="unsupported selection strategy"):
            plan_support_set(_dataset("dataset_root"), **arguments)

    def test_strict_k_shot_rejects_custom_strategy(self, tmp_path: Path) -> None:
        runner = GeneratedStrategyRunner(tmp_path / "artifacts")
        arguments = _plan_args(runner, protocol="strict_k_shot")
        with pytest.raises(ValueError, match="strict_k_shot"):
            plan_support_set(_dataset("dataset_root"), **arguments)


class TestPreparationThreading:
    def test_prepare_threads_custom_strategy(self, tmp_path: Path) -> None:
        data_root = tmp_path / "data"
        for index in range(8):
            path = data_root / "bottle" / "train" / "good" / f"{index:03d}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fake-image")
        dataset = _dataset(str(data_root), count=8)
        embeddings = DinoEmbeddingManifest(
            dataset_digest=dataset.digest,
            category="bottle",
            model_id="facebook/dinov2-small",
            manifest_path="features/x.json",
            requested_revision="r",
            resolved_revision="r",
            pooling="normalized_cls_token",
            preprocessing="x",
            image_files=sorted(_embeddings(8)),
            embeddings=_embeddings(8),
            device="cpu",
            digest="c" * 64,
        )
        implementation = _implementation()
        run = ExperimentRun(
            plan_id="plan_x",
            hypothesis_id="hypothesis_x",
            protocol="pool_compression_m8",
            dataset="MVTec AD",
            category="bottle",
            detector="anomalydino",
            selection_strategy=CUSTOM_NAME,
            shots=2,
            seed=11,
        )
        service = ExperimentPreparationService(tmp_path / "artifacts")
        prepared = service.prepare(
            project_id="project_x",
            run=run,
            dataset=dataset,
            dataset_manifest_path=tmp_path / "manifest.json",
            embeddings=embeddings,
            candidate_pool_size=8,
            custom_strategies={CUSTOM_NAME: implementation},
            strategy_runner=GeneratedStrategyRunner(tmp_path / "artifacts"),
        )
        support = json.loads(Path(prepared.support_manifest_path).read_text(encoding="utf-8"))
        assert support["strategy"] == CUSTOM_NAME
        assert len(support["selected_files"]) == 2
        assert prepared.dataset_view_digest
