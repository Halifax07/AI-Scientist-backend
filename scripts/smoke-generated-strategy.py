"""End-to-end smoke for the AI-generated selection strategy pipeline (no GPU).

Exercises: hypothesis operationalization -> deterministic runtime implementation ->
static validation -> behavioral smoke test -> human approval gate -> campaign
initialization with the custom strategy -> real preparation (support manifest and
dataset view) dispatching the custom strategy out-of-process.
"""

from __future__ import annotations

import asyncio
import json
import random
import shutil
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

from fsad_scientist.agents.mock_runtime import MockScientistRuntime
from fsad_scientist.config import PROJECT_ROOT
from fsad_scientist.datasets.scanner import MvtecDatasetScanner
from fsad_scientist.domain.enums import HypothesisStatus, ResearchStage
from fsad_scientist.domain.models import ComputeBudget, ProjectSpec
from fsad_scientist.experiments.preparation import ExperimentPreparationService
from fsad_scientist.experiments.strategy_runner import GeneratedStrategyRunner
from fsad_scientist.features.dinov2 import DinoEmbeddingManifest
from fsad_scientist.repository import JsonProjectRepository
from fsad_scientist.workflow import ResearchWorkflow


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    smoke_root = PROJECT_ROOT / "artifacts" / "smoke" / "generated_strategy"
    dataset_root = smoke_root / "synthetic_mvtec"
    _build_synthetic_dataset(dataset_root)

    dataset = MvtecDatasetScanner().scan(dataset_root, dataset_name="MVTec AD")
    dataset_path = smoke_root / "dataset_manifest.json"
    MvtecDatasetScanner().save(dataset, dataset_path)

    with tempfile.TemporaryDirectory(prefix="fsad-smoke-ledger-") as ledger:
        workflow = ResearchWorkflow(
            repository=JsonProjectRepository(Path(ledger)),
            runtime=MockScientistRuntime(),
            artifact_root=PROJECT_ROOT / "artifacts",
        )
        project = workflow.create_project(
            ProjectSpec(budget=ComputeBudget(max_experiments=8))
        )
        while project.stage != ResearchStage.AWAITING_EXPERIMENT_APPROVAL:
            project = asyncio.run(workflow.advance(project.id))

        hypothesis = next(
            item
            for item in project.hypotheses
            if item.analysis_contract is not None
            and item.analysis_contract.kind == "query_adaptation"
        )
        project = asyncio.run(
            workflow.implement_experiment_method(project.id, hypothesis_id=hypothesis.id)
        )
        implementation = next(
            item
            for item in project.method_implementations
            if item.hypothesis_id == hypothesis.id
        )
        if implementation.status != "validated":
            _fail(f"implementation not validated: {implementation.status}")

        hypothesis = next(
            item for item in project.hypotheses if item.id == hypothesis.id
        )
        hypothesis.status = HypothesisStatus.SHORTLISTED
        assert project.experiment_plan is not None
        project.experiment_plan.hypothesis_ids = [hypothesis.id]
        project = workflow.repository.save(project)

        approved = workflow.approve_experiment_plan(project.id, approved_by="smoke-reviewer")
        implementation = next(
            item
            for item in approved.method_implementations
            if item.hypothesis_id == hypothesis.id
        )
        if implementation.status != "approved":
            _fail(f"implementation not approved after plan approval: {implementation.status}")

        project = workflow.attach_dataset_audit(
            approved.id,
            manifest=dataset,
            manifest_path=str(dataset_path),
        )
        project = workflow.initialize_experiment_campaign(
            project.id,
            dataset=dataset,
            detector="anomalydino",
            max_rounds=3,
            max_runs=8,
        )
        campaign = project.experiment_campaign
        if campaign is None:
            _fail("campaign was not initialized")
        if campaign.treatment != implementation.name:
            _fail(f"campaign treatment {campaign.treatment} != {implementation.name}")

        custom_runs = [run for run in project.runs if run.selection_strategy == implementation.name]
        if not custom_runs:
            _fail("no queued run uses the custom strategy")
        run = custom_runs[0]

        vectors = _fake_embeddings(dataset, category=run.category)
        embeddings = DinoEmbeddingManifest(
            dataset_digest=dataset.digest,
            category=run.category,
            model_id="smoke/fake-embedder",
            manifest_path=str(smoke_root / "fake_embedding_manifest.json"),
            requested_revision="smoke",
            resolved_revision="smoke",
            pooling="smoke_unit_norm",
            preprocessing="none",
            image_files=sorted(vectors),
            embeddings=vectors,
            device="cpu",
            digest="c" * 64,
        )
        prepared = ExperimentPreparationService(PROJECT_ROOT / "artifacts").prepare(
            project_id=project.id,
            run=run,
            dataset=dataset,
            dataset_manifest_path=dataset_path,
            embeddings=embeddings,
            candidate_pool_size=campaign.candidate_pool_size,
            custom_strategies={implementation.name: implementation},
            strategy_runner=GeneratedStrategyRunner(PROJECT_ROOT / "artifacts"),
        )
        support = json.loads(Path(prepared.support_manifest_path).read_text(encoding="utf-8"))
        if support["strategy"] != implementation.name:
            _fail(f"support manifest strategy {support['strategy']} != {implementation.name}")
        if set(support["selected_files"]) - set(support["candidate_pool_files"]):
            _fail("selected files escape the candidate pool")
        if len(support["selected_files"]) != run.shots:
            _fail(f"selected {len(support['selected_files'])} files, expected {run.shots}")

        print(
            json.dumps(
                {
                    "purpose": "generated_strategy_pipeline_smoke_only",
                    "scientific_result": False,
                    "strategy_name": implementation.name,
                    "code_digest": implementation.code_digest,
                    "implementation_status": implementation.status,
                    "smoke_summary": implementation.smoke_result.summary
                    if implementation.smoke_result
                    else None,
                    "campaign": {
                        "treatment": campaign.treatment,
                        "control": campaign.control,
                    },
                    "dataset_digest": dataset.digest,
                    "support_digest": support["digest"],
                    "selected_files": support["selected_files"],
                    "dataset_view_digest": prepared.dataset_view_digest,
                    "artifact_path": implementation.artifact_path,
                },
                ensure_ascii=False,
                indent=2,
            )
        )


def _fake_embeddings(dataset, *, category: str) -> dict[str, list[float]]:
    generator = random.Random(17)
    candidates = dataset.support_candidates(category)
    vectors = {}
    for index, relative_path in enumerate(candidates):
        vector = [round(generator.random(), 6) for _ in range(8)]
        norm = sum(value * value for value in vector) ** 0.5 or 1.0
        vectors[relative_path] = [round(value / norm, 6) for value in vector]
        if index >= 7:
            break
    return vectors


def _build_synthetic_dataset(root: Path) -> None:
    if root.exists():
        shutil.rmtree(root)
    training = root / "bottle" / "train" / "good"
    test_good = root / "bottle" / "test" / "good"
    test_bad = root / "bottle" / "test" / "broken"
    masks = root / "bottle" / "ground_truth" / "broken"
    for directory in (training, test_good, test_bad, masks):
        directory.mkdir(parents=True, exist_ok=True)

    for index, color in enumerate(
        ((40, 100, 180), (60, 120, 190), (80, 90, 170), (45, 135, 165), (70, 105, 175))
    ):
        image = Image.new("RGB", (224, 224), color)
        draw = ImageDraw.Draw(image)
        draw.ellipse((62 + index, 28, 162 + index, 202), outline="white", width=8)
        image.save(training / f"{index:03}.png")

    Image.new("RGB", (224, 224), (55, 110, 180)).save(test_good / "100.png")
    bad = Image.new("RGB", (224, 224), (55, 110, 180))
    ImageDraw.Draw(bad).rectangle((90, 90, 135, 135), fill=(230, 30, 30))
    bad.save(test_bad / "101.png")
    mask = Image.new("L", (224, 224), 0)
    ImageDraw.Draw(mask).rectangle((90, 90, 135, 135), fill=255)
    mask.save(masks / "101_mask.png")


def _fail(message: str) -> None:
    payload = {
        "purpose": "generated_strategy_pipeline_smoke_only",
        "failed": message,
    }
    print(json.dumps(payload, ensure_ascii=False))
    raise SystemExit(1)


if __name__ == "__main__":
    main()
