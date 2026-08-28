"""End-to-end smoke for the AI-generated detector pipeline (CPU-only).

Exercises: detector implementation via the deterministic runtime -> static
validation -> real two-run behavioral smoke on a synthetic MVTec view ->
human approval gate -> campaign initialization with the generated detector.
No real campaign execution is performed.
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from pathlib import Path

from fsad_scientist.agents.mock_runtime import MockScientistRuntime
from fsad_scientist.config import PROJECT_ROOT
from fsad_scientist.datasets.scanner import MvtecDatasetScanner
from fsad_scientist.datasets.synthetic import build_synthetic_mvtec_smoke_dataset
from fsad_scientist.domain.enums import ResearchStage
from fsad_scientist.domain.models import ComputeBudget, ProjectSpec
from fsad_scientist.repository import JsonProjectRepository
from fsad_scientist.workflow import ResearchWorkflow


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    smoke_root = PROJECT_ROOT / "artifacts" / "smoke" / "generated_detector"
    dataset_root = smoke_root / "synthetic_mvtec"
    build_synthetic_mvtec_smoke_dataset(dataset_root)
    dataset = MvtecDatasetScanner().scan(dataset_root, dataset_name="MVTec AD")
    dataset_path = smoke_root / "dataset_manifest.json"
    MvtecDatasetScanner().save(dataset, dataset_path)

    with tempfile.TemporaryDirectory(prefix="fsad-smoke-detector-ledger-") as ledger:
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
            workflow.implement_experiment_detector(
                project.id,
                name_stem="nearest_prototype",
                hypothesis_id=hypothesis.id,
                reference_description="PatchCore 式最近邻记忆库",
            )
        )
        implementation = next(
            item for item in project.method_implementations if item.kind == "detector"
        )
        if implementation.status != "validated":
            _fail(f"detector not validated: {implementation.status}")

        assert project.experiment_plan is not None
        if implementation.name not in project.experiment_plan.detectors:
            project.experiment_plan.detectors.append(implementation.name)
        project = workflow.repository.save(project)

        approved = workflow.approve_experiment_plan(project.id, approved_by="smoke-reviewer")
        implementation = next(
            item
            for item in approved.method_implementations
            if item.kind == "detector"
        )
        if implementation.status != "approved":
            _fail(f"detector not approved after plan approval: {implementation.status}")

        project = workflow.attach_dataset_audit(
            approved.id,
            manifest=dataset,
            manifest_path=str(dataset_path),
        )
        project = workflow.initialize_experiment_campaign(
            project.id,
            dataset=dataset,
            hypothesis_id=hypothesis.id,
            detector=implementation.name,
            device="cpu",
            max_rounds=3,
            max_runs=8,
        )
        campaign = project.experiment_campaign
        if campaign is None:
            _fail("campaign was not initialized")
        if campaign.detector != implementation.name:
            _fail(f"campaign detector {campaign.detector} != {implementation.name}")
        if any(run.detector != implementation.name for run in project.runs):
            _fail("queued runs do not use the generated detector")

        print(
            json.dumps(
                {
                    "purpose": "generated_detector_pipeline_smoke_only",
                    "scientific_result": False,
                    "detector_name": implementation.name,
                    "code_digest": implementation.code_digest,
                    "implementation_status": implementation.status,
                    "smoke_summary": implementation.smoke_result.summary
                    if implementation.smoke_result
                    else None,
                    "campaign": {
                        "detector": campaign.detector,
                        "treatment": campaign.treatment,
                        "control": campaign.control,
                    },
                    "queued_runs": len(project.runs),
                    "artifact_path": implementation.artifact_path,
                },
                ensure_ascii=False,
                indent=2,
            )
        )


def _fail(message: str) -> None:
    payload = {
        "purpose": "generated_detector_pipeline_smoke_only",
        "failed": message,
    }
    print(json.dumps(payload, ensure_ascii=False))
    raise SystemExit(1)


if __name__ == "__main__":
    main()
