from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import uvicorn

from fsad_scientist.agents.mock_runtime import MockScientistRuntime
from fsad_scientist.config import get_settings
from fsad_scientist.datasets.models import DatasetManifest
from fsad_scientist.datasets.scanner import MvtecDatasetScanner
from fsad_scientist.datasets.view import DatasetViewBuilder
from fsad_scientist.domain.enums import ResearchStage
from fsad_scientist.domain.models import ProjectSpec
from fsad_scientist.evidence.search import LiteratureSearchService
from fsad_scientist.experiments.models import SupportSetManifest
from fsad_scientist.experiments.support_selection import plan_support_set
from fsad_scientist.features.dinov2 import DinoEmbeddingManifest, DinoV2Embedder
from fsad_scientist.repository import JsonProjectRepository
from fsad_scientist.workflow import ResearchWorkflow


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(prog="fsad-scientist")
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve = subparsers.add_parser("serve", help="Start the FastAPI service")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", default=8000, type=int)
    serve.add_argument("--reload", action="store_true")

    subparsers.add_parser(
        "demo-ledger",
        help="Create a mock project and advance it to the human approval gate",
    )

    search = subparsers.add_parser("search-papers", help="Search arXiv and Crossref")
    search.add_argument("query")
    search.add_argument("--limit", type=int, default=8)

    scan = subparsers.add_parser("scan-dataset", help="Audit a MVTec AD directory")
    scan.add_argument("root", type=Path)
    scan.add_argument("--name", default="MVTec AD")

    profile = subparsers.add_parser(
        "profile-dinov2",
        help="Extract cached train/good DINOv2 embeddings",
    )
    profile.add_argument("manifest", type=Path)
    profile.add_argument("category")
    profile.add_argument("--model", default=None)
    profile.add_argument("--device", default="auto")
    profile.add_argument("--batch-size", type=int, default=8)

    support = subparsers.add_parser("plan-support", help="Freeze a support-set manifest")
    support.add_argument("manifest", type=Path)
    support.add_argument("category")
    support.add_argument("--protocol", required=True)
    support.add_argument("--strategy", required=True)
    support.add_argument("--shots", required=True, type=int)
    support.add_argument("--seed", type=int, default=0)
    support.add_argument("--pool-size", type=int, default=30)
    support.add_argument("--embeddings", type=Path)

    view = subparsers.add_parser("build-view", help="Materialize an immutable dataset view")
    view.add_argument("dataset_manifest", type=Path)
    view.add_argument("support_manifest", type=Path)

    args = parser.parse_args()
    if args.command == "serve":
        uvicorn.run(
            "fsad_scientist.main:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
        )
    elif args.command == "demo-ledger":
        asyncio.run(_create_demo_ledger())
    elif args.command == "search-papers":
        asyncio.run(_search_papers(args.query, args.limit))
    elif args.command == "scan-dataset":
        _scan_dataset(args.root, args.name)
    elif args.command == "profile-dinov2":
        _profile_dinov2(
            args.manifest,
            args.category,
            model=args.model,
            device=args.device,
            batch_size=args.batch_size,
        )
    elif args.command == "plan-support":
        _plan_support(
            args.manifest,
            args.category,
            protocol=args.protocol,
            strategy=args.strategy,
            shots=args.shots,
            seed=args.seed,
            pool_size=args.pool_size,
            embeddings_path=args.embeddings,
        )
    elif args.command == "build-view":
        _build_view(args.dataset_manifest, args.support_manifest)


async def _create_demo_ledger() -> None:
    settings = get_settings()
    workflow = ResearchWorkflow(
        repository=JsonProjectRepository(settings.storage_path),
        runtime=MockScientistRuntime(),
    )
    project = workflow.create_project(ProjectSpec())
    while project.stage != ResearchStage.AWAITING_EXPERIMENT_APPROVAL:
        project = await workflow.advance(project.id)
    print(
        json.dumps(
            {
                "project_id": project.id,
                "stage": project.stage,
                "next_action": project.next_action,
                "hypotheses": [item.title for item in project.hypotheses],
                "plan_digest": project.experiment_plan.preregistration_digest
                if project.experiment_plan
                else None,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


async def _search_papers(query: str, limit: int) -> None:
    settings = get_settings()
    result = await LiteratureSearchService(mailto=settings.evidence_mailto).search(
        query,
        limit=limit,
    )
    print(result.model_dump_json(indent=2))


def _scan_dataset(root: Path, name: str) -> None:
    settings = get_settings()
    scanner = MvtecDatasetScanner()
    manifest = scanner.scan(root, dataset_name=name)
    destination = settings.artifact_path / "datasets" / f"{manifest.digest}.json"
    scanner.save(manifest, destination)
    _print_artifact(manifest, destination)


def _profile_dinov2(
    manifest_path: Path,
    category: str,
    *,
    model: str | None,
    device: str,
    batch_size: int,
) -> None:
    settings = get_settings()
    dataset = DatasetManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    result = DinoV2Embedder(
        settings.artifact_path,
        model_id=model or settings.dinov2_profile_model,
        device=device,
        batch_size=batch_size,
    ).extract(dataset, category=category)
    fallback_path = settings.artifact_path / "features" / dataset.digest[:16]
    _print_artifact(result, Path(result.manifest_path or fallback_path))


def _plan_support(
    manifest_path: Path,
    category: str,
    *,
    protocol: str,
    strategy: str,
    shots: int,
    seed: int,
    pool_size: int,
    embeddings_path: Path | None,
) -> None:
    settings = get_settings()
    dataset = DatasetManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    embeddings = None
    extractor = "none"
    if embeddings_path:
        feature_manifest = DinoEmbeddingManifest.model_validate_json(
            embeddings_path.read_text(encoding="utf-8")
        )
        embeddings = feature_manifest.embeddings
        revision = feature_manifest.resolved_revision or "floating"
        extractor = f"{feature_manifest.model_id}@{revision}"
    result = plan_support_set(
        dataset,
        category=category,
        protocol=protocol,
        strategy=strategy,
        shots=shots,
        seed=seed,
        candidate_pool_size=pool_size,
        embeddings=embeddings,
        feature_extractor=extractor,
    )
    destination = settings.artifact_path / "support_sets" / f"{result.digest}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    temporary.replace(destination)
    _print_artifact(result, destination)


def _build_view(dataset_path: Path, support_path: Path) -> None:
    settings = get_settings()
    dataset = DatasetManifest.model_validate_json(dataset_path.read_text(encoding="utf-8"))
    support = SupportSetManifest.model_validate_json(support_path.read_text(encoding="utf-8"))
    result = DatasetViewBuilder(settings.artifact_path).build(dataset, support)
    print(result.model_dump_json(indent=2))


def _print_artifact(model: object, path: Path) -> None:
    payload = json.loads(model.model_dump_json())  # type: ignore[attr-defined]
    payload["artifact_path"] = str(path.resolve())
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
