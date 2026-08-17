from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated, TypeVar, cast

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from fsad_scientist import __version__
from fsad_scientist.agents.agentscope_client import AgentScopeUnavailableError
from fsad_scientist.agents.contracts import ScientistRuntime
from fsad_scientist.agents.evidence_runtime import EvidenceEnabledRuntime
from fsad_scientist.agents.mock_runtime import MockScientistRuntime
from fsad_scientist.agents.qwen_runtime import QwenScientistRuntime
from fsad_scientist.api.schemas import (
    ApprovalRequest,
    ClaimVerifyRequest,
    CreateProjectRequest,
    DatasetScanRequest,
    DatasetViewRequest,
    DinoExtractRequest,
    EvidenceSearchRequest,
    EvidenceVerifyRequest,
    ExecuteNextExperimentRequest,
    ExecuteNextExperimentResponse,
    ExecuteRunRequest,
    FullTextRequest,
    GenerateDetectorRequest,
    GenerateMethodRequest,
    HealthResponse,
    InitializeExperimentCampaignRequest,
    PrepareRunRequest,
    ProjectDatasetAuditRequest,
    RunResultRequest,
    StartNextResearchCycleRequest,
    SupportPlanRequest,
)
from fsad_scientist.config import Settings, get_settings
from fsad_scientist.datasets.models import DatasetManifest, DatasetViewManifest
from fsad_scientist.datasets.scanner import MvtecDatasetScanner
from fsad_scientist.datasets.view import DatasetViewBuilder
from fsad_scientist.domain.models import (
    EvidenceRecord,
    ExperimentRun,
    MethodImplementation,
    ProjectSpec,
    ResearchProject,
)
from fsad_scientist.evidence.claims import QwenClaimVerifier
from fsad_scientist.evidence.fulltext import ArxivFullTextService, FullTextDocument
from fsad_scientist.evidence.search import LiteratureSearchResult, LiteratureSearchService
from fsad_scientist.experiments.adapters import MethodRegistry, resolve_detector_command
from fsad_scientist.experiments.models import (
    ExecutionRecord,
    PreparedRunArtifacts,
    SupportSetManifest,
)
from fsad_scientist.experiments.preparation import ExperimentPreparationService
from fsad_scientist.experiments.runner import ExperimentRunner
from fsad_scientist.experiments.strategy_runner import GeneratedStrategyRunner
from fsad_scientist.experiments.support_selection import plan_support_set
from fsad_scientist.features.dinov2 import DinoEmbeddingManifest, DinoV2Embedder
from fsad_scientist.repository import JsonProjectRepository, ProjectNotFoundError
from fsad_scientist.workflow import ResearchWorkflow, WorkflowError


def _get_workflow(request: Request) -> ResearchWorkflow:
    return cast(ResearchWorkflow, request.app.state.workflow)


WorkflowDependency = Annotated[ResearchWorkflow, Depends(_get_workflow)]


def _get_evidence_service(request: Request) -> LiteratureSearchService:
    return cast(LiteratureSearchService, request.app.state.evidence_service)


EvidenceDependency = Annotated[LiteratureSearchService, Depends(_get_evidence_service)]


def create_app(
    *,
    settings: Settings | None = None,
    storage_path: Path | None = None,
    runtime: ScientistRuntime | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    evidence_service = LiteratureSearchService(mailto=settings.evidence_mailto)
    runtime = runtime or _build_runtime(settings, evidence_service=evidence_service)
    repository = JsonProjectRepository(storage_path or settings.storage_path)
    workflow = ResearchWorkflow(
        repository=repository,
        runtime=runtime,
        artifact_root=settings.artifact_path,
    )
    settings.artifact_path.mkdir(parents=True, exist_ok=True)

    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        description=(
            "Auditable autonomous research workflow for few-shot industrial visual "
            "anomaly detection."
        ),
    )
    app.state.workflow = workflow
    app.state.runtime_name = runtime.name
    app.state.evidence_service = evidence_service
    app.state.settings = settings
    app.state.experiment_locks = {}
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(ProjectNotFoundError)
    async def project_not_found(_: Request, exc: ProjectNotFoundError):
        return _error_response(status.HTTP_404_NOT_FOUND, f"Project not found: {exc.args[0]}")

    @app.exception_handler(WorkflowError)
    async def workflow_error(_: Request, exc: WorkflowError):
        return _error_response(status.HTTP_409_CONFLICT, str(exc))

    @app.exception_handler(AgentScopeUnavailableError)
    async def agent_runtime_unavailable(_: Request, exc: AgentScopeUnavailableError):
        return _error_response(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc))

    @app.get("/health", response_model=HealthResponse)
    async def health(request: Request) -> HealthResponse:
        return HealthResponse(
            status="ok",
            runtime=cast(str, request.app.state.runtime_name),
            version=__version__,
        )

    @app.get("/api/v1/projects", response_model=list[ResearchProject])
    async def list_projects(
        workflow: WorkflowDependency,
    ) -> list[ResearchProject]:
        return workflow.repository.list()

    @app.post(
        "/api/v1/projects",
        response_model=ResearchProject,
        status_code=status.HTTP_201_CREATED,
    )
    async def create_project(
        body: CreateProjectRequest,
        workflow: WorkflowDependency,
    ) -> ResearchProject:
        return workflow.create_project(body.spec)

    @app.post(
        "/api/v1/projects/demo",
        response_model=ResearchProject,
        status_code=status.HTTP_201_CREATED,
    )
    async def create_demo_project(
        workflow: WorkflowDependency,
    ) -> ResearchProject:
        return workflow.create_project(ProjectSpec())

    @app.get("/api/v1/projects/{project_id}", response_model=ResearchProject)
    async def get_project(
        project_id: str,
        workflow: WorkflowDependency,
    ) -> ResearchProject:
        return workflow.repository.get(project_id)

    @app.post("/api/v1/evidence/search", response_model=LiteratureSearchResult)
    async def search_evidence(
        body: EvidenceSearchRequest,
        service: EvidenceDependency,
    ) -> LiteratureSearchResult:
        return await service.search(
            body.query,
            limit=body.limit,
            providers=tuple(body.providers),
        )

    @app.post("/api/v1/evidence/verify", response_model=EvidenceRecord)
    async def verify_evidence(
        body: EvidenceVerifyRequest,
        service: EvidenceDependency,
    ) -> EvidenceRecord:
        return await service.verify(body.record)

    @app.post("/api/v1/evidence/fulltext", response_model=FullTextDocument)
    async def retrieve_fulltext(
        body: FullTextRequest,
        request: Request,
    ) -> FullTextDocument:
        settings = cast(Settings, request.app.state.settings)
        return await ArxivFullTextService(settings.artifact_path).fetch_and_extract(
            body.record,
            force=body.force,
        )

    @app.post("/api/v1/evidence/claims/verify", response_model=EvidenceRecord)
    async def verify_claims(
        body: ClaimVerifyRequest,
        request: Request,
    ) -> EvidenceRecord:
        settings = cast(Settings, request.app.state.settings)
        document = _load_artifact_model(
            body.fulltext_manifest_path,
            settings.artifact_path,
            FullTextDocument,
        )
        return await QwenClaimVerifier(
            model=settings.reasoning_model,
            api_key=settings.dashscope_api_key_value,
        ).verify(
            body.record,
            document,
        )

    @app.post(
        "/api/v1/projects/{project_id}/evidence/search",
        response_model=ResearchProject,
    )
    async def search_and_attach_evidence(
        project_id: str,
        body: EvidenceSearchRequest,
        workflow: WorkflowDependency,
        service: EvidenceDependency,
    ) -> ResearchProject:
        result = await service.search(
            body.query,
            limit=body.limit,
            providers=tuple(body.providers),
        )
        return workflow.attach_evidence(project_id, evidence=result.records)

    @app.post("/api/v1/datasets/scan", response_model=DatasetManifest)
    async def scan_dataset(body: DatasetScanRequest, request: Request) -> DatasetManifest:
        scanner = MvtecDatasetScanner()
        manifest = await asyncio.to_thread(
            scanner.scan,
            Path(body.root),
            dataset_name=body.dataset_name,
        )
        artifact_root = cast(Settings, request.app.state.settings).artifact_path
        scanner.save(
            manifest,
            artifact_root / "datasets" / f"{manifest.digest}.json",
        )
        return manifest

    @app.post(
        "/api/v1/projects/{project_id}/dataset/audit",
        response_model=ResearchProject,
    )
    async def audit_project_dataset(
        project_id: str,
        body: ProjectDatasetAuditRequest,
        workflow: WorkflowDependency,
        request: Request,
    ) -> ResearchProject:
        settings = cast(Settings, request.app.state.settings)
        artifact_root = settings.artifact_path
        manifest_path = artifact_root / "datasets" / "mock_manifest.json"

        if settings.runtime == "mock":
            manifest = _create_mock_manifest(body.dataset_name, str(Path(body.root).resolve()))
            manifest_path = artifact_root / "datasets" / f"{manifest.digest}.json"
            MvtecDatasetScanner.save(manifest, manifest_path)
        else:
            scanner = MvtecDatasetScanner()
            manifest = await asyncio.to_thread(
                scanner.scan,
                Path(body.root),
                dataset_name=body.dataset_name,
            )
            manifest_path = artifact_root / "datasets" / f"{manifest.digest}.json"
            scanner.save(manifest, manifest_path)

        return workflow.attach_dataset_audit(
            project_id,
            manifest=manifest,
            manifest_path=str(manifest_path.resolve()),
        )

    @app.post("/api/v1/features/dinov2/extract", response_model=DinoEmbeddingManifest)
    async def extract_dinov2(
        body: DinoExtractRequest,
        request: Request,
    ) -> DinoEmbeddingManifest:
        settings = cast(Settings, request.app.state.settings)
        dataset = _load_artifact_model(
            body.dataset_manifest_path,
            settings.artifact_path,
            DatasetManifest,
        )
        embedder = DinoV2Embedder(
            settings.artifact_path,
            model_id=body.model_id,
            revision=body.revision,
            device=body.device,
            batch_size=body.batch_size,
        )
        return await asyncio.to_thread(
            embedder.extract,
            dataset,
            category=body.category,
            image_files=body.image_files,
            force=body.force,
        )

    @app.post("/api/v1/support-sets/plan", response_model=SupportSetManifest)
    async def build_support_plan(
        body: SupportPlanRequest,
        request: Request,
    ) -> SupportSetManifest:
        settings = cast(Settings, request.app.state.settings)
        dataset = _load_artifact_model(
            body.dataset_manifest_path,
            settings.artifact_path,
            DatasetManifest,
        )
        embeddings = None
        feature_extractor = "none"
        if body.embedding_manifest_path:
            feature_manifest = _load_artifact_model(
                body.embedding_manifest_path,
                settings.artifact_path,
                DinoEmbeddingManifest,
            )
            if feature_manifest.dataset_digest != dataset.digest:
                raise HTTPException(409, "embedding manifest belongs to another dataset")
            embeddings = feature_manifest.embeddings
            feature_revision = (
                feature_manifest.resolved_revision
                or feature_manifest.requested_revision
                or "floating"
            )
            feature_extractor = (
                f"{feature_manifest.model_id}@{feature_revision}"
            )
        support = plan_support_set(
            dataset,
            category=body.category,
            protocol=body.protocol,
            strategy=body.strategy,
            shots=body.shots,
            seed=body.seed,
            candidate_pool_size=body.candidate_pool_size,
            embeddings=embeddings,
            feature_extractor=feature_extractor,
        )
        path = settings.artifact_path / "support_sets" / f"{support.digest}.json"
        _write_model(path, support)
        return support

    @app.post("/api/v1/dataset-views/build", response_model=DatasetViewManifest)
    async def build_dataset_view(
        body: DatasetViewRequest,
        request: Request,
    ) -> DatasetViewManifest:
        settings = cast(Settings, request.app.state.settings)
        dataset = _load_artifact_model(
            body.dataset_manifest_path,
            settings.artifact_path,
            DatasetManifest,
        )
        support = _load_artifact_model(
            body.support_manifest_path,
            settings.artifact_path,
            SupportSetManifest,
        )
        builder = DatasetViewBuilder(settings.artifact_path)
        return await asyncio.to_thread(
            builder.build,
            dataset,
            support,
            include_test=body.include_test,
        )

    @app.post("/api/v1/projects/{project_id}/advance", response_model=ResearchProject)
    async def advance_project(
        project_id: str,
        workflow: WorkflowDependency,
    ) -> ResearchProject:
        return await workflow.advance(project_id)

    @app.post(
        "/api/v1/projects/{project_id}/research-cycles/next",
        response_model=ResearchProject,
    )
    async def start_next_research_cycle(
        project_id: str,
        body: StartNextResearchCycleRequest,
        workflow: WorkflowDependency,
    ) -> ResearchProject:
        return await workflow.start_next_research_cycle(
            project_id,
            user_guidance=body.user_guidance,
        )

    @app.post("/api/v1/projects/{project_id}/approve", response_model=ResearchProject)
    async def approve_plan(
        project_id: str,
        body: ApprovalRequest,
        workflow: WorkflowDependency,
    ) -> ResearchProject:
        return workflow.approve_experiment_plan(project_id, approved_by=body.approved_by)

    @app.post(
        "/api/v1/projects/{project_id}/experiment-methods/generate",
        response_model=ResearchProject,
    )
    async def generate_experiment_method(
        project_id: str,
        body: GenerateMethodRequest,
        workflow: WorkflowDependency,
    ) -> ResearchProject:
        return await workflow.implement_experiment_method(
            project_id,
            hypothesis_id=body.hypothesis_id,
        )

    @app.post(
        "/api/v1/projects/{project_id}/experiment-methods/generate-detector",
        response_model=ResearchProject,
    )
    async def generate_experiment_detector(
        project_id: str,
        body: GenerateDetectorRequest,
        workflow: WorkflowDependency,
    ) -> ResearchProject:
        return await workflow.implement_experiment_detector(
            project_id,
            name_stem=body.name_stem,
            hypothesis_id=body.hypothesis_id,
            reference_description=body.reference_description,
        )

    @app.post(
        "/api/v1/projects/{project_id}/experiment-campaign/initialize",
        response_model=ResearchProject,
    )
    async def initialize_experiment_campaign(
        project_id: str,
        body: InitializeExperimentCampaignRequest,
        workflow: WorkflowDependency,
        request: Request,
    ) -> ResearchProject:
        settings = cast(Settings, request.app.state.settings)
        dataset = _load_artifact_model(
            body.dataset_manifest_path,
            settings.artifact_path,
            DatasetManifest,
        )
        return workflow.initialize_experiment_campaign(
            project_id,
            dataset=dataset,
            hypothesis_id=body.hypothesis_id,
            device=body.device,
            detector=body.detector,
            max_rounds=body.max_rounds,
            max_runs=body.max_runs,
        )

    @app.post(
        "/api/v1/projects/{project_id}/experiment-campaign/execute-next",
        response_model=ExecuteNextExperimentResponse,
    )
    async def execute_next_campaign_experiment(
        project_id: str,
        body: ExecuteNextExperimentRequest,
        workflow: WorkflowDependency,
        request: Request,
    ) -> ExecuteNextExperimentResponse:
        locks = cast(dict[str, asyncio.Lock], request.app.state.experiment_locks)
        lock = locks.setdefault(project_id, asyncio.Lock())
        async with lock:
            settings = cast(Settings, request.app.state.settings)
            project = workflow.repository.get(project_id)
            campaign = project.experiment_campaign
            if campaign is None:
                raise HTTPException(409, "The project has no experiment campaign")
            if body.candidate_pool_size != campaign.candidate_pool_size:
                raise HTTPException(
                    409,
                    "candidate_pool_size is preregistered and cannot change during execution",
                )
            run, guidance_decision = await workflow.select_next_experiment(
                project_id,
                user_guidance=body.user_guidance,
            )
            project = workflow.repository.get(project_id)
            campaign = project.experiment_campaign
            if campaign is None:
                raise HTTPException(409, "The project has no experiment campaign")
            dataset_path = _resolve_artifact_path(
                campaign.dataset_manifest_path,
                settings.artifact_path,
            )
            dataset = _load_artifact_model(
                str(dataset_path),
                settings.artifact_path,
                DatasetManifest,
            )
            if dataset.digest != campaign.dataset_digest:
                raise HTTPException(409, "Dataset changed after campaign preregistration")

            output_dir = settings.artifact_path / "runs" / project_id / run.id
            output_dir.mkdir(parents=True, exist_ok=True)

            if settings.runtime == "mock":
                record = _create_mock_execution_record(run, output_dir)
                updated = workflow.record_run_result(
                    project_id,
                    run_id=run.id,
                    metrics=record.normalized_result.metrics if record.normalized_result else {},
                    artifact_paths=[],
                    code_revision="mock",
                    environment_digest="mock",
                    success=True,
                    verified=True,  # Mock results are valid for development
                    result_source="synthetic_test",
                    preparation_path=str((output_dir / "preparation.json").resolve()),
                    execution_record_path=str((output_dir / "execution.json").resolve()),
                    duration_seconds=record.duration_seconds,
                    error=None,
                )
                return ExecuteNextExperimentResponse(
                    run_id=run.id,
                    guidance_decision=guidance_decision,
                    prepared=None,
                    execution=record,
                    project=updated,
                )

            embeddings = await asyncio.to_thread(
                DinoV2Embedder(
                    settings.artifact_path,
                    model_id=settings.dinov2_profile_model,
                    device=campaign.device,
                    batch_size=8,
                ).extract,
                dataset,
                category=run.category,
                force=body.force_embeddings,
            )
            custom_strategies, strategy_runner = _custom_strategy_context(project, settings)
            prepared = await asyncio.to_thread(
                ExperimentPreparationService(settings.artifact_path).prepare,
                project_id=project_id,
                run=run,
                dataset=dataset,
                dataset_manifest_path=dataset_path,
                embeddings=embeddings,
                candidate_pool_size=campaign.candidate_pool_size,
                custom_strategies=custom_strategies,
                strategy_runner=strategy_runner,
            )
            view = _load_artifact_model(
                prepared.dataset_view_manifest_path,
                settings.artifact_path,
                DatasetViewManifest,
            )
            support = _load_artifact_model(
                prepared.support_manifest_path,
                settings.artifact_path,
                SupportSetManifest,
            )
            output_dir = settings.artifact_path / "runs" / project_id / run.id
            try:
                command = resolve_detector_command(
                    project,
                    run,
                    MethodRegistry(settings.artifact_path.parents[0]),
                    dataset_view=Path(view.view_root),
                    output_dir=output_dir,
                    device=campaign.device,
                )
            except ValueError as exc:
                raise HTTPException(409, str(exc)) from exc
            workflow.mark_run_running(project_id, run_id=run.id)
            record = await ExperimentRunner(
                project_root=settings.artifact_path.parents[0],
                artifact_root=settings.artifact_path,
            ).execute(
                run,
                command,
                output_dir=output_dir,
                dataset_view=view,
                support_manifest=support,
                timeout_seconds=body.timeout_seconds,
            )
            normalized = record.normalized_result
            if normalized is not None:
                _attach_support_geometry(normalized.metrics, support)
            updated = workflow.record_run_result(
                project_id,
                run_id=run.id,
                metrics=normalized.metrics if normalized else {},
                artifact_paths=[
                    record.stdout_path,
                    record.stderr_path,
                    *record.discovered_artifacts,
                ],
                code_revision=record.code_revision,
                environment_digest=record.environment_digest,
                success=record.status == "succeeded" and normalized is not None,
                verified=record.status == "succeeded" and normalized is not None,
                result_source="real_executor",
                preparation_path=str(
                    (
                        settings.artifact_path
                        / "prepared_runs"
                        / project_id
                        / f"{run.id}.json"
                    ).resolve()
                ),
                execution_record_path=str((output_dir / "execution.json").resolve()),
                duration_seconds=record.duration_seconds,
                error=record.error,
            )
            return ExecuteNextExperimentResponse(
                run_id=run.id,
                guidance_decision=guidance_decision,
                prepared=prepared,
                execution=record,
                project=updated,
            )

    @app.post(
        "/api/v1/projects/{project_id}/experiment-campaign/review",
        response_model=ResearchProject,
    )
    async def review_experiment_campaign_round(
        project_id: str,
        workflow: WorkflowDependency,
    ) -> ResearchProject:
        return await workflow.review_experiment_round(project_id)

    @app.post(
        "/api/v1/projects/{project_id}/runs/{run_id}/result",
        response_model=ResearchProject,
    )
    async def record_result(
        project_id: str,
        run_id: str,
        body: RunResultRequest,
        workflow: WorkflowDependency,
    ) -> ResearchProject:
        try:
            return workflow.record_run_result(
                project_id,
                run_id=run_id,
                metrics=body.metrics,
                artifact_paths=body.artifact_paths,
                code_revision=body.code_revision,
                environment_digest=body.environment_digest,
                success=body.success,
                verified=body.verified,
                result_source=body.result_source,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post(
        "/api/v1/projects/{project_id}/runs/{run_id}/execute",
        response_model=ExecutionRecord,
    )
    async def execute_run(
        project_id: str,
        run_id: str,
        body: ExecuteRunRequest,
        workflow: WorkflowDependency,
        request: Request,
    ) -> ExecutionRecord:
        settings = cast(Settings, request.app.state.settings)
        project = workflow.repository.get(project_id)
        run = next((item for item in project.runs if item.id == run_id), None)
        if run is None:
            raise HTTPException(404, f"Unknown run id: {run_id}")
        dataset = _load_artifact_model(
            body.dataset_manifest_path,
            settings.artifact_path,
            DatasetManifest,
        )
        support = _load_artifact_model(
            body.support_manifest_path,
            settings.artifact_path,
            SupportSetManifest,
        )
        _validate_run_support(run, dataset, support)
        view = await asyncio.to_thread(
            DatasetViewBuilder(settings.artifact_path).build,
            dataset,
            support,
        )
        output_dir = settings.artifact_path / "runs" / project_id / run_id
        try:
            command = resolve_detector_command(
                project,
                run,
                MethodRegistry(settings.artifact_path.parents[0]),
                dataset_view=Path(view.view_root),
                output_dir=output_dir,
                device=body.device,
            )
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        workflow.mark_run_running(project_id, run_id=run_id)
        record = await ExperimentRunner(
            project_root=settings.artifact_path.parents[0],
            artifact_root=settings.artifact_path,
        ).execute(
            run,
            command,
            output_dir=output_dir,
            dataset_view=view,
            support_manifest=support,
            timeout_seconds=body.timeout_seconds,
        )
        normalized = record.normalized_result
        if normalized is not None:
            _attach_support_geometry(normalized.metrics, support)
        workflow.record_run_result(
            project_id,
            run_id=run_id,
            metrics=normalized.metrics if normalized else {},
            artifact_paths=[record.stdout_path, record.stderr_path, *record.discovered_artifacts],
            code_revision=record.code_revision,
            environment_digest=record.environment_digest,
            success=record.status == "succeeded" and normalized is not None,
            verified=record.status == "succeeded" and normalized is not None,
            result_source="real_executor",
            execution_record_path=str((output_dir / "execution.json").resolve()),
            duration_seconds=record.duration_seconds,
            error=record.error,
        )
        return record

    @app.post(
        "/api/v1/projects/{project_id}/runs/{run_id}/prepare",
        response_model=PreparedRunArtifacts,
    )
    async def prepare_run(
        project_id: str,
        run_id: str,
        body: PrepareRunRequest,
        workflow: WorkflowDependency,
        request: Request,
    ) -> PreparedRunArtifacts:
        settings = cast(Settings, request.app.state.settings)
        project = workflow.repository.get(project_id)
        run = next((item for item in project.runs if item.id == run_id), None)
        if run is None:
            raise HTTPException(404, f"Unknown run id: {run_id}")
        dataset_path = _resolve_artifact_path(
            body.dataset_manifest_path,
            settings.artifact_path,
        )
        dataset = _load_artifact_model(
            str(dataset_path),
            settings.artifact_path,
            DatasetManifest,
        )
        embeddings = None
        if body.embedding_manifest_path:
            embeddings = _load_artifact_model(
                body.embedding_manifest_path,
                settings.artifact_path,
                DinoEmbeddingManifest,
            )
        custom_strategies, strategy_runner = _custom_strategy_context(project, settings)
        return await asyncio.to_thread(
            ExperimentPreparationService(settings.artifact_path).prepare,
            project_id=project_id,
            run=run,
            dataset=dataset,
            dataset_manifest_path=dataset_path,
            embeddings=embeddings,
            candidate_pool_size=body.candidate_pool_size,
            custom_strategies=custom_strategies,
            strategy_runner=strategy_runner,
        )

    @app.post("/api/v1/projects/{project_id}/results/finalize", response_model=ResearchProject)
    async def finalize_results(
        project_id: str,
        workflow: WorkflowDependency,
    ) -> ResearchProject:
        return workflow.finalize_results(project_id)

    return app


def _build_runtime(
    settings: Settings,
    *,
    evidence_service: LiteratureSearchService,
) -> ScientistRuntime:
    if settings.runtime == "mock":
        runtime: ScientistRuntime = MockScientistRuntime()
    elif settings.runtime == "agentscope":
        runtime = QwenScientistRuntime(
            model=settings.reasoning_model,
            api_key=settings.dashscope_api_key_value,
        )
    else:
        raise ValueError(f"Unsupported AISCIENTIST_RUNTIME: {settings.runtime}")
    if settings.live_evidence:
        return EvidenceEnabledRuntime(runtime, evidence_service)
    return runtime


def _error_response(status_code: int, detail: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"detail": detail})


ModelType = TypeVar("ModelType")


def _load_artifact_model(
    value: str,
    artifact_root: Path,
    model_type: type[ModelType],
) -> ModelType:
    path = _resolve_artifact_path(value, artifact_root)
    if not path.is_file():
        raise HTTPException(404, f"manifest not found: {path}")
    try:
        return model_type.model_validate_json(path.read_text(encoding="utf-8"))  # type: ignore[attr-defined,no-any-return]
    except (ValueError, AttributeError) as exc:
        raise HTTPException(422, f"invalid manifest: {exc}") from exc


def _resolve_artifact_path(value: str, artifact_root: Path) -> Path:
    path = Path(value)
    path = (artifact_root / path).resolve() if not path.is_absolute() else path.resolve()
    try:
        path.relative_to(artifact_root.resolve())
    except ValueError as exc:
        raise HTTPException(400, "manifest path must be inside the artifact root") from exc
    return path


def _custom_strategy_context(
    project: ResearchProject,
    settings: Settings,
) -> tuple[dict[str, MethodImplementation], GeneratedStrategyRunner]:
    approved = {
        item.name: item
        for item in project.method_implementations
        if item.kind == "selection_strategy" and item.status == "approved"
    }
    return approved, GeneratedStrategyRunner(settings.artifact_path)


def _write_model(path: Path, model: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(model.model_dump_json(indent=2), encoding="utf-8")  # type: ignore[attr-defined]
    temporary.replace(path)


def _create_mock_manifest(dataset_name: str, root: str) -> DatasetManifest:
    """Create a mock dataset manifest for development without real data."""
    import hashlib
    import json

    from fsad_scientist.datasets.models import DatasetAuditIssue, DatasetFileRecord

    categories = ["bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut", "pill", "screw", "tile", "transistor", "wood", "zipper"]
    files: list[DatasetFileRecord] = []
    for category in categories:
        for i in range(5):
            files.append(
                DatasetFileRecord(
                    relative_path=f"{category}/train/good/image_{i:04d}.png",
                    category=category,
                    split="train",
                    anomaly_type="good",
                    kind="image",
                    byte_size=1024 * 50,
                    sha256=hashlib.sha256(f"{category}_{i}".encode()).hexdigest(),
                )
            )
        for anomaly_type in ["good", "broken_large", "broken_small", "contamination"]:
            for i in range(3):
                files.append(
                    DatasetFileRecord(
                        relative_path=f"{category}/test/{anomaly_type}/image_{i:04d}.png",
                        category=category,
                        split="test",
                        anomaly_type=anomaly_type,
                        kind="image",
                        byte_size=1024 * 50,
                        sha256=hashlib.sha256(f"{category}_{anomaly_type}_{i}".encode()).hexdigest(),
                    )
                )
                files.append(
                    DatasetFileRecord(
                        relative_path=f"{category}/ground_truth/{anomaly_type}/image_{i:04d}_mask.png",
                        category=category,
                        split="ground_truth",
                        anomaly_type=anomaly_type,
                        kind="mask",
                        byte_size=1024 * 5,
                        sha256=hashlib.sha256(f"{category}_{anomaly_type}_{i}_mask".encode()).hexdigest(),
                    )
                )

    counts = {
        "files": len(files),
        **{f"category:{cat}": sum(f.category == cat for f in files) for cat in categories},
    }

    digest_payload = {
        "dataset": dataset_name,
        "format": "mvtec_ad",
        "files": [item.model_dump(mode="json") for item in files],
    }
    digest = hashlib.sha256(
        json.dumps(digest_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()

    return DatasetManifest(
        dataset=dataset_name,
        root=root,
        categories=sorted(categories),
        files=files,
        counts=counts,
        issues=[
            DatasetAuditIssue(
                severity="warning",
                code="MOCK_DATA",
                message="Mock data - replace with real MVTec AD dataset for actual experiments",
            )
        ],
        digest=digest,
    )


def _create_mock_execution_record(
    run: ExperimentRun,
    output_dir: Path,
) -> ExecutionRecord:
    """Create a mock execution record for development without real data."""
    import random
    from fsad_scientist.experiments.models import (
        ExecutionRecord,
        NormalizedExperimentResult,
    )
    from fsad_scientist.domain.models import utc_now

    strategy_bias = 0.015 if run.selection_strategy == "k_center" else 0.0
    seed_variance = (run.seed % 3) * 0.003
    category_variance = hash(run.category) % 10 * 0.001

    image_auroc = 0.85 + strategy_bias + seed_variance + category_variance + random.uniform(-0.01, 0.01)
    image_auroc = max(0.5, min(0.99, image_auroc))

    pixel_auroc = 0.78 + strategy_bias * 0.5 + random.uniform(-0.02, 0.02)
    pixel_auroc = max(0.4, min(0.98, pixel_auroc))

    aupro = 0.82 + strategy_bias * 0.8 + random.uniform(-0.015, 0.015)
    aupro = max(0.45, min(0.97, aupro))

    stdout_path = output_dir / "stdout.log"
    stderr_path = output_dir / "stderr.log"
    record_path = output_dir / "execution.json"

    stdout_path.write_text(f"[MOCK] Run {run.id} - {run.detector} on {run.category}\n[MOCK] Strategy: {run.selection_strategy}, K={run.shots}, seed={run.seed}\n[MOCK] image_auroc={image_auroc:.4f}\n", encoding="utf-8")
    stderr_path.write_text("", encoding="utf-8")

    record = ExecutionRecord(
        run_id=run.id,
        method=run.detector,
        status="succeeded",
        command=["python", "-m", "mock_detector"],
        cwd=str(output_dir),
        output_dir=str(output_dir),
        environment_overrides={},
        dataset_view_digest="mock_digest",
        support_manifest_digest="mock_support_digest",
        code_revision="mock",
        environment_digest="mock",
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
        normalized_result=NormalizedExperimentResult(
            parser="mock",
            metrics={
                "image_auroc": round(image_auroc, 4),
                "image_ap": round(image_auroc * 0.95, 4),
                "pixel_auroc": round(pixel_auroc, 4),
                "aupro": round(aupro, 4),
                "image_auroc_treated": round(image_auroc + strategy_bias, 4),
            },
            source_files=[],
        ),
    )

    record_path.write_text(record.model_dump_json(indent=2), encoding="utf-8")
    return record


def _validate_run_support(run, dataset: DatasetManifest, support: SupportSetManifest) -> None:
    mismatches = []
    expected = {
        "dataset": run.dataset,
        "category": run.category,
        "protocol": run.protocol,
        "strategy": run.selection_strategy,
        "shots": run.shots,
        "seed": run.seed,
    }
    actual = {
        "dataset": support.dataset,
        "category": support.category,
        "protocol": support.protocol,
        "strategy": support.strategy,
        "shots": support.shots,
        "seed": support.seed,
    }
    for name, value in expected.items():
        actual_value = actual[name]
        if name == "dataset":
            matches = str(value).casefold() == str(actual_value).casefold()
        else:
            matches = value == actual_value
        if not matches:
            mismatches.append(f"{name}: run={value!r}, support={actual_value!r}")
    source_digest = support.selection_metadata.get("dataset_manifest_digest")
    if source_digest and source_digest != dataset.digest:
        mismatches.append("dataset manifest digest")
    if mismatches:
        raise HTTPException(409, "run/support mismatch: " + "; ".join(mismatches))


def _attach_support_geometry(
    metrics: dict[str, float], support: SupportSetManifest
) -> None:
    for metric_name in (
        "coverage_radius",
        "mean_coverage_distance",
        "selected_pairwise_distance",
        "effective_rank",
    ):
        value = support.selection_metadata.get(metric_name)
        if isinstance(value, (int, float)):
            metrics[metric_name] = float(value)
