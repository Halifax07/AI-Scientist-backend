from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import suppress
from pathlib import Path
from typing import Annotated, TypeVar, cast

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from fsad_scientist import __version__
from fsad_scientist.agents.agentscope_client import (
    AgentOutputValidationError,
    AgentScopeUnavailableError,
)
from fsad_scientist.agents.contracts import ScientistRuntime
from fsad_scientist.agents.evidence_runtime import EvidenceEnabledRuntime
from fsad_scientist.agents.mock_runtime import MockScientistRuntime
from fsad_scientist.agents.qwen_runtime import QwenScientistRuntime
from fsad_scientist.api.schemas import (
    ApprovalRequest,
    AutoStartExperimentRequest,
    ClaimVerifyRequest,
    CreateDemoRequest,
    CreateProjectRequest,
    DatasetScanRequest,
    DatasetViewRequest,
    DinoExtractRequest,
    EvidenceSearchRequest,
    EvidenceVerifyRequest,
    ExecuteNextExperimentRequest,
    ExecuteNextExperimentResponse,
    ExecuteParallelExperimentRequest,
    ExecuteRunRequest,
    FullTextRequest,
    GenerateDetectorRequest,
    GenerateMethodRequest,
    HealthResponse,
    InitializeExperimentCampaignRequest,
    PrepareRunRequest,
    ProjectDatasetAuditRequest,
    RankHypothesesRequest,
    ReviewExperimentRoundRequest,
    RunResultRequest,
    StartNextResearchCycleRequest,
    SupportPlanRequest,
)
from fsad_scientist.config import Settings, get_settings
from fsad_scientist.datasets.models import DatasetManifest, DatasetViewManifest
from fsad_scientist.datasets.scanner import MvtecDatasetScanner
from fsad_scientist.datasets.view import DatasetViewBuilder
from fsad_scientist.domain.enums import ResearchStage, RunStatus
from fsad_scientist.domain.models import (
    DatasetSpec,
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
from fsad_scientist.workflow import (
    InvalidTransitionError,
    ResearchWorkflow,
    ResultsRequiredError,
    WorkflowError,
)


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

    @app.exception_handler(AgentOutputValidationError)
    async def agent_output_invalid(_: Request, exc: AgentOutputValidationError):
        return _error_response(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc))

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
        summary="Create a pre-configured few-shot industrial anomaly detection demo project",
        description=(
            "Shorthand for creating a ProjectSpec with preset='fsad', which loads "
            "platform-embedded evidence, gaps and hypotheses for the MVTec AD demo. "
            "Use POST /api/v1/projects for custom research domains."
        ),
    )
    async def create_demo_project(
        workflow: WorkflowDependency,
    ) -> ResearchProject:
        demo_spec = ProjectSpec(
            preset="fsad",
            title="少样本工业视觉异常检测自主研究",
            domain="少样本工业视觉异常检测",
            application_context="新产品上线时仅能获取极少量正常样本",
            datasets=[
                DatasetSpec(name="MVTec AD", role="primary"),
                DatasetSpec(name="VisA", role="validation"),
            ],
        )
        return workflow.create_project(demo_spec)

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
        "/api/v1/projects/{project_id}/automation/ideation",
        response_model=ResearchProject,
    )
    async def automate_ideation_to_ranking(
        project_id: str,
        workflow: WorkflowDependency,
    ) -> ResearchProject:
        """Run formalisation, evidence, gaps and hypothesis generation automatically."""

        return await workflow.advance_to_hypothesis_ranking(project_id)

    @app.post(
        "/api/v1/projects/{project_id}/hypotheses/rank",
        response_model=ResearchProject,
    )
    async def rank_hypotheses(
        project_id: str,
        body: RankHypothesesRequest,
        workflow: WorkflowDependency,
    ) -> ResearchProject:
        return await workflow.rank_hypotheses(
            project_id,
            rankings=body.rankings,
            auto_preregister=body.auto_preregister,
        )

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
        "/api/v1/projects/{project_id}/experiment-plan/regenerate",
        response_model=ResearchProject,
    )
    async def regenerate_experiment_plan(
        project_id: str,
        workflow: WorkflowDependency,
    ) -> ResearchProject:
        return await workflow.regenerate_experiment_plan(project_id)

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
            execution_mode=body.execution_mode,
            parallelism=body.max_parallel_runs,
            selected_hypothesis_ids=body.selected_hypothesis_ids,
        )

    @app.post(
        "/api/v1/projects/{project_id}/experiment-campaign/auto-start",
        response_model=ResearchProject,
    )
    async def auto_start_experiment_campaign(
        project_id: str,
        body: AutoStartExperimentRequest,
        workflow: WorkflowDependency,
        request: Request,
    ) -> ResearchProject:
        settings = cast(Settings, request.app.state.settings)
        dataset_path = _resolve_artifact_path(
            body.dataset_manifest_path,
            settings.artifact_path,
        )
        dataset = _load_artifact_model(
            str(dataset_path),
            settings.artifact_path,
            DatasetManifest,
        )
        return await workflow.auto_start_parallel_campaign(
            project_id,
            dataset=dataset,
            hypothesis_id=body.hypothesis_id,
            selected_hypothesis_ids=body.selected_hypothesis_ids,
            device=body.device,
            detector=body.detector,
            max_rounds=body.max_rounds,
            max_runs=body.max_runs,
            parallelism=body.max_parallel_runs,
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
        "/api/v1/projects/{project_id}/experiment-campaign/execute-stream",
    )
    async def execute_parallel_campaign_stream(
        project_id: str,
        body: ExecuteParallelExperimentRequest,
        workflow: WorkflowDependency,
        request: Request,
    ) -> StreamingResponse:
        """Run queued innovation Rounds concurrently and stream structured events."""

        # Validate before returning StreamingResponse.  Exceptions raised after
        # headers are sent cannot be converted into the normal JSON error shape,
        # which used to make a missing/finished campaign look like a silent
        # ``Failed to fetch`` in the browser.
        project = workflow.repository.get(project_id)
        campaign = project.experiment_campaign
        if campaign is None or campaign.execution_mode != "parallel":
            raise HTTPException(409, "The project has no parallel experiment campaign")
        if campaign.status != "active":
            raise HTTPException(409, "The parallel campaign is not accepting runs")

        return StreamingResponse(
            _stream_parallel_execution(
                project_id,
                body=body,
                workflow=workflow,
                settings=cast(Settings, request.app.state.settings),
                locks=cast(dict[str, asyncio.Lock], request.app.state.experiment_locks),
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get(
        "/api/v1/projects/{project_id}/experiment-campaign/events",
    )
    async def experiment_campaign_events(
        project_id: str,
        workflow: WorkflowDependency,
        after: int = 0,
    ) -> StreamingResponse:
        """Replay persisted progress events for a disconnected UI client."""

        project = workflow.repository.get(project_id)
        events = [item for item in project.experiment_progress if item.sequence > after]

        async def replay() -> AsyncIterator[str]:
            for item in events:
                yield _sse_data(item.model_dump(mode="json"))
            yield _sse_data(
                {
                    "event_type": "stream_completed",
                    "message": "已 replay 当前已持久化的实验进度。",
                    "sequence": events[-1].sequence if events else after,
                    "payload": {"project_id": project_id},
                }
            )

        return StreamingResponse(
            replay(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )

    @app.post(
        "/api/v1/projects/{project_id}/experiment-campaign/review",
        response_model=ResearchProject,
    )
    async def review_experiment_campaign_round(
        project_id: str,
        workflow: WorkflowDependency,
        body: ReviewExperimentRoundRequest | None = None,
    ) -> ResearchProject:
        return await workflow.review_experiment_round(
            project_id,
            user_guidance=body.user_guidance if body else None,
            round_id=body.round_id if body else None,
        )

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


async def _stream_parallel_execution(
    project_id: str,
    *,
    body: ExecuteParallelExperimentRequest,
    workflow: ResearchWorkflow,
    settings: Settings,
    locks: dict[str, asyncio.Lock],
) -> AsyncIterator[str]:
    """Execute the selected campaign queue with bounded concurrency.

    A producer performs the real work and puts durable, structured events on an
    in-memory queue.  The consumer yields SSE frames immediately, so the UI can
    render every Run and Round as soon as the local executor reports it.
    """

    lock = locks.setdefault(project_id, asyncio.Lock())
    async with lock:
        project = workflow.repository.get(project_id)
        campaign = project.experiment_campaign
        # 这里不能像普通端点那样抛 HTTPException: StreamingResponse 在生成器
        # 真正产出帧之前就已发出 headers, 之后抛出的任何异常都无法再转成 JSON
        # 错误响应 —— Starlette 只会记录 "response already started" 并掐断连接,
        # 浏览器表现为 ERR_INCOMPLETE_CHUNKED_ENCODING, 用户完全看不到错误。
        # 因此生成器内的异常状态一律改为: 尽力恢复, 然后发完整事件帧干净收尾,
        # 让前端拿到最新 project 快照自行刷新 (端点层 722 行已做过常规 409 校验,
        # 这里只兜端点校验与生成器启动之间的竞态窗口)。
        if campaign is None or campaign.execution_mode != "parallel":
            yield _sse_data(
                {
                    "event_type": "stream_completed",
                    "message": "该项目当前没有可继续的并行实验队列，请刷新后查看最新状态。",
                    "progress": 1.0,
                    "payload": {"project": project.model_dump(mode="json")},
                }
            )
            return
        campaign_run_ids = {
            run_id
            for experiment_round in campaign.rounds
            for run_id in experiment_round.run_ids
        }
        parallelism = min(
            body.max_parallel_runs or campaign.parallelism,
            campaign.parallelism,
            settings_for_parallelism(project, settings),
        )
        parallelism = max(1, parallelism)
        event_queue: asyncio.Queue[dict[str, object]] = asyncio.Queue()
        state_lock = asyncio.Lock()
        round_ready_emitted: set[str] = set()
        round_guidance_emitted: set[str] = set()
        embedding_cache: dict[str, DinoEmbeddingManifest] = {}
        embedding_locks: dict[str, asyncio.Lock] = {}

        def progress_for(run_ids: set[str]) -> float:
            current = workflow.repository.get(project_id)
            terminal = sum(
                run.id in run_ids
                and run.status in {RunStatus.SUCCEEDED, RunStatus.FAILED}
                for run in current.runs
            )
            return min(terminal / max(len(run_ids), 1), 1.0)

        async def emit(
            event_type: str,
            message: str,
            *,
            run_id: str | None = None,
            round_id: str | None = None,
            hypothesis_id: str | None = None,
            status: str | None = None,
            progress: float | None = None,
            payload: dict[str, object] | None = None,
            snapshot: bool = True,
        ) -> None:
            async with state_lock:
                event = workflow.record_experiment_progress(
                    project_id,
                    event_type=event_type,
                    message=message,
                    campaign_id=campaign.id,
                    round_id=round_id,
                    hypothesis_id=hypothesis_id,
                    run_id=run_id,
                    status=status,
                    progress=progress,
                    payload=payload,
                )
                frame = event.model_dump(mode="json")
                if snapshot:
                    frame["project"] = workflow.repository.get(project_id).model_dump(mode="json")
                await event_queue.put(frame)

        async def producer() -> None:
            try:
                selected: list[ExperimentRun] = []
                try:
                    selected = workflow.select_parallel_runs(
                        project_id,
                        run_ids=body.run_ids,
                    )
                except (ResultsRequiredError, InvalidTransitionError):
                    # 队列为空 (例如上一次执行流被浏览器关闭/断网打断, 遗留的
                    # RUNNING 孤儿运行让 campaign 永远停在 active, 但没有任何
                    # 可排队运行; 每次重试都会在 headers 已发出的生成器里抛错,
                    # 表现为前端 ERR_INCOMPLETE_CHUNKED_ENCODING 的静默死锁)。
                    # 先清理孤儿运行, 让 refresh_after_run 把状态机推到真实位置,
                    # 再给 select 一次机会; 仍无运行则干净地结束本批。
                    workflow.recover_stale_parallel_runs(project_id)
                    current_campaign = workflow.repository.get(
                        project_id
                    ).experiment_campaign
                    if (
                        current_campaign is not None
                        and current_campaign.status == "active"
                    ):
                        try:
                            selected = workflow.select_parallel_runs(
                                project_id,
                                run_ids=body.run_ids,
                            )
                        except (ResultsRequiredError, InvalidTransitionError):
                            selected = []
                total = len(selected)
                selected_run_ids = {item.id for item in selected}
                if not selected:
                    final = workflow.repository.get(project_id)
                    final_campaign = final.experiment_campaign
                    await emit(
                        "batch_completed",
                        (
                            "当前没有排队等待的实验运行；若上一次执行曾中断，"
                            "遗留的运行状态已清理。请按界面提示继续"
                            "（提交指导、汇总结果或重新执行失败实验）。"
                        ),
                        status=final_campaign.status if final_campaign else None,
                        progress=progress_for(campaign_run_ids),
                        payload={
                            "campaign_status": (
                                final_campaign.status if final_campaign else None
                            ),
                            "next_action": (
                                final_campaign.next_action if final_campaign else None
                            ),
                            "batch_run_count": 0,
                            "campaign_run_count": len(campaign_run_ids),
                        },
                    )
                    await emit(
                        "stream_completed",
                        "实验流已结束；所有状态均已写入 Research Ledger。",
                        status=final.stage.value,
                        progress=progress_for(campaign_run_ids),
                        payload={"project_id": project_id, "batch_completed": True},
                    )
                    return
                await emit(
                    "campaign_started",
                    f"已启动 {total} 个实验运行，最多同时执行 {parallelism} 个。",
                    status=campaign.status,
                    progress=0.0,
                    payload={
                        "total_runs": total,
                        "parallelism": parallelism,
                        "run_ids": [item.id for item in selected],
                        "hypothesis_ids": sorted({item.hypothesis_id for item in selected}),
                    },
                )
                semaphore = asyncio.Semaphore(parallelism)

                async def run_one(run: ExperimentRun) -> None:
                    try:
                        await emit(
                            "run_queued",
                            f"{run.id} 已进入并行执行队列。",
                            run_id=run.id,
                            round_id=run.round_id,
                            hypothesis_id=run.hypothesis_id,
                            status=run.status,
                        )
                        async with semaphore:
                            async with state_lock:
                                workflow.mark_run_running(project_id, run_id=run.id)
                            await emit(
                                "run_started",
                                f"{run.id} 已开始执行。",
                                run_id=run.id,
                                round_id=run.round_id,
                                hypothesis_id=run.hypothesis_id,
                                status="running",
                            )
                            record = await _execute_campaign_run(
                                project_id,
                                run=run,
                                body=body,
                                workflow=workflow,
                                settings=settings,
                                state_lock=state_lock,
                                embedding_cache=embedding_cache,
                                embedding_locks=embedding_locks,
                            )
                            async with state_lock:
                                current = workflow.repository.get(project_id)
                                current_run = next(
                                    item for item in current.runs if item.id == run.id
                                )
                                terminal = sum(
                                    item.id in selected_run_ids
                                    and item.status
                                    in {RunStatus.SUCCEEDED, RunStatus.FAILED}
                                    for item in current.runs
                                )
                            await emit(
                                "run_finished",
                                (
                                    f"{run.id} 已完成。"
                                    if current_run.status == RunStatus.SUCCEEDED
                                    else f"{run.id} 执行失败：{current_run.error or '未知错误'}"
                                ),
                                run_id=run.id,
                                round_id=run.round_id,
                                hypothesis_id=run.hypothesis_id,
                                status=current_run.status,
                                progress=min(terminal / max(total, 1), 1.0),
                                payload={
                                    "run": current_run.model_dump(mode="json"),
                                    "execution": record.model_dump(mode="json"),
                                },
                            )
                            async with state_lock:
                                refreshed = workflow.repository.get(project_id)
                                refreshed_campaign = refreshed.experiment_campaign
                                ready_rounds = []
                                guidance_rounds = []
                                if refreshed_campaign is not None:
                                    for experiment_round in refreshed_campaign.rounds:
                                        if (
                                            experiment_round.status == "awaiting_guidance"
                                            and experiment_round.id not in round_guidance_emitted
                                        ):
                                            round_guidance_emitted.add(experiment_round.id)
                                            guidance_rounds.append(
                                                (
                                                    experiment_round.id,
                                                    experiment_round.index,
                                                    experiment_round.hypothesis_id,
                                                    experiment_round.status,
                                                    workflow.experiment_planner.summarize_round(
                                                        refreshed,
                                                        round_id=experiment_round.id,
                                                    ),
                                                )
                                            )
                                        if (
                                            experiment_round.status == "ready_for_feedback"
                                            and experiment_round.id not in round_ready_emitted
                                        ):
                                            round_ready_emitted.add(experiment_round.id)
                                            ready_rounds.append(
                                                (
                                                    experiment_round.id,
                                                    experiment_round.index,
                                                    experiment_round.hypothesis_id,
                                                    experiment_round.status,
                                                    workflow.experiment_planner.summarize_round(
                                                        refreshed,
                                                        round_id=experiment_round.id,
                                                    ),
                                                )
                                            )
                            for (
                                guidance_round_id,
                                guidance_round_index,
                                guidance_hypothesis_id,
                                guidance_status,
                                guidance_summary,
                            ) in guidance_rounds:
                                await emit(
                                    "round_guidance_required",
                                    (
                                        f"Round {guidance_round_index} 的第 1 次迭代已完成，"
                                        "请提交一次指导后继续第 2、3 次迭代。"
                                    ),
                                    round_id=guidance_round_id,
                                    hypothesis_id=guidance_hypothesis_id,
                                    status=guidance_status,
                                    payload={"summary": guidance_summary},
                                )
                            for (
                                ready_round_id,
                                ready_round_index,
                                ready_hypothesis_id,
                                ready_status,
                                ready_summary,
                            ) in ready_rounds:
                                await emit(
                                    "round_ready",
                                    (
                                        f"Round {ready_round_index} 的三次迭代已完成，"
                                        "正在汇总结果。"
                                    ),
                                    round_id=ready_round_id,
                                    hypothesis_id=ready_hypothesis_id,
                                    status=ready_status,
                                    payload={
                                        "round_index": ready_round_index,
                                        "summary": ready_summary,
                                    },
                                )
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        # A worker-level failure should become a terminal Run and
                        # a normal progress event; one bad innovation must not
                        # abort all other selected innovations in the batch.
                        error = f"{type(exc).__name__}: {exc}"
                        async with state_lock:
                            current = workflow.repository.get(project_id)
                            current_run = next(
                                (item for item in current.runs if item.id == run.id),
                                None,
                            )
                            if current_run is not None and current_run.status in {
                                RunStatus.QUEUED,
                                RunStatus.RUNNING,
                            }:
                                workflow.record_run_result(
                                    project_id,
                                    run_id=run.id,
                                    metrics={},
                                    artifact_paths=[],
                                    code_revision=None,
                                    environment_digest=None,
                                    success=False,
                                    verified=False,
                                    result_source="real_executor",
                                    error=error,
                                )
                                current = workflow.repository.get(project_id)
                                current_run = next(
                                    item for item in current.runs if item.id == run.id
                                )
                            terminal = sum(
                                item.id in selected_run_ids
                                and item.status
                                in {RunStatus.SUCCEEDED, RunStatus.FAILED}
                                for item in current.runs
                            )
                        await emit(
                            "run_finished",
                            f"{run.id} 执行失败：{error}",
                            run_id=run.id,
                            round_id=run.round_id,
                            hypothesis_id=run.hypothesis_id,
                            status="failed",
                            progress=min(terminal / max(total, 1), 1.0),
                            payload={"error": error},
                        )

                await asyncio.gather(*(run_one(item) for item in selected))
                completed = workflow.repository.get(project_id)
                completed_campaign = completed.experiment_campaign
                if (
                    body.auto_review
                    and completed_campaign is not None
                    and completed_campaign.status == "awaiting_feedback"
                ):
                    ready_round_ids = {
                        item.id
                        for item in completed_campaign.rounds
                        if item.status == "ready_for_feedback"
                    }
                    completed = await workflow.complete_parallel_campaign(project_id)
                    completed_rounds = (
                        completed.experiment_campaign.rounds
                        if completed.experiment_campaign
                        else []
                    )
                    for experiment_round in completed_rounds:
                        if (
                            ready_round_ids
                            and experiment_round.id not in ready_round_ids
                        ):
                            continue
                        await emit(
                            "round_completed",
                            f"Round {experiment_round.index} 已完成创新点结果审查。",
                            round_id=experiment_round.id,
                            hypothesis_id=experiment_round.hypothesis_id,
                            status=experiment_round.status,
                            payload={
                                "summary": experiment_round.result_summary,
                                "feedback": experiment_round.feedback.model_dump(mode="json")
                                if experiment_round.feedback
                                else None,
                            },
                        )
                final_project = workflow.repository.get(project_id)
                final_campaign = final_project.experiment_campaign
                campaign_completed = bool(
                    final_campaign is not None and final_campaign.status == "completed"
                )
                final_progress = progress_for(campaign_run_ids)
                completion_event = "campaign_completed" if campaign_completed else "batch_completed"
                await emit(
                    completion_event,
                    (
                        "所选创新点已完成并行实验与 Round 汇总。"
                        if campaign_completed
                        else "本次并行批次已结束，仍有未执行队列可继续启动。"
                    ),
                    status=final_campaign.status if final_campaign else None,
                    progress=final_progress,
                    payload={
                        "campaign_status": final_campaign.status if final_campaign else None,
                        "next_action": final_campaign.next_action if final_campaign else None,
                        "batch_run_count": total,
                        "campaign_run_count": len(campaign_run_ids),
                    },
                )
                final_stage = final_project.stage
                if campaign_completed and body.auto_review:
                    try:
                        finalized = workflow.finalize_results(project_id)
                        await emit(
                            "results_locked",
                            "全部选中创新点的实验结果已锁定，开始统一统计分析。",
                            status=finalized.stage.value,
                            progress=1.0,
                            payload={
                                "verified_runs": sum(
                                    run.status == RunStatus.SUCCEEDED and run.verified
                                    for run in finalized.runs
                                ),
                                "failed_runs": sum(
                                    run.status == RunStatus.FAILED for run in finalized.runs
                                ),
                            },
                        )
                        analyzed = await workflow.advance(project_id)
                        final_stage = analyzed.stage
                        await emit(
                            "statistics_completed",
                            "成对统计与假设判定已完成，各创新点结果保持独立。",
                            status=analyzed.stage.value,
                            progress=1.0,
                            payload={
                                "finding_count": len(analyzed.findings),
                                "stage": analyzed.stage.value,
                            },
                        )
                        if analyzed.stage == ResearchStage.RESULTS_ANALYZED:
                            reviewed = await workflow.advance(project_id)
                            final_stage = reviewed.stage
                            if reviewed.stage == ResearchStage.HYPOTHESES_PROPOSED:
                                await emit(
                                    "hypothesis_revision_ready",
                                    (
                                        "当前证据不足以支持原主张，AI Scientist 已生成修订假设，"
                                        "等待下一次用户排名。"
                                    ),
                                    status=reviewed.stage.value,
                                    progress=1.0,
                                    payload={
                                        "hypothesis_count": len(reviewed.hypotheses),
                                        "research_cycle": reviewed.research_cycle,
                                    },
                                )
                            elif reviewed.stage == ResearchStage.INNOVATION_REVIEWED:
                                await emit(
                                    "innovation_review_completed",
                                    "创新审查已完成，结果包含新颖性、机制、边界和复现性依据。",
                                    status=reviewed.stage.value,
                                    progress=1.0,
                                    payload={
                                        "innovation_count": len(reviewed.innovations),
                                        "innovations": [
                                            {
                                                "id": item.id,
                                                "hypothesis_id": item.hypothesis_id,
                                                "title": item.title,
                                                "status": item.status,
                                                "confidence": item.confidence,
                                                "core_finding": item.core_finding,
                                                "difference_from_prior_work": (
                                                    item.difference_from_prior_work
                                                ),
                                                "boundary_conditions": item.boundary_conditions,
                                                "reproducibility_evidence": (
                                                    item.reproducibility_evidence
                                                ),
                                            }
                                            for item in reviewed.innovations
                                        ],
                                    },
                                )
                                report = await workflow.advance(project_id)
                                final_stage = report.stage
                                if report.stage == ResearchStage.REPORT_READY:
                                    await emit(
                                        "report_ready",
                                        "研究输出清单已生成，可导出报告和复现材料。",
                                        status=report.stage.value,
                                        progress=1.0,
                                        payload={
                                            "artifact_count": len(report.artifacts),
                                        },
                                    )
                    except Exception as exc:
                        final_stage = workflow.repository.get(project_id).stage
                        await emit(
                            "finalization_failed",
                            f"实验已完成，但结果收尾暂未完成：{type(exc).__name__}: {exc}",
                            status=final_stage.value,
                            progress=1.0,
                            payload={"error": f"{type(exc).__name__}: {exc}"},
                        )
                final_project = workflow.repository.get(project_id)
                await emit(
                    "stream_completed",
                    "实验流已结束；所有状态均已写入 Research Ledger。",
                    status=final_project.stage.value,
                    progress=final_progress,
                    payload={
                        "project_id": project_id,
                        "batch_completed": True,
                        "campaign_completed": campaign_completed,
                        "stage": final_stage.value,
                    },
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                with suppress(Exception):
                    async with state_lock:
                        workflow.fail_parallel_campaign(project_id, reason=error)
                await emit(
                    "campaign_failed",
                    f"并行实验流失败：{error}",
                    status="failed",
                    payload={"error": error},
                )
                await emit(
                    "stream_completed",
                    "实验流因错误结束；请查看失败 Run 和 Research Ledger。",
                    status="failed",
                    payload={"project_id": project_id},
                )

        producer_task = asyncio.create_task(producer())
        try:
            while True:
                try:
                    frame = await asyncio.wait_for(event_queue.get(), timeout=15.0)
                except TimeoutError:
                    yield _sse_data(
                        {
                            "event_type": "heartbeat",
                            "message": "实验仍在运行，等待本地执行器返回结构化状态。",
                        }
                    )
                    continue
                yield _sse_data(frame)
                if frame.get("event_type") == "stream_completed":
                    break
        finally:
            if not producer_task.done():
                producer_task.cancel()
            with suppress(asyncio.CancelledError):
                await producer_task


async def _execute_campaign_run(
    project_id: str,
    *,
    run: ExperimentRun,
    body: ExecuteParallelExperimentRequest,
    workflow: ResearchWorkflow,
    settings: Settings,
    state_lock: asyncio.Lock,
    embedding_cache: dict[str, DinoEmbeddingManifest],
    embedding_locks: dict[str, asyncio.Lock],
) -> ExecutionRecord:
    """Execute one already-reserved run and persist success or failure."""

    project = workflow.repository.get(project_id)
    campaign = project.experiment_campaign
    if campaign is None:
        raise HTTPException(409, "The project has no experiment campaign")
    output_dir = settings.artifact_path / "runs" / project_id / run.id
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
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

        if settings.runtime == "mock":
            record = _create_mock_execution_record(run, output_dir)
            updated_metrics = (
                record.normalized_result.metrics if record.normalized_result else {}
            )
            async with state_lock:
                workflow.record_run_result(
                    project_id,
                    run_id=run.id,
                    metrics=updated_metrics,
                    artifact_paths=[],
                    code_revision="mock",
                    environment_digest="mock",
                    success=True,
                    verified=True,
                    result_source="synthetic_test",
                    preparation_path=str((output_dir / "preparation.json").resolve()),
                    execution_record_path=str((output_dir / "execution.json").resolve()),
                    duration_seconds=record.duration_seconds,
                    error=None,
                )
            return record

        category_lock = embedding_locks.setdefault(run.category, asyncio.Lock())
        async with category_lock:
            embeddings = embedding_cache.get(run.category)
            if embeddings is None:
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
                embedding_cache[run.category] = embeddings
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
        command = resolve_detector_command(
            project,
            run,
            MethodRegistry(settings.artifact_path.parents[0]),
            dataset_view=Path(view.view_root),
            output_dir=output_dir,
            device=campaign.device,
        )
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
        async with state_lock:
            workflow.record_run_result(
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
        return record
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        current = workflow.repository.get(project_id)
        current_run = next((item for item in current.runs if item.id == run.id), None)
        if current_run is not None and current_run.status == RunStatus.RUNNING:
            async with state_lock:
                workflow.record_run_result(
                    project_id,
                    run_id=run.id,
                    metrics={},
                    artifact_paths=[],
                    code_revision=None,
                    environment_digest=None,
                    success=False,
                    verified=False,
                    result_source="real_executor",
                    execution_record_path=str((output_dir / "execution.json").resolve()),
                    error=error,
                )
        return ExecutionRecord(
            run_id=run.id,
            method=run.detector,
            status="failed",
            command=[],
            cwd=str(output_dir),
            output_dir=str(output_dir),
            environment_overrides={},
            dataset_view_digest="",
            support_manifest_digest="",
            environment_digest="",
            stdout_path=str(output_dir / "stdout.log"),
            stderr_path=str(output_dir / "stderr.log"),
            error=error,
        )


def settings_for_parallelism(project: ResearchProject, settings: Settings) -> int:
    """Resolve a conservative upper bound for local hardware execution."""

    del settings
    return max(1, min(project.spec.budget.max_parallel_runs, 32))


def _sse_data(payload: dict[str, object]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False, default=str)}\n\n"


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

    categories = [
        "bottle",
        "cable",
        "capsule",
        "carpet",
        "grid",
        "hazelnut",
        "leather",
        "metal_nut",
        "pill",
        "screw",
        "tile",
        "transistor",
        "wood",
        "zipper",
    ]
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

    strategy_bias = 0.015 if run.selection_strategy == "k_center" else 0.0
    seed_variance = (run.seed % 3) * 0.003
    category_variance = hash(run.category) % 10 * 0.001

    image_auroc = (
        0.85
        + strategy_bias
        + seed_variance
        + category_variance
        + random.uniform(-0.01, 0.01)
    )
    image_auroc = max(0.5, min(0.99, image_auroc))

    pixel_auroc = 0.78 + strategy_bias * 0.5 + random.uniform(-0.02, 0.02)
    pixel_auroc = max(0.4, min(0.98, pixel_auroc))

    aupro = 0.82 + strategy_bias * 0.8 + random.uniform(-0.015, 0.015)
    aupro = max(0.45, min(0.97, aupro))

    stdout_path = output_dir / "stdout.log"
    stderr_path = output_dir / "stderr.log"
    record_path = output_dir / "execution.json"

    stdout_path.write_text(
        (
            f"[MOCK] Run {run.id} - {run.detector} on {run.category}\n"
            f"[MOCK] Strategy: {run.selection_strategy}, K={run.shots}, "
            f"seed={run.seed}\n"
            f"[MOCK] image_auroc={image_auroc:.4f}\n"
        ),
        encoding="utf-8",
    )
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
