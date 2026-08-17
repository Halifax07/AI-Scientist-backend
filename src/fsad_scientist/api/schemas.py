from typing import Literal

from pydantic import BaseModel, Field

from fsad_scientist.domain.models import (
    EvidenceRecord,
    ExperimentGuidanceDecision,
    ProjectSpec,
    ResearchProject,
)
from fsad_scientist.experiments.models import ExecutionRecord, PreparedRunArtifacts


class CreateProjectRequest(BaseModel):
    spec: ProjectSpec = Field(default_factory=ProjectSpec)


class ApprovalRequest(BaseModel):
    approved_by: str = Field(min_length=1, max_length=120)


class StartNextResearchCycleRequest(BaseModel):
    user_guidance: str = Field(min_length=2, max_length=3000)


class RunResultRequest(BaseModel):
    metrics: dict[str, float] = Field(default_factory=dict)
    artifact_paths: list[str] = Field(default_factory=list)
    code_revision: str | None = None
    environment_digest: str | None = None
    success: bool = True
    verified: bool = True
    result_source: Literal["real_executor", "external_import", "synthetic_test"] = (
        "external_import"
    )


class HealthResponse(BaseModel):
    status: str
    runtime: str
    version: str


class EvidenceSearchRequest(BaseModel):
    query: str = Field(min_length=2, max_length=500)
    limit: int = Field(default=8, ge=1, le=50)
    providers: list[Literal["arxiv", "crossref"]] = Field(
        default_factory=lambda: ["arxiv", "crossref"]
    )


class EvidenceVerifyRequest(BaseModel):
    record: EvidenceRecord


class FullTextRequest(BaseModel):
    record: EvidenceRecord
    force: bool = False


class ClaimVerifyRequest(BaseModel):
    record: EvidenceRecord
    fulltext_manifest_path: str = Field(min_length=1)


class DatasetScanRequest(BaseModel):
    root: str = Field(min_length=1)
    dataset_name: str = Field(default="MVTec AD", min_length=1, max_length=120)


class ProjectDatasetAuditRequest(DatasetScanRequest):
    pass


class InitializeExperimentCampaignRequest(BaseModel):
    dataset_manifest_path: str = Field(min_length=1)
    detector: Literal["anomalydino", "patchcore", "subspacead"] = "anomalydino"
    device: str = Field(default="cuda:0", pattern=r"^(cpu|cuda(?::\d+)?)$")
    max_rounds: int = Field(default=3, ge=1, le=10)
    max_runs: int = Field(default=24, ge=2, le=240)


class ExecuteNextExperimentRequest(BaseModel):
    candidate_pool_size: int = Field(default=30, ge=2, le=1000)
    timeout_seconds: float = Field(default=3600.0, gt=0, le=86400)
    force_embeddings: bool = False
    user_guidance: str = Field(
        default="按预注册约束和系统优先级执行，不做额外调整。",
        min_length=1,
        max_length=3000,
    )


class ExecuteNextExperimentResponse(BaseModel):
    run_id: str
    guidance_decision: ExperimentGuidanceDecision
    prepared: PreparedRunArtifacts
    execution: ExecutionRecord
    project: ResearchProject


class DinoExtractRequest(BaseModel):
    dataset_manifest_path: str = Field(min_length=1)
    category: str = Field(min_length=1, max_length=120)
    image_files: list[str] | None = None
    model_id: str = Field(default="facebook/dinov2-small", min_length=1)
    revision: str | None = None
    device: str = "auto"
    batch_size: int = Field(default=8, ge=1, le=128)
    force: bool = False


class SupportPlanRequest(BaseModel):
    dataset_manifest_path: str = Field(min_length=1)
    category: str = Field(min_length=1, max_length=120)
    protocol: str
    strategy: str
    shots: int = Field(ge=1)
    seed: int = 0
    candidate_pool_size: int = Field(default=30, ge=1)
    embedding_manifest_path: str | None = None


class DatasetViewRequest(BaseModel):
    dataset_manifest_path: str = Field(min_length=1)
    support_manifest_path: str = Field(min_length=1)
    include_test: bool = True


class ExecuteRunRequest(BaseModel):
    dataset_manifest_path: str = Field(min_length=1)
    support_manifest_path: str = Field(min_length=1)
    device: str = "cuda:0"
    timeout_seconds: float = Field(default=3600.0, gt=0, le=86400)


class PrepareRunRequest(BaseModel):
    dataset_manifest_path: str = Field(min_length=1)
    embedding_manifest_path: str | None = None
    candidate_pool_size: int = Field(default=30, ge=1)
