from typing import Literal

from pydantic import BaseModel, Field

from fsad_scientist.domain.models import (
    EvidenceRecord,
    ExperimentGuidanceDecision,
    HypothesisRanking,
    ProjectSpec,
    ResearchProject,
)
from fsad_scientist.experiments.models import ExecutionRecord, PreparedRunArtifacts


class CreateProjectRequest(BaseModel):
    spec: ProjectSpec = Field(default_factory=ProjectSpec)


class ApprovalRequest(BaseModel):
    approved_by: str = Field(min_length=1, max_length=120)


class GenerateMethodRequest(BaseModel):
    hypothesis_id: str = Field(min_length=1)


class GenerateDetectorRequest(BaseModel):
    hypothesis_id: str = Field(min_length=1)
    name_stem: str = Field(min_length=1, max_length=120)
    reference_description: str | None = None


class StartNextResearchCycleRequest(BaseModel):
    user_guidance: str = Field(min_length=2, max_length=3000)


class RankHypothesesRequest(BaseModel):
    """Human ranking gate after all candidate generation is automatic."""

    rankings: list[HypothesisRanking] = Field(min_length=1, max_length=1000)
    auto_preregister: bool = True


class AutoStartExperimentRequest(BaseModel):
    dataset_manifest_path: str = Field(min_length=1)
    hypothesis_id: str = Field(min_length=1)
    selected_hypothesis_ids: list[str] | None = Field(default=None, max_length=100)
    detector: str = Field(default="anomalydino", min_length=1, max_length=120)
    device: str = Field(default="cuda:0", pattern=r"^(cpu|cuda(?::\d+)?)$")
    max_rounds: int = Field(default=20, ge=1, le=100)
    max_runs: int = Field(default=240, ge=6, le=1000)
    max_parallel_runs: int | None = Field(default=None, ge=1, le=32)


class ExecuteParallelExperimentRequest(BaseModel):
    """Batch execution controls; run_ids enables selective replay/resume."""

    run_ids: list[str] | None = Field(default=None, max_length=1000)
    max_parallel_runs: int | None = Field(default=None, ge=1, le=32)
    timeout_seconds: float = Field(default=3600.0, gt=0, le=86400)
    force_embeddings: bool = False
    auto_review: bool = True


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
    hypothesis_id: str = Field(min_length=1)
    selected_hypothesis_ids: list[str] | None = Field(default=None, max_length=100)
    detector: str = Field(default="anomalydino", min_length=1, max_length=120)
    device: str = Field(default="cuda:0", pattern=r"^(cpu|cuda(?::\d+)?)$")
    max_rounds: int = Field(default=20, ge=1, le=100)
    max_runs: int = Field(default=24, ge=2, le=240)
    execution_mode: Literal["sequential", "parallel"] = "sequential"
    max_parallel_runs: int | None = Field(default=None, ge=1, le=32)


class ExecuteNextExperimentRequest(BaseModel):
    candidate_pool_size: int = Field(default=30, ge=2, le=1000)
    timeout_seconds: float = Field(default=3600.0, gt=0, le=86400)
    force_embeddings: bool = False
    user_guidance: str | None = Field(default=None, min_length=2, max_length=3000)


class ReviewExperimentRoundRequest(BaseModel):
    """One human decision gate between iteration 1 and iterations 2–3."""

    round_id: str | None = Field(default=None, min_length=1, max_length=120)
    user_guidance: str | None = Field(default=None, min_length=2, max_length=3000)


class ExecuteNextExperimentResponse(BaseModel):
    run_id: str
    guidance_decision: ExperimentGuidanceDecision
    prepared: PreparedRunArtifacts | None = None
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
