from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, computed_field, field_validator

from fsad_scientist.domain.enums import (
    EvidenceStatus,
    HypothesisStatus,
    ProjectStatus,
    ResearchStage,
    RunStatus,
)


def utc_now() -> datetime:
    return datetime.now(UTC)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


def _execution_strategy(value: str) -> str | None:
    normalized = " ".join(value.casefold().replace("_", " ").replace("-", " ").split())
    if normalized == "random" or "random" in normalized or "随机" in normalized:
        return "random"
    if any(
        marker in normalized
        for marker in (
            "k center",
            "diversity",
            "representative",
            "coverage",
            "多样性",
            "代表性",
            "覆盖",
        )
    ):
        return "k_center"
    return None


class DatasetSpec(BaseModel):
    name: str
    root: str | None = None
    role: Literal["primary", "validation", "private"] = "primary"
    normal_only_training: bool = True
    may_select_support_from_train: bool = True


class ComputeBudget(BaseModel):
    gpu_hours: float = Field(default=12.0, gt=0)
    max_parallel_runs: int = Field(default=2, ge=1, le=32)
    max_llm_calls: int = Field(default=300, ge=1)
    max_experiments: int = Field(default=240, ge=1)


class ResearchConstraints(BaseModel):
    shots: list[int] = Field(default_factory=lambda: [1, 2, 4, 8])
    require_image_level_score: bool = True
    require_pixel_localization: bool = True
    no_real_anomalies_for_adaptation: bool = True
    training_free_preferred: bool = True
    human_approval_before_execution: bool = True
    strict_k_shot_protocol: bool = True
    pool_compression_protocol: bool = True
    candidate_pool_size: int = Field(default=30, ge=2)
    max_research_cycles: int = Field(default=3, ge=1, le=10)

    @field_validator("shots")
    @classmethod
    def validate_shots(cls, values: list[int]) -> list[int]:
        cleaned = sorted(set(values))
        if not cleaned or any(value <= 0 for value in cleaned):
            raise ValueError("shots must contain positive integers")
        return cleaned


class ProjectSpec(BaseModel):
    title: str = "少样本工业视觉异常检测自主研究"
    domain: str = "少样本工业视觉异常检测"
    application_context: str = "新产品上线时仅能获取极少量正常样本"
    objective: str = "自主发现并验证具有科学价值的可证伪改进方向"
    datasets: list[DatasetSpec] = Field(
        default_factory=lambda: [
            DatasetSpec(name="MVTec AD", role="primary"),
            DatasetSpec(name="VisA", role="validation"),
        ]
    )
    constraints: ResearchConstraints = Field(default_factory=ResearchConstraints)
    budget: ComputeBudget = Field(default_factory=ComputeBudget)
    user_guidance: list[str] = Field(default_factory=list)


class ClaimVerification(BaseModel):
    claim: str
    verdict: Literal["supports", "contradicts", "not_found"]
    page_number: int | None = Field(default=None, ge=1)
    quote: str | None = Field(default=None, max_length=600)
    rationale: str
    anchored: bool = False
    verifier: str


class EvidenceRecord(BaseModel):
    id: str = Field(default_factory=lambda: new_id("evidence"))
    title: str
    source_type: Literal["paper", "dataset", "experiment", "user", "system"]
    url: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    authors: list[str] = Field(default_factory=list)
    published_year: int | None = None
    venue: str | None = None
    abstract: str | None = None
    source_provider: str | None = None
    claims: list[str] = Field(default_factory=list)
    claim_checks: list[ClaimVerification] = Field(default_factory=list)
    status: EvidenceStatus = EvidenceStatus.UNVERIFIED
    verification_scope: Literal["none", "bibliographic", "claim"] = "none"
    verification_notes: list[str] = Field(default_factory=list)
    retrieved_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ResearchGap(BaseModel):
    id: str = Field(default_factory=lambda: new_id("gap"))
    title: str
    description: str
    why_unresolved: str
    evidence_ids: list[str] = Field(default_factory=list)
    expected_scientific_value: float = Field(ge=0, le=1)
    estimated_cost: float = Field(ge=0, le=1)
    status: Literal["candidate", "selected", "rejected"] = "candidate"


class HypothesisScore(BaseModel):
    novelty: float = Field(ge=0, le=1)
    falsifiability: float = Field(ge=0, le=1)
    feasibility: float = Field(ge=0, le=1)
    scientific_value: float = Field(ge=0, le=1)
    evidence_strength: float = Field(ge=0, le=1)
    elo: float = 1000.0

    @property
    def weighted_total(self) -> float:
        return round(
            0.25 * self.novelty
            + 0.25 * self.falsifiability
            + 0.20 * self.feasibility
            + 0.20 * self.scientific_value
            + 0.10 * self.evidence_strength,
            4,
        )


class AnalysisContract(BaseModel):
    kind: Literal["selection_main_effect", "detector_interaction", "query_adaptation"]
    metric: str
    treatment: str
    control: str
    alpha: float = Field(default=0.05, gt=0, lt=1)
    minimum_pairs: int = Field(default=6, ge=2)


class Hypothesis(BaseModel):
    id: str = Field(default_factory=lambda: new_id("hypothesis"))
    gap_id: str
    title: str
    claim: str
    null_hypothesis: str
    rationale: str
    independent_variables: list[str]
    dependent_variables: list[str]
    predicted_direction: str
    falsification_conditions: list[str]
    evidence_ids: list[str] = Field(default_factory=list)
    closest_prior_work: list[str] = Field(default_factory=list)
    analysis_contract: AnalysisContract | None = None
    score: HypothesisScore | None = None
    status: HypothesisStatus = HypothesisStatus.CANDIDATE
    revision: int = 1
    parent_hypothesis_id: str | None = None
    # Human ranking is intentionally separate from the AI skeptic score.  The
    # user can select several innovations, assign a priority and leave a short
    # review without overwriting the machine-audited score.
    user_selected: bool | None = None
    user_priority: int | None = Field(default=None, ge=1, le=1000)
    user_score: float | None = Field(default=None, ge=0, le=100)
    user_review_note: str | None = Field(default=None, max_length=3000)

    @computed_field
    @property
    def execution_readiness(self) -> Literal["executable", "requires_implementation"]:
        contract = self.analysis_contract
        if (
            contract is not None
            and contract.kind == "selection_main_effect"
            and _execution_strategy(contract.treatment) in {"random", "k_center"}
            and _execution_strategy(contract.control) in {"random", "k_center"}
            and _execution_strategy(contract.treatment) != _execution_strategy(contract.control)
        ):
            return "executable"
        return "requires_implementation"

    @computed_field
    @property
    def experiment_guidance(self) -> list[str]:
        contract = self.analysis_contract
        if contract is None:
            return ["先补充自变量、对照组、主指标和最小样本量，再进入实验。"]
        if self.execution_readiness == "executable":
            return [
                f"围绕“{self.title}”独立建立实验活动，不与其他创新点混用结果。",
                (
                    f"实验组为 {contract.treatment}，对照组为 {contract.control}，"
                    f"主指标为 {contract.metric}。"
                ),
                f"至少形成 {contract.minimum_pairs} 组同类别、同 K、同 seed 的成对结果。",
                (
                    "并行模式下每个实验 Round 自动完成 3 次迭代；旧串行模式才在第 1 次迭代后"
                    "接受一次用户指导，且任何模式都不得改变已批准的对照边界。"
                ),
            ]
        return [
            f"该创新需要实现 {contract.treatment} 与 {contract.control} 的可调用算法适配器。",
            f"实现后以 {contract.metric} 为主指标，至少收集 {contract.minimum_pairs} 组成对结果。",
        "方法代码、参数和失败日志必须先登记到 Research Ledger，之后才能声明进入验证。",
    ]


class HypothesisRanking(BaseModel):
    """A user's auditable ranking decision for one generated hypothesis."""

    hypothesis_id: str = Field(min_length=1)
    selected: bool = True
    priority: int = Field(default=1, ge=1, le=1000)
    score: float = Field(default=50.0, ge=0, le=100)
    note: str | None = Field(default=None, max_length=3000)


class ExperimentPlan(BaseModel):
    id: str = Field(default_factory=lambda: new_id("plan"))
    hypothesis_ids: list[str]
    hypothesis_contracts: dict[str, AnalysisContract] = Field(default_factory=dict)
    protocols: list[str]
    detectors: list[str]
    selection_strategies: list[str]
    datasets: list[str]
    categories: list[str]
    shots: list[int]
    seeds: list[int]
    metrics: list[str]
    analysis_methods: list[str]
    stages: list[str]
    stopping_conditions: list[str]
    estimated_gpu_hours: float = Field(ge=0)
    preregistration_digest: str
    approved: bool = False
    approved_by: str | None = None
    approved_at: datetime | None = None
    method_implementation_digests: dict[str, str] = Field(default_factory=dict)


class ExperimentRun(BaseModel):
    id: str = Field(default_factory=lambda: new_id("run"))
    plan_id: str
    hypothesis_id: str
    protocol: str
    dataset: str
    category: str
    detector: str
    selection_strategy: str
    shots: int
    seed: int
    iteration: int = Field(default=1, ge=1, le=3)
    round_id: str | None = None
    node_id: str | None = None
    phase: Literal[
        "feasibility",
        "sensitivity",
        "main_study",
        "replication",
        "ablation",
        "cross_dataset",
    ] = "feasibility"
    status: RunStatus = RunStatus.PLANNED
    metrics: dict[str, float] = Field(default_factory=dict)
    artifact_paths: list[str] = Field(default_factory=list)
    code_revision: str | None = None
    environment_digest: str | None = None
    verified: bool = False
    result_source: Literal["real_executor", "external_import", "synthetic_test"] | None = None
    preparation_path: str | None = None
    execution_record_path: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    duration_seconds: float | None = Field(default=None, ge=0)
    error: str | None = None


class ExperimentGuidanceDecision(BaseModel):
    """Guarded interpretation of human advice or an automatic low-level run action."""

    advisor: str
    selected_run_id: str
    interpretation: str
    disposition: Literal["applied", "partially_applied", "not_applicable", "rejected"]
    rationale: str
    execution_notes: list[str] = Field(default_factory=list)
    protected_constraints: list[str] = Field(default_factory=list)


class UserGuidanceRecord(BaseModel):
    """Auditable human input and its effect on an autonomous research action."""

    id: str = Field(default_factory=lambda: new_id("guidance"))
    scope: Literal["experiment_execution", "round_iteration", "research_cycle"]
    target_action: Literal[
        "execute_next_experiment",
        "continue_round_iterations",
        "start_next_research_cycle",
    ]
    text: str = Field(min_length=1, max_length=3000)
    research_cycle: int = Field(ge=1)
    round_id: str | None = None
    advisor: str | None = None
    interpretation: str | None = None
    disposition: Literal[
        "received",
        "applied",
        "partially_applied",
        "not_applicable",
        "rejected",
    ] = "received"
    rationale: str | None = None
    selected_run_id: str | None = None
    affected_ids: list[str] = Field(default_factory=list)
    protected_constraints: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)


class DatasetAuditRecord(BaseModel):
    id: str = Field(default_factory=lambda: new_id("dataset_audit"))
    dataset: str
    root: str
    manifest_path: str
    digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    categories: list[str]
    counts: dict[str, int] = Field(default_factory=dict)
    issue_count: int = Field(default=0, ge=0)
    verified: bool = False
    audited_at: datetime = Field(default_factory=utc_now)


class ExperimentCell(BaseModel):
    category: str
    shots: int = Field(ge=1)
    seed: int = Field(ge=0)


class ReasoningStep(BaseModel):
    """AI 推理链中的单个步骤"""
    step: int = Field(ge=1, description="步骤编号")
    observation: str = Field(description="观察到的具体事实")
    conclusion: str = Field(description="从这个事实得出的结论")
    confidence: Literal["高", "中", "低"] = "中"


class AlternativeDecision(BaseModel):
    """考虑过但未选择的备选方案"""
    decision: str = Field(description="考虑过的方案名称")
    rejected_reason: str = Field(description="为什么没有选择该方案")


class ExpectedImprovement(BaseModel):
    """预期改进指标"""
    metric: str = Field(description="预期改进的指标名")
    direction: Literal["increase", "decrease"] = Field(description="改进方向")
    estimated_delta: float = Field(description="预估变化量")
    confidence: Literal["高", "中", "低"] = "中"


class ExperimentFeedbackProposal(BaseModel):
    advisor: str
    decision: Literal[
        "expand",
        "replicate",
        "diagnose",
        "stop",
        "adapt_k",
        "focus_category",
        "ablate",
        "early_stop",
    ] = "expand"
    rationale: str
    reasoning_chain: list[ReasoningStep] = Field(
        default_factory=list,
        description="AI 推理步骤列表，每个元素是一个推理步骤"
    )
    alternative_decisions: list[AlternativeDecision] = Field(
        default_factory=list,
        description="考虑过但未选择的方案及原因"
    )
    expected_improvement: ExpectedImprovement | None = Field(
        default=None,
        description="预期改进指标"
    )
    observed_patterns: list[str] = Field(default_factory=list)
    next_phase: Literal[
        "sensitivity",
        "main_study",
        "replication",
        "ablation",
        "cross_dataset",
        "complete",
    ] = "sensitivity"
    recommended_cells: list[ExperimentCell] = Field(default_factory=list)
    strategy_adjustment: dict[str, Any] = Field(
        default_factory=dict,
        description="针对当前策略的具体调整建议"
    )
    expected_information_gain: float = Field(default=0.0, ge=0, le=1)
    stop: bool = False


class ExperimentNodeRecord(BaseModel):
    id: str = Field(default_factory=lambda: new_id("experiment_node"))
    round_id: str
    iteration: int = Field(default=1, ge=1, le=3)
    parent_id: str | None = None
    phase: Literal[
        "feasibility",
        "sensitivity",
        "main_study",
        "replication",
        "ablation",
        "cross_dataset",
    ]
    objective: str
    information_gain: float = Field(ge=0, le=1)
    falsification_value: float = Field(ge=0, le=1)
    estimated_cost: float = Field(gt=0)
    novelty: float = Field(default=0.0, ge=0, le=1)
    priority: float = Field(ge=0)
    config: dict[str, Any]
    run_ids: list[str] = Field(default_factory=list)
    status: Literal["pending", "running", "succeeded", "failed", "pruned"] = "pending"
    result_summary: dict[str, Any] = Field(default_factory=dict)
    error_history: list[str] = Field(default_factory=list)


class ExperimentRound(BaseModel):
    id: str = Field(default_factory=lambda: new_id("round"))
    index: int = Field(ge=1)
    phase: Literal[
        "feasibility",
        "sensitivity",
        "main_study",
        "replication",
        "ablation",
        "cross_dataset",
    ]
    objective: str
    rationale: str
    hypothesis_id: str = ""
    treatment: str = "k_center"
    control: str = "random"
    metric: str = "image_auroc"
    iteration_target: int = Field(default=3, ge=3, le=3)
    completed_iterations: int = Field(default=0, ge=0, le=3)
    guidance_received: bool = False
    node_ids: list[str] = Field(default_factory=list)
    run_ids: list[str] = Field(default_factory=list)
    status: Literal[
        "planned",
        "running",
        "awaiting_guidance",
        "ready_for_feedback",
        "completed",
        "failed",
    ] = "planned"
    result_summary: dict[str, Any] = Field(default_factory=dict)
    feedback: ExperimentFeedbackProposal | None = None
    efficiency: dict[str, float | int] = Field(default_factory=dict)
    started_at: datetime | None = None
    completed_at: datetime | None = None


class ExperimentCampaign(BaseModel):
    id: str = Field(default_factory=lambda: new_id("campaign"))
    hypothesis_id: str
    hypothesis_ids: list[str] = Field(default_factory=list)
    dataset_audit_id: str
    dataset_manifest_path: str
    dataset_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    protocol: str = "pool_compression_m30"
    candidate_pool_size: int = Field(default=30, ge=2)
    detector: str = "anomalydino"
    treatment: str = "k_center"
    control: str = "random"
    metric: str = "image_auroc"
    device: str = "cuda:0"
    max_rounds: int = Field(default=3, ge=1, le=100)
    iterations_per_round: int = Field(default=3, ge=3, le=3)
    max_runs: int = Field(default=24, ge=2, le=1000)
    exhaustive_run_count: int = Field(default=0, ge=0)
    current_round: int = Field(default=1, ge=1)
    execution_mode: Literal["sequential", "parallel"] = "sequential"
    parallelism: int = Field(default=1, ge=1, le=32)
    selected_hypothesis_ids: list[str] = Field(default_factory=list)
    status: Literal[
        "active",
        "awaiting_guidance",
        "awaiting_feedback",
        "completed",
        "failed",
    ] = "active"
    termination_reason: str | None = None
    nodes: list[ExperimentNodeRecord] = Field(default_factory=list)
    rounds: list[ExperimentRound] = Field(default_factory=list)
    next_action: str = "execute_next_experiment"
    created_at: datetime = Field(default_factory=utc_now)
    completed_at: datetime | None = None


class AnalysisFinding(BaseModel):
    id: str = Field(default_factory=lambda: new_id("finding"))
    hypothesis_id: str
    statement: str
    effect_size: float | None = None
    confidence_interval: tuple[float, float] | None = None
    p_value: float | None = Field(default=None, ge=0, le=1)
    sample_size: int = Field(default=0, ge=0)
    analysis_method: str | None = None
    claim_verdict: Literal["supported", "rejected", "inconclusive", "not_tested"] = (
        "inconclusive"
    )
    supporting_run_ids: list[str] = Field(default_factory=list)
    contradicting_run_ids: list[str] = Field(default_factory=list)
    boundary_conditions: list[str] = Field(default_factory=list)
    verified: bool = False


class InnovationCandidate(BaseModel):
    id: str = Field(default_factory=lambda: new_id("innovation"))
    hypothesis_id: str
    title: str
    core_finding: str
    difference_from_prior_work: str
    mechanism_evidence: list[str]
    supporting_finding_ids: list[str]
    boundary_conditions: list[str]
    reproducibility_evidence: list[str]
    confidence: Literal["low", "medium", "high"]
    status: Literal[
        "unverified_candidate",
        "evidence_supported_candidate",
        "rejected",
    ] = "unverified_candidate"


class WorkflowEvent(BaseModel):
    id: str = Field(default_factory=lambda: new_id("event"))
    stage: ResearchStage
    actor: str
    action: str
    summary: str
    created_at: datetime = Field(default_factory=utc_now)
    payload: dict[str, Any] = Field(default_factory=dict)


class ExperimentProgressEvent(BaseModel):
    """Durable, structured progress emitted by the parallel experiment runner.

    The UI consumes these records as an SSE stream, while persisting them in the
    Research Ledger makes a disconnected run replayable and auditable.
    """

    id: str = Field(default_factory=lambda: new_id("experiment_event"))
    sequence: int = Field(ge=1)
    event_type: Literal[
        "campaign_started",
        "run_queued",
        "run_started",
        "run_finished",
        "round_guidance_required",
        "round_ready",
        "round_completed",
        "batch_completed",
        "campaign_completed",
        "results_locked",
        "statistics_completed",
        "innovation_review_completed",
        "hypothesis_revision_ready",
        "report_ready",
        "finalization_failed",
        "campaign_failed",
        "stream_completed",
    ]
    message: str
    campaign_id: str | None = None
    round_id: str | None = None
    hypothesis_id: str | None = None
    run_id: str | None = None
    status: str | None = None
    progress: float | None = Field(default=None, ge=0, le=1)
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class ArtifactRecord(BaseModel):
    id: str = Field(default_factory=lambda: new_id("artifact"))
    kind: str
    title: str
    path: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    provenance: list[str] = Field(default_factory=list)
    verified: bool = False
    created_at: datetime = Field(default_factory=utc_now)


class StaticValidationReport(BaseModel):
    passed: bool = False
    issues: list[str] = Field(default_factory=list)


class MethodSmokeResult(BaseModel):
    passed: bool = False
    summary: str = ""
    selected: list[str] | None = None
    deterministic: bool | None = None


class MethodImplementation(BaseModel):
    """AI-generated experiment code registered for one hypothesis strategy."""

    id: str = Field(default_factory=lambda: new_id("method"))
    kind: Literal["selection_strategy", "detector"] = "selection_strategy"
    name: str
    hypothesis_id: str
    source_code: str
    code_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    static_validation: StaticValidationReport = Field(default_factory=StaticValidationReport)
    smoke_result: MethodSmokeResult | None = None
    status: Literal["draft", "validated", "approved", "rejected"] = "draft"
    artifact_path: str | None = None
    provenance: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)


class ResearchProject(BaseModel):
    id: str = Field(default_factory=lambda: new_id("project"))
    spec: ProjectSpec
    stage: ResearchStage = ResearchStage.CREATED
    status: ProjectStatus = ProjectStatus.ACTIVE
    evidence: list[EvidenceRecord] = Field(default_factory=list)
    gaps: list[ResearchGap] = Field(default_factory=list)
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    hypothesis_history: list[Hypothesis] = Field(default_factory=list)
    experiment_plan: ExperimentPlan | None = None
    experiment_plan_history: list[ExperimentPlan] = Field(default_factory=list)
    dataset_audits: list[DatasetAuditRecord] = Field(default_factory=list)
    experiment_campaign: ExperimentCampaign | None = None
    experiment_campaign_history: list[ExperimentCampaign] = Field(default_factory=list)
    runs: list[ExperimentRun] = Field(default_factory=list)
    guidance_records: list[UserGuidanceRecord] = Field(default_factory=list)
    findings: list[AnalysisFinding] = Field(default_factory=list)
    finding_history: list[AnalysisFinding] = Field(default_factory=list)
    innovations: list[InnovationCandidate] = Field(default_factory=list)
    artifacts: list[ArtifactRecord] = Field(default_factory=list)
    method_implementations: list[MethodImplementation] = Field(default_factory=list)
    events: list[WorkflowEvent] = Field(default_factory=list)
    experiment_progress: list[ExperimentProgressEvent] = Field(default_factory=list)
    next_action: str = "formalize_scope"
    research_cycle: int = 1
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    def record_event(
        self,
        *,
        actor: str,
        action: str,
        summary: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        self.events.append(
            WorkflowEvent(
                stage=self.stage,
                actor=actor,
                action=action,
                summary=summary,
                payload=payload or {},
            )
        )
        self.updated_at = utc_now()
