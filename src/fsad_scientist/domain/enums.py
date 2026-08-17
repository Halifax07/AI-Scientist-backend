from enum import StrEnum


class ResearchStage(StrEnum):
    CREATED = "created"
    SCOPE_FORMALIZED = "scope_formalized"
    EVIDENCE_READY = "evidence_ready"
    GAPS_DISCOVERED = "gaps_discovered"
    HYPOTHESES_PROPOSED = "hypotheses_proposed"
    HYPOTHESES_REVIEWED = "hypotheses_reviewed"
    AWAITING_EXPERIMENT_APPROVAL = "awaiting_experiment_approval"
    EXPERIMENTS_QUEUED = "experiments_queued"
    RESULTS_READY = "results_ready"
    RESULTS_ANALYZED = "results_analyzed"
    INNOVATION_REVIEWED = "innovation_reviewed"
    REPORT_READY = "report_ready"
    FAILED = "failed"


class ProjectStatus(StrEnum):
    ACTIVE = "active"
    WAITING_HUMAN = "waiting_human"
    WAITING_EXTERNAL = "waiting_external"
    COMPLETED = "completed"
    FAILED = "failed"


class EvidenceStatus(StrEnum):
    UNVERIFIED = "unverified"
    METADATA_VERIFIED = "metadata_verified"
    VERIFIED = "verified"
    REJECTED = "rejected"


class HypothesisStatus(StrEnum):
    CANDIDATE = "candidate"
    SHORTLISTED = "shortlisted"
    APPROVED = "approved"
    SUPPORTED = "supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    REJECTED = "rejected"
    INCONCLUSIVE = "inconclusive"
    REVISED = "revised"


class RunStatus(StrEnum):
    PLANNED = "planned"
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
