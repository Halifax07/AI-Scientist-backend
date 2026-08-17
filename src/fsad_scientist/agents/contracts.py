from __future__ import annotations

from typing import Any, Protocol

from fsad_scientist.domain.models import (
    AnalysisFinding,
    ArtifactRecord,
    EvidenceRecord,
    ExperimentCell,
    ExperimentFeedbackProposal,
    ExperimentGuidanceDecision,
    ExperimentPlan,
    ExperimentRun,
    Hypothesis,
    InnovationCandidate,
    ResearchGap,
    ResearchProject,
)


class ScientistRuntime(Protocol):
    """Cognitive operations required by the durable workflow.

    A runtime may be deterministic (tests), Qwen through AgentScope, or another
    explicitly approved implementation. Experiment execution is deliberately not
    part of this interface; it is a deterministic tool boundary.
    """

    name: str

    async def formalize_scope(self, project: ResearchProject) -> ArtifactRecord: ...

    async def gather_evidence(
        self, project: ResearchProject
    ) -> tuple[list[EvidenceRecord], ArtifactRecord]: ...

    async def discover_gaps(self, project: ResearchProject) -> list[ResearchGap]: ...

    async def propose_hypotheses(self, project: ResearchProject) -> list[Hypothesis]: ...

    async def review_hypotheses(self, project: ResearchProject) -> list[Hypothesis]: ...

    async def design_experiments(self, project: ResearchProject) -> ExperimentPlan: ...

    async def recommend_next_experiments(
        self,
        project: ResearchProject,
        *,
        round_summary: dict[str, Any],
        allowed_cells: list[ExperimentCell],
    ) -> ExperimentFeedbackProposal: ...

    async def interpret_experiment_guidance(
        self,
        project: ResearchProject,
        *,
        guidance: str,
        candidate_runs: list[ExperimentRun],
    ) -> ExperimentGuidanceDecision: ...

    async def analyze_results(self, project: ResearchProject) -> list[AnalysisFinding]: ...

    async def revise_hypotheses(self, project: ResearchProject) -> list[Hypothesis]: ...

    async def review_innovations(
        self, project: ResearchProject
    ) -> list[InnovationCandidate]: ...

    async def build_report_manifest(self, project: ResearchProject) -> ArtifactRecord: ...
