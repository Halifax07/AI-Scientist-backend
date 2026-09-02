from __future__ import annotations

import asyncio
import hashlib
import json
from itertools import product
from pathlib import Path
from typing import Any, Literal

from fsad_scientist.agents.contracts import ScientistRuntime
from fsad_scientist.agents.mock_runtime import MockScientistRuntime
from fsad_scientist.config import PROJECT_ROOT
from fsad_scientist.datasets.models import DatasetManifest
from fsad_scientist.domain.enums import (
    HypothesisStatus,
    ProjectStatus,
    ResearchStage,
    RunStatus,
)
from fsad_scientist.domain.models import (
    AnalysisContract,
    ArtifactRecord,
    DatasetAuditRecord,
    EvidenceRecord,
    ExperimentFeedbackProposal,
    ExperimentGuidanceDecision,
    ExperimentProgressEvent,
    ExperimentRun,
    Hypothesis,
    HypothesisRanking,
    MethodImplementation,
    ProjectSpec,
    ResearchProject,
    StaticValidationReport,
    UserGuidanceRecord,
    utc_now,
)
from fsad_scientist.experiments.code_safety import (
    BUILTIN_DETECTORS,
    BUILTIN_STRATEGIES,
    sanitize_strategy_name,
    validate_detector_source,
    validate_strategy_source,
)
from fsad_scientist.experiments.detector_runner import (
    assemble_detector_file,
    run_detector_smoke,
)
from fsad_scientist.experiments.loop import (
    AdaptiveExperimentPlanner,
    is_supported_primary_metric,
    normalize_primary_metric,
)
from fsad_scientist.experiments.strategy_runner import (
    GeneratedStrategyRunner,
    assemble_strategy_file,
    run_strategy_smoke,
)
from fsad_scientist.repository import JsonProjectRepository


class WorkflowError(RuntimeError):
    pass


class InvalidTransitionError(WorkflowError):
    pass


class ApprovalRequiredError(WorkflowError):
    pass


class ResultsRequiredError(WorkflowError):
    pass


class ResearchWorkflow:
    """Durable, auditable state machine for autonomous scientific discovery."""

    def __init__(
        self,
        *,
        repository: JsonProjectRepository,
        runtime: ScientistRuntime,
        artifact_root: Path | None = None,
    ) -> None:
        self.repository = repository
        self.runtime = runtime
        self.artifact_root = (artifact_root or PROJECT_ROOT / "artifacts").resolve()
        self.experiment_planner = AdaptiveExperimentPlanner()

    def create_project(self, spec: ProjectSpec) -> ResearchProject:
        project = ResearchProject(spec=spec)
        project.record_event(
            actor="human",
            action="create_project",
            summary="用户提供研究领域、数据、现实约束和计算预算。",
        )
        return self.repository.save(project)

    async def advance_to_hypothesis_ranking(self, project_id: str) -> ResearchProject:
        """Run all machine-only discovery stages and stop at the ranking gate.

        Problem formalisation, evidence retrieval, gap discovery and candidate
        generation are deterministic workflow transitions from the user's point
        of view.  The only intentional pause is after candidates exist, where a
        human can review and score them before any experiment budget is spent.
        """

        project = self.repository.get(project_id)
        automatic_stages = {
            ResearchStage.CREATED,
            ResearchStage.SCOPE_FORMALIZED,
            ResearchStage.EVIDENCE_READY,
            ResearchStage.GAPS_DISCOVERED,
        }
        while project.stage in automatic_stages:
            project = await self.advance(project.id)
        if project.stage not in {
            ResearchStage.HYPOTHESES_PROPOSED,
            ResearchStage.HYPOTHESES_REVIEWED,
            ResearchStage.AWAITING_EXPERIMENT_APPROVAL,
            ResearchStage.EXPERIMENTS_QUEUED,
            ResearchStage.RESULTS_READY,
            ResearchStage.RESULTS_ANALYZED,
            ResearchStage.INNOVATION_REVIEWED,
            ResearchStage.REPORT_READY,
        }:
            raise InvalidTransitionError(
                f"The project cannot enter hypothesis ranking from {project.stage}"
            )
        if project.stage == ResearchStage.HYPOTHESES_PROPOSED:
            project = await self._ensure_hypothesis_review(project)
        return project

    async def rank_hypotheses(
        self,
        project_id: str,
        *,
        rankings: list[HypothesisRanking],
        auto_preregister: bool = True,
    ) -> ResearchProject:
        """Apply a human ranking and optionally generate the preregistration.

        The skeptic/meta-review agent still supplies machine scores, but those
        scores are shown as advice.  User selection and priority are persisted
        separately and become the sole source of the experiment portfolio.
        """

        project = self.repository.get(project_id)
        if project.stage == ResearchStage.HYPOTHESES_PROPOSED:
            project = await self._ensure_hypothesis_review(project)
        elif project.stage != ResearchStage.HYPOTHESES_REVIEWED:
            raise InvalidTransitionError(
                "Hypothesis ranking is only available after automatic candidate generation"
            )

        if not rankings:
            raise InvalidTransitionError("At least one hypothesis ranking is required")
        by_id = {item.id: item for item in project.hypotheses}
        unknown = [item.hypothesis_id for item in rankings if item.hypothesis_id not in by_id]
        if unknown:
            raise InvalidTransitionError(
                "Unknown hypothesis ids in ranking: " + ", ".join(unknown)
            )
        ranking_by_id = {item.hypothesis_id: item for item in rankings}
        if len(ranking_by_id) != len(rankings):
            raise InvalidTransitionError("Each hypothesis may appear only once in a ranking")
        selected = [item for item in rankings if item.selected]
        if not selected:
            raise InvalidTransitionError("Select at least one innovation for validation")

        for hypothesis in project.hypotheses:
            ranking = ranking_by_id.get(hypothesis.id)
            hypothesis.user_selected = bool(ranking and ranking.selected)
            hypothesis.user_priority = ranking.priority if ranking else None
            hypothesis.user_score = ranking.score if ranking else None
            hypothesis.user_review_note = ranking.note if ranking else None
            if hypothesis.user_selected:
                hypothesis.status = HypothesisStatus.SHORTLISTED
            elif hypothesis.status not in {
                HypothesisStatus.SUPPORTED,
                HypothesisStatus.REJECTED,
                HypothesisStatus.REVISED,
            }:
                hypothesis.status = HypothesisStatus.CANDIDATE

        selected_ids = [item.hypothesis_id for item in sorted(
            selected,
            key=lambda item: (item.priority, -item.score, item.hypothesis_id),
        )]
        unselected = [
            item
            for item in project.hypotheses
            if item.id not in set(selected_ids)
        ]
        unselected.sort(key=lambda item: -(item.score.elo if item.score else 0.0))
        project.hypotheses = [
            *(by_id[item_id] for item_id in selected_ids),
            *unselected,
        ]
        # The ranking gate is the only user interaction before execution.  If a
        # selected candidate names a custom selection strategy, generate and
        # validate that adapter in the background now; the user should not have
        # to leave the ranking screen and perform a separate implementation step.
        self._ensure_executable_hypotheses(project)
        self.repository.save(project)
        project = await self._prepare_selected_hypothesis_implementations(
            project.id,
            selected_ids=selected_ids,
        )
        self._move(
            project,
            stage=ResearchStage.HYPOTHESES_REVIEWED,
            status=ProjectStatus.ACTIVE,
            next_action="design_preregistered_experiment",
            actor="human_hypothesis_reviewer",
            summary=(
                f"用户已完成假设审阅与优先级筛选，选择 {len(selected_ids)} 个创新点进入验证。"
            ),
            payload={
                "selected_hypothesis_ids": selected_ids,
                "ranking_count": len(rankings),
                "human_gate": "ranking_only",
            },
        )
        project = self.repository.save(project)
        if auto_preregister:
            project = await self.advance(project.id)
        return project

    async def _ensure_hypothesis_review(
        self,
        project: ResearchProject,
    ) -> ResearchProject:
        """Materialize machine scores before showing the human ranking table."""

        if project.hypotheses and all(item.score is not None for item in project.hypotheses):
            return project
        project.hypotheses = await self.runtime.review_hypotheses(project)
        project.record_event(
            actor="skeptic_and_meta_review_agents",
            action="automatic_hypothesis_review",
            summary="候选假设已由后台反驳与元审查智能体自动评分，等待用户排序筛选。",
            payload={
                "hypothesis_count": len(project.hypotheses),
                "human_gate": "ranking_only",
            },
        )
        return self.repository.save(project)

    async def _prepare_selected_hypothesis_implementations(
        self,
        project_id: str,
        *,
        selected_ids: list[str],
    ) -> ResearchProject:
        """Make every user-selected innovation executable before planning.

        Built-in ``random``/``k_center`` comparisons need no generation.  For a
        custom strategy, the runtime creates a pure function and the existing
        static-validation plus smoke-test gates register it as ``validated``.
        A failed adapter is surfaced at the ranking request instead of being
        silently dropped from the user's selected portfolio.
        """

        project = self.repository.get(project_id)
        for hypothesis_id in selected_ids:
            project = self.repository.get(project_id)
            hypothesis = next(
                (item for item in project.hypotheses if item.id == hypothesis_id),
                None,
            )
            if hypothesis is None or hypothesis.analysis_contract is None:
                raise InvalidTransitionError(
                    f"Selected innovation has no analysis contract: {hypothesis_id}"
                )
            contract = hypothesis.analysis_contract
            if contract.kind not in {"selection_main_effect", "query_adaptation"}:
                raise InvalidTransitionError(
                    f"Selected innovation {hypothesis_id} is not supported by the current "
                    "experiment executor"
                )
            missing = [
                name
                for name in (contract.treatment, contract.control)
                if name not in BUILTIN_STRATEGIES
                and not any(
                    implementation.hypothesis_id == hypothesis_id
                    and implementation.kind == "selection_strategy"
                    and implementation.name == name
                    and implementation.status in {"validated", "approved"}
                    for implementation in project.method_implementations
                )
            ]
            if not missing:
                continue
            try:
                project = await self.implement_experiment_method(
                    project_id,
                    hypothesis_id=hypothesis_id,
                )
            except (InvalidTransitionError, ValueError) as exc:
                raise InvalidTransitionError(
                    f"创新点 {hypothesis_id} 的自动方法实现失败：{exc}"
                ) from exc
            refreshed = self.repository.get(project_id)
            refreshed_hypothesis = next(
                (item for item in refreshed.hypotheses if item.id == hypothesis_id),
                None,
            )
            if refreshed_hypothesis is None or refreshed_hypothesis.analysis_contract is None:
                raise InvalidTransitionError(
                    f"自动方法实现后找不到创新点：{hypothesis_id}"
                )
            unresolved = [
                name
                for name in (
                    refreshed_hypothesis.analysis_contract.treatment,
                    refreshed_hypothesis.analysis_contract.control,
                )
                if name not in BUILTIN_STRATEGIES
                and not any(
                    implementation.hypothesis_id == hypothesis_id
                    and implementation.kind == "selection_strategy"
                    and implementation.name == name
                    and implementation.status in {"validated", "approved"}
                    for implementation in refreshed.method_implementations
                )
            ]
            if unresolved:
                raise InvalidTransitionError(
                    f"创新点 {hypothesis_id} 仍缺少可执行策略：{', '.join(unresolved)}"
                )
            project = refreshed
        return project

    def record_experiment_progress(
        self,
        project_id: str,
        *,
        event_type: str,
        message: str,
        campaign_id: str | None = None,
        round_id: str | None = None,
        hypothesis_id: str | None = None,
        run_id: str | None = None,
        status: str | None = None,
        progress: float | None = None,
        payload: dict[str, Any] | None = None,
    ) -> ExperimentProgressEvent:
        """Append one durable structured event for SSE/replay consumers."""

        project = self.repository.get(project_id)
        event = ExperimentProgressEvent(
            sequence=(project.experiment_progress[-1].sequence + 1)
            if project.experiment_progress
            else 1,
            event_type=event_type,  # type: ignore[arg-type]
            message=message,
            campaign_id=campaign_id,
            round_id=round_id,
            hypothesis_id=hypothesis_id,
            run_id=run_id,
            status=status,
            progress=progress,
            payload=payload or {},
        )
        project.experiment_progress.append(event)
        self.repository.save(project)
        return event

    def select_parallel_runs(
        self,
        project_id: str,
        *,
        run_ids: list[str] | None = None,
        limit: int | None = None,
    ) -> list[ExperimentRun]:
        """Select queued runs from all active parallel innovation Rounds."""

        project = self.repository.get(project_id)
        campaign = project.experiment_campaign
        if campaign is None or campaign.execution_mode != "parallel":
            raise InvalidTransitionError("The project has no parallel experiment campaign")
        if campaign.status != "active":
            raise InvalidTransitionError("The parallel campaign is not accepting runs")
        candidates = self.experiment_planner.queued_runs(project)
        # Before every innovation Round receives its one midpoint decision,
        # only its first pre-registered iteration may be dispatched.  This
        # prevents a client from accidentally running iterations 2–3 before
        # the human-in-the-loop gate is shown.
        pending_round_ids = {
            item.id
            for item in campaign.rounds
            if not item.guidance_received
        }
        if pending_round_ids:
            candidates = [
                item
                for item in candidates
                if item.round_id not in pending_round_ids or item.iteration == 1
            ]
        if run_ids is not None:
            requested = set(run_ids)
            unknown = requested - {item.id for item in candidates}
            if unknown:
                raise InvalidTransitionError(
                    "Requested runs are not queued in the active campaign: "
                    + ", ".join(sorted(unknown))
                )
            candidates = [item for item in candidates if item.id in requested]
        if limit is not None:
            candidates = candidates[:limit]
        if not candidates:
            raise ResultsRequiredError("No queued experiment is available in the parallel campaign")
        project.record_event(
            actor="parallel_experiment_scheduler",
            action="select_parallel_runs",
            summary=f"已选择 {len(candidates)} 个跨创新点实验运行并准备并行执行。",
            payload={
                "run_ids": [item.id for item in candidates],
                "round_ids": sorted({item.round_id for item in candidates if item.round_id}),
            },
        )
        self.repository.save(project)
        return candidates

    def fail_parallel_campaign(
        self,
        project_id: str,
        *,
        reason: str,
    ) -> ResearchProject:
        """Persist a fatal scheduler error without fabricating experiment results."""

        project = self.repository.get(project_id)
        campaign = project.experiment_campaign
        if campaign is None or campaign.execution_mode != "parallel":
            raise InvalidTransitionError("The project has no parallel experiment campaign")
        if campaign.status != "completed":
            campaign.status = "failed"
            campaign.termination_reason = f"parallel_stream_failed: {reason[:500]}"
            campaign.next_action = "inspect_failed_parallel_campaign"
            campaign.completed_at = utc_now()
            project.status = ProjectStatus.WAITING_EXTERNAL
            project.next_action = campaign.next_action
            project.record_event(
                actor="parallel_experiment_scheduler",
                action="fail_parallel_campaign",
                summary="并行实验调度发生致命错误；系统保留已完成结果并停止继续执行。",
                payload={"reason": reason},
            )
            return self.repository.save(project)
        return project

    async def complete_parallel_campaign(self, project_id: str) -> ResearchProject:
        """Analyze each completed parallel Round and close the campaign.

        The advisor is invoked per innovation on an immutable project snapshot so
        Qwen calls can run concurrently without races in the durable ledger.
        """

        project = self.repository.get(project_id)
        campaign = project.experiment_campaign
        if campaign is None or campaign.execution_mode != "parallel":
            raise InvalidTransitionError("The project has no parallel experiment campaign")
        campaign_run_ids = {
            run_id
            for experiment_round in campaign.rounds
            for run_id in experiment_round.run_ids
        }
        if any(
            run.status not in {RunStatus.SUCCEEDED, RunStatus.FAILED}
            for run in project.runs
            if run.id in campaign_run_ids
        ):
            raise ResultsRequiredError("Parallel campaign still has non-terminal runs")

        ready = [
            item
            for item in campaign.rounds
            if item.status in {"ready_for_feedback", "completed"}
        ]
        if not ready:
            raise ResultsRequiredError("No completed parallel Round is ready for analysis")

        async def review_one(experiment_round):
            snapshot = project.model_copy(deep=True)
            snapshot_campaign = snapshot.experiment_campaign
            if snapshot_campaign is None:
                raise InvalidTransitionError("The project has no experiment campaign")
            snapshot_campaign.hypothesis_id = experiment_round.hypothesis_id
            snapshot_campaign.treatment = experiment_round.treatment
            snapshot_campaign.control = experiment_round.control
            snapshot_campaign.metric = experiment_round.metric
            summary = self.experiment_planner.summarize_round(
                snapshot, round_id=experiment_round.id
            )
            try:
                proposal = await self.runtime.recommend_next_experiments(
                    snapshot,
                    round_summary=summary,
                    allowed_cells=[],
                )
            except Exception as exc:  # pragma: no cover - defensive runtime boundary
                proposal = ExperimentFeedbackProposal(
                    advisor=self.runtime.name,
                    decision="diagnose",
                    rationale=f"自动分析智能体暂不可用：{type(exc).__name__}: {exc}",
                    next_phase="complete",
                    stop=False,
                )
            return experiment_round.id, summary, proposal

        reviews = await asyncio.gather(*(review_one(item) for item in ready))
        for round_id, summary, proposal in reviews:
            current = next(item for item in campaign.rounds if item.id == round_id)
            current.result_summary = summary
            current.feedback = proposal
            current.status = "completed"
            current.completed_at = current.completed_at or utc_now()
            self.experiment_planner._update_nodes_for_round(
                campaign, current, project, summary
            )
            project.record_event(
                actor=proposal.advisor,
                action="complete_parallel_round",
                summary=(
                    f"Round {current.index}（{current.hypothesis_id}）已完成统计汇总，"
                    "结果已加入创新点审查队列。"
                ),
                payload={
                    "round_id": current.id,
                    "hypothesis_id": current.hypothesis_id,
                    "summary": summary,
                    "feedback": proposal.model_dump(mode="json"),
                },
            )
        campaign.status = "completed"
        campaign.current_round = max(item.index for item in campaign.rounds)
        campaign.next_action = "analyze_verified_results"
        campaign.termination_reason = "selected_innovations_completed_in_parallel"
        campaign.completed_at = utc_now()
        project.status = ProjectStatus.WAITING_EXTERNAL
        project.next_action = campaign.next_action
        return self.repository.save(project)

    async def start_next_research_cycle(
        self,
        project_id: str,
        *,
        user_guidance: str,
    ) -> ResearchProject:
        """Record human direction before the hypothesis-revision transition."""

        project = self.repository.get(project_id)
        if project.stage != ResearchStage.RESULTS_ANALYZED:
            raise InvalidTransitionError("The project is not ready for a new research cycle")
        if not self._should_revise(project):
            raise InvalidTransitionError(
                "The current findings do not open an evidence-driven revision cycle"
            )
        guidance = user_guidance.strip()
        if len(guidance) < 2:
            raise InvalidTransitionError("Guidance for the next research cycle is required")
        record = UserGuidanceRecord(
            scope="research_cycle",
            target_action="start_next_research_cycle",
            text=guidance,
            research_cycle=project.research_cycle,
        )
        project.guidance_records.append(record)
        project.spec.user_guidance.append(
            f"研究循环 {project.research_cycle + 1} 人工指导：{guidance}"
        )
        project.record_event(
            actor="human",
            action="guide_next_research_cycle",
            summary="用户已在投入下一循环预算前提交研究指导，等待 AI Scientist 修订假设。",
            payload={
                "guidance_id": record.id,
                "guidance": guidance,
                "target_research_cycle": project.research_cycle + 1,
            },
        )
        self.repository.save(project)
        return await self.advance(project_id, cycle_guidance_id=record.id)

    async def advance(
        self,
        project_id: str,
        *,
        cycle_guidance_id: str | None = None,
    ) -> ResearchProject:
        project = self.repository.get(project_id)

        if project.stage == ResearchStage.CREATED:
            artifact = await self.runtime.formalize_scope(project)
            project.artifacts.append(artifact)
            self._move(
                project,
                stage=ResearchStage.SCOPE_FORMALIZED,
                status=ProjectStatus.ACTIVE,
                next_action="gather_evidence",
                actor="supervisor",
                summary="研究目标已转化为结构化、可审计的科学问题。",
            )

        elif project.stage == ResearchStage.SCOPE_FORMALIZED:
            evidence, artifact = await self.runtime.gather_evidence(project)
            merged = {_evidence_key(item): item for item in project.evidence}
            for item in evidence:
                merged.setdefault(_evidence_key(item), item)
            project.evidence = list(merged.values())
            project.artifacts.append(artifact)
            self._move(
                project,
                stage=ResearchStage.EVIDENCE_READY,
                status=ProjectStatus.ACTIVE,
                next_action="discover_research_gaps",
                actor="evidence_agent",
                summary="文献候选和证据校验计划已建立。",
                payload={"evidence_count": len(evidence)},
            )

        elif project.stage == ResearchStage.EVIDENCE_READY:
            project.gaps = await self.runtime.discover_gaps(project)
            self._move(
                project,
                stage=ResearchStage.GAPS_DISCOVERED,
                status=ProjectStatus.ACTIVE,
                next_action="propose_falsifiable_hypotheses",
                actor="data_and_gap_agents",
                summary="系统已根据证据和场景约束生成研究空白候选。",
                payload={"gap_count": len(project.gaps)},
            )

        elif project.stage == ResearchStage.GAPS_DISCOVERED:
            project.hypotheses = await self.runtime.propose_hypotheses(project)
            self._move(
                project,
                stage=ResearchStage.HYPOTHESES_PROPOSED,
                status=ProjectStatus.ACTIVE,
                next_action="debate_rank_and_evolve_hypotheses",
                actor="hypothesis_agent",
                summary="创新候选已转化为具有明确零假设和证伪条件的科学假设。",
                payload={"hypothesis_count": len(project.hypotheses)},
            )

        elif project.stage == ResearchStage.HYPOTHESES_PROPOSED:
            project.hypotheses = await self.runtime.review_hypotheses(project)
            self._ensure_executable_hypotheses(project)
            self._move(
                project,
                stage=ResearchStage.HYPOTHESES_REVIEWED,
                status=ProjectStatus.ACTIVE,
                next_action="design_preregistered_experiment",
                actor="skeptic_and_meta_review_agents",
                summary="候选假设已完成反驳、可证伪性审查和排序。",
                payload={
                    "shortlisted": [
                        item.id
                        for item in project.hypotheses
                        if item.status == HypothesisStatus.SHORTLISTED
                    ]
                },
            )

        elif project.stage == ResearchStage.HYPOTHESES_REVIEWED:
            self._ensure_executable_hypotheses(project)
            if project.experiment_plan is not None:
                project.experiment_plan_history.append(
                    project.experiment_plan.model_copy(deep=True)
                )
            project.experiment_plan = await self.runtime.design_experiments(project)
            self._scope_experiment_plan_to_primary_hypothesis(project)
            self._move(
                project,
                stage=ResearchStage.AWAITING_EXPERIMENT_APPROVAL,
                status=ProjectStatus.WAITING_HUMAN,
                next_action="human_approve_preregistered_plan",
                actor="experiment_planner",
                summary="预注册实验计划已生成，等待人工确认预算和安全边界。",
                payload={
                    "plan_id": project.experiment_plan.id,
                    "digest": project.experiment_plan.preregistration_digest,
                },
            )

        elif project.stage == ResearchStage.AWAITING_EXPERIMENT_APPROVAL:
            raise ApprovalRequiredError("The preregistered experiment plan needs approval")

        elif project.stage == ResearchStage.EXPERIMENTS_QUEUED:
            raise ResultsRequiredError(
                "Experiment runs are queued; execute them or import verified results first"
            )

        elif project.stage == ResearchStage.RESULTS_READY:
            if project.findings:
                project.finding_history.extend(
                    item.model_copy(deep=True) for item in project.findings
                )
            project.findings = await self.runtime.analyze_results(project)
            hypothesis_by_id = {item.id: item for item in project.hypotheses}
            verdict_status = {
                "supported": HypothesisStatus.SUPPORTED,
                "rejected": HypothesisStatus.REJECTED,
                "inconclusive": HypothesisStatus.INCONCLUSIVE,
            }
            for finding in project.findings:
                hypothesis = hypothesis_by_id.get(finding.hypothesis_id)
                status_for_verdict = verdict_status.get(finding.claim_verdict)
                if hypothesis is not None and status_for_verdict is not None:
                    hypothesis.status = status_for_verdict
            self._move(
                project,
                stage=ResearchStage.RESULTS_ANALYZED,
                status=ProjectStatus.ACTIVE,
                next_action="review_innovation_candidates",
                actor="statistics_and_vision_review_agents",
                summary="真实运行结果已完成统计分析，未满足条件的假设不会被强行接受。",
                payload={"finding_count": len(project.findings)},
            )

        elif project.stage == ResearchStage.RESULTS_ANALYZED:
            cycle_guidance = next(
                (
                    item
                    for item in project.guidance_records
                    if item.id == cycle_guidance_id and item.scope == "research_cycle"
                ),
                None,
            )
            revised = (
                await self.runtime.revise_hypotheses(project)
                if self._should_revise(project)
                else []
            )
            if revised:
                previous = [item.model_copy(deep=True) for item in project.hypotheses]
                revised_parent_ids = {
                    item.parent_hypothesis_id
                    for item in revised
                    if item.parent_hypothesis_id is not None
                }
                for item in previous:
                    if item.id in revised_parent_ids:
                        item.status = HypothesisStatus.REVISED
                project.hypothesis_history.extend(previous)
                project.finding_history.extend(
                    item.model_copy(deep=True) for item in project.findings
                )
                if project.experiment_plan is not None:
                    project.experiment_plan_history.append(
                        project.experiment_plan.model_copy(deep=True)
                    )
                project.hypotheses = revised
                project.findings = []
                project.experiment_plan = None
                if project.experiment_campaign is not None:
                    project.experiment_campaign_history.append(
                        project.experiment_campaign.model_copy(deep=True)
                    )
                    project.experiment_campaign = None
                project.research_cycle += 1
                if cycle_guidance is not None:
                    cycle_guidance.advisor = self.runtime.name
                    cycle_guidance.interpretation = (
                        "AI Scientist 已把该建议作为新假设的修订约束，并将其交给后续"
                        "辩论、预注册和预算审查继续校验。"
                    )
                    cycle_guidance.disposition = "applied"
                    cycle_guidance.rationale = (
                        "人工建议影响假设范围和下一循环关注重点，但不会绕过证据、"
                        "可证伪性与预注册边界。"
                    )
                    cycle_guidance.affected_ids = [item.id for item in revised]
                    cycle_guidance.protected_constraints = [
                        "新假设仍需通过反驳与 Elo 排名",
                        "新实验计划必须重新预注册并由人批准",
                        "历史结果和原假设不可覆盖",
                    ]
                self._move(
                    project,
                    stage=ResearchStage.HYPOTHESES_PROPOSED,
                    status=ProjectStatus.ACTIVE,
                    next_action="debate_rank_and_evolve_revised_hypotheses",
                    actor="hypothesis_revision_agent",
                    summary="真实结果尚未支持主张；系统已缩小或改写机制假设并启动下一研究轮。",
                    payload={
                        "research_cycle": project.research_cycle,
                        "revised_hypothesis_ids": [item.id for item in revised],
                        "parent_hypothesis_ids": sorted(revised_parent_ids),
                        "guidance_id": cycle_guidance.id if cycle_guidance else None,
                    },
                )
            else:
                if cycle_guidance is not None:
                    cycle_guidance.advisor = self.runtime.name
                    cycle_guidance.interpretation = "本次运行未生成满足结构约束的修订假设。"
                    cycle_guidance.disposition = "not_applicable"
                    cycle_guidance.rationale = "系统保留原始建议，但未据此伪造新的研究主张。"
                if project.experiment_campaign is not None:
                    project.experiment_campaign_history.append(
                        project.experiment_campaign.model_copy(deep=True)
                    )
                    project.experiment_campaign = None
                project.innovations = await self.runtime.review_innovations(project)
                self._move(
                    project,
                    stage=ResearchStage.INNOVATION_REVIEWED,
                    status=ProjectStatus.ACTIVE,
                    next_action="build_competition_report",
                    actor="innovation_review_agent",
                    summary="候选发现已完成新颖性、证据强度、边界和复现性审查。",
                    payload={"innovation_count": len(project.innovations)},
                )

        elif project.stage == ResearchStage.INNOVATION_REVIEWED:
            artifact = await self.runtime.build_report_manifest(project)
            project.artifacts.append(artifact)
            self._move(
                project,
                stage=ResearchStage.REPORT_READY,
                status=ProjectStatus.COMPLETED,
                next_action="export_report_and_reproduction_bundle",
                actor="scientific_reporter",
                summary="赛题要求的研究报告清单与复现证据索引已生成。",
            )

        else:
            raise InvalidTransitionError(f"No transition is available from {project.stage}")

        return self.repository.save(project)

    def approve_experiment_plan(self, project_id: str, *, approved_by: str) -> ResearchProject:
        project = self.repository.get(project_id)
        if project.stage != ResearchStage.AWAITING_EXPERIMENT_APPROVAL:
            raise InvalidTransitionError("Project is not waiting for experiment approval")
        if project.experiment_plan is None:
            raise InvalidTransitionError("Project has no experiment plan")

        excluded_hypothesis_ids = self._scope_experiment_plan_to_primary_hypothesis(
            project
        )
        if excluded_hypothesis_ids:
            self.repository.save(project)
            raise InvalidTransitionError(
                "预注册计划已移除当前工具链不可执行的创新点；请刷新页面复核后再次批准"
            )

        self._enforce_method_implementation_gate(project)
        project.experiment_plan.approved = True
        project.experiment_plan.approved_by = approved_by
        project.experiment_plan.approved_at = utc_now()
        for hypothesis in project.hypotheses:
            if hypothesis.id in project.experiment_plan.hypothesis_ids:
                hypothesis.status = HypothesisStatus.APPROVED

        new_runs = self._build_feasibility_runs(project)
        project.runs.extend(new_runs)
        self._move(
            project,
            stage=ResearchStage.EXPERIMENTS_QUEUED,
            status=ProjectStatus.WAITING_EXTERNAL,
            next_action="execute_or_import_verified_results",
            actor="human_and_experiment_planner",
            summary="实验计划已批准；首批可行性运行清单已冻结并排队。",
            payload={
                "approved_by": approved_by,
                "queued_runs": len(new_runs),
                "total_runs": len(project.runs),
            },
        )
        return self.repository.save(project)

    async def implement_experiment_method(
        self,
        project_id: str,
        *,
        hypothesis_id: str,
    ) -> ResearchProject:
        """Generate, statically validate, smoke-test and register one custom strategy.

        Approved implementations are immutable; validated ones are reused without a
        new LLM call; draft/rejected ones are regenerated. Only implementations that
        pass static validation and the behavioral smoke test reach ``validated``
        status, and only ``approved`` implementations may enter a campaign.
        """

        project = self.repository.get(project_id)
        hypothesis = next(
            (item for item in project.hypotheses if item.id == hypothesis_id),
            None,
        )
        if hypothesis is None:
            raise InvalidTransitionError(f"Unknown hypothesis id: {hypothesis_id}")
        contract = hypothesis.analysis_contract
        if contract is None:
            raise InvalidTransitionError("The hypothesis has no analysis contract")
        canonical_treatment = _strategy_alias(contract.treatment) or contract.treatment
        canonical_control = _strategy_alias(contract.control) or contract.control
        replacement_names = {
            "treatment": canonical_treatment,
            "control": canonical_control,
        }
        generation_targets: list[tuple[str, str]] = []
        used_names: set[str] = set()
        for slot, original_name, canonical_name in (
            ("treatment", contract.treatment, canonical_treatment),
            ("control", contract.control, canonical_control),
        ):
            if canonical_name in BUILTIN_STRATEGIES:
                continue
            strategy_name = sanitize_strategy_name(original_name)
            if strategy_name in used_names:
                strategy_name = sanitize_strategy_name(f"{strategy_name}_{slot}")
            used_names.add(strategy_name)
            replacement_names[slot] = strategy_name
            generation_targets.append((slot, strategy_name))

        if not generation_targets:
            strategy_name = sanitize_strategy_name(f"ai_strategy_{hypothesis.id}")
            replacement_names["treatment"] = strategy_name
            replacement_names["control"] = (
                canonical_control
                if canonical_treatment == "k_center"
                else canonical_treatment
            )
            generation_targets.append(("treatment", strategy_name))

        implementation_digests: dict[str, str] = {}
        for slot, strategy_name in generation_targets:
            peer_slot = "control" if slot == "treatment" else "treatment"
            implementation = await self._implement_selection_strategy(
                project,
                hypothesis=hypothesis,
                strategy_name=strategy_name,
                control_name=replacement_names[peer_slot],
                reserved_digests=set(implementation_digests.values()),
            )
            if implementation.status not in {"approved", "validated"}:
                return self.repository.save(project)
            if implementation.code_digest in implementation_digests.values():
                implementation.status = "rejected"
                project.record_event(
                    actor="method_registry",
                    action="implement_experiment_method",
                    summary=(
                        f"策略 {implementation.name} 与另一实验臂实现完全相同，已拒绝。"
                    ),
                    payload={"code_digest": implementation.code_digest},
                )
                return self.repository.save(project)
            implementation_digests[strategy_name] = implementation.code_digest

        self._sync_generated_strategy_references(
            project,
            hypothesis=hypothesis,
            contract=contract,
            treatment_name=replacement_names["treatment"],
            control_name=replacement_names["control"],
            implementation_digests=implementation_digests,
        )
        return self.repository.save(project)

    async def _implement_selection_strategy(
        self,
        project: ResearchProject,
        *,
        hypothesis: Hypothesis,
        strategy_name: str,
        control_name: str,
        reserved_digests: set[str],
    ) -> MethodImplementation:
        existing = next(
            (
                item
                for item in project.method_implementations
                if item.hypothesis_id == hypothesis.id
                and item.kind == "selection_strategy"
                and item.name == strategy_name
                and item.status in {"approved", "validated"}
                and item.code_digest not in reserved_digests
            ),
            None,
        )
        if existing is not None:
            self._reject_non_mock_deterministic_fallback(
                existing,
                implementation_kind="选样策略",
            )
            if existing.name in BUILTIN_STRATEGIES:
                raise InvalidTransitionError(
                    f"Generated strategy {existing.name} cannot use a built-in name"
                )
            project.record_event(
                actor="method_registry",
                action="implement_experiment_method",
                summary=f"复用已有 {existing.status} 实现 {existing.name}，未发起新的生成。",
                payload={"code_digest": existing.code_digest, "status": existing.status},
            )
            return existing

        approved_conflict = next(
            (
                item
                for item in project.method_implementations
                if item.hypothesis_id == hypothesis.id
                and item.kind == "selection_strategy"
                and item.name == strategy_name
                and item.status == "approved"
            ),
            None,
        )
        if approved_conflict is not None:
            raise InvalidTransitionError(
                f"Approved strategy {strategy_name} is immutable and conflicts with another arm"
            )

        implementation = await self.runtime.implement_selection_strategy(
            project,
            hypothesis=hypothesis,
            strategy_name=strategy_name,
            control_name=control_name,
        )
        self._reject_non_mock_deterministic_fallback(
            implementation,
            implementation_kind="选样策略",
        )
        implementation.name = strategy_name
        project.method_implementations = [
            item
            for item in project.method_implementations
            if not (
                item.hypothesis_id == hypothesis.id
                and item.kind == "selection_strategy"
                and item.name == strategy_name
            )
        ]
        project.method_implementations.append(implementation)

        validation = validate_strategy_source(implementation.source_code)
        implementation.static_validation = StaticValidationReport(
            passed=validation.passed,
            issues=validation.issues,
        )
        if not validation.passed:
            implementation.status = "rejected"
            project.record_event(
                actor="code_safety_validator",
                action="implement_experiment_method",
                summary=f"生成的策略 {implementation.name} 未通过静态校验，已拒绝。",
                payload={"issues": validation.issues},
            )
            return implementation

        assembled = assemble_strategy_file(implementation.source_code)
        digest = hashlib.sha256(assembled.encode("utf-8")).hexdigest()
        if digest != implementation.code_digest:
            implementation.status = "rejected"
            project.record_event(
                actor="code_safety_validator",
                action="implement_experiment_method",
                summary=f"策略 {implementation.name} 的注册摘要与源码不一致，已拒绝。",
                payload={"expected": implementation.code_digest, "actual": digest},
            )
            return implementation

        strategy_path = self.artifact_root / "generated_methods" / digest / "strategy.py"
        strategy_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = strategy_path.with_suffix(".py.tmp")
        temporary.write_text(assembled, encoding="utf-8")
        temporary.replace(strategy_path)
        implementation.artifact_path = str(strategy_path.resolve())
        project.artifacts.append(
            ArtifactRecord(
                kind="generated_strategy",
                title=f"生成选样策略 {implementation.name}",
                path=str(strategy_path.resolve()),
                payload={
                    "name": implementation.name,
                    "code_digest": implementation.code_digest,
                    "hypothesis_id": hypothesis.id,
                },
                provenance=[self.runtime.name, "code_safety_validator"],
                verified=False,
            )
        )

        smoke = run_strategy_smoke(
            implementation,
            GeneratedStrategyRunner(self.artifact_root),
        )
        implementation.smoke_result = smoke
        if not smoke.passed:
            implementation.status = "rejected"
            project.record_event(
                actor="strategy_smoke_runner",
                action="implement_experiment_method",
                summary=f"策略 {implementation.name} 冒烟测试未通过，已拒绝。",
                payload={"smoke_summary": smoke.summary},
            )
            return implementation

        implementation.status = "validated"
        project.record_event(
            actor=self.runtime.name,
            action="implement_experiment_method",
            summary=(
                f"策略 {implementation.name} 已生成并通过静态校验与冒烟测试，"
                "等待计划批准后注册执行。"
            ),
            payload={
                "name": implementation.name,
                "code_digest": implementation.code_digest,
                "strategy_name": strategy_name,
                "control_name": control_name,
            },
        )
        return implementation

    @staticmethod
    def _plan_preregistration_digest(plan) -> str:
        payload = plan.model_dump(
            mode="json",
            exclude={
                "id",
                "preregistration_digest",
                "approved",
                "approved_by",
                "approved_at",
            },
        )
        return hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()

    def _scope_experiment_plan_to_primary_hypothesis(
        self,
        project: ResearchProject,
    ) -> list[str]:
        """Keep every executable innovation in the preregistered portfolio.

        The historical name is retained for saved-project/API compatibility.  A
        campaign no longer collapses the plan to one primary hypothesis: each
        retained hypothesis receives one Round of three internal iterations.
        """
        plan = project.experiment_plan
        if plan is None:
            return []

        hypotheses = {item.id: item for item in project.hypotheses}
        generated_detectors = [
            item
            for item in project.method_implementations
            if item.kind == "detector"
            and item.status in {"validated", "approved"}
            and item.name in plan.detectors
        ]
        generated_detector_hypothesis_ids = {
            item.hypothesis_id for item in generated_detectors
        }
        requested_detectors = list(plan.detectors)
        if generated_detectors:
            plan.detectors = list(dict.fromkeys(item.name for item in generated_detectors))
        supported_hypothesis_ids: list[str] = []
        normalized_contracts: dict[str, AnalysisContract] = {}
        for hypothesis_id in plan.hypothesis_ids:
            hypothesis = hypotheses.get(hypothesis_id)
            contract = hypothesis.analysis_contract if hypothesis is not None else None
            if (
                contract is None
                or contract.kind not in {"selection_main_effect", "query_adaptation"}
                or not is_supported_primary_metric(contract.metric)
            ):
                continue
            normalized_metric = normalize_primary_metric(contract.metric)
            if generated_detectors and (
                hypothesis_id not in generated_detector_hypothesis_ids
                or normalized_metric not in {"image_auroc", "image_ap"}
            ):
                continue
            normalized_contract = contract.model_copy(
                update={"metric": normalized_metric}
            )
            hypothesis.analysis_contract = normalized_contract
            planned_contract = plan.hypothesis_contracts.get(hypothesis_id)
            if planned_contract is not None:
                if not is_supported_primary_metric(planned_contract.metric):
                    continue
                normalized_contract = planned_contract.model_copy(
                    update={"metric": normalize_primary_metric(planned_contract.metric)}
                )
            supported_hypothesis_ids.append(hypothesis_id)
            normalized_contracts[hypothesis_id] = normalized_contract.model_copy(deep=True)
        if not supported_hypothesis_ids:
            raise InvalidTransitionError(
                "预注册计划没有当前执行器可支持的创新点，请重新设计实验计划"
            )
        if (
            plan.hypothesis_ids == supported_hypothesis_ids
            and plan.hypothesis_contracts == normalized_contracts
            and requested_detectors == plan.detectors
        ):
            return []

        excluded_hypothesis_ids = [
            hypothesis_id
            for hypothesis_id in plan.hypothesis_ids
            if hypothesis_id not in supported_hypothesis_ids
        ]
        plan.hypothesis_ids = supported_hypothesis_ids
        plan.hypothesis_contracts = normalized_contracts
        plan.preregistration_digest = self._plan_preregistration_digest(plan)
        project.record_event(
            actor="experiment_plan_scope_guard",
            action="scope_experiment_plan",
            summary="预注册计划已保留全部可执行创新点，每个创新点对应一个实验 Round。",
            payload={
                "retained_hypothesis_ids": supported_hypothesis_ids,
                "excluded_hypothesis_ids": excluded_hypothesis_ids,
            },
        )
        return excluded_hypothesis_ids

    @staticmethod
    def _contract_is_supported_primary(contract: AnalysisContract | None) -> bool:
        if contract is None or contract.kind not in {
            "selection_main_effect",
            "query_adaptation",
        }:
            return False
        if not is_supported_primary_metric(contract.metric):
            return False
        protocol_markers = (
            "compression ratio",
            "compression rate",
            "pool size",
            "candidate pool",
            "压缩率",
            "候选池",
            "池大小",
        )
        names = (contract.treatment.casefold(), contract.control.casefold())
        return not any(
            marker in name for name in names for marker in protocol_markers
        )

    @staticmethod
    def _contract_is_experimentable(
        project: ResearchProject,
        hypothesis: Hypothesis,
    ) -> bool:
        contract = hypothesis.analysis_contract
        if contract is None or contract.kind not in {
            "selection_main_effect",
            "query_adaptation",
        }:
            return False
        approved = BUILTIN_STRATEGIES | {
            item.name
            for item in project.method_implementations
            if item.kind == "selection_strategy" and item.status == "approved"
        }
        return contract.treatment in approved and contract.control in approved

    @staticmethod
    def _primary_hypothesis_priority(
        project: ResearchProject,
        hypothesis: Hypothesis,
    ) -> int:
        """Prefer executable custom hypotheses, then contracts using built-ins.

        The planner may return a natural-language custom strategy before its
        implementation endpoint has been called. Such a candidate must not hide
        an executable built-in core hypothesis during plan scoping. Once the user
        has generated and validated the custom implementation, it remains the
        preferred candidate even when a built-in candidate is also present.
        """

        contract = hypothesis.analysis_contract
        if contract is None:
            return 2
        custom_names = [
            name
            for name in (contract.treatment, contract.control)
            if name not in BUILTIN_STRATEGIES
        ]
        if not custom_names:
            return 1
        for name in custom_names:
            implementation = next(
                (
                    item
                    for item in project.method_implementations
                    if item.kind == "selection_strategy"
                    and item.hypothesis_id == hypothesis.id
                    and item.name == name
                ),
                None,
            )
            if implementation is None or implementation.status not in {
                "validated",
                "approved",
            }:
                return 2
            if not implementation.static_validation.passed or not (
                implementation.smoke_result is not None
                and implementation.smoke_result.passed
            ):
                return 2
        return 0

    def _sync_generated_strategy_references(
        self,
        project: ResearchProject,
        *,
        hypothesis: Hypothesis,
        contract: AnalysisContract,
        treatment_name: str,
        control_name: str,
        implementation_digests: dict[str, str],
    ) -> None:
        generated_names = set(implementation_digests)
        invalid_names = generated_names & BUILTIN_STRATEGIES
        if invalid_names:
            raise InvalidTransitionError(
                f"Generated strategies cannot use built-in names: {sorted(invalid_names)}"
            )
        replacement = contract.model_copy(
            update={"treatment": treatment_name, "control": control_name}
        )
        hypothesis.analysis_contract = replacement
        plan = project.experiment_plan
        if plan is None or hypothesis.id not in plan.hypothesis_ids:
            return
        plan.hypothesis_contracts[hypothesis.id] = replacement.model_copy(deep=True)
        replacements = {
            contract.treatment: treatment_name,
            contract.control: control_name,
        }
        plan.selection_strategies = list(
            dict.fromkeys(
                replacements.get(name, name) for name in plan.selection_strategies
            )
        )
        for name in (treatment_name, control_name):
            if name not in plan.selection_strategies:
                plan.selection_strategies.append(name)
        for original_name, replacement_name in replacements.items():
            if original_name != replacement_name:
                plan.method_implementation_digests.pop(original_name, None)
        plan.method_implementation_digests.update(implementation_digests)
        plan.preregistration_digest = self._plan_preregistration_digest(plan)

    async def implement_experiment_detector(
        self,
        project_id: str,
        *,
        name_stem: str,
        hypothesis_id: str,
        reference_description: str | None = None,
    ) -> ResearchProject:
        """Generate, statically validate, smoke-test and register one detector.

        Mirrors ``implement_experiment_method``: approved implementations are
        immutable, validated ones are reused, draft/rejected are regenerated,
        and only validated implementations may be approved for campaigns.
        """

        project = self.repository.get(project_id)
        hypothesis = next(
            (item for item in project.hypotheses if item.id == hypothesis_id),
            None,
        )
        if hypothesis is None:
            raise InvalidTransitionError(f"Unknown hypothesis id: {hypothesis_id}")
        if not name_stem.strip():
            raise InvalidTransitionError("Detector name stem is required")

        existing = next(
            (
                item
                for item in project.method_implementations
                if item.hypothesis_id == hypothesis.id and item.kind == "detector"
            ),
            None,
        )
        if existing is not None and existing.status in {"approved", "validated"}:
            self._reject_non_mock_deterministic_fallback(
                existing,
                implementation_kind="检测器",
            )
            project.record_event(
                actor="method_registry",
                action="implement_experiment_detector",
                summary=f"复用已有 {existing.status} 检测器实现 {existing.name}，未发起新的生成。",
                payload={"code_digest": existing.code_digest, "status": existing.status},
            )
            return self.repository.save(project)

        implementation = await self.runtime.implement_detector(
            project,
            hypothesis=hypothesis,
            name_stem=name_stem,
            reference_description=reference_description,
        )
        self._reject_non_mock_deterministic_fallback(
            implementation,
            implementation_kind="检测器",
        )
        project.method_implementations = [
            item
            for item in project.method_implementations
            if not (item.hypothesis_id == hypothesis.id and item.kind == "detector")
        ]
        project.method_implementations.append(implementation)

        validation = validate_detector_source(implementation.source_code)
        implementation.static_validation = StaticValidationReport(
            passed=validation.passed,
            issues=validation.issues,
        )
        if not validation.passed:
            implementation.status = "rejected"
            project.record_event(
                actor="code_safety_validator",
                action="implement_experiment_detector",
                summary=f"生成的检测器 {implementation.name} 未通过静态校验，已拒绝。",
                payload={"issues": validation.issues},
            )
            return self.repository.save(project)

        assembled = assemble_detector_file(implementation.source_code)
        digest = hashlib.sha256(assembled.encode("utf-8")).hexdigest()
        if digest != implementation.code_digest:
            implementation.status = "rejected"
            project.record_event(
                actor="code_safety_validator",
                action="implement_experiment_detector",
                summary=f"检测器 {implementation.name} 的注册摘要与源码不一致，已拒绝。",
                payload={"expected": implementation.code_digest, "actual": digest},
            )
            return self.repository.save(project)

        detector_path = self.artifact_root / "generated_methods" / digest / "detector.py"
        detector_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = detector_path.with_suffix(".py.tmp")
        temporary.write_text(assembled, encoding="utf-8")
        temporary.replace(detector_path)
        implementation.artifact_path = str(detector_path.resolve())
        project.artifacts.append(
            ArtifactRecord(
                kind="generated_detector",
                title=f"生成检测器 {implementation.name}",
                path=str(detector_path.resolve()),
                payload={
                    "name": implementation.name,
                    "code_digest": implementation.code_digest,
                    "hypothesis_id": hypothesis.id,
                    "reference_description": reference_description,
                },
                provenance=[self.runtime.name, "code_safety_validator"],
                verified=False,
            )
        )

        smoke = await run_detector_smoke(implementation, self.artifact_root)
        implementation.smoke_result = smoke
        if not smoke.passed:
            implementation.status = "rejected"
            project.record_event(
                actor="detector_smoke_runner",
                action="implement_experiment_detector",
                summary=f"检测器 {implementation.name} 冒烟测试未通过，已拒绝。",
                payload={"smoke_summary": smoke.summary},
            )
            return self.repository.save(project)

        implementation.status = "validated"
        project.record_event(
            actor=self.runtime.name,
            action="implement_experiment_detector",
            summary=(
                f"检测器 {implementation.name} 已生成并通过静态校验与冒烟测试，"
                "等待计划批准后注册执行。"
            ),
            payload={
                "name": implementation.name,
                "code_digest": implementation.code_digest,
                "name_stem": name_stem,
            },
        )
        return self.repository.save(project)

    def _reject_non_mock_deterministic_fallback(
        self,
        implementation: MethodImplementation,
        *,
        implementation_kind: str,
    ) -> None:
        if type(self.runtime) is MockScientistRuntime:
            return
        fallback_markers = [
            marker
            for marker in implementation.provenance
            if "fallback" in marker
        ]
        if any("deterministic-fallback" in marker for marker in fallback_markers):
            raise InvalidTransitionError(
                f"非 mock 运行时返回了 deterministic-fallback {implementation_kind}实现，"
                "已拒绝注册；请修复 AgentScope 调用后重试。"
                f"失败标记：{', '.join(fallback_markers)}"
            )

    def attach_evidence(
        self,
        project_id: str,
        *,
        evidence: list[EvidenceRecord],
        actor: str = "evidence_retrieval_tool",
    ) -> ResearchProject:
        project = self.repository.get(project_id)
        existing = {_evidence_key(item): item for item in project.evidence}
        for item in evidence:
            existing[_evidence_key(item)] = item
        project.evidence = list(existing.values())
        project.record_event(
            actor=actor,
            action="attach_evidence",
            summary=f"已附加 {len(evidence)} 条真实来源的文献元数据。",
            payload={
                "evidence_ids": [item.id for item in evidence],
                "verification_scopes": [item.verification_scope for item in evidence],
            },
        )
        return self.repository.save(project)

    def attach_dataset_audit(
        self,
        project_id: str,
        *,
        manifest: DatasetManifest,
        manifest_path: str,
    ) -> ResearchProject:
        project = self.repository.get(project_id)
        audit = DatasetAuditRecord(
            dataset=manifest.dataset,
            root=manifest.root,
            manifest_path=manifest_path,
            digest=manifest.digest,
            categories=manifest.categories,
            counts=manifest.counts,
            issue_count=len(manifest.issues),
            verified=manifest.is_valid,
        )
        project.dataset_audits = [
            item for item in project.dataset_audits if item.digest != audit.digest
        ]
        project.dataset_audits.append(audit)
        project.record_event(
            actor="dataset_auditor",
            action="attach_dataset_audit",
            summary=(
                f"MVTec AD 数据已完成结构、文件与掩码审计："
                f"{len(manifest.categories)} 个类别，{len(manifest.files)} 个文件。"
            ),
            payload={
                "audit_id": audit.id,
                "dataset_digest": audit.digest,
                "verified": audit.verified,
                "issue_count": audit.issue_count,
            },
        )
        return self.repository.save(project)

    def initialize_experiment_campaign(
        self,
        project_id: str,
        *,
        dataset: DatasetManifest,
        hypothesis_id: str,
        device: str = "cuda:0",
        detector: str = "anomalydino",
        max_rounds: int = 3,
        max_runs: int = 24,
        execution_mode: Literal["sequential", "parallel"] = "sequential",
        parallelism: int | None = None,
        selected_hypothesis_ids: list[str] | None = None,
    ) -> ResearchProject:
        project = self.repository.get(project_id)
        if project.stage != ResearchStage.EXPERIMENTS_QUEUED:
            raise InvalidTransitionError("Approve the preregistered plan before starting a loop")
        if project.experiment_campaign is not None:
            if project.experiment_campaign.status != "completed":
                raise InvalidTransitionError(
                    "The project already has an active experiment campaign"
                )
            project.experiment_campaign_history.append(
                project.experiment_campaign.model_copy(deep=True)
            )
            project.experiment_campaign = None
        if project.experiment_plan is None:
            raise InvalidTransitionError("The project has no current experiment plan")
        current_plan_id = project.experiment_plan.id
        replaceable_runs = [
            run
            for run in project.runs
            if run.plan_id == current_plan_id and run.round_id is None
        ]
        if any(
            run.status in {RunStatus.RUNNING, RunStatus.SUCCEEDED, RunStatus.FAILED}
            for run in replaceable_runs
        ):
            raise InvalidTransitionError(
                "Cannot replace the fixed feasibility queue after execution has started"
            )
        audit = next(
            (
                item
                for item in reversed(project.dataset_audits)
                if item.digest == dataset.digest and item.verified
            ),
            None,
        )
        if audit is None:
            raise InvalidTransitionError("Run and attach a verified dataset audit first")

        replaced_count = len(replaceable_runs)
        replaceable_ids = {run.id for run in replaceable_runs}
        historical_runs = [run for run in project.runs if run.id not in replaceable_ids]
        try:
            campaign, runs = self.experiment_planner.initialize(
                project,
                audit=audit,
                dataset=dataset,
                hypothesis_id=hypothesis_id,
                device=device,
                detector=detector,
                max_rounds=max_rounds,
                max_runs=max_runs,
                execution_mode=execution_mode,
                parallelism=parallelism,
                selected_hypothesis_ids=selected_hypothesis_ids,
            )
        except ValueError as exc:
            raise InvalidTransitionError(str(exc)) from exc
        project.runs = [*historical_runs, *runs]
        project.experiment_campaign = campaign
        project.status = ProjectStatus.WAITING_EXTERNAL
        project.next_action = campaign.next_action
        project.record_event(
            actor="adaptive_experiment_planner",
            action="initialize_experiment_campaign",
            summary=(
                "已建立创新点驱动的闭环实验队列；每个创新点对应一个 Round，"
                + (
                    "多个 Round 已预注册并行执行，每个 Round 固定三次自动迭代。"
                    if execution_mode == "parallel"
                    else "每个 Round 固定三次内部迭代，并在第 1 次迭代后接受一次用户指导。"
                )
            ),
            payload={
                "campaign_id": campaign.id,
                "hypothesis_id": campaign.hypothesis_id,
                "replaced_fixed_runs": replaced_count,
                "initial_runs": len(runs),
                "max_rounds": campaign.max_rounds,
                "max_runs": campaign.max_runs,
                "exhaustive_run_count": campaign.exhaustive_run_count,
                "execution_mode": campaign.execution_mode,
                "parallelism": campaign.parallelism,
                "selected_hypothesis_ids": campaign.selected_hypothesis_ids,
            },
        )
        return self.repository.save(project)

    async def auto_start_parallel_campaign(
        self,
        project_id: str,
        *,
        dataset: DatasetManifest,
        hypothesis_id: str,
        selected_hypothesis_ids: list[str] | None = None,
        device: str = "cuda:0",
        detector: str = "anomalydino",
        max_rounds: int = 20,
        max_runs: int = 240,
        parallelism: int | None = None,
    ) -> ResearchProject:
        """Automatically preregister and start a selected innovation portfolio."""

        project = self.repository.get(project_id)
        if project.stage == ResearchStage.HYPOTHESES_REVIEWED:
            project = await self.advance(project_id)
        if project.stage == ResearchStage.AWAITING_EXPERIMENT_APPROVAL:
            project = self.approve_experiment_plan(
                project_id,
                approved_by="automatic_preregistration_after_human_ranking",
            )
        if project.stage != ResearchStage.EXPERIMENTS_QUEUED:
            raise InvalidTransitionError(
                "Rank hypotheses and complete dataset audit before starting the parallel campaign"
            )
        if project.experiment_campaign is not None:
            if project.experiment_campaign.execution_mode == "parallel":
                return project
            if project.experiment_campaign.status != "completed":
                raise InvalidTransitionError(
                    "The project already has an active experiment campaign"
                )
        selected = selected_hypothesis_ids or [
            item.id
            for item in project.hypotheses
            if item.user_selected is not False
            and item.status in {HypothesisStatus.SHORTLISTED, HypothesisStatus.APPROVED}
        ]
        if not selected:
            raise InvalidTransitionError(
                "Select at least one innovation before starting experiments"
            )
        return self.initialize_experiment_campaign(
            project_id,
            dataset=dataset,
            hypothesis_id=hypothesis_id,
            device=device,
            detector=detector,
            max_rounds=max_rounds,
            max_runs=max_runs,
            execution_mode="parallel",
            parallelism=parallelism,
            selected_hypothesis_ids=selected,
        )

    async def select_next_experiment(
        self,
        project_id: str,
        *,
        user_guidance: str | None = None,
    ) -> tuple[ExperimentRun, ExperimentGuidanceDecision]:
        """Interpret human advice and select only from the frozen current queue."""

        project = self.repository.get(project_id)
        campaign = project.experiment_campaign
        if campaign is None or campaign.status != "active":
            raise InvalidTransitionError("The experiment campaign is not accepting runs")
        guidance = (user_guidance or "").strip()
        human_guidance = bool(guidance)
        if not guidance:
            guidance = "系统按本 Round 已预注册的迭代顺序自动执行。"
        candidates = self.experiment_planner.queued_runs(project)
        if not candidates:
            raise ResultsRequiredError("No queued experiment is available in the current round")

        decision = await self.runtime.interpret_experiment_guidance(
            project,
            guidance=guidance,
            candidate_runs=candidates,
        )
        candidate_by_id = {run.id: run for run in candidates}
        selected = candidate_by_id.get(decision.selected_run_id)
        if selected is None:
            selected = candidates[0]
            decision = decision.model_copy(
                deep=True,
                update={
                    "selected_run_id": selected.id,
                    "disposition": "partially_applied",
                    "rationale": (
                        decision.rationale
                        + " 动作校验器拒绝了队列外选择，并回退到当前最高优先级任务。"
                    ),
                },
            )

        if human_guidance:
            record = UserGuidanceRecord(
                scope="experiment_execution",
                target_action="execute_next_experiment",
                text=guidance,
                research_cycle=project.research_cycle,
                round_id=campaign.rounds[-1].id,
                advisor=decision.advisor,
                interpretation=decision.interpretation,
                disposition=decision.disposition,
                rationale=decision.rationale,
                selected_run_id=selected.id,
                affected_ids=[selected.id],
                protected_constraints=decision.protected_constraints,
            )
            project.guidance_records.append(record)
            project.record_event(
                actor="human_guidance_agent",
                action="interpret_experiment_guidance",
                summary=(
                    f"用户在真实运行前提交指导；AI Scientist 判定为 "
                    f"{decision.disposition}，选择 {selected.id}。"
                ),
                payload={
                    "guidance_id": record.id,
                    "guidance": guidance,
                    "decision": decision.model_dump(mode="json"),
                    "candidate_run_ids": [run.id for run in candidates],
                },
            )
        self.repository.save(project)
        return selected, decision

    async def review_experiment_round(
        self,
        project_id: str,
        *,
        user_guidance: str | None = None,
        round_id: str | None = None,
    ) -> ResearchProject:
        project = self.repository.get(project_id)
        campaign = project.experiment_campaign
        if campaign is None:
            raise InvalidTransitionError("The project has no experiment campaign")
        if campaign.execution_mode == "parallel" and campaign.status == "awaiting_feedback":
            # Parallel Rounds are reviewed together after their midpoint gates
            # have all been completed; there is no second human approval gate.
            return await self.complete_parallel_campaign(project_id)
        if campaign.status not in {"awaiting_guidance", "awaiting_feedback"}:
            raise ResultsRequiredError("The current experiment round is not ready for feedback")

        guidance_rounds = [
            item for item in campaign.rounds
            if item.status == "awaiting_guidance"
        ]
        if campaign.execution_mode == "parallel":
            if round_id is not None:
                current = next(
                    (item for item in guidance_rounds if item.id == round_id),
                    None,
                )
                if current is None:
                    raise InvalidTransitionError(
                        "The requested parallel Round is not waiting for guidance"
                    )
            else:
                current = guidance_rounds[0] if guidance_rounds else None
            if current is None:
                raise ResultsRequiredError("No parallel Round is waiting for guidance")
        else:
            current = campaign.rounds[-1]
        guidance = (user_guidance or "").strip()
        if campaign.status == "awaiting_guidance" and not guidance:
            raise InvalidTransitionError("每个 Round 中途必须提交一次用户指导")
        summary = self.experiment_planner.summarize_round(
            project,
            round_id=current.id,
        )
        allowed_cells = (
            self.experiment_planner.remaining_round_cells(
                project,
                round_id=current.id,
            )
            if campaign.execution_mode == "parallel"
            else self.experiment_planner.allowed_next_cells(
                project,
                hypothesis_id=current.hypothesis_id,
            )
        )
        advisor_project = project
        if campaign.execution_mode == "parallel":
            # The campaign keeps a stable primary hypothesis for summaries,
            # while the advisor must receive the exact innovation represented
            # by the Round whose guidance form was submitted.
            advisor_project = project.model_copy(deep=True)
            advisor_campaign = advisor_project.experiment_campaign
            if advisor_campaign is not None:
                advisor_campaign.hypothesis_id = current.hypothesis_id
                advisor_campaign.treatment = current.treatment
                advisor_campaign.control = current.control
                advisor_campaign.metric = current.metric
        try:
            proposal = await self.runtime.recommend_next_experiments(
                advisor_project,
                round_summary=summary,
                allowed_cells=allowed_cells,
                user_guidance=guidance or None,
            )
        except TypeError as exc:
            # Keep compatibility with test/custom runtimes that implement the
            # pre-guidance advisor contract.  Built-in Qwen and mock runtimes
            # accept the explicit guidance field above; only an unexpected
            # keyword is retried without it.
            if "user_guidance" not in str(exc):
                raise
            proposal = await self.runtime.recommend_next_experiments(
                advisor_project,
                round_summary=summary,
                allowed_cells=allowed_cells,
            )
        if campaign.status == "awaiting_guidance":
            try:
                new_runs = self.experiment_planner.apply_midpoint_guidance(
                    project,
                    proposal=proposal,
                    summary=summary,
                    round_id=current.id,
                )
            except ValueError as exc:
                raise InvalidTransitionError(str(exc)) from exc
            record = UserGuidanceRecord(
                scope="round_iteration",
                target_action="continue_round_iterations",
                text=guidance,
                research_cycle=project.research_cycle,
                round_id=current.id,
                advisor=proposal.advisor,
                interpretation=(
                    "该建议用于调整本 Round 第 2、3 次迭代的类别、K 与随机种子优先级。"
                ),
                disposition="applied",
                rationale=proposal.rationale,
                affected_ids=[run.id for run in new_runs],
                protected_constraints=[
                    "创新点与分析契约不变",
                    "固定三次迭代",
                    "测试标签不参与支持集选择",
                ],
            )
            project.guidance_records.append(record)
            existing_run_ids = {run.id for run in project.runs}
            project.runs.extend(
                run for run in new_runs if run.id not in existing_run_ids
            )
            project.status = ProjectStatus.WAITING_EXTERNAL
            project.next_action = campaign.next_action
            project.record_event(
                actor="human_guidance_agent",
                action="continue_round_iterations",
                summary=(
                    f"用户已在 Round {current.index} 中途提交唯一一次指导；"
                    "系统已据此排定第 2、3 次自动迭代。"
                ),
                payload={
                    "guidance_id": record.id,
                    "guidance": guidance,
                    "new_run_ids": [run.id for run in new_runs],
                },
            )
            return self.repository.save(project)

        try:
            new_runs = self.experiment_planner.apply_feedback(
                project,
                proposal=proposal,
                summary=summary,
            )
        except ValueError as exc:
            raise InvalidTransitionError(str(exc)) from exc
        project.runs.extend(new_runs)
        project.status = ProjectStatus.WAITING_EXTERNAL
        project.next_action = campaign.next_action
        project.record_event(
            actor=proposal.advisor,
            action="review_experiment_round",
            summary=(
                f"Round {summary['round_index']} 的三次迭代已汇总；"
                f"决策={proposal.decision}，下一创新点新增 {len(new_runs)} 次初始运行。"
            ),
            payload={
                "round_summary": summary,
                "feedback": proposal.model_dump(mode="json"),
                "new_run_ids": [run.id for run in new_runs],
            },
        )
        return self.repository.save(project)

    def mark_run_running(self, project_id: str, *, run_id: str) -> ResearchProject:
        project = self.repository.get(project_id)
        if project.stage != ResearchStage.EXPERIMENTS_QUEUED:
            raise InvalidTransitionError("Project is not accepting experiment execution")
        run = next((item for item in project.runs if item.id == run_id), None)
        if run is None:
            raise KeyError(f"Unknown run id: {run_id}")
        if run.status != RunStatus.QUEUED:
            raise InvalidTransitionError(f"Run {run_id} is not queued")
        run.status = RunStatus.RUNNING
        run.started_at = utc_now()
        self.experiment_planner.refresh_after_run(project)
        project.record_event(
            actor="experiment_executor",
            action="start_run",
            summary=f"实验 {run.id} 已开始执行。",
            payload={"run_id": run.id},
        )
        return self.repository.save(project)

    def record_run_result(
        self,
        project_id: str,
        *,
        run_id: str,
        metrics: dict[str, float],
        artifact_paths: list[str] | None = None,
        code_revision: str | None = None,
        environment_digest: str | None = None,
        success: bool = True,
        verified: bool = True,
        result_source: Literal[
            "real_executor", "external_import", "synthetic_test"
        ] = "external_import",
        preparation_path: str | None = None,
        execution_record_path: str | None = None,
        duration_seconds: float | None = None,
        error: str | None = None,
    ) -> ResearchProject:
        project = self.repository.get(project_id)
        if project.stage != ResearchStage.EXPERIMENTS_QUEUED:
            raise InvalidTransitionError("Project is not accepting experiment results")

        run = next((item for item in project.runs if item.id == run_id), None)
        if run is None:
            raise KeyError(f"Unknown run id: {run_id}")
        run.metrics = metrics
        run.artifact_paths = artifact_paths or []
        run.code_revision = code_revision
        run.environment_digest = environment_digest
        run.status = RunStatus.SUCCEEDED if success else RunStatus.FAILED
        run.verified = verified and success
        run.result_source = result_source
        run.preparation_path = preparation_path
        run.execution_record_path = execution_record_path
        run.duration_seconds = duration_seconds
        run.error = error
        run.finished_at = utc_now()
        self.experiment_planner.refresh_after_run(project)
        if project.experiment_campaign is not None:
            project.next_action = project.experiment_campaign.next_action
        project.record_event(
            actor="experiment_executor",
            action="record_run_result",
            summary=f"实验 {run.id} 已返回 {'成功' if success else '失败'} 状态。",
            payload={"run_id": run.id, "verified": run.verified},
        )
        return self.repository.save(project)

    def finalize_results(self, project_id: str) -> ResearchProject:
        project = self.repository.get(project_id)
        if project.stage != ResearchStage.EXPERIMENTS_QUEUED:
            raise InvalidTransitionError("Project is not waiting for experiment results")
        if (
            project.experiment_campaign is not None
            and project.experiment_campaign.status != "completed"
        ):
            raise ResultsRequiredError(
                "The adaptive experiment campaign must finish its feedback rounds first"
            )
        unfinished = [
            run.id
            for run in project.runs
            if run.status in {RunStatus.PLANNED, RunStatus.QUEUED, RunStatus.RUNNING}
        ]
        if unfinished:
            raise ResultsRequiredError(
                f"{len(unfinished)} experiment runs have not reached a terminal state"
            )
        if not any(run.status == RunStatus.SUCCEEDED and run.verified for run in project.runs):
            raise ResultsRequiredError("At least one verified successful run is required")

        self._move(
            project,
            stage=ResearchStage.RESULTS_READY,
            status=ProjectStatus.ACTIVE,
            next_action="analyze_verified_results",
            actor="experiment_executor",
            summary="实验批次已完成，真实结果已锁定，进入统计分析。",
            payload={
                "successful": sum(run.status == RunStatus.SUCCEEDED for run in project.runs),
                "failed": sum(run.status == RunStatus.FAILED for run in project.runs),
            },
        )
        return self.repository.save(project)

    @staticmethod
    def _move(
        project: ResearchProject,
        *,
        stage: ResearchStage,
        status: ProjectStatus,
        next_action: str,
        actor: str,
        summary: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        project.stage = stage
        project.status = status
        project.next_action = next_action
        project.record_event(
            actor=actor,
            action=stage.value,
            summary=summary,
            payload=payload,
        )

    @staticmethod
    def _build_feasibility_runs(project: ResearchProject) -> list[ExperimentRun]:
        plan = project.experiment_plan
        if plan is None:
            return []

        hypothesis_ids = plan.hypothesis_ids
        datasets = plan.datasets[:1]
        categories = plan.categories[:3]
        detectors = plan.detectors
        shots = plan.shots[:2]
        seeds = plan.seeds[:3]
        max_runs = min(project.spec.budget.max_experiments, 240)
        runs: list[ExperimentRun] = []

        protocol_strategies = {
            "strict_k_shot": ["random"],
            "pool_compression_m30": ["random", "k_center"],
        }
        for protocol in plan.protocols:
            # Interleave hypotheses so a bounded legacy feasibility queue still
            # contains at least one evidence item for every approved innovation.
            combinations = product(datasets, categories, detectors, shots, seeds, hypothesis_ids)
            for dataset, category, detector, shot, seed, hypothesis_id in combinations:
                hypothesis = next(
                    (item for item in project.hypotheses if item.id == hypothesis_id),
                    None,
                )
                contract = hypothesis.analysis_contract if hypothesis is not None else None
                strategies = (
                    [contract.control, contract.treatment]
                    if contract is not None
                    else protocol_strategies.get(protocol, plan.selection_strategies[:2])
                )
                for strategy in strategies:
                    if len(runs) >= max_runs:
                        return runs
                    runs.append(
                        ExperimentRun(
                            plan_id=plan.id,
                            hypothesis_id=hypothesis_id,
                            protocol=protocol,
                            dataset=dataset,
                            category=category,
                            detector=detector,
                            selection_strategy=strategy,
                            shots=shot,
                            seed=seed,
                            status=RunStatus.QUEUED,
                        )
                    )
        return runs

    @staticmethod
    def _should_revise(project: ResearchProject) -> bool:
        if project.research_cycle >= project.spec.constraints.max_research_cycles:
            return False
        verdicts = {
            finding.claim_verdict
            for finding in project.findings
            if finding.claim_verdict != "not_tested"
        }
        return "supported" not in verdicts and bool(
            verdicts & {"rejected", "inconclusive"}
        )

    @staticmethod
    def _enforce_method_implementation_gate(project: ResearchProject) -> None:
        """Block approval of plans referencing custom strategies without validated code."""

        if project.experiment_plan is None:
            return
        plan = project.experiment_plan
        strategy_names: set[tuple[str, str]] = set()
        for hypothesis_id in plan.hypothesis_ids:
            hypothesis = next(
                (item for item in project.hypotheses if item.id == hypothesis_id),
                None,
            )
            if hypothesis is None or hypothesis.analysis_contract is None:
                continue
            contract = hypothesis.analysis_contract
            planned_contract = plan.hypothesis_contracts.get(hypothesis_id)
            if planned_contract is not None and planned_contract != contract:
                raise InvalidTransitionError(
                    f"假设 {hypothesis_id} 的分析契约已偏离预注册计划，请重新生成实验计划"
                )
            if contract.kind not in {"selection_main_effect", "query_adaptation"}:
                continue
            for name in (contract.treatment, contract.control):
                if name not in BUILTIN_STRATEGIES:
                    strategy_names.add((hypothesis_id, name))
        approved_digests: dict[str, str] = {}
        for hypothesis_id in plan.hypothesis_ids:
            hypothesis = next(
                (item for item in project.hypotheses if item.id == hypothesis_id),
                None,
            )
            if hypothesis is None or hypothesis.analysis_contract is None:
                continue
            contract = hypothesis.analysis_contract
            custom_names = [
                name
                for name in (contract.treatment, contract.control)
                if name not in BUILTIN_STRATEGIES
            ]
            if len(custom_names) != 2:
                continue
            implementations = [
                next(
                    (
                        item
                        for item in project.method_implementations
                        if item.kind == "selection_strategy"
                        and item.name == name
                        and item.hypothesis_id == hypothesis_id
                    ),
                    None,
                )
                for name in custom_names
            ]
            if (
                all(item is not None for item in implementations)
                and implementations[0].code_digest == implementations[1].code_digest
            ):
                raise InvalidTransitionError(
                    f"假设 {hypothesis_id} 的 treatment/control 使用了相同策略实现，不能批准"
                )
        for hypothesis_id, name in sorted(strategy_names):
            implementation = next(
                (
                    item
                    for item in project.method_implementations
                    if item.kind == "selection_strategy"
                    and item.name == name
                    and item.hypothesis_id == hypothesis_id
                ),
                None,
            )
            if implementation is None:
                raise InvalidTransitionError(
                    f"策略 {name} 没有已注册实现；请先调用实验方法生成端点"
                )
            static_ok = implementation.static_validation.passed
            smoke_ok = (
                implementation.smoke_result is not None
                and implementation.smoke_result.passed
            )
            if not (static_ok and smoke_ok):
                raise InvalidTransitionError(
                    f"策略 {name} 未通过静态校验或冒烟测试，不能批准"
                )
            preregistered = plan.method_implementation_digests.get(name)
            if preregistered is not None and preregistered != implementation.code_digest:
                raise InvalidTransitionError(
                    f"策略 {name} 在预注册后发生变化，请重新生成并更新预注册摘要"
                )
            implementation.status = "approved"
            approved_digests[name] = implementation.code_digest
        for name in plan.detectors:
            if name.casefold() in BUILTIN_DETECTORS:
                continue
            implementation = next(
                (
                    item
                    for item in project.method_implementations
                    if item.kind == "detector" and item.name == name
                ),
                None,
            )
            if implementation is None:
                raise InvalidTransitionError(
                    f"检测器 {name} 没有已注册实现；请先调用检测器生成端点"
                )
            static_ok = implementation.static_validation.passed
            smoke_ok = (
                implementation.smoke_result is not None
                and implementation.smoke_result.passed
            )
            if not (static_ok and smoke_ok):
                raise InvalidTransitionError(
                    f"检测器 {name} 未通过静态校验或冒烟测试，不能批准"
                )
            preregistered = plan.method_implementation_digests.get(name)
            if preregistered is not None and preregistered != implementation.code_digest:
                raise InvalidTransitionError(
                    f"检测器 {name} 在预注册后发生变化，请重新生成并更新预注册摘要"
                )
            implementation.status = "approved"
            approved_digests[name] = implementation.code_digest
        if approved_digests:
            project.record_event(
                actor="human_and_method_registry",
                action="approve_experiment_plan",
                summary="计划批准已把通过校验的自定义实现（策略或检测器）注册为可执行。",
                payload={"approved_method_digests": approved_digests},
            )

    @staticmethod
    def _ensure_executable_hypotheses(project: ResearchProject) -> None:
        """Normalize every supported innovation without inventing a replacement claim."""

        for hypothesis in project.hypotheses:
            contract = hypothesis.analysis_contract
            if contract is None or contract.kind != "query_adaptation":
                continue
            control = _strategy_alias(contract.control)
            if control == "random" and contract.control != "random":
                original = contract.control
                hypothesis.analysis_contract = contract.model_copy(
                    update={"control": "random"}
                )
                project.record_event(
                    actor="research_brief_operationalizer",
                    action="operationalize_hypothesis",
                    summary=(
                        "已将查询自适应假设的对照条件归一为 random 基线；"
                        "treatment 保持自定义策略名，等待实现生成与注册。"
                    ),
                    payload={"hypothesis_id": hypothesis.id, "original_control": original},
                )

        for hypothesis in project.hypotheses:
            contract = hypothesis.analysis_contract
            if contract is None or contract.kind != "selection_main_effect":
                continue
            treatment = _strategy_alias(contract.treatment)
            control = _strategy_alias(contract.control)
            if treatment == "k_center" and control == "random":
                if (contract.treatment, contract.control) != (treatment, control):
                    original = {
                        "treatment": contract.treatment,
                        "control": contract.control,
                    }
                    hypothesis.analysis_contract = contract.model_copy(
                        update={"treatment": treatment, "control": control}
                    )
                    hypothesis.status = HypothesisStatus.SHORTLISTED
                    project.record_event(
                        actor="research_brief_operationalizer",
                        action="operationalize_hypothesis",
                        summary=(
                            "已将参考集多样性假设映射为可执行的 k-center 对 random 成对对照；"
                            "科学主张、零假设和证伪条件保持不变。"
                        ),
                        payload={"hypothesis_id": hypothesis.id, "original": original},
                    )
                continue

        executable = [
            hypothesis
            for hypothesis in project.hypotheses
            if hypothesis.execution_readiness == "executable"
            or any(
                implementation.hypothesis_id == hypothesis.id
                and implementation.kind == "selection_strategy"
                and implementation.status in {"validated", "approved"}
                for implementation in project.method_implementations
            )
        ]
        if not executable:
            raise InvalidTransitionError(
                "当前创新点都需要先实现方法适配器；系统不会替换成无关假设。"
            )


def _evidence_key(item: EvidenceRecord) -> str:
    if item.doi:
        return f"doi:{item.doi.casefold()}"
    if item.arxiv_id:
        return f"arxiv:{item.arxiv_id.casefold()}"
    return f"title:{' '.join(item.title.casefold().split())}"


def _strategy_alias(value: str) -> str | None:
    normalized = " ".join(value.casefold().replace("_", " ").replace("-", " ").split())
    if (
        normalized == "random"
        or "random" in normalized
        or "随机" in normalized
        or normalized == "none"
        or "no adaptation" in normalized
        or "无适配" in normalized
    ):
        return "random"
    baseline_markers = (
        "static single sample",
        "single sample prototype",
        "static prototype",
        "fixed prototype",
        "no refinement",
        "without refinement",
        "静态单样本",
        "单样本原型",
        "静态原型",
        "固定原型",
        "无校准",
        "无修正",
    )
    if any(marker in normalized for marker in baseline_markers):
        return "random"
    diversity_markers = (
        "k center",
        "diversity",
        "representative",
        "coverage",
        "farthest",
        "多样性",
        "代表性",
        "覆盖",
    )
    if any(marker in normalized for marker in diversity_markers):
        return "k_center"
    return None
