from __future__ import annotations

from itertools import product
from typing import Any, Literal

from fsad_scientist.agents.contracts import ScientistRuntime
from fsad_scientist.datasets.models import DatasetManifest
from fsad_scientist.domain.enums import (
    HypothesisStatus,
    ProjectStatus,
    ResearchStage,
    RunStatus,
)
from fsad_scientist.domain.models import (
    AnalysisContract,
    DatasetAuditRecord,
    EvidenceRecord,
    ExperimentGuidanceDecision,
    ExperimentRun,
    Hypothesis,
    ProjectSpec,
    ResearchProject,
    UserGuidanceRecord,
    utc_now,
)
from fsad_scientist.experiments.loop import AdaptiveExperimentPlanner
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
    ) -> None:
        self.repository = repository
        self.runtime = runtime
        self.experiment_planner = AdaptiveExperimentPlanner()

    def create_project(self, spec: ProjectSpec) -> ResearchProject:
        project = ResearchProject(spec=spec)
        project.record_event(
            actor="human",
            action="create_project",
            summary="用户提供研究领域、数据、现实约束和计算预算。",
        )
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
            self._ensure_executable_core_hypothesis(project)
            if project.experiment_plan is not None:
                project.experiment_plan_history.append(
                    project.experiment_plan.model_copy(deep=True)
                )
            project.experiment_plan = await self.runtime.design_experiments(project)
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
        device: str = "cuda:0",
        detector: str = "anomalydino",
        max_rounds: int = 3,
        max_runs: int = 24,
    ) -> ResearchProject:
        project = self.repository.get(project_id)
        if project.stage != ResearchStage.EXPERIMENTS_QUEUED:
            raise InvalidTransitionError("Approve the preregistered plan before starting a loop")
        if project.experiment_campaign is not None:
            raise InvalidTransitionError("The project already has an experiment campaign")
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
                device=device,
                detector=detector,
                max_rounds=max_rounds,
                max_runs=max_runs,
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
                "已用渐进式实验树替换一次性穷举队列；首轮仅冻结最小成对可行性实验，"
                "后续单元将由真实结果驱动。"
            ),
            payload={
                "campaign_id": campaign.id,
                "hypothesis_id": campaign.hypothesis_id,
                "replaced_fixed_runs": replaced_count,
                "initial_runs": len(runs),
                "max_rounds": campaign.max_rounds,
                "max_runs": campaign.max_runs,
                "exhaustive_run_count": campaign.exhaustive_run_count,
            },
        )
        return self.repository.save(project)

    async def select_next_experiment(
        self,
        project_id: str,
        *,
        user_guidance: str,
    ) -> tuple[ExperimentRun, ExperimentGuidanceDecision]:
        """Interpret human advice and select only from the frozen current queue."""

        project = self.repository.get(project_id)
        campaign = project.experiment_campaign
        if campaign is None or campaign.status != "active":
            raise InvalidTransitionError("The experiment campaign is not accepting runs")
        guidance = user_guidance.strip()
        if not guidance:
            raise InvalidTransitionError("Guidance is required before a real experiment")
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

    async def review_experiment_round(self, project_id: str) -> ResearchProject:
        project = self.repository.get(project_id)
        campaign = project.experiment_campaign
        if campaign is None:
            raise InvalidTransitionError("The project has no experiment campaign")
        if campaign.status != "awaiting_feedback":
            raise ResultsRequiredError("The current experiment round is not ready for feedback")

        summary = self.experiment_planner.summarize_current_round(project)
        allowed_cells = self.experiment_planner.allowed_next_cells(project)
        proposal = await self.runtime.recommend_next_experiments(
            project,
            round_summary=summary,
            allowed_cells=allowed_cells,
        )
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
                f"第 {summary['round_index']} 轮真实结果已反馈到规划器；"
                f"决策={proposal.decision}，下一轮新增 {len(new_runs)} 次运行。"
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

        hypothesis_ids = plan.hypothesis_ids[:1]
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
            strategies = protocol_strategies.get(protocol, plan.selection_strategies[:2])
            combinations = product(
                hypothesis_ids,
                datasets,
                categories,
                detectors,
                shots,
                seeds,
                strategies,
            )
            for hypothesis_id, dataset, category, detector, shot, seed, strategy in combinations:
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
    def _ensure_executable_core_hypothesis(project: ResearchProject) -> None:
        """Operationalize the team research brief without changing its scientific claim."""

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
                return

        selected_gap = next(
            (gap for gap in project.gaps if gap.status == "selected"),
            project.gaps[0] if project.gaps else None,
        )
        if selected_gap is None:
            raise InvalidTransitionError(
                "A research gap is required to operationalize the experiment hypothesis"
            )
        core = Hypothesis(
            gap_id=selected_gap.id,
            title="正常参考样本的代表性是否比数量更重要",
            claim=(
                "在候选正常样本池固定且 K≤8 时，DINOv2 特征空间的 k-center 覆盖选样"
                "相较随机选样，将提高 AnomalyDINO 的 Image AUROC，并降低跨支持集重采样波动。"
            ),
            null_hypothesis=(
                "在相同类别、K、候选池和随机种子下，k-center 与 random 的 Image AUROC"
                "成对差异为零，且稳定性没有改善。"
            ),
            rationale=(
                "团队阶段性调研将正常样本代表性确定为主问题；该对照可直接在 MVTec AD"
                "正常训练图像上实施，不使用测试异常参与选择。"
            ),
            independent_variables=["支持集选择策略", "K", "类别", "随机种子"],
            dependent_variables=[
                "Image AUROC",
                "Pixel AUROC",
                "跨支持集标准差",
                "特征覆盖半径",
            ],
            predicted_direction=(
                "k-center 的平均成对效应为正，且覆盖半径与性能损失或波动正相关。"
            ),
            falsification_conditions=[
                "达到预注册最小配对数后，Image AUROC 成对效应置信区间仍包含零且效应可忽略",
                "收益无法跨至少三个 MVTec 类别复现",
                "覆盖半径改善但检测性能与稳定性不随之改善",
            ],
            evidence_ids=[item.id for item in project.evidence],
            closest_prior_work=[
                item.title
                for item in project.evidence
                if any(name in item.title.casefold() for name in ("patchcore", "anomalydino"))
            ],
            analysis_contract=AnalysisContract(
                kind="selection_main_effect",
                metric="image_auroc",
                treatment="k_center",
                control="random",
                minimum_pairs=6,
            ),
            status=HypothesisStatus.SHORTLISTED,
        )
        project.hypotheses.append(core)
        project.record_event(
            actor="research_brief_operationalizer",
            action="register_core_experiment_hypothesis",
            summary=(
                "现有候选中没有可由当前真实工具链直接检验的支持集选择主效应；"
                "系统已把团队调研确定的核心问题注册为可证伪实验假设。"
            ),
            payload={"hypothesis_id": core.id},
        )


def _evidence_key(item: EvidenceRecord) -> str:
    if item.doi:
        return f"doi:{item.doi.casefold()}"
    if item.arxiv_id:
        return f"arxiv:{item.arxiv_id.casefold()}"
    return f"title:{' '.join(item.title.casefold().split())}"


def _strategy_alias(value: str) -> str | None:
    normalized = " ".join(value.casefold().replace("_", " ").replace("-", " ").split())
    if normalized == "random" or "random" in normalized or "随机" in normalized:
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
