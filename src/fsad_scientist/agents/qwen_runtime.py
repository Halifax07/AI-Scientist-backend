from __future__ import annotations

from typing import Any

from fsad_scientist.agents.agentscope_client import AgentScopeJsonClient
from fsad_scientist.agents.mock_runtime import MockScientistRuntime
from fsad_scientist.domain.enums import EvidenceStatus, HypothesisStatus
from fsad_scientist.domain.models import (
    ArtifactRecord,
    ExperimentCell,
    ExperimentFeedbackProposal,
    ExperimentGuidanceDecision,
    ExperimentRun,
    Hypothesis,
    HypothesisScore,
    ResearchGap,
    ResearchProject,
    new_id,
)


class QwenScientistRuntime(MockScientistRuntime):
    """Qwen-backed cognitive stages with deterministic scientific safeguards.

    Literature retrieval, experiment execution and statistics remain tool-boundary
    operations. During the first scaffold phase, those operations use the parent
    implementation and keep their outputs unverified.
    """

    name = "qwen-agentscope-runtime"

    def __init__(
        self,
        *,
        model: str = "qwen3.7-plus",
        api_key: str | None = None,
    ) -> None:
        self.client = AgentScopeJsonClient(model=model, api_key=api_key)

    async def formalize_scope(self, project: ResearchProject) -> ArtifactRecord:
        response = await self.client.complete(
            role_name="Supervisor",
            system_prompt=(
                "你是自主科研项目经理。用户只提供研究领域、数据、现实约束和预算。"
                "把它转化为结构化研究范围，但不要替用户预设最终创新结论。"
                "除论文标题和标准技术名词外，所有自然语言字段使用简体中文。"
            ),
            payload={
                "project_spec": project.spec.model_dump(mode="json"),
                "required_keys": [
                    "problem_statement",
                    "independent_variables",
                    "dependent_variables",
                    "control_variables",
                    "integrity_rules",
                ],
            },
        )
        return ArtifactRecord(
            kind="research_scope",
            title="Qwen 生成的结构化研究范围",
            payload=response,
            provenance=["user_scope", self.name],
            verified=False,
        )

    async def discover_gaps(self, project: ResearchProject) -> list[ResearchGap]:
        response = await self.client.complete(
            role_name="GapDiscoveryAgent",
            system_prompt=(
                "你负责从已有证据候选、工业数据约束和方法差异中发现研究空白。"
                "提出 3 至 6 个互不重复的空白。不得把未经校验的文献候选视为事实。"
                "除论文标题和标准技术名词外，所有自然语言字段使用简体中文。"
            ),
            payload={
                "scope": project.spec.model_dump(mode="json"),
                "evidence_candidates": [
                    item.model_dump(mode="json") for item in project.evidence
                ],
                "output_schema": {
                    "gaps": [
                        {
                            "title": "string",
                            "description": "string",
                            "why_unresolved": "string",
                            "evidence_ids": ["evidence_id"],
                            "expected_scientific_value": "0..1",
                            "estimated_cost": "0..1",
                            "status": "candidate|selected|rejected",
                        }
                    ]
                },
            },
        )
        return [ResearchGap.model_validate(item) for item in response.get("gaps", [])]

    async def propose_hypotheses(self, project: ResearchProject) -> list[Hypothesis]:
        response = await self.client.complete(
            role_name="HypothesisAgent",
            system_prompt=(
                "你负责把研究空白转化为可证伪科学假设。每个假设必须包含零假设、"
                "变量、预测方向和明确的证伪条件；从不同机制提出 3 至 6 个候选，"
                "不要写成模糊的工程目标。除论文标题和标准技术名词外，"
                "所有自然语言字段使用简体中文。"
            ),
            payload={
                "gaps": [item.model_dump(mode="json") for item in project.gaps],
                "evidence_candidates": [
                    item.model_dump(mode="json") for item in project.evidence
                ],
                "required_fields": [
                    "gap_id",
                    "title",
                    "claim",
                    "null_hypothesis",
                    "rationale",
                    "independent_variables",
                    "dependent_variables",
                    "predicted_direction",
                    "falsification_conditions",
                    "evidence_ids",
                    "closest_prior_work",
                    "analysis_contract",
                ],
                "strict_output_schema": {
                    "hypotheses": [
                        {
                            "gap_id": "existing gap id",
                            "title": "string",
                            "claim": "string",
                            "null_hypothesis": "string",
                            "rationale": "string",
                            "independent_variables": ["string"],
                            "dependent_variables": ["string"],
                            "predicted_direction": "string",
                            "falsification_conditions": ["string"],
                            "evidence_ids": ["existing evidence id"],
                            "closest_prior_work": ["string"],
                            "analysis_contract": {
                                "kind": (
                                    "selection_main_effect|detector_interaction|"
                                    "query_adaptation"
                                ),
                                "metric": "string",
                                "treatment": "string",
                                "control": "string",
                                "alpha": 0.05,
                                "minimum_pairs": 6,
                            },
                        }
                    ]
                },
                "return": {"hypotheses": "array"},
            },
        )
        valid_gap_ids = {item.id for item in project.gaps}
        valid_evidence_ids = {item.id for item in project.evidence}
        hypotheses: list[Hypothesis] = []
        for item in response.get("hypotheses", []):
            if not isinstance(item, dict):
                continue
            normalized = _normalize_hypothesis_payload(item)
            if normalized.get("gap_id") not in valid_gap_ids:
                continue
            normalized["evidence_ids"] = [
                evidence_id
                for evidence_id in normalized["evidence_ids"]
                if evidence_id in valid_evidence_ids
            ]
            hypotheses.append(Hypothesis.model_validate(normalized))
        if not hypotheses:
            raise ValueError("Qwen returned no schema-valid hypotheses")
        return hypotheses

    async def review_hypotheses(self, project: ResearchProject) -> list[Hypothesis]:
        claim_verified = sum(
            item.status == EvidenceStatus.VERIFIED
            and item.verification_scope == "claim"
            for item in project.evidence
        )
        maximum_evidence_strength = 0.85 if claim_verified else 0.5
        response = await self.client.complete(
            role_name="SkepticMetaReviewer",
            system_prompt=(
                "你是严格的反方审稿人。进行两两比较，优先选择新颖、可证伪、"
                "在预算内可验证且有科学价值的假设。证据尚未校验时，"
                "evidence_strength 不得超过输入给出的上限。最多 shortlist 两个。"
                "所有评审性自然语言内容使用简体中文。"
            ),
            payload={
                "budget": project.spec.budget.model_dump(mode="json"),
                "claim_verified_evidence_count": claim_verified,
                "maximum_evidence_strength": maximum_evidence_strength,
                "hypotheses": [
                    item.model_dump(mode="json") for item in project.hypotheses
                ],
                "return": {
                    "reviews": [
                        {
                            "id": "hypothesis_id",
                            "novelty": "0..1",
                            "falsifiability": "0..1",
                            "feasibility": "0..1",
                            "scientific_value": "0..1",
                            "evidence_strength": "0..0.5",
                            "elo": "number",
                            "status": "shortlisted|candidate",
                        }
                    ]
                },
            },
        )
        reviews = {item["id"]: item for item in response.get("reviews", [])}
        result: list[Hypothesis] = []
        shortlist_count = 0
        for hypothesis in project.hypotheses:
            updated = hypothesis.model_copy(deep=True)
            review: dict[str, Any] | None = reviews.get(hypothesis.id)
            if review:
                updated.score = HypothesisScore(
                    novelty=review["novelty"],
                    falsifiability=review["falsifiability"],
                    feasibility=review["feasibility"],
                    scientific_value=review["scientific_value"],
                    evidence_strength=min(
                        review["evidence_strength"],
                        maximum_evidence_strength,
                    ),
                    elo=review["elo"],
                )
                if review.get("status") == "shortlisted" and shortlist_count < 2:
                    updated.status = HypothesisStatus.SHORTLISTED
                    shortlist_count += 1
                else:
                    updated.status = HypothesisStatus.CANDIDATE
            result.append(updated)

        if shortlist_count == 0 and result:
            result.sort(key=lambda item: item.score.elo if item.score else 0, reverse=True)
            result[0].status = HypothesisStatus.SHORTLISTED
        return sorted(result, key=lambda item: item.score.elo if item.score else 0, reverse=True)

    async def recommend_next_experiments(
        self,
        project: ResearchProject,
        *,
        round_summary: dict[str, Any],
        allowed_cells: list[ExperimentCell],
    ) -> ExperimentFeedbackProposal:
        """Use Qwen as a scientific advisor inside a deterministic action boundary."""

        try:
            response = await self.client.complete(
                role_name="AdaptiveExperimentPlanner",
                system_prompt=(
                    "你是少样本工业视觉异常检测的自适应实验规划智能体。"
                    "只根据真实运行摘要决定下一轮最有信息量的实验，不得虚构指标。"
                    "你只能从 allowed_cells 中选择最多两个单元；每个单元会由系统强制生成 "
                    "random 与 k_center 成对运行。优先证伪价值、跨类别复现、K 敏感性与失败诊断，"
                    "并减少无效穷举。paired_metric_summaries 只用于解释机制和边界，不能悄悄替换"
                    "预注册主指标；如果 primary_metric_saturated=true，应优先选择更难类别。"
                    "pair_count/cumulative_pair_count 是全活动累计配对数，"
                    "round_pair_count 是本轮新增数。"
                    "mean_difference/positive_pair_fraction 仅描述本轮；"
                    "跨轮总体方向必须读取 cumulative_primary_summary。"
                    "只有达到 minimum_pairs 后才能建议 stop。"
                    "所有自然语言字段使用简体中文。"
                ),
                payload={
                    "hypothesis": next(
                        (
                            item.model_dump(mode="json")
                            for item in project.hypotheses
                            if project.experiment_campaign is not None
                            and item.id == project.experiment_campaign.hypothesis_id
                        ),
                        None,
                    ),
                    "round_summary": round_summary,
                    "recent_human_guidance": [
                        item.model_dump(mode="json")
                        for item in project.guidance_records[-8:]
                    ],
                    "allowed_cells": [
                        item.model_dump(mode="json") for item in allowed_cells[:100]
                    ],
                    "output_schema": {
                        "advisor": self.name,
                        "decision": "expand|replicate|diagnose|stop",
                        "rationale": "string",
                        "observed_patterns": ["string"],
                        "next_phase": (
                            "sensitivity|main_study|replication|ablation|"
                            "cross_dataset|complete"
                        ),
                        "recommended_cells": [
                            {"category": "string", "shots": "integer", "seed": "integer"}
                        ],
                        "expected_information_gain": "0..1",
                        "stop": "boolean",
                    },
                },
            )
            response["advisor"] = self.name
            proposal = ExperimentFeedbackProposal.model_validate(response)
            if proposal.stop and int(round_summary.get("pair_count", 0)) < int(
                round_summary.get("minimum_pairs", 6)
            ):
                proposal.stop = False
                proposal.decision = "expand"
                proposal.next_phase = "replication"
                proposal.rationale += " 系统否决了提前停止：尚未达到预注册最小成对样本数。"
            return proposal
        except Exception as exc:
            fallback = await super().recommend_next_experiments(
                project,
                round_summary=round_summary,
                allowed_cells=allowed_cells,
            )
            fallback.advisor = f"{self.name}:deterministic-fallback"
            fallback.observed_patterns.append(
                f"Qwen 规划调用未产生有效结构化结果：{type(exc).__name__}"
            )
            return fallback

    async def interpret_experiment_guidance(
        self,
        project: ResearchProject,
        *,
        guidance: str,
        candidate_runs: list[ExperimentRun],
    ) -> ExperimentGuidanceDecision:
        """Let Qwen interpret intent while a deterministic allow-list owns the action."""

        if not candidate_runs:
            raise ValueError("No queued experiment is available for guidance")
        candidates = [
            {
                "run_id": run.id,
                "category": run.category,
                "shots": run.shots,
                "seed": run.seed,
                "selection_strategy": run.selection_strategy,
                "detector": run.detector,
                "protocol": run.protocol,
            }
            for run in candidate_runs
        ]
        allowed_ids = {item["run_id"] for item in candidates}
        try:
            response = await self.client.complete(
                role_name="HumanExperimentGuidanceAgent",
                system_prompt=(
                    "你负责解释用户在单次真实实验执行前的指导。你只能从 candidate_runs "
                    "中选择一个 run_id，可以调整执行优先级，但绝不能修改预注册配置、指标、"
                    "数据边界或生成任意命令。若建议需要新增类别、K、seed、检测器或指标，"
                    "将 disposition 标为 not_applicable，并选择系统默认候选，同时说明应在"
                    "下一实验轮或下一研究循环重新预注册。所有自然语言使用简体中文。"
                ),
                payload={
                    "user_guidance": guidance,
                    "candidate_runs": candidates,
                    "output_schema": {
                        "selected_run_id": "one exact candidate run_id",
                        "interpretation": "string",
                        "disposition": (
                            "applied|partially_applied|not_applicable|rejected"
                        ),
                        "rationale": "string",
                        "execution_notes": ["string"],
                        "protected_constraints": ["string"],
                    },
                },
            )
            response["advisor"] = self.name
            decision = ExperimentGuidanceDecision.model_validate(response)
            if decision.selected_run_id not in allowed_ids:
                raise ValueError("Qwen selected a run outside the registered queue")
            required_guards = [
                "预注册配置保持不变",
                "测试异常标签不得用于支持集选择",
                "原始指导与解释写入 Research Ledger",
            ]
            decision.protected_constraints = list(
                dict.fromkeys([*decision.protected_constraints, *required_guards])
            )
            return decision
        except Exception as exc:
            fallback = await super().interpret_experiment_guidance(
                project,
                guidance=guidance,
                candidate_runs=candidate_runs,
            )
            fallback.advisor = f"{self.name}:deterministic-fallback"
            fallback.rationale += f" Qwen 解释回退：{type(exc).__name__}。"
            return fallback

    async def revise_hypotheses(self, project: ResearchProject) -> list[Hypothesis]:
        response = await self.client.complete(
            role_name="HypothesisRevisionAgent",
            system_prompt=(
                "根据真实实验 finding 修订被证伪或证据不足的假设。必须缩小或改变机制主张，"
                "不得只改措辞；保留可证伪零假设、分析契约和明确边界。不要改写已支持假设。"
                "human_guidance_for_next_cycle 是用户对下一研究循环的明确建议，应说明如何采纳；"
                "若与真实证据、预算或研究完整性规则冲突，只采纳可执行部分并在 rationale 中解释。"
                "除标准技术名词外，所有自然语言字段使用简体中文。"
            ),
            payload={
                "research_cycle": project.research_cycle,
                "hypotheses": [item.model_dump(mode="json") for item in project.hypotheses],
                "findings": [item.model_dump(mode="json") for item in project.findings],
                "human_guidance_for_next_cycle": [
                    item.model_dump(mode="json")
                    for item in project.guidance_records
                    if item.scope == "research_cycle"
                    and item.research_cycle == project.research_cycle
                ],
                "return": {"hypotheses": "array of full Hypothesis records without id"},
            },
        )
        previous = {item.id: item for item in project.hypotheses}
        revised: list[Hypothesis] = []
        for payload in response.get("hypotheses", []):
            parent_id = payload.get("parent_hypothesis_id")
            parent = previous.get(parent_id)
            if parent is None:
                continue
            payload["id"] = new_id("hypothesis")
            payload["revision"] = parent.revision + 1
            payload["status"] = HypothesisStatus.CANDIDATE
            payload["score"] = None
            revised.append(Hypothesis.model_validate(payload))
        return revised


def _normalize_hypothesis_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize container types without rewriting model-generated science content."""

    normalized = dict(payload)
    for field in (
        "independent_variables",
        "dependent_variables",
        "falsification_conditions",
        "evidence_ids",
        "closest_prior_work",
    ):
        normalized[field] = _string_list(normalized.get(field))

    if not isinstance(normalized.get("analysis_contract"), dict):
        normalized["analysis_contract"] = None

    # Identity, score, and lifecycle fields are owned by the durable workflow.
    for field in ("id", "score", "status", "revision", "parent_hypothesis_id"):
        normalized.pop(field, None)
    return normalized


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value if item is not None]
    return [str(value)]
