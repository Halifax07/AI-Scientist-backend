from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from typing import Any

from fsad_scientist.agents.agentscope_client import (
    AgentOutputValidationError,
    AgentScopeJsonClient,
)
from fsad_scientist.agents.mock_runtime import MockScientistRuntime
from fsad_scientist.domain.enums import EvidenceStatus, HypothesisStatus
from fsad_scientist.domain.models import (
    ArtifactRecord,
    ExperimentCardPresentationSpec,
    ExperimentCell,
    ExperimentDesignSpec,
    ExperimentFeedbackProposal,
    ExperimentGuidanceDecision,
    ExperimentPlan,
    ExperimentRun,
    Hypothesis,
    HypothesisScore,
    MethodImplementation,
    ResearchGap,
    ResearchProject,
    new_id,
)
from fsad_scientist.experiments.code_safety import (
    extract_detector_source,
    extract_select_function,
    implementation_detector_name,
    sanitize_strategy_name,
    validate_detector_source,
    validate_strategy_source,
)
from fsad_scientist.experiments.design import (
    normalize_design_conditions_payload,
    normalize_presentation_spec_payload,
    result_aware_presentation_spec,
    validate_design,
)
from fsad_scientist.experiments.detector_runner import assemble_detector_file
from fsad_scientist.experiments.strategy_runner import assemble_strategy_file


async def _generate_validated_source(
    *,
    client: Any,
    role_name: str,
    system_prompt: str,
    request_payload: dict[str, Any],
    extractor: Callable[[str], str],
    validator: Callable[[str], Any],
    repair_focus: str,
    failure_label: str,
) -> tuple[str, list[str]]:
    source = ""
    issues: list[str] = []
    for attempt in range(4):
        repairing = attempt > 0
        payload = request_payload
        prompt = system_prompt
        if repairing:
            payload = {
                **request_payload,
                "previous_source_code": source,
                "validation_issues": issues,
                "repair_instruction": "逐条修复全部结构化输出与静态校验问题。",
            }
            prompt = (
                system_prompt
                + "上一版响应未通过结构化解析、源码提取或静态校验。"
                "必须逐条消除 validation_issues；"
                + repair_focus
                + "只返回包含 source_code 的 JSON 对象。"
            )
        try:
            response = await client.complete(
                role_name=role_name,
                system_prompt=prompt,
                payload=payload,
            )
            source = extractor(str(response.get("source_code", "")))
        except ValueError as exc:
            issues = [str(exc)]
            continue
        validation = validator(source)
        if validation.passed:
            return source, issues
        issues = validation.issues
    raise ValueError(f"四次生成后的{failure_label}仍未通过：" + "；".join(issues))


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

    @staticmethod
    def _project_context_block(project: ResearchProject) -> str:
        """Render the user-supplied research context for LLM system prompts.

        The platform default is the broad ``machine_vision_anomaly_detection``
        preset, where prompts must talk about the user's ``objective`` and
        ``domain`` rather than any specific dataset. Only when the user
        explicitly opts into ``fsad`` (or ``generic``) does the prompt narrow
        its scope to the embedded demo content.
        """
        spec = project.spec
        domain = (spec.domain or "").strip() or "用户自定义研究领域"
        objective = (spec.objective or "").strip()
        application_context = (spec.application_context or "").strip()
        if spec.preset == "fsad":
            default_objective = (
                "在仅有极少量正常参考图像、且适配阶段没有真实异常样本时,"
                "自主发现能够改善工业异常检测性能或稳定性的机制。"
            )
            return (
                "【平台选题】少样本工业视觉异常检测演示（MVTec AD 风格的少样本 "
                "正常参考样本场景,平台内置演示证据与空白候选）。\n"
                f"【研究领域】{domain}\n"
                f"【研究目标】{objective or default_objective}\n"
                f"【应用上下文】{application_context or '用户提供的研究场景和约束条件'}"
            )
        if spec.preset == "generic" or spec.preset is None:
            return (
                "【平台选题】用户通过通用科研工作台自定义的研究领域;"
                "不预设任何特定数据集或方法;所有 AI 生成内容必须围绕"
                "用户在 objective 中给出的研究方向与关键词。\n"
                f"【研究领域】{domain}\n"
                f"【研究目标】{objective or domain}\n"
                f"【应用上下文】{application_context or '用户提供的研究场景和约束条件'}"
            )
        # 平台默认:machine_vision_anomaly_detection 等宽泛方向。
        default_objective = (
            "围绕用户输入的研究方向与关键词,自主发现问题空白、"
            "提出可证伪的创新机制,并设计验证实验。"
        )
        return (
            "【平台选题】基于机器视觉的异常检测,平台默认宽泛方向;"
            "不预设任何特定数据集或方法;所有 AI 生成内容必须围绕"
            "用户在 objective 与 user_guidance 中给出的研究方向与关键词。\n"
            f"【研究领域】{domain}\n"
            f"【研究目标】{objective or default_objective}\n"
            f"【应用上下文】{application_context or '用户提供的研究场景和约束条件'}"
        )

    @staticmethod
    def _design_failure_plan(
        plan: ExperimentPlan,
        reason: str,
        errors: list[str] | None = None,
        *,
        status: str = "needs_correction",
    ) -> ExperimentPlan:
        payload = plan.model_dump(mode="json")
        payload["design_generation_status"] = status
        payload["design_generation_fallback_reason"] = reason
        payload["design_generation_errors"] = list(errors or [reason])
        payload.pop("preregistration_digest", None)
        digest = hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return ExperimentPlan(**payload, preregistration_digest=digest)

    @staticmethod
    def _fallback_is_executable(plan: ExperimentPlan) -> bool:
        """Only expose deterministic fallback when it covers the frozen portfolio."""

        design_hypothesis_ids = {
            design.hypothesis_id
            for design in plan.designs
            if design.hypothesis_id is not None
        }
        if not plan.designs or set(plan.hypothesis_ids) != design_hypothesis_ids:
            return False
        try:
            for design in plan.designs:
                validate_design(
                    design,
                    plan,
                    allowed_categories=set(plan.categories),
                    allowed_detectors=set(plan.detectors),
                    allowed_strategies=set(plan.selection_strategies),
                )
        except ValueError:
            return False
        return True

    async def formalize_scope(self, project: ResearchProject) -> ArtifactRecord:
        try:
            response = await self.client.complete(
                role_name="Supervisor",
                system_prompt=(
                    "你是自主科研项目经理。用户只提供研究领域、数据、现实约束和预算。"
                    "把它转化为结构化研究范围，但不要替用户预设最终创新结论。"
                    "始终根据用户在 ProjectSpec.objective / domain 中提供的"
                    "研究领域与关键词展开，不得偷换为与用户输入无关的方向。"
                    f"\n{self._project_context_block(project)}\n"
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
        except AgentOutputValidationError:
            fallback = await super().formalize_scope(project)
            return fallback.model_copy(
                update={
                    "provenance": [
                        *fallback.provenance,
                        self.name,
                        "invalid_json_fallback",
                    ]
                }
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
                "你负责从已有证据候选、用户研究场景与方法差异中发现研究空白。"
                "提出 3 至 6 个互不重复的空白。不得把未经校验的文献候选视为事实。"
                "必须严格围绕用户在 ProjectSpec 中给出的 objective 与"
                "domain 展开，禁止偷换为与用户输入无关的工业视觉或异常检测话题。"
                f"\n{self._project_context_block(project)}\n"
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
                "你负责把研究空白转化为可证伪科学假设。"
                "所有假设必须围绕用户在 ProjectSpec.objective / domain 中给出的"
                "研究领域与关键词展开；不要回到少样本工业视觉异常检测、"
                "MVTec AD 或 PatchCore 等与用户输入无关的默认话题。"
                f"\n{self._project_context_block(project)}\n\n"
                "每个假设必须包含零假设、"
                "变量、预测方向和明确的证伪条件；从不同机制提出 3 至 6 个候选，"
                "不要写成模糊的工程目标。候选构成必须同时满足四条强制规则："
                "(1) 至少 1 个候选使用 design_mode=custom_design，不提供 treatment/control，"
                "检验现有 random/k_center 因素矩阵无法表达的机制变量；注意 kind 字段只能填"
                "selection_main_effect、detector_interaction 或 query_adaptation 三者之一，"
                "禁止在 kind 中填写 custom_design，custom_design 只允许出现在 design_mode 字段；"
                "(2) 至少 1 个候选为 selection_main_effect 或 query_adaptation，且 treatment "
                "必须是全新实现名（control 用 random），并在 rationale 中用不超过 3 句写明"
                "该名称的确定性步骤（输入→操作→输出）；"
                "(3) 其余候选可以使用已注册条件，但不得全部退回 random 对 k_center 的默认对照。"
                "(4) 机制避让与多样性：全部候选不得把同一机制换名复述；当用户给出的上下文"
                "自然引导到以下演示中已过度使用的机制族时，本轮一律搁置，改从其他因果轴提出假设："
                "参考库/候选池压缩或修剪（含查询感知、预算感知的动态剪枝）；"
                "类别广度与样本深度的总预算权衡；候选池规模扩大带来的边际递减或收益曲线；"
                "K 与预训练表征饱和度的交互或饱和点。"
                "可优先探索的支持集选取与检测机制轴（举例，不限于）：支持集内簇覆盖与冗余结构、"
                "对最坏类别或高方差类别的优先保护、选取规则与检测器内部表征粒度是否匹配、"
                "伪样本与噪声对边界的鲁棒性、采样稳定性与 seed 依赖等。"
                "claim 的主语必须是待检验的因果机制与预测方向，不得把\"提出某方法\"当作 claim。"
                "analysis_contract.kind 可为 selection_main_effect、"
                "detector_interaction 或 query_adaptation；它只是兼容旧流程和安全边界，"
                "不要要求所有假设固定两种 selection strategy，也不要固定 k_center/random。"
                "treatment/control 可以描述方法、检测器或其他兼容字段；当前工具链可直接执行的"
                "方法应使用已注册实现，尚未注册的方法可以作为 requires_implementation 候选。"
                "后续 ExperimentDesignSpec 决定因素、条件、复现/探索方式和分析形式。"
                "analysis_contract.metric 必须严格使用以下标识之一："
                "image_auroc、pixel_auroc、image_ap、aupro；不要返回 Image AUROC、Pixel AUROC、"
                "AUPRO 等展示标签或 Recall at FPR=5% 等执行器不支持的指标。"
                "analysis_contract.design_mode 可选择 paired_comparison 或 custom_design；"
                "custom_design 可以不提供 treatment/control，由后续设计绑定 Run 字段。"
                "paired_comparison 的 treatment 与 control 必须是两个不同的条件；"
                "不得比较同一个方法、检测器或策略与自身。"
                "detector_interaction 的 treatment/control 必须逐字使用 provided "
                "registered_detectors/detectors 名单中的精确检测器名（如 patchcore、"
                "subspacead）；禁止拼接检测器名与策略名（如 subspacead_kcenter、"
                "k_center_subspacead），名单外的检测器名会在提交时被后端拒绝。"
                "想比较同一检测器内 k_center 与 random 支持集选择策略时，改用 "
                "selection_main_effect 并把 treatment/control 精确填为 random 与 "
                "k_center，不要把策略名接到检测器名上。"
                "selection_main_effect/query_adaptation 的 treatment/control 使用 "
                "random、k_center 或已注册实现名；需要新条件时按上方强制规则 (2) "
                "给出新实现名，系统会在提交后自动生成并校验实现，"
                "但不要把它伪装成检测器名。"
                "除论文标题和标准技术名词外，"
                "所有自然语言字段使用简体中文。"
            ),
            payload={
                "gaps": [item.model_dump(mode="json") for item in project.gaps],
                "evidence_candidates": [
                    item.model_dump(mode="json") for item in project.evidence
                ],
                "execution_capabilities": {
                    "implemented_contract_kinds": [
                        "selection_main_effect",
                        "detector_interaction",
                        "query_adaptation",
                    ],
                    "registered_selection_strategies": [
                        "random",
                        "k_center",
                        *[
                            item.name
                            for item in project.method_implementations
                            if item.kind == "selection_strategy"
                            and item.status in {"validated", "approved"}
                        ],
                    ],
                    "registered_detectors": [
                        "patchcore",
                        "anomalydino",
                        "subspacead",
                        *[
                            item.name
                            for item in project.method_implementations
                            if item.kind == "detector"
                            and item.status in {"validated", "approved"}
                        ],
                    ],
                    "detectors": ["anomalydino", "patchcore", "subspacead"],
                    "datasets": ["MVTec AD"],
                    "shots": project.spec.constraints.shots,
                    "rule": "可执行性由已注册实现和后续合法 ExperimentDesignSpec 共同决定",
                },
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
                                    "selection_main_effect|detector_interaction|query_adaptation"
                                ),
                                "metric": "image_auroc|pixel_auroc|image_ap|aupro",
                                "design_mode": "paired_comparison|custom_design",
                                "treatment": "method_or_detector_name or null",
                                "control": "method_or_detector_name or null",
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
            contract = normalized.get("analysis_contract")
            if isinstance(contract, dict):
                # kind only accepts the three causal axes; custom design is
                # expressed through design_mode. Tolerate a model that wrote
                # custom_design into kind or omitted kind on a custom card.
                if contract.get("design_mode") == "custom_design":
                    if contract.get("kind") not in (
                        "selection_main_effect",
                        "detector_interaction",
                        "query_adaptation",
                    ):
                        contract["kind"] = "selection_main_effect"
                elif contract.get("kind") == "custom_design":
                    contract["design_mode"] = "custom_design"
                    contract["kind"] = "selection_main_effect"
                if not _has_distinct_conditions(contract):
                    continue
            normalized["evidence_ids"] = [
                evidence_id
                for evidence_id in normalized["evidence_ids"]
                if evidence_id in valid_evidence_ids
            ]
            hypotheses.append(Hypothesis.model_validate(normalized))
        if not hypotheses:
            raise AgentOutputValidationError(
                "Qwen 返回的假设没有形成两个不同的实验条件，请重新生成研究假设。"
            )
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
        # Models occasionally use the input-facing ``hypothesis_id`` name,
        # return the reviews under ``scores``/``rankings``, or nest the actual
        # dimensions below ``score``. Normalize those harmless schema variants
        # before applying the durable score model.
        raw_reviews: Any = (
            response.get("reviews")
            or response.get("hypothesis_reviews")
            or response.get("scores")
            or response.get("rankings")
            or response.get("hypotheses")
            or []
        )
        if isinstance(raw_reviews, dict):
            raw_reviews = [
                (
                    {**value, "id": key}
                    if isinstance(value, dict)
                    else {"id": key, "elo": value}
                )
                for key, value in raw_reviews.items()
            ]
        reviews: dict[str, dict[str, Any]] = {}
        if isinstance(raw_reviews, list):
            for item in raw_reviews:
                if not isinstance(item, dict):
                    continue
                hypothesis_id = (
                    item.get("id")
                    or item.get("hypothesis_id")
                    or item.get("hypothesisId")
                )
                if hypothesis_id:
                    reviews[str(hypothesis_id)] = item
        result: list[Hypothesis] = []
        shortlist_count = 0

        # The reviewer may omit a hypothesis or echo a mismatched id.  Fill a
        # neutral score so every candidate is rankable; never leave score None
        # (the frontend renders an empty AI-score cell for None).
        def _review_number(value: Any, fallback: float) -> float:
            try:
                number = float(value)
            except (TypeError, ValueError):
                return fallback
            return number if math.isfinite(number) else fallback

        def _bounded_review_number(value: Any, fallback: float) -> float:
            return max(0.0, min(1.0, _review_number(value, fallback)))

        for hypothesis in project.hypotheses:
            updated = hypothesis.model_copy(deep=True)
            review: dict[str, Any] | None = reviews.get(hypothesis.id)
            if review:
                score_payload = review.get("score")
                if not isinstance(score_payload, dict):
                    score_payload = review
                updated.score = HypothesisScore(
                    novelty=_bounded_review_number(score_payload.get("novelty"), 0.5),
                    falsifiability=_bounded_review_number(
                        score_payload.get("falsifiability"), 0.5
                    ),
                    feasibility=_bounded_review_number(score_payload.get("feasibility"), 0.5),
                    scientific_value=_bounded_review_number(
                        score_payload.get("scientific_value"), 0.5
                    ),
                    evidence_strength=min(
                        _bounded_review_number(score_payload.get("evidence_strength"), 0.5),
                        maximum_evidence_strength,
                    ),
                    elo=_review_number(score_payload.get("elo"), 1000.0),
                )
                if review.get("status") == "shortlisted" and shortlist_count < 2:
                    updated.status = HypothesisStatus.SHORTLISTED
                    shortlist_count += 1
                else:
                    updated.status = HypothesisStatus.CANDIDATE
            else:
                # Reviewer did not return this hypothesis: keep it rankable as a
                # neutral candidate instead of leaving score unset.
                updated.score = updated.score or HypothesisScore(
                    novelty=0.5,
                    falsifiability=0.5,
                    feasibility=0.5,
                    scientific_value=0.5,
                    evidence_strength=min(0.5, maximum_evidence_strength),
                    elo=1000.0,
                )
                updated.status = HypothesisStatus.CANDIDATE
            result.append(updated)

        if shortlist_count == 0 and result:
            result.sort(key=lambda item: item.score.elo if item.score else 0, reverse=True)
            result[0].status = HypothesisStatus.SHORTLISTED
        return sorted(result, key=lambda item: item.score.elo if item.score else 0, reverse=True)

    async def design_experiments(self, project: ResearchProject) -> ExperimentPlan:
        """Generate structured conditions while keeping the local plan boundary authoritative."""

        fallback = await super().design_experiments(project)
        try:
            response = await self.client.complete(
                role_name="ExperimentDesignAgent",
                system_prompt=(
                    "你负责为已批准的科学假设生成轻量通用实验设计。"
                    "所有设计必须服务于用户在 ProjectSpec 中给出的研究领域与目标；"
                    "若该课题不是少样本工业视觉异常检测，请不要硬编码 FSAD"
                    "默认场景或默认检测器列表。"
                    f"\n{self._project_context_block(project)}\n"
                    "必须为每个假设自主选择 design_mode=paired_comparison 或 custom_design；"
                    "paired_comparison 才能使用 treatment/control，custom_design 不得依赖它们。"
                    "设计必须绑定现有 Run 字段 selection_strategy、detector、category、"
                    "shots、seed 或 protocol；"
                    "每个条件必须给出完整 factor_values，条件 ID 唯一。"
                    "允许 group_comparison、factor_effects、interaction_summary、ordered_trend、"
                    "distribution_summary；interaction_summary 将归一化为 factor_effects；"
                    "不要把设计限制为 k_center/random，所有值必须来自 allowed_values。"
                    "必须同时生成 question、rationale 和受控 presentation_spec；"
                    "布局可以在输出枚举内自由组合变化，但字段值必须逐字来自 output_schema："
                    "block.kind 只能是 narrative、progress、metrics、chart、table、runs、"
                    "evidence、decision、diagnostics、insight、callout、key_value、timeline "
                    "之一，不存在 bar_chart、distribution_summary、box_plot、scatter、"
                    "histogram 等块类型（analysis.mode 或 block.source 里的词不能当 kind 用）；"
                    "chart_mark 只能取 bar、line、point、heatmap、interval 之一，"
                    "其他图形词汇（boxplot、scatter、histogram、pie 等）一律不用；"
                    "只有 kind=chart 的块才填 chart_mark；"
                    "kind=chart 的块其 source 不得为 runs 或 evidence，"
                    "图表必须基于汇总数据（condition_statistics、factor_effects、"
                    "interaction_summary、ordered_trend 或 distribution_summary）。"
                    "除技术名词外，自然语言使用简体中文。"
                ),
                payload={
                    "research_context": {
                        "objective": project.spec.objective,
                        "application_context": project.spec.application_context,
                        "user_guidance": project.spec.user_guidance,
                        "budget": project.spec.budget.model_dump(mode="json"),
                        "constraints": project.spec.constraints.model_dump(mode="json"),
                    },
                    "hypotheses": [
                        item.model_dump(mode="json")
                        for item in project.hypotheses
                        if item.id in fallback.hypothesis_ids
                    ],
                    "allowed_values": {
                        "selection_strategy": fallback.selection_strategies,
                        "detector": fallback.detectors,
                        "category": fallback.categories,
                        "shots": fallback.shots,
                        "seed": fallback.seeds,
                        "protocol": fallback.protocols,
                    },
                    "output_schema": {
                        "designs": [
                            {
                                "id": "design_id",
                                "hypothesis_id": "hypothesis_id",
                                "design_mode": "paired_comparison|custom_design",
                                "support_selection_strategy": "allowed strategy or null",
                                "question": "string",
                                "rationale": "string",
                                "factors": [
                                    {
                                        "name": "factor_name",
                                        "field": (
                                            "selection_strategy|detector|category|shots|seed|protocol"
                                        ),
                                        "levels": ["allowed value"],
                                    }
                                ],
                                "conditions": [
                                    {
                                        "id": "condition_id",
                                        "label": "string",
                                        "factor_values": {"factor_name": "allowed value"},
                                    }
                                ],
                                "analysis": {
                                    "mode": (
                                        "group_comparison|factor_effects|interaction_summary|ordered_trend|"
                                        "distribution_summary"
                                    ),
                                    "primary_metric": "image_auroc|pixel_auroc|image_ap|aupro",
                                    "minimum_pairs": 2,
                                },
                                "presentation_spec": {
                                    "schema_version": 2,
                                    "layout": "stack|split|grid|sequence",
                                    "density": "compact|comfortable",
                                    "blocks": [
                                        {
                                            "id": "block_id",
                                            "kind": (
                                                "narrative|progress|metrics|chart|table|runs|"
                                                "evidence|decision|diagnostics|insight|callout|key_value|timeline"
                                            ),
                                            "source": (
                                                "design|progress|condition_statistics|condition_effects|"
                                                "factor_effects|interaction_summary|ordered_trend|"
                                                "distribution_summary|runs|evidence|feedback|diagnostics"
                                            ),
                                            "chart_mark": "bar|line|point|heatmap|interval or null",
                                            "span": "full|half|third",
                                            "title": "string or null",
                                            "content": "plain text or null",
                                            "config": "JSON object with display data only",
                                        }
                                    ],
                                },
                            }
                        ]
                    },
                },
            )
            designs: list[ExperimentDesignSpec] = []
            design_errors: list[str] = []
            for raw in response.get("designs", []):
                if not isinstance(raw, dict):
                    design_errors.append("AI returned a non-object experiment design")
                    continue
                try:
                    raw_design = dict(raw)
                    if "conditions" in raw_design:
                        raw_design["conditions"] = normalize_design_conditions_payload(
                            raw_design["conditions"]
                        )
                    raw_spec = raw_design.pop("presentation_spec", None)
                    if raw_spec is None:
                        raise ValueError(
                            f"{raw_design.get('id', 'design')}: missing presentation_spec"
                        )
                    # Validate experiment semantics independently, then repair only
                    # additive presentation omissions before strict DSL validation.
                    design_without_spec = ExperimentDesignSpec.model_validate(raw_design)
                    repaired_spec = normalize_presentation_spec_payload(
                        raw_spec,
                        analysis_mode=design_without_spec.analysis.mode,
                    )
                    design = ExperimentDesignSpec.model_validate(
                        {**raw_design, "presentation_spec": repaired_spec}
                    )
                    if design.hypothesis_id not in fallback.hypothesis_ids:
                        design_errors.append("AI design references an unknown hypothesis")
                        continue
                    validate_design(
                        design,
                        fallback,
                        allowed_categories=set(fallback.categories),
                        allowed_detectors=set(fallback.detectors),
                        allowed_strategies=set(fallback.selection_strategies),
                    )
                except (TypeError, ValueError) as exc:
                    design_errors.append(str(exc))
                    continue
                designs.append(design)
            payload = fallback.model_dump(mode="json")
            if not designs:
                # No usable AI design at all: keep the deterministic portfolio so
                # the approved plan never silently drops executable hypotheses.
                return self._design_failure_plan(
                    fallback,
                    "AI did not return a valid executable experiment design; "
                    "deterministic fallback designs were kept.",
                    design_errors,
                    status="fallback" if self._fallback_is_executable(fallback) else "needs_correction",
                )
            ai_by_hypothesis = {
                item.hypothesis_id: item
                for item in designs
                if item.hypothesis_id is not None
            }
            expected_ids = set(fallback.hypothesis_ids)
            actual_ids = set(ai_by_hypothesis)
            if expected_ids - actual_ids:
                # The deterministic fallback already carries one validated design
                # per executable hypothesis; let it fill coverage the AI skipped
                # instead of leaving the plan to be pruned at scoping time.
                fallback_by_hypothesis = {
                    item.hypothesis_id: item
                    for item in fallback.designs
                    if item.hypothesis_id is not None
                }
                missing = sorted(expected_ids - actual_ids)
                covered = sorted(
                    hypothesis_id
                    for hypothesis_id in missing
                    if hypothesis_id in fallback_by_hypothesis
                )
                if covered:
                    design_errors.append(
                        "以下假设由确定性兜底设计补齐（AI 未返回合法设计）："
                        + ", ".join(covered)
                    )
                if covered:
                    payload["design_generation_status"] = "ai_selected"
                    payload["design_generation_fallback_reason"] = None
                else:
                    payload["design_generation_status"] = "needs_correction"
                    payload["design_generation_fallback_reason"] = (
                        "AI did not provide a valid design for every executable hypothesis."
                    )
                    design_errors.append(f"missing designs: {missing}")
                ai_by_hypothesis.update(
                    {
                        hypothesis_id: fallback_by_hypothesis[hypothesis_id]
                        for hypothesis_id in covered
                    }
                )
            else:
                payload["design_generation_status"] = "ai_selected"
                payload["design_generation_fallback_reason"] = None
            payload["design_generation_errors"] = design_errors
            payload["designs"] = [
                item.model_dump(mode="json")
                for item in (
                    ai_by_hypothesis[hypothesis_id]
                    for hypothesis_id in fallback.hypothesis_ids
                    if hypothesis_id in ai_by_hypothesis
                )
            ]
            payload.pop("preregistration_digest", None)
            digest = hashlib.sha256(
                json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
            ).hexdigest()
            return ExperimentPlan(**payload, preregistration_digest=digest)
        except Exception as exc:
            fallback_status = (
                "fallback" if self._fallback_is_executable(fallback) else "needs_correction"
            )
            return self._design_failure_plan(
                fallback,
                (
                    "Qwen 实验设计调用失败，已保留可执行的 deterministic fallback。"
                    if fallback_status == "fallback"
                    else "AI experiment design generation failed; correction is required."
                ),
                [str(exc)],
                status=fallback_status,
            )

    async def recommend_next_experiments(
        self,
        project: ResearchProject,
        *,
        round_summary: dict[str, Any],
        allowed_cells: list[ExperimentCell],
        user_guidance: str | None = None,
    ) -> ExperimentFeedbackProposal:
        """Use Qwen as a scientific advisor inside a deterministic action boundary."""

        legacy_design = None
        if "design_mode" not in round_summary and round_summary.get("design_id"):
            plan = project.experiment_plan
            legacy_design = next(
                (item for item in (plan.designs if plan is not None else [])
                 if item.id == round_summary.get("design_id")),
                None,
            )
        is_explicit_design = round_summary.get("design_mode") == "custom_design" or (
            legacy_design is not None and legacy_design.design_mode == "custom_design"
        )
        if is_explicit_design:
            return await self._recommend_explicit_design_feedback(
                project,
                round_summary=round_summary,
                allowed_cells=allowed_cells,
                user_guidance=user_guidance,
            )

        try:
            response = await self.client.complete(
                role_name="AdaptiveExperimentPlanner",
                system_prompt=(
                    "你是自主科研平台的自适应实验规划智能体。"
                    "所有分析与建议都必须围绕用户在 ProjectSpec 中声明的研究领域与目标；"
                    "若该课题不是少样本工业视觉异常检测，请不要沿用 FSAD 演示的"
                    "默认语境或硬编码工业视觉假设。\n"
                    f"{self._project_context_block(project)}\n\n"
                    "【当前实验语义】一个 Round 只验证一个创新点，固定包含 3 次预注册迭代。"
                    "默认执行模式是 parallel：多个已选创新点各自拥有独立 Round，三次迭代由本机"
                    "并行调度；每个 Round 的第 1 次迭代完成后必须等待一次用户指导，"
                    "再自动执行第二 2、3 次迭代。"
                    "不能要求用户逐 Run 批准，也不能把不同创新点的结果合并。"
                    "用户在首轮后提交的指导意见是本 Round 后续两次迭代的软约束；应尽量采纳，"
                    "但不得突破预注册的创新点、数据隔离、三次迭代和测试标签隔离约束。"
                    "兼容旧 sequential 模式时，completed_iterations=1 才允许一次中途指导；"
                    "completed_iterations=3 时只汇总当前 Round。\n\n"
                    "【决策类型】你可以给出以下决策：\n"
                    "1. expand: 扩展到新类别，检验效应跨类别泛化能力\n"
                    "2. replicate: 增加随机种子，提高统计可信度\n"
                    "3. diagnose: 诊断异常结果或失败原因\n"
                    "4. stop: 收集足够证据后停止实验\n"
                    "5. adapt_k: 根据当前 K 值敏感性分析结果，调整 K 值\n"
                    "6. focus_category: 聚焦效应最显著的类别进行深入分析\n"
                    "7. ablate: 消融实验，移除或修改某个组件\n"
                    "8. early_stop: 效应已显著强于基线，可以提前停止\n\n"
                    "【决策依据】应综合考虑：\n"
                    "- 当前配对数是否达到最小要求（minimum_pairs）\n"
                    "- 效应量是否足够显著\n"
                    "- 正向配对比例是否稳定\n"
                    "- 各类别效应是否一致\n"
                    "- 置信区间宽度\n\n"
                    "【输出要求】\n"
                    "1. 在 reasoning_chain 中列出你的完整推理步骤，格式为：\n"
                    "   - observation: 观察到的具体事实（数字或现象）\n"
                    "   - conclusion: 从这个事实得出的结论\n"
                    "   - confidence: 对该结论的置信度（高/中/低）\n"
                    "2. 在 alternative_decisions 中说明你考虑过但未选择的方案及其原因\n"
                    "3. 在 expected_improvement 中说明预期的改进方向和幅度\n"
                    "4. pair_count/cumulative_pair_count 描述当前创新点 Round 的累计配对数，"
                    "round_pair_count 是当前 Round 已形成的配对数\n"
                    "5. mean_difference/positive_pair_fraction 描述当前 Round；"
                    "不同创新点之间不得直接合并为一个效应量\n"
                    "6. 只有达到 minimum_pairs 后才能建议 stop\n"
                    "7. 所有自然语言字段使用简体中文"
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
                    "round_contract": {
                        "hypothesis_id": round_summary.get("hypothesis_id"),
                        "iteration_target": 3,
                        "completed_iterations": round_summary.get("completed_iterations", 0),
                        "execution_mode": (
                            project.experiment_campaign.execution_mode
                            if project.experiment_campaign is not None
                            else "sequential"
                        ),
                        "human_guidance_gate": (
                            "after_iteration_1_per_round"
                            if project.experiment_campaign is not None
                            and project.experiment_campaign.execution_mode == "parallel"
                            else "after_iteration_1_only"
                        ),
                    },
                    "user_guidance": user_guidance or "",
                    "recent_human_guidance": [
                        item.model_dump(mode="json")
                        for item in project.guidance_records[-8:]
                    ],
                    "allowed_cells": [
                        item.model_dump(mode="json") for item in allowed_cells[:100]
                    ],
                    "output_schema": {
                        "advisor": self.name,
                        "decision": (
                            "expand|replicate|diagnose|stop|adapt_k|focus_category|ablate|early_stop"
                        ),
                        "rationale": "string",
                        "reasoning_chain": [
                            {
                                "step": "integer (starting from 1)",
                                "observation": "观察到的具体事实",
                                "conclusion": "从这个事实得出的结论",
                                "confidence": "高|中|低"
                            }
                        ],
                        "alternative_decisions": [
                            {
                                "decision": "考虑过的方案名称",
                                "rejected_reason": "为什么没有选择该方案"
                            }
                        ],
                        "expected_improvement": {
                            "metric": "指标名称",
                            "direction": "increase|decrease",
                            "estimated_delta": "number",
                            "confidence": "高|中|低"
                        },
                        "observed_patterns": ["string"],
                        "next_phase": (
                            "sensitivity|main_study|replication|ablation|"
                            "cross_dataset|complete"
                        ),
                        "recommended_cells": [
                            {"category": "string", "shots": "integer", "seed": "integer"}
                        ],
                        "strategy_adjustment": {
                            "focus_on_k": "integer or null (建议聚焦的 K 值)",
                            "priority_category": "string or null (优先测试的类别)",
                            "ablation_target": "string or null (消融目标)"
                        },
                        "expected_information_gain": "0..1",
                        "stop": "boolean",
                    },
                },
            )
            response["advisor"] = self.name
            proposal = ExperimentFeedbackProposal.model_validate(response)
            explicit_design = round_summary.get("design_mode") == "custom_design" or (
                legacy_design is not None and legacy_design.design_mode == "custom_design"
            )
            if proposal.stop:
                if explicit_design:
                    evidence_ready = round_summary.get("evidence_status") in {
                        "sample_threshold_met",
                        "sufficient",
                    }
                else:
                    evidence_ready = int(round_summary.get("pair_count", 0)) >= int(
                        round_summary.get("minimum_pairs", 6)
                    )
                if not evidence_ready:
                    proposal.stop = False
                    proposal.decision = "expand"
                    proposal.next_phase = "replication"
                    proposal.rationale += (
                        " 系统否决了提前停止：尚未达到预注册最小证据量。"
                    )
            return proposal
        except Exception as exc:
            fallback = await super().recommend_next_experiments(
                project,
                round_summary=round_summary,
                allowed_cells=allowed_cells,
                user_guidance=user_guidance,
            )
            fallback.advisor = f"{self.name}:deterministic-fallback"
            fallback.observed_patterns.append(
                f"Qwen 规划调用未产生有效结构化结果：{type(exc).__name__}"
            )
            return fallback

    async def _recommend_explicit_design_feedback(
        self,
        project: ResearchProject,
        *,
        round_summary: dict[str, Any],
        allowed_cells: list[ExperimentCell],
        user_guidance: str | None = None,
    ) -> ExperimentFeedbackProposal:
        design_id = str(round_summary["design_id"])
        plan = project.experiment_plan
        design = next(
            (item for item in plan.designs if item.id == design_id),
            None,
        ) if plan is not None else None
        if design is None:
            raise ValueError(f"Unknown experiment design: {design_id}")

        fallback_spec = result_aware_presentation_spec(design, round_summary)
        try:
            response = await self.client.complete(
                role_name="AdaptiveExperimentPlanner",
                system_prompt=(
                    "你是自主科研平台的实验结果规划智能体。"
                    "所有分析与建议都必须围绕用户在 ProjectSpec 中声明的研究领域与目标；"
                    "若该课题不是少样本工业视觉异常检测，请不要沿用 FSAD 演示的"
                    "默认语境或硬编码工业视觉假设。\n"
                    f"{self._project_context_block(project)}\n\n"
                    f"用户本轮指导：{user_guidance or '未提供'}\n\n"
                    "当前 Round 使用已批准的显式实验设计；只能在该设计的因素、条件和预算边界内"
                    "提出建议。\n"
                    "请依据 analysis_mode、condition_statistics、condition_effects、"
                    "factor_effects、"
                    "interaction_summary、ordered_trend、distribution_summary、sample_size、"
                    "evidence_status 和 failed_run_ids 判断下一步。推断统计尚未执行，"
                    "不得声称统计检验、统计显著性或科学证据已经充分。\n"
                    "同时选择一个受控 presentation_spec，反映真实结果：失败时包含"
                    "runs、diagnostics、evidence；趋势有效点不足时使用 ordered_trend table；"
                    "没有交互数据时不要使用空 heatmap，改用 condition_statistics 或 "
                    "factor_effects table。"
                    "只能使用给定 block kind/source/chart_mark；允许 insight、callout、key_value、timeline"
                    "等纯语义组件，可通过 content/config 提供数据，不得输出可执行代码；"
                    "不得输出 React、HTML、CSS 或 ECharts options。"
                ),
                payload={
                    "design": design.model_dump(mode="json"),
                    "round_summary": round_summary,
                    "allowed_cells": [
                        item.model_dump(mode="json") for item in allowed_cells[:100]
                    ],
                    "output_schema": {
                        "advisor": self.name,
                        "decision": (
                            "expand|replicate|diagnose|stop|adapt_k|focus_category|ablate|early_stop"
                        ),
                        "rationale": "string",
                        "reasoning_chain": "same structured list as the feedback contract",
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
                        "presentation_spec": {
                            "schema_version": 2,
                            "layout": "stack|split|grid|sequence",
                            "density": "compact|comfortable",
                            "blocks": [
                                {
                                    "id": "string",
                                    "kind": (
                                        "narrative|progress|metrics|chart|table|runs|"
                                        "evidence|decision|diagnostics|insight|callout|key_value|timeline"
                                    ),
                                    "source": (
                                        "design|progress|condition_statistics|condition_effects|"
                                        "factor_effects|interaction_summary|ordered_trend|"
                                        "distribution_summary|runs|evidence|feedback|diagnostics"
                                    ),
                                    "chart_mark": "bar|line|point|heatmap|interval|null",
                                    "span": "full|half|third",
                                    "title": "string or null",
                                    "content": "plain text or null",
                                    "config": "JSON object with display data only",
                                }
                            ],
                        },
                    },
                },
            )
            if not isinstance(response, dict):
                raise ValueError("Qwen returned a non-object feedback result")
            response = dict(response)
            response["advisor"] = self.name
            raw_spec = response.pop("presentation_spec", None)
            proposal = ExperimentFeedbackProposal.model_validate(response)
            if raw_spec is not None:
                try:
                    candidate_spec = ExperimentCardPresentationSpec.model_validate(
                        raw_spec
                    )
                    merged_design = design.model_copy(
                        update={"presentation_spec": candidate_spec}, deep=True
                    )
                    proposal.presentation_spec = result_aware_presentation_spec(
                        merged_design,
                        {**round_summary, "feedback": proposal.model_dump(mode="json")},
                    )
                except Exception:
                    proposal.presentation_spec = fallback_spec
            else:
                proposal.presentation_spec = fallback_spec
            if proposal.stop and round_summary.get("evidence_status") != "sample_threshold_met":
                proposal.stop = False
                proposal.decision = "expand"
                proposal.next_phase = "replication"
                proposal.rationale += " 系统否决了提前停止：尚未达到最小样本门槛。"
            return proposal
        except Exception as exc:
            fallback = await super().recommend_next_experiments(
                project,
                round_summary=round_summary,
                allowed_cells=allowed_cells,
                user_guidance=user_guidance,
            )
            fallback.advisor = f"{self.name}:deterministic-fallback"
            fallback.presentation_spec = fallback_spec
            fallback.observed_patterns.append(
                f"Qwen 设计反馈调用未产生有效结构化结果：{type(exc).__name__}"
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
                    "你负责解释用户指导或系统自动执行动作。底层一次请求只选择一个已排队的 run_id；"
                    "系统会连续调用该接口完成当前 Round 的三次内部迭代。用户指导只在第 1 次迭代"
                    "结束后通过 Round 审查接口提交一次，不应被解释成每个 run 都需要人工批准。你只能"
                    "从 candidate_runs 中选择一个 run_id，可以调整执行优先级，但绝不能修改预注册"
                    "配置、"
                    "指标、数据边界或生成任意命令。若建议需要新增类别、K、seed、检测器或指标，将"
                    "disposition 标为 not_applicable，并选择系统默认候选，同时说明应在下一实验"
                    " Round"
                    "或下一研究循环重新预注册。所有自然语言使用简体中文。"
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

    async def implement_selection_strategy(
        self,
        project: ResearchProject,
        *,
        hypothesis: Hypothesis,
        strategy_name: str,
        control_name: str,
    ) -> MethodImplementation:
        """Ask Qwen for a pure selection function; deterministic fallback on failure."""

        validation_issues: list[str] = []
        try:
            system_prompt = (
                "你是支持集选样策略实现专家，负责为当前研究领域生成候选样本选择函数。"
                f"\n{self._project_context_block(project)}\n\n"
                f"为策略 {strategy_name}（对照 {control_name}）编写实现。"
                "只返回一个纯 Python 函数，必须恰好是一个模块级 def select；"
                "函数体不得包含 import 语句，不得定义嵌套函数或类（运行模板已导入"
                "math、random、numpy 等允许模块）。"
                "不得读写文件、联网、启动子进程或调用 eval/exec；"
                "固定 seed 时必须确定性输出。函数只依赖候选正常样本的特征向量，"
                "不得接触任何测试/异常数据。除代码外所有自然语言使用简体中文。"
            )
            request_payload = {
                "hypothesis": hypothesis.model_dump(mode="json"),
                "strategy_name": strategy_name,
                "control_name": control_name,
                "function_contract": {
                    "signature": (
                        "def select(candidate_ids: list[str], "
                        "embeddings: dict[str, list[float]], k: int, seed: int) "
                        "-> list[str]"
                    ),
                    "semantics": (
                        "从 candidate_ids 中选出恰好 k 个不重复样本，"
                        "返回其文件 id 列表；只能基于 embeddings 与 seed。"
                    ),
                    "scale": "candidate_ids 不超过 30 个；向量维度不超过 384。",
                    "forbidden": [
                        "import 语句",
                        "嵌套函数或类定义",
                        "文件/网络/子进程/eval/exec",
                        "非确定性随机源（必须用 random.Random(seed) 或纯计算）",
                        "接触测试或异常标签",
                    ],
                },
                "output_schema": {
                    "source_code": "完整 def select 函数源码",
                    "explanation": "策略机制的一句话说明（简体中文）",
                },
                "return": {"source_code": "string"},
            }
            source, validation_issues = await _generate_validated_source(
                client=self.client,
                role_name="MethodImplementerAgent",
                system_prompt=system_prompt,
                request_payload=request_payload,
                extractor=extract_select_function,
                validator=validate_strategy_source,
                repair_focus="select 函数体内不得定义任何嵌套函数或类；",
                failure_label="选样策略",
            )
            assembled = assemble_strategy_file(source)
            digest = hashlib.sha256(assembled.encode("utf-8")).hexdigest()
            provenance = [self.name, "static_contract:accepted"]
            if validation_issues:
                provenance.append(f"{self.name}:validation-repair")
            return MethodImplementation(
                kind="selection_strategy",
                name=sanitize_strategy_name(strategy_name),
                hypothesis_id=hypothesis.id,
                source_code=source,
                code_digest=digest,
                provenance=provenance,
                status="draft",
            )
        except Exception as exc:
            fallback = await super().implement_selection_strategy(
                project,
                hypothesis=hypothesis,
                strategy_name=strategy_name,
                control_name=control_name,
            )
            fallback.provenance.append(
                f"{self.name}:deterministic-fallback:{type(exc).__name__}"
            )
            fallback_issues = validation_issues or (
                [str(exc)] if isinstance(exc, ValueError) else []
            )
            if fallback_issues:
                fallback.provenance.append(
                    f"{self.name}:validation-fallback:{'; '.join(fallback_issues)}"
                )
            return fallback

    async def implement_detector(
        self,
        project: ResearchProject,
        *,
        hypothesis: Hypothesis,
        name_stem: str,
        reference_description: str | None,
    ) -> MethodImplementation:
        """Ask Qwen for an anomaly-score core function; deterministic fallback on failure."""

        validation_issues: list[str] = []
        try:
            system_prompt = (
                "你是当前研究领域的检测器/评分函数实现专家。"
                f"\n{self._project_context_block(project)}\n\n"
                "为以下假设实现一个新评分逻辑的核心函数，"
                "保持与用户研究领域一致（少样本工业异常检测仍是内置演示场景，"
                "其他领域请按用户提供的 objective 调整）。"
                "优先使用 numpy、math、statistics 等轻量纯计算；不要构造或加载"
                "任何运行时模型、预训练权重或网络资源，不要调用 torchvision/transformers"
                "模型，也不要下载。只允许模块级 import（白名单：math/random/numpy/"
                "scipy/sklearn/PIL/cv2/torch/torchvision/transformers/timm）与顶层普通函数；"
                "辅助函数名不得以下划线开头，不得定义嵌套函数或类；必须恰好包含一个函数"
                "def anomaly_score(image, support_images, seed) -> float（异常/不相似度评分，"
                "分数越高表示越异常；其他领域可保留相同签名）。单图调用必须轻量。"
                "在少样本工业异常检测场景下，分数越高越异常：核心必须计算测试图与正常支持图"
                "之间的非负偏离距离，并直接返回随偏离增大的统计量；禁止对距离取负、取倒数或"
                "转换成相似度。应把支持图缩放到测试图尺寸，计算对齐像素的 RGB 绝对或平方距离，"
                "对多个支持图逐像素取最小距离，再用 95% 到 99% 高分位聚合为图像分数。"
                "禁止只用全图均值、标准差或直方图，因为局部缺陷会被平均掉。"
                "必须保证局部明显颜色/纹理缺陷得到高于正常图的分数。"
                "禁止读写文件、联网、启动子进程、eval/exec、"
                "torch.hub.load/hub.load 下载。固定 seed 必须确定性输出；数据读取、标签"
                "推导、AUROC 计算与输出由系统模板负责，不要重复实现。除代码外所有自然语言"
                "使用简体中文。"
            )
            request_payload = {
                "hypothesis": hypothesis.model_dump(mode="json"),
                "name_stem": name_stem,
                "reference_description": reference_description,
                "function_contract": {
                    "signature": (
                        "def anomaly_score(image, support_images, seed) -> float"
                    ),
                    "image": "numpy HxWx3 uint8 RGB 数组（单张测试图）",
                    "support_images": "numpy 数组列表（预注册的 K 张正常参考图）",
                    "semantics": "返回异常分数，分数越高越异常",
                    "template_provides": [
                        "数据视图读取",
                        "ground_truth 掩码标签推导",
                        "image_auroc/image_ap 计算",
                        "metrics.json 输出",
                    ],
                },
                "output_schema": {
                    "source_code": "模块级 import + 顶层辅助函数 + anomaly_score 的完整源码",
                    "explanation": "方法机制的一句话说明（简体中文）",
                },
                "return": {"source_code": "string"},
            }
            source, validation_issues = await _generate_validated_source(
                client=self.client,
                role_name="DetectorImplementerAgent",
                system_prompt=system_prompt,
                request_payload=request_payload,
                extractor=extract_detector_source,
                validator=validate_detector_source,
                repair_focus="不要引入新的函数、类、导入或外部依赖；",
                failure_label="检测器",
            )
            assembled = assemble_detector_file(source)
            digest = hashlib.sha256(assembled.encode("utf-8")).hexdigest()
            provenance = [self.name, "static_contract:accepted"]
            if validation_issues:
                provenance.append(f"{self.name}:validation-repair")
            return MethodImplementation(
                kind="detector",
                name=implementation_detector_name(name_stem, digest),
                hypothesis_id=hypothesis.id,
                source_code=source,
                code_digest=digest,
                provenance=provenance,
                status="draft",
            )
        except Exception as exc:
            fallback = await super().implement_detector(
                project,
                hypothesis=hypothesis,
                name_stem=name_stem,
                reference_description=reference_description,
            )
            fallback.provenance.append(
                f"{self.name}:deterministic-fallback:{type(exc).__name__}"
            )
            fallback_issues = validation_issues or (
                [str(exc)] if isinstance(exc, ValueError) else []
            )
            if fallback_issues:
                fallback.provenance.append(
                    f"{self.name}:validation-fallback:{'; '.join(fallback_issues)}"
                )
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


def _has_distinct_conditions(contract: dict[str, Any]) -> bool:
    if contract.get("design_mode") == "custom_design":
        return True
    treatment = _comparison_key(contract.get("treatment"))
    control = _comparison_key(contract.get("control"))
    return bool(treatment and control and treatment != control)


def _comparison_key(value: Any) -> str:
    return " ".join(str(value).casefold().replace("_", " ").replace("-", " ").split())
