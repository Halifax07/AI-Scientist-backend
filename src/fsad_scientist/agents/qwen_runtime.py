from __future__ import annotations

import hashlib
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
    MethodImplementation,
    ResearchGap,
    ResearchProject,
    new_id,
    ReasoningStep,
    AlternativeDecision,
    ExpectedImprovement,
)
from fsad_scientist.experiments.code_safety import (
    extract_detector_source,
    extract_select_function,
    implementation_detector_name,
    sanitize_strategy_name,
    validate_detector_source,
    validate_strategy_source,
)
from fsad_scientist.experiments.detector_runner import assemble_detector_file
from fsad_scientist.experiments.strategy_runner import assemble_strategy_file


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
                    "你的输出必须包含完整的推理过程，让非专业用户也能理解决策逻辑。\n\n"
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
                    "4. pair_count/cumulative_pair_count 是全活动累计配对数，round_pair_count 是本轮新增数\n"
                    "5. mean_difference/positive_pair_fraction 仅描述本轮；跨轮总体方向必须读取 cumulative_primary_summary\n"
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
                "你是少样本异常检测的支持集选样策略实现专家。"
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
            response = await self.client.complete(
                role_name="MethodImplementerAgent",
                system_prompt=system_prompt,
                payload=request_payload,
            )
            source = extract_select_function(str(response.get("source_code", "")))
            validation = validate_strategy_source(source)
            if not validation.passed:
                validation_issues = validation.issues
                response = await self.client.complete(
                    role_name="MethodImplementerAgent",
                    system_prompt=(
                        system_prompt
                        + "上一版源码未通过静态校验。请只修复下列问题，保持策略语义不变；"
                        "仍然只返回完整的 def select 函数源码，不要返回解释或 Markdown。"
                    ),
                    payload={
                        **request_payload,
                        "previous_source_code": source,
                        "validation_issues": validation.issues,
                        "repair_instruction": "修复上一版源码的全部静态校验问题。",
                    },
                )
                source = extract_select_function(str(response.get("source_code", "")))
                validation = validate_strategy_source(source)
                if not validation.passed:
                    validation_issues = validation.issues
                    raise ValueError(
                        "定向修复后的选样策略仍未通过静态校验："
                        + "；".join(validation.issues)
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
            if validation_issues:
                fallback.provenance.append(
                    f"{self.name}:validation-fallback:{'; '.join(validation_issues)}"
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
                "你是工业异常检测方法实现专家。为以下假设实现一个新检测器的核心打分"
                "逻辑。优先使用 numpy、math、statistics 等轻量纯计算；不要构造或加载"
                "任何运行时模型、预训练权重或网络资源，不要调用 torchvision/transformers"
                "模型，也不要下载。只允许模块级 import（白名单：math/random/numpy/"
                "scipy/sklearn/PIL/cv2/torch/torchvision/transformers/timm）与顶层普通函数；"
                "辅助函数名不得以下划线开头，不得定义嵌套函数或类；必须恰好包含一个函数"
                "def anomaly_score(image, support_images, seed) -> float，单图调用必须轻量。"
                "分数越高表示越异常。禁止读写文件、联网、启动子进程、eval/exec、"
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
            response = await self.client.complete(
                role_name="DetectorImplementerAgent",
                system_prompt=system_prompt,
                payload=request_payload,
            )
            source = extract_detector_source(str(response.get("source_code", "")))
            validation = validate_detector_source(source)
            if not validation.passed:
                validation_issues = validation.issues
                response = await self.client.complete(
                    role_name="DetectorImplementerAgent",
                    system_prompt=(
                        system_prompt
                        + "上一版源码未通过静态校验。请只修复下列问题，保持检测器语义不变；"
                        "仍然只返回完整源码，不要返回解释或 Markdown。"
                    ),
                    payload={
                        **request_payload,
                        "previous_source_code": source,
                        "validation_issues": validation.issues,
                        "repair_instruction": "修复上一版源码的全部静态校验问题。",
                    },
                )
                source = extract_detector_source(str(response.get("source_code", "")))
                validation = validate_detector_source(source)
                if not validation.passed:
                    validation_issues = validation.issues
                    raise ValueError(
                        "定向修复后的检测器仍未通过静态校验："
                        + "；".join(validation.issues)
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
            if validation_issues:
                fallback.provenance.append(
                    f"{self.name}:validation-fallback:{'; '.join(validation_issues)}"
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
