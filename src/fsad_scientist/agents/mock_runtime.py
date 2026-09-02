from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from fsad_scientist.domain.enums import EvidenceStatus, HypothesisStatus, RunStatus
from fsad_scientist.domain.models import (
    AlternativeDecision,
    AnalysisContract,
    AnalysisFinding,
    ArtifactRecord,
    EvidenceRecord,
    ExpectedImprovement,
    ExperimentCell,
    ExperimentFeedbackProposal,
    ExperimentGuidanceDecision,
    ExperimentPlan,
    ExperimentRun,
    Hypothesis,
    HypothesisScore,
    InnovationCandidate,
    MethodImplementation,
    ReasoningStep,
    ResearchGap,
    ResearchProject,
    new_id,
)
from fsad_scientist.experiments.code_safety import (
    implementation_detector_name,
    sanitize_strategy_name,
)
from fsad_scientist.experiments.detector_runner import assemble_detector_file
from fsad_scientist.experiments.strategy_runner import assemble_strategy_file
from fsad_scientist.science.statistics import compare_paired_runs

MOCK_DETECTOR_SOURCE = (
    "import numpy as np\n"
    "\n"
    "\n"
    "def anomaly_score(image, support_images, seed):\n"
    "    differences = [\n"
    "        image.astype(np.float32) - support.astype(np.float32)\n"
    "        for support in support_images\n"
    "    ]\n"
    "    distances = [\n"
    "        float(np.mean(difference ** 2))\n"
    "        for difference in differences\n"
    "    ]\n"
    "    return min(distances)\n"
)

def _mock_strategy_source(strategy_name: str) -> str:
    normalized = sanitize_strategy_name(strategy_name)
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    prefer_high_norm = not any(
        marker in normalized for marker in ("unconstrained", "fastref", "baseline")
    )
    rotation = int(digest[:8], 16)
    return (
        "def select(candidate_ids, embeddings, k, seed):\n"
        "    norms = {\n"
        "        file_id: sum(value * value for value in embeddings[file_id])\n"
        "        for file_id in candidate_ids\n"
        "    }\n"
        "    ranked = sorted(\n"
        "        candidate_ids,\n"
        "        key=lambda file_id: (norms[file_id], file_id),\n"
        f"        reverse={prefer_high_norm!r},\n"
        "    )\n"
        f"    offset = {rotation} % len(ranked)\n"
        "    rotated = ranked[offset:] + ranked[:offset]\n"
        "    return rotated[:k]\n"
    )


class MockScientistRuntime:
    """Deterministic development runtime.

    It exercises the complete scientific workflow without API keys or invented
    experiment metrics. Every generated research artifact is explicitly marked
    unverified and must later be replaced or checked by real tools.
    """

    name = "mock-scientist-runtime"

    async def formalize_scope(self, project: ResearchProject) -> ArtifactRecord:
        constraints = project.spec.constraints
        return ArtifactRecord(
            kind="research_scope",
            title="结构化研究任务",
            verified=False,
            provenance=["user_scope", self.name],
            payload={
                "problem_statement": (
                    "在仅有极少量正常参考图像、且适配阶段没有真实异常样本时，"
                    "自主发现能够改善工业异常检测性能或稳定性的机制。"
                ),
                "independent_variables": [
                    "参考样本数量 K",
                    "参考样本构成",
                    "检测器结构",
                    "正常类内部变化程度",
                ],
                "dependent_variables": [
                    "图像级检测性能",
                    "像素级定位性能",
                    "跨支持集重采样稳定性",
                ],
                "protocols": [
                    name
                    for enabled, name in [
                        (constraints.strict_k_shot_protocol, "strict_k_shot"),
                        (constraints.pool_compression_protocol, "pool_compression"),
                    ]
                    if enabled
                ],
                "integrity_rules": [
                    "测试异常标签不得用于参考样本选择",
                    "LLM 不得生成实验指标",
                    "创新结论必须链接真实 Run ID",
                ],
            },
        )

    async def gather_evidence(
        self, project: ResearchProject
    ) -> tuple[list[EvidenceRecord], ArtifactRecord]:
        evidence = [
            EvidenceRecord(
                title="Towards Total Recall in Industrial Anomaly Detection",
                source_type="paper",
                url="https://arxiv.org/abs/2106.08265",
                arxiv_id="2106.08265",
                claims=["PatchCore 是局部 patch 记忆库类基线。"],
                status=EvidenceStatus.UNVERIFIED,
            ),
            EvidenceRecord(
                title="AnomalyDINO: Boosting Patch-based Few-shot Anomaly Detection with DINOv2",
                source_type="paper",
                url="https://arxiv.org/abs/2405.14529",
                arxiv_id="2405.14529",
                claims=["冻结 DINOv2 patch 特征可用于训练自由少样本异常检测。"],
                status=EvidenceStatus.UNVERIFIED,
            ),
            EvidenceRecord(
                title="SubspaceAD: Training-Free Few-Shot Anomaly Detection via Subspace Modeling",
                source_type="paper",
                url="https://arxiv.org/abs/2602.23013",
                arxiv_id="2602.23013",
                claims=["正常 patch 特征可通过子空间重建残差进行异常评分。"],
                status=EvidenceStatus.UNVERIFIED,
            ),
            EvidenceRecord(
                title=(
                    "FastRef: Fast Prototype Refinement for Few-shot Industrial "
                    "Anomaly Detection"
                ),
                source_type="paper",
                url="https://arxiv.org/abs/2506.21398",
                arxiv_id="2506.21398",
                claims=["查询图像统计可在测试时修正少样本正常原型。"],
                status=EvidenceStatus.UNVERIFIED,
            ),
        ]
        artifact = ArtifactRecord(
            kind="evidence_search_plan",
            title="文献检索与证据校验计划",
            verified=False,
            provenance=[self.name],
            payload={
                "queries": [
                    "few-shot industrial anomaly detection support set selection",
                    "normal reference diversity anomaly detection stability",
                    "subspace modeling few-shot anomaly detection",
                    "query adaptive prototype anomaly detection",
                ],
                "required_checks": [
                    "题目、作者、DOI 或 arXiv ID 一致",
                    "结论必须定位到原文段落或表格",
                    "与研究空白相关的负面结果也必须保留",
                ],
            },
        )
        return evidence, artifact

    async def discover_gaps(self, project: ResearchProject) -> list[ResearchGap]:
        evidence_ids = [item.id for item in project.evidence]
        return [
            ResearchGap(
                title="参考集质量与稳定性缺少系统研究",
                description=(
                    "现有比较通常关注平均性能，较少把支持集重采样造成的方差和"
                    "最坏情况表现作为主要研究对象。"
                ),
                why_unresolved="需要大规模配对重采样实验，而不是单次固定支持集。",
                evidence_ids=evidence_ids,
                expected_scientific_value=0.90,
                estimated_cost=0.45,
                status="selected",
            ),
            ResearchGap(
                title="代表性的定义可能依赖检测器结构",
                description=(
                    "近邻记忆模型关注局部覆盖，子空间模型关注有效秩；统一的样本"
                    "选择目标可能并不适用于所有检测器。"
                ),
                why_unresolved="需要 strategy × detector 的交互实验和机制指标。",
                evidence_ids=evidence_ids,
                expected_scientific_value=0.86,
                estimated_cost=0.55,
                status="selected",
            ),
            ResearchGap(
                title="测试时信息能否抵消劣质参考集",
                description="查询自适应原型可能降低模型对初始参考图像构成的敏感性。",
                why_unresolved="需要把 FastRef 类修正与多种参考集选择方法组合比较。",
                evidence_ids=evidence_ids,
                expected_scientific_value=0.78,
                estimated_cost=0.68,
            ),
            ResearchGap(
                title="从被动选样本扩展到主动采集",
                description="系统可以根据当前未覆盖区域决定下一张最值得采集的正常图像。",
                why_unresolved="公开基准通常没有连续采集动作和采集成本标签。",
                evidence_ids=evidence_ids,
                expected_scientific_value=0.92,
                estimated_cost=0.90,
            ),
        ]

    async def propose_hypotheses(self, project: ResearchProject) -> list[Hypothesis]:
        gaps = {gap.title: gap for gap in project.gaps}
        evidence_ids = [item.id for item in project.evidence]
        return [
            Hypothesis(
                gap_id=gaps["参考集质量与稳定性缺少系统研究"].id,
                title="覆盖感知选样优先改善最坏情况稳定性",
                claim=(
                    "当 K≤4 时，k-center 正常参考集相较随机参考集对平均 AUROC 的"
                    "提升可能有限，但会显著降低跨支持集重采样方差和最坏十分位损失。"
                ),
                null_hypothesis=(
                    "在相同 K 和候选池下，k-center 与随机选择的稳定性指标没有差异。"
                ),
                rationale="极少样本下，未覆盖的正常模式会造成随机且严重的假阳性。",
                independent_variables=["K", "选择策略", "正常特征覆盖半径"],
                dependent_variables=["AUROC", "AUPRO", "跨重采样标准差", "最坏十分位"],
                predicted_direction="覆盖半径减小，方差和最坏情况损失下降。",
                falsification_conditions=[
                    "配对置信区间包含零且效应量可忽略",
                    "收益不能在至少三个类别上复现",
                ],
                evidence_ids=evidence_ids,
                analysis_contract=AnalysisContract(
                    kind="selection_main_effect",
                    metric="image_auroc",
                    treatment="k_center",
                    control="random",
                    minimum_pairs=6,
                ),
            ),
            Hypothesis(
                gap_id=gaps["代表性的定义可能依赖检测器结构"].id,
                title="覆盖选样收益具有结构类与纹理类边界",
                claim=(
                    "k-center 相对 random 的 Image AUROC 收益在结构异常类别上高于"
                    "纹理异常类别，且该差异可由正常特征覆盖半径解释。"
                ),
                null_hypothesis="k-center 的成对收益在结构类和纹理类之间没有差异。",
                rationale="局部几何覆盖对结构变化和重复纹理的作用机制可能不同。",
                independent_variables=["类别类型", "选择策略", "覆盖半径", "K"],
                dependent_variables=["Image AUROC", "Pixel AUROC", "AUPRO"],
                predicted_direction="结构异常类别获得更高的成对收益。",
                falsification_conditions=[
                    "类别组间效应差异的置信区间包含零",
                    "差异不能跨至少三个类别复现",
                ],
                evidence_ids=evidence_ids,
                analysis_contract=AnalysisContract(
                    kind="selection_main_effect",
                    metric="image_auroc",
                    treatment="k_center",
                    control="random",
                    minimum_pairs=12,
                ),
            ),
            Hypothesis(
                gap_id=gaps["测试时信息能否抵消劣质参考集"].id,
                title="覆盖感知选样的收益随 K 增大而衰减",
                claim=(
                    "k-center 相对 random 的收益在 K=1/2 时最大，并随 K 增加到 4/8"
                    "而显著衰减。"
                ),
                null_hypothesis="k-center 与 random 的成对效应不随 K 改变。",
                rationale="极低 K 下正常模式遗漏最严重，代表性选样的边际价值应更高。",
                independent_variables=["K", "选择策略", "类别"],
                dependent_variables=["Image AUROC", "跨 seed 方差"],
                predicted_direction="选择策略与 K 呈负向交互。",
                falsification_conditions=["成对收益不随 K 增大而下降"],
                evidence_ids=evidence_ids,
                analysis_contract=AnalysisContract(
                    kind="selection_main_effect",
                    metric="image_auroc",
                    treatment="k_center",
                    control="random",
                    minimum_pairs=6,
                ),
            ),
            Hypothesis(
                gap_id=gaps["测试时信息能否抵消劣质参考集"].id,
                title="查询自适应原型降低初始支持集敏感度",
                claim=(
                    "测试时原型修正对随机或低覆盖参考集的收益高于对高覆盖参考集的收益。"
                ),
                null_hypothesis="原型修正收益与初始参考集覆盖质量无关。",
                rationale="查询中的正常区域可以补足初始参考原型未覆盖的外观变化。",
                independent_variables=["原型修正", "参考集覆盖质量", "K"],
                dependent_variables=["AUROC", "AUPRO", "方差"],
                predicted_direction="低覆盖参考集获得更大的修正收益。",
                falsification_conditions=["修正收益不随参考集覆盖质量变化"],
                evidence_ids=evidence_ids,
                analysis_contract=AnalysisContract(
                    kind="query_adaptation",
                    metric="image_auroc",
                    treatment="query_adaptive",
                    control="no_adaptation",
                    minimum_pairs=6,
                ),
            ),
        ]

    async def review_hypotheses(self, project: ResearchProject) -> list[Hypothesis]:
        score_values = [
            (0.84, 0.96, 0.91, 0.88, 0.45, 1180.0),
            (0.90, 0.91, 0.80, 0.92, 0.42, 1168.0),
            (0.80, 0.86, 0.61, 0.83, 0.38, 1008.0),
        ]
        reviewed: list[Hypothesis] = []
        for index, hypothesis in enumerate(project.hypotheses):
            values = score_values[min(index, len(score_values) - 1)]
            updated = hypothesis.model_copy(deep=True)
            updated.score = HypothesisScore(
                novelty=values[0],
                falsifiability=values[1],
                feasibility=values[2],
                scientific_value=values[3],
                evidence_strength=values[4],
                elo=values[5],
            )
            updated.status = (
                HypothesisStatus.SHORTLISTED if index < 2 else HypothesisStatus.CANDIDATE
            )
            reviewed.append(updated)
        return sorted(
            reviewed,
            key=lambda item: item.score.elo if item.score else 0,
            reverse=True,
        )

    async def design_experiments(self, project: ResearchProject) -> ExperimentPlan:
        registered = [
            item
            for item in project.method_implementations
            if item.status in {"validated", "approved"}
        ]
        hypothesis_contracts = {
            item.id: item.analysis_contract.model_dump(mode="json")
            for item in project.hypotheses
            if item.status == HypothesisStatus.SHORTLISTED
            and item.analysis_contract is not None
        }
        payload = {
            "hypothesis_ids": [
                item.id
                for item in project.hypotheses
                if (
                    item.execution_readiness == "executable"
                    or any(
                        implementation.hypothesis_id == item.id
                        and implementation.kind == "selection_strategy"
                        for implementation in registered
                    )
                )
                and item.user_selected is not False
                and item.status
                in {HypothesisStatus.SHORTLISTED, HypothesisStatus.CANDIDATE}
            ],
            "hypothesis_contracts": hypothesis_contracts,
            "protocols": ["strict_k_shot", "pool_compression_m30"],
            "detectors": [
                "patchcore",
                "anomalydino",
                "subspacead",
                *[item.name for item in registered if item.kind == "detector"],
            ],
            "selection_strategies": [
                "random",
                "k_center",
                *[item.name for item in registered if item.kind == "selection_strategy"],
            ],
            "method_implementation_digests": {
                item.name: item.code_digest for item in registered
            },
            "datasets": ["MVTec AD", "VisA"],
            "categories": ["bottle", "carpet", "capsule", "cable", "transistor"],
            "shots": project.spec.constraints.shots,
            "seeds": list(range(10)),
            "metrics": [
                "image_auroc",
                "image_ap",
                "pixel_auroc",
                "aupro",
                "support_set_std",
                "worst_decile",
                "coverage_radius",
                "effective_rank",
            ],
            "analysis_methods": [
                "paired_bootstrap_95ci",
                "paired_permutation_test",
                "strategy_x_k_x_detector_interaction",
                "category_level_replication",
            ],
            "stages": [
                "feasibility",
                "sensitivity",
                "main_factorial_study",
                "replication",
                "ablation",
                "cross_dataset_validation",
            ],
            "stopping_conditions": [
                "GPU 或实验次数预算耗尽",
                "假设已稳定证伪",
                "后续候选实验的信息增益/成本低于阈值",
            ],
            "estimated_gpu_hours": min(project.spec.budget.gpu_hours, 12.0),
        }
        digest = hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return ExperimentPlan(**payload, preregistration_digest=digest)

    async def recommend_next_experiments(
        self,
        project: ResearchProject,
        *,
        round_summary: dict[str, Any],
        allowed_cells: list[ExperimentCell],
        user_guidance: str | None = None,
    ) -> ExperimentFeedbackProposal:
        """Deterministic fallback for the result-to-next-plan decision."""

        pair_count = int(round_summary.get("pair_count", 0))
        minimum_pairs = int(round_summary.get("minimum_pairs", 6))
        failed = list(round_summary.get("failed_run_ids", []))
        mean_difference = round_summary.get("mean_difference")
        positive_fraction = round_summary.get("positive_pair_fraction")
        primary_saturated = bool(round_summary.get("primary_metric_saturated", False))

        reasoning_chain: list[ReasoningStep] = []
        alternative_decisions: list[AlternativeDecision] = []

        if failed or pair_count == 0:
            decision = "diagnose"
            phase = "sensitivity"
            rationale = (
                "当前轮存在失败运行或尚未形成有效成对结果；下一轮优先选择最小诊断单元，"
                "区分执行链路问题、类别边界和 K 敏感性。"
            )
            reasoning_chain.append(ReasoningStep(
                step=1,
                observation=f"有效成对单元数 = {pair_count}, 失败运行数 = {len(failed)}",
                conclusion="样本不足或存在执行问题，需要诊断",
                confidence="高",
            ))
            alternative_decisions.append(AlternativeDecision(
                decision="expand",
                rejected_reason="在失败运行原因未知的情况下扩展实验会增加风险",
            ))
        elif pair_count >= minimum_pairs and positive_fraction is not None:
            decision = "stop"
            phase = "complete"
            rationale = (
                "已达到预注册的最小成对样本数，当前证据可以进入正式配对统计与创新审查。"
            )
            reasoning_chain.append(ReasoningStep(
                step=1,
                observation=f"配对数 {pair_count} >= 最小要求 {minimum_pairs}",
                conclusion="已达到统计检验所需的最小样本量",
                confidence="高",
            ))
            reasoning_chain.append(ReasoningStep(
                step=2,
                observation=f"正向配对比例 = {positive_fraction:.2%}",
                conclusion="效应方向一致，可以得出结论",
                confidence="高" if positive_fraction >= 0.8 else "中",
            ))
        elif primary_saturated:
            decision = "expand"
            phase = "main_study"
            rationale = (
                "预注册主指标在当前类别上饱和，零差异不能区分两种策略；"
                "扩展到更难类别，同时保留像素指标作为机制和边界诊断。"
            )
            reasoning_chain.append(ReasoningStep(
                step=1,
                observation="主指标已饱和（所有值 >= 0.995）",
                conclusion="当前类别无法区分策略效果",
                confidence="高",
            ))
            reasoning_chain.append(ReasoningStep(
                step=2,
                observation="需要测试更难类别或改变检测器",
                conclusion="扩展到新类别以寻找有效区分区域",
                confidence="中",
            ))
            alternative_decisions.append(AlternativeDecision(
                decision="stop",
                rejected_reason="指标饱和时无法得出有意义的结论",
            ))
        elif mean_difference is not None and mean_difference > 0 and (
            positive_fraction or 0
        ) >= 0.75:
            decision = "expand"
            phase = "replication"
            rationale = (
                "首批效应方向较一致但证据量仍不足；扩展到新类别，以检验收益是否具有跨类别复现性。"
            )
            reasoning_chain.append(ReasoningStep(
                step=1,
                observation=f"平均效应量 = {mean_difference:.4f} > 0",
                conclusion="处理组（k_center）优于对照组（random）",
                confidence="高",
            ))
            reasoning_chain.append(ReasoningStep(
                step=2,
                observation=f"正向配对比例 = {positive_fraction:.2%} >= 75%",
                conclusion="效应在多个独立单元上复现",
                confidence="高",
            ))
            reasoning_chain.append(ReasoningStep(
                step=3,
                observation=f"但配对数 {pair_count} < 最小要求 {minimum_pairs}",
                conclusion="样本量不足以得出强结论，需要扩展",
                confidence="中",
            ))
        elif mean_difference is not None and mean_difference <= 0:
            decision = "diagnose"
            phase = "sensitivity"
            rationale = (
                "观察到零效应或反向效应；改变类别或 K 做边界诊断，避免在无效区域盲目扩大计算。"
            )
            reasoning_chain.append(ReasoningStep(
                step=1,
                observation=f"平均效应量 = {mean_difference:.4f} <= 0",
                conclusion="处理组未优于对照组，需要诊断原因",
                confidence="高",
            ))
            reasoning_chain.append(ReasoningStep(
                step=2,
                observation="可能是类别特性、K值或执行问题",
                conclusion="需要进行诊断实验确定原因",
                confidence="中",
            ))
            alternative_decisions.append(AlternativeDecision(
                decision="expand",
                rejected_reason="在效应为负的情况下扩展会增加计算浪费",
            ))
        else:
            decision = "replicate"
            phase = "replication"
            rationale = (
                "当前成对效应不稳定且样本不足；增加独立单元以判断波动来自随机种子还是类别差异。"
            )
            reasoning_chain.append(ReasoningStep(
                step=1,
                observation=f"配对数 = {pair_count}, 效应量 = {mean_difference}",
                conclusion="样本量不足以判断效应稳定性",
                confidence="中",
            ))
            reasoning_chain.append(ReasoningStep(
                step=2,
                observation=f"正向比例 = {positive_fraction}",
                conclusion="需要更多独立样本来验证效应一致性",
                confidence="中",
            ))

        expected_improvement: ExpectedImprovement | None = None
        if mean_difference is not None and mean_difference > 0:
            expected_improvement = ExpectedImprovement(
                metric="image_auroc",
                direction="increase",
                estimated_delta=mean_difference * 0.2,
                confidence="中",
            )

        guidance_note = (user_guidance or "").strip()
        guided_cells = list(allowed_cells)
        if guidance_note and guided_cells:
            normalized_guidance = guidance_note.casefold()

            def guidance_score(cell: ExperimentCell) -> int:
                score = 0
                category = cell.category.casefold()
                if category in normalized_guidance:
                    score += 4
                if (
                    f"k={cell.shots}" in normalized_guidance
                    or f"k {cell.shots}" in normalized_guidance
                    or f"{cell.shots}-shot" in normalized_guidance
                ):
                    score += 2
                if f"seed={cell.seed}" in normalized_guidance:
                    score += 1
                return score

            guided_cells = [
                cell
                for _, cell in sorted(
                    enumerate(guided_cells),
                    key=lambda item: (-guidance_score(item[1]), item[0]),
                )
            ]
            rationale += (
                " 已记录用户指导，并在不改变预注册空间的前提下用于后续迭代排序："
                f"{guidance_note[:120]}"
            )

        return ExperimentFeedbackProposal(
            advisor=self.name,
            decision=decision,
            rationale=rationale,
            reasoning_chain=reasoning_chain,
            alternative_decisions=alternative_decisions,
            expected_improvement=expected_improvement,
            observed_patterns=[
                f"有效成对单元 {pair_count}/{minimum_pairs}",
                f"平均处理效应 {mean_difference}",
                f"同向比例 {positive_fraction}",
                f"失败运行 {len(failed)}",
                f"主指标饱和 {primary_saturated}",
            ],
            next_phase=phase,
            recommended_cells=[] if decision == "stop" else guided_cells[:2],
            strategy_adjustment={},
            expected_information_gain=0.85 if decision == "diagnose" else 0.72,
            stop=decision == "stop",
        )

    async def interpret_experiment_guidance(
        self,
        project: ResearchProject,
        *,
        guidance: str,
        candidate_runs: list[ExperimentRun],
    ) -> ExperimentGuidanceDecision:
        """Interpret advice as priority only; never mutate a preregistered run."""

        if not candidate_runs:
            raise ValueError("No queued experiment is available for guidance")
        normalized = guidance.casefold().strip()
        requested_shots = {
            int(value)
            for value in re.findall(r"(?:^|\s)k\s*[=:：]?\s*(\d+)", normalized)
        }
        requested_seeds = {
            int(value)
            for value in re.findall(r"(?:seed|随机种子)\s*[=:：]?\s*(\d+)", normalized)
        }

        ranked: list[tuple[int, int, ExperimentRun, list[str]]] = []
        for index, run in enumerate(candidate_runs):
            score = 0
            matches: list[str] = []
            if run.id.casefold() in normalized:
                score += 100
                matches.append("指定 Run ID")
            if run.category.casefold() in normalized:
                negative = re.search(
                    rf"(?:不要|跳过|避开|暂不).{{0,8}}{re.escape(run.category.casefold())}",
                    normalized,
                )
                score += -40 if negative else 25
                matches.append(f"类别 {run.category}")
            if requested_shots and run.shots in requested_shots:
                score += 18
                matches.append(f"K={run.shots}")
            if requested_seeds and run.seed in requested_seeds:
                score += 12
                matches.append(f"seed={run.seed}")
            strategy_aliases = {
                "k_center": ("k-center", "k_center", "代表性", "覆盖选样"),
                "random": ("random", "随机选样", "随机策略"),
            }
            if any(
                marker in normalized
                for marker in strategy_aliases.get(run.selection_strategy, ())
            ):
                score += 10
                matches.append(f"策略 {run.selection_strategy}")
            ranked.append((score, -index, run, matches))

        best_score, _, selected, matches = max(ranked, key=lambda item: (item[0], item[1]))
        default_markers = ("按预注册", "按系统", "默认顺序", "无额外", "不做额外")
        if best_score > 0:
            disposition = "applied"
            interpretation = "用户希望优先执行：" + "、".join(matches)
            rationale = "该建议可通过调整当前已冻结队列的执行顺序落实。"
        elif any(marker in normalized for marker in default_markers):
            disposition = "applied"
            interpretation = "用户授权按预注册优先级执行，不请求改变实验配置。"
            rationale = "采用实验树中信息增益/成本优先级最高的待运行任务。"
        else:
            disposition = "not_applicable"
            interpretation = "当前指导未能映射到本轮任何已预注册的待运行任务。"
            rationale = "系统保留该意见并按确定性优先级执行，配置变更需进入下一轮预注册。"

        return ExperimentGuidanceDecision(
            advisor=self.name,
            selected_run_id=selected.id,
            interpretation=interpretation,
            disposition=disposition,
            rationale=rationale,
            execution_notes=[
                f"本次执行锁定为 {selected.category} / K={selected.shots} / "
                f"seed={selected.seed} / {selected.selection_strategy}。"
            ],
            protected_constraints=[
                "不改变预注册的类别、K、seed、检测器、策略和评价指标",
                "不允许测试异常标签进入支持集选择",
                "用户原文与 AI 解释写入 Research Ledger",
            ],
        )

    async def implement_selection_strategy(
        self,
        project: ResearchProject,
        *,
        hypothesis: Hypothesis,
        strategy_name: str,
        control_name: str,
    ) -> MethodImplementation:
        """Deterministic stub: a valid, distinct-from-builtin coverage strategy.

        The registered name equals the sanitized contract strategy name so the
        plan approval gate can match implementations to hypothesis contracts.
        """

        source_code = _mock_strategy_source(strategy_name)
        assembled = assemble_strategy_file(source_code)
        digest = hashlib.sha256(assembled.encode("utf-8")).hexdigest()
        return MethodImplementation(
            kind="selection_strategy",
            name=sanitize_strategy_name(strategy_name),
            hypothesis_id=hypothesis.id,
            source_code=source_code,
            code_digest=digest,
            provenance=[
                self.name,
                f"deterministic-stub-for:{strategy_name}",
                f"control:{control_name}",
            ],
            status="draft",
        )

    async def implement_detector(
        self,
        project: ResearchProject,
        *,
        hypothesis: Hypothesis,
        name_stem: str,
        reference_description: str | None,
    ) -> MethodImplementation:
        """Deterministic stub: a real numpy nearest-pixel-distance detector."""

        assembled = assemble_detector_file(MOCK_DETECTOR_SOURCE)
        digest = hashlib.sha256(assembled.encode("utf-8")).hexdigest()
        provenance = [self.name, f"deterministic-stub-for:{name_stem}"]
        if reference_description:
            provenance.append(f"reference:{reference_description}")
        return MethodImplementation(
            kind="detector",
            name=implementation_detector_name(name_stem, digest),
            hypothesis_id=hypothesis.id,
            source_code=MOCK_DETECTOR_SOURCE,
            code_digest=digest,
            provenance=provenance,
            status="draft",
        )

    async def analyze_results(self, project: ResearchProject) -> list[AnalysisFinding]:
        runs = [run for run in project.runs if run.status == RunStatus.SUCCEEDED and run.verified]
        if not runs:
            raise ValueError("No verified experiment results are available")

        findings: list[AnalysisFinding] = []
        for hypothesis in project.hypotheses:
            contract = hypothesis.analysis_contract
            if contract is None or contract.kind not in {
                "selection_main_effect",
                "query_adaptation",
            }:
                findings.append(
                    AnalysisFinding(
                        hypothesis_id=hypothesis.id,
                        statement="当前实验批次未直接识别该假设预注册的因果量。",
                        boundary_conditions=["需要执行与 analysis_contract 匹配的实验节点"],
                        claim_verdict="not_tested",
                        verified=False,
                    )
                )
                continue
            try:
                comparison = compare_paired_runs(
                    runs,
                    hypothesis_id=hypothesis.id,
                    metric=contract.metric,
                    treatment=contract.treatment,
                    control=contract.control,
                    alpha=contract.alpha,
                )
            except ValueError:
                findings.append(
                    AnalysisFinding(
                        hypothesis_id=hypothesis.id,
                        statement="尚无足够的预注册成对真实结果，当前证据不足。",
                        boundary_conditions=["需要相同数据、类别、检测器、K 和 seed 的配对运行"],
                        # 配对数不足时标记为 not_tested，避免触发无限修订循环。
                        claim_verdict="not_tested",
                        verified=False,
                    )
                )
                continue

            lower, upper = comparison.confidence_interval
            enough_pairs = comparison.pair_count >= contract.minimum_pairs
            if enough_pairs and lower > 0 and comparison.permutation_p_value < contract.alpha:
                verdict = "supported"
            elif enough_pairs and upper <= 0:
                verdict = "rejected"
            elif not enough_pairs:
                # 配对数不足以满足预注册要求时标记为 not_tested，避免触发无限修订循环
                verdict = "not_tested"
            else:
                verdict = "inconclusive"
            supporting_ids = [
                run_id
                for difference, pair in zip(
                    comparison.differences,
                    comparison.pair_run_ids,
                    strict=True,
                )
                if difference > 0
                for run_id in pair
            ]
            contradicting_ids = [
                run_id
                for difference, pair in zip(
                    comparison.differences,
                    comparison.pair_run_ids,
                    strict=True,
                )
                if difference <= 0
                for run_id in pair
            ]
            findings.append(
                AnalysisFinding(
                    hypothesis_id=hypothesis.id,
                    statement=(
                        f"{contract.treatment} 相对 {contract.control} 的配对 "
                        f"{contract.metric} 差异已用 bootstrap 与符号置换检验计算。"
                    ),
                    effect_size=comparison.mean_difference,
                    confidence_interval=comparison.confidence_interval,
                    p_value=comparison.permutation_p_value,
                    sample_size=comparison.pair_count,
                    analysis_method="paired_bootstrap_95ci+paired_sign_permutation",
                    claim_verdict=verdict,
                    supporting_run_ids=sorted(set(supporting_ids)),
                    contradicting_run_ids=sorted(set(contradicting_ids)),
                    boundary_conditions=["当前结论仅覆盖已导入且标记 verified 的配对运行"],
                    verified=enough_pairs,
                )
            )
        return findings

    async def review_innovations(
        self, project: ResearchProject
    ) -> list[InnovationCandidate]:
        hypotheses = {item.id: item for item in project.hypotheses}
        evidence_by_id = {item.id: item for item in project.evidence}
        candidates: list[InnovationCandidate] = []
        for finding in project.findings:
            hypothesis = hypotheses[finding.hypothesis_id]
            linked_evidence = [
                evidence_by_id[evidence_id]
                for evidence_id in hypothesis.evidence_ids
                if evidence_id in evidence_by_id
            ]
            required_verified = min(2, len(linked_evidence))
            evidence_is_verified = required_verified > 0 and sum(
                item.status == EvidenceStatus.VERIFIED
                and item.verification_scope == "claim"
                for item in linked_evidence
            ) >= required_verified
            supported = (
                finding.verified
                and finding.claim_verdict == "supported"
                and finding.effect_size is not None
                and finding.confidence_interval is not None
                and finding.effect_size > 0
                and finding.confidence_interval[0] > 0
            )
            status = (
                "evidence_supported_candidate"
                if supported and evidence_is_verified
                else "unverified_candidate"
            )
            candidates.append(
                InnovationCandidate(
                    hypothesis_id=hypothesis.id,
                    title=hypothesis.title,
                    core_finding=finding.statement,
                    difference_from_prior_work=(
                        "重点从固定支持集的平均精度转向参考集几何、检测器结构与"
                        "跨支持集稳定性的交互；需由真实检索工具确认最近工作。"
                    ),
                    mechanism_evidence=[hypothesis.predicted_direction],
                    supporting_finding_ids=[finding.id] if finding.verified else [],
                    boundary_conditions=finding.boundary_conditions,
                    reproducibility_evidence=finding.supporting_run_ids,
                    confidence="medium" if supported else "low",
                    status=status,
                )
            )
        return candidates

    async def revise_hypotheses(self, project: ResearchProject) -> list[Hypothesis]:
        findings = {item.hypothesis_id: item for item in project.findings}
        cycle_guidance = next(
            (
                item
                for item in reversed(project.guidance_records)
                if item.scope == "research_cycle"
                and item.research_cycle == project.research_cycle
            ),
            None,
        )
        revised: list[Hypothesis] = []
        for hypothesis in project.hypotheses:
            finding = findings.get(hypothesis.id)
            if finding is None or finding.claim_verdict not in {"rejected", "inconclusive"}:
                continue
            updated = hypothesis.model_copy(
                deep=True,
                update={
                    "score": None,
                    "status": HypothesisStatus.CANDIDATE,
                    "revision": hypothesis.revision + 1,
                    "parent_hypothesis_id": hypothesis.id,
                },
            )
            updated.id = new_id("hypothesis")
            updated.title = f"{hypothesis.title}（修订 r{updated.revision}）"
            updated.claim = (
                f"在上一轮边界条件内重新检验：{hypothesis.claim} "
                "主效应改为优先考察跨支持集方差和最坏十分位，而非只看均值。"
                + (
                    f" 用户进一步要求下一循环重点考虑：{cycle_guidance.text}"
                    if cycle_guidance is not None
                    else ""
                )
            )
            updated.rationale = (
                f"上一轮 verdict={finding.claim_verdict}，n={finding.sample_size}；"
                "修订缩小结论范围并提高对稳定性指标的优先级。"
                + (
                    " 该修订同时纳入了人工指导，并仍需通过新的预注册计划接受审查。"
                    if cycle_guidance is not None
                    else ""
                )
            )
            updated.falsification_conditions = [
                *hypothesis.falsification_conditions,
                "修订后预注册的稳定性主终点仍未达到最小效应或重复要求",
            ]
            revised.append(updated)
        return revised

    async def build_report_manifest(self, project: ResearchProject) -> ArtifactRecord:
        return ArtifactRecord(
            kind="report_manifest",
            title="赛题研究报告导出清单",
            verified=all(item.verified for item in project.findings) and bool(project.findings),
            provenance=[self.name, *[run.id for run in project.runs if run.verified]],
            payload={
                "required_sections": [
                    "Problem Statement",
                    "Rationale",
                    "Technical Details",
                    "Datasets",
                    "Paper Title",
                    "Abstract",
                    "Methods",
                    "Experiments",
                    "Results",
                    "Innovation Candidates",
                    "Limitations",
                    "References",
                    "Reproduction Instructions",
                ],
                "counts": {
                    "evidence": len(project.evidence),
                    "hypotheses": len(project.hypotheses),
                    "runs": len(project.runs),
                    "verified_runs": sum(run.verified for run in project.runs),
                    "findings": len(project.findings),
                    "innovations": len(project.innovations),
                },
            },
        )
