from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ToolCapability:
    name: str
    owner: str
    purpose: str
    deterministic: bool
    implementation_status: str
    input_contract: str
    output_contract: str


class ScientificToolRegistry:
    """Authoritative tool boundary exposed to reasoning agents."""

    def __init__(self) -> None:
        self._tools = {
            item.name: item
            for item in [
                ToolCapability(
                    name="search_papers",
                    owner="evidence_service",
                    purpose="检索论文元数据和原文候选",
                    deterministic=False,
                    implementation_status="implemented",
                    input_contract="query, date_range, domains",
                    output_contract="EvidenceRecord[] (bibliographic scope)",
                ),
                ToolCapability(
                    name="verify_reference",
                    owner="evidence_service",
                    purpose="校验 DOI/arXiv、题目、作者和原文主张",
                    deterministic=True,
                    implementation_status="implemented_requires_qwen_for_claims",
                    input_contract="EvidenceRecord",
                    output_contract="EvidenceRecord + page-anchored ClaimVerification[]",
                ),
                ToolCapability(
                    name="profile_dataset",
                    owner="data_service",
                    purpose="生成数据清单、DINO 特征和正常变化指标",
                    deterministic=True,
                    implementation_status="implemented",
                    input_contract="DatasetSpec",
                    output_contract="dataset card + feature manifest",
                ),
                ToolCapability(
                    name="select_support_set",
                    owner="selection_service",
                    purpose="按预注册策略冻结参考样本清单",
                    deterministic=True,
                    implementation_status="implemented",
                    input_contract="candidate manifest, K, strategy, seed",
                    output_contract="SupportSetManifest",
                ),
                ToolCapability(
                    name="submit_experiment",
                    owner="experiment_service",
                    purpose="在隔离环境中执行异常检测实验",
                    deterministic=True,
                    implementation_status="implemented",
                    input_contract="ExperimentRun + SupportSetManifest",
                    output_contract="metrics.json + artifacts + provenance",
                ),
                ToolCapability(
                    name="analyze_statistics",
                    owner="statistics_service",
                    purpose="执行配对统计、置信区间和交互分析",
                    deterministic=True,
                    implementation_status="paired_effects_implemented",
                    input_contract="verified ExperimentRun[]",
                    output_contract="AnalysisFinding[]",
                ),
                ToolCapability(
                    name="review_visual_failures",
                    owner="vision_review_service",
                    purpose="由 Qwen-VL 审查图像和热力图失败模式",
                    deterministic=False,
                    implementation_status="interface_only",
                    input_contract="images + anomaly maps + numeric metrics",
                    output_contract="grounded visual observations",
                ),
            ]
        }

    def get(self, name: str) -> ToolCapability:
        return self._tools[name]

    def list(self) -> list[ToolCapability]:
        return sorted(self._tools.values(), key=lambda item: item.name)
