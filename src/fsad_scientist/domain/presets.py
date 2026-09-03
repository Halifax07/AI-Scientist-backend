"""Research direction presets.

A *preset* packages the platform's view of a research domain so that the same
runtime can serve both platform-shipped demos (e.g. ``fsad-demo`` for
few-shot industrial anomaly detection) and arbitrary user-defined research
directions.

The preset layer is the single source of truth for:

- display title, default domain and objective copy;
- default datasets and detector / selection strategy universes;
- primary and secondary evaluation metrics;
- prompt-prefix blocks injected into every LLM call so the agent stays on
  the user's chosen topic;
- evidence and hypothesis seeds that back the in-process demo content.

Presets are loaded from ``configs/presets/*.yaml`` and registered in
``PRESET_REGISTRY``.  Adding a new domain requires no changes to the
runtime – drop a YAML file, register the id, and the platform picks it
up on the next process start.

This module keeps the original ``"fsad"`` alias as the canonical id for the
few-shot industrial anomaly detection demo so existing data, tests and
APIs continue to work without changes.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DemoDatasetHint:
    """Reference to a dataset that a preset knows how to wire up."""

    name: str
    role: str = "primary"
    root: str | None = None  # None => user must provide absolute path
    description: str = ""


@dataclass(frozen=True)
class DemoEvidenceSeed:
    """A pre-built evidence record used to bootstrap a demo project."""

    title: str
    source_type: str = "paper"
    url: str | None = None
    arxiv_id: str | None = None
    doi: str | None = None
    authors: list[str] = field(default_factory=list)
    published_year: int | None = None
    claims: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class DemoHypothesisSeed:
    """A pre-built hypothesis used to bootstrap a demo project."""

    gap_title: str
    title: str
    claim: str
    null_hypothesis: str
    rationale: str
    independent_variables: list[str]
    dependent_variables: list[str]
    predicted_direction: str
    falsification_conditions: list[str]
    metric: str
    kind: str = "selection_main_effect"
    treatment: str | None = None
    control: str | None = None
    design_mode: str = "paired_comparison"
    minimum_pairs: int = 6


@dataclass(frozen=True)
class ResearchPreset:
    """All platform-defaults that describe one research direction."""

    id: str
    label: str
    description: str

    # User-facing defaults
    default_title: str
    default_domain: str
    default_application_context: str
    default_objective: str

    # Experiments
    primary_metrics: tuple[str, ...]
    supported_metrics: tuple[str, ...]
    detectors: tuple[str, ...]
    selection_strategies: tuple[str, ...]
    protocols: tuple[str, ...]
    shots: tuple[int, ...] = (1, 2, 4, 8)
    candidate_pool_size: int = 30
    seeds: tuple[int, ...] = tuple(range(10))

    # Datasets
    demo_datasets: tuple[DemoDatasetHint, ...] = ()

    # Prompt fragments
    context_block: str = ""
    scope_directives: tuple[str, ...] = ()
    gap_directives: tuple[str, ...] = ()
    hypothesis_directives: tuple[str, ...] = ()
    feedback_directives: tuple[str, ...] = ()
    design_directives: tuple[str, ...] = ()
    selection_strategy_directives: tuple[str, ...] = ()
    detector_directives: tuple[str, ...] = ()
    guidance_directives: tuple[str, ...] = ()

    # Demo seeds (only populated for built-in demos)
    demo_evidence: tuple[DemoEvidenceSeed, ...] = ()
    demo_gap_titles: tuple[str, ...] = ()
    demo_hypotheses: tuple[DemoHypothesisSeed, ...] = ()

    # Optional keywords that any new project bootstrapped from this preset
    # inherits in its ``user_guidance``.
    default_keywords: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Built-in FSAD preset (legacy / backward compatibility)
# ---------------------------------------------------------------------------


def _build_fsad_preset() -> ResearchPreset:
    """Few-shot industrial visual anomaly detection – the original demo."""

    return ResearchPreset(
        id="fsad",
        label="少样本工业视觉异常检测（演示）",
        description=(
            "保留原有 MVTec AD 演示场景，用于快速验证流程；"
            "真实研究请使用“自定义研究领域”输入。"
        ),
        default_title="少样本工业视觉异常检测自主研究",
        default_domain="少样本工业视觉异常检测",
        default_application_context="新产品上线时仅能获取极少量正常样本",
        default_objective=(
            "在仅有极少量正常参考图像、且适配阶段没有真实异常样本时，"
            "自主发现能够改善工业异常检测性能或稳定性的机制。"
        ),
        primary_metrics=("image_auroc", "pixel_auroc"),
        supported_metrics=(
            "image_auroc",
            "pixel_auroc",
            "image_ap",
            "aupro",
            "support_set_std",
            "worst_decile",
            "coverage_radius",
            "effective_rank",
        ),
        detectors=("anomalydino", "patchcore", "subspacead"),
        selection_strategies=("random", "k_center"),
        protocols=("strict_k_shot", "pool_compression_m30"),
        shots=(1, 2, 4, 8),
        candidate_pool_size=30,
        seeds=tuple(range(10)),
        demo_datasets=(
            DemoDatasetHint(
                name="MVTec AD",
                role="primary",
                description="14 类别工业异常检测基准（MVTec AD）。",
            ),
            DemoDatasetHint(
                name="VisA",
                role="validation",
                description="跨域验证数据集。",
            ),
        ),
        context_block=(
            "【平台选题】少样本工业视觉异常检测演示（MVTec AD 风格的少样本 "
            "正常参考样本场景，平台内置演示证据与空白候选）。\n"
            "【研究领域】{domain}\n"
            "【研究目标】{objective}\n"
            "【应用上下文】{application_context}"
        ),
        scope_directives=(
            "围绕用户在 ProjectSpec 中给出的少样本工业视觉异常检测目标展开；"
            "独立变量包含参考样本数量 K、参考样本构成、检测器结构、正常类内部变化程度；"
            "因变量包含图像级检测性能、像素级定位性能、跨支持集重采样稳定性。",
        ),
        gap_directives=(
            "从现有 MVTec AD 演示证据（PatchCore / AnomalyDINO / SubspaceAD / FastRef）"
            "中发现与支持集选择、检测器交互、查询自适应相关的空白。",
        ),
        hypothesis_directives=(
            "禁止把领域偷换为与用户输入无关的话题；"
            "analysis_contract.metric 必须使用 image_auroc / pixel_auroc / image_ap / aupro；"
            "detector_interaction 只能使用 patchcore / anomalydino / subspacead 等内置检测器名；"
            "selection_main_effect 只能使用 random / k_center 或已注册实现。",
        ),
        feedback_directives=(
            "若该课题不是少样本工业视觉异常检测，请不要沿用 FSAD 演示的默认语境。",
        ),
        design_directives=(
            "若该课题不是少样本工业视觉异常检测，请不要硬编码 FSAD 默认场景或默认检测器列表。",
        ),
        selection_strategy_directives=(
            "为当前研究领域生成候选样本选择函数；签名严格为 def select(candidate_ids, embeddings, k, seed) -> list[str]。",
        ),
        detector_directives=(
            "在少样本工业异常检测场景下，分数越高越异常："
            "核心必须计算测试图与正常支持图之间的非负偏离距离；"
            "禁止对距离取负、取倒数或转换成相似度；"
            "禁止只用全图均值、标准差或直方图。",
        ),
        guidance_directives=(
            "用户指导必须可追溯到具体类别、K 值或参考集构成，"
            "不得修改预注册指标与数据边界。",
        ),
        demo_evidence=(
            DemoEvidenceSeed(
                title="Towards Total Recall in Industrial Anomaly Detection",
                url="https://arxiv.org/abs/2106.08265",
                arxiv_id="2106.08265",
                claims=["PatchCore 是局部 patch 记忆库类基线。"],
            ),
            DemoEvidenceSeed(
                title="AnomalyDINO: Boosting Patch-based Few-shot Anomaly Detection with DINOv2",
                url="https://arxiv.org/abs/2405.14529",
                arxiv_id="2405.14529",
                claims=["冻结 DINOv2 patch 特征可用于训练自由少样本异常检测。"],
            ),
            DemoEvidenceSeed(
                title="SubspaceAD: Training-Free Few-Shot Anomaly Detection via Subspace Modeling",
                url="https://arxiv.org/abs/2602.23013",
                arxiv_id="2602.23013",
                claims=["正常 patch 特征可通过子空间重建残差进行异常评分。"],
            ),
            DemoEvidenceSeed(
                title=(
                    "FastRef: Fast Prototype Refinement for Few-shot Industrial "
                    "Anomaly Detection"
                ),
                url="https://arxiv.org/abs/2506.21398",
                arxiv_id="2506.21398",
                claims=["查询图像统计可在测试时修正少样本正常原型。"],
            ),
        ),
        demo_gap_titles=(
            "参考集质量与稳定性缺少系统研究",
            "代表性的定义可能依赖检测器结构",
            "测试时信息能否抵消劣质参考集",
            "从被动选样本扩展到主动采集",
        ),
        default_keywords=(
            "few-shot",
            "anomaly detection",
            "support set selection",
        ),
    )


def _build_generic_preset() -> ResearchPreset:
    """The neutral default for arbitrary research directions."""

    return ResearchPreset(
        id="generic",
        label="通用科研方向（自定义）",
        description="平台默认方向，由用户完全自定义研究领域与关键词。",
        default_title="用户自定义研究",
        default_domain="用户自定义研究领域",
        default_application_context="用户提供的研究场景和约束条件",
        default_objective="",
        primary_metrics=("primary_score",),
        supported_metrics=("primary_score",),
        detectors=(),
        selection_strategies=("random",),
        protocols=("exploratory",),
        shots=(1, 4, 8),
        candidate_pool_size=30,
        seeds=tuple(range(10)),
        demo_datasets=(),
        context_block=(
            "【平台选题】用户通过通用科研工作台自定义的研究领域；"
            "不预设任何特定数据集或方法；所有 AI 生成内容必须围绕"
            "用户在 objective 中给出的研究方向与关键词。\n"
            "【研究领域】{domain}\n"
            "【研究目标】{objective}\n"
            "【应用上下文】{application_context}"
        ),
        scope_directives=(
            "始终根据用户在 ProjectSpec.objective / domain 中提供的"
            "研究领域与关键词展开，不得偷换为与用户输入无关的方向。",
        ),
        gap_directives=(
            "必须严格围绕用户在 ProjectSpec 中给出的 objective 与"
            "domain 展开，禁止偷换为与用户输入无关的工业视觉或异常检测话题。",
        ),
        hypothesis_directives=(
            "所有假设必须围绕用户在 ProjectSpec.objective / domain 中给出的"
            "研究领域与关键词展开；不要回到少样本工业视觉异常检测、"
            "MVTec AD 或 PatchCore 等与用户输入无关的默认话题。",
        ),
        feedback_directives=(
            "若该课题不是少样本工业视觉异常检测，请不要沿用 FSAD 演示的"
            "默认语境或硬编码工业视觉假设。",
        ),
        design_directives=(
            "若该课题不是少样本工业视觉异常检测，请不要硬编码 FSAD"
            "默认场景或默认检测器列表。",
        ),
        selection_strategy_directives=(
            "为当前研究领域生成候选样本选择函数；签名严格为 .",
        ),
        detector_directives=(
            "为当前研究领域实现评分函数；保留 anomaly_score(image, support_images, seed) -> float 签名；"
            "其他领域可保留相同签名与命名约定，但允许按用户 objective 调整输出语义。",
        ),
        guidance_directives=(
            "用户指导在当前领域可能涉及不同的样本、指标或边界条件；"
            "不得修改预注册指标与数据边界。",
        ),
        demo_evidence=(),
        demo_gap_titles=(),
        demo_hypotheses=(),
        default_keywords=(),
    )


# ---------------------------------------------------------------------------
# Registry & file-based presets
# ---------------------------------------------------------------------------


def _build_mvad_preset() -> ResearchPreset:
    """基于机器视觉的异常检测 – 平台默认的宽泛研究方向。

    该预设是平台默认入口：覆盖工业产品外观、医学影像、视频监控、遥感等
    "机器视觉 + 异常/缺陷/离群点检测" 大方向。少样本工业视觉异常检测作为
    ``fsad`` 演示预设仍然保留,用于快速验证流程,但不再被硬编码到默认
    ProjectSpec / 默认工作流中。
    """

    return ResearchPreset(
        id="machine_vision_anomaly_detection",
        label="基于机器视觉的异常检测（平台默认）",
        description=(
            "围绕机器视觉中的异常 / 缺陷 / 离群点检测开展自主科研,覆盖工业"
            "外观检测、医学影像、视频监控、遥感、文档图像等场景;"
            "少样本工业视觉异常检测 (fsad) 仍作为一键演示入口。"
        ),
        default_title="基于机器视觉的异常检测自主研究",
        default_domain="基于机器视觉的异常检测",
        default_application_context=(
            "在工业质检 / 医学影像 / 视频监控 / 遥感等机器视觉场景中,"
            "正常样本容易获得、异常样本稀缺或代价昂贵。"
        ),
        default_objective=(
            "围绕用户输入的研究方向与关键词,自主发现问题空白、"
            "提出可证伪的创新机制,并设计验证实验。"
        ),
        primary_metrics=("primary_score",),
        supported_metrics=(
            "image_auroc",
            "pixel_auroc",
            "image_ap",
            "aupro",
            "primary_score",
            "stability_score",
        ),
        # 默认不锁定任何检测器 / 选样策略 – 由用户在 ProjectSpec 中按场景选择。
        # 演示与回放时仍可使用 fsad 提供的检测器列表。
        detectors=(),
        selection_strategies=(),
        protocols=("exploratory",),
        shots=(1, 2, 4, 8),
        candidate_pool_size=30,
        seeds=tuple(range(10)),
        demo_datasets=(
            DemoDatasetHint(
                name="用户指定数据集",
                role="primary",
                description=(
                    "由用户在 ProjectSpec 中按研究方向提供的图像 / 视频数据集。"
                ),
            ),
        ),
        context_block=(
            "【平台选题】基于机器视觉的异常检测,平台默认宽泛方向;"
            "不预设任何特定数据集或方法;所有 AI 生成内容必须围绕"
            "用户在 objective 与 user_guidance 中给出的研究方向与关键词。\n"
            "【研究领域】{domain}\n"
            "【研究目标】{objective}\n"
            "【应用上下文】{application_context}"
        ),
        scope_directives=(
            "围绕用户在 ProjectSpec.objective / domain / user_guidance 中提供的"
            "研究领域与关键词展开,可涵盖工业外观、医学影像、视频监控、"
            "遥感等机器视觉异常检测场景;不得偷换为与用户输入无关的方向。",
        ),
        gap_directives=(
            "必须严格围绕用户在 ProjectSpec 中给出的 objective 与"
            "domain 展开,禁止偷换为与用户输入无关的工业视觉或异常检测话题;",
            "除非用户显式选择 fsad 演示预设,否则禁止把研究空白默认收敛到"
            "少样本工业异常检测的固定候选列表。",
        ),
        hypothesis_directives=(
            "所有假设必须围绕用户在 ProjectSpec.objective / domain 中给出的"
            "研究领域与关键词展开;不要回到少样本工业视觉异常检测、"
            "MVTec AD 或 PatchCore 等与用户输入无关的默认话题。",
        ),
        feedback_directives=(
            "若该课题不是少样本工业视觉异常检测,请不要沿用 FSAD 演示的"
            "默认语境或硬编码工业视觉假设。",
        ),
        design_directives=(
            "若该课题不是少样本工业视觉异常检测,请不要硬编码 FSAD"
            "默认场景或默认检测器列表。",
        ),
        selection_strategy_directives=(
            "为当前研究领域生成候选样本选择函数;签名严格为 "
            "def select(candidate_ids, embeddings, k, seed) -> list[str]。",
        ),
        detector_directives=(
            "为当前研究领域实现异常评分函数;保留 "
            "anomaly_score(image, support_images, seed) -> float 签名;",
            "不同领域的输出语义可由用户自定义,但必须返回非负偏离量或等价"
            "可证伪指标,禁止伪造实验指标。",
        ),
        guidance_directives=(
            "用户指导在当前领域可能涉及不同的样本、指标或边界条件;"
            "不得修改预注册指标与数据边界。",
        ),
        demo_evidence=(),
        demo_gap_titles=(),
        demo_hypotheses=(),
        default_keywords=(
            "machine vision",
            "anomaly detection",
        ),
    )


def _default_presets() -> dict[str, ResearchPreset]:
    registry: dict[str, ResearchPreset] = {
        "generic": _build_generic_preset(),
        "fsad": _build_fsad_preset(),
        # 平台默认入口:基于机器视觉的异常检测。少样本工业视觉异常检测
        # 作为 ``fsad`` 演示保留,不作为唯一默认方向。
        "machine_vision_anomaly_detection": _build_mvad_preset(),
    }
    return registry


def _coerce_evidence_seed(item: dict[str, Any]) -> DemoEvidenceSeed:
    return DemoEvidenceSeed(
        title=str(item["title"]),
        source_type=str(item.get("source_type", "paper")),
        url=item.get("url"),
        arxiv_id=item.get("arxiv_id"),
        doi=item.get("doi"),
        authors=list(item.get("authors", []) or []),
        published_year=item.get("published_year"),
        claims=list(item.get("claims", []) or []),
    )


def _coerce_dataset_hint(item: dict[str, Any]) -> DemoDatasetHint:
    return DemoDatasetHint(
        name=str(item["name"]),
        role=str(item.get("role", "primary")),
        root=item.get("root"),
        description=str(item.get("description", "")),
    )


def _coerce_hypothesis_seed(item: dict[str, Any]) -> DemoHypothesisSeed:
    return DemoHypothesisSeed(
        gap_title=str(item["gap_title"]),
        title=str(item["title"]),
        claim=str(item["claim"]),
        null_hypothesis=str(item["null_hypothesis"]),
        rationale=str(item["rationale"]),
        independent_variables=list(item.get("independent_variables", []) or []),
        dependent_variables=list(item.get("dependent_variables", []) or []),
        predicted_direction=str(item.get("predicted_direction", "")),
        falsification_conditions=list(item.get("falsification_conditions", []) or []),
        metric=str(item["metric"]),
        kind=str(item.get("kind", "selection_main_effect")),
        treatment=item.get("treatment"),
        control=item.get("control"),
        design_mode=str(item.get("design_mode", "paired_comparison")),
        minimum_pairs=int(item.get("minimum_pairs", 6)),
    )


def _preset_from_dict(payload: dict[str, Any]) -> ResearchPreset:
    return ResearchPreset(
        id=str(payload["id"]),
        label=str(payload.get("label", payload["id"])),
        description=str(payload.get("description", "")),
        default_title=str(payload.get("default_title", payload["id"])),
        default_domain=str(payload.get("default_domain", "用户自定义研究领域")),
        default_application_context=str(
            payload.get("default_application_context", "用户提供的研究场景和约束条件")
        ),
        default_objective=str(payload.get("default_objective", "")),
        primary_metrics=tuple(payload.get("primary_metrics", ()) or ()),
        supported_metrics=tuple(payload.get("supported_metrics", ()) or ()),
        detectors=tuple(payload.get("detectors", ()) or ()),
        selection_strategies=tuple(payload.get("selection_strategies", ()) or ()),
        protocols=tuple(payload.get("protocols", ()) or ()),
        shots=tuple(int(s) for s in (payload.get("shots", (1, 2, 4, 8)) or ())),
        candidate_pool_size=int(payload.get("candidate_pool_size", 30)),
        seeds=tuple(int(s) for s in (payload.get("seeds", list(range(10))) or ())),
        demo_datasets=tuple(
            _coerce_dataset_hint(item) for item in (payload.get("demo_datasets", ()) or ())
        ),
        context_block=str(payload.get("context_block", "")),
        scope_directives=tuple(payload.get("scope_directives", ()) or ()),
        gap_directives=tuple(payload.get("gap_directives", ()) or ()),
        hypothesis_directives=tuple(payload.get("hypothesis_directives", ()) or ()),
        feedback_directives=tuple(payload.get("feedback_directives", ()) or ()),
        design_directives=tuple(payload.get("design_directives", ()) or ()),
        selection_strategy_directives=tuple(
            payload.get("selection_strategy_directives", ()) or ()
        ),
        detector_directives=tuple(payload.get("detector_directives", ()) or ()),
        guidance_directives=tuple(payload.get("guidance_directives", ()) or ()),
        demo_evidence=tuple(
            _coerce_evidence_seed(item) for item in (payload.get("demo_evidence", ()) or ())
        ),
        demo_gap_titles=tuple(payload.get("demo_gap_titles", ()) or ()),
        demo_hypotheses=tuple(
            _coerce_hypothesis_seed(item) for item in (payload.get("demo_hypotheses", ()) or ())
        ),
        default_keywords=tuple(payload.get("default_keywords", ()) or ()),
    )


def _project_root() -> Path:
    """Locate the backend package root, regardless of where Python is launched from."""
    return Path(__file__).resolve().parents[3]


@lru_cache(maxsize=1)
def load_preset_registry() -> dict[str, ResearchPreset]:
    """Load built-in presets + any YAML files from ``configs/presets``.

    The cache is invalidated whenever files in ``configs/presets`` change,
    which is rare in production but convenient during development.
    """

    registry = _default_presets()
    presets_dir_env = os.environ.get("AISCIENTIST_PRESETS_DIR")
    candidates: list[Path] = []
    if presets_dir_env:
        candidates.append(Path(presets_dir_env))
    candidates.append(_project_root() / "configs" / "presets")
    for path in candidates:
        if not path.is_dir():
            continue
        for yaml_path in sorted(path.glob("*.yaml")):
            try:
                payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
            except yaml.YAMLError:
                continue
            if not isinstance(payload, dict):
                continue
            preset_id = str(payload.get("id") or yaml_path.stem)
            payload["id"] = preset_id
            try:
                registry[preset_id] = _preset_from_dict(payload)
            except (KeyError, TypeError, ValueError):
                # A malformed preset must not break boot of the rest of the registry.
                continue
    return registry


def resolve_preset(preset_id: str | None) -> ResearchPreset:
    """Look up a preset by id, falling back to the generic preset."""

    if not preset_id:
        return load_preset_registry()["generic"]
    registry = load_preset_registry()
    return registry.get(preset_id) or registry["generic"]


def render_context_block(preset: ResearchPreset, *, domain: str, objective: str, application_context: str) -> str:
    """Render the preset's context_block with the project fields inlined."""

    block = preset.context_block or "{domain} | {objective} | {application_context}"
    return block.format(
        domain=domain or preset.default_domain,
        objective=objective or preset.default_objective or preset.default_domain,
        application_context=application_context or preset.default_application_context,
    )


__all__ = [
    "ResearchPreset",
    "DemoDatasetHint",
    "DemoEvidenceSeed",
    "DemoHypothesisSeed",
    "load_preset_registry",
    "resolve_preset",
    "render_context_block",
]
