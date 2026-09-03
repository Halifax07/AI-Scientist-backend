"""Validation and compilation for lightweight, general experiment designs."""

from __future__ import annotations

import itertools
import json
from typing import Any

from fsad_scientist.domain.models import (
    AnalysisContract,
    ExperimentCardBlockSpec,
    ExperimentCardPresentationSpec,
    ExperimentConditionSpec,
    ExperimentDesignSpec,
    ExperimentPlan,
)

RUN_FIELDS = frozenset(
    {"selection_strategy", "detector", "category", "shots", "seed", "protocol"}
)
ANALYSIS_MODES = frozenset(
    {"group_comparison", "factor_effects", "ordered_trend", "distribution_summary"}
)
MAX_PRESENTATION_BLOCKS = 16


def _canonical_metric(value: str) -> str:
    key = "".join(character for character in value.strip().casefold() if character.isalnum())
    return {
        "imageauroc": "image_auroc",
        "pixelauroc": "pixel_auroc",
        "imageap": "image_ap",
        "aupro": "aupro",
    }.get(key, value.strip())


def normalize_design(
    plan: ExperimentPlan,
    *,
    hypothesis_id: str,
    contract: AnalysisContract | None = None,
) -> ExperimentDesignSpec:
    """Return a selected design without mutating a persisted plan.

    Plans written before general designs existed are represented as the two
    preregistered treatment/control conditions at runtime only.
    """

    candidates = [
        design
        for design in plan.designs
        if design.hypothesis_id in {None, hypothesis_id}
    ]
    if candidates:
        selected = next(
            (design for design in candidates if design.hypothesis_id == hypothesis_id),
            candidates[0],
        )
        return selected.model_copy(deep=True)
    contract = contract or plan.hypothesis_contracts.get(hypothesis_id)
    if contract is None:
        raise ValueError(f"No analysis contract exists for hypothesis: {hypothesis_id}")
    return ExperimentDesignSpec(
        id=f"legacy_{hypothesis_id}",
        name="legacy_treatment_control",
        hypothesis_id=hypothesis_id,
        design_type="paired_comparison",
        design_mode="paired_comparison",
        factors=[
            {
                "name": "selection_strategy",
                "field": "selection_strategy",
                "levels": [contract.control, contract.treatment],
            }
        ],
        conditions=[
            {
                "id": "control",
                "label": "control",
                "factor_values": {"selection_strategy": contract.control},
            },
            {
                "id": "treatment",
                "label": "treatment",
                "factor_values": {"selection_strategy": contract.treatment},
            },
        ],
        analysis={
            "mode": "group_comparison",
            "primary_metric": contract.metric,
            "alpha": contract.alpha,
            "minimum_pairs": contract.minimum_pairs,
        },
        purpose="main_study",
    )


def validate_design(
    design: ExperimentDesignSpec,
    plan: ExperimentPlan,
    *,
    budget: int | None = None,
    allowed_categories: set[str] | None = None,
    allowed_detectors: set[str] | None = None,
    allowed_strategies: set[str] | None = None,
) -> None:
    """Validate a design against the frozen plan and executable capabilities."""

    if design.analysis.mode not in ANALYSIS_MODES:
        raise ValueError(f"Unsupported experiment analysis mode: {design.analysis.mode}")
    if design.design_mode == "paired_comparison" and len(design.conditions) not in {0, 2}:
        raise ValueError("paired_comparison must contain exactly two conditions")
    allowed_metrics = {_canonical_metric(metric) for metric in plan.metrics}
    if _canonical_metric(design.analysis.primary_metric or "") not in allowed_metrics:
        raise ValueError(
            f"Primary metric is not allowed by the experiment plan: "
            f"{design.analysis.primary_metric}"
        )
    if not design.factors:
        raise ValueError("An experiment design must declare at least one factor")
    factor_names: set[str] = set()
    factor_fields: dict[str, str] = {}
    field_owners: dict[str, str] = {}
    for factor in design.factors:
        if factor.name in factor_names:
            raise ValueError(f"Duplicate experiment factor: {factor.name}")
        factor_names.add(factor.name)
        field = factor.field or factor.run_field
        if field not in RUN_FIELDS:
            raise ValueError(f"Unsupported Run factor field: {field}")
        if field in field_owners:
            raise ValueError(
                f"Run field {field} cannot be bound by multiple factors: "
                f"{field_owners[field]} and {factor.name}"
            )
        field_owners[field] = factor.name
        factor_fields[factor.name] = field
        allowed_levels = _allowed_levels(
            field,
            plan,
            allowed_categories=allowed_categories,
            allowed_detectors=allowed_detectors,
            allowed_strategies=allowed_strategies,
        )
        invalid = [level for level in factor.levels if level not in allowed_levels]
        if invalid:
            raise ValueError(
                f"Factor {factor.name} contains values outside the plan: {invalid}"
            )

    condition_ids: set[str] = set()
    condition_signatures: set[str] = set()
    for condition in design.conditions:
        if condition.id in condition_ids:
            raise ValueError(f"Duplicate experiment condition: {condition.id}")
        condition_ids.add(condition.id)
        signature = repr(sorted(condition.factor_values.items(), key=lambda item: item[0]))
        if signature in condition_signatures:
            raise ValueError(f"Experiment conditions must be unique: {condition.id}")
        condition_signatures.add(signature)
        if set(condition.factor_values) != factor_names:
            missing = sorted(factor_names - set(condition.factor_values))
            extra = sorted(set(condition.factor_values) - factor_names)
            raise ValueError(
                f"Condition {condition.id} must bind every factor; missing={missing}, extra={extra}"
            )
        for name, value in condition.factor_values.items():
            factor = next(item for item in design.factors if item.name == name)
            if value not in factor.levels:
                raise ValueError(
                    f"Condition {condition.id} assigns {value!r} outside factor {name} levels"
                )

    assignments = (
        [condition.factor_values for condition in design.conditions]
        if design.conditions
        else [
            dict(zip(factor_names_ordered, values, strict=True))
            for values in itertools.product(*(factor.levels for factor in design.factors))
            for factor_names_ordered in [[factor.name for factor in design.factors]]
        ]
    )
    assignment_signatures = {
        repr(sorted(assignment.items(), key=lambda item: item[0]))
        for assignment in assignments
    }
    if len(assignment_signatures) < 2:
        raise ValueError(
            "An experiment design must contain at least two executable condition assignments"
        )
    if (
        design.support_selection_strategy is not None
        and design.support_selection_strategy not in (
            allowed_strategies or set(plan.selection_strategies)
        )
    ):
        raise ValueError(
            "Default support selection strategy is outside the experiment plan: "
            f"{design.support_selection_strategy}"
        )
    default_strategies: list[tuple[str, str]] = []
    if "selection_strategy" not in field_owners:
        if design.support_selection_strategy is not None:
            default_strategies = [
                (design.hypothesis_id or "design", design.support_selection_strategy)
            ]
        else:
            applicable_hypothesis_ids = (
                [design.hypothesis_id]
                if design.hypothesis_id is not None
                else plan.hypothesis_ids
            )
            default_strategies = [
                (hypothesis_id, contract.treatment)
                for hypothesis_id in applicable_hypothesis_ids
                if (
                    (contract := plan.hypothesis_contracts.get(hypothesis_id)) is not None
                    and contract.treatment is not None
                )
            ]
    for assignment in assignments:
        values_by_field = {
            factor_fields[name]: value for name, value in assignment.items()
        }
        protocol = values_by_field.get("protocol")
        strategy = values_by_field.get("selection_strategy")
        if protocol == "strict_k_shot" and strategy not in {None, "random"}:
            raise ValueError(
                "Unsupported protocol/selection_strategy combination: "
                f"strict_k_shot requires random, got {strategy}"
            )
        if protocol == "strict_k_shot" and strategy is None:
            incompatible_defaults = [
                (hypothesis_id, default_strategy)
                for hypothesis_id, default_strategy in default_strategies
                if default_strategy != "random"
            ]
            if incompatible_defaults:
                details = ", ".join(
                    f"{hypothesis_id}={default_strategy}"
                    for hypothesis_id, default_strategy in incompatible_defaults
                )
                raise ValueError(
                    "Unsupported default protocol/selection_strategy combination: "
                    "strict_k_shot requires AnalysisContract.treatment=random when "
                    f"selection_strategy is not a design factor; got {details}"
                )

    condition_count = len(design.conditions) or _factorial_size(design)
    if (
        design.presentation_spec is not None
        and len(design.presentation_spec.blocks) > MAX_PRESENTATION_BLOCKS
    ):
        raise ValueError("The presentation spec cannot contain more than 16 blocks")
    design_limit = design.max_runs or design.budget
    if design_limit is not None and condition_count > design_limit:
        raise ValueError("The compiled design exceeds max_runs")
    if budget is not None and condition_count > budget:
        raise ValueError("The experiment design exceeds the available run budget")
    if design.analysis.baseline_condition_id is not None:
        condition_ids = (
            {condition.id for condition in design.conditions}
            if design.conditions
            else {f"condition_{index + 1}" for index in range(condition_count)}
        )
        if design.analysis.baseline_condition_id not in condition_ids:
            raise ValueError(
                "baseline_condition_id must reference a compiled experiment condition"
            )
    if design.analysis.mode == "ordered_trend":
        ordered_factor = design.analysis.ordered_factor
        if ordered_factor is None:
            if len(design.factors) != 1:
                raise ValueError("ordered_trend requires analysis.ordered_factor")
            ordered_factor = design.factors[0].name
        if ordered_factor not in factor_names:
            raise ValueError(f"Unknown ordered trend factor: {ordered_factor}")
        if not all(_is_orderable(level) for level in next(
            factor.levels for factor in design.factors if factor.name == ordered_factor
        )):
            raise ValueError("ordered_trend requires numeric or string-orderable levels")


def validate_plan_designs(
    plan: ExperimentPlan,
    *,
    budget: int | None = None,
    allowed_categories: set[str] | None = None,
    allowed_detectors: set[str] | None = None,
    allowed_strategies: set[str] | None = None,
) -> None:
    """Validate every design and its plan-level identity constraints."""

    design_ids = [design.id for design in plan.designs]
    if len(design_ids) != len(set(design_ids)):
        raise ValueError("Experiment design IDs must be unique")
    for design in plan.designs:
        validate_design(
            design,
            plan,
            budget=budget,
            allowed_categories=allowed_categories,
            allowed_detectors=allowed_detectors,
            allowed_strategies=allowed_strategies,
        )


def compile_design(
    design: ExperimentDesignSpec,
    *,
    plan: ExperimentPlan | None = None,
    budget: int | None = None,
    allowed_categories: set[str] | None = None,
    allowed_detectors: set[str] | None = None,
    allowed_strategies: set[str] | None = None,
) -> list[ExperimentConditionSpec]:
    """Compile explicit or factorial conditions into deterministic conditions."""

    if plan is not None:
        validate_design(
            design,
            plan,
            budget=budget,
            allowed_categories=allowed_categories,
            allowed_detectors=allowed_detectors,
            allowed_strategies=allowed_strategies,
        )
    conditions = design.conditions
    if not conditions:
        conditions = [
            ExperimentConditionSpec(
                id=f"condition_{index + 1}",
                factor_values=dict(zip(names, values, strict=True)),
            )
            for index, values in enumerate(
                itertools.product(*(factor.levels for factor in design.factors))
            )
            for names in [[factor.name for factor in design.factors]]
        ]
    design_limit = design.max_runs or design.budget
    if design_limit is not None and len(conditions) > design_limit:
        raise ValueError("The compiled design exceeds max_runs")
    if budget is not None and len(conditions) > budget:
        raise ValueError("The compiled design exceeds the available run budget")
    return [condition.model_copy(deep=True) for condition in conditions]


def default_presentation_spec(
    design: ExperimentDesignSpec,
) -> ExperimentCardPresentationSpec:
    """Build a deterministic, data-only card layout for a design mode."""

    narrative = ExperimentCardBlockSpec(
        id="narrative", kind="narrative", source="design", span="full"
    )
    progress = ExperimentCardBlockSpec(
        id="progress", kind="progress", source="progress", span="full"
    )
    evidence = ExperimentCardBlockSpec(
        id="evidence", kind="evidence", source="evidence", span="full"
    )
    mode_blocks = {
        "group_comparison": (
            "stack",
            [
                narrative,
                progress,
                ExperimentCardBlockSpec(
                    id="conditions",
                    kind="table",
                    source="condition_statistics",
                    span="full",
                ),
                ExperimentCardBlockSpec(
                    id="effects", kind="chart", source="condition_effects", chart_mark="bar"
                ),
                evidence,
            ],
        ),
        "factor_effects": (
            "grid",
            [
                narrative,
                progress,
                ExperimentCardBlockSpec(
                    id="factors",
                    kind="chart",
                    source="factor_effects",
                    chart_mark="bar",
                    span="half",
                ),
                ExperimentCardBlockSpec(
                    id="interaction",
                    kind="chart",
                    source="interaction_summary",
                    chart_mark="heatmap",
                    span="half",
                ),
                evidence,
            ],
        ),
        "ordered_trend": (
            "sequence",
            [
                narrative,
                progress,
                ExperimentCardBlockSpec(
                    id="trend", kind="chart", source="ordered_trend", chart_mark="line"
                ),
                ExperimentCardBlockSpec(
                    id="conditions",
                    kind="table",
                    source="condition_statistics",
                    span="full",
                ),
                evidence,
            ],
        ),
        "distribution_summary": (
            "split",
            [
                narrative,
                progress,
                ExperimentCardBlockSpec(
                    id="distribution",
                    kind="chart",
                    source="distribution_summary",
                    chart_mark="interval",
                    span="half",
                ),
                ExperimentCardBlockSpec(
                    id="runs", kind="runs", source="runs", span="half"
                ),
                evidence,
            ],
        ),
    }
    layout, blocks = mode_blocks[design.analysis.mode]
    return ExperimentCardPresentationSpec(layout=layout, blocks=blocks)


def normalize_presentation_spec_payload(
    raw_spec: Any,
    *,
    analysis_mode: str,
) -> dict[str, Any]:
    """Repair only additive omissions in an AI presentation-spec payload.

    The returned mapping is still validated by ``ExperimentCardPresentationSpec``;
    this helper does not discard invalid AI content or relax the DSL contract.
    """

    if not isinstance(raw_spec, dict):
        raise ValueError("presentation_spec must be an object")
    if analysis_mode not in ANALYSIS_MODES:
        raise ValueError(f"Unsupported experiment analysis mode: {analysis_mode}")

    raw_blocks = raw_spec.get("blocks", [])
    if not isinstance(raw_blocks, list) or any(
        not isinstance(block, dict) for block in raw_blocks
    ):
        raise ValueError("presentation_spec.blocks must be a list of objects")

    blocks = [dict(block) for block in raw_blocks]
    for block in blocks:
        if block.get("source") in {"group_comparison", "comparison"}:
            block["source"] = "condition_statistics"
        if (
            block.get("kind") == "chart"
            and block.get("source") == "distribution_summary"
            and block.get("chart_mark") == "point"
        ):
            # Keep the distribution-chart intent while using a supported mark.
            block["chart_mark"] = "bar"
    present = {
        (block.get("kind"), block.get("source"))
        for block in blocks
    }
    required = (
        {"id": "narrative", "kind": "narrative", "source": "design", "span": "full"},
        {"id": "progress", "kind": "progress", "source": "progress", "span": "full"},
        {"id": "evidence", "kind": "evidence", "source": "evidence", "span": "full"},
    )
    for block in required:
        key = (block["kind"], block["source"])
        if key not in present:
            blocks.append(block)
            present.add(key)

    result_sources = {
        "group_comparison": "condition_statistics",
        "factor_effects": "factor_effects",
        "ordered_trend": "ordered_trend",
        "distribution_summary": "distribution_summary",
    }
    result_kinds = {
        "metrics",
        "chart",
        "table",
        "runs",
        "diagnostics",
        "insight",
        "key_value",
        "timeline",
    }
    if not any(block.get("kind") in result_kinds for block in blocks):
        blocks.append(
            {
                "id": f"{analysis_mode}-data",
                "kind": "table",
                "source": result_sources[analysis_mode],
                "span": "full",
            }
        )

    normalized = dict(raw_spec)
    normalized["blocks"] = blocks
    return normalized


def normalize_design_conditions_payload(raw_conditions: Any) -> Any:
    """Drop only semantically identical duplicate AI condition objects."""

    if not isinstance(raw_conditions, list):
        return raw_conditions

    normalized: list[Any] = []
    seen: set[str] = set()
    for raw_condition in raw_conditions:
        if not isinstance(raw_condition, dict):
            normalized.append(raw_condition)
            continue
        try:
            condition = ExperimentConditionSpec.model_validate(raw_condition)
        except (TypeError, ValueError):
            normalized.append(raw_condition)
            continue
        signature = json.dumps(
            condition.factor_values,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=repr,
        )
        if signature not in seen:
            normalized.append(raw_condition)
            seen.add(signature)
    return normalized


def _legacy_result_aware_presentation_spec(
    design: ExperimentDesignSpec,
    round_summary: dict[str, Any],
) -> ExperimentCardPresentationSpec:
    """Choose a small, safe card layout from the observed round result."""

    source_spec = design.presentation_spec or default_presentation_spec(design)
    blocks = [
        ExperimentCardBlockSpec(id="narrative", kind="narrative", source="design"),
        ExperimentCardBlockSpec(id="progress", kind="progress", source="progress"),
    ]
    mode = design.analysis.mode
    condition_effects = round_summary.get("condition_effects") or []
    factor_effects = round_summary.get("factor_effects") or []
    interaction_summary = round_summary.get("interaction_summary") or []
    ordered_trend = round_summary.get("ordered_trend") or []
    distribution = round_summary.get("distribution_summary") or {}

    if mode == "ordered_trend":
        valid_points = [point for point in ordered_trend if point.get("mean") is not None]
        if len(valid_points) >= 2:
            blocks.append(
                ExperimentCardBlockSpec(
                    id="trend", kind="chart", source="ordered_trend", chart_mark="line"
                )
            )
        else:
            blocks.append(
                ExperimentCardBlockSpec(
                    id="trend-data", kind="table", source="ordered_trend"
                )
            )
        blocks.append(
            ExperimentCardBlockSpec(
                id="conditions", kind="table", source="condition_statistics"
            )
        )
    elif mode == "factor_effects":
        if factor_effects:
            blocks.append(
                ExperimentCardBlockSpec(
                    id="factors", kind="chart", source="factor_effects", chart_mark="bar"
                )
            )
        else:
            blocks.append(
                ExperimentCardBlockSpec(
                    id="factor-data", kind="table", source="condition_statistics"
                )
            )
        has_interaction_data = _has_complete_interaction_data(
            design, interaction_summary
        )
        if has_interaction_data:
            blocks.append(
                ExperimentCardBlockSpec(
                    id="interaction",
                    kind="chart",
                    source="interaction_summary",
                    chart_mark="heatmap",
                )
            )
        elif interaction_summary:
            blocks.append(
                ExperimentCardBlockSpec(
                    id="interaction-data",
                    kind="table",
                    source="interaction_summary",
                )
            )
        else:
            blocks.append(
                ExperimentCardBlockSpec(
                    id="interaction-data",
                    kind="table",
                    source="factor_effects" if factor_effects else "condition_statistics",
                )
            )
    elif mode == "distribution_summary":
        has_distribution_data = all(
            distribution.get(key) is not None for key in ("minimum", "mean", "maximum")
        )
        if has_distribution_data:
            blocks.append(
                ExperimentCardBlockSpec(
                    id="distribution",
                    kind="chart",
                    source="distribution_summary",
                    chart_mark="interval",
                )
            )
        else:
            blocks.append(
                ExperimentCardBlockSpec(
                    id="distribution-data", kind="table", source="condition_statistics"
                )
            )
        blocks.append(
            ExperimentCardBlockSpec(id="runs", kind="runs", source="runs", span="half")
        )
    else:
        blocks.append(
            ExperimentCardBlockSpec(
                id="conditions", kind="table", source="condition_statistics"
            )
        )
        if condition_effects:
            blocks.append(
                ExperimentCardBlockSpec(
                    id="effects", kind="chart", source="condition_effects", chart_mark="bar"
                )
            )

    if round_summary.get("failed_run_ids"):
        if not any(block.kind == "runs" for block in blocks):
            blocks.append(
                ExperimentCardBlockSpec(id="runs", kind="runs", source="runs")
            )
        if not any(block.kind == "diagnostics" for block in blocks):
            blocks.append(
                ExperimentCardBlockSpec(
                    id="diagnostics", kind="diagnostics", source="diagnostics"
                )
            )
    blocks.append(ExperimentCardBlockSpec(id="evidence", kind="evidence", source="evidence"))
    layout = (
        source_spec.layout
        if source_spec.layout in {"stack", "split", "grid", "sequence"}
        else "stack"
    )
    density = source_spec.density
    return ExperimentCardPresentationSpec(layout=layout, density=density, blocks=blocks[:16])


def _block_has_observed_data(
    block: ExperimentCardBlockSpec,
    design: ExperimentDesignSpec,
    summary: dict[str, Any],
) -> bool:
    """Keep a model-selected block only when its source can be rendered honestly."""

    source = block.source
    if source in {"design", "progress", "evidence", "runs"}:
        return True
    if source == "feedback":
        return bool(summary.get("feedback"))
    if source == "diagnostics":
        return bool(summary.get("failed_run_ids"))
    values = summary.get(source)
    if source == "distribution_summary":
        return isinstance(values, dict) and any(
            values.get(key) is not None for key in ("mean", "minimum", "maximum")
        )
    if not values:
        return False
    if block.kind == "chart":
        if source == "ordered_trend":
            return sum(item.get("mean") is not None for item in values) >= 2
        if source == "interaction_summary":
            return _has_complete_interaction_data(design, values)
        if source in {"condition_statistics", "condition_effects", "factor_effects"}:
            return any(
                (item.get("mean") if source == "condition_statistics" else item.get("effect"))
                is not None
                for item in values
            )
    return True


def result_aware_presentation_spec(
    design: ExperimentDesignSpec,
    round_summary: dict[str, Any],
) -> ExperimentCardPresentationSpec:
    """Merge result feedback into the existing safe AI card specification."""

    source_spec = design.presentation_spec or default_presentation_spec(design)
    blocks = [
        block.model_copy(deep=True)
        for block in source_spec.blocks
        if _block_has_observed_data(block, design, round_summary)
    ]
    present = {(block.kind, block.source) for block in blocks}

    required = [
        ExperimentCardBlockSpec(id="narrative", kind="narrative", source="design"),
        ExperimentCardBlockSpec(id="progress", kind="progress", source="progress"),
        ExperimentCardBlockSpec(id="evidence", kind="evidence", source="evidence"),
    ]
    for block in required:
        if (block.kind, block.source) not in present:
            blocks.append(block)
            present.add((block.kind, block.source))

    # Keep an informative data table when a result is too incomplete for the
    # AI-requested chart. This is an additive downgrade, not a card rebuild.
    mode = design.analysis.mode
    if mode == "ordered_trend":
        points = round_summary.get("ordered_trend") or []
        if sum(item.get("mean") is not None for item in points) < 2 and not any(
            block.kind == "table" and block.source == "ordered_trend" for block in blocks
        ):
            blocks.append(ExperimentCardBlockSpec(id="trend-data", kind="table", source="ordered_trend"))
    elif mode == "factor_effects":
        interaction = round_summary.get("interaction_summary") or []
        if interaction and not _has_complete_interaction_data(design, interaction) and not any(
            block.kind == "table" and block.source == "interaction_summary" for block in blocks
        ):
            blocks.append(ExperimentCardBlockSpec(id="interaction-data", kind="table", source="interaction_summary"))
        if not round_summary.get("factor_effects") and not any(
            block.kind == "table" and block.source in {"condition_statistics", "factor_effects"}
            for block in blocks
        ):
            blocks.append(ExperimentCardBlockSpec(id="factor-data", kind="table", source="condition_statistics"))

    if not any(
        block.kind in {"metrics", "chart", "table", "runs", "diagnostics", "insight", "key_value", "timeline"}
        for block in blocks
    ):
        for candidate in default_presentation_spec(design).blocks:
            if candidate.kind in {"narrative", "progress", "evidence"}:
                continue
            if _block_has_observed_data(candidate, design, round_summary):
                blocks.append(candidate)
                break
        else:
            blocks.append(ExperimentCardBlockSpec(id="runs", kind="runs", source="runs"))

    if round_summary.get("failed_run_ids"):
        for block in (
            ExperimentCardBlockSpec(id="runs", kind="runs", source="runs"),
            ExperimentCardBlockSpec(id="diagnostics", kind="diagnostics", source="diagnostics"),
        ):
            if (block.kind, block.source) not in present:
                blocks.append(block)
                present.add((block.kind, block.source))
    return ExperimentCardPresentationSpec(
        layout=source_spec.layout,
        density=source_spec.density,
        blocks=blocks[:MAX_PRESENTATION_BLOCKS],
    )


def is_result_compatible_presentation_spec(
    spec: ExperimentCardPresentationSpec,
    design: ExperimentDesignSpec,
    round_summary: dict[str, Any],
) -> bool:
    """Check that a model-selected layout matches observed data availability."""

    blocks = spec.blocks
    if round_summary.get("failed_run_ids"):
        required = {"runs", "diagnostics", "evidence"}
        if not required.issubset({block.kind for block in blocks}):
            return False

    if design.analysis.mode == "ordered_trend":
        valid_points = [
            point for point in (round_summary.get("ordered_trend") or [])
            if point.get("mean") is not None
        ]
        if len(valid_points) < 2:
            if any(
                block.kind == "chart"
                and block.source == "ordered_trend"
                and block.chart_mark in {"line", "point"}
                for block in blocks
            ):
                return False
            if not any(
                block.kind == "table" and block.source == "ordered_trend"
                for block in blocks
            ):
                return False

    interaction_summary = round_summary.get("interaction_summary") or []
    interaction_data = _has_complete_interaction_data(design, interaction_summary)
    if not interaction_data and any(
        block.kind == "chart"
        and block.source == "interaction_summary"
        and block.chart_mark == "heatmap"
        for block in blocks
    ):
        return False
    if design.analysis.mode != "factor_effects" or interaction_data:
        return True
    allowed_table_sources = (
        {"interaction_summary"}
        if interaction_summary
        else {"condition_statistics", "factor_effects"}
    )
    return any(
        block.kind == "table" and block.source in allowed_table_sources
        for block in blocks
    )


def _has_complete_interaction_data(
    design: ExperimentDesignSpec,
    interaction_summary: list[dict[str, Any]],
) -> bool:
    return any(
        _has_complete_interaction_item(design, item) for item in interaction_summary
    )


def _has_complete_interaction_item(
    design: ExperimentDesignSpec,
    item: dict[str, Any],
) -> bool:
    factor_a = next(
        (factor for factor in design.factors if factor.name == item.get("factor_a")),
        None,
    )
    factor_b = next(
        (factor for factor in design.factors if factor.name == item.get("factor_b")),
        None,
    )
    if factor_a is None or factor_b is None:
        return False
    cell_means = item.get("cell_means")
    if not isinstance(cell_means, dict):
        return False
    required = {
        f"{factor_a.name}={level_a}|{factor_b.name}={level_b}"
        for level_a, level_b in itertools.product(factor_a.levels, factor_b.levels)
    }
    return bool(required) and all(
        key in cell_means and cell_means[key] is not None for key in required
    )


def _allowed_levels(
    field: str | None,
    plan: ExperimentPlan,
    *,
    allowed_categories: set[str] | None,
    allowed_detectors: set[str] | None,
    allowed_strategies: set[str] | None,
) -> list[Any]:
    if field == "selection_strategy":
        return list(allowed_strategies or plan.selection_strategies)
    if field == "detector":
        return list(allowed_detectors or plan.detectors)
    if field == "category":
        values = list(allowed_categories or plan.categories)
        return [value for value in values if value in plan.categories]
    if field == "shots":
        return list(plan.shots)
    if field == "seed":
        return list(plan.seeds)
    if field == "protocol":
        return list(plan.protocols)
    return []


def _factorial_size(design: ExperimentDesignSpec) -> int:
    return max(1, len(list(itertools.product(*(factor.levels for factor in design.factors)))))


def _is_orderable(value: Any) -> bool:
    return isinstance(value, (int, float, str)) and not isinstance(value, bool)


def design_fingerprint(design: ExperimentDesignSpec) -> str:
    """Return a stable JSON representation useful to callers and tests."""

    return json.dumps(design.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)


validate_experiment_design = validate_design
validate_experiment_plan_designs = validate_plan_designs
