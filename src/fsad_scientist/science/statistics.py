from __future__ import annotations

import itertools
import math
import random
from statistics import fmean
from typing import Any, Literal

from pydantic import BaseModel, Field

from fsad_scientist.domain.enums import RunStatus
from fsad_scientist.domain.models import ExperimentRun


class PairedEffectResult(BaseModel):
    metric: str
    treatment: str
    control: str
    differences: list[float]
    pair_run_ids: list[tuple[str, str]]
    mean_difference: float
    confidence_interval: tuple[float, float]
    permutation_p_value: float = Field(ge=0, le=1)
    bootstrap_samples: int
    permutation_samples: int

    @property
    def pair_count(self) -> int:
        return len(self.differences)


def compare_paired_runs(
    runs: list[ExperimentRun],
    *,
    hypothesis_id: str,
    metric: str,
    treatment: str,
    control: str,
    alpha: float = 0.05,
    bootstrap_samples: int = 10_000,
    permutation_samples: int = 20_000,
    random_seed: int = 17,
) -> PairedEffectResult:
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if bootstrap_samples < 100:
        raise ValueError("bootstrap_samples must be at least 100")
    if permutation_samples < 100:
        raise ValueError("permutation_samples must be at least 100")

    grouped: dict[tuple[object, ...], dict[str, tuple[str, float]]] = {}
    for run in runs:
        if (
            run.hypothesis_id != hypothesis_id
            or run.status != RunStatus.SUCCEEDED
            or not run.verified
            or metric not in run.metrics
        ):
            continue
        key = (
            run.protocol,
            run.dataset,
            run.category,
            run.detector,
            run.shots,
            run.seed,
        )
        grouped.setdefault(key, {})[run.selection_strategy] = (run.id, run.metrics[metric])

    differences: list[float] = []
    pairs: list[tuple[str, str]] = []
    for strategies in grouped.values():
        if treatment not in strategies or control not in strategies:
            continue
        treatment_id, treatment_value = strategies[treatment]
        control_id, control_value = strategies[control]
        difference = treatment_value - control_value
        if not math.isfinite(difference):
            continue
        differences.append(difference)
        pairs.append((treatment_id, control_id))
    if len(differences) < 2:
        raise ValueError("at least two verified paired differences are required")

    generator = random.Random(random_seed)
    bootstrap_means = [
        fmean(generator.choice(differences) for _ in differences)
        for _ in range(bootstrap_samples)
    ]
    confidence_interval = (
        _quantile(bootstrap_means, alpha / 2),
        _quantile(bootstrap_means, 1 - alpha / 2),
    )
    permutation_p_value, actual_permutations = _sign_flip_p_value(
        differences,
        max_samples=permutation_samples,
        generator=generator,
    )
    return PairedEffectResult(
        metric=metric,
        treatment=treatment,
        control=control,
        differences=differences,
        pair_run_ids=pairs,
        mean_difference=fmean(differences),
        confidence_interval=confidence_interval,
        permutation_p_value=permutation_p_value,
        bootstrap_samples=bootstrap_samples,
        permutation_samples=actual_permutations,
    )


def _sign_flip_p_value(
    differences: list[float],
    *,
    max_samples: int,
    generator: random.Random,
) -> tuple[float, int]:
    observed = abs(fmean(differences))
    sample_count = 2 ** len(differences)
    if sample_count <= max_samples:
        means = (
            abs(fmean(sign * value for sign, value in zip(signs, differences, strict=True)))
            for signs in itertools.product((-1, 1), repeat=len(differences))
        )
        extreme = sum(value >= observed - 1e-15 for value in means)
        return extreme / sample_count, sample_count

    extreme = 0
    for _ in range(max_samples):
        permuted = fmean(
            (-value if generator.random() < 0.5 else value) for value in differences
        )
        extreme += abs(permuted) >= observed - 1e-15
    return (extreme + 1) / (max_samples + 1), max_samples


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


class SignificanceLevel(BaseModel):
    """统计显著性等级"""
    level: Literal[
        "highly_significant",  # p < 0.001
        "significant",        # p < 0.01
        "marginally_significant",  # p < 0.05
        "not_significant",    # p >= 0.05
        "insufficient",       # 样本不足
    ]
    confidence: Literal["高", "中", "低"]
    p_value: float | None = None
    interpretation: str


def calculate_significance_level(
    pair_count: int,
    mean_difference: float,
    positive_fraction: float,
    p_value: float | None = None,
) -> SignificanceLevel:
    """计算统计显著性等级

    Args:
        pair_count: 有效配对数
        mean_difference: 平均效应量
        positive_fraction: 正向配对比例 (0-1)
        p_value: 置换检验 p 值 (可选)

    Returns:
        SignificanceLevel: 包含等级、置信度和解释的对象
    """
    if pair_count < 2:
        return SignificanceLevel(
            level="insufficient",
            confidence="低",
            p_value=None,
            interpretation="样本不足，需要至少 2 个配对才能进行统计检验。",
        )

    if p_value is not None:
        if p_value < 0.001:
            return SignificanceLevel(
                level="highly_significant",
                confidence="高",
                p_value=p_value,
                interpretation=f"结果极其显著（p = {p_value:.4f}），可以强烈拒绝零假设。",
            )
        elif p_value < 0.01:
            return SignificanceLevel(
                level="significant",
                confidence="高",
                p_value=p_value,
                interpretation=f"结果非常显著（p = {p_value:.4f}），可以拒绝零假设。",
            )
        elif p_value < 0.05:
            return SignificanceLevel(
                level="marginally_significant",
                confidence="中",
                p_value=p_value,
                interpretation=f"结果显著（p = {p_value:.4f}），可以拒绝零假设。",
            )
        else:
            return SignificanceLevel(
                level="not_significant",
                confidence="中",
                p_value=p_value,
                interpretation=f"结果不显著（p = {p_value:.4f}），无法拒绝零假设。",
            )

    if positive_fraction >= 0.95 and pair_count >= 6:
        return SignificanceLevel(
            level="highly_significant",
            confidence="高",
            p_value=None,
            interpretation="正向比例高达 95% 以上，配对数充足，结果极其显著。",
        )
    elif positive_fraction >= 0.80 and pair_count >= 4:
        return SignificanceLevel(
            level="significant",
            confidence="中",
            p_value=None,
            interpretation="正向比例达 80%，配对数充足，结果显著。",
        )
    elif positive_fraction >= 0.60 and pair_count >= 3:
        return SignificanceLevel(
            level="marginally_significant",
            confidence="低",
            p_value=None,
            interpretation="正向比例约 60%，配对数较少，边缘显著。",
        )
    elif mean_difference > 0 and positive_fraction > 0.5:
        return SignificanceLevel(
            level="not_significant",
            confidence="低",
            p_value=None,
            interpretation="效应为正但样本量不足或正向比例不够高，无法得出显著结论。",
        )
    else:
        return SignificanceLevel(
            level="not_significant",
            confidence="低",
            p_value=None,
            interpretation="未观察到显著效应。",
        )


def calculate_efficiency_metrics(
    selected_runs: int,
    exhaustive_run_count: int,
    cumulative_pairs: int,
    minimum_pairs: int,
) -> dict[str, Any]:
    """计算实验效率指标

    Args:
        selected_runs: 已选择的实验运行数
        exhaustive_run_count: 穷举所需的实验运行数
        cumulative_pairs: 累计配对数
        minimum_pairs: 最小要求配对数

    Returns:
        dict: 效率指标字典
    """
    runs_avoided = max(exhaustive_run_count - selected_runs, 0)
    savings_ratio = runs_avoided / exhaustive_run_count if exhaustive_run_count > 0 else 0.0
    pairs_progress = cumulative_pairs / minimum_pairs if minimum_pairs > 0 else 0.0

    return {
        "selected_runs": selected_runs,
        "exhaustive_run_count": exhaustive_run_count,
        "runs_avoided": runs_avoided,
        "savings_ratio": round(savings_ratio, 4),
        "savings_percentage": round(savings_ratio * 100, 2),
        "cumulative_pairs": cumulative_pairs,
        "minimum_pairs": minimum_pairs,
        "pairs_progress": round(min(pairs_progress, 1.0), 4),
        "pairs_percentage": round(min(pairs_progress * 100, 100), 2),
    }
