from __future__ import annotations

import itertools
import math
import random
from statistics import fmean

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
