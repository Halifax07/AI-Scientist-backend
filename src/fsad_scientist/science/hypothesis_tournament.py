from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class EvolutionOperator(StrEnum):
    REFINE = "refine"
    COMBINE = "combine"
    SIMPLIFY = "simplify"
    DIVERGE = "diverge"


@dataclass(frozen=True, slots=True)
class PairwiseJudgment:
    hypothesis_a: str
    hypothesis_b: str
    outcome_for_a: float
    confidence: float = 1.0
    rationale: str = ""

    def __post_init__(self) -> None:
        if self.outcome_for_a not in {0.0, 0.5, 1.0}:
            raise ValueError("outcome_for_a must be 0, 0.5, or 1")
        if not 0 < self.confidence <= 1:
            raise ValueError("confidence must be in (0, 1]")
        if self.hypothesis_a == self.hypothesis_b:
            raise ValueError("A hypothesis cannot debate itself")


class EloTournament:
    """Pairwise hypothesis ranking inspired by scientific debate tournaments."""

    def __init__(self, hypothesis_ids: list[str], *, initial: float = 1000, k: float = 32) -> None:
        if len(set(hypothesis_ids)) != len(hypothesis_ids):
            raise ValueError("hypothesis ids must be unique")
        self.ratings = dict.fromkeys(hypothesis_ids, float(initial))
        self.k = k
        self.history: list[PairwiseJudgment] = []

    def apply(self, judgment: PairwiseJudgment) -> dict[str, float]:
        if judgment.hypothesis_a not in self.ratings or judgment.hypothesis_b not in self.ratings:
            raise KeyError("Both hypotheses must be registered in the tournament")

        rating_a = self.ratings[judgment.hypothesis_a]
        rating_b = self.ratings[judgment.hypothesis_b]
        expected_a = 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
        expected_b = 1 - expected_a
        weighted_k = self.k * judgment.confidence

        self.ratings[judgment.hypothesis_a] = rating_a + weighted_k * (
            judgment.outcome_for_a - expected_a
        )
        self.ratings[judgment.hypothesis_b] = rating_b + weighted_k * (
            (1 - judgment.outcome_for_a) - expected_b
        )
        self.history.append(judgment)
        return self.ratings.copy()

    def ranked(self) -> list[tuple[str, float]]:
        return sorted(self.ratings.items(), key=lambda item: (-item[1], item[0]))


@dataclass(frozen=True, slots=True)
class HypothesisEvolutionRequest:
    operator: EvolutionOperator
    parent_ids: tuple[str, ...]
    unresolved_objection: str
    required_preserved_evidence_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.parent_ids:
            raise ValueError("At least one parent hypothesis is required")
        if self.operator == EvolutionOperator.COMBINE and len(self.parent_ids) < 2:
            raise ValueError("COMBINE requires at least two parents")

