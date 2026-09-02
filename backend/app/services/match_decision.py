from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable


class MatchDecision(str, Enum):
    MATCH = "match"
    POSSIBLE_MATCH = "possible_match"
    NON_MATCH = "non_match"


@dataclass(frozen=True)
class MatchDecisionThresholds:
    """
    Probability thresholds used to convert Splink probabilities
    into application-level match decisions.
    """

    match_threshold: float = 0.90
    possible_match_threshold: float = 0.50

    def __post_init__(self) -> None:
        if not 0.0 <= self.possible_match_threshold <= 1.0:
            raise ValueError(
                "possible_match_threshold must be between 0.0 and 1.0."
            )

        if not 0.0 <= self.match_threshold <= 1.0:
            raise ValueError(
                "match_threshold must be between 0.0 and 1.0."
            )

        if self.possible_match_threshold > self.match_threshold:
            raise ValueError(
                "possible_match_threshold cannot be greater than "
                "match_threshold."
            )


@dataclass(frozen=True)
class MatchDecisionResult:
    record_a_id: int
    record_b_id: int
    match_probability: float
    decision: MatchDecision


def classify_match_probability(
    match_probability: float,
    thresholds: MatchDecisionThresholds | None = None,
) -> MatchDecision:
    """
    Convert a Splink match probability into an application decision.
    """

    if not 0.0 <= match_probability <= 1.0:
        raise ValueError("match_probability must be between 0.0 and 1.0.")

    thresholds = thresholds or MatchDecisionThresholds()

    if match_probability >= thresholds.match_threshold:
        return MatchDecision.MATCH

    if match_probability >= thresholds.possible_match_threshold:
        return MatchDecision.POSSIBLE_MATCH

    return MatchDecision.NON_MATCH


def classify_match_results(
    results: Iterable[tuple[int, int, float]],
    thresholds: MatchDecisionThresholds | None = None,
) -> list[MatchDecisionResult]:
    """
    Classify multiple Splink prediction results.
    """

    thresholds = thresholds or MatchDecisionThresholds()

    classified: list[MatchDecisionResult] = []

    for record_a_id, record_b_id, probability in results:
        if record_a_id == record_b_id:
            continue

        probability = float(probability)

        classified.append(
            MatchDecisionResult(
                record_a_id=min(record_a_id, record_b_id),
                record_b_id=max(record_a_id, record_b_id),
                match_probability=probability,
                decision=classify_match_probability(
                    probability,
                    thresholds,
                ),
            )
        )

    classified.sort(
        key=lambda item: (
            -item.match_probability,
            item.record_a_id,
            item.record_b_id,
        )
    )

    return classified