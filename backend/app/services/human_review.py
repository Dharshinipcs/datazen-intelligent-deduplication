from __future__ import annotations

from dataclasses import dataclass
from threading import Lock


@dataclass(frozen=True)
class HumanReviewDecision:
    record_a_id: int
    record_b_id: int
    decision: str


class HumanReviewStore:
    """
    In-memory store for Human Review decisions.

    Decisions are keyed by dataset and record pair.

    Prototype limitation:
        Review decisions disappear when the backend process restarts.
    """

    def __init__(self) -> None:
        self._decisions: dict[
            str,
            dict[tuple[int, int], HumanReviewDecision],
        ] = {}
        self._lock = Lock()

    def save(
        self,
        dataset_id: str,
        record_a_id: int,
        record_b_id: int,
        decision: str,
    ) -> HumanReviewDecision:
        if record_a_id == record_b_id:
            raise ValueError(
                "Human Review requires two different records."
            )

        if decision not in {"match", "non_match"}:
            raise ValueError(
                "Human Review decision must be 'match' or 'non_match'."
            )

        normalized_a = min(record_a_id, record_b_id)
        normalized_b = max(record_a_id, record_b_id)

        review_decision = HumanReviewDecision(
            record_a_id=normalized_a,
            record_b_id=normalized_b,
            decision=decision,
        )

        with self._lock:
            dataset_decisions = self._decisions.setdefault(
                dataset_id,
                {},
            )
            dataset_decisions[
                (normalized_a, normalized_b)
            ] = review_decision

        return review_decision

    def get(
        self,
        dataset_id: str,
        record_a_id: int,
        record_b_id: int,
    ) -> HumanReviewDecision | None:
        normalized_a = min(record_a_id, record_b_id)
        normalized_b = max(record_a_id, record_b_id)

        with self._lock:
            dataset_decisions = self._decisions.get(
                dataset_id,
                {},
            )
            return dataset_decisions.get(
                (normalized_a, normalized_b)
            )

    def is_reviewed(
        self,
        dataset_id: str,
        record_a_id: int,
        record_b_id: int,
    ) -> bool:
        return (
            self.get(
                dataset_id,
                record_a_id,
                record_b_id,
            )
            is not None
        )

    def get_all(
        self,
        dataset_id: str,
    ) -> list[HumanReviewDecision]:
        with self._lock:
            dataset_decisions = self._decisions.get(
                dataset_id,
                {}
            )
            return list(dataset_decisions.values())

    def clear(self, dataset_id: str) -> None:
        with self._lock:
            self._decisions.pop(dataset_id, None)


human_review_store = HumanReviewStore()