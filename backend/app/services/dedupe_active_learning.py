from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import dedupe

from app.services.blocking import (
    LearnedDedupePattern,
)


@dataclass
class ActiveLearningSession:
    dataset_id: str
    linker: Any
    records: dict[int, dict[str, str]]
    sample_size: int
    blocked_proportion: float


def create_active_learning_session(
    dataset_id: str,
    linker: Any,
    records: dict[int, dict[str, str]],
    sample_size: int = 1500,
    blocked_proportion: float = 0.9,
) -> ActiveLearningSession:
    """
    Initialize Dedupe's active-learning process for a dataset.

    Dedupe samples candidate pairs internally. The application can
    then request uncertain pairs and ask the user to label them.
    """

    if not records:
        raise ValueError(
            "Cannot initialize active learning with no records."
        )

    if sample_size <= 0:
        raise ValueError(
            "sample_size must be greater than zero."
        )

    if not 0.0 <= blocked_proportion <= 1.0:
        raise ValueError(
            "blocked_proportion must be between 0.0 and 1.0."
        )

    linker.prepare_training(
        records,
        sample_size=sample_size,
        blocked_proportion=blocked_proportion,
    )

    return ActiveLearningSession(
        dataset_id=dataset_id,
        linker=linker,
        records=records,
        sample_size=sample_size,
        blocked_proportion=blocked_proportion,
    )


def get_uncertain_pairs(
    session: ActiveLearningSession,
    limit: int = 10,
) -> list[tuple[dict[str, str], dict[str, str]]]:
    """
    Return the record pairs Dedupe currently considers most
    informative to label.
    """

    if limit <= 0:
        raise ValueError(
            "limit must be greater than zero."
        )

    pairs = session.linker.uncertain_pairs()

    return list(pairs[:limit])


def mark_labeled_pairs(
    session: ActiveLearningSession,
    matches: list[tuple[dict[str, str], dict[str, str]]],
    distinct: list[tuple[dict[str, str], dict[str, str]]],
) -> None:
    """
    Feed user Match/Distinct labels back into Dedupe.
    """

    session.linker.mark_pairs(
        {
            "match": matches,
            "distinct": distinct,
        }
    )


def train_dedupe_model(
    session: ActiveLearningSession,
    recall: float = 1.0,
    index_predicates: bool = True,
) -> None:
    """
    Train Dedupe's pairwise classifier and learned predicates.

    The resulting predicates are learned matching patterns.
    They are NOT directly exposed as final application blocking
    rules.
    """

    if not 0.0 <= recall <= 1.0:
        raise ValueError(
            "recall must be between 0.0 and 1.0."
        )

    session.linker.train(
        recall=recall,
        index_predicates=index_predicates,
    )


def get_learned_predicates(
    session: ActiveLearningSession,
) -> list[Any]:
    """
    Extract the raw predicates learned by Dedupe after training.

    These remain internal Dedupe artifacts.
    """

    linker = session.linker

    if not hasattr(linker, "predicates"):
        return []

    predicates = getattr(linker, "predicates")

    if predicates is None:
        return []

    return list(predicates)


def get_learned_dedupe_patterns(
    session: ActiveLearningSession,
) -> list[LearnedDedupePattern]:
    """
    Convert Dedupe's learned predicates into neutral application-level
    learned patterns.

    These patterns describe what Dedupe learned, but deliberately do
    NOT claim to be Splink blocking rules.
    """

    patterns: list[LearnedDedupePattern] = []

    for predicate in get_learned_predicates(session):
        pattern = predicate_to_learned_pattern(predicate)

        if pattern is not None:
            patterns.append(pattern)

    return _deduplicate_patterns(patterns)


def predicate_to_learned_pattern(
    predicate: Any,
) -> LearnedDedupePattern | None:
    """
    Convert a Dedupe predicate into a neutral learned pattern.

    Unsupported predicate structures are ignored rather than being
    incorrectly converted into blocking rules.
    """

    field_names = _extract_predicate_fields(predicate)

    if not field_names:
        return None

    strategy = _infer_strategy(predicate)

    return LearnedDedupePattern(
        fields=tuple(field_names),
        strategy=strategy,
    )


def _extract_predicate_fields(
    predicate: Any,
) -> list[str]:
    """
    Extract field names from Dedupe predicate objects.

    Simple predicates expose a field attribute.
    Compound predicates can contain nested predicates.
    """

    if hasattr(predicate, "field"):
        field = getattr(predicate, "field")

        if isinstance(field, str):
            return [field]

    if hasattr(predicate, "fields"):
        fields = getattr(predicate, "fields")

        if isinstance(fields, (list, tuple)):
            return [
                field
                for field in fields
                if isinstance(field, str)
            ]

    if hasattr(predicate, "predicates"):
        nested = getattr(predicate, "predicates")

        if isinstance(nested, (list, tuple)):
            fields: list[str] = []

            for child in nested:
                fields.extend(
                    _extract_predicate_fields(child)
                )

            return list(dict.fromkeys(fields))

    return []


def _infer_strategy(
    predicate: Any,
) -> str:
    """
    Describe the kind of pattern learned by Dedupe.

    This describes the Dedupe-side learning result only.
    """

    name = type(predicate).__name__.lower()

    if "compound" in name:
        return "compound"

    if "levenshtein" in name:
        return "fuzzy"

    if "tfidf" in name:
        return "tfidf"

    if "search" in name:
        return "search"

    if "canopy" in name:
        return "canopy"

    if "exists" in name:
        return "exists"

    if "index" in name:
        return "index"

    if "exact" in name:
        return "exact"

    return "learned"


def _deduplicate_patterns(
    patterns: list[LearnedDedupePattern],
) -> list[LearnedDedupePattern]:
    """
    Remove duplicate learned patterns while preserving order.
    """

    seen: set[tuple[tuple[str, ...], str]] = set()
    unique_patterns: list[LearnedDedupePattern] = []

    for pattern in patterns:
        key = (
            pattern.fields,
            pattern.strategy,
        )

        if key in seen:
            continue

        seen.add(key)
        unique_patterns.append(pattern)

    return unique_patterns