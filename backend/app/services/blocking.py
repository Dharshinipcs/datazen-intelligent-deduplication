from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LearnedDedupePattern:
    """
    A pattern learned by Dedupe during active learning.

    This is NOT a final blocking rule.

    It represents evidence learned from Dedupe that our application
    can later interpret when generating downstream blocking rules.
    """

    fields: tuple[str, ...]
    strategy: str


@dataclass(frozen=True)
class SplinkBlockingRule:
    """
    Application-level representation of a Splink blocking rule.

    The rule is deliberately represented independently of the
    Splink runtime so that rule generation can be tested without
    requiring a Splink linker.
    """

    fields: tuple[str, ...]
    strategy: str
    sql_condition: str


@dataclass(frozen=True)
class BlockingStrategy:
    """
    Final blocking strategy generated for downstream candidate
    generation.

    The source records where the strategy came from.
    """

    rules: tuple[SplinkBlockingRule, ...]
    source: str = "dedupe_pattern_generator"


def build_blocking_strategy(
    rules: list[SplinkBlockingRule],
) -> BlockingStrategy:
    """
    Build the final application-level blocking strategy.

    These rules are intended for downstream Splink candidate
    generation and are NOT Dedupe predicates.
    """

    if not rules:
        raise ValueError("At least one Splink blocking rule is required.")

    return BlockingStrategy(
        rules=tuple(rules),
    )


def generate_splink_blocking_rules(
    patterns: list[LearnedDedupePattern],
) -> list[SplinkBlockingRule]:
    """
    Convert learned Dedupe patterns into efficient Splink
    blocking rules.

    Important:
        Dedupe learns comparison/matching patterns.
        This function interprets those patterns and generates
        independent Splink blocking rules.

    Current conservative strategy:

        single-field pattern
            -> exact equality blocking rule

    This is particularly appropriate for normalized fields such as
    email, where exact equality provides a highly selective and
    inexpensive candidate-generation condition.

    Multi-field patterns are represented as compound equality rules.
    """

    if not patterns:
        raise ValueError(
            "No learned Dedupe patterns are available "
            "to generate blocking rules."
        )

    generated_rules: list[SplinkBlockingRule] = []

    for pattern in patterns:
        fields = tuple(
            field
            for field in pattern.fields
            if isinstance(field, str) and field.strip()
        )

        if not fields:
            continue

        sql_condition = _build_exact_sql_condition(fields)

        generated_rules.append(
            SplinkBlockingRule(
                fields=fields,
                strategy="exact",
                sql_condition=sql_condition,
            )
        )

    return _deduplicate_splink_rules(generated_rules)


def _build_exact_sql_condition(
    fields: tuple[str, ...],
) -> str:
    """
    Build a Splink-compatible SQL equality condition.

    Examples:

        ("email",)
            -> "l.email = r.email"

        ("email", "city")
            -> "l.email = r.email AND l.city = r.city"
    """

    conditions = [
        f"l.{_quote_identifier(field)} = r.{_quote_identifier(field)}"
        for field in fields
    ]

    return " AND ".join(conditions)


def _quote_identifier(field: str) -> str:
    """
    Validate a column identifier before placing it into SQL.

    Column names originate from the uploaded dataset, so they must
    not be interpolated into SQL without validation.
    """

    field = field.strip()

    if not field:
        raise ValueError("Blocking field name cannot be empty.")

    if not all(
        character.isalnum() or character == "_"
        for character in field
    ):
        raise ValueError(
            f"Unsupported column name for SQL blocking rule: {field}"
        )

    return field


def _deduplicate_splink_rules(
    rules: list[SplinkBlockingRule],
) -> list[SplinkBlockingRule]:
    """Remove duplicate generated rules while preserving order."""

    seen: set[tuple[tuple[str, ...], str, str]] = set()
    unique_rules: list[SplinkBlockingRule] = []

    for rule in rules:
        key = (
            rule.fields,
            rule.strategy,
            rule.sql_condition,
        )

        if key in seen:
            continue

        seen.add(key)
        unique_rules.append(rule)

    return unique_rules