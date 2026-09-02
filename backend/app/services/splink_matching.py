from __future__ import annotations

from typing import Any

import pandas as pd
from splink import Linker, SettingsCreator, block_on
from splink.comparison_library import (
    EmailComparison,
    ExactMatch,
    NameComparison,
)
from splink.internals.duckdb.database_api import DuckDBAPI

from app.models.schemas import DatasetSchema
from app.services.blocking import SplinkBlockingRule


INTERNAL_UNIQUE_ID_COLUMN = "unique_id"


def run_splink_matching(
    records: dict[int, dict[str, str]],
    blocking_rules: list[SplinkBlockingRule],
    schema: DatasetSchema,
) -> pd.DataFrame:
    """
    Run Splink probabilistic matching on blocked candidate pairs.

    Splink comparisons are selected from the detected semantic schema,
    not from raw column names.

    Pipeline:

        records
            ↓
        semantic schema
            ↓
        field comparisons
            ↓
        blocking rules
            ↓
        EM parameter estimation
            ↓
        probabilistic prediction
    """

    if not records:
        raise ValueError(
            "Cannot run matching on an empty dataset."
        )

    if not blocking_rules:
        raise ValueError(
            "At least one blocking rule is required."
        )

    dataframe = _records_to_dataframe(records)

    comparisons = _build_comparisons(
        dataframe=dataframe,
        schema=schema,
    )

    if not comparisons:
        raise ValueError(
            "No supported semantic fields were found for "
            "Splink matching."
        )

    splink_blocking_rules = _build_blocking_rules(
        dataframe=dataframe,
        blocking_rules=blocking_rules,
    )

    settings = SettingsCreator(
        link_type="dedupe_only",
        comparisons=comparisons,
        blocking_rules_to_generate_predictions=splink_blocking_rules,
        unique_id_column_name=INTERNAL_UNIQUE_ID_COLUMN,
        retain_matching_columns=True,
        retain_intermediate_calculation_columns=False,
    )

    linker = Linker(
        dataframe,
        settings,
        db_api=DuckDBAPI(),
        set_up_basic_logging=False,
    )

    _estimate_splink_parameters(
        linker=linker,
        blocking_rules=splink_blocking_rules,
    )

    predictions = linker.inference.predict()

    return predictions.as_pandas_dataframe()


def _records_to_dataframe(
    records: dict[int, dict[str, str]],
) -> pd.DataFrame:
    """Convert application records into a DataFrame."""

    rows: list[dict[str, Any]] = []

    for record_id, record in records.items():

        if not isinstance(record, dict):
            raise ValueError(
                f"Record {record_id} must be represented "
                "as a dictionary."
            )

        row = dict(record)

        if INTERNAL_UNIQUE_ID_COLUMN in row:
            raise ValueError(
                "Dataset contains reserved internal column "
                f"'{INTERNAL_UNIQUE_ID_COLUMN}'."
            )

        row[INTERNAL_UNIQUE_ID_COLUMN] = int(record_id)

        rows.append(row)

    dataframe = pd.DataFrame(rows)

    if dataframe.empty:
        raise ValueError(
            "Cannot run matching on an empty dataset."
        )

    return dataframe


def _build_comparisons(
    dataframe: pd.DataFrame,
    schema: DatasetSchema,
) -> list[Any]:
    """
    Build Splink comparisons using detected semantic types.

    Semantic mapping:

        email              -> EmailComparison
        person_name        -> NameComparison
        phone              -> ExactMatch
        address            -> ExactMatch
        city               -> ExactMatch
        organization_name  -> ExactMatch

    Identifier fields are excluded because technical identifiers
    must never be used as entity-matching evidence.

    Generic or unknown semantic types are excluded rather than
    guessed.
    """

    semantic_by_column = {
        field.column_name: field.semantic_type
        for field in schema.fields
    }

    comparisons: list[Any] = []

    for column in dataframe.columns:

        if column == INTERNAL_UNIQUE_ID_COLUMN:
            continue

        semantic_type = semantic_by_column.get(column)

        if semantic_type == "identifier":
            continue

        if semantic_type == "email":
            comparisons.append(
                EmailComparison(column)
            )

        elif semantic_type == "person_name":
            comparisons.append(
                NameComparison(column)
            )

        elif semantic_type in {
            "phone",
            "address",
            "city",
            "organization_name",
        }:
            comparisons.append(
                ExactMatch(column)
            )

    return comparisons


def _build_blocking_rules(
    dataframe: pd.DataFrame,
    blocking_rules: list[SplinkBlockingRule],
) -> list[Any]:
    """Convert application blocking rules into Splink rules."""

    splink_rules: list[Any] = []

    for rule in blocking_rules:

        if rule.strategy != "exact":
            raise ValueError(
                "Unsupported Splink blocking strategy: "
                f"{rule.strategy}"
            )

        if not rule.fields:
            raise ValueError(
                "Blocking rule must contain at least one field."
            )

        missing_fields = [
            field
            for field in rule.fields
            if field not in dataframe.columns
        ]

        if missing_fields:
            raise ValueError(
                "Blocking rule references missing fields: "
                + ", ".join(missing_fields)
            )

        splink_rules.append(
            block_on(*rule.fields)
        )

    return splink_rules


def _estimate_splink_parameters(
    linker: Linker,
    blocking_rules: list[Any],
) -> None:
    """
    Estimate Splink match prior and m/u parameters.

    The match prior is estimated first from the deterministic
    blocking rules, followed by EM estimation of comparison
    parameters.
    """

    if not blocking_rules:
        raise ValueError(
            "At least one blocking rule is required for "
            "Splink parameter estimation."
        )

    # Estimate the probability that two randomly selected records
    # belong to the same entity.
    linker.training.estimate_probability_two_random_records_match(
        deterministic_matching_rules=blocking_rules,
        recall=0.9,
    )

    # Estimate m/u parameters using the same blocking rules used
    # during candidate generation and prediction.
    for blocking_rule in blocking_rules:
        linker.training.estimate_parameters_using_expectation_maximisation(
            blocking_rule=blocking_rule,
            estimate_without_term_frequencies=False,
        )