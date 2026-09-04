from __future__ import annotations

from typing import Any

import pandas as pd
from splink import Linker, SettingsCreator, block_on
from splink.internals.duckdb.database_api import DuckDBAPI

from app.services.blocking import SplinkBlockingRule


INTERNAL_UNIQUE_ID_COLUMN = "unique_id"


def generate_candidate_pairs(
    records: dict[int, dict[str, str]],
    blocking_rules: list[SplinkBlockingRule],
) -> list[tuple[int, int]]:
    """
    Generate candidate record pairs using Splink deterministic blocking.

    Candidate generation only determines which record pairs should be
    considered by the later matching stage. It does not calculate
    match probabilities or make match/non-match decisions.
    """
    if not records:
        raise ValueError("Cannot generate candidates from an empty dataset.")

    if not blocking_rules:
        raise ValueError("At least one blocking rule is required.")

    dataframe = _records_to_dataframe(records)

    splink_rules = []

    for rule in blocking_rules:
        if rule.strategy != "exact":
            raise ValueError(
                f"Unsupported Splink blocking strategy: {rule.strategy}"
            )

        if not rule.fields:
            raise ValueError("Blocking rule must contain at least one field.")

        missing_fields = [
            field for field in rule.fields
            if field not in dataframe.columns
        ]

        if missing_fields:
            raise ValueError(
                "Blocking rule references missing fields: "
                + ", ".join(missing_fields)
            )

        splink_rules.append(block_on(*rule.fields))

    settings = SettingsCreator(
        link_type="dedupe_only",
        blocking_rules_to_generate_predictions=splink_rules,
        unique_id_column_name=INTERNAL_UNIQUE_ID_COLUMN,
    )

    linker = Linker(
        dataframe,
        settings,
        db_api=DuckDBAPI(),
        set_up_basic_logging=False,
    )

    candidate_dataframe = (
        linker.inference
        .deterministic_link()
        .as_pandas_dataframe()
    )

    return _extract_candidate_pairs(candidate_dataframe)


def _records_to_dataframe(
    records: dict[int, dict[str, str]],
) -> pd.DataFrame:
    """
    Convert the prepared Dedupe records into a Splink input DataFrame.
    """
    rows: list[dict[str, Any]] = []

    for record_id, record in records.items():
        if not isinstance(record, dict):
            raise ValueError(
                f"Record {record_id} must be represented as a dictionary."
            )

        row = dict(record)

        if INTERNAL_UNIQUE_ID_COLUMN in row:
            raise ValueError(
                f"Dataset contains reserved internal column "
                f"'{INTERNAL_UNIQUE_ID_COLUMN}'."
            )

        row[INTERNAL_UNIQUE_ID_COLUMN] = int(record_id)
        rows.append(row)

    dataframe = pd.DataFrame(rows)

    if dataframe.empty:
        raise ValueError("Cannot generate candidates from an empty dataset.")

    return dataframe


def _extract_candidate_pairs(
    candidate_dataframe: pd.DataFrame,
) -> list[tuple[int, int]]:
    """
    Extract Splink's left/right unique IDs as stable candidate pairs.
    """
    required_columns = {
        f"{INTERNAL_UNIQUE_ID_COLUMN}_l",
        f"{INTERNAL_UNIQUE_ID_COLUMN}_r",
    }

    missing_columns = required_columns - set(candidate_dataframe.columns)

    if missing_columns:
        raise ValueError(
            "Splink candidate output is missing expected columns: "
            + ", ".join(sorted(missing_columns))
        )

    pairs: set[tuple[int, int]] = set()

    for _, row in candidate_dataframe.iterrows():
        record_a_id = int(row[f"{INTERNAL_UNIQUE_ID_COLUMN}_l"])
        record_b_id = int(row[f"{INTERNAL_UNIQUE_ID_COLUMN}_r"])

        if record_a_id == record_b_id:
            continue

        pair = tuple(sorted((record_a_id, record_b_id)))
        pairs.add(pair)

    return sorted(pairs)