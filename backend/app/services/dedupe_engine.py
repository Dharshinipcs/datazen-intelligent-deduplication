from __future__ import annotations

from typing import Any

import dedupe
import pandas as pd

from app.models.schemas import DatasetSchema


# Semantic types that are currently useful for entity resolution.
# They are mapped to Dedupe 3.x variable objects below.
SUPPORTED_SEMANTIC_TYPES = {
    "person_name",
    "email",
    "identifier",
    "phone",
    "address",
    "organization_name",
}


def build_dedupe_fields(
    schema: DatasetSchema,
) -> list[Any]:
    """
    Convert the detected semantic schema into Dedupe 3.x
    variable objects.

    Unknown semantic types are ignored rather than guessed.
    """
    fields: list[Any] = []

    for field in schema.fields:
        if field.semantic_type not in SUPPORTED_SEMANTIC_TYPES:
            continue

        # All currently supported entity-resolution fields are
        # represented as strings. Dedupe's variable class handles
        # the comparison logic during learning.
        fields.append(
            dedupe.variables.String(field.column_name)
        )

    if not fields:
        raise ValueError(
            "No supported entity-resolution fields were detected."
        )

    return fields


def dataframe_to_dedupe_records(
    df: pd.DataFrame,
    fields: list[Any],
) -> dict[int, dict[str, str]]:
    """
    Convert a standardized DataFrame into the record format expected
    by Dedupe.

    Record IDs are stable integer positions within the standardized
    dataset.
    """
    records: dict[int, dict[str, str]] = {}

    columns = [
        field.field
        for field in fields
    ]

    for row_index, row in df.reset_index(drop=True).iterrows():
        record: dict[str, str] = {}

        for column in columns:
            value = row[column]

            if pd.isna(value):
                record[column] = ""
            else:
                record[column] = str(value)

        records[int(row_index)] = record

    if not records:
        raise ValueError(
            "Cannot create Dedupe records from an empty dataset."
        )

    return records


def create_dedupe_linker(
    fields: list[Any],
) -> dedupe.Dedupe:
    """
    Create a Dedupe 3.x active-learning linker.
    """
    return dedupe.Dedupe(fields)
