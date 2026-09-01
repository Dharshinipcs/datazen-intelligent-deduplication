from pathlib import Path

import pandas as pd

from app.models.schemas import (
    DatasetSchema,
    DatasetStandardizationPlan,
    StandardizedField,
)


# Semantic types that currently have explicit standardization rules.
STANDARDIZATION_RULES = {
    "identifier": [
        "trim",
    ],
    "person_name": [
        "trim",
        "collapse_whitespace",
        "lowercase",
    ],
    "email": [
        "trim",
        "lowercase",
    ],
}


def build_standardization_plan(
    dataset_id: str,
    schema: DatasetSchema,
) -> DatasetStandardizationPlan:
    """
    Build a semantic-driven standardization plan.

    The detected semantic type determines which transformations
    should be applied to each dataset column.
    """
    fields = []

    for field in schema.fields:
        transformations = STANDARDIZATION_RULES.get(
            field.semantic_type,
            ["trim"],
        )

        fields.append(
            StandardizedField(
                column_name=field.column_name,
                semantic_type=field.semantic_type,
                transformations=transformations,
            )
        )

    return DatasetStandardizationPlan(
        dataset_id=dataset_id,
        fields=fields,
    )


def _trim(value):
    """Remove leading and trailing whitespace from a value."""
    if pd.isna(value):
        return value

    return str(value).strip()


def _collapse_whitespace(value):
    """Replace consecutive whitespace with a single space."""
    if pd.isna(value):
        return value

    return " ".join(str(value).split())


def _lowercase(value):
    """Convert textual values to lowercase."""
    if pd.isna(value):
        return value

    return str(value).lower()


def _apply_transformation(value, transformation: str):
    """Apply one supported transformation to a value."""
    if transformation == "trim":
        return _trim(value)

    if transformation == "collapse_whitespace":
        return _collapse_whitespace(value)

    if transformation == "lowercase":
        return _lowercase(value)

    raise ValueError(
        f"Unsupported standardization transformation: {transformation}"
    )


def apply_standardization(
    df: pd.DataFrame,
    plan: DatasetStandardizationPlan,
) -> pd.DataFrame:
    """
    Apply a standardization plan to a copy of the dataset.

    The input DataFrame is never modified directly.
    """
    standardized_df = df.copy()

    for field in plan.fields:
        column_name = field.column_name

        if column_name not in standardized_df.columns:
            raise ValueError(
                f"Column '{column_name}' from the standardization plan "
                "does not exist in the dataset."
            )

        for transformation in field.transformations:
            standardized_df[column_name] = standardized_df[
                column_name
            ].map(
                lambda value: _apply_transformation(
                    value,
                    transformation,
                )
            )

    return standardized_df