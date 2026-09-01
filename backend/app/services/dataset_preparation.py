from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.models.schemas import (
    DatasetSchema,
    DatasetStandardizationPlan,
)
from app.services.profiling import load_dataset, profile_dataset
from app.services.schema_detection import detect_schema
from app.services.standardization import (
    apply_standardization,
    build_standardization_plan,
)


def prepare_dataset(
    file_path: Path,
    file_extension: str,
    dataset_id: str,
) -> tuple[
    pd.DataFrame,
    DatasetSchema,
    DatasetStandardizationPlan,
]:
    """
    Load, profile, detect semantic schema, and standardize a dataset.

    Returns:
        standardized_df:
            Standardized copy of the original dataset.

        schema:
            Detected semantic schema.

        plan:
            Standardization plan used to produce standardized_df.

    The original dataset on disk is never modified.
    """

    # 1. Load raw dataset.
    df = load_dataset(
        file_path=file_path,
        file_extension=file_extension,
    )

    # 2. Profile dataset.
    profile = profile_dataset(
        file_path=file_path,
        file_extension=file_extension,
    )

    # 3. Detect semantic field types.
    fields = detect_schema(profile)

    schema = DatasetSchema(
        dataset_id=dataset_id,
        fields=fields,
    )

    # 4. Build standardization plan.
    plan = build_standardization_plan(
        dataset_id=dataset_id,
        schema=schema,
    )

    # 5. Apply transformations to a copy.
    standardized_df = apply_standardization(
        df=df,
        plan=plan,
    )

    return standardized_df, schema, plan