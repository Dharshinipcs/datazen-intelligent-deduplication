from fastapi import APIRouter, HTTPException, status

from app.config import RAW_DATA_DIR
from app.models.schemas import DatasetStandardizationPlan
from app.services.metadata import load_metadata
from app.services.profiling import load_dataset
from app.services.schema_detection import detect_schema
from app.services.profiling import profile_dataset
from app.services.standardization import (
    apply_standardization,
    build_standardization_plan,
)


router = APIRouter(
    prefix="/api/datasets",
    tags=["standardization"],
)


@router.post(
    "/{dataset_id}/standardize",
)
def standardize_uploaded_dataset(dataset_id: str):
    """
    Build and apply a semantic-driven standardization plan
    to a copy of the uploaded dataset.

    The original raw dataset is never modified.
    """
    try:
        metadata = load_metadata(dataset_id)
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found.",
        )

    file_path = RAW_DATA_DIR / metadata.stored_filename

    if not file_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Stored dataset file not found.",
        )

    try:
        # Load the original dataset.
        df = load_dataset(
            file_path=file_path,
            file_extension=metadata.file_extension,
        )

        # Detect semantic meaning from the dataset profile.
        profile = profile_dataset(
            file_path=file_path,
            file_extension=metadata.file_extension,
        )

        fields = detect_schema(profile)

        from app.models.schemas import DatasetSchema

        schema = DatasetSchema(
            dataset_id=dataset_id,
            fields=fields,
        )

        # Build the standardization plan.
        plan = build_standardization_plan(
            dataset_id=dataset_id,
            schema=schema,
        )

        # Apply transformations to a copy.
        standardized_df = apply_standardization(
            df=df,
            plan=plan,
        )

        return {
            "dataset_id": dataset_id,
            "plan": plan,
            "row_count": len(standardized_df),
            "column_count": len(standardized_df.columns),
            "columns": list(standardized_df.columns),
            "preview": standardized_df.head(5).fillna("").to_dict(
                orient="records"
            ),
        }

    except (ValueError, OSError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unable to standardize dataset: {exc}",
        ) from exc