from fastapi import APIRouter, HTTPException, status

from app.config import RAW_DATA_DIR
from app.models.schemas import DatasetStandardizationPlan
from app.services.dataset_preparation import prepare_dataset
from app.services.metadata import load_metadata


router = APIRouter(
    prefix="/api/datasets",
    tags=["standardization"],
)


@router.post(
    "/{dataset_id}/standardize",
)
def standardize_uploaded_dataset(dataset_id: str):
    """
    Prepare and standardize an uploaded dataset.

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
        standardized_df, schema, plan = prepare_dataset(
            file_path=file_path,
            file_extension=metadata.file_extension,
            dataset_id=dataset_id,
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