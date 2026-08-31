from fastapi import APIRouter, HTTPException, status

from app.config import RAW_DATA_DIR
from app.models.schemas import DatasetProfile, DatasetSchema
from app.services.metadata import load_metadata
from app.services.profiling import profile_dataset
from app.services.schema_detection import detect_schema


router = APIRouter(
    prefix="/api/datasets",
    tags=["profiling"],
)


@router.post(
    "/{dataset_id}/profile",
    response_model=DatasetProfile,
)
def profile_uploaded_dataset(dataset_id: str) -> DatasetProfile:
    """Profile a previously uploaded dataset."""
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
        profile = profile_dataset(
            file_path=file_path,
            file_extension=metadata.file_extension,
        )

        return DatasetProfile.model_validate(profile)

    except (ValueError, OSError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unable to profile dataset: {exc}",
        ) from exc


@router.post(
    "/{dataset_id}/schema",
    response_model=DatasetSchema,
)
def detect_uploaded_schema(dataset_id: str) -> DatasetSchema:
    """Detect semantic field types for a previously uploaded dataset."""
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
        profile = profile_dataset(
            file_path=file_path,
            file_extension=metadata.file_extension,
        )

        fields = detect_schema(profile)

        return DatasetSchema(
            dataset_id=dataset_id,
            fields=fields,
        )

    except (ValueError, OSError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unable to detect dataset schema: {exc}",
        ) from exc