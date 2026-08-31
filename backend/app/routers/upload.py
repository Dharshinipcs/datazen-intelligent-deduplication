from datetime import datetime, timezone

from fastapi import APIRouter, File, UploadFile

from app.models.schemas import DatasetMetadata
from app.services.metadata import save_metadata
from app.services.storage import (
    create_dataset_id,
    get_raw_dataset_path,
    initialize_storage,
)
from app.services.validation import (
    save_upload_with_limit,
    validate_upload_filename,
)


router = APIRouter(
    prefix="/api/datasets",
    tags=["datasets"],
)


@router.post("/upload", response_model=DatasetMetadata)
async def upload_dataset(file: UploadFile = File(...)):
    """Upload and persist a raw dataset."""
    initialize_storage()

    extension = validate_upload_filename(file.filename)

    dataset_id = create_dataset_id()
    destination = get_raw_dataset_path(
        dataset_id=dataset_id,
        filename=file.filename,
    )

    file_size = await save_upload_with_limit(
        upload_file=file,
        destination=destination,
    )

    metadata = DatasetMetadata(
        dataset_id=dataset_id,
        original_filename=file.filename,
        stored_filename=destination.name,
        file_extension=extension,
        file_size_bytes=file_size,
        uploaded_at=datetime.now(timezone.utc),
    )

    save_metadata(metadata)

    return metadata
