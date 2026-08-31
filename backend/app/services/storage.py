from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile

from app.config import (
    METADATA_DIR,
    PROCESSED_DATA_DIR,
    PROFILES_DIR,
    RAW_DATA_DIR,
    RESULTS_DIR,
)


def initialize_storage() -> None:
    """Create all persistent storage directories if they do not exist."""
    for directory in (
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        METADATA_DIR,
        PROFILES_DIR,
        RESULTS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def create_dataset_id() -> str:
    """Create a unique identifier for an uploaded dataset."""
    return uuid4().hex


def sanitize_filename(filename: str) -> str:
    """
    Return a safe filename containing only the final path component.

    This prevents path traversal such as ../../some_file.csv.
    """
    return Path(filename).name


def get_raw_dataset_path(dataset_id: str, filename: str) -> Path:
    """Return the storage path for a raw uploaded dataset."""
    safe_filename = sanitize_filename(filename)
    return RAW_DATA_DIR / f"{dataset_id}__{safe_filename}"


async def save_upload(
    upload_file: UploadFile,
    destination: Path,
) -> int:
    """
    Save an uploaded file in chunks.

    Returns:
        Number of bytes written.
    """
    bytes_written = 0

    with destination.open("wb") as output_file:
        while chunk := await upload_file.read(1024 * 1024):
            output_file.write(chunk)
            bytes_written += len(chunk)

    await upload_file.close()

    return bytes_written
