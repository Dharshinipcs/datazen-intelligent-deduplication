import json
from pathlib import Path

from app.config import METADATA_DIR
from app.models.schemas import DatasetMetadata


def get_metadata_path(dataset_id: str) -> Path:
    """Return the metadata JSON path for a dataset."""
    return METADATA_DIR / f"{dataset_id}.json"


def save_metadata(metadata: DatasetMetadata) -> Path:
    """Persist dataset metadata as JSON."""
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    metadata_path = get_metadata_path(metadata.dataset_id)

    with metadata_path.open("w", encoding="utf-8") as metadata_file:
        json.dump(
            metadata.model_dump(mode="json"),
            metadata_file,
            indent=2,
        )

    return metadata_path


def load_metadata(dataset_id: str) -> DatasetMetadata:
    """Load dataset metadata from JSON."""
    metadata_path = get_metadata_path(dataset_id)

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata not found for dataset: {dataset_id}"
        )

    with metadata_path.open("r", encoding="utf-8") as metadata_file:
        data = json.load(metadata_file)

    return DatasetMetadata.model_validate(data)
