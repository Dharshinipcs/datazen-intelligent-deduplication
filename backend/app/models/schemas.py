from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class DatasetMetadata(BaseModel):
    """Metadata describing an uploaded dataset."""

    model_config = ConfigDict(extra="forbid")

    dataset_id: str
    original_filename: str
    stored_filename: str
    file_extension: str
    file_size_bytes: int
    uploaded_at: datetime


class ColumnProfile(BaseModel):
    """Profile information for a single dataset column."""

    model_config = ConfigDict(extra="forbid")

    name: str
    dtype: str
    row_count: int
    null_count: int
    null_percentage: float
    unique_count: int
    unique_percentage: float
    sample_values: list[str]


class DatasetProfile(BaseModel):
    """Domain-agnostic profile of an uploaded dataset."""

    model_config = ConfigDict(extra="forbid")

    row_count: int
    column_count: int
    columns: list[ColumnProfile]


class SemanticField(BaseModel):
    """Semantic interpretation of a dataset column."""

    model_config = ConfigDict(extra="forbid")

    column_name: str
    semantic_type: str
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[str]


class DatasetSchema(BaseModel):
    """Detected semantic schema of an uploaded dataset."""

    model_config = ConfigDict(extra="forbid")

    dataset_id: str
    fields: list[SemanticField]

class StandardizedField(BaseModel):
    """Standardization result for a single dataset column."""

    model_config = ConfigDict(extra="forbid")

    column_name: str
    semantic_type: str
    transformations: list[str]


class DatasetStandardizationPlan(BaseModel):
    """Standardization plan derived from semantic field detection."""

    model_config = ConfigDict(extra="forbid")

    dataset_id: str
    fields: list[StandardizedField]