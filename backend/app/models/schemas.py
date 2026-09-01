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


# ---------------------------------------------------------------------------
# DEDUPE API SCHEMAS
# ---------------------------------------------------------------------------


class DedupePrepareRequest(BaseModel):
    """Configuration used when starting a Dedupe active-learning session."""

    model_config = ConfigDict(extra="forbid")

    sample_size: int = Field(default=1500, gt=0)
    blocked_proportion: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
    )


class DedupePrepareResponse(BaseModel):
    """Summary returned after initializing Dedupe active learning."""

    model_config = ConfigDict(extra="forbid")

    dataset_id: str
    row_count: int
    column_count: int
    fields: list[str]
    sample_size: int
    blocked_proportion: float
    status: str


class DedupeRecord(BaseModel):
    """A single record presented to the active-learning UI."""

    model_config = ConfigDict(extra="forbid")

    record_id: int
    data: dict[str, str]


class DedupePair(BaseModel):
    """A pair of records that the user must label."""

    model_config = ConfigDict(extra="forbid")

    record_a: DedupeRecord
    record_b: DedupeRecord


class DedupeUncertainPairsResponse(BaseModel):
    """Uncertain pairs selected by Dedupe."""

    model_config = ConfigDict(extra="forbid")

    dataset_id: str
    pairs: list[DedupePair]


class DedupeLabelRequest(BaseModel):
    """User labels for Dedupe training pairs."""

    model_config = ConfigDict(extra="forbid")

    matches: list[DedupePair] = Field(default_factory=list)
    distinct: list[DedupePair] = Field(default_factory=list)


class DedupeLabelResponse(BaseModel):
    """Result after adding labels to the active-learning session."""

    model_config = ConfigDict(extra="forbid")

    dataset_id: str
    matches_added: int
    distinct_added: int
    status: str


class DedupeTrainRequest(BaseModel):
    """Configuration used to train the Dedupe model."""

    model_config = ConfigDict(extra="forbid")

    recall: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
    )
    index_predicates: bool = True

class LearnedDedupePatternResponse(BaseModel):
    """
    A pattern learned by Dedupe during active-learning training.

    This represents what Dedupe learned from the labeled pairs.
    It is NOT yet a Splink blocking rule.
    """

    model_config = ConfigDict(extra="forbid")

    fields: list[str]
    pattern_type: str
    description: str


class DedupeTrainResponse(BaseModel):
    """
    Result after training the Dedupe model.

    Dedupe learns matching patterns here.
    These patterns are later converted by our application
    into Splink blocking rules.
    """

    model_config = ConfigDict(extra="forbid")

    dataset_id: str
    status: str
    learned_patterns: list[LearnedDedupePatternResponse]

class DedupeLearnedPatternsResponse(BaseModel):
    """
    Explicit API response for the patterns learned by Dedupe.

    This endpoint exposes Dedupe's learned knowledge before
    any Splink blocking rules are generated.
    """

    model_config = ConfigDict(extra="forbid")

    dataset_id: str
    patterns: list[LearnedDedupePatternResponse]

class SplinkBlockingRuleResponse(BaseModel):
    """A blocking rule generated for downstream Splink."""

    model_config = ConfigDict(extra="forbid")

    fields: list[str]
    strategy: str
    sql_condition: str

class SplinkBlockingStrategyResponse(BaseModel):
    """Final blocking strategy generated for Splink."""

    model_config = ConfigDict(extra="forbid")

    dataset_id: str
    source: str
    rules: list[SplinkBlockingRuleResponse]