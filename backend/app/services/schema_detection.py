import re

from app.models.schemas import ColumnProfile, SemanticField


EMAIL_PATTERN = re.compile(
    r"^[^@\s]+@[^@\s]+\.[^@\s]+$"
)

PHONE_PATTERN = re.compile(
    r"^\+?[\d\s().-]{7,20}$"
)

POSTAL_CODE_PATTERN = re.compile(
    r"^\d{4,10}$"
)


def normalize_column_name(name: str) -> str:
    """Normalize a column name for semantic matching."""
    value = name.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def name_tokens(name: str) -> set[str]:
    """Return normalized tokens from a column name."""
    normalized = normalize_column_name(name)
    return set(normalized.split("_"))


def looks_like_email(values: list[str]) -> bool:
    """Check whether sample values strongly resemble email addresses."""
    if not values:
        return False

    matches = sum(
        bool(EMAIL_PATTERN.match(value.strip()))
        for value in values
    )

    return matches / len(values) >= 0.8


def looks_like_phone(values: list[str]) -> bool:
    """Check whether sample values strongly resemble phone numbers."""
    if not values:
        return False

    matches = sum(
        bool(PHONE_PATTERN.match(value.strip()))
        for value in values
    )

    return matches / len(values) >= 0.8


def looks_like_postal_code(values: list[str]) -> bool:
    """Check whether sample values resemble numeric postal codes."""
    if not values:
        return False

    matches = sum(
        bool(POSTAL_CODE_PATTERN.match(value.strip()))
        for value in values
    )

    return matches / len(values) >= 0.8


def detect_semantic_field(column: ColumnProfile) -> SemanticField:
    """
    Detect the semantic meaning of a single column.

    Detection uses explainable evidence from:
    - column name
    - sample values
    - data profile statistics

    Returns:
        SemanticField containing semantic type, confidence,
        and supporting evidence.
    """
    tokens = name_tokens(column.name)
    values = column.sample_values

    # ---------------------------------------------------------
    # Strong semantic signals from column names
    # ---------------------------------------------------------

    if (
        tokens & {"email", "email_address", "mail"}
        or "email" in tokens
    ):
        return SemanticField(
            column_name=column.name,
            semantic_type="email",
            confidence=0.99,
            evidence=["column_name"],
        )

    if tokens & {
        "phone",
        "mobile",
        "telephone",
        "contact_number",
        "phone_number",
    }:
        return SemanticField(
            column_name=column.name,
            semantic_type="phone",
            confidence=0.98,
            evidence=["column_name"],
        )

    if tokens & {
        "postal",
        "postcode",
        "zip",
        "zipcode",
        "pincode",
    }:
        return SemanticField(
            column_name=column.name,
            semantic_type="postal_code",
            confidence=0.98,
            evidence=["column_name"],
        )

    if tokens & {
        "city",
        "town",
    }:
        return SemanticField(
            column_name=column.name,
            semantic_type="city",
            confidence=0.98,
            evidence=["column_name"],
        )

    if tokens & {
        "state",
        "province",
        "region",
    }:
        return SemanticField(
            column_name=column.name,
            semantic_type="state",
            confidence=0.97,
            evidence=["column_name"],
        )

    if tokens & {
        "country",
        "nation",
    }:
        return SemanticField(
            column_name=column.name,
            semantic_type="country",
            confidence=0.98,
            evidence=["column_name"],
        )

    if tokens & {
        "address",
        "street",
        "addr",
    }:
        return SemanticField(
            column_name=column.name,
            semantic_type="address",
            confidence=0.97,
            evidence=["column_name"],
        )

    if tokens & {
        "customer_id",
        "supplier_id",
        "vendor_id",
        "employee_id",
        "account_id",
        "record_id",
        "entity_id",
        "identifier",
        "id",
    }:
        return SemanticField(
            column_name=column.name,
            semantic_type="identifier",
            confidence=0.97,
            evidence=["column_name"],
        )

    if tokens & {
        "first_name",
        "firstname",
        "given_name",
    }:
        return SemanticField(
            column_name=column.name,
            semantic_type="person_name",
            confidence=0.96,
            evidence=["column_name"],
        )

    if tokens & {
        "last_name",
        "lastname",
        "surname",
        "family_name",
    }:
        return SemanticField(
            column_name=column.name,
            semantic_type="person_name",
            confidence=0.96,
            evidence=["column_name"],
        )

    if tokens & {
        "name",
        "full_name",
        "fullname",
        "customer_name",
        "supplier_name",
        "vendor_name",
        "employee_name",
    }:
        return SemanticField(
            column_name=column.name,
            semantic_type="person_name",
            confidence=0.90,
            evidence=["column_name"],
        )

    if tokens & {
        "company",
        "company_name",
        "organization",
        "organization_name",
        "org_name",
        "business_name",
    }:
        return SemanticField(
            column_name=column.name,
            semantic_type="organization_name",
            confidence=0.96,
            evidence=["column_name"],
        )

    # ---------------------------------------------------------
    # Value-based semantic signals
    # ---------------------------------------------------------

    if looks_like_email(values):
        return SemanticField(
            column_name=column.name,
            semantic_type="email",
            confidence=0.95,
            evidence=["email_pattern"],
        )

    if looks_like_phone(values):
        return SemanticField(
            column_name=column.name,
            semantic_type="phone",
            confidence=0.90,
            evidence=["phone_pattern"],
        )

    if looks_like_postal_code(values):
        return SemanticField(
            column_name=column.name,
            semantic_type="postal_code",
            confidence=0.75,
            evidence=["postal_code_pattern"],
        )

    # ---------------------------------------------------------
    # Generic type fallback
    # ---------------------------------------------------------

    if column.dtype.startswith(("int", "float")):
        semantic_type = "numeric"
    elif column.dtype in {"datetime", "datetime64[ns]"}:
        semantic_type = "date"
    else:
        semantic_type = "text"

    return SemanticField(
        column_name=column.name,
        semantic_type=semantic_type,
        confidence=0.50,
        evidence=["data_type"],
    )


def detect_schema(profile: dict) -> list[SemanticField]:
    """Detect semantic types for all columns in a dataset profile."""
    columns = [
        ColumnProfile.model_validate(column)
        for column in profile["columns"]
    ]

    return [
        detect_semantic_field(column)
        for column in columns
    ]
