from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from app.config import RAW_DATA_DIR
from app.models.schemas import (
    DedupeLabelRequest,
    DedupeLabelResponse,
    DedupePair,
    DedupePrepareRequest,
    DedupePrepareResponse,
    DedupeRecord,
    DedupeTrainRequest,
    DedupeTrainResponse,
    DedupeLearnedPatternsResponse,
    LearnedDedupePatternResponse,
    SplinkBlockingRuleResponse,
    SplinkBlockingStrategyResponse,
    DedupeUncertainPairsResponse,
    CandidateGenerationResponse,
    CandidatePairResponse,
    SplinkMatchResponse,
    SplinkMatchingResponse,
    HumanReviewDecisionRequest,
    HumanReviewDecisionResponse,
    HumanReviewItem,
    HumanReviewResponse,
)
from app.models.schemas import MatchDecisionItem, MatchDecisionResponse
from app.services.blocking import (
    generate_splink_blocking_rules,
)
from app.services.dataset_preparation import prepare_dataset
from app.services.dedupe_active_learning import (
    create_active_learning_session,
    get_learned_dedupe_patterns,
    get_uncertain_pairs,
    mark_labeled_pairs,
    train_dedupe_model,
)
from app.services.dedupe_engine import (
    build_dedupe_fields,
    create_dedupe_linker,
    dataframe_to_dedupe_records,
)
from app.services.dedupe_session import dedupe_sessions
from app.services.metadata import load_metadata
from app.services.candidate_generation import generate_candidate_pairs
from app.services.splink_matching import run_splink_matching
from app.services.match_decision import classify_match_results
from app.services.human_review import human_review_store


router = APIRouter(
    prefix="/api/datasets",
    tags=["dedupe"],
)


def _load_dataset_metadata(dataset_id: str):
    """Load dataset metadata and verify the raw file exists."""

    try:
        metadata = load_metadata(dataset_id)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found.",
        ) from exc

    file_path = RAW_DATA_DIR / metadata.stored_filename

    if not file_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Stored dataset file not found.",
        )

    return metadata, file_path


def _get_session(dataset_id: str):
    """Get an active Dedupe session."""

    try:
        return dedupe_sessions.get(dataset_id)
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                "No active Dedupe session exists for this dataset. "
                "Prepare the dataset first."
            ),
        ) from exc


def _pair_to_api_pair(
    pair: tuple[dict[str, str], dict[str, str]],
) -> DedupePair:
    """Convert a Dedupe pair into the API representation."""

    record_a, record_b = pair

    return DedupePair(
        record_a=DedupeRecord(
            record_id=_extract_record_id(record_a),
            data=record_a,
        ),
        record_b=DedupeRecord(
            record_id=_extract_record_id(record_b),
            data=record_b,
        ),
    )


def _extract_record_id(
    record: dict[str, str],
) -> int:
    """Extract the original integer record identifier."""

    value = record.get("_dedupe_record_id")

    if value is not None:
        try:
            return int(value)
        except (TypeError, ValueError):
            pass

    value = record.get("id")

    if value is not None:
        try:
            return int(value)
        except (TypeError, ValueError):
            pass

    return -1


def _api_pair_to_tuple(
    pair: DedupePair,
) -> tuple[dict[str, str], dict[str, str]]:
    """Convert an API pair back into Dedupe's representation."""

    return pair.record_a.data, pair.record_b.data


@router.post(
    "/{dataset_id}/dedupe/prepare",
    response_model=DedupePrepareResponse,
)
def prepare_dedupe(
    dataset_id: str,
    request: DedupePrepareRequest,
):
    """
    Prepare a dataset for Dedupe active learning.
    """

    metadata, file_path = _load_dataset_metadata(dataset_id)

    try:
        standardized_df, schema, _plan = prepare_dataset(
            file_path=file_path,
            file_extension=metadata.file_extension,
            dataset_id=dataset_id,
        )

        dedupe_fields = build_dedupe_fields(schema)

        if not dedupe_fields:
            raise ValueError(
                "No fields are available for Dedupe matching."
            )

        records = dataframe_to_dedupe_records(
            standardized_df,
            dedupe_fields,
        )

        if len(records) < 2:
            raise ValueError(
                "At least two records are required for deduplication."
            )

        linker = create_dedupe_linker(dedupe_fields)

        session = create_active_learning_session(
            dataset_id=dataset_id,
            linker=linker,
            records=records,
            schema=schema,
            sample_size=request.sample_size,
            blocked_proportion=request.blocked_proportion,
        )

        dedupe_sessions.save(
            dataset_id=dataset_id,
            session=session,
        )

        return DedupePrepareResponse(
            dataset_id=dataset_id,
            row_count=len(standardized_df),
            column_count=len(standardized_df.columns),
            fields=[
                field.column_name
                for field in schema.fields
            ],
            sample_size=request.sample_size,
            blocked_proportion=request.blocked_proportion,
            status="active_learning_ready",
        )

    except (ValueError, OSError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unable to prepare dataset for Dedupe: {exc}"
            ),
        ) from exc


@router.get(
    "/{dataset_id}/dedupe/uncertain-pairs",
    response_model=DedupeUncertainPairsResponse,
)
def get_dedupe_uncertain_pairs(
    dataset_id: str,
    limit: int = 10,
):
    """Return the most informative pairs selected by Dedupe."""

    if limit <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="limit must be greater than zero.",
        )

    session = _get_session(dataset_id)

    try:
        pairs = get_uncertain_pairs(
            session=session,
            limit=limit,
        )

        return DedupeUncertainPairsResponse(
            dataset_id=dataset_id,
            pairs=[
                _pair_to_api_pair(pair)
                for pair in pairs
            ],
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc


@router.post(
    "/{dataset_id}/dedupe/label",
    response_model=DedupeLabelResponse,
)
def label_dedupe_pairs(
    dataset_id: str,
    request: DedupeLabelRequest,
):
    """Add Match/Distinct labels to the active-learning session."""

    session = _get_session(dataset_id)

    matches = [
        _api_pair_to_tuple(pair)
        for pair in request.matches
    ]

    distinct = [
        _api_pair_to_tuple(pair)
        for pair in request.distinct
    ]

    if not matches and not distinct:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "At least one Match or Distinct pair is required."
            ),
        )

    try:
        mark_labeled_pairs(
            session=session,
            matches=matches,
            distinct=distinct,
        )

        return DedupeLabelResponse(
            dataset_id=dataset_id,
            matches_added=len(matches),
            distinct_added=len(distinct),
            status="labels_added",
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc


@router.post(
    "/{dataset_id}/dedupe/train",
    response_model=DedupeTrainResponse,
)
def train_dedupe(
    dataset_id: str,
    request: DedupeTrainRequest,
):
    """
    Train Dedupe and expose its learned patterns.

    Important:
        The learned patterns are NOT treated as final blocking rules.
        They are passed to the application's blocking strategy
        generator in the next stage.
    """

    session = _get_session(dataset_id)

    try:
        train_dedupe_model(
            session=session,
            recall=request.recall,
            index_predicates=request.index_predicates,
        )

        patterns = get_learned_dedupe_patterns(session)

        return DedupeTrainResponse(
            dataset_id=dataset_id,
            status="trained",
            learned_patterns=[
                LearnedDedupePatternResponse(
                    fields=list(pattern.fields),
                    pattern_type=pattern.strategy,
                    description=(
                        f"Dedupe learned a {pattern.strategy} "
                        f"matching pattern on: {', '.join(pattern.fields)}"
                    ),
                )
                for pattern in patterns
            ],
        )

    except (ValueError, AssertionError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Unable to train Dedupe model. "
                "Add more Match/Distinct labels and retry. "
                f"Details: {exc}"
            ),
        ) from exc


@router.get(
    "/{dataset_id}/dedupe/learned-patterns",
    response_model=DedupeLearnedPatternsResponse,
)
def get_dedupe_learned_patterns(
    dataset_id: str,
):
    """
    Return patterns learned by Dedupe.

    These are intermediate learning results, not Splink rules.
    """

    session = _get_session(dataset_id)

    try:
        patterns = get_learned_dedupe_patterns(session)

        if not patterns:
            raise ValueError(
                "No learned Dedupe patterns are available. "
                "Train the Dedupe model first."
            )

        return DedupeLearnedPatternsResponse(
            dataset_id=dataset_id,
            patterns=[
                LearnedDedupePatternResponse(
                    fields=list(pattern.fields),
                    pattern_type=pattern.strategy,
                    description=(
                        f"Dedupe learned a {pattern.strategy} "
                        f"matching pattern on: {', '.join(pattern.fields)}"
                    ),
                )
                for pattern in patterns
            ],
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc


@router.post(
    "/{dataset_id}/dedupe/blocking-strategy/generate",
    response_model=SplinkBlockingStrategyResponse,
)
def generate_blocking_strategy(
    dataset_id: str,
):
    """
    Generate the application's final Splink blocking strategy
    from patterns learned by Dedupe.
    """

    session = _get_session(dataset_id)

    try:
        patterns = get_learned_dedupe_patterns(session)

        if not patterns:
            raise ValueError(
                "No learned Dedupe patterns are available. "
                "Train the Dedupe model first."
            )

        rules = generate_splink_blocking_rules(
            patterns,
        )

        if not rules:
            raise ValueError(
                "Unable to generate Splink blocking rules from "
                "the learned Dedupe patterns."
            )

        return SplinkBlockingStrategyResponse(
            dataset_id=dataset_id,
            source="dedupe_pattern_generator",
            rules=[
                SplinkBlockingRuleResponse(
                    fields=list(rule.fields),
                    strategy=rule.strategy,
                    sql_condition=rule.sql_condition,
                )
                for rule in rules
            ],
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

@router.post(
    "/{dataset_id}/dedupe/candidates/generate",
    response_model=CandidateGenerationResponse,
)
def generate_candidates(dataset_id: str) -> CandidateGenerationResponse:
    session = _get_session(dataset_id)

    learned_patterns = get_learned_dedupe_patterns(session)

    try:
        blocking_rules = generate_splink_blocking_rules(learned_patterns)

        candidate_pairs = generate_candidate_pairs(
            records=session.records,
            blocking_rules=blocking_rules,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    candidates = [
        CandidatePairResponse(
            record_a_id=record_a_id,
            record_b_id=record_b_id,
        )
        for record_a_id, record_b_id in candidate_pairs
    ]

    return CandidateGenerationResponse(
        dataset_id=dataset_id,
        candidate_pair_count=len(candidates),
        blocking_rule_count=len(blocking_rules),
        candidates=candidates,
        status="generated",
    )

@router.post(
    "/{dataset_id}/dedupe/match",
    response_model=SplinkMatchingResponse,
)
def run_matching(dataset_id: str) -> SplinkMatchingResponse:
    session = _get_session(dataset_id)

    learned_patterns = get_learned_dedupe_patterns(session)

    try:
        blocking_rules = generate_splink_blocking_rules(learned_patterns)

        prediction_dataframe = run_splink_matching(
            records=session.records,
            blocking_rules=blocking_rules,
            schema=session.schema,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    matches = _prediction_dataframe_to_matches(
        prediction_dataframe
    )

    return SplinkMatchingResponse(
        dataset_id=dataset_id,
        match_pair_count=len(matches),
        blocking_rule_count=len(blocking_rules),
        matches=matches,
        status="matched",
    )


def _prediction_dataframe_to_matches(
    prediction_dataframe,
) -> list[SplinkMatchResponse]:
    required_columns = {
        "unique_id_l",
        "unique_id_r",
        "match_probability",
    }

    missing_columns = required_columns - set(prediction_dataframe.columns)

    if missing_columns:
        raise ValueError(
            "Splink prediction output is missing expected columns: "
            + ", ".join(sorted(missing_columns))
        )

    matches: list[SplinkMatchResponse] = []

    for _, row in prediction_dataframe.iterrows():
        record_a_id = int(row["unique_id_l"])
        record_b_id = int(row["unique_id_r"])
        match_probability = float(row["match_probability"])

        if record_a_id == record_b_id:
            continue

        matches.append(
            SplinkMatchResponse(
                record_a_id=min(record_a_id, record_b_id),
                record_b_id=max(record_a_id, record_b_id),
                match_probability=match_probability,
            )
        )

    matches.sort(
        key=lambda item: (
            -item.match_probability,
            item.record_a_id,
            item.record_b_id,
        )
    )

    return matches

@router.post(
    "/{dataset_id}/dedupe/decisions",
    response_model=MatchDecisionResponse,
)
def get_match_decisions(dataset_id: str) -> MatchDecisionResponse:
    session = _get_session(dataset_id)

    learned_patterns = get_learned_dedupe_patterns(session)

    try:
        blocking_rules = generate_splink_blocking_rules(learned_patterns)

        prediction_dataframe = run_splink_matching(
            records=session.records,
            blocking_rules=blocking_rules,
            schema=session.schema,
        )

        required_columns = {
            "unique_id_l",
            "unique_id_r",
            "match_probability",
        }

        missing_columns = required_columns - set(
            prediction_dataframe.columns
        )

        if missing_columns:
            raise ValueError(
                "Splink prediction output is missing expected columns: "
                + ", ".join(sorted(missing_columns))
            )

        raw_results = [
            (
                int(row["unique_id_l"]),
                int(row["unique_id_r"]),
                float(row["match_probability"]),
            )
            for _, row in prediction_dataframe.iterrows()
        ]

        classified_results = classify_match_results(raw_results)

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    results = [
        MatchDecisionItem(
            record_a_id=item.record_a_id,
            record_b_id=item.record_b_id,
            match_probability=item.match_probability,
            decision=item.decision.value,
        )
        for item in classified_results
    ]

    match_count = sum(
        item.decision == "match"
        for item in results
    )

    possible_match_count = sum(
        item.decision == "possible_match"
        for item in results
    )

    non_match_count = sum(
        item.decision == "non_match"
        for item in results
    )

    return MatchDecisionResponse(
        dataset_id=dataset_id,
        match_count=match_count,
        possible_match_count=possible_match_count,
        non_match_count=non_match_count,
        results=results,
        status="classified",
    )

@router.get(
    "/{dataset_id}/dedupe/human-review",
    response_model=HumanReviewResponse,
)
def get_human_review_queue(
    dataset_id: str,
) -> HumanReviewResponse:
    session = _get_session(dataset_id)

    learned_patterns = get_learned_dedupe_patterns(session)

    try:
        blocking_rules = generate_splink_blocking_rules(
            learned_patterns
        )

        prediction_dataframe = run_splink_matching(
            records=session.records,
            blocking_rules=blocking_rules,
            schema=session.schema,
        )

        required_columns = {
            "unique_id_l",
            "unique_id_r",
            "match_probability",
        }

        missing_columns = required_columns - set(
            prediction_dataframe.columns
        )

        if missing_columns:
            raise ValueError(
                "Splink prediction output is missing expected columns: "
                + ", ".join(sorted(missing_columns))
            )

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    reviewed_decisions = {
        (
            decision.record_a_id,
            decision.record_b_id,
        )
        for decision in human_review_store.get_all(dataset_id)
    }

    items: list[HumanReviewItem] = []

    for _, row in prediction_dataframe.iterrows():
        record_a_id = int(row["unique_id_l"])
        record_b_id = int(row["unique_id_r"])
        probability = float(row["match_probability"])

        if record_a_id == record_b_id:
            continue

        normalized_pair = (
            min(record_a_id, record_b_id),
            max(record_a_id, record_b_id),
        )

        if normalized_pair in reviewed_decisions:
            continue

        if not 0.50 <= probability < 0.90:
            continue

        record_a = session.records.get(record_a_id)
        record_b = session.records.get(record_b_id)

        if record_a is None or record_b is None:
            continue

        items.append(
            HumanReviewItem(
                record_a=DedupeRecord(
                    record_id=record_a_id,
                    data=record_a,
                ),
                record_b=DedupeRecord(
                    record_id=record_b_id,
                    data=record_b,
                ),
                match_probability=probability,
            )
        )

    items.sort(
        key=lambda item: (
            -item.match_probability,
            item.record_a.record_id,
            item.record_b.record_id,
        )
    )

    return HumanReviewResponse(
        dataset_id=dataset_id,
        review_count=len(items),
        items=items,
        status="ready",
    )


@router.post(
    "/{dataset_id}/dedupe/human-review",
    response_model=HumanReviewDecisionResponse,
)
def submit_human_review(
    dataset_id: str,
    request: HumanReviewDecisionRequest,
) -> HumanReviewDecisionResponse:
    session = _get_session(dataset_id)

    if request.record_a_id not in session.records:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Record {request.record_a_id} not found.",
        )

    if request.record_b_id not in session.records:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Record {request.record_b_id} not found.",
        )

    try:
        decision = human_review_store.save(
            dataset_id=dataset_id,
            record_a_id=request.record_a_id,
            record_b_id=request.record_b_id,
            decision=request.decision,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    return HumanReviewDecisionResponse(
        dataset_id=dataset_id,
        record_a_id=decision.record_a_id,
        record_b_id=decision.record_b_id,
        decision=decision.decision,
        status="saved",
    )