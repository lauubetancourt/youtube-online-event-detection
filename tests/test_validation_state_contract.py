from __future__ import annotations

import pytest

from youtube_pipeline.validation_state import (
    COMPLETE,
    CONTEXT_LIMIT_EXCEEDED,
    EXECUTED,
    FAILED,
    INVALID_CITATION,
    INVALID_QUERY,
    MISSING_PAYLOAD,
    NOT_ELIGIBLE,
    NO_EVIDENCE,
    PARTIAL,
    PENDING_BATCHES,
    VIDEO_FAILURES,
    ValidationState,
    normalize_g1_state,
    normalize_g2_state,
)
from youtube_pipeline.validation_status import (
    CONFIRMED,
    INSUFFICIENT_EVIDENCE,
    NOT_CONFIRMED,
)


def _complete_g2(status: str) -> dict:
    return {
        "summary_completeness": "complete",
        "event_batch_status": "complete",
        "next_batch_required": False,
        "pending_video_ids": [],
        "failed_video_ids": [],
        "overall_validation_status": status,
    }


@pytest.mark.parametrize(
    "status", [CONFIRMED, INSUFFICIENT_EVIDENCE, NOT_CONFIRMED]
)
def test_g1_complete_semantic_decisions(status: str) -> None:
    state = normalize_g1_state({"validation_status": status})
    assert state == ValidationState(EXECUTED, COMPLETE, status, None)


def test_g1_legacy_context_limit_has_no_semantic_decision() -> None:
    state = normalize_g1_state(
        {
            "validation_status": "insufficient_evidence",
            "generation_error": True,
            "limitations": ["context_limit_exceeded"],
        }
    )
    assert state == ValidationState(
        FAILED, PARTIAL, None, CONTEXT_LIMIT_EXCEEDED
    )


@pytest.mark.parametrize("status", [CONFIRMED, INSUFFICIENT_EVIDENCE])
def test_g2_complete_semantic_decisions(status: str) -> None:
    state = normalize_g2_state(_complete_g2(status))
    assert state == ValidationState(EXECUTED, COMPLETE, status, None)


def test_g2_pending_batch_suppresses_legacy_overall_status() -> None:
    summary = {
        **_complete_g2(CONFIRMED),
        "summary_completeness": "partial",
        "event_batch_status": "partial_pending_batch",
        "next_batch_required": True,
        "pending_video_ids": ["video-2"],
    }
    assert normalize_g2_state(summary) == ValidationState(
        EXECUTED, PARTIAL, None, PENDING_BATCHES
    )


def test_g2_failed_video_suppresses_legacy_overall_status() -> None:
    summary = {
        **_complete_g2(CONFIRMED),
        "summary_completeness": "partial",
        "event_batch_status": "partial_errors",
        "failed_video_ids": ["video-1"],
    }
    assert normalize_g2_state(summary) == ValidationState(
        EXECUTED, PARTIAL, None, VIDEO_FAILURES
    )


@pytest.mark.parametrize(
    ("reason_text", "expected"),
    [
        ("missing_query_payload", MISSING_PAYLOAD),
        ("invalid_video_query", INVALID_QUERY),
        ("invalid_citation", INVALID_CITATION),
    ],
)
def test_g2_specific_video_failures_are_stable_reason_codes(
    reason_text: str, expected: str
) -> None:
    summary = {
        **_complete_g2(CONFIRMED),
        "summary_completeness": "partial",
        "event_batch_status": "partial_errors",
        "failed_video_ids": ["video-1"],
        "failure_reason_codes": [reason_text],
    }
    assert normalize_g2_state(summary) == ValidationState(
        EXECUTED, PARTIAL, None, expected
    )


def test_legacy_incomplete_query_payload_overrides_processed_success() -> None:
    summary = _complete_g2(CONFIRMED)
    state = normalize_g2_state(
        summary,
        query_records=[
            {"video_id": "video-2", "query_status": "invalid_invented_content"}
        ],
        raw_records=[
            {
                "record_type": "video_query_generation_fallback",
                "video_id": "video-1",
                "parsed_json": None,
                "parse_error": "Response did not contain output text.",
                "schema_errors": ["missing_payload"],
                "raw_response": {
                    "status": "incomplete",
                    "incomplete_details": {"reason": "max_output_tokens"},
                },
            }
        ],
    )
    assert state == ValidationState(EXECUTED, PARTIAL, None, MISSING_PAYLOAD)


def test_no_evidence_is_terminal_but_not_semantically_evaluated() -> None:
    expected = ValidationState(NOT_ELIGIBLE, COMPLETE, None, NO_EVIDENCE)
    assert normalize_g1_state(None, evidence_available=False) == expected
    assert normalize_g2_state(None, evidence_available=False) == expected


def test_partial_state_cannot_publish_semantic_decision() -> None:
    with pytest.raises(ValueError, match="partial stage"):
        ValidationState(EXECUTED, PARTIAL, INSUFFICIENT_EVIDENCE, PENDING_BATCHES)


def test_legacy_partial_never_uses_global_status_as_authority() -> None:
    legacy = {
        "summary_completeness": "partial",
        "event_batch_status": "partial_errors",
        "next_batch_required": False,
        "pending_video_ids": [],
        "failed_video_ids": ["video-9"],
        "overall_validation_status": "partially_confirmed",
    }
    state = normalize_g2_state(legacy)
    assert state.validation_status is None
    assert state.reason_code == VIDEO_FAILURES
