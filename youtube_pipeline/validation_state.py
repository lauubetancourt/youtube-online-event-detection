"""Transversal validation-state contract for G1 and G2 artifacts.

Semantic validation, technical execution and processing completeness are
independent dimensions.  The helpers in this module also provide a read-only
compatibility boundary for historical artifacts.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Final, Iterable, Mapping

from youtube_pipeline.validation_status import normalize_validation_status


EXECUTED: Final = "EXECUTED"
NOT_EXECUTED: Final = "NOT_EXECUTED"
NOT_ELIGIBLE: Final = "NOT_ELIGIBLE"
FAILED: Final = "FAILED"
EXECUTION_STATUS_VALUES: Final[frozenset[str]] = frozenset(
    {EXECUTED, NOT_EXECUTED, NOT_ELIGIBLE, FAILED}
)

COMPLETE: Final = "COMPLETE"
PARTIAL: Final = "PARTIAL"
COMPLETENESS_STATUS_VALUES: Final[frozenset[str]] = frozenset({COMPLETE, PARTIAL})

NO_EVIDENCE: Final = "NO_EVIDENCE"
CONTEXT_LIMIT_EXCEEDED: Final = "CONTEXT_LIMIT_EXCEEDED"
INVALID_QUERY: Final = "INVALID_QUERY"
INVALID_CITATION: Final = "INVALID_CITATION"
MISSING_PAYLOAD: Final = "MISSING_PAYLOAD"
PROVIDER_ERROR: Final = "PROVIDER_ERROR"
TIMEOUT: Final = "TIMEOUT"
PENDING_BATCHES: Final = "PENDING_BATCHES"
VIDEO_FAILURES: Final = "VIDEO_FAILURES"
REASON_CODE_VALUES: Final[frozenset[str]] = frozenset(
    {
        NO_EVIDENCE,
        CONTEXT_LIMIT_EXCEEDED,
        INVALID_QUERY,
        INVALID_CITATION,
        MISSING_PAYLOAD,
        PROVIDER_ERROR,
        TIMEOUT,
        PENDING_BATCHES,
        VIDEO_FAILURES,
    }
)


@dataclass(frozen=True, slots=True)
class ValidationState:
    execution_status: str
    completeness_status: str
    validation_status: str | None
    reason_code: str | None

    def __post_init__(self) -> None:
        if self.execution_status not in EXECUTION_STATUS_VALUES:
            raise ValueError(f"Unknown execution status: {self.execution_status!r}")
        if self.completeness_status not in COMPLETENESS_STATUS_VALUES:
            raise ValueError(
                f"Unknown completeness status: {self.completeness_status!r}"
            )
        normalize_validation_status(self.validation_status, allow_none=True)
        if self.reason_code is not None and self.reason_code not in REASON_CODE_VALUES:
            raise ValueError(f"Unknown reason code: {self.reason_code!r}")
        if self.completeness_status != COMPLETE and self.validation_status is not None:
            raise ValueError("A partial stage cannot expose a semantic decision.")
        if self.execution_status in {NOT_EXECUTED, NOT_ELIGIBLE, FAILED} and (
            self.validation_status is not None
        ):
            raise ValueError(
                "A non-executed, non-eligible or failed stage cannot expose a "
                "semantic decision."
            )

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)


def _strings(value: object) -> list[str]:
    if isinstance(value, (list, tuple, set, frozenset)):
        return [text for item in value for text in _strings(item)]
    if value in (None, ""):
        return []
    return [str(value)]


def _error_text(*values: object) -> str:
    return " ".join(
        item.lower()
        for value in values
        for item in _strings(value)
    )


def _explicit_state(record: Mapping[str, Any]) -> ValidationState | None:
    keys = {
        "execution_status",
        "completeness_status",
        "validation_status",
        "reason_code",
    }
    if not keys.issubset(record):
        return None
    validation = normalize_validation_status(
        record.get("validation_status"), allow_none=True
    )
    return ValidationState(
        execution_status=str(record["execution_status"]),
        completeness_status=str(record["completeness_status"]),
        validation_status=validation,
        reason_code=(
            str(record["reason_code"])
            if record.get("reason_code") not in (None, "")
            else None
        ),
    )


def no_evidence_state() -> ValidationState:
    """Return the legitimate terminal state for a stage without evidence."""

    return ValidationState(NOT_ELIGIBLE, COMPLETE, None, NO_EVIDENCE)


def normalize_g1_state(
    report: Mapping[str, Any] | None,
    *,
    evidence_available: bool = True,
) -> ValidationState:
    """Normalize a current or legacy G1 report without rewriting it."""

    if not evidence_available:
        return no_evidence_state()
    if report is None:
        return ValidationState(NOT_EXECUTED, PARTIAL, None, MISSING_PAYLOAD)
    explicit = _explicit_state(report)
    if explicit is not None:
        return explicit

    error_text = _error_text(
        report.get("limitations"),
        report.get("errors"),
        (report.get("citation_verification") or {}).get("errors")
        if isinstance(report.get("citation_verification"), Mapping)
        else None,
    )
    if report.get("generation_error"):
        if "context_limit_exceeded" in error_text:
            reason = CONTEXT_LIMIT_EXCEEDED
        elif "timeout" in error_text:
            reason = TIMEOUT
        elif "citation" in error_text:
            reason = INVALID_CITATION
        elif "payload" in error_text or "parse" in error_text or "schema" in error_text:
            reason = MISSING_PAYLOAD
        else:
            reason = PROVIDER_ERROR
        return ValidationState(FAILED, PARTIAL, None, reason)

    status = normalize_validation_status(report.get("validation_status"))
    return ValidationState(EXECUTED, COMPLETE, status, None)


def g2_is_complete(summary: Mapping[str, Any]) -> bool:
    """Apply the exact G2 completeness rule from the public contract."""

    return (
        summary.get("summary_completeness") == "complete"
        and summary.get("event_batch_status") == "complete"
        and summary.get("next_batch_required") is False
        and summary.get("pending_video_ids") == []
        and summary.get("failed_video_ids") == []
    )


def _g2_failure_reason(
    summary: Mapping[str, Any],
    video_reports: Iterable[Mapping[str, Any]],
) -> str | None:
    text = _error_text(
        summary.get("failure_reason_codes"),
        summary.get("global_limitations"),
        *(
            (
                report.get("reason_code"),
                report.get("status_reason"),
                report.get("limitations"),
                (report.get("citation_verification") or {}).get("errors")
                if isinstance(report.get("citation_verification"), Mapping)
                else None,
            )
            for report in video_reports
        ),
    )
    if "missing_payload" in text or "missing_query_payload" in text:
        return MISSING_PAYLOAD
    if "invalid_query" in text or "invalid_video_query" in text:
        return INVALID_QUERY
    if "invalid_citation" in text or "citation" in text:
        return INVALID_CITATION
    if "timeout" in text:
        return TIMEOUT
    if summary.get("failed_video_ids"):
        return VIDEO_FAILURES
    if summary.get("next_batch_required") is True or summary.get(
        "pending_video_ids"
    ):
        return PENDING_BATCHES
    if "provider" in text or "serper" in text:
        return PROVIDER_ERROR
    return None


def normalize_g2_state(
    summary: Mapping[str, Any] | None,
    *,
    video_reports: Iterable[Mapping[str, Any]] = (),
    query_records: Iterable[Mapping[str, Any]] = (),
    raw_records: Iterable[Mapping[str, Any]] = (),
    evidence_available: bool = True,
    attempted: bool = True,
    technical_failure_reason: str | None = None,
) -> ValidationState:
    """Normalize a current or legacy G2 summary.

    A legacy ``overall_validation_status`` is consulted only after the exact
    completeness predicate succeeds.
    """

    if not evidence_available:
        return no_evidence_state()
    if summary is None:
        if attempted:
            reason = technical_failure_reason or MISSING_PAYLOAD
            return ValidationState(FAILED, PARTIAL, None, reason)
        return ValidationState(NOT_EXECUTED, PARTIAL, None, technical_failure_reason)
    video_reports = tuple(video_reports)
    query_records = tuple(query_records)
    raw_records = tuple(raw_records)
    explicit = _explicit_state(summary)
    if explicit is not None:
        return explicit
    if any(
        record.get("record_type") == "video_query_generation_fallback"
        and (
            record.get("parsed_json") is None
            or bool(record.get("parse_error"))
            or bool(record.get("schema_errors"))
        )
        for record in raw_records
    ):
        return ValidationState(EXECUTED, PARTIAL, None, MISSING_PAYLOAD)
    if any(
        record.get("query_status") == "invalid_invented_content"
        for record in query_records
    ):
        return ValidationState(EXECUTED, PARTIAL, None, INVALID_QUERY)
    failed_report_reasons = {
        report.get("reason_code")
        for report in video_reports
        if report.get("execution_status") == FAILED
    }
    if INVALID_CITATION in failed_report_reasons:
        return ValidationState(EXECUTED, PARTIAL, None, INVALID_CITATION)
    if not g2_is_complete(summary):
        reason = _g2_failure_reason(summary, video_reports) or PENDING_BATCHES
        return ValidationState(EXECUTED, PARTIAL, None, reason)
    raw_status = summary.get("validation_status")
    if raw_status in (None, ""):
        raw_status = summary.get("overall_validation_status")
    status = normalize_validation_status(raw_status)
    return ValidationState(EXECUTED, COMPLETE, status, None)


__all__ = [
    "COMPLETE",
    "COMPLETENESS_STATUS_VALUES",
    "CONTEXT_LIMIT_EXCEEDED",
    "EXECUTED",
    "EXECUTION_STATUS_VALUES",
    "FAILED",
    "INVALID_CITATION",
    "INVALID_QUERY",
    "MISSING_PAYLOAD",
    "NOT_ELIGIBLE",
    "NOT_EXECUTED",
    "NO_EVIDENCE",
    "PARTIAL",
    "PENDING_BATCHES",
    "PROVIDER_ERROR",
    "REASON_CODE_VALUES",
    "TIMEOUT",
    "VIDEO_FAILURES",
    "ValidationState",
    "g2_is_complete",
    "no_evidence_state",
    "normalize_g1_state",
    "normalize_g2_state",
]
