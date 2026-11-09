"""Deterministic parsing and causal comparison for external publication dates."""
from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any


EXACT = "EXACT"
DATE_ONLY = "DATE_ONLY"
UNKNOWN = "UNKNOWN"

BEFORE_REFERENCE = "BEFORE_REFERENCE"
AT_REFERENCE = "AT_REFERENCE"
AFTER_REFERENCE = "AFTER_REFERENCE"
SAME_DAY_TIME_UNKNOWN = "SAME_DAY_TIME_UNKNOWN"
UNKNOWN_TEMPORAL_RELATION = "UNKNOWN_TEMPORAL_RELATION"

CAUSALLY_AVAILABLE = "CAUSALLY_AVAILABLE"
FUTURE_RELATIVE_TO_CUTOFF = "FUTURE_RELATIVE_TO_CUTOFF"

_DAY_FIRST_DATE = re.compile(r"\d{1,2}/\d{1,2}/\d{4}")
_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


@dataclass(frozen=True, slots=True)
class ExternalPublicationTime:
    raw_value: Any
    precision: str
    timestamp_utc: datetime | None = None
    calendar_date: date | None = None

    def __post_init__(self) -> None:
        if self.precision not in {EXACT, DATE_ONLY, UNKNOWN}:
            raise ValueError("Unsupported external publication-time precision.")
        if self.precision == EXACT:
            if self.timestamp_utc is None or self.timestamp_utc.tzinfo is None:
                raise ValueError("EXACT publication time requires an aware timestamp.")
            if self.calendar_date != self.timestamp_utc.astimezone(timezone.utc).date():
                raise ValueError("EXACT publication date must match its UTC timestamp.")
        elif self.precision == DATE_ONLY:
            if self.calendar_date is None or self.timestamp_utc is not None:
                raise ValueError("DATE_ONLY requires a date and no synthetic timestamp.")
        elif self.timestamp_utc is not None or self.calendar_date is not None:
            raise ValueError("UNKNOWN publication time cannot contain parsed values.")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "raw_publication_date": self.raw_value,
            "parsed_publication_time_utc": (
                self.timestamp_utc.isoformat() if self.timestamp_utc is not None else None
            ),
            "publication_date": (
                self.calendar_date.isoformat() if self.calendar_date is not None else None
            ),
            "publication_time_precision": self.precision,
        }


def _parse_aware_datetime(raw: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed.astimezone(timezone.utc)


def parse_external_publication_time(value: object) -> ExternalPublicationTime:
    """Parse only explicit supported formats; never invoke free-form heuristics."""

    if not isinstance(value, str) or not value.strip():
        return ExternalPublicationTime(value, UNKNOWN)
    raw = value.strip()
    if _DAY_FIRST_DATE.fullmatch(raw):
        try:
            parsed_date = datetime.strptime(raw, "%d/%m/%Y").date()
        except ValueError:
            return ExternalPublicationTime(value, UNKNOWN)
        return ExternalPublicationTime(value, DATE_ONLY, calendar_date=parsed_date)
    if _ISO_DATE.fullmatch(raw):
        try:
            parsed_date = datetime.strptime(raw, "%Y-%m-%d").date()
        except ValueError:
            return ExternalPublicationTime(value, UNKNOWN)
        return ExternalPublicationTime(value, DATE_ONLY, calendar_date=parsed_date)
    parsed_timestamp = _parse_aware_datetime(raw)
    if parsed_timestamp is not None:
        return ExternalPublicationTime(
            value,
            EXACT,
            timestamp_utc=parsed_timestamp,
            calendar_date=parsed_timestamp.date(),
        )
    return ExternalPublicationTime(value, UNKNOWN)


def external_publication_time_from_record(
    record: Mapping[str, Any],
) -> ExternalPublicationTime:
    """Use persisted parsed fields when present, falling back for legacy artifacts."""

    precision = record.get("publication_time_precision")
    raw = record.get("raw_publication_date")
    if raw is None:
        raw = record.get("external_source_published_at")
    if raw is None:
        raw = record.get("published_at")
    if precision == EXACT:
        timestamp = _parse_aware_datetime(
            str(record.get("parsed_publication_time_utc") or "")
        )
        if timestamp is not None:
            return ExternalPublicationTime(
                raw,
                EXACT,
                timestamp_utc=timestamp,
                calendar_date=timestamp.date(),
            )
        return ExternalPublicationTime(raw, UNKNOWN)
    if precision == DATE_ONLY:
        publication_date = record.get("publication_date")
        if isinstance(publication_date, str) and _ISO_DATE.fullmatch(publication_date):
            try:
                parsed_date = datetime.strptime(publication_date, "%Y-%m-%d").date()
            except ValueError:
                return ExternalPublicationTime(raw, UNKNOWN)
            return ExternalPublicationTime(raw, DATE_ONLY, calendar_date=parsed_date)
        return ExternalPublicationTime(raw, UNKNOWN)
    if precision == UNKNOWN:
        return ExternalPublicationTime(raw, UNKNOWN)
    return parse_external_publication_time(raw)


def _reference_time_utc(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        parsed = _parse_aware_datetime(value.strip())
        if parsed is None:
            raise ValueError("Reference time must be an ISO-8601 timestamp with timezone.")
    else:
        raise TypeError("Reference time must be a datetime or string.")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("Reference time must be timezone-aware.")
    return parsed.astimezone(timezone.utc)


def temporal_relation_to_reference(
    publication: ExternalPublicationTime,
    reference_utc: str | datetime,
) -> str:
    reference = _reference_time_utc(reference_utc)
    if publication.precision == EXACT:
        assert publication.timestamp_utc is not None
        if publication.timestamp_utc < reference:
            return BEFORE_REFERENCE
        if publication.timestamp_utc == reference:
            return AT_REFERENCE
        return AFTER_REFERENCE
    if publication.precision == DATE_ONLY:
        assert publication.calendar_date is not None
        if publication.calendar_date < reference.date():
            return BEFORE_REFERENCE
        if publication.calendar_date > reference.date():
            return AFTER_REFERENCE
        return SAME_DAY_TIME_UNKNOWN
    return UNKNOWN_TEMPORAL_RELATION


def temporal_relation_to_cutoff(
    publication: ExternalPublicationTime,
    cutoff_utc: str | datetime,
) -> str:
    """Classify against the existing strict ``published_at < cutoff`` boundary."""

    relation = temporal_relation_to_reference(publication, cutoff_utc)
    if relation == BEFORE_REFERENCE:
        return CAUSALLY_AVAILABLE
    if relation in {AT_REFERENCE, AFTER_REFERENCE}:
        return FUTURE_RELATIVE_TO_CUTOFF
    if relation == SAME_DAY_TIME_UNKNOWN:
        return SAME_DAY_TIME_UNKNOWN
    return UNKNOWN_TEMPORAL_RELATION


__all__ = [
    "AFTER_REFERENCE",
    "AT_REFERENCE",
    "BEFORE_REFERENCE",
    "CAUSALLY_AVAILABLE",
    "DATE_ONLY",
    "EXACT",
    "ExternalPublicationTime",
    "FUTURE_RELATIVE_TO_CUTOFF",
    "SAME_DAY_TIME_UNKNOWN",
    "UNKNOWN",
    "UNKNOWN_TEMPORAL_RELATION",
    "external_publication_time_from_record",
    "parse_external_publication_time",
    "temporal_relation_to_cutoff",
    "temporal_relation_to_reference",
]
