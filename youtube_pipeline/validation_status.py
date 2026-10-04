"""Shared validation-status contract for internal G-1 and external G-2.

New artifacts use the three uppercase statuses.  Legacy spellings remain readable
at artifact/configuration boundaries, but are never emitted by normalizers.
"""
from __future__ import annotations

from typing import Final


CONFIRMED: Final = "CONFIRMED"
INSUFFICIENT_EVIDENCE: Final = "INSUFFICIENT_EVIDENCE"
NOT_CONFIRMED: Final = "NOT_CONFIRMED"

VALIDATION_STATUS_VALUES: Final[frozenset[str]] = frozenset(
    {CONFIRMED, INSUFFICIENT_EVIDENCE, NOT_CONFIRMED}
)

LEGACY_STATUS_MAPPING: Final[dict[str, str]] = {
    "confirmed": CONFIRMED,
    "partially_confirmed": INSUFFICIENT_EVIDENCE,
    "ambiguous": INSUFFICIENT_EVIDENCE,
    "insufficient_evidence": INSUFFICIENT_EVIDENCE,
    "not_confirmed": NOT_CONFIRMED,
}


def normalize_validation_status(
    value: object,
    *,
    allow_none: bool = False,
) -> str | None:
    """Return the authoritative three-state spelling for new processing.

    Unknown values fail explicitly so a legacy or malformed artifact cannot be
    silently assigned a different methodological meaning.
    """

    if value is None:
        if allow_none:
            return None
        raise ValueError("validation status is required")
    if not isinstance(value, str):
        raise TypeError("validation status must be a string")
    normalized = value.strip()
    if normalized in VALIDATION_STATUS_VALUES:
        return normalized
    mapped = LEGACY_STATUS_MAPPING.get(normalized.lower())
    if mapped is not None:
        return mapped
    raise ValueError(f"Unknown validation status: {value!r}")


def is_legacy_validation_status(value: object) -> bool:
    return isinstance(value, str) and value.strip().lower() in LEGACY_STATUS_MAPPING


__all__ = [
    "CONFIRMED",
    "INSUFFICIENT_EVIDENCE",
    "LEGACY_STATUS_MAPPING",
    "NOT_CONFIRMED",
    "VALIDATION_STATUS_VALUES",
    "is_legacy_validation_status",
    "normalize_validation_status",
]
