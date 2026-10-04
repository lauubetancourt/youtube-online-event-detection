from __future__ import annotations

import pytest

from youtube_pipeline.validation_status import (
    CONFIRMED,
    INSUFFICIENT_EVIDENCE,
    NOT_CONFIRMED,
    normalize_validation_status,
)


@pytest.mark.parametrize(
    ("legacy", "expected"),
    [
        ("confirmed", CONFIRMED),
        ("partially_confirmed", INSUFFICIENT_EVIDENCE),
        ("ambiguous", INSUFFICIENT_EVIDENCE),
        ("insufficient_evidence", INSUFFICIENT_EVIDENCE),
        ("not_confirmed", NOT_CONFIRMED),
    ],
)
def test_legacy_status_mapping_is_explicit(legacy: str, expected: str) -> None:
    assert normalize_validation_status(legacy) == expected


@pytest.mark.parametrize(
    "status", [CONFIRMED, INSUFFICIENT_EVIDENCE, NOT_CONFIRMED]
)
def test_new_statuses_are_idempotent(status: str) -> None:
    assert normalize_validation_status(status) == status


def test_unknown_status_fails_explicitly() -> None:
    with pytest.raises(ValueError, match="Unknown validation status"):
        normalize_validation_status("possibly_confirmed")
