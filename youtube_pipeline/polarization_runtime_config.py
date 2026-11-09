"""Configuration and prepared-comment scope for polarization runtimes."""
from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any, Iterable

from youtube_pipeline.signal_observations import require_text


YOUTUBE_COMMENTS_CSV_V1 = "youtube_comments_csv_v1"
SILVER_COMMENTS_V1 = "silver_comments_v1"
PROVISIONAL_RUNTIME_INPUT = "PROVISIONAL_RUNTIME_INPUT"
CAUSAL_MINIMAL_V1_INPUT = "CAUSAL_MINIMAL_V1"
ALL_FROM_COMMENT_STREAM = "ALL_FROM_COMMENT_STREAM"
EXPLICIT_FILTER = "EXPLICIT_FILTER"


def normalize_video_id(value: object) -> str | None:
    """Return a canonical nonblank ID, ignoring null-like prepared values."""

    if value is None:
        return None
    try:
        if value != value:  # NaN without importing a dataframe library here.
            return None
    except (TypeError, ValueError):
        return None
    text = str(value).strip()
    return text or None


def _video_ids_hash(video_ids: tuple[str, ...]) -> str:
    return hashlib.sha256("\n".join(video_ids).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class VideoScope:
    mode: str
    effective_video_ids: tuple[str, ...]
    effective_video_ids_hash: str
    configured_video_count: int | None
    matched_video_count: int | None
    unmatched_video_ids: tuple[str, ...]

    @property
    def effective_video_count(self) -> int:
        return len(self.effective_video_ids)

    @property
    def unmatched_video_count(self) -> int:
        return len(self.unmatched_video_ids)

    def to_manifest(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "video_scope_mode": self.mode,
            "effective_video_count": self.effective_video_count,
            "effective_video_ids_hash": self.effective_video_ids_hash,
        }
        if self.mode == EXPLICIT_FILTER:
            result.update({
                "configured_video_count": self.configured_video_count,
                "matched_video_count": self.matched_video_count,
                "unmatched_video_count": self.unmatched_video_count,
                "unmatched_video_ids": list(self.unmatched_video_ids),
            })
        return result


def resolve_video_scope(
    prepared_video_ids: Iterable[object],
    configured_video_ids: tuple[str, ...] | None,
) -> VideoScope:
    """Resolve an optional configured filter against prepared comments only."""

    available = tuple(sorted({
        video_id
        for value in prepared_video_ids
        if (video_id := normalize_video_id(value)) is not None
    }))
    if not available:
        raise ValueError("Prepared comments contain no nonblank video_id values.")
    if configured_video_ids is None:
        return VideoScope(
            mode=ALL_FROM_COMMENT_STREAM,
            effective_video_ids=available,
            effective_video_ids_hash=_video_ids_hash(available),
            configured_video_count=None,
            matched_video_count=None,
            unmatched_video_ids=(),
        )
    configured = set(configured_video_ids)
    effective = tuple(video_id for video_id in available if video_id in configured)
    unmatched = tuple(sorted(configured.difference(available)))
    if not effective:
        raise ValueError(
            "Configured polarization runtime video_ids do not match any "
            "prepared comments."
        )
    return VideoScope(
        mode=EXPLICIT_FILTER,
        effective_video_ids=effective,
        effective_video_ids_hash=_video_ids_hash(effective),
        configured_video_count=len(configured_video_ids),
        matched_video_count=len(effective),
        unmatched_video_ids=unmatched,
    )


@dataclass(frozen=True, slots=True)
class PolarizationRuntimeConfig:
    """Optionally filter the prepared comment stream without owning time."""

    video_ids: tuple[str, ...] | None = None
    input_schema: str = YOUTUBE_COMMENTS_CSV_V1
    source_role: str = PROVISIONAL_RUNTIME_INPUT

    def __post_init__(self) -> None:
        if self.video_ids is not None:
            if not isinstance(self.video_ids, (list, tuple)):
                raise TypeError("video_ids must be null or a list/tuple.")
            video_ids = tuple(self.video_ids)
            if not video_ids:
                raise ValueError("video_ids must not be an empty list; omit it to use all videos.")
            if len(video_ids) != len(set(video_ids)):
                raise ValueError("video_ids must be unique.")
            for video_id in video_ids:
                require_text("video_id", video_id)
            object.__setattr__(self, "video_ids", video_ids)
        supported = {
            YOUTUBE_COMMENTS_CSV_V1: PROVISIONAL_RUNTIME_INPUT,
            SILVER_COMMENTS_V1: CAUSAL_MINIMAL_V1_INPUT,
        }
        expected_role = supported.get(self.input_schema)
        if expected_role is None:
            raise ValueError(
                f"input_schema must be one of {sorted(supported)!r}."
            )
        if self.source_role != expected_role:
            raise ValueError(
                f"source_role must be {expected_role!r} for "
                f"input_schema={self.input_schema!r}."
            )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> PolarizationRuntimeConfig:
        if not isinstance(payload, Mapping):
            raise TypeError("Polarization runtime configuration must be an object.")
        unknown = set(payload) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown polarization runtime fields: {sorted(unknown)}")
        return cls(**dict(payload))


__all__ = [
    "ALL_FROM_COMMENT_STREAM",
    "CAUSAL_MINIMAL_V1_INPUT",
    "EXPLICIT_FILTER",
    "PROVISIONAL_RUNTIME_INPUT",
    "SILVER_COMMENTS_V1",
    "VideoScope",
    "YOUTUBE_COMMENTS_CSV_V1",
    "PolarizationRuntimeConfig",
    "normalize_video_id",
    "resolve_video_scope",
]
