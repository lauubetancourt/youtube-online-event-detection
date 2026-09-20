"""Configuration-only contract for the standalone polarization runtime."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any

from youtube_pipeline.signal_observations import require_text


YOUTUBE_COMMENTS_CSV_V1 = "youtube_comments_csv_v1"
SILVER_COMMENTS_V1 = "silver_comments_v1"
PROVISIONAL_RUNTIME_INPUT = "PROVISIONAL_RUNTIME_INPUT"
CAUSAL_MINIMAL_V1_INPUT = "CAUSAL_MINIMAL_V1"


@dataclass(frozen=True, slots=True)
class PolarizationRuntimeConfig:
    """Select an explicit real-input slice without owning temporal semantics."""

    video_ids: tuple[str, ...]
    input_schema: str = YOUTUBE_COMMENTS_CSV_V1
    source_role: str = PROVISIONAL_RUNTIME_INPUT

    def __post_init__(self) -> None:
        if not isinstance(self.video_ids, (list, tuple)):
            raise TypeError("video_ids must be a list or tuple.")
        video_ids = tuple(self.video_ids)
        if not video_ids or len(video_ids) != len(set(video_ids)):
            raise ValueError("video_ids must be nonempty and unique.")
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
    "CAUSAL_MINIMAL_V1_INPUT",
    "PROVISIONAL_RUNTIME_INPUT",
    "SILVER_COMMENTS_V1",
    "YOUTUBE_COMMENTS_CSV_V1",
    "PolarizationRuntimeConfig",
]
