"""Configuration contract for one causal historical end-to-end replay."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd


FIXED_STEP_V1 = "fixed_step_v1"
PRECOMPUTED_STANCES_V1 = "precomputed_stances_v1"


def _utc(name: str, value: object) -> datetime:
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if not isinstance(value, datetime):
        raise TypeError(f"{name} must be a datetime or ISO-8601 string.")
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError(f"{name} must be timezone-aware UTC.")
    return value


def _duration(value: str) -> timedelta:
    if not isinstance(value, str) or not value.strip():
        raise TypeError("step must be a non-empty duration string.")
    parsed = pd.to_timedelta(value)
    if pd.isna(parsed) or parsed <= pd.Timedelta(0):
        raise ValueError("step must be a positive duration.")
    return parsed.to_pytimedelta()


@dataclass(frozen=True, slots=True)
class CausalEndToEndReplayConfig:
    """Frozen replay clock and persisted stance reuse configuration."""

    start_utc: datetime
    end_utc: datetime
    step: str
    output_dir: str | Path
    stance_artifact_path: str | Path
    stance_manifest_path: str | Path
    parameter_status: str
    clock_policy_id: str = FIXED_STEP_V1
    stance_mode: str = PRECOMPUTED_STANCES_V1

    def __post_init__(self) -> None:
        start = _utc("start_utc", self.start_utc)
        end = _utc("end_utc", self.end_utc)
        object.__setattr__(self, "start_utc", start)
        object.__setattr__(self, "end_utc", end)
        if start >= end:
            raise ValueError("start_utc must precede end_utc.")
        _duration(self.step)
        for name in ("output_dir", "stance_artifact_path", "stance_manifest_path"):
            value = getattr(self, name)
            if not isinstance(value, (str, Path)) or not str(value).strip():
                raise TypeError(f"{name} must be a non-empty path.")
        if self.parameter_status not in {
            "TEST_ONLY",
            "DEVELOPMENT_NOT_CALIBRATED",
        }:
            raise ValueError("Unsupported replay parameter_status.")
        if self.clock_policy_id != FIXED_STEP_V1:
            raise ValueError(f"clock_policy_id must be {FIXED_STEP_V1!r}.")
        if self.stance_mode != PRECOMPUTED_STANCES_V1:
            raise ValueError(f"stance_mode must be {PRECOMPUTED_STANCES_V1!r}.")

    @property
    def step_delta(self) -> timedelta:
        return _duration(self.step)

    @classmethod
    def from_mapping(
        cls, payload: Mapping[str, Any]
    ) -> CausalEndToEndReplayConfig:
        if not isinstance(payload, Mapping):
            raise TypeError("Causal replay configuration must be an object.")
        unknown = set(payload) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown causal replay fields: {sorted(unknown)}")
        values = dict(payload)
        for name in ("start_utc", "end_utc"):
            if name in values:
                values[name] = _utc(name, values[name])
        return cls(**values)


__all__ = [
    "FIXED_STEP_V1",
    "PRECOMPUTED_STANCES_V1",
    "CausalEndToEndReplayConfig",
]
