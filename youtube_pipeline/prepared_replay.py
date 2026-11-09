from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .detectors import DEFAULT_DETECTOR, create_detector
from .monitoring import build_event_time_window_stream
from .replay import read_dataset_for_playback, replay_events
from .stream_runtime import Stream, require_streamz


DEFAULT_PREPARED_TIMESTAMP_COLUMN = "event_time_utc"
DEFAULT_REPLAY_WINDOW_SIZE = "20min"
DEFAULT_REPLAY_SPEED = 120.0
DEFAULT_REPLAY_MAX_SLEEP_SECONDS = 0.2


@dataclass(frozen=True)
class PreparedDatasetConfig:
    """A dataset already prepared for replay by the current pipeline."""

    path: str | Path
    timestamp_column: str = DEFAULT_PREPARED_TIMESTAMP_COLUMN

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "PreparedDatasetConfig":
        config_payload = payload.get("prepared_dataset", payload)
        if not isinstance(config_payload, dict):
            raise ValueError("Prepared dataset config must be an object.")
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(config_payload) - allowed)
        if unknown:
            raise ValueError(f"Unknown prepared dataset config fields: {unknown}")
        return cls(**config_payload)

    def __post_init__(self) -> None:
        if not str(self.path).strip():
            raise ValueError("Prepared dataset path must not be empty.")
        if not isinstance(self.timestamp_column, str):
            raise TypeError("timestamp_column must be a string.")
        if not self.timestamp_column.strip():
            raise ValueError("timestamp_column must not be empty.")


@dataclass(frozen=True)
class ReplayConfig:
    """Parameters and output of the existing prepared-dataset replay stage."""

    output_snapshots: str | Path
    window_size: str = DEFAULT_REPLAY_WINDOW_SIZE
    speed: float = DEFAULT_REPLAY_SPEED
    max_sleep_seconds: float | None = DEFAULT_REPLAY_MAX_SLEEP_SECONDS
    start: str | None = None
    end: str | None = None

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "ReplayConfig":
        config_payload = payload.get("replay", payload)
        if not isinstance(config_payload, dict):
            raise ValueError("Replay config must be an object.")
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(config_payload) - allowed)
        if unknown:
            raise ValueError(f"Unknown replay config fields: {unknown}")
        return cls(**config_payload)

    def __post_init__(self) -> None:
        if not str(self.output_snapshots).strip():
            raise ValueError("Replay output_snapshots must not be empty.")
        if not isinstance(self.window_size, str):
            raise TypeError("window_size must be a string.")
        if not self.window_size.strip():
            raise ValueError("window_size must not be empty.")
        if self.speed <= 0:
            raise ValueError("speed must be > 0.")


def run_prepared_replay(
    dataset: PreparedDatasetConfig,
    replay: ReplayConfig,
    *,
    detector_name: str = DEFAULT_DETECTOR,
    detector_params: dict[str, Any] | None = None,
) -> Path:
    """Run the existing playback behavior from resolved component configs."""

    if not isinstance(dataset, PreparedDatasetConfig):
        raise TypeError("dataset must be PreparedDatasetConfig.")
    if not isinstance(replay, ReplayConfig):
        raise TypeError("replay must be ReplayConfig.")
    require_streamz()

    events = read_dataset_for_playback(
        dataset.path,
        ts_col=dataset.timestamp_column,
    )
    source = Stream()
    snapshots: list[dict[str, Any]] = []
    resolved_detector_params = dict(detector_params or {})
    resolved_detector_params.setdefault("ts_col", dataset.timestamp_column)
    detector = create_detector(detector_name, **resolved_detector_params)

    (
        build_event_time_window_stream(
            source,
            window_size=replay.window_size,
            ts_col=dataset.timestamp_column,
        ).sink(snapshots.append)
    )

    replay_events(
        source=source,
        events_df=events,
        ts_col=dataset.timestamp_column,
        speed=replay.speed,
        max_sleep_seconds=replay.max_sleep_seconds,
        start=replay.start,
        end=replay.end,
        event_hooks=[detector.on_event],
    )

    snapshot_df = pd.json_normalize(snapshots, sep=".")
    output = Path(replay.output_snapshots)
    output.parent.mkdir(parents=True, exist_ok=True)
    snapshot_df.to_csv(output, index=False)
    return output


__all__ = [
    "DEFAULT_PREPARED_TIMESTAMP_COLUMN",
    "DEFAULT_REPLAY_MAX_SLEEP_SECONDS",
    "DEFAULT_REPLAY_SPEED",
    "DEFAULT_REPLAY_WINDOW_SIZE",
    "PreparedDatasetConfig",
    "ReplayConfig",
    "run_prepared_replay",
]
