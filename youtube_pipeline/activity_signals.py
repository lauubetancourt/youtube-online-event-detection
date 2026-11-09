from __future__ import annotations

import math
import re
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import partial
from numbers import Integral, Real
from typing import Any, Callable
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pandas as pd


CLOSED_INTERVAL = "closed"
LEFT_CLOSED_RIGHT_OPEN_INTERVAL = "left_closed_right_open"
SUPPORTED_INTERVAL_POLICIES = frozenset(
    {
        CLOSED_INTERVAL,
        LEFT_CLOSED_RIGHT_OPEN_INTERVAL,
    }
)
_SIGNAL_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_]*$")


def _require_non_empty_string(field_name: str, value: object) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string.")
    if not value.strip():
        raise ValueError(f"{field_name} must not be empty.")


def _require_utc_datetime(field_name: str, value: object) -> None:
    if not isinstance(value, datetime):
        raise TypeError(f"{field_name} must be a datetime.")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware UTC.")
    if value.utcoffset() != timedelta(0):
        raise ValueError(f"{field_name} must use UTC.")


@dataclass(frozen=True, slots=True)
class ActivitySignalDefinition:
    """Semantic identity of one activity signal, independent of its detector."""

    signal_id: str
    metric: str
    source: str
    scope: str
    unit: str
    window: str
    cadence: str
    time_basis: str
    timezone: str
    interval_policy: str

    def __post_init__(self) -> None:
        for field_name in (
            "signal_id",
            "metric",
            "source",
            "scope",
            "unit",
            "window",
            "cadence",
            "time_basis",
            "timezone",
            "interval_policy",
        ):
            _require_non_empty_string(field_name, getattr(self, field_name))
        if _SIGNAL_ID_PATTERN.fullmatch(self.signal_id) is None:
            raise ValueError(
                "signal_id must contain only lowercase letters, digits, and "
                "underscores, and must start with a letter or digit."
            )
        if self.interval_policy not in SUPPORTED_INTERVAL_POLICIES:
            supported = ", ".join(sorted(SUPPORTED_INTERVAL_POLICIES))
            raise ValueError(
                f"Unsupported interval_policy {self.interval_policy!r}; "
                f"expected one of: {supported}."
            )
        try:
            ZoneInfo(self.timezone)
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"Unknown signal timezone: {self.timezone!r}.") from exc


@dataclass(frozen=True, slots=True)
class ActivityObservation:
    """One causal value of an activity signal over an explicit time window."""

    signal: ActivitySignalDefinition
    observation_time_utc: datetime
    window_start_utc: datetime
    window_end_utc: datetime
    value: int | float
    support_count: int
    quality: str = "passed"

    def __post_init__(self) -> None:
        if not isinstance(self.signal, ActivitySignalDefinition):
            raise TypeError("signal must be an ActivitySignalDefinition.")
        for field_name in (
            "observation_time_utc",
            "window_start_utc",
            "window_end_utc",
        ):
            _require_utc_datetime(field_name, getattr(self, field_name))
        if self.window_start_utc >= self.window_end_utc:
            raise ValueError("window_start_utc must be before window_end_utc.")
        if self.window_end_utc > self.observation_time_utc:
            raise ValueError(
                "window_end_utc must not be later than observation_time_utc."
            )
        if isinstance(self.value, bool) or not isinstance(self.value, Real):
            raise TypeError("value must be a real number.")
        if not math.isfinite(float(self.value)):
            raise ValueError("value must be finite.")
        if isinstance(self.support_count, bool) or not isinstance(
            self.support_count, Integral
        ):
            raise TypeError("support_count must be an integer.")
        if self.support_count < 0:
            raise ValueError("support_count must be >= 0.")
        _require_non_empty_string("quality", self.quality)


def _align_timestamp_to_cadence(
    timestamp: pd.Timestamp,
    cadence: pd.Timedelta,
) -> pd.Timestamp:
    cadence_ns = int(cadence.value)
    if cadence_ns <= 0:
        raise ValueError("Signal cadence must be > 0.")
    timestamp_ns = int(timestamp.value)
    remainder = timestamp_ns % cadence_ns
    if remainder == 0:
        return timestamp
    return pd.Timestamp(timestamp_ns + (cadence_ns - remainder), tz="UTC")


def _duration_signal_id_token(value: str) -> str:
    duration_ns = int(pd.to_timedelta(value).value)
    if duration_ns <= 0:
        raise ValueError("Signal durations must be > 0.")
    for suffix, unit_ns in (
        ("s", 1_000_000_000),
        ("ms", 1_000_000),
        ("us", 1_000),
    ):
        if duration_ns % unit_ns == 0:
            return f"{duration_ns // unit_ns}{suffix}"
    return f"{duration_ns}ns"


def event_window_comment_count_definition(
    *,
    window: str,
    cadence: str,
    time_basis: str,
) -> ActivitySignalDefinition:
    """Build the explicit semantics of an event-time comment-count signal."""

    return ActivitySignalDefinition(
        signal_id=(
            "comment_count_event_window_"
            f"{_duration_signal_id_token(window)}_step_"
            f"{_duration_signal_id_token(cadence)}"
        ),
        metric="comment_count",
        source="prepared_comments",
        scope="selected_comment_stream",
        unit="comments",
        window=window,
        cadence=cadence,
        time_basis=time_basis,
        timezone="UTC",
        interval_policy=CLOSED_INTERVAL,
    )


def event_window_unique_author_count_definition(
    *,
    window: str,
    cadence: str,
    time_basis: str,
) -> ActivitySignalDefinition:
    """Build the explicit semantics of an event-time unique-author signal."""

    return ActivitySignalDefinition(
        signal_id=(
            "unique_author_count_event_window_"
            f"{_duration_signal_id_token(window)}_step_"
            f"{_duration_signal_id_token(cadence)}"
        ),
        metric="unique_authors",
        source="prepared_comments",
        scope="selected_comment_stream",
        unit="authors/window",
        window=window,
        cadence=cadence,
        time_basis=time_basis,
        timezone="UTC",
        interval_policy=CLOSED_INTERVAL,
    )


ActivityWindowMeasurement = Callable[
    [tuple[dict[str, Any], ...]],
    tuple[int | float, str],
]


def comment_count_measurement(
    events: tuple[dict[str, Any], ...],
) -> tuple[int, str]:
    return len(events), "passed"


def unique_author_count_measurement(
    events: tuple[dict[str, Any], ...],
    *,
    author_column: str,
) -> tuple[int, str]:
    """Count known authors; missing IDs degrade quality but are not invented."""

    _require_non_empty_string("author_column", author_column)
    author_ids: set[str] = set()
    missing_count = 0
    for event in events:
        value = event.get(author_column)
        if value is None or bool(pd.isna(value)):
            missing_count += 1
            continue
        normalized = str(value).strip()
        if not normalized:
            missing_count += 1
            continue
        author_ids.add(normalized)
    quality = "degraded_missing_author_id" if missing_count else "passed"
    return len(author_ids), quality


class EventWindowActivitySignal:
    """Incremental event-time window shared by activity metric functions."""

    def __init__(
        self,
        *,
        definition: ActivitySignalDefinition,
        timestamp_column: str,
        measurement: ActivityWindowMeasurement,
    ) -> None:
        if not isinstance(definition, ActivitySignalDefinition):
            raise TypeError("definition must be an ActivitySignalDefinition.")
        if definition.interval_policy != CLOSED_INTERVAL:
            raise ValueError(
                "EventWindowActivitySignal requires a closed interval policy."
            )
        _require_non_empty_string("timestamp_column", timestamp_column)
        if definition.time_basis != timestamp_column:
            raise ValueError(
                "definition.time_basis must match timestamp_column for this signal."
            )
        if not callable(measurement):
            raise TypeError("measurement must be callable.")

        self.definition = definition
        self.timestamp_column = timestamp_column
        self.measurement = measurement
        self.window = pd.to_timedelta(definition.window)
        self.cadence = pd.to_timedelta(definition.cadence)
        if self.window <= pd.Timedelta(0):
            raise ValueError("Signal window must be > 0.")
        if self.cadence <= pd.Timedelta(0):
            raise ValueError("Signal cadence must be > 0.")

        self._buffer: deque[dict[str, Any]] = deque()
        self._next_tick: pd.Timestamp | None = None
        self._last_advanced_cutoff: pd.Timestamp | None = None

    def start_at(self, start_time_utc: datetime) -> None:
        """Set the first scheduled tick without inventing a synthetic event.

        This is used by causal replay, where the simulation clock and the
        signal cadence are independent. Historical lookback events may still
        be supplied after initialization as long as they precede the first
        scheduled tick.
        """

        _require_utc_datetime("start_time_utc", start_time_utc)
        if self._next_tick is not None or self._buffer:
            raise RuntimeError("Activity signal scheduling is already initialized.")
        self._next_tick = _align_timestamp_to_cadence(
            pd.Timestamp(start_time_utc), self.cadence
        )

    def advance_to(
        self,
        data_cutoff_utc: datetime,
        *,
        on_observation: Callable[[ActivityObservation], None] | None = None,
    ) -> tuple[ActivityObservation, ...]:
        """Emit every due tick strictly before one causal data cutoff."""

        _require_utc_datetime("data_cutoff_utc", data_cutoff_utc)
        if self._next_tick is None:
            raise RuntimeError("Call start_at before advancing an empty signal.")
        cutoff = pd.Timestamp(data_cutoff_utc)
        if (
            self._last_advanced_cutoff is not None
            and cutoff < self._last_advanced_cutoff
        ):
            raise ValueError("Activity signal data cutoffs must not regress.")
        emitted: list[ActivityObservation] = []
        while self._next_tick < cutoff:
            observation = self._build_observation(self._next_tick)
            emitted.append(observation)
            if on_observation is not None:
                on_observation(observation)
            self._next_tick += self.cadence
        self._last_advanced_cutoff = cutoff
        return tuple(emitted)

    def on_event(
        self,
        event: dict[str, Any],
        *,
        on_observation: Callable[[ActivityObservation], None] | None = None,
        on_event_accepted: Callable[[dict[str, Any]], None] | None = None,
    ) -> tuple[ActivityObservation, ...]:
        """Process one event and emit observations in historical reference order."""

        event_ts = pd.to_datetime(
            event.get(self.timestamp_column),
            utc=True,
            errors="coerce",
        )
        if pd.isna(event_ts):
            return ()

        timestamp = pd.Timestamp(event_ts)
        if (
            self._last_advanced_cutoff is not None
            and timestamp < self._last_advanced_cutoff
        ):
            raise ValueError(
                "Activity event precedes a completed causal prefix; "
                "late arrivals are unsupported."
            )
        if self._next_tick is None:
            self._next_tick = _align_timestamp_to_cadence(timestamp, self.cadence)

        emitted: list[ActivityObservation] = []

        def emit(observation: ActivityObservation) -> None:
            emitted.append(observation)
            if on_observation is not None:
                on_observation(observation)

        while self._next_tick is not None and self._next_tick < timestamp:
            emit(self._build_observation(self._next_tick))
            self._next_tick += self.cadence

        normalized_event = dict(event)
        normalized_event[self.timestamp_column] = timestamp
        self._buffer.append(normalized_event)
        self._trim_buffer(timestamp)
        if on_event_accepted is not None:
            on_event_accepted(normalized_event)

        while self._next_tick is not None and self._next_tick == timestamp:
            emit(self._build_observation(self._next_tick))
            self._next_tick += self.cadence

        return tuple(emitted)

    def _build_observation(self, tick: pd.Timestamp) -> ActivityObservation:
        self._trim_buffer(tick)
        support = tuple(self._buffer)
        value, quality = self.measurement(support)
        return ActivityObservation(
            signal=self.definition,
            observation_time_utc=tick.to_pydatetime(),
            window_start_utc=(tick - self.window).to_pydatetime(),
            window_end_utc=tick.to_pydatetime(),
            value=value,
            support_count=len(support),
            quality=quality,
        )

    def _trim_buffer(self, reference_ts: pd.Timestamp) -> None:
        left = reference_ts - self.window
        while self._buffer:
            item_ts = pd.to_datetime(
                self._buffer[0].get(self.timestamp_column),
                utc=True,
                errors="coerce",
            )
            if pd.isna(item_ts) or pd.Timestamp(item_ts) >= left:
                break
            self._buffer.popleft()


class EventWindowCommentCountSignal(EventWindowActivitySignal):
    """Compatibility constructor for the reference comment-count signal."""

    def __init__(
        self,
        *,
        definition: ActivitySignalDefinition,
        timestamp_column: str,
    ) -> None:
        if definition.metric != "comment_count":
            raise ValueError("EventWindowCommentCountSignal requires comment_count.")
        super().__init__(
            definition=definition,
            timestamp_column=timestamp_column,
            measurement=comment_count_measurement,
        )


ActivitySignalFactory = Callable[
    [ActivitySignalDefinition],
    EventWindowActivitySignal,
]


def _create_comment_count_signal(
    definition: ActivitySignalDefinition,
) -> EventWindowActivitySignal:
    return EventWindowCommentCountSignal(
        definition=definition,
        timestamp_column=definition.time_basis,
    )


def _create_unique_author_count_signal(
    definition: ActivitySignalDefinition,
) -> EventWindowActivitySignal:
    if definition.metric != "unique_authors":
        raise ValueError(
            "The unique-author activity signal requires metric='unique_authors'."
        )
    return EventWindowActivitySignal(
        definition=definition,
        timestamp_column=definition.time_basis,
        measurement=partial(
            unique_author_count_measurement,
            author_column="author_id",
        ),
    )


COMMENT_COUNT_SIGNAL_ID = "comment_count_event_window_120s_step_30s"
UNIQUE_AUTHOR_COUNT_SIGNAL_ID = (
    "unique_author_count_event_window_120s_step_30s"
)
ACTIVITY_SIGNAL_REGISTRY: dict[str, ActivitySignalFactory] = {
    COMMENT_COUNT_SIGNAL_ID: _create_comment_count_signal,
    UNIQUE_AUTHOR_COUNT_SIGNAL_ID: _create_unique_author_count_signal,
}


def get_activity_signal_ids() -> tuple[str, ...]:
    """Return canonical IDs for activity signal families available at runtime."""

    return tuple(sorted(ACTIVITY_SIGNAL_REGISTRY))


def _activity_signal_registry_key(
    definition: ActivitySignalDefinition,
) -> str:
    registered_families = (
        (COMMENT_COUNT_SIGNAL_ID, event_window_comment_count_definition),
        (
            UNIQUE_AUTHOR_COUNT_SIGNAL_ID,
            event_window_unique_author_count_definition,
        ),
    )
    for registry_key, definition_factory in registered_families:
        expected = definition_factory(
            window=definition.window,
            cadence=definition.cadence,
            time_basis=definition.time_basis,
        )
        if definition.signal_id == expected.signal_id:
            return registry_key
    available = ", ".join(get_activity_signal_ids())
    raise ValueError(
        f"Unknown activity signal {definition.signal_id!r}; "
        f"available signal families: {available}."
    )


def create_activity_signal(
    definition: ActivitySignalDefinition,
) -> EventWindowActivitySignal:
    """Create fresh incremental state for one registered signal definition."""

    if not isinstance(definition, ActivitySignalDefinition):
        raise TypeError("definition must be an ActivitySignalDefinition.")
    factory = ACTIVITY_SIGNAL_REGISTRY[_activity_signal_registry_key(definition)]
    return factory(definition)


__all__ = [
    "ACTIVITY_SIGNAL_REGISTRY",
    "ActivityObservation",
    "ActivitySignalFactory",
    "ActivitySignalDefinition",
    "ActivityWindowMeasurement",
    "CLOSED_INTERVAL",
    "COMMENT_COUNT_SIGNAL_ID",
    "EventWindowActivitySignal",
    "EventWindowCommentCountSignal",
    "LEFT_CLOSED_RIGHT_OPEN_INTERVAL",
    "SUPPORTED_INTERVAL_POLICIES",
    "UNIQUE_AUTHOR_COUNT_SIGNAL_ID",
    "comment_count_measurement",
    "create_activity_signal",
    "event_window_comment_count_definition",
    "event_window_unique_author_count_definition",
    "get_activity_signal_ids",
    "unique_author_count_measurement",
]
