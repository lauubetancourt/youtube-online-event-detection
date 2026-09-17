from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import math
from numbers import Real
from types import MappingProxyType
from typing import Any, Protocol, overload

from youtube_pipeline.activity_signals import ActivityObservation
from youtube_pipeline.signal_observations import SignalObservation, VALUE_AVAILABLE


NumericObservation = ActivityObservation | SignalObservation


def observation_signal_id(observation: NumericObservation) -> str:
    """Bridge the historical nested identity without changing either contract."""
    if isinstance(observation, ActivityObservation):
        return observation.signal.signal_id
    if isinstance(observation, SignalObservation):
        return observation.signal_id
    raise TypeError("observation must be an ActivityObservation or SignalObservation.")


def require_numeric_observation(observation: NumericObservation) -> str:
    """Adapters accept values only; absence handling belongs to dispatch."""
    signal_id = observation_signal_id(observation)
    if isinstance(observation, SignalObservation) and observation.status != VALUE_AVAILABLE:
        raise ValueError("Only VALUE_AVAILABLE observations can update a detector.")
    if isinstance(observation.value, bool) or not isinstance(observation.value, Real):
        raise TypeError("Detector observation value must be a real number.")
    if not math.isfinite(float(observation.value)):
        raise ValueError("Detector observation value must be finite.")
    return signal_id


_COMPONENT_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_]*$")


def _require_component_id(field_name: str, value: object) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string.")
    if _COMPONENT_ID_PATTERN.fullmatch(value) is None:
        raise ValueError(
            f"{field_name} must contain only lowercase letters, digits, "
            "and underscores, and must start with a letter or digit."
        )


def _freeze_metadata(value: Any, *, location: str) -> Any:
    if value is None or isinstance(value, (str, int, bool, datetime)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{location} must contain only finite numbers.")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError(f"{location} keys must be non-empty strings.")
            frozen[key] = _freeze_metadata(item, location=f"{location}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_metadata(item, location=f"{location}[]") for item in value
        )
    raise TypeError(
        f"{location} contains unsupported value type {type(value).__name__}."
    )


@dataclass(frozen=True, slots=True)
class DetectionResult:
    """Detector-neutral decision for one numeric signal observation."""

    detector_id: str
    signal_id: str
    observation_time_utc: datetime
    triggered: bool
    quality: str
    score: float | None = None
    detector_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_component_id("detector_id", self.detector_id)
        _require_component_id("signal_id", self.signal_id)
        if not isinstance(self.observation_time_utc, datetime):
            raise TypeError("observation_time_utc must be a datetime.")
        if (
            self.observation_time_utc.tzinfo is None
            or self.observation_time_utc.utcoffset() != timedelta(0)
        ):
            raise ValueError("observation_time_utc must be timezone-aware UTC.")
        if type(self.triggered) is not bool:
            raise TypeError("triggered must be a boolean.")
        if not isinstance(self.quality, str):
            raise TypeError("quality must be a string.")
        if not self.quality.strip():
            raise ValueError("quality must not be empty.")
        if self.score is not None:
            if isinstance(self.score, bool) or not isinstance(self.score, Real):
                raise TypeError("score must be a real number or None.")
            if not math.isfinite(float(self.score)):
                raise ValueError("score must be finite when provided.")
            object.__setattr__(self, "score", float(self.score))
        if not isinstance(self.detector_metadata, Mapping):
            raise TypeError("detector_metadata must be a mapping.")
        object.__setattr__(
            self,
            "detector_metadata",
            _freeze_metadata(
                self.detector_metadata,
                location="detector_metadata",
            ),
        )


class ActivityObservationDetector(Protocol):
    """Minimal detector interface for the neutral activity-observation path."""

    def on_observation(self, observation: ActivityObservation) -> DetectionResult:
        ...


class NumericObservationDetector(Protocol):
    """Value-only interface shared by PH, ADWIN and KSWIN; no legacy lifecycle."""

    def on_observation(self, observation: NumericObservation) -> DetectionResult:
        ...


@dataclass(frozen=True, slots=True)
class ActivityDetectionRouteConfig:
    """Explicitly associate one activity signal with one detector strategy."""

    signal_id: str
    detector_id: str

    def __post_init__(self) -> None:
        for field_name in ("signal_id", "detector_id"):
            _require_component_id(field_name, getattr(self, field_name))


@overload
def dispatch_activity_observation(
    *, route: ActivityDetectionRouteConfig, observation: ActivityObservation,
    detectors: Mapping[str, ActivityObservationDetector | NumericObservationDetector],
) -> DetectionResult: ...


@overload
def dispatch_activity_observation(
    *, route: ActivityDetectionRouteConfig, observation: SignalObservation,
    detectors: Mapping[str, ActivityObservationDetector | NumericObservationDetector],
) -> DetectionResult | None: ...


def dispatch_activity_observation(
    *,
    route: ActivityDetectionRouteConfig,
    observation: NumericObservation,
    detectors: Mapping[str, ActivityObservationDetector | NumericObservationDetector],
) -> DetectionResult | None:
    """Route numeric values; explicit absences leave detector state untouched.

    Skipping absent values is only a technical dispatch policy, not a decision
    about resetting state after long gaps. The historical function name remains
    compatible; routing does not branch on signal kind or detector identity.
    """

    if not isinstance(route, ActivityDetectionRouteConfig):
        raise TypeError("route must be an ActivityDetectionRouteConfig.")
    signal_id = observation_signal_id(observation)
    if signal_id != route.signal_id:
        raise ValueError(
            "Observation signal_id does not match the configured activity route."
        )
    try:
        detector = detectors[route.detector_id]
    except KeyError as exc:
        raise ValueError(
            f"No detector instance is available for {route.detector_id!r}."
        ) from exc
    on_observation = getattr(detector, "on_observation", None)
    if not callable(on_observation):
        raise TypeError(
            f"Detector {route.detector_id!r} has no on_observation operation."
        )
    if isinstance(observation, SignalObservation) and observation.status != VALUE_AVAILABLE:
        return None
    require_numeric_observation(observation)
    result = on_observation(observation)
    if not isinstance(result, DetectionResult):
        raise TypeError(
            f"Detector {route.detector_id!r} must return DetectionResult."
        )
    if result.detector_id != route.detector_id:
        raise ValueError("DetectionResult detector_id does not match the route.")
    if result.signal_id != route.signal_id:
        raise ValueError("DetectionResult signal_id does not match the route.")
    if result.observation_time_utc != observation.observation_time_utc:
        raise ValueError(
            "DetectionResult observation time does not match the observation."
        )
    return result


__all__ = [
    "ActivityObservationDetector",
    "NumericObservation",
    "NumericObservationDetector",
    "ActivityDetectionRouteConfig",
    "DetectionResult",
    "dispatch_activity_observation",
    "observation_signal_id",
    "require_numeric_observation",
]
