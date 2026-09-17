from __future__ import annotations

import math
from dataclasses import dataclass, replace
from numbers import Real
from typing import Any, Callable, Protocol

import pandas as pd
from river import drift

from youtube_pipeline.activity_signals import (
    ActivityObservation,
    CLOSED_INTERVAL,
    EventWindowCommentCountSignal,
    _align_timestamp_to_cadence as _align_timestamp_to_slide,
    event_window_comment_count_definition,
)
from youtube_pipeline.activity_detection import (
    ActivityObservationDetector,
    DetectionResult,
    NumericObservation,
    NumericObservationDetector,
    require_numeric_observation,
)


class TriggerDetector(ActivityObservationDetector, Protocol):
    """Legacy XIAO lifecycle in addition to neutral observation detection."""

    completed_triggers: list[dict[str, Any]]

    def on_event(self, event: dict[str, Any]) -> None:
        ...

    def finalize(self, final_ts: pd.Timestamp | None = None) -> None:
        ...


def _timedelta_steps(window_td: pd.Timedelta, slide_td: pd.Timedelta, label: str) -> int:
    window_ns = int(window_td.value)
    slide_ns = int(slide_td.value)
    if window_ns <= 0:
        raise ValueError(f"{label} must be > 0.")
    if slide_ns <= 0:
        raise ValueError("slide_interval must be > 0.")
    if window_ns % slide_ns != 0:
        raise ValueError(f"{label} must be an exact multiple of slide_interval.")
    return window_ns // slide_ns


@dataclass(frozen=True, slots=True)
class PageHinkleyConfig:
    """Technical River defaults, not event-detection validated parameters."""

    min_instances: int = 30
    delta: float = 0.005
    threshold: float = 50.0
    alpha: float = 0.9999
    mode: str = "both"

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "PageHinkleyConfig":
        config_payload = payload.get("page_hinkley", payload)
        if not isinstance(config_payload, dict):
            raise ValueError("Page-Hinkley config must be an object.")
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(config_payload) - allowed)
        if unknown:
            raise ValueError(f"Unknown Page-Hinkley config fields: {unknown}")
        return cls(**config_payload)

    def __post_init__(self) -> None:
        if isinstance(self.min_instances, bool) or not isinstance(
            self.min_instances, int
        ):
            raise TypeError("min_instances must be an integer.")
        if self.min_instances < 1:
            raise ValueError("min_instances must be >= 1.")

        for field_name in ("delta", "threshold", "alpha"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{field_name} must be a real number.")
            numeric_value = float(value)
            if not math.isfinite(numeric_value):
                raise ValueError(f"{field_name} must be finite.")
            object.__setattr__(self, field_name, numeric_value)

        if self.delta < 0:
            raise ValueError("delta must be >= 0.")
        if self.threshold <= 0:
            raise ValueError("threshold must be > 0.")
        if not 0 < self.alpha <= 1:
            raise ValueError("alpha must be in the interval (0, 1].")
        if not isinstance(self.mode, str):
            raise TypeError("mode must be a string.")
        if self.mode not in {"up", "down", "both"}:
            raise ValueError("mode must be one of: up, down, both.")


class PageHinkleyAdapter:
    """Adapt River Page-Hinkley to the neutral observation/result contract.

    Availability through this adapter does not make Page-Hinkley a validated,
    recommended, or definitive detector for event detection.
    """

    detector_id = "page_hinkley"

    def __init__(self, *, config: PageHinkleyConfig | None = None) -> None:
        if config is not None and not isinstance(config, PageHinkleyConfig):
            raise TypeError("config must be PageHinkleyConfig or None.")
        self.config = config or PageHinkleyConfig()
        self.observations_processed = 0
        self._detector = drift.PageHinkley(
            min_instances=self.config.min_instances,
            delta=self.config.delta,
            threshold=self.config.threshold,
            alpha=self.config.alpha,
            mode=self.config.mode,
        )

    def on_observation(self, observation: NumericObservation) -> DetectionResult:
        """Update Page-Hinkley with the observation's numeric value only."""

        signal_id = require_numeric_observation(observation)

        # River resets Page-Hinkley at the start of the update after a drift.
        # Mirror that public lifecycle in the adapter-owned warm-up counter.
        if self._detector.drift_detected:
            self.observations_processed = 0

        self._detector.update(observation.value)
        self.observations_processed += 1

        return DetectionResult(
            detector_id=self.detector_id,
            signal_id=signal_id,
            observation_time_utc=observation.observation_time_utc,
            triggered=bool(self._detector.drift_detected),
            quality=observation.quality,
            score=None,
            detector_metadata={
                "observations_processed": self.observations_processed,
                "min_instances": self.config.min_instances,
                "mode": self.config.mode,
                "warmup_complete": (
                    self.observations_processed >= self.config.min_instances
                ),
            },
        )


@dataclass(frozen=True, slots=True)
class ADWINConfig:
    """Technical River defaults, not event-detection validated parameters."""

    delta: float = 0.002
    clock: int = 32
    max_buckets: int = 5
    min_window_length: int = 5
    grace_period: int = 10

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "ADWINConfig":
        config_payload = payload.get("adwin", payload)
        if not isinstance(config_payload, dict):
            raise ValueError("ADWIN config must be an object.")
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(config_payload) - allowed)
        if unknown:
            raise ValueError(f"Unknown ADWIN config fields: {unknown}")
        return cls(**config_payload)

    def __post_init__(self) -> None:
        if isinstance(self.delta, bool) or not isinstance(self.delta, Real):
            raise TypeError("delta must be a real number.")
        numeric_delta = float(self.delta)
        if not math.isfinite(numeric_delta):
            raise ValueError("delta must be finite.")
        if not 0 < numeric_delta < 1:
            raise ValueError("delta must be in the interval (0, 1).")
        object.__setattr__(self, "delta", numeric_delta)

        for field_name in (
            "clock",
            "max_buckets",
            "min_window_length",
            "grace_period",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field_name} must be an integer.")

        if self.clock < 1:
            raise ValueError("clock must be >= 1.")
        if self.max_buckets < 1:
            raise ValueError("max_buckets must be >= 1.")
        if self.min_window_length < 1:
            raise ValueError("min_window_length must be >= 1.")
        if self.grace_period < 0:
            raise ValueError("grace_period must be >= 0.")


class ADWINAdapter:
    """Adapt River ADWIN to the neutral observation/result contract.

    The adaptive window is detector state measured in observations. It is not
    the physical-time window declared by ``ActivitySignalDefinition``.
    Availability through this adapter does not make ADWIN calibrated, validated,
    or selected as the definitive event detector.
    """

    detector_id = "adwin"

    def __init__(self, *, config: ADWINConfig | None = None) -> None:
        if config is not None and not isinstance(config, ADWINConfig):
            raise TypeError("config must be ADWINConfig or None.")
        self.config = config or ADWINConfig()
        self.observations_processed = 0
        self._detector = drift.ADWIN(
            delta=self.config.delta,
            clock=self.config.clock,
            max_buckets=self.config.max_buckets,
            min_window_length=self.config.min_window_length,
            grace_period=self.config.grace_period,
        )

    def on_observation(self, observation: NumericObservation) -> DetectionResult:
        """Update ADWIN with the observation's numeric value only."""

        signal_id = require_numeric_observation(observation)

        # River resets ADWIN at the start of the update after a drift. Mirror
        # that public lifecycle in the adapter-owned segment counter.
        if self._detector.drift_detected:
            self.observations_processed = 0

        self._detector.update(observation.value)
        self.observations_processed += 1

        return DetectionResult(
            detector_id=self.detector_id,
            signal_id=signal_id,
            observation_time_utc=observation.observation_time_utc,
            triggered=bool(self._detector.drift_detected),
            quality=observation.quality,
            score=None,
            detector_metadata={
                "observations_processed": self.observations_processed,
                "width": self._detector.width,
                "estimation": self._detector.estimation,
                "variance": self._detector.variance,
            },
        )


@dataclass(frozen=True, slots=True)
class KSWINConfig:
    """Public River parameters; library defaults are technical, not calibrated.

    ``window_size`` counts scalar observations, not seconds. The sampling code
    requires window_size >= 2 * stat_size. River's constructor misses this check.
    Initial ``window`` data is input/state, intentionally not configuration.
    With seed=None, random sampling is not reproducible across new instances.
    """

    alpha: float = 0.005
    window_size: int = 100
    stat_size: int = 30
    seed: int | None = None

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "KSWINConfig":
        if not isinstance(payload, dict):
            raise TypeError("KSWIN config must be an object.")
        unknown = sorted(set(payload) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(f"Unknown KSWIN config fields: {unknown}")
        return cls(**payload)

    def __post_init__(self) -> None:
        if isinstance(self.alpha, bool) or not isinstance(self.alpha, Real):
            raise TypeError("alpha must be a real number.")
        alpha = float(self.alpha)
        if not math.isfinite(alpha):
            raise ValueError("alpha must be finite.")
        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be in [0, 1], as accepted by River.")
        object.__setattr__(self, "alpha", alpha)
        for name in ("window_size", "stat_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer.")
            if value < 1:
                raise ValueError(f"{name} must be positive for a nonempty KS test.")
        if self.window_size < 2 * self.stat_size:
            raise ValueError("window_size must be >= 2 * stat_size for sampling without replacement.")
        if self.seed is not None and (isinstance(self.seed, bool) or not isinstance(self.seed, int)):
            raise TypeError("seed must be an integer or None.")


class KSWINAdapter:
    """Detect distribution changes of scalar values, never opinion vectors.

    For MEC(t), KSWIN compares recent scalar MEC values with a random historical
    sample. It does not compare OpinionDistribution(t) with its predecessor.
    This is technical availability, not an official polarization-detector choice.
    """

    detector_id = "kswin"

    def __init__(self, *, config: KSWINConfig | None = None) -> None:
        if config is not None and not isinstance(config, KSWINConfig):
            raise TypeError("config must be KSWINConfig or None.")
        self.config = config if config is not None else KSWINConfig()
        self.observations_processed = 0  # lifetime successful updates, gaps excluded
        self._detector = drift.KSWIN(
            alpha=self.config.alpha, window_size=self.config.window_size,
            stat_size=self.config.stat_size, seed=self.config.seed,
        )

    def on_observation(self, observation: NumericObservation) -> DetectionResult:
        signal_id = require_numeric_observation(observation)
        # River trims to recent stat_size on drift, then clears state and reseeds
        # at the next update. No manual reset, cooldown or XIAO lifecycle here.
        self._detector.update(observation.value)
        self.observations_processed += 1
        return DetectionResult(
            detector_id=self.detector_id, signal_id=signal_id,
            observation_time_utc=observation.observation_time_utc,
            triggered=bool(self._detector.drift_detected), quality=observation.quality,
            score=None,
            detector_metadata={
                "observations_processed": self.observations_processed,
                "segment_observations": self._detector.n,
                "window_length": len(self._detector.window),
                "window_size": self.config.window_size,
                "stat_size": self.config.stat_size,
                "alpha": self.config.alpha, "seed": self.config.seed,
            },
        )


@dataclass(frozen=True, slots=True)
class XiaoEMAConfig:
    """Authoritative methodological parameters for the XIAO EMA detector."""

    ts_col: str = "event_time_utc"
    text_col: str = "text"
    window_size: str = "120s"
    slide_interval: str = "30s"
    slow_window: str = "10min"
    sensitivity_threshold: float = 1.5
    v_min: int = 46
    cooldown: str = "3min"
    warmup_windows: int = 10
    extreme_volume_multiplier: float = 2.0

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "XiaoEMAConfig":
        config_payload = payload.get("xiao_ema", payload)
        if not isinstance(config_payload, dict):
            raise ValueError("XIAO EMA config must be an object.")
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(config_payload) - allowed)
        if unknown:
            raise ValueError(f"Unknown XIAO EMA config fields: {unknown}")
        return cls(**config_payload)

    def __post_init__(self) -> None:
        for field_name in (
            "ts_col",
            "text_col",
            "window_size",
            "slide_interval",
            "slow_window",
            "cooldown",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str):
                raise TypeError(f"{field_name} must be a string.")
            if not value.strip():
                raise ValueError(f"{field_name} must not be empty.")
        if self.sensitivity_threshold <= 0:
            raise ValueError("sensitivity_threshold must be > 0.")
        if self.v_min <= 0:
            raise ValueError("v_min must be > 0.")
        if self.warmup_windows < 0:
            raise ValueError("warmup_windows must be >= 0.")
        if self.extreme_volume_multiplier <= 0:
            raise ValueError("extreme_volume_multiplier must be > 0.")

        slide_td = pd.to_timedelta(self.slide_interval)
        _timedelta_steps(
            pd.to_timedelta(self.window_size),
            slide_td,
            label="window_size",
        )
        _timedelta_steps(
            pd.to_timedelta(self.slow_window),
            slide_td,
            label="slow_window",
        )
        _timedelta_steps(
            pd.to_timedelta(self.cooldown),
            slide_td,
            label="cooldown",
        )


class XiaoEMATriggerDetector:
    """
    EMA trigger inspired by Xiao et al. (2025), adapted to the audited stream cadence.

    Implementation choices required by the prototype request:
      - Sliding window volume: 2 minutes
      - Slide interval: 30 seconds
      - Fast EMA span: 2 minutes / 30 seconds = 4 steps
      - Slow EMA span: 10 minutes / 30 seconds = 20 steps
      - Warm-up: first 10 windows update EMAs silently unless volume > v_extreme
      - Post warm-up trigger: EMA_fast > EMA_slow * sensitivity_threshold and volume > v_min
      - Cooldown / lock: 3 minutes
    """

    def __init__(
        self,
        *,
        config: XiaoEMAConfig | None = None,
        ts_col: str | None = None,
        text_col: str | None = None,
        window_size: str | None = None,
        slide_interval: str | None = None,
        slow_window: str | None = None,
        sensitivity_threshold: float | None = None,
        v_min: int | None = None,
        cooldown: str | None = None,
        log_fn: Callable[[str], None] | None = None,
    ) -> None:
        if config is not None and not isinstance(config, XiaoEMAConfig):
            raise TypeError("config must be XiaoEMAConfig or None.")
        overrides = {
            key: value
            for key, value in {
                "ts_col": ts_col,
                "text_col": text_col,
                "window_size": window_size,
                "slide_interval": slide_interval,
                "slow_window": slow_window,
                "sensitivity_threshold": sensitivity_threshold,
                "v_min": v_min,
                "cooldown": cooldown,
            }.items()
            if value is not None
        }
        if config is not None and overrides:
            raise ValueError(
                "Pass either XiaoEMAConfig or legacy keyword overrides, not both."
            )
        effective = config or XiaoEMAConfig()
        if overrides:
            effective = replace(effective, **overrides)

        self.config = effective
        self.ts_col = effective.ts_col
        self.text_col = effective.text_col
        self.window_td = pd.to_timedelta(effective.window_size)
        self.slide_td = pd.to_timedelta(effective.slide_interval)
        self.slow_window_td = pd.to_timedelta(effective.slow_window)
        self.cooldown_td = pd.to_timedelta(effective.cooldown)

        self.fast_span_steps = _timedelta_steps(
            self.window_td, self.slide_td, label="window_size"
        )
        self.slow_span_steps = _timedelta_steps(
            self.slow_window_td, self.slide_td, label="slow_window"
        )
        _timedelta_steps(self.cooldown_td, self.slide_td, label="cooldown")

        # Academic EMA form: alpha = 2 / (n + 1)
        self.fast_alpha = 2.0 / float(self.fast_span_steps + 1)
        self.slow_alpha = 2.0 / float(self.slow_span_steps + 1)
        self.sensitivity_threshold = float(effective.sensitivity_threshold)
        self.log_fn = log_fn or print
        self.windows_processed = 0
        self.warmup_windows = int(effective.warmup_windows)
        self.v_min = int(effective.v_min)
        self.v_extreme = int(self.v_min * effective.extreme_volume_multiplier)

        self.signal_definition = event_window_comment_count_definition(
            window=effective.window_size,
            cadence=effective.slide_interval,
            time_basis=effective.ts_col,
        )
        self.signal = EventWindowCommentCountSignal(
            definition=self.signal_definition,
            timestamp_column=effective.ts_col,
        )
        self._ema_fast: float | None = None
        self._ema_slow: float | None = None
        self._lock_until: pd.Timestamp | None = None
        self._active_trigger: dict[str, Any] | None = None
        self.completed_triggers: list[dict[str, Any]] = []

    def on_event(self, event: dict[str, Any]) -> None:
        self.signal.on_event(
            event,
            on_observation=self.on_observation,
            on_event_accepted=self._collect_if_locked,
        )

    def on_observation(self, observation: ActivityObservation) -> DetectionResult:
        """Update XIAO from an explicit activity observation."""

        if not isinstance(observation, ActivityObservation):
            raise TypeError("observation must be an ActivityObservation.")
        if observation.signal.time_basis != self.ts_col:
            raise ValueError("Observation time_basis does not match XIAO ts_col.")
        if pd.to_timedelta(observation.signal.window) != self.window_td:
            raise ValueError("Observation window does not match XIAO window_size.")
        if pd.to_timedelta(observation.signal.cadence) != self.slide_td:
            raise ValueError("Observation cadence does not match XIAO slide_interval.")
        if observation.signal.interval_policy != CLOSED_INTERVAL:
            raise ValueError("XIAO reference observations must use closed intervals.")
        numeric_value = float(observation.value)
        if not numeric_value.is_integer() or numeric_value < 0:
            raise ValueError("XIAO reference observations require non-negative counts.")

        tick_ts = pd.Timestamp(observation.observation_time_utc)
        volume = int(numeric_value)
        active_trigger_before = self._active_trigger
        self._advance_observation(tick_ts=tick_ts, volume=volume)
        triggered = (
            self._active_trigger is not None
            and self._active_trigger is not active_trigger_before
        )
        strength = (
            None
            if self._ema_fast is None
            or self._ema_slow is None
            or self._ema_slow <= 0
            else float(self._ema_fast / self._ema_slow)
        )
        return DetectionResult(
            detector_id=DEFAULT_DETECTOR,
            signal_id=observation.signal.signal_id,
            observation_time_utc=observation.observation_time_utc,
            triggered=triggered,
            quality=observation.quality,
            score=strength,
            detector_metadata={
                "volume": volume,
                "ema_fast": self._ema_fast,
                "ema_slow": self._ema_slow,
                "strength": strength,
                "windows_processed": self.windows_processed,
                "warmup_complete": self.windows_processed > self.warmup_windows,
                "cooldown_active": self._lock_until is not None,
            },
        )

    def finalize(self, final_ts: pd.Timestamp | None = None) -> None:
        if self._active_trigger is None:
            return
        self._close_trigger(final_ts)

    def _advance_observation(self, *, tick_ts: pd.Timestamp, volume: int) -> None:
        if self._lock_until is not None and tick_ts >= self._lock_until:
            self._close_trigger(tick_ts)

        self.windows_processed += 1

        if self._ema_fast is None:
            self._ema_fast = float(volume)
        else:
            self._ema_fast = (
                self.fast_alpha * float(volume)
                + (1.0 - self.fast_alpha) * self._ema_fast
            )

        if self._ema_slow is None:
            self._ema_slow = float(volume)
        else:
            self._ema_slow = (
                self.slow_alpha * float(volume)
                + (1.0 - self.slow_alpha) * self._ema_slow
            )

        if self._lock_until is not None:
            return

        if self._ema_fast is None or self._ema_slow is None or self._ema_slow <= 0:
            return

        strength = self._ema_fast / self._ema_slow

        if self.windows_processed <= self.warmup_windows:
            if volume > self.v_extreme:
                self._open_trigger(tick_ts=tick_ts, volume=volume, strength=strength)
            return

        if volume <= self.v_min:
            return

        if self._ema_fast <= self._ema_slow * self.sensitivity_threshold:
            return

        self._open_trigger(tick_ts=tick_ts, volume=volume, strength=strength)

    def _open_trigger(
        self,
        *,
        tick_ts: pd.Timestamp,
        volume: int,
        strength: float,
    ) -> None:
        self._lock_until = tick_ts + self.cooldown_td
        self._active_trigger = {
            "trigger_time": tick_ts,
            "cooldown_until": self._lock_until,
            "volume": volume,
            "strength": strength,
            "comments": [],
        }
        self.log_fn(
            "[TRIGGER] Evento detectado a las "
            f"{tick_ts.isoformat()}. "
            f"Volumen actual: {volume}. "
            f"Fuerza del pico (EMA_R / EMA_L): {strength:.2f}."
        )

    def _collect_if_locked(self, event: dict[str, Any]) -> None:
        if self._active_trigger is None or self._lock_until is None:
            return
        event_ts = pd.to_datetime(event.get(self.ts_col), utc=True, errors="coerce")
        if pd.isna(event_ts):
            return
        if pd.Timestamp(event_ts) >= self._lock_until:
            return
        self._active_trigger["comments"].append(
            {
                self.ts_col: pd.Timestamp(event_ts),
                self.text_col: event.get(self.text_col),
            }
        )

    def _close_trigger(self, closed_at: pd.Timestamp | None) -> None:
        if self._active_trigger is None:
            self._lock_until = None
            return
        trigger = dict(self._active_trigger)
        trigger["closed_at"] = closed_at
        self.completed_triggers.append(trigger)
        self._active_trigger = None
        self._lock_until = None


DEFAULT_DETECTOR = "xiao_ema"
PAGE_HINKLEY_DETECTOR = "page_hinkley"
ADWIN_DETECTOR = "adwin"
KSWIN_DETECTOR = "kswin"
DETECTOR_PARAM_ALIASES = {
    "trigger_window_size": "window_size",
    "trigger_slide_interval": "slide_interval",
    "trigger_slow_window": "slow_window",
    "trigger_threshold": "sensitivity_threshold",
    "trigger_min_volume": "v_min",
    "trigger_cooldown": "cooldown",
}
DETECTOR_REGISTRY: dict[str, Callable[..., ActivityObservationDetector | NumericObservationDetector]] = {
    DEFAULT_DETECTOR: XiaoEMATriggerDetector,
    PAGE_HINKLEY_DETECTOR: PageHinkleyAdapter,
    ADWIN_DETECTOR: ADWINAdapter,
    KSWIN_DETECTOR: KSWINAdapter,
}


def normalize_detector_params(params: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in params.items():
        normalized[DETECTOR_PARAM_ALIASES.get(key, key)] = value
    return normalized


def get_detector_names() -> tuple[str, ...]:
    return tuple(sorted(DETECTOR_REGISTRY))


def create_detector(
    name: str | None = DEFAULT_DETECTOR,
    **params: Any,
) -> ActivityObservationDetector | NumericObservationDetector:
    detector_name = name or DEFAULT_DETECTOR
    try:
        factory = DETECTOR_REGISTRY[detector_name]
    except KeyError as exc:
        available = ", ".join(get_detector_names())
        raise ValueError(
            f"Unknown detector '{detector_name}'. Available detectors: {available}."
        ) from exc
    return factory(**normalize_detector_params(params))


__all__ = [
    "ADWINAdapter",
    "ADWINConfig",
    "ADWIN_DETECTOR",
    "ActivityObservationDetector",
    "NumericObservationDetector",
    "KSWINAdapter",
    "KSWINConfig",
    "KSWIN_DETECTOR",
    "DEFAULT_DETECTOR",
    "DETECTOR_PARAM_ALIASES",
    "DETECTOR_REGISTRY",
    "PAGE_HINKLEY_DETECTOR",
    "PageHinkleyAdapter",
    "PageHinkleyConfig",
    "TriggerDetector",
    "XiaoEMAConfig",
    "XiaoEMATriggerDetector",
    "_align_timestamp_to_slide",
    "_timedelta_steps",
    "create_detector",
    "get_detector_names",
    "normalize_detector_params",
]
