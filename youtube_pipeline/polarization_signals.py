"""Causal, synthetic-ready polarization ticks, independent of detector/runtime state."""
from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, fields
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd

from youtube_pipeline.comment_stance import CLASSIFIED, ERROR, UNCERTAIN, CommentStance
from youtube_pipeline.polarization_measures import (
    DISTRIBUTION_READY, LIKERT5_POSITIONS,
    NO_CLASSIFIABLE_OPINIONS as EMPTY_DISTRIBUTION,
    OpinionDistribution, PolarizationMeasurement, PolarizationMeasure,
    create_polarization_measure, get_polarization_measure_names,
)
from youtube_pipeline.signal_observations import (
    NO_CLASSIFIABLE_OPINIONS, NO_INPUT, NO_REPRESENTABLE_AUTHORS, VALUE_AVAILABLE,
    SignalObservation, require_text, require_utc, to_json_mapping,
)

AUTHOR_LATEST_POLICY = "likert5_author_latest_classified_v1"
UTC_ORIGIN = datetime(1970, 1, 1, tzinfo=timezone.utc)


def _duration(name: str, value: str) -> timedelta:
    require_text(name, value)
    parsed = pd.to_timedelta(value)
    if pd.isna(parsed) or parsed.value <= 0 or parsed.value % 1000:
        raise ValueError(f"{name} must be positive and have microsecond precision.")
    return timedelta(microseconds=int(parsed.value) // 1000)


@dataclass(frozen=True, slots=True)
class PolarizationTemporalSignalConfig:
    """Window/cadence/support are experimental, not calibrated defaults.

    Execution coverage is [start_time_utc, end_time_utc); signal windows are closed.
    A replay stance is assumed available with its comment (not an online LLM model).
    """

    proposition_id: str
    classification_run_id: str
    window: str
    cadence: str
    start_time_utc: datetime
    end_time_utc: datetime
    enabled_measures: tuple[str, ...]
    interval_policy: str = "closed"
    time_basis: str = "event_time_utc"
    timezone: str = "UTC"
    tick_origin_utc: datetime = UTC_ORIGIN
    distribution_policy_id: str = AUTHOR_LATEST_POLICY
    min_support_threshold: int | None = None

    def __post_init__(self) -> None:
        for name in ("proposition_id", "classification_run_id"):
            require_text(name, getattr(self, name))
        _duration("window", self.window)
        _duration("cadence", self.cadence)
        for name in ("start_time_utc", "end_time_utc", "tick_origin_utc"):
            require_utc(name, getattr(self, name))
        if self.start_time_utc >= self.end_time_utc:
            raise ValueError("start_time_utc must precede end_time_utc.")
        if (self.interval_policy, self.time_basis, self.timezone) != (
            "closed", "event_time_utc", "UTC"
        ):
            raise ValueError("V1 requires closed event-time windows in UTC.")
        if self.distribution_policy_id != AUTHOR_LATEST_POLICY:
            raise ValueError("Unsupported V1 distribution policy.")
        if not isinstance(self.enabled_measures, (list, tuple)):
            raise TypeError("enabled_measures must be a list or tuple.")
        measures = tuple(self.enabled_measures)
        for measure in measures:
            require_text("enabled measure", measure)
        if not measures or len(set(measures)) != len(measures):
            raise ValueError("enabled_measures must be nonempty and unique.")
        if not set(measures) <= set(get_polarization_measure_names()):
            raise ValueError("Unknown enabled polarization measure.")
        object.__setattr__(self, "enabled_measures", measures)
        if self.min_support_threshold is not None:
            if isinstance(self.min_support_threshold, bool) or not isinstance(self.min_support_threshold, int):
                raise TypeError("min_support_threshold must be an integer or None.")
            if self.min_support_threshold < 1:
                raise ValueError("min_support_threshold must be positive.")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> PolarizationTemporalSignalConfig:
        if not isinstance(payload, Mapping):
            raise TypeError("Polarization temporal configuration must be a mapping.")
        unknown = set(payload) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown polarization temporal fields: {sorted(unknown)}.")
        values = dict(payload)
        for name in ("start_time_utc", "end_time_utc", "tick_origin_utc"):
            if isinstance(values.get(name), str):
                values[name] = datetime.fromisoformat(values[name].replace("Z", "+00:00"))
        return cls(**values)


@dataclass(frozen=True, slots=True)
class PolarizationTemporalTick:
    observation_time_utc: datetime
    window_start_utc: datetime
    window_end_utc: datetime
    data_cutoff_utc: datetime
    status: str
    distribution: OpinionDistribution
    selected_comment_ids: tuple[str, ...]
    measurements: tuple[PolarizationMeasurement, ...]
    observations: tuple[SignalObservation, ...]

    def to_mapping(self) -> dict[str, Any]:
        return to_json_mapping(self)


class PolarizationTemporalSignal:
    """Accept complete causal prefixes, then emit every closed tick before cutoff.

    ``advance`` validates the entire batch before changing state. Records must be
    globally ordered by (event_time_utc, comment_id); no sorting or late arrivals.
    A subsequent batch may start at the preceding cutoff, never before it.
    Selecting authors is deliberately recomputed once per tick for correctness.
    State is per execution; cross-restart durability and detector gaps are deferred.
    """

    def __init__(self, config: PolarizationTemporalSignalConfig,
                 measures: Mapping[str, PolarizationMeasure]) -> None:
        if not isinstance(config, PolarizationTemporalSignalConfig):
            raise TypeError("config must be PolarizationTemporalSignalConfig.")
        if set(measures) != set(config.enabled_measures):
            raise ValueError("Measures must match enabled_measures exactly.")
        for name, measure in measures.items():
            if measure.measure_id != name or not callable(getattr(measure, "measure", None)):
                raise ValueError("Measure adapter identity/operation mismatch.")
        self.config = config
        self._measures = dict(measures)
        self._window = _duration("window", config.window)
        self._cadence = _duration("cadence", config.cadence)
        remainder = (config.start_time_utc - config.tick_origin_utc) % self._cadence
        self._next_tick = config.start_time_utc + (
            self._cadence - remainder if remainder else timedelta(0)
        )
        self._buffer: list[CommentStance] = []
        self._last_key: tuple[datetime, str] | None = None
        self._cutoff: datetime | None = None
        self._seen: set[tuple[str, str]] = set()

    def advance(self, stances: Iterable[CommentStance] = (), *,
                data_cutoff_utc: datetime) -> tuple[PolarizationTemporalTick, ...]:
        require_utc("data_cutoff_utc", data_cutoff_utc)
        if self._cutoff is not None and data_cutoff_utc < self._cutoff:
            raise ValueError("data_cutoff_utc must not regress.")
        batch = list(stances)
        last_key, seen = self._last_key, set(self._seen)
        for stance in batch:
            if not isinstance(stance, CommentStance):
                raise TypeError("Input records must be CommentStance.")
            key = (stance.event_time_utc, stance.comment_id)
            identity = (stance.proposition_id, stance.comment_id)
            if last_key is not None and key < last_key:
                raise ValueError("CommentStance order must be (event_time_utc, comment_id).")
            if identity in seen:
                raise ValueError("Duplicate comment_id for a proposition.")
            if stance.event_time_utc >= data_cutoff_utc:
                raise ValueError("event_time_utc must be strictly before data_cutoff_utc.")
            if self._cutoff is not None and stance.event_time_utc < self._cutoff:
                raise ValueError("Input precedes a completed causal prefix; late arrivals unsupported.")
            last_key = key
            seen.add(identity)
        buffer = self._buffer + [s for s in batch if s.proposition_id == self.config.proposition_id]
        tick = self._next_tick
        results = []
        while tick < data_cutoff_utc and tick < self.config.end_time_utc:
            left = tick - self._window
            buffer = [s for s in buffer if s.event_time_utc >= left]
            records = [s for s in buffer if s.event_time_utc <= tick]
            results.append(self._evaluate_tick(tick, data_cutoff_utc, records))
            tick += self._cadence
        # Commit only after validation and every measure calculation succeeded.
        self._buffer, self._next_tick = buffer, tick
        self._last_key, self._seen, self._cutoff = last_key, seen, data_cutoff_utc
        return tuple(results)

    def _build_distribution(self, records: list[CommentStance]) -> tuple[OpinionDistribution, str, tuple[str, ...]]:
        classified = [s for s in records if s.classification_status == CLASSIFIED]
        latest: dict[str, CommentStance] = {}
        excluded = Counter(s.classification_status.lower() for s in records
                           if s.classification_status != CLASSIFIED)
        for stance in classified:
            if stance.author_id is None:
                excluded["missing_author"] += 1
            else:
                # Input order already guarantees max(event_time, comment_id).
                latest[stance.author_id] = stance
        excluded["superseded_classified"] = len(classified) - excluded["missing_author"] - len(latest)
        counts = [0] * 5
        for stance in latest.values():
            assert stance.stance_category is not None
            counts[stance.stance_category - 1] += 1
        support = len(latest)
        status = (NO_INPUT if not records else
                  NO_CLASSIFIABLE_OPINIONS if not classified else
                  NO_REPRESENTABLE_AUTHORS if not support else VALUE_AVAILABLE)
        reasons = set()
        if excluded["missing_author"]:
            reasons.add("missing_author")
        if any(s.classification_status == UNCERTAIN for s in records):
            reasons.add("uncertain_input")
        if any(s.classification_status == ERROR for s in records):
            reasons.add("error_input")
        if any(s.quality != "passed" for s in records):
            reasons.add("input_quality")
        threshold = self.config.min_support_threshold
        if support and threshold is not None and support < threshold:
            reasons.add("low_support")
        if status != VALUE_AVAILABLE:
            reasons.add(status.lower())
            quality = status.lower()
        else:
            quality = "low_support" if "low_support" in reasons else "degraded" if reasons else "passed"
        distribution = OpinionDistribution(
            proposition_id=self.config.proposition_id,
            distribution_policy_id=self.config.distribution_policy_id,
            classification_run_id=self.config.classification_run_id,
            positions=LIKERT5_POSITIONS, bin_counts=tuple(counts),
            weights=tuple(c / support for c in counts) if support else None,
            support_count=support, unique_author_count=support,
            input_count=len(records), classified_comment_count=len(classified),
            excluded_counts=dict(excluded), quality=quality,
            quality_reasons=tuple(sorted(reasons)),
            status=DISTRIBUTION_READY if support else EMPTY_DISTRIBUTION,
        )
        return distribution, status, tuple(latest[a].comment_id for a in sorted(latest))

    def _evaluate_tick(self, tick: datetime, cutoff: datetime,
                       records: list[CommentStance]) -> PolarizationTemporalTick:
        distribution, status, selected = self._build_distribution(records)
        measurements, observations = [], []
        for name in self.config.enabled_measures:
            measurement = self._measures[name].measure(distribution) if status == VALUE_AVAILABLE else None
            if measurement is not None:
                measurements.append(measurement)
            metadata = {
                "window_start_utc": (tick - self._window).isoformat(),
                "window_end_utc": tick.isoformat(), "data_cutoff_utc": cutoff.isoformat(),
                "interval_policy": self.config.interval_policy,
                "proposition_id": distribution.proposition_id,
                "classification_run_id": distribution.classification_run_id,
                "distribution_policy_id": distribution.distribution_policy_id,
                "measure_id": name,
                "measure_parameters": dict(measurement.parameters) if measurement else None,
                "input_count": distribution.input_count,
                "classified_comment_count": distribution.classified_comment_count,
                "unique_author_count": distribution.unique_author_count,
                "excluded_counts": dict(distribution.excluded_counts),
                "quality_reasons": distribution.quality_reasons,
                "input_quality_labels": tuple(sorted({s.quality for s in records if s.quality != "passed"})),
            }
            observations.append(SignalObservation(
                signal_id=f"polarization_{name}", observation_time_utc=tick,
                value=measurement.value if measurement else None,
                status=status, support_count=distribution.support_count,
                quality=measurement.quality if measurement else distribution.quality,
                metadata=metadata,
            ))
        return PolarizationTemporalTick(
            tick, tick - self._window, tick, cutoff, status, distribution,
            selected, tuple(measurements), tuple(observations),
        )


def create_polarization_temporal_signal(config: PolarizationTemporalSignalConfig,
                                        measure_configs: Mapping[str, Any]) -> PolarizationTemporalSignal:
    """Compose only explicit resolved parameters; no library defaults duplicated here."""
    if set(measure_configs) != set(config.enabled_measures):
        raise ValueError("Resolved measure configurations must match enabled_measures.")
    if any(value is None for value in measure_configs.values()):
        raise ValueError("Every enabled measure requires resolved configuration.")
    return PolarizationTemporalSignal(config, {
        name: create_polarization_measure(name, config=measure_configs[name])
        for name in config.enabled_measures
    })
