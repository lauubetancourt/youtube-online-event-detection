from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
import math
from numbers import Real
from typing import Any

from youtube_pipeline.activity_detection import DetectionResult
from youtube_pipeline.activity_signals import ActivityObservation


SUPPORTED_CANDIDATE_LIFECYCLE_STATES = frozenset({"point", "open", "closed"})


def _require_non_empty_string(field_name: str, value: object) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string.")
    if not value.strip():
        raise ValueError(f"{field_name} must not be empty.")


def _require_utc_datetime(field_name: str, value: object) -> None:
    if not isinstance(value, datetime):
        raise TypeError(f"{field_name} must be a datetime.")
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError(f"{field_name} must be timezone-aware UTC.")


@dataclass(frozen=True, slots=True)
class EventCandidateLineage:
    """Minimal references needed to trace a candidate without copying RunConfig."""

    run_id: str
    config_hash: str
    dataset_ref: str
    run_manifest_ref: str | None = None

    def __post_init__(self) -> None:
        for field_name in ("run_id", "config_hash", "dataset_ref"):
            _require_non_empty_string(field_name, getattr(self, field_name))
        if self.run_manifest_ref is not None:
            _require_non_empty_string("run_manifest_ref", self.run_manifest_ref)


@dataclass(frozen=True, slots=True)
class EventCandidate:
    """Detector-neutral promotion of one triggered observation.

    Comment/video associations and RAG context deliberately remain outside this
    contract. Historical event IDs can be supplied as ``candidate_id`` by a
    compatibility adapter without changing their formulas.
    """

    candidate_id: str
    observation: ActivityObservation
    detection_result: DetectionResult
    lineage: EventCandidateLineage
    evidence_window_start_utc: datetime
    evidence_window_end_utc: datetime
    lifecycle_state: str | None = None
    closed_at_utc: datetime | None = None

    def __post_init__(self) -> None:
        _require_non_empty_string("candidate_id", self.candidate_id)
        if not isinstance(self.observation, ActivityObservation):
            raise TypeError("observation must be an ActivityObservation.")
        if not isinstance(self.detection_result, DetectionResult):
            raise TypeError("detection_result must be a DetectionResult.")
        if not isinstance(self.lineage, EventCandidateLineage):
            raise TypeError("lineage must be EventCandidateLineage.")
        if not self.detection_result.triggered:
            raise ValueError("Only a triggered DetectionResult can become a candidate.")
        if self.detection_result.signal_id != self.observation.signal.signal_id:
            raise ValueError("DetectionResult signal_id does not match the observation.")
        if (
            self.detection_result.observation_time_utc
            != self.observation.observation_time_utc
        ):
            raise ValueError("DetectionResult time does not match the observation.")
        if self.detection_result.quality != self.observation.quality:
            raise ValueError("DetectionResult must propagate observation quality unchanged.")

        _require_utc_datetime(
            "evidence_window_start_utc", self.evidence_window_start_utc
        )
        _require_utc_datetime("evidence_window_end_utc", self.evidence_window_end_utc)
        if self.evidence_window_start_utc >= self.evidence_window_end_utc:
            raise ValueError("evidence_window_start_utc must be before its end.")
        if self.evidence_window_end_utc > self.observation_time_utc:
            raise ValueError("Candidate evidence must not extend beyond detection time.")

        if (
            self.lifecycle_state is not None
            and self.lifecycle_state not in SUPPORTED_CANDIDATE_LIFECYCLE_STATES
        ):
            raise ValueError(
                "lifecycle_state must be one of point, open, closed, or None."
            )
        if self.closed_at_utc is not None:
            _require_utc_datetime("closed_at_utc", self.closed_at_utc)
            if self.lifecycle_state != "closed":
                raise ValueError("closed_at_utc requires lifecycle_state='closed'.")
            if self.closed_at_utc < self.observation_time_utc:
                raise ValueError("closed_at_utc must not precede detection time.")

    @property
    def detector_id(self) -> str:
        return self.detection_result.detector_id

    @property
    def signal_id(self) -> str:
        return self.detection_result.signal_id

    @property
    def observation_time_utc(self) -> datetime:
        return self.detection_result.observation_time_utc

    @property
    def quality(self) -> str:
        return self.detection_result.quality


def promote_detection_result(
    *,
    candidate_id: str,
    observation: ActivityObservation,
    detection_result: DetectionResult,
    lineage: EventCandidateLineage,
    evidence_window_start_utc: datetime | None = None,
    evidence_window_end_utc: datetime | None = None,
    lifecycle_state: str | None = None,
    closed_at_utc: datetime | None = None,
) -> EventCandidate:
    """Promote a triggered result without inventing evidence or a new ID formula."""

    return EventCandidate(
        candidate_id=candidate_id,
        observation=observation,
        detection_result=detection_result,
        lineage=lineage,
        evidence_window_start_utc=(
            observation.window_start_utc
            if evidence_window_start_utc is None
            else evidence_window_start_utc
        ),
        evidence_window_end_utc=(
            observation.window_end_utc
            if evidence_window_end_utc is None
            else evidence_window_end_utc
        ),
        lifecycle_state=lifecycle_state,
        closed_at_utc=closed_at_utc,
    )


def project_completed_xiao_trigger(
    candidate: EventCandidate,
    *,
    cooldown_until_utc: datetime,
    comments: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Return the legacy completed-trigger shape without changing XIAO itself.

    The comments are supplied by the caller and are not treated as the candidate's
    evidence inventory. The evidence layer remains responsible for traceable
    comment and video associations.
    """

    if not isinstance(candidate, EventCandidate):
        raise TypeError("candidate must be an EventCandidate.")
    if candidate.detector_id != "xiao_ema":
        raise ValueError("Only xiao_ema candidates can use the XIAO projection.")
    if candidate.lifecycle_state != "closed":
        raise ValueError("A completed XIAO trigger requires lifecycle_state='closed'.")
    _require_utc_datetime("cooldown_until_utc", cooldown_until_utc)
    if cooldown_until_utc < candidate.observation_time_utc:
        raise ValueError("cooldown_until_utc must not precede trigger time.")

    metadata = candidate.detection_result.detector_metadata
    volume = metadata.get("volume")
    if isinstance(volume, bool) or not isinstance(volume, Real):
        raise ValueError("XIAO detector_metadata must contain numeric volume.")
    numeric_volume = float(volume)
    if (
        not math.isfinite(numeric_volume)
        or not numeric_volume.is_integer()
        or numeric_volume < 0
    ):
        raise ValueError("XIAO volume must be a finite non-negative integer value.")

    strength = metadata.get("strength")
    if strength is None:
        strength = candidate.detection_result.score
    if isinstance(strength, bool) or not isinstance(strength, Real):
        raise ValueError("XIAO result must contain numeric strength or score.")
    numeric_strength = float(strength)
    if not math.isfinite(numeric_strength):
        raise ValueError("XIAO strength must be finite.")

    comment_records: list[dict[str, Any]] = []
    for comment in comments:
        if not isinstance(comment, Mapping):
            raise TypeError("comments must contain mapping records.")
        comment_records.append(dict(comment))

    return {
        "trigger_time": candidate.observation_time_utc,
        "cooldown_until": cooldown_until_utc,
        "volume": int(numeric_volume),
        "strength": numeric_strength,
        "comments": comment_records,
        "closed_at": candidate.closed_at_utc,
    }


__all__ = [
    "EventCandidate",
    "EventCandidateLineage",
    "SUPPORTED_CANDIDATE_LIFECYCLE_STATES",
    "project_completed_xiao_trigger",
    "promote_detection_result",
]
