"""Neutral triggered-result projection for future temporal association."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from numbers import Integral
from pathlib import Path
from typing import Any

from youtube_pipeline.activity_detection import DetectionResult


ACTIVITY = "ACTIVITY"
POLARIZATION = "POLARIZATION"
SIGNAL_FAMILIES = frozenset({ACTIVITY, POLARIZATION})
PRESERVE_STATE_V1 = "preserve_state_v1"
GAP_POLICY_METHODOLOGICAL_STATUS = "NOT_CALIBRATED"


def _require_text(name: str, value: object) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    if not value.strip():
        raise ValueError(f"{name} must not be empty.")
    return value


def _require_utc(name: str, value: object) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"{name} must be a datetime.")
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError(f"{name} must be timezone-aware UTC.")
    return value


@dataclass(frozen=True, slots=True)
class DetectionGapPolicyConfig:
    """Configured continuity policy for numeric detector streams."""

    policy_id: str = PRESERVE_STATE_V1

    def __post_init__(self) -> None:
        if self.policy_id != PRESERVE_STATE_V1:
            raise ValueError(f"policy_id must be {PRESERVE_STATE_V1!r}.")

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> DetectionGapPolicyConfig:
        if not isinstance(payload, dict):
            raise TypeError("Detection gap policy configuration must be an object.")
        unknown = set(payload) - {"policy_id"}
        if unknown:
            raise ValueError(f"Unknown detection gap policy fields: {sorted(unknown)}")
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class DetectionAlert:
    """One triggered statistical decision prepared for later association.

    The alert carries no association, candidate taxonomy, evidence, or event claim.
    Its identity is a deterministic projection of the originating result envelope.
    """

    alert_id: str
    detection_result_id: str
    detection_result_ref: str
    run_id: str
    config_hash: str
    signal_id: str
    signal_family: str
    detector_id: str
    alert_time_utc: datetime
    data_cutoff_utc: datetime
    triggered: bool
    quality: str
    support_count: int | None

    def __post_init__(self) -> None:
        for name in (
            "alert_id",
            "detection_result_id",
            "detection_result_ref",
            "run_id",
            "config_hash",
            "signal_id",
            "detector_id",
            "quality",
        ):
            _require_text(name, getattr(self, name))
        if self.signal_family not in SIGNAL_FAMILIES:
            raise ValueError(
                f"signal_family must be one of {sorted(SIGNAL_FAMILIES)!r}."
            )
        _require_utc("alert_time_utc", self.alert_time_utc)
        _require_utc("data_cutoff_utc", self.data_cutoff_utc)
        if self.alert_time_utc >= self.data_cutoff_utc:
            raise ValueError("alert_time_utc must be strictly before data_cutoff_utc.")
        if self.triggered is not True:
            raise ValueError("DetectionAlert requires triggered=True.")
        if self.support_count is not None:
            if isinstance(self.support_count, bool) or not isinstance(
                self.support_count, Integral
            ):
                raise TypeError("support_count must be an integer or None.")
            if self.support_count < 0:
                raise ValueError("support_count must be >= 0.")
            object.__setattr__(self, "support_count", int(self.support_count))

    def to_mapping(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["alert_time_utc"] = self.alert_time_utc.isoformat()
        payload["data_cutoff_utc"] = self.data_cutoff_utc.isoformat()
        return payload

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> DetectionAlert:
        if not isinstance(payload, dict):
            raise TypeError("DetectionAlert payload must be an object.")
        expected = {
            "alert_id",
            "detection_result_id",
            "detection_result_ref",
            "run_id",
            "config_hash",
            "signal_id",
            "signal_family",
            "detector_id",
            "alert_time_utc",
            "data_cutoff_utc",
            "triggered",
            "quality",
            "support_count",
        }
        unknown = set(payload) - expected
        missing = expected - set(payload)
        if unknown:
            raise ValueError(f"Unknown DetectionAlert fields: {sorted(unknown)}")
        if missing:
            raise ValueError(f"Missing DetectionAlert fields: {sorted(missing)}")

        def parse_time(name: str) -> datetime:
            value = payload[name]
            if not isinstance(value, str):
                raise TypeError(f"{name} must be an ISO-8601 string.")
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError as exc:
                raise ValueError(f"{name} must be a valid ISO-8601 datetime.") from exc

        normalized = dict(payload)
        normalized["alert_time_utc"] = parse_time("alert_time_utc")
        normalized["data_cutoff_utc"] = parse_time("data_cutoff_utc")
        return cls(**normalized)


def detection_result_id(*, run_id: str, result: DetectionResult) -> str:
    _require_text("run_id", run_id)
    if not isinstance(result, DetectionResult):
        raise TypeError("result must be a DetectionResult.")
    identity = json.dumps(
        {
            "run_id": run_id,
            "signal_id": result.signal_id,
            "detector_id": result.detector_id,
            "observation_time_utc": result.observation_time_utc.isoformat(),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return "detection_result_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]


def detection_alert_from_result(
    result: DetectionResult,
    *,
    run_id: str,
    config_hash: str,
    signal_family: str,
    support_count: int | None,
    data_cutoff_utc: datetime,
    detection_artifact_ref: str,
) -> DetectionAlert | None:
    """Project exactly one triggered result; non-triggers remain result-only."""

    if not isinstance(result, DetectionResult):
        raise TypeError("result must be a DetectionResult.")
    if not result.triggered:
        return None
    result_id = detection_result_id(run_id=run_id, result=result)
    artifact = _require_text("detection_artifact_ref", detection_artifact_ref)
    alert_identity = json.dumps(
        {"detection_result_id": result_id, "signal_family": signal_family},
        sort_keys=True,
        separators=(",", ":"),
    )
    return DetectionAlert(
        alert_id=(
            "alert_"
            + hashlib.sha256(alert_identity.encode("utf-8")).hexdigest()[:24]
        ),
        detection_result_id=result_id,
        detection_result_ref=f"{artifact}#{result_id}",
        run_id=run_id,
        config_hash=config_hash,
        signal_id=result.signal_id,
        signal_family=signal_family,
        detector_id=result.detector_id,
        alert_time_utc=result.observation_time_utc,
        data_cutoff_utc=data_cutoff_utc,
        triggered=True,
        quality=result.quality,
        support_count=support_count,
    )


def load_detection_alerts(path: str | Path) -> tuple[DetectionAlert, ...]:
    source = Path(path)
    alerts: list[DetectionAlert] = []
    for line_number, line in enumerate(
        source.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid DetectionAlert JSON on line {line_number}."
            ) from exc
        alerts.append(DetectionAlert.from_mapping(payload))
    return tuple(alerts)


__all__ = [
    "ACTIVITY",
    "GAP_POLICY_METHODOLOGICAL_STATUS",
    "POLARIZATION",
    "PRESERVE_STATE_V1",
    "SIGNAL_FAMILIES",
    "DetectionAlert",
    "DetectionGapPolicyConfig",
    "detection_alert_from_result",
    "detection_result_id",
    "load_detection_alerts",
]
