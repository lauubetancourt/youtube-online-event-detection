"""Causal timestamp association for neutral activity and polarization alerts."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from youtube_pipeline.detection_alerts import ACTIVITY, POLARIZATION, DetectionAlert


BOUNDED_TEMPORAL_CLUSTERS_V1 = "bounded_temporal_clusters_v1"
ASSOCIATION_TYPES = frozenset({ACTIVITY, POLARIZATION, "ACTIVITY_POLARIZATION"})
FINALIZED = "FINALIZED"
TOLERANCE_ELAPSED = "TOLERANCE_ELAPSED"
END_OF_STREAM = "END_OF_STREAM"
ASSOCIATION_METHODOLOGICAL_STATUS = "EXPERIMENTAL_NOT_CALIBRATED"
ASSOCIATION_ARTIFACT = "detection_associations.jsonl"
ASSOCIATION_MANIFEST = "detection_association_manifest.json"


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


def _duration(value: str) -> timedelta:
    _require_text("tolerance", value)
    try:
        parsed = pd.to_timedelta(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("tolerance must be a valid duration.") from exc
    if pd.isna(parsed) or parsed < pd.Timedelta(0):
        raise ValueError("tolerance must be a non-negative duration.")
    return parsed.to_pytimedelta()


@dataclass(frozen=True, slots=True)
class DetectionAssociationConfig:
    """Experimental parameters for bounded, causal alert grouping."""

    tolerance: str
    parameter_status: str
    policy_id: str = BOUNDED_TEMPORAL_CLUSTERS_V1

    def __post_init__(self) -> None:
        if self.policy_id != BOUNDED_TEMPORAL_CLUSTERS_V1:
            raise ValueError(
                f"policy_id must be {BOUNDED_TEMPORAL_CLUSTERS_V1!r}."
            )
        _duration(self.tolerance)
        if self.parameter_status not in {
            "TEST_ONLY",
            "DEVELOPMENT_NOT_CALIBRATED",
        }:
            raise ValueError(
                "parameter_status must be TEST_ONLY or "
                "DEVELOPMENT_NOT_CALIBRATED."
            )

    @property
    def tolerance_delta(self) -> timedelta:
        return _duration(self.tolerance)

    @classmethod
    def from_mapping(
        cls, payload: dict[str, Any]
    ) -> DetectionAssociationConfig:
        if not isinstance(payload, dict):
            raise TypeError("Association configuration must be an object.")
        unknown = set(payload) - {
            "policy_id",
            "tolerance",
            "parameter_status",
        }
        if unknown:
            raise ValueError(f"Unknown association fields: {sorted(unknown)}")
        missing = {"tolerance", "parameter_status"} - set(payload)
        if missing:
            raise ValueError(
                f"Missing association fields: {sorted(missing)}"
            )
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class DetectionAssociation:
    """Finalized, non-probabilistic grouping of one or more source alerts."""

    association_id: str
    run_id: str
    policy_id: str
    tolerance: str
    parameter_status: str
    association_type: str
    activity_alert_ids: tuple[str, ...]
    polarization_alert_ids: tuple[str, ...]
    start_time_utc: datetime
    end_time_utc: datetime
    reference_time_utc: datetime
    finalized_at_cutoff_utc: datetime
    status: str
    finalization_reason: str

    def __post_init__(self) -> None:
        for name in (
            "association_id",
            "run_id",
            "policy_id",
            "tolerance",
            "parameter_status",
        ):
            _require_text(name, getattr(self, name))
        _duration(self.tolerance)
        if self.parameter_status not in {
            "TEST_ONLY",
            "DEVELOPMENT_NOT_CALIBRATED",
        }:
            raise ValueError(
                "parameter_status must be TEST_ONLY or "
                "DEVELOPMENT_NOT_CALIBRATED."
            )
        if self.association_type not in ASSOCIATION_TYPES:
            raise ValueError(
                f"association_type must be one of {sorted(ASSOCIATION_TYPES)!r}."
            )
        activity_ids = tuple(self.activity_alert_ids)
        polarization_ids = tuple(self.polarization_alert_ids)
        for name, values in (
            ("activity_alert_ids", activity_ids),
            ("polarization_alert_ids", polarization_ids),
        ):
            if len(set(values)) != len(values):
                raise ValueError(f"{name} must not contain duplicates.")
            for value in values:
                _require_text(f"{name} item", value)
        if set(activity_ids).intersection(polarization_ids):
            raise ValueError("An alert cannot belong to both signal families.")
        if not activity_ids and not polarization_ids:
            raise ValueError("A detection association requires at least one alert.")
        expected_type = (
            "ACTIVITY_POLARIZATION"
            if activity_ids and polarization_ids
            else ACTIVITY if activity_ids else POLARIZATION
        )
        if self.association_type != expected_type:
            raise ValueError("association_type does not match its alert families.")
        object.__setattr__(self, "activity_alert_ids", tuple(sorted(activity_ids)))
        object.__setattr__(
            self, "polarization_alert_ids", tuple(sorted(polarization_ids))
        )
        for name in (
            "start_time_utc",
            "end_time_utc",
            "reference_time_utc",
            "finalized_at_cutoff_utc",
        ):
            _require_utc(name, getattr(self, name))
        if self.start_time_utc > self.end_time_utc:
            raise ValueError("start_time_utc must not follow end_time_utc.")
        tolerance_delta = _duration(self.tolerance)
        if self.end_time_utc - self.start_time_utc > tolerance_delta:
            raise ValueError("Association alert span must not exceed tolerance.")
        if self.reference_time_utc != self.end_time_utc:
            raise ValueError("reference_time_utc must equal the latest alert time.")
        if self.finalized_at_cutoff_utc <= self.end_time_utc:
            raise ValueError("Association finalization must be causally after its alerts.")
        if self.status != FINALIZED:
            raise ValueError(f"status must be {FINALIZED!r}.")
        if self.finalization_reason not in {TOLERANCE_ELAPSED, END_OF_STREAM}:
            raise ValueError("Unsupported finalization_reason.")
        if (
            self.finalization_reason == TOLERANCE_ELAPSED
            and self.finalized_at_cutoff_utc
            <= self.start_time_utc + tolerance_delta
        ):
            raise ValueError(
                "Tolerance-elapsed finalization requires cutoff after its boundary."
            )

    @property
    def alert_count(self) -> int:
        return len(self.activity_alert_ids) + len(self.polarization_alert_ids)

    def to_mapping(self) -> dict[str, Any]:
        payload = asdict(self)
        for name in (
            "start_time_utc",
            "end_time_utc",
            "reference_time_utc",
            "finalized_at_cutoff_utc",
        ):
            payload[name] = getattr(self, name).isoformat()
        payload["activity_alert_ids"] = list(self.activity_alert_ids)
        payload["polarization_alert_ids"] = list(self.polarization_alert_ids)
        return payload

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> DetectionAssociation:
        if not isinstance(payload, Mapping):
            raise TypeError("DetectionAssociation payload must be a mapping.")
        expected = {
            "association_id",
            "run_id",
            "policy_id",
            "tolerance",
            "parameter_status",
            "association_type",
            "activity_alert_ids",
            "polarization_alert_ids",
            "start_time_utc",
            "end_time_utc",
            "reference_time_utc",
            "finalized_at_cutoff_utc",
            "status",
            "finalization_reason",
        }
        unknown = set(payload) - expected
        missing = expected - set(payload)
        if unknown:
            raise ValueError(
                f"Unknown DetectionAssociation fields: {sorted(unknown)}"
            )
        if missing:
            raise ValueError(
                f"Missing DetectionAssociation fields: {sorted(missing)}"
            )
        parsed = dict(payload)
        for name in (
            "start_time_utc",
            "end_time_utc",
            "reference_time_utc",
            "finalized_at_cutoff_utc",
        ):
            value = parsed[name]
            if not isinstance(value, str):
                raise TypeError(f"{name} must be an ISO-8601 string.")
            try:
                parsed[name] = datetime.fromisoformat(value)
            except ValueError as exc:
                raise ValueError(f"{name} must be a valid ISO-8601 datetime.") from exc
        for name in ("activity_alert_ids", "polarization_alert_ids"):
            value = parsed[name]
            if not isinstance(value, list):
                raise TypeError(f"{name} must be a list.")
            parsed[name] = tuple(value)
        return cls(**parsed)


def _alert_key(alert: DetectionAlert) -> tuple[Any, ...]:
    return (
        alert.alert_time_utc,
        alert.signal_family,
        alert.signal_id,
        alert.detector_id,
        alert.alert_id,
    )


def _association_id(
    *,
    run_id: str,
    config: DetectionAssociationConfig,
    alerts: Iterable[DetectionAlert],
) -> str:
    identity = json.dumps(
        {
            "run_id": run_id,
            "policy_id": config.policy_id,
            "tolerance": config.tolerance,
            "boundary": "closed_le",
            "alert_ids": sorted(alert.alert_id for alert in alerts),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return "association_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]


class TemporalDetectionAssociator:
    """Incremental causal buffer using earliest-alert bounded clusters.

    All alerts within ``anchor_time <= time <= anchor_time + tolerance`` join the
    earliest pending cluster. The anchor never extends transitively. A cluster is
    emitted only once the current cutoff is strictly after its closed boundary, or
    when end-of-stream explicitly flushes pending alerts.
    """

    def __init__(self, *, run_id: str, config: DetectionAssociationConfig) -> None:
        self.run_id = _require_text("run_id", run_id)
        if not isinstance(config, DetectionAssociationConfig):
            raise TypeError("config must be DetectionAssociationConfig.")
        self.config = config
        self._pending: dict[str, DetectionAlert] = {}
        self._seen_alert_ids: set[str] = set()
        self._last_cutoff_utc: datetime | None = None
        self._finalized_through_utc: datetime | None = None
        self._closed = False

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    @property
    def seen_alert_count(self) -> int:
        return len(self._seen_alert_ids)

    def advance(
        self,
        alerts: Iterable[DetectionAlert],
        *,
        data_cutoff_utc: datetime,
        end_of_stream: bool = False,
    ) -> tuple[DetectionAssociation, ...]:
        if self._closed:
            raise RuntimeError("Association stream is already closed.")
        cutoff = _require_utc("data_cutoff_utc", data_cutoff_utc)
        if self._last_cutoff_utc is not None and cutoff < self._last_cutoff_utc:
            raise ValueError("Association data cutoffs must not regress.")

        incoming = list(alerts)
        for alert in incoming:
            if not isinstance(alert, DetectionAlert):
                raise TypeError("alerts must contain DetectionAlert values.")
            if alert.run_id != self.run_id:
                raise ValueError("DetectionAlert run_id does not match the association run.")
            if alert.alert_id in self._seen_alert_ids:
                raise ValueError(f"Duplicate DetectionAlert {alert.alert_id!r}.")
            if alert.data_cutoff_utc > cutoff:
                raise ValueError("DetectionAlert is not causally available at this cutoff.")
            if (
                self._finalized_through_utc is not None
                and alert.alert_time_utc <= self._finalized_through_utc
            ):
                raise ValueError(
                    "Late DetectionAlert would revise a finalized association; "
                    "late-arrival handling is not implemented."
                )

        for alert in incoming:
            self._seen_alert_ids.add(alert.alert_id)
            self._pending[alert.alert_id] = alert

        finalized: list[DetectionAssociation] = []
        tolerance = self.config.tolerance_delta
        while self._pending:
            ordered = sorted(self._pending.values(), key=_alert_key)
            anchor = ordered[0]
            boundary = anchor.alert_time_utc + tolerance
            if not end_of_stream and cutoff <= boundary:
                break
            cluster = [alert for alert in ordered if alert.alert_time_utc <= boundary]
            reason = END_OF_STREAM if end_of_stream and cutoff <= boundary else TOLERANCE_ELAPSED
            association = self._finalize_cluster(
                cluster,
                cutoff=cutoff,
                finalization_reason=reason,
            )
            finalized.append(association)
            for alert in cluster:
                del self._pending[alert.alert_id]
            self._finalized_through_utc = boundary

        self._last_cutoff_utc = cutoff
        if end_of_stream:
            self._closed = True
        return tuple(finalized)

    def _finalize_cluster(
        self,
        alerts: list[DetectionAlert],
        *,
        cutoff: datetime,
        finalization_reason: str,
    ) -> DetectionAssociation:
        ordered = sorted(alerts, key=_alert_key)
        activity_ids = tuple(
            alert.alert_id for alert in ordered if alert.signal_family == ACTIVITY
        )
        polarization_ids = tuple(
            alert.alert_id for alert in ordered if alert.signal_family == POLARIZATION
        )
        association_type = (
            "ACTIVITY_POLARIZATION"
            if activity_ids and polarization_ids
            else ACTIVITY if activity_ids else POLARIZATION
        )
        start = ordered[0].alert_time_utc
        end = ordered[-1].alert_time_utc
        return DetectionAssociation(
            association_id=_association_id(
                run_id=self.run_id,
                config=self.config,
                alerts=ordered,
            ),
            run_id=self.run_id,
            policy_id=self.config.policy_id,
            tolerance=self.config.tolerance,
            parameter_status=self.config.parameter_status,
            association_type=association_type,
            activity_alert_ids=activity_ids,
            polarization_alert_ids=polarization_ids,
            start_time_utc=start,
            end_time_utc=end,
            reference_time_utc=end,
            finalized_at_cutoff_utc=cutoff,
            status=FINALIZED,
            finalization_reason=finalization_reason,
        )


@dataclass(frozen=True, slots=True)
class DetectionAssociationArtifacts:
    association_path: Path
    manifest_path: Path
    alerts_input: int
    associations_produced: int
    activity_only: int
    polarization_only: int
    joint: int


def _write_jsonl_exclusive(path: Path, rows: Iterable[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    with path.open("x", encoding="utf-8") as target:
        for row in rows:
            encoded = json.dumps(
                row, ensure_ascii=False, sort_keys=True, allow_nan=False
            ) + "\n"
            target.write(encoded)
            digest.update(encoded.encode("utf-8"))
    return digest.hexdigest()


def run_temporal_detection_association(
    *,
    config: DetectionAssociationConfig,
    alerts: Iterable[DetectionAlert],
    run_id: str,
    config_hash: str,
    data_cutoff_utc: datetime,
    output_dir: str | Path,
    input_policy_id: str | None = None,
    input_alert_artifacts: Mapping[str, str] | None = None,
) -> DetectionAssociationArtifacts:
    """Finalize one causal replay/batch stream and persist neutral associations."""

    _require_text("config_hash", config_hash)
    if input_policy_id is not None:
        _require_text("input_policy_id", input_policy_id)
    if input_alert_artifacts is not None:
        if not isinstance(input_alert_artifacts, Mapping):
            raise TypeError("input_alert_artifacts must be a mapping or None.")
        for name, value in input_alert_artifacts.items():
            _require_text("input alert artifact name", name)
            _require_text("input alert artifact reference", value)

    alert_rows = list(alerts)
    associator = TemporalDetectionAssociator(run_id=run_id, config=config)
    associations = associator.advance(
        alert_rows,
        data_cutoff_utc=data_cutoff_utc,
        end_of_stream=True,
    )
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    association_path = root / ASSOCIATION_ARTIFACT
    manifest_path = root / ASSOCIATION_MANIFEST
    if association_path.exists() or manifest_path.exists():
        raise FileExistsError("Association outputs already exist; choose a fresh output.")
    association_hash = _write_jsonl_exclusive(
        association_path,
        (association.to_mapping() for association in associations),
    )
    counts = {
        ACTIVITY: sum(item.association_type == ACTIVITY for item in associations),
        POLARIZATION: sum(
            item.association_type == POLARIZATION for item in associations
        ),
        "ACTIVITY_POLARIZATION": sum(
            item.association_type == "ACTIVITY_POLARIZATION"
            for item in associations
        ),
    }
    source_config_hashes = sorted({alert.config_hash for alert in alert_rows})
    manifest = {
        "artifact_version": "temporal_detection_association_v1",
        "status": "executed",
        "methodological_status": ASSOCIATION_METHODOLOGICAL_STATUS,
        "run_id": run_id,
        "config_hash": config_hash,
        "association_policy": {
            "policy_id": config.policy_id,
            "tolerance": config.tolerance,
            "parameter_status": config.parameter_status,
            "boundary": "abs_time_difference_lte_tolerance",
            "cluster_rule": "earliest_alert_anchor_bounded_span",
            "finalization": "cutoff_strictly_after_anchor_plus_tolerance_or_end_of_stream",
            "transitive_chaining": False,
        },
        "input": {
            "input_policy_id": input_policy_id,
            "alert_artifacts": dict(input_alert_artifacts or {}),
            "source_config_hashes": source_config_hashes,
            "shared_execution_identity": "run_id",
            "config_hash_equality_required": False,
        },
        "counts": {
            "alerts_input": len(alert_rows),
            "activity_alerts": sum(
                alert.signal_family == ACTIVITY for alert in alert_rows
            ),
            "polarization_alerts": sum(
                alert.signal_family == POLARIZATION for alert in alert_rows
            ),
            "alerts_associated": sum(item.alert_count for item in associations),
            "associations": len(associations),
            "activity_only": counts[ACTIVITY],
            "polarization_only": counts[POLARIZATION],
            "joint": counts["ACTIVITY_POLARIZATION"],
        },
        "causality": {
            "data_cutoff_utc": data_cutoff_utc.isoformat(),
            "end_of_stream": True,
            "future_lookahead": False,
            "late_arrival_policy": "not_implemented_A9_deferred",
        },
        "artifacts": {
            "detection_associations": ASSOCIATION_ARTIFACT,
            "detection_associations_sha256": association_hash,
        },
        "downstream": {
            "event_candidates_created": False,
            "evidence_retrieved": False,
            "rag_executed": False,
        },
    }
    with manifest_path.open("x", encoding="utf-8") as target:
        target.write(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return DetectionAssociationArtifacts(
        association_path=association_path,
        manifest_path=manifest_path,
        alerts_input=len(alert_rows),
        associations_produced=len(associations),
        activity_only=counts[ACTIVITY],
        polarization_only=counts[POLARIZATION],
        joint=counts["ACTIVITY_POLARIZATION"],
    )


def load_detection_associations(path: str | Path) -> tuple[DetectionAssociation, ...]:
    """Load finalized associations from their neutral JSONL artifact."""

    source = Path(path)
    associations: list[DetectionAssociation] = []
    for line_number, line in enumerate(
        source.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid association JSON on line {line_number}."
            ) from exc
        associations.append(DetectionAssociation.from_mapping(payload))
    return tuple(associations)


__all__ = [
    "ACTIVITY",
    "ASSOCIATION_ARTIFACT",
    "ASSOCIATION_MANIFEST",
    "ASSOCIATION_METHODOLOGICAL_STATUS",
    "BOUNDED_TEMPORAL_CLUSTERS_V1",
    "DetectionAssociation",
    "DetectionAssociationArtifacts",
    "DetectionAssociationConfig",
    "END_OF_STREAM",
    "FINALIZED",
    "POLARIZATION",
    "TOLERANCE_ELAPSED",
    "TemporalDetectionAssociator",
    "load_detection_associations",
    "run_temporal_detection_association",
]
