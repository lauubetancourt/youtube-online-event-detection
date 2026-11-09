"""Configured detector execution over persisted temporal polarization values."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from youtube_pipeline.activity_detection import (
    ActivityDetectionRouteConfig,
    DetectionResult,
    dispatch_numeric_observation,
)
from youtube_pipeline.configuration.models import RunConfig
from youtube_pipeline.detectors import create_detector
from youtube_pipeline.detection_alerts import (
    GAP_POLICY_METHODOLOGICAL_STATUS,
    POLARIZATION,
    PRESERVE_STATE_V1,
    DetectionGapPolicyConfig,
    detection_alert_from_result,
    detection_result_id,
)
from youtube_pipeline.signal_observations import SignalObservation, to_json_mapping


DETECTION_ARTIFACT = "polarization_detection_results.jsonl"
DETECTION_MANIFEST = "polarization_detection_manifest.json"
DETECTION_ALERT_ARTIFACT = "detection_alerts.jsonl"
TECHNICAL_GAP_POLICY = "skip_absent_value_preserve_detector_state_v1"
METHODOLOGICAL_GAP_STATUS = GAP_POLICY_METHODOLOGICAL_STATUS
SUPPORTED_POLARIZATION_DETECTORS = frozenset({"page_hinkley", "adwin", "kswin"})


@dataclass(frozen=True, slots=True)
class PolarizationDetectionRuntimeArtifacts:
    result_path: Path
    manifest_path: Path
    alert_path: Path
    routes_count: int
    observations_available: int
    observations_skipped_no_value: int
    results_produced: int
    trigger_count: int
    alerts_produced: int
    future_leak_count: int


def _parse_time(value: object, name: str) -> datetime:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be an ISO-8601 string.")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware.")
    return parsed


def _observation_from_row(row: Mapping[str, Any]) -> tuple[SignalObservation, str]:
    if not isinstance(row, Mapping):
        raise TypeError("Polarization observation rows must be mappings.")
    metadata = row.get("metadata")
    if not isinstance(metadata, Mapping):
        raise TypeError("Polarization observation metadata must be a mapping.")
    observation = SignalObservation(
        signal_id=row["signal_id"],
        observation_time_utc=_parse_time(
            row["observation_time_utc"], "observation_time_utc"
        ),
        value=row.get("value"),
        status=row["status"],
        support_count=row["support_count"],
        quality=row["quality"],
        metadata=metadata,
    )
    distribution_time = row.get(
        "distribution_observation_time_utc",
        row["observation_time_utc"],
    )
    if not isinstance(distribution_time, str):
        raise TypeError("distribution_observation_time_utc must be a string.")
    return observation, distribution_time


def load_polarization_signal_observations(
    path: str | Path,
) -> tuple[tuple[SignalObservation, str], ...]:
    source = Path(path)
    rows = [
        json.loads(line)
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return tuple(_observation_from_row(row) for row in rows)


def configured_polarization_routes(
    config: RunConfig,
) -> tuple[ActivityDetectionRouteConfig, ...]:
    if config.detection is None or config.detection.routes is None:
        return ()
    routes = tuple(
        route
        for route in config.detection.routes
        if route.signal_id.startswith("polarization_")
    )
    for route in routes:
        if route.detector_id not in SUPPORTED_POLARIZATION_DETECTORS:
            supported = ", ".join(sorted(SUPPORTED_POLARIZATION_DETECTORS))
            raise ValueError(
                f"Polarization route {route.signal_id!r} uses unsupported detector "
                f"{route.detector_id!r}; expected one of: {supported}."
            )
    return routes


def _detector_parameters(config: RunConfig, detector_id: str) -> Any:
    assert config.detection is not None
    detector_config = getattr(config.detection, detector_id, None)
    if detector_config is None:
        raise ValueError(
            f"Polarization route requires detection.{detector_id} configuration."
        )
    return detector_config


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


def run_polarization_detection_runtime(
    config: RunConfig,
    observations: Iterable[tuple[SignalObservation, str]],
    *,
    output_dir: str | Path,
    config_hash: str,
    input_policy_manifest: Mapping[str, Any] | None = None,
    signal_artifact_path: str | Path | None = None,
) -> PolarizationDetectionRuntimeArtifacts:
    """Route available scalar values; absences never update detector state."""

    routes = configured_polarization_routes(config)
    if not routes:
        raise ValueError("No polarization detection routes are configured.")
    if config.signals is None or config.signals.polarization is None:
        raise ValueError("Polarization detection requires signals.polarization.")
    if config.detection is None:
        raise ValueError("Polarization detection requires detection configuration.")
    effective_gap_policy = (
        config.detection.gap_policy or DetectionGapPolicyConfig()
    )
    if effective_gap_policy.policy_id != PRESERVE_STATE_V1:
        raise ValueError("Unsupported polarization detection gap policy.")

    route_by_signal = {route.signal_id: route for route in routes}
    route_detectors = {
        route.signal_id: create_detector(
            route.detector_id,
            config=_detector_parameters(config, route.detector_id),
        )
        for route in routes
    }
    received = list(observations)
    last_time: datetime | None = None
    identities: set[tuple[str, datetime]] = set()
    for observation, _ in received:
        if not isinstance(observation, SignalObservation):
            raise TypeError("observations must contain SignalObservation values.")
        if last_time is not None and observation.observation_time_utc < last_time:
            raise ValueError("Polarization observations must not regress in time.")
        last_time = observation.observation_time_utc
        identity = (observation.signal_id, observation.observation_time_utc)
        if identity in identities:
            raise ValueError("Duplicate signal_id/observation_time_utc observation.")
        identities.add(identity)
    received.sort(key=lambda item: (
        item[0].observation_time_utc,
        item[0].signal_id,
    ))

    rows: list[dict[str, Any]] = []
    alert_rows: list[dict[str, Any]] = []
    available = Counter()
    skipped = Counter()
    triggers = Counter()
    future_leak_count = 0
    for observation, distribution_time in received:
        route = route_by_signal.get(observation.signal_id)
        if route is None:
            continue
        cutoff_raw = observation.metadata.get("data_cutoff_utc")
        if cutoff_raw is None:
            raise ValueError("Polarization observation requires data_cutoff_utc metadata.")
        cutoff = _parse_time(cutoff_raw, "metadata.data_cutoff_utc")
        if observation.observation_time_utc >= cutoff:
            future_leak_count += 1
            raise ValueError("Polarization observation violates its causal cutoff.")
        detector = route_detectors[route.signal_id]
        result = dispatch_numeric_observation(
            route=route,
            observation=observation,
            detectors={route.detector_id: detector},
        )
        if result is None:
            skipped[route.signal_id] += 1
            continue
        available[route.signal_id] += 1
        triggers[route.signal_id] += int(result.triggered)
        rows.append(_result_row(
            run_id=config.identity.run_id,
            route=route,
            observation=observation,
            distribution_time=distribution_time,
            result=result,
        ))
        alert = detection_alert_from_result(
            result,
            run_id=config.identity.run_id,
            config_hash=config_hash,
            signal_family=POLARIZATION,
            support_count=observation.support_count,
            data_cutoff_utc=cutoff,
            detection_artifact_ref=DETECTION_ARTIFACT,
        )
        if alert is not None:
            alert_rows.append(alert.to_mapping())

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    result_path = root / DETECTION_ARTIFACT
    manifest_path = root / DETECTION_MANIFEST
    alert_path = root / DETECTION_ALERT_ARTIFACT
    if result_path.exists() or manifest_path.exists() or alert_path.exists():
        raise FileExistsError(
            "Polarization detection outputs already exist; choose a fresh output."
        )
    result_hash = _write_jsonl_exclusive(result_path, rows)
    alert_hash = _write_jsonl_exclusive(alert_path, alert_rows)
    temporal = config.signals.polarization
    route_rows = []
    for route in routes:
        detector_config = _detector_parameters(config, route.detector_id)
        route_rows.append({
            "signal_id": route.signal_id,
            "measure_id": route.signal_id.removeprefix("polarization_"),
            "detector_id": route.detector_id,
            "detector_parameters": to_json_mapping(detector_config),
            "detector_instances": 1,
            "observations_available": available[route.signal_id],
            "observations_skipped_no_value": skipped[route.signal_id],
            "detection_results": available[route.signal_id],
            "trigger_count": triggers[route.signal_id],
        })
    manifest = {
        "artifact_version": "polarization_detection_runtime_v1",
        "status": "executed",
        "methodological_status": "TECHNICALLY_AVAILABLE_NOT_CALIBRATED",
        "run_id": config.identity.run_id,
        "config_hash": config_hash,
        "input_policy": (
            to_json_mapping(input_policy_manifest)
            if input_policy_manifest is not None else None
        ),
        "signal_artifact": (
            str(signal_artifact_path) if signal_artifact_path is not None else None
        ),
        "temporal_signal": {
            "window": temporal.window,
            "cadence": temporal.cadence,
            "interval_policy": temporal.interval_policy,
            "distribution_policy_id": temporal.distribution_policy_id,
        },
        "routes": route_rows,
        "counts": {
            "observations_received": len(received),
            "observations_available": sum(available.values()),
            "observations_skipped_no_value": sum(skipped.values()),
            "detection_results": len(rows),
            "trigger_count": sum(triggers.values()),
            "detection_alerts": len(alert_rows),
        },
        "gap_policy": {
            "policy_id": effective_gap_policy.policy_id,
            "configuration_status": (
                "explicit"
                if config.detection.gap_policy is not None
                else "legacy_implicit_equivalent"
            ),
            "technical_behavior": TECHNICAL_GAP_POLICY,
            "detector_state_on_absence": "unchanged",
            "reset": False,
            "imputation": False,
            "interpolation": False,
            "methodological_status": METHODOLOGICAL_GAP_STATUS,
        },
        "causality": {
            "rule": "observation_time_utc < data_cutoff_utc",
            "future_leak_count": future_leak_count,
            "ordering": "observation_time_utc_then_signal_id",
        },
        "artifacts": {
            "detection_results": DETECTION_ARTIFACT,
            "detection_results_sha256": result_hash,
            "detection_alerts": DETECTION_ALERT_ARTIFACT,
            "detection_alerts_sha256": alert_hash,
        },
        "alerts": {
            "signal_family": POLARIZATION,
            "production_rule": "triggered_detection_results_only",
            "count": len(alert_rows),
            "future_association_status": "not_executed",
        },
        "downstream": {
            "event_candidates_created": False,
            "activity_polarization_association": False,
            "rag_executed": False,
        },
    }
    with manifest_path.open("x", encoding="utf-8") as target:
        target.write(json.dumps(
            manifest, ensure_ascii=False, indent=2, allow_nan=False
        ) + "\n")
    return PolarizationDetectionRuntimeArtifacts(
        result_path=result_path,
        manifest_path=manifest_path,
        alert_path=alert_path,
        routes_count=len(routes),
        observations_available=sum(available.values()),
        observations_skipped_no_value=sum(skipped.values()),
        results_produced=len(rows),
        trigger_count=sum(triggers.values()),
        alerts_produced=len(alert_rows),
        future_leak_count=future_leak_count,
    )


def _result_row(
    *,
    run_id: str,
    route: ActivityDetectionRouteConfig,
    observation: SignalObservation,
    distribution_time: str,
    result: DetectionResult,
) -> dict[str, Any]:
    return {
        "detection_result_id": detection_result_id(run_id=run_id, result=result),
        "run_id": run_id,
        "signal_id": result.signal_id,
        "measure_id": observation.metadata["measure_id"],
        "observation_time_utc": result.observation_time_utc.isoformat(),
        "distribution_observation_time_utc": distribution_time,
        "value": observation.value,
        "status": observation.status,
        "support_count": observation.support_count,
        "quality": result.quality,
        "detector_id": result.detector_id,
        "triggered": result.triggered,
        "score": result.score,
        "detector_metadata": to_json_mapping(result.detector_metadata),
        "distribution_policy_id": observation.metadata["distribution_policy_id"],
        "classification_run_id": observation.metadata["classification_run_id"],
        "data_cutoff_utc": observation.metadata["data_cutoff_utc"],
        "route": {
            "signal_id": route.signal_id,
            "detector_id": route.detector_id,
        },
    }


__all__ = [
    "DETECTION_ARTIFACT",
    "DETECTION_ALERT_ARTIFACT",
    "DETECTION_MANIFEST",
    "METHODOLOGICAL_GAP_STATUS",
    "SUPPORTED_POLARIZATION_DETECTORS",
    "TECHNICAL_GAP_POLICY",
    "PolarizationDetectionRuntimeArtifacts",
    "configured_polarization_routes",
    "load_polarization_signal_observations",
    "run_polarization_detection_runtime",
]
