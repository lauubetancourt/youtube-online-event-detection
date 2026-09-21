"""Thin causal replay orchestration from Silver records to candidate validation."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from youtube_pipeline.activity_detection import (
    DetectionResult,
    NumericObservationDetector,
    dispatch_numeric_observation,
)
from youtube_pipeline.activity_signals import (
    ActivityObservation,
    create_activity_signal,
)
from youtube_pipeline.association_candidates import (
    EVENT_CANDIDATE_ARTIFACT,
    EventCandidateProjector,
    AssociatedEventCandidate,
)
from youtube_pipeline.candidate_evidence import (
    CandidateEvidenceArtifacts,
    build_candidate_evidence,
    derive_candidate_evidence_scope,
)
from youtube_pipeline.candidate_rag import (
    CANDIDATE_VALIDATION_RESULTS_FILE,
    CandidateRagArtifacts,
    run_candidate_rag_validation,
)
from youtube_pipeline.comment_stance import CommentStance
from youtube_pipeline.configuration.models import RunConfig
from youtube_pipeline.configuration.paths import resolve_run_config
from youtube_pipeline.configuration.serialization import run_config_to_mapping
from youtube_pipeline.detectors import create_detector
from youtube_pipeline.g2_gate import G2GatePolicyConfig
from youtube_pipeline.detection_alerts import (
    ACTIVITY,
    POLARIZATION,
    PRESERVE_STATE_V1,
    DetectionAlert,
    detection_alert_from_result,
    detection_result_id,
)
from youtube_pipeline.detection_association import (
    ASSOCIATION_ARTIFACT,
    DetectionAssociation,
    TemporalDetectionAssociator,
)
from youtube_pipeline.experimental_input import (
    CAUSAL_MINIMAL_POLICY_ID,
    CausalInputRecord,
    prepare_causal_minimal_batch,
    read_input_table,
    source_sha256,
)
from youtube_pipeline.polarization_signal_composition import (
    create_configured_polarization_signal,
)
from youtube_pipeline.polarization_signals import PolarizationTemporalTick
from youtube_pipeline.signal_observations import SignalObservation, to_json_mapping


REPLAY_MANIFEST = "causal_replay_manifest.json"
RESOLVED_CONFIG = "resolved_config.json"
ACTIVITY_OBSERVATIONS = "activity_observations.jsonl"
POLARIZATION_DISTRIBUTIONS = "opinion_distributions.jsonl"
POLARIZATION_OBSERVATIONS = "polarization_signal_observations.jsonl"
STANCE_ARTIFACT = "comment_stances.jsonl"
DETECTION_RESULTS = "detection_results.jsonl"
DETECTION_ALERTS = "detection_alerts.jsonl"
VALIDATION_RESULTS = "candidate_validation_results.jsonl"
INPUT_POLICY_MANIFEST = "causal_input_manifest.json"


DetectorFactory = Callable[[str, Any], NumericObservationDetector]


@dataclass(frozen=True, slots=True)
class CausalReplayArtifacts:
    output_dir: Path
    manifest_path: Path
    candidate_path: Path
    validation_results_path: Path
    comments_released: int
    activity_observations: int
    polarization_observations: int
    detection_results: int
    alerts: int
    associations: int
    candidates: int
    validations: int
    future_leak_count: int


def _json_safe(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    with path.open("x", encoding="utf-8") as target:
        for row in rows:
            encoded = json.dumps(
                _json_safe(row), ensure_ascii=False, sort_keys=True, allow_nan=False
            ) + "\n"
            target.write(encoded)
            digest.update(encoded.encode("utf-8"))
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}.")
    return payload


def _load_stances(path: Path) -> tuple[CommentStance, ...]:
    values: list[CommentStance] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        try:
            values.append(CommentStance(
                comment_id=row["comment_id"],
                author_id=row.get("author_id"),
                event_time_utc=datetime.fromisoformat(
                    row["event_time_utc"].replace("Z", "+00:00")
                ),
                proposition_id=row["proposition_id"],
                stance_category=row.get("stance_category"),
                classification_status=row["classification_status"],
                quality=row.get("quality", "passed"),
            ))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid CommentStance on line {line_number}.") from exc
    ordered = tuple(sorted(values, key=lambda item: (item.event_time_utc, item.comment_id)))
    if tuple(values) != ordered:
        raise ValueError("Precomputed stances require deterministic temporal ordering.")
    if len({item.comment_id for item in values}) != len(values):
        raise ValueError("Precomputed stances contain duplicate comment IDs.")
    return tuple(values)


def _validate_stance_reuse(
    *,
    config: RunConfig,
    records: tuple[CausalInputRecord, ...],
    stances: tuple[CommentStance, ...],
    manifest: Mapping[str, Any],
    source_path: Path,
) -> None:
    assert config.polarization is not None
    assert config.polarization.classifier is not None
    classifier = config.polarization.classifier
    assert config.topic_profile is not None
    expected = {
        "classifier_run_id": config.identity.run_id,
        "model": classifier.model,
        "prompt_id": classifier.prompt_id,
        "prompt_version": classifier.prompt_version,
    }
    for name, value in expected.items():
        if manifest.get(name) != value:
            raise ValueError(f"Stance manifest {name} is incompatible with the replay.")
    if manifest.get("topic_id") != config.topic_profile.topic_id:
        raise ValueError("Stance manifest topic_id is incompatible with the replay.")
    proposition = manifest.get("proposition")
    proposition_id = (
        proposition.get("proposition_id")
        if isinstance(proposition, Mapping)
        else manifest.get("proposition_id")
    )
    expected_proposition = to_json_mapping(classifier.proposition)
    if proposition_id != classifier.proposition.proposition_id:
        raise ValueError("Stance manifest proposition is incompatible with the replay.")
    if proposition != expected_proposition:
        raise ValueError(
            "Stance manifest proposition/context snapshot is incompatible with the replay."
        )
    context_hash = hashlib.sha256(
        classifier.proposition.context.encode("utf-8")
    ).hexdigest()
    if manifest.get("context_hash") != context_hash:
        raise ValueError("Stance manifest context hash is incompatible with the replay.")
    if manifest.get("classification_policy") != {
        "policy_id": "comment_stance_likert5_structured_v1",
        "missing_parent_policy": "SKIPPED",
        "thread_context": "immediate_parent_only",
        "text_authority": "original",
    }:
        raise ValueError("Stance manifest classification policy is incompatible.")
    input_ref = manifest.get("input_ref")
    if input_ref is not None and Path(str(input_ref)).absolute() != source_path.absolute():
        raise ValueError("Stance manifest input_ref differs from the Silver source.")
    if manifest.get("input_policy_id") != CAUSAL_MINIMAL_POLICY_ID:
        raise ValueError("Stance manifest input policy is not causal_minimal_v1.")
    source_hash = source_sha256(source_path)
    if manifest.get("input_source_sha256") != source_hash:
        raise ValueError("Stance manifest source hash differs from the Silver source.")
    record_ids = {item.comment_id for item in records}
    stance_ids = {item.comment_id for item in stances}
    if stance_ids != record_ids:
        missing = sorted(record_ids - stance_ids)
        extra = sorted(stance_ids - record_ids)
        raise ValueError(
            "Precomputed stance IDs must equal the causal replay universe; "
            f"missing={missing[:5]}, extra={extra[:5]}."
        )
    ids_hash = hashlib.sha256(
        "\n".join(sorted(record_ids)).encode("utf-8")
    ).hexdigest()
    if manifest.get("classified_comment_ids_sha256") != ids_hash:
        raise ValueError("Stance manifest canonical comment-ID hash is incompatible.")
    by_id = {item.comment_id: item for item in records}
    for stance in stances:
        source = by_id[stance.comment_id]
        if stance.event_time_utc != source.event_time_utc:
            raise ValueError("Stance event time differs from its canonical comment.")
        if stance.author_id != source.author_id:
            raise ValueError("Stance author differs from its canonical comment.")
        if stance.proposition_id != classifier.proposition.proposition_id:
            raise ValueError("CommentStance proposition differs from configured proposition.")


def _clock_cutoffs(start: datetime, end: datetime, step: timedelta) -> tuple[datetime, ...]:
    cutoffs = [start]
    current = start
    while current < end:
        current = min(current + step, end)
        cutoffs.append(current)
    return tuple(cutoffs)


def _activity_row(observation: ActivityObservation, cutoff: datetime) -> dict[str, Any]:
    return {
        "signal_id": observation.signal.signal_id,
        "observation_time_utc": observation.observation_time_utc.isoformat(),
        "window_start_utc": observation.window_start_utc.isoformat(),
        "window_end_utc": observation.window_end_utc.isoformat(),
        "data_cutoff_utc": cutoff.isoformat(),
        "value": observation.value,
        "support_count": observation.support_count,
        "quality": observation.quality,
        "signal": to_json_mapping(observation.signal),
    }


def _polarization_rows(
    ticks: Iterable[PolarizationTemporalTick],
    stance_by_id: Mapping[str, CommentStance],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    distributions: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    for tick in ticks:
        distributions.append({
            "observation_time_utc": tick.observation_time_utc.isoformat(),
            "window_start_utc": tick.window_start_utc.isoformat(),
            "window_end_utc": tick.window_end_utc.isoformat(),
            "data_cutoff_utc": tick.data_cutoff_utc.isoformat(),
            "status": tick.status,
            "distribution": to_json_mapping(tick.distribution),
            "selected_inputs": [
                {
                    "comment_id": comment_id,
                    "author_id": stance_by_id[comment_id].author_id,
                    "event_time_utc": stance_by_id[comment_id].event_time_utc.isoformat(),
                    "stance_category": stance_by_id[comment_id].stance_category,
                }
                for comment_id in tick.selected_comment_ids
            ],
        })
        for observation in tick.observations:
            row = observation.to_mapping()
            row["distribution_observation_time_utc"] = (
                tick.observation_time_utc.isoformat()
            )
            observations.append(row)
    return distributions, observations


def _detector_config(config: RunConfig, detector_id: str) -> Any:
    assert config.detection is not None
    value = getattr(config.detection, detector_id, None)
    if value is None:
        raise ValueError(f"Missing detection.{detector_id} configuration.")
    return value


def _default_detector_factory(detector_id: str, detector_config: Any) -> Any:
    return create_detector(detector_id, config=detector_config)


def _all_routes(config: RunConfig) -> dict[str, Any]:
    assert config.detection is not None
    routes = list(config.detection.routes or ())
    if config.detection.activity_route is not None:
        routes.append(config.detection.activity_route)
    result: dict[str, Any] = {}
    for route in routes:
        if route.signal_id in result:
            raise ValueError(f"Duplicate detector route for {route.signal_id!r}.")
        result[route.signal_id] = route
    return result


def _candidate_video_ids(
    association: DetectionAssociation,
    alert_video_ids: Mapping[str, tuple[str, ...]],
) -> tuple[str, ...]:
    ids: set[str] = set()
    for alert_id in association.activity_alert_ids + association.polarization_alert_ids:
        ids.update(alert_video_ids.get(alert_id, ()))
    return tuple(sorted(ids))


def _result_row(
    *,
    run_id: str,
    result: DetectionResult,
    observation: ActivityObservation | SignalObservation,
    cutoff: datetime,
    signal_family: str,
) -> dict[str, Any]:
    return {
        "detection_result_id": detection_result_id(run_id=run_id, result=result),
        "run_id": run_id,
        "signal_id": result.signal_id,
        "signal_family": signal_family,
        "detector_id": result.detector_id,
        "observation_time_utc": result.observation_time_utc.isoformat(),
        "data_cutoff_utc": cutoff.isoformat(),
        "value": observation.value,
        "support_count": observation.support_count,
        "quality": result.quality,
        "triggered": result.triggered,
        "score": result.score,
        "detector_metadata": to_json_mapping(result.detector_metadata),
    }


def run_causal_end_to_end_replay(
    config: RunConfig,
    *,
    base_dir: str | Path,
    execute_rag: bool = False,
    detector_factory: DetectorFactory | None = None,
    g1_runner: Callable[[Any], dict[str, Any]] | None = None,
    g2_runner: Callable[[Any], dict[str, Any]] | None = None,
) -> CausalReplayArtifacts:
    """Execute one fixed-config causal replay without calibrating any parameter."""

    resolved = resolve_run_config(config, base_dir=base_dir)
    actual = resolved.config
    if actual.simulation is None or actual.simulation.causal_end_to_end is None:
        raise ValueError("simulation.causal_end_to_end is required.")
    replay = actual.simulation.causal_end_to_end
    assert actual.data is not None and actual.data.local_files is not None
    assert actual.data.experimental_input is not None
    assert actual.signals is not None and actual.signals.activity is not None
    assert actual.signals.polarization is not None
    assert actual.detection is not None
    assert actual.association is not None
    assert actual.polarization is not None and actual.polarization.runtime is not None
    assert actual.rag is not None
    assert actual.rag.candidate_evidence is not None
    assert actual.rag.candidate_validation is not None
    if (
        actual.detection.gap_policy is not None
        and actual.detection.gap_policy.policy_id != PRESERVE_STATE_V1
    ):
        raise ValueError("Causal replay V1 requires preserve_state_v1.")
    temporal = actual.signals.polarization
    if temporal.start_time_utc < replay.start_utc or temporal.end_time_utc > replay.end_utc:
        raise ValueError("Polarization temporal bounds must fit inside replay bounds.")

    root = Path(replay.output_dir)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Causal replay output directory must be empty or absent.")
    root.mkdir(parents=True, exist_ok=True)
    source_path = Path(actual.data.local_files.comments_path)
    source = read_input_table(source_path)
    prepared = prepare_causal_minimal_batch(
        source,
        actual.data.experimental_input,
        source_path=source_path,
        data_cutoff_utc=replay.end_utc,
    )
    allowed_videos = set(actual.polarization.runtime.video_ids)
    activity_window = pd.to_timedelta(actual.signals.activity.window).to_pytimedelta()
    polarization_window = pd.to_timedelta(temporal.window).to_pytimedelta()
    history_start = min(
        replay.start_utc - activity_window,
        temporal.start_time_utc - polarization_window,
    )
    records = tuple(
        record for record in prepared.records
        if record.video_id in allowed_videos
        and history_start <= record.event_time_utc < replay.end_utc
    )
    stance_history_start = temporal.start_time_utc - polarization_window
    stance_records = tuple(
        record for record in records if record.event_time_utc >= stance_history_start
    )
    stances = _load_stances(Path(replay.stance_artifact_path))
    stance_manifest = _read_json(Path(replay.stance_manifest_path))
    _validate_stance_reuse(
        config=actual,
        records=stance_records,
        stances=stances,
        manifest=stance_manifest,
        source_path=source_path,
    )
    stance_by_id = {item.comment_id: item for item in stances}
    record_by_id = {item.comment_id: item for item in records}

    activity_signal = create_activity_signal(actual.signals.activity)
    activity_signal.start_at(replay.start_utc)
    polarization_signal = create_configured_polarization_signal(actual)
    route_by_signal = _all_routes(actual)
    expected_signals = {actual.signals.activity.signal_id} | {
        f"polarization_{measure}" for measure in temporal.enabled_measures
    }
    missing_routes = sorted(expected_signals - set(route_by_signal))
    if missing_routes:
        raise ValueError(f"Missing detector routes: {missing_routes}.")
    factory = detector_factory or _default_detector_factory
    detectors = {
        signal_id: factory(route.detector_id, _detector_config(actual, route.detector_id))
        for signal_id, route in route_by_signal.items()
        if signal_id in expected_signals
    }
    associator = TemporalDetectionAssociator(
        run_id=actual.identity.run_id,
        config=actual.association,
    )
    projector = EventCandidateProjector(
        run_id=actual.identity.run_id,
        config_hash=resolved.config_hash,
        input_policy_id=CAUSAL_MINIMAL_POLICY_ID,
        association_artifact_ref=ASSOCIATION_ARTIFACT,
    )
    evidence_config = replace(
        actual.rag.candidate_evidence,
        source_path=str(source_path),
        output_dir=str(root / "candidate_evidence"),
    )
    g2_gate_config = actual.rag.g2_gate or G2GatePolicyConfig()

    activity_rows: list[dict[str, Any]] = []
    distribution_rows: list[dict[str, Any]] = []
    polarization_rows: list[dict[str, Any]] = []
    detection_rows: list[dict[str, Any]] = []
    alerts: list[DetectionAlert] = []
    associations: list[DetectionAssociation] = []
    candidates: list[AssociatedEventCandidate] = []
    alert_video_ids: dict[str, tuple[str, ...]] = {}
    released_stances: list[CommentStance] = []
    released_records: list[CausalInputRecord] = []
    record_index = 0
    future_leaks = Counter()
    scheduler_counts = Counter()
    evidence_status_by_candidate: dict[str, str] = {}
    evidence_readiness_transitions: list[dict[str, str]] = []

    def audit_evidence_readiness(cutoff: datetime) -> None:
        for candidate in candidates:
            scope = derive_candidate_evidence_scope(
                candidate,
                evidence_config,
                current_data_cutoff_utc=cutoff,
            )
            previous = evidence_status_by_candidate.get(candidate.candidate_id)
            if previous != scope.status:
                evidence_status_by_candidate[candidate.candidate_id] = scope.status
                evidence_readiness_transitions.append({
                    "candidate_id": candidate.candidate_id,
                    "data_cutoff_utc": cutoff.isoformat(),
                    "status": scope.status,
                })

    def dispatch(
        observation: ActivityObservation | SignalObservation,
        *,
        cutoff: datetime,
        family: str,
        video_ids: Iterable[str],
    ) -> DetectionAlert | None:
        if observation.observation_time_utc >= cutoff:
            future_leaks["signal"] += 1
            raise AssertionError("Signal observation is not strictly before its cutoff.")
        route = route_by_signal[observation.signal.signal_id if isinstance(
            observation, ActivityObservation
        ) else observation.signal_id]
        result = dispatch_numeric_observation(
            route=route,
            observation=observation,
            detectors={route.detector_id: detectors[route.signal_id]},
        )
        if result is None:
            return None
        detection_rows.append(_result_row(
            run_id=actual.identity.run_id,
            result=result,
            observation=observation,
            cutoff=cutoff,
            signal_family=family,
        ))
        alert = detection_alert_from_result(
            result,
            run_id=actual.identity.run_id,
            config_hash=resolved.config_hash,
            signal_family=family,
            support_count=observation.support_count,
            data_cutoff_utc=cutoff,
            detection_artifact_ref=DETECTION_RESULTS,
        )
        if alert is not None:
            alert_video_ids[alert.alert_id] = tuple(sorted(set(video_ids)))
        return alert

    def process_activity(
        observations: Iterable[ActivityObservation], cutoff: datetime
    ) -> list[DetectionAlert]:
        new_alerts: list[DetectionAlert] = []
        for observation in observations:
            activity_rows.append(_activity_row(observation, cutoff))
            window_videos = {
                item.video_id for item in released_records
                if observation.window_start_utc <= item.event_time_utc <= observation.window_end_utc
            }
            alert = dispatch(
                observation,
                cutoff=cutoff,
                family=ACTIVITY,
                video_ids=window_videos,
            )
            if alert is not None:
                new_alerts.append(alert)
        return new_alerts

    cutoffs = _clock_cutoffs(replay.start_utc, replay.end_utc, replay.step_delta)
    for cutoff in cutoffs:
        new_activity_observations: list[ActivityObservation] = []
        new_stances: list[CommentStance] = []
        while record_index < len(records) and records[record_index].event_time_utc < cutoff:
            record = records[record_index]
            released_records.append(record)
            stance = stance_by_id.get(record.comment_id)
            if stance is not None:
                released_stances.append(stance)
                new_stances.append(stance)
            new_activity_observations.extend(
                activity_signal.on_event({
                    "comment_id": record.comment_id,
                    "video_id": record.video_id,
                    "author_id": record.author_id,
                    "event_time_utc": record.event_time_utc,
                })
            )
            record_index += 1
        scheduler_counts["comments_released"] += len(new_stances)
        new_activity_observations.extend(activity_signal.advance_to(cutoff))
        scheduler_counts["activity_ticks"] += len(new_activity_observations)
        new_alerts = process_activity(new_activity_observations, cutoff)

        ticks = polarization_signal.advance(new_stances, data_cutoff_utc=cutoff)
        scheduler_counts["polarization_ticks"] += len(ticks)
        tick_distributions, tick_observations = _polarization_rows(ticks, stance_by_id)
        distribution_rows.extend(tick_distributions)
        polarization_rows.extend(tick_observations)
        for tick in ticks:
            videos = {
                record_by_id[comment_id].video_id
                for comment_id in tick.selected_comment_ids
            }
            for observation in tick.observations:
                alert = dispatch(
                    observation,
                    cutoff=cutoff,
                    family=POLARIZATION,
                    video_ids=videos,
                )
                if alert is not None:
                    new_alerts.append(alert)
        new_alerts.sort(key=lambda item: (
            item.alert_time_utc, item.signal_family, item.signal_id,
            item.detector_id, item.alert_id,
        ))
        alerts.extend(new_alerts)
        finalized = associator.advance(new_alerts, data_cutoff_utc=cutoff)
        associations.extend(finalized)
        for association in finalized:
            candidate = projector.promote(
                association,
                created_at_cutoff_utc=cutoff,
                associated_video_ids=_candidate_video_ids(
                    association, alert_video_ids
                ),
            )
            if candidate is not None:
                candidates.append(candidate)
        audit_evidence_readiness(cutoff)

    eos_associations = associator.advance(
        (), data_cutoff_utc=replay.end_utc, end_of_stream=True
    )
    associations.extend(eos_associations)
    for association in eos_associations:
        candidate = projector.promote(
            association,
            created_at_cutoff_utc=replay.end_utc,
            associated_video_ids=_candidate_video_ids(association, alert_video_ids),
        )
        if candidate is not None:
            candidates.append(candidate)
    audit_evidence_readiness(replay.end_utc)

    if record_index != len(records):
        future_leaks["source_release"] += len(records) - record_index
        raise AssertionError("Replay ended before all in-range causal records were released.")

    paths = {
        "resolved_config": root / RESOLVED_CONFIG,
        "input_policy_manifest": root / INPUT_POLICY_MANIFEST,
        "stances": root / STANCE_ARTIFACT,
        "activity_observations": root / ACTIVITY_OBSERVATIONS,
        "opinion_distributions": root / POLARIZATION_DISTRIBUTIONS,
        "polarization_observations": root / POLARIZATION_OBSERVATIONS,
        "detection_results": root / DETECTION_RESULTS,
        "detection_alerts": root / DETECTION_ALERTS,
        "associations": root / ASSOCIATION_ARTIFACT,
        "candidates": root / EVENT_CANDIDATE_ARTIFACT,
        "validations": root / VALIDATION_RESULTS,
        "activity_manifest": root / "activity_signal_manifest.json",
        "polarization_manifest": root / "polarization_signal_manifest.json",
        "detection_manifest": root / "detection_manifest.json",
        "association_manifest": root / "detection_association_manifest.json",
        "candidate_manifest": root / "event_candidate_manifest.json",
    }
    _write_json(paths["resolved_config"], run_config_to_mapping(actual, path_base=base_dir))
    input_manifest = dict(prepared.manifest)
    input_manifest["selected_video_ids"] = sorted(allowed_videos)
    input_manifest["selected_records"] = len(records)
    _write_json(paths["input_policy_manifest"], input_manifest)
    artifact_hashes = {
        "stances": _write_jsonl(
            paths["stances"], (to_json_mapping(item) for item in released_stances)
        ),
        "activity_observations": _write_jsonl(paths["activity_observations"], activity_rows),
        "opinion_distributions": _write_jsonl(paths["opinion_distributions"], distribution_rows),
        "polarization_observations": _write_jsonl(paths["polarization_observations"], polarization_rows),
        "detection_results": _write_jsonl(paths["detection_results"], detection_rows),
        "detection_alerts": _write_jsonl(
            paths["detection_alerts"], (item.to_mapping() for item in alerts)
        ),
        "associations": _write_jsonl(
            paths["associations"], (item.to_mapping() for item in associations)
        ),
        "candidates": _write_jsonl(
            paths["candidates"], (item.to_mapping() for item in candidates)
        ),
    }
    _write_json(paths["activity_manifest"], {
        "artifact_version": "causal_activity_signal_v1",
        "run_id": actual.identity.run_id,
        "signal_id": actual.signals.activity.signal_id,
        "window": actual.signals.activity.window,
        "cadence": actual.signals.activity.cadence,
        "observation_count": len(activity_rows),
        "artifact": str(paths["activity_observations"]),
        "causal_rule": "observation_time_utc < data_cutoff_utc",
    })
    _write_json(paths["polarization_manifest"], {
        "artifact_version": "causal_polarization_signal_v1",
        "run_id": actual.identity.run_id,
        "window": temporal.window,
        "cadence": temporal.cadence,
        "enabled_measures": list(temporal.enabled_measures),
        "distribution_count": len(distribution_rows),
        "observation_count": len(polarization_rows),
        "distribution_artifact": str(paths["opinion_distributions"]),
        "observation_artifact": str(paths["polarization_observations"]),
        "causal_rule": "observation_time_utc < data_cutoff_utc",
    })
    _write_json(paths["detection_manifest"], {
        "artifact_version": "causal_detection_v1",
        "run_id": actual.identity.run_id,
        "routes": [
            {"signal_id": route.signal_id, "detector_id": route.detector_id}
            for route in route_by_signal.values() if route.signal_id in expected_signals
        ],
        "gap_policy": PRESERVE_STATE_V1,
        "gap_policy_status": "NOT_CALIBRATED",
        "detection_results": len(detection_rows),
        "triggered_alerts": len(alerts),
        "result_artifact": str(paths["detection_results"]),
        "alert_artifact": str(paths["detection_alerts"]),
    })
    _write_json(paths["association_manifest"], {
        "artifact_version": "causal_detection_association_v1",
        "run_id": actual.identity.run_id,
        "policy_id": actual.association.policy_id,
        "tolerance": actual.association.tolerance,
        "parameter_status": actual.association.parameter_status,
        "end_of_stream_flush": True,
        "association_count": len(associations),
        "artifact": str(paths["associations"]),
    })
    _write_json(paths["candidate_manifest"], {
        "artifact_version": "causal_event_candidate_v1",
        "run_id": actual.identity.run_id,
        "projection_policy_id": "association_to_candidate_v1",
        "candidate_count": len(candidates),
        "artifact": str(paths["candidates"]),
        "evidence_embedded": False,
        "score_fusion": False,
    })

    evidence: CandidateEvidenceArtifacts = build_candidate_evidence(
        candidates=candidates,
        config=evidence_config,
        current_data_cutoff_utc=replay.end_utc,
    )
    validation_rows: list[dict[str, Any]] = []
    rag_artifacts: list[CandidateRagArtifacts] = []
    rag_errors: list[dict[str, str]] = []
    if execute_rag:
        for candidate in candidates:
            validation_config = replace(
                actual.rag.candidate_validation,
                candidates_path=str(paths["candidates"]),
                evidence_dir=str(evidence.output_dir),
                output_dir=str(root / "candidate_rag" / candidate.candidate_id),
                candidate_id=candidate.candidate_id,
            )
            try:
                kwargs: dict[str, Any] = {"execute": True}
                if g1_runner is not None:
                    kwargs["g1_runner"] = g1_runner
                if g2_runner is not None:
                    kwargs["g2_runner"] = g2_runner
                result = run_candidate_rag_validation(
                    validation_config,
                    g2_gate=g2_gate_config,
                    **kwargs,
                )
                rag_artifacts.append(result)
                validation_rows.extend(
                    json.loads(line)
                    for line in result.validation_results_path.read_text(
                        encoding="utf-8"
                    ).splitlines()
                    if line.strip()
                )
            except AssertionError:
                raise
            except Exception as exc:  # isolate one candidate without hiding the error
                rag_errors.append({
                    "candidate_id": candidate.candidate_id,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                })
    artifact_hashes["validations"] = _write_jsonl(paths["validations"], validation_rows)
    rag_manifests = [_read_json(item.manifest_path) for item in rag_artifacts]
    rag_token_usage = {
        name: sum(int(manifest.get("token_usage", {}).get(name, 0) or 0)
                  for manifest in rag_manifests)
        for name in ("input_tokens", "output_tokens", "total_tokens")
    }
    rag_token_usage["usage_complete"] = all(
        bool(manifest.get("token_usage", {}).get("usage_complete", False))
        for manifest in rag_manifests
    ) if rag_manifests else True
    external_call_counts = {
        "serper_request_count": sum(
            int(manifest.get("external_calls", {}).get("serper_request_count", 0) or 0)
            for manifest in rag_manifests
        ),
        "query_call_count": sum(
            int(manifest.get("external_calls", {}).get("query_call_count", 0) or 0)
            for manifest in rag_manifests
        ),
        "validation_call_count": sum(
            int(manifest.get("external_calls", {}).get("validation_call_count", 0) or 0)
            for manifest in rag_manifests
        ),
    }

    candidate_counts = Counter(item.candidate_type for item in candidates)
    evidence_manifest = _read_json(evidence.manifest_path)
    evidence_counts = evidence_manifest.get("counts", {})
    future_leak_count = (
        sum(future_leaks.values())
        + int(evidence_manifest.get("causality", {}).get("future_leak_count", 0))
        + sum(
            int(row.get("internal_future_leak_count", 0))
            + int(row.get("external_future_leak_count", 0))
            for row in validation_rows
        )
    )
    if future_leak_count:
        raise AssertionError("End-to-end causal audit found future leakage.")
    manifest = {
        "artifact_version": "causal_end_to_end_replay_v1",
        "status": "executed_successfully",
        "prototype_status": "EXPERIMENT_READY",
        "methodological_status": replay.parameter_status,
        "run_id": actual.identity.run_id,
        "config_hash": resolved.config_hash,
        "source": {
            "path": str(source_path),
            "sha256": prepared.manifest["source_sha256"],
            "input_policy_id": CAUSAL_MINIMAL_POLICY_ID,
            "gold_dependency": False,
        },
        "replay_clock": {
            "policy_id": replay.clock_policy_id,
            "start_utc": replay.start_utc.isoformat(),
            "end_utc": replay.end_utc.isoformat(),
            "step": replay.step,
            "cutoff_count": len(cutoffs),
            "strict_availability_rule": "event_time_utc < current_data_cutoff_utc",
        },
        "scheduling": {
            "replay_step_independent_from_signal_cadence": True,
            "catch_up_all_ticks_strictly_before_cutoff": True,
            "activity_window": actual.signals.activity.window,
            "activity_cadence": actual.signals.activity.cadence,
            "polarization_window": temporal.window,
            "polarization_cadence": temporal.cadence,
            **dict(scheduler_counts),
        },
        "stance": {
            "mode": replay.stance_mode,
            "artifact": str(replay.stance_artifact_path),
            "manifest": str(replay.stance_manifest_path),
            "released_with_canonical_comment_only": True,
            "future_stances_preloaded_into_signal_state": False,
            "topic_id": actual.topic_profile.topic_id,
            "topic_description": actual.topic_profile.topic_description,
            "proposition": actual.topic_profile.proposition.proposition_text,
            "context_hash": hashlib.sha256(
                actual.topic_profile.proposition.context.encode("utf-8")
            ).hexdigest(),
            "prompt_id": actual.topic_profile.stance_prompt_id,
            "prompt_version": actual.topic_profile.stance_prompt_version,
        },
        "detection": {
            "routes": [
                {"signal_id": route.signal_id, "detector_id": route.detector_id}
                for route in route_by_signal.values()
                if route.signal_id in expected_signals
            ],
            "instances_per_route": 1,
            "gap_policy": PRESERVE_STATE_V1,
            "gap_policy_status": "NOT_CALIBRATED",
        },
        "association": {
            "policy_id": actual.association.policy_id,
            "tolerance": actual.association.tolerance,
            "parameter_status": actual.association.parameter_status,
            "causal_buffering": True,
            "end_of_stream_flush": True,
        },
        "candidate_projection": {
            "policy_id": "association_to_candidate_v1",
            "one_per_finalized_association": True,
        },
        "evidence": {
            "policy_id": evidence_config.policy_id,
            "mode": evidence_config.mode,
            "lookback": evidence_config.lookback,
            "lookahead": evidence_config.lookahead,
            "cutoff_utc": replay.end_utc.isoformat(),
            "source_is_same_canonical_silver_universe": True,
            "readiness_transitions": evidence_readiness_transitions,
        },
        "rag": {
            "executed": execute_rag,
            "g1_runner": "injected" if g1_runner is not None else "configured_default",
            "g2_runner": "injected" if g2_runner is not None else "configured_default",
            "candidate_failures_isolated": True,
            "errors": rag_errors,
            "token_usage": rag_token_usage,
            "external_calls": external_call_counts,
            "g2_gate": to_json_mapping(g2_gate_config),
        },
        "counts": {
            "comments_released": len(released_records),
            "activity_observations": len(activity_rows),
            "polarization_observations": len(polarization_rows),
            "detection_results": len(detection_rows),
            "detection_alerts": len(alerts),
            "detection_associations": len(associations),
            "event_candidates": len(candidates),
            "activity_candidates": candidate_counts[ACTIVITY],
            "polarization_candidates": candidate_counts[POLARIZATION],
            "activity_polarization_candidates": candidate_counts["ACTIVITY_POLARIZATION"],
            "evidence_ready": int(evidence_counts.get("evidence_ready", 0)),
            "no_evidence": int(evidence_counts.get("no_evidence_found", 0)),
            "pending_evidence": int(evidence_counts.get("pending_evidence_horizon", 0)),
            "blocked_evidence": int(evidence_counts.get("blocked_missing_video_lineage", 0)),
            "g1_executions": sum(row.get("g1_result") is not None for row in validation_rows),
            "g2_executions": sum(row.get("g2_result") is not None for row in validation_rows),
            "g2_eligible": sum(
                row.get("g2_execution_status") != "NOT_ELIGIBLE"
                for row in validation_rows
            ),
            "g2_skipped_by_gate": sum(
                row.get("g2_execution_status") == "SKIPPED_BY_G1_GATE"
                for row in validation_rows
            ),
            "g2_failed": sum(
                row.get("g2_execution_status") == "FAILED"
                for row in validation_rows
            ),
            "candidate_validation_results": len(validation_rows),
        },
        "causality": {
            "source_future_leak_count": 0,
            "activity_future_leak_count": 0,
            "polarization_future_leak_count": 0,
            "alert_future_leak_count": 0,
            "association_future_leak_count": 0,
            "candidate_future_leak_count": 0,
            "internal_evidence_future_leak_count": 0,
            "external_evidence_future_leak_count": 0,
            "future_leak_count": future_leak_count,
        },
        "artifacts": {
            name: str(path) for name, path in paths.items()
        } | {
            "candidate_evidence_dir": str(evidence.output_dir),
            "candidate_rag_dirs": [str(item.output_dir) for item in rag_artifacts],
            "sha256": artifact_hashes,
        },
        "limitations": [
            "Methodological parameters remain uncalibrated.",
            "Historical web retrieval cannot prove historical index availability.",
            "Arrival time is unavailable; replay uses authoritative event_time_utc.",
            "No durable checkpoints, late-arrival recovery, or production streaming.",
        ],
        "calibration_performed": False,
        "scientific_validation_performed": False,
    }
    manifest_path = root / REPLAY_MANIFEST
    _write_json(manifest_path, manifest)
    return CausalReplayArtifacts(
        output_dir=root,
        manifest_path=manifest_path,
        candidate_path=paths["candidates"],
        validation_results_path=paths["validations"],
        comments_released=len(released_records),
        activity_observations=len(activity_rows),
        polarization_observations=len(polarization_rows),
        detection_results=len(detection_rows),
        alerts=len(alerts),
        associations=len(associations),
        candidates=len(candidates),
        validations=len(validation_rows),
        future_leak_count=future_leak_count,
    )


__all__ = [
    "ACTIVITY_OBSERVATIONS",
    "DETECTION_ALERTS",
    "DETECTION_RESULTS",
    "INPUT_POLICY_MANIFEST",
    "POLARIZATION_DISTRIBUTIONS",
    "POLARIZATION_OBSERVATIONS",
    "REPLAY_MANIFEST",
    "RESOLVED_CONFIG",
    "STANCE_ARTIFACT",
    "VALIDATION_RESULTS",
    "CausalReplayArtifacts",
    "run_causal_end_to_end_replay",
]
