"""Standalone comments-to-polarization runtime with optional neutral detection."""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from youtube_pipeline.comment_stance import CLASSIFIED, CommentStance
from youtube_pipeline.configuration.models import RunConfig
from youtube_pipeline.configuration.paths import resolve_run_config
from youtube_pipeline.experimental_input import (
    adapt_stance_records,
    prepare_causal_minimal_batch,
    read_input_table,
    source_sha256,
)
from youtube_pipeline.polarization_signal_composition import create_configured_polarization_signal
from youtube_pipeline.polarization_detection_runtime import (
    DETECTION_ALERT_ARTIFACT,
    DETECTION_ARTIFACT,
    DETECTION_MANIFEST,
    configured_polarization_routes,
    load_polarization_signal_observations,
    run_polarization_detection_runtime,
)
from youtube_pipeline.polarization_runtime_config import (
    VideoScope,
    normalize_video_id,
    resolve_video_scope,
)
from youtube_pipeline.polarization_signals import PolarizationTemporalTick
from youtube_pipeline.signal_observations import VALUE_AVAILABLE, to_json_mapping
from youtube_pipeline.stance_classifier import StanceCommentInput, StanceModelTransport
from youtube_pipeline.stance_classifier_io import (
    CLASSIFIER_MANIFEST,
    CLASSIFIER_RESOLVED_CONFIG,
    STANCE_ARTIFACT,
    run_configured_comment_stance_classification,
)
from youtube_pipeline.storage import canonicalize_comment_event_time


DISTRIBUTION_ARTIFACT = "opinion_distributions.jsonl"
SIGNAL_ARTIFACT = "polarization_signal_observations.jsonl"
RUNTIME_MANIFEST = "polarization_runtime_manifest.json"
REQUIRED_COMMENT_COLUMNS = frozenset({
    "video_id", "comment_id", "text", "author_id", "published_at",
    "is_reply", "reply_to_comment_id",
})


def _duration(value: str) -> timedelta:
    parsed = pd.to_timedelta(value)
    return timedelta(microseconds=int(parsed.value) // 1000)


def _optional_text(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value)
    return text if text.strip() else None


def _write_jsonl_exclusive(path: Path, rows: Iterable[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    with path.open("x", encoding="utf-8") as target:
        for row in rows:
            encoded = json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
            target.write(encoded)
            digest.update(encoded.encode("utf-8"))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class PolarizationRuntimePlan:
    comments: tuple[StanceCommentInput, ...]
    comments_path: Path
    videos_path: Path
    source_row_count: int
    selected_video_ids: tuple[str, ...]
    video_scope: VideoScope
    selected_video_metadata: tuple[dict[str, str | None], ...]
    selection_start_utc: datetime
    selection_end_utc: datetime
    selected_time_min_utc: datetime
    selected_time_max_utc: datetime
    expected_classifier_calls: int
    input_policy_manifest: dict[str, Any] | None


@dataclass(frozen=True, slots=True)
class PolarizationRuntimeArtifacts:
    run_id: str
    stance_path: Path
    distribution_path: Path
    signal_path: Path
    manifest_path: Path
    resolved_config_path: Path
    input_count: int
    tick_count: int
    observation_count: int
    detection_result_path: Path | None = None
    detection_manifest_path: Path | None = None
    detection_result_count: int = 0
    detection_trigger_count: int = 0
    detection_alert_path: Path | None = None
    detection_alert_count: int = 0


def _require_runtime_config(config: RunConfig):
    if config.data is None or config.data.local_files is None:
        raise ValueError("Polarization runtime requires data.local_files.")
    if config.signals is None or config.signals.polarization is None:
        raise ValueError("Polarization runtime requires signals.polarization.")
    if config.polarization is None or config.polarization.runtime is None:
        raise ValueError("Polarization runtime requires polarization.runtime.")
    if config.polarization.classifier is None:
        raise ValueError("Polarization runtime requires polarization.classifier.")
    return (
        config.data.local_files,
        config.signals.polarization,
        config.polarization.runtime,
    )


def prepare_polarization_runtime(config: RunConfig) -> PolarizationRuntimePlan:
    """Read and validate the configured real-input slice without model calls."""
    local_files, temporal, runtime = _require_runtime_config(config)
    comments_path, videos_path = Path(local_files.comments_path), Path(local_files.videos_path)
    selection_start = temporal.start_time_utc - _duration(temporal.window)
    selection_end = temporal.end_time_utc
    comments = read_input_table(comments_path)
    source_row_count = len(comments)
    input_policy_manifest = None
    policy_records = None
    if config.data is not None and config.data.experimental_input is not None:
        comments = canonicalize_comment_event_time(comments)
        prepared = prepare_causal_minimal_batch(
            comments,
            config.data.experimental_input,
            source_path=comments_path,
            data_cutoff_utc=selection_end,
        )
        comments = prepared.dataframe
        policy_records = prepared.records
        input_policy_manifest = prepared.manifest
        comments = comments.assign(event_time_utc_runtime=comments["event_time_utc"])
    else:
        missing = REQUIRED_COMMENT_COLUMNS - set(comments.columns)
        if missing:
            raise ValueError(f"Comments input lacks required columns: {sorted(missing)}")
        timestamps = pd.to_datetime(comments["published_at"], utc=True, errors="coerce")
        if timestamps.isna().any():
            raise ValueError("Comments input contains invalid published_at timestamps.")
        comments = comments.assign(event_time_utc_runtime=timestamps)
        if comments["comment_id"].isna().any():
            raise ValueError("Comments input requires non-null comment_id values.")

    scope = resolve_video_scope(
        (record.video_id for record in policy_records)
        if policy_records is not None
        else comments["video_id"].tolist(),
        runtime.video_ids,
    )
    effective_video_ids = scope.effective_video_ids
    normalized_video_ids = comments["video_id"].map(normalize_video_id)
    selected = comments[
        normalized_video_ids.isin(effective_video_ids)
        & (comments["event_time_utc_runtime"] >= selection_start)
        & (comments["event_time_utc_runtime"] < selection_end)
    ].copy()
    if selected.empty:
        raise ValueError("Configured polarization runtime selection contains no comments.")
    if selected["comment_id"].astype(str).duplicated().any():
        raise ValueError("Selected polarization comments require unique comment_id values.")
    selected = selected.sort_values(["event_time_utc_runtime", "comment_id"], kind="stable")

    comments_input: list[StanceCommentInput] = []
    expected_calls = 0
    if policy_records is not None:
        selected_ids = selected["comment_id"].astype(str).tolist()
        comments_input = [
            adapted.classifier_input
            for adapted in adapt_stance_records(
                policy_records, selected_comment_ids=selected_ids
            )
        ]
    else:
        parent_ids = set(selected.loc[selected["is_reply"] == True, "reply_to_comment_id"].dropna().astype(str))
        parent_rows = comments[comments["comment_id"].astype(str).isin(parent_ids)]
        lookup = {}
        for parent_id, group in parent_rows.groupby(parent_rows["comment_id"].astype(str), sort=False):
            comparable = group[["text", "author_id", "published_at"]].fillna("").astype(str)
            if len(comparable.drop_duplicates()) != 1:
                raise ValueError(f"Parent comment {parent_id!r} is ambiguous in the source input.")
            lookup[parent_id] = next(group.itertuples(index=False))

        for row in selected.itertuples(index=False):
            event_time = row.event_time_utc_runtime.to_pydatetime()
            parent_id = _optional_text(row.reply_to_comment_id) if bool(row.is_reply) else None
            parent_row = lookup.get(parent_id) if parent_id is not None else None
            parent_text = _optional_text(parent_row.text) if parent_row is not None else None
            parent_time = (
                parent_row.event_time_utc_runtime.to_pydatetime() if parent_row is not None else None
            )
            comments_input.append(StanceCommentInput(
                comment_id=str(row.comment_id), author_id=_optional_text(row.author_id),
                event_time_utc=event_time, text="" if pd.isna(row.text) else str(row.text),
                parent_comment_id=parent_id, parent_text=parent_text,
                parent_event_time_utc=parent_time, is_deleted=False,
            ))

    for comment in comments_input:
        if comment.text.strip() and (
            comment.parent_comment_id is None or (
                comment.parent_text is not None
                and comment.parent_event_time_utc is not None
                and comment.parent_event_time_utc <= comment.event_time_utc
            )
        ):
            expected_calls += 1

    videos = read_input_table(videos_path)
    if not {"video_id", "title"} <= set(videos.columns):
        raise ValueError("Videos input requires video_id and title columns.")
    video_lookup = {
        video_id: row
        for row in videos.itertuples(index=False)
        if (video_id := normalize_video_id(row.video_id)) is not None
    }
    metadata = tuple({
        "video_id": video_id,
        "title": (
            _optional_text(video_lookup[video_id].title)
            if video_id in video_lookup else None
        ),
        "published_at": (
            _optional_text(getattr(video_lookup[video_id], "publishedAt", None))
            if video_id in video_lookup else None
        ),
    } for video_id in effective_video_ids)
    return PolarizationRuntimePlan(
        tuple(comments_input), comments_path, videos_path, source_row_count,
        effective_video_ids, scope, metadata, selection_start, selection_end,
        comments_input[0].event_time_utc, comments_input[-1].event_time_utc,
        expected_calls, input_policy_manifest,
    )


def _load_stances(path: Path) -> tuple[CommentStance, ...]:
    stances = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        stances.append(CommentStance(
            comment_id=row["comment_id"], author_id=row["author_id"],
            event_time_utc=datetime.fromisoformat(row["event_time_utc"].replace("Z", "+00:00")),
            proposition_id=row["proposition_id"], stance_category=row["stance_category"],
            classification_status=row["classification_status"], quality=row["quality"],
        ))
    if tuple(stances) != tuple(sorted(stances, key=lambda s: (s.event_time_utc, s.comment_id))):
        raise ValueError("Persisted CommentStance records are not deterministically ordered.")
    return tuple(stances)


def _runtime_rows(ticks: tuple[PolarizationTemporalTick, ...], stances: tuple[CommentStance, ...]):
    by_id = {stance.comment_id: stance for stance in stances}
    distribution_rows, observation_rows = [], []
    for tick in ticks:
        window_records = [stance for stance in stances if (
            stance.proposition_id == tick.distribution.proposition_id
            and tick.window_start_utc <= stance.event_time_utc <= tick.window_end_utc
        )]
        input_authors = {stance.author_id for stance in window_records if stance.author_id is not None}
        selected = tuple(by_id[comment_id] for comment_id in tick.selected_comment_ids)
        lineage = [{
            "comment_id": stance.comment_id, "author_id": stance.author_id,
            "event_time_utc": stance.event_time_utc.isoformat(),
            "stance_category": stance.stance_category,
        } for stance in selected]
        coverage = {
            "classified_fraction": (
                tick.distribution.classified_comment_count / tick.distribution.input_count
                if tick.distribution.input_count else None
            ),
            "represented_author_fraction": (
                tick.distribution.support_count / len(input_authors) if input_authors else None
            ),
        }
        distribution_rows.append({
            "observation_time_utc": tick.observation_time_utc.isoformat(),
            "window_start_utc": tick.window_start_utc.isoformat(),
            "window_end_utc": tick.window_end_utc.isoformat(),
            "data_cutoff_utc": tick.data_cutoff_utc.isoformat(),
            "status": tick.status,
            "distribution": to_json_mapping(tick.distribution),
            "selected_inputs": lineage,
            "coverage": coverage,
        })
        for observation in tick.observations:
            row = observation.to_mapping()
            row["distribution_observation_time_utc"] = tick.observation_time_utc.isoformat()
            observation_rows.append(row)
    return distribution_rows, observation_rows


def _sanity_checks(ticks: tuple[PolarizationTemporalTick, ...],
                   stances: tuple[CommentStance, ...],
                   enabled_measures: tuple[str, ...]) -> dict[str, Any]:
    by_id = {stance.comment_id: stance for stance in stances}
    failures = []
    for tick in ticks:
        distribution = tick.distribution
        selected = [by_id[comment_id] for comment_id in tick.selected_comment_ids]
        if tick.status == VALUE_AVAILABLE and not math.isclose(
            math.fsum(distribution.weights or ()), 1.0, rel_tol=0.0, abs_tol=1e-9
        ):
            failures.append(f"weights:{tick.observation_time_utc.isoformat()}")
        if distribution.support_count != len(selected):
            failures.append(f"support:{tick.observation_time_utc.isoformat()}")
        authors = [stance.author_id for stance in selected]
        if None in authors or len(authors) != len(set(authors)):
            failures.append(f"authors:{tick.observation_time_utc.isoformat()}")
        for stance in selected:
            if stance.classification_status != CLASSIFIED:
                failures.append(f"nonclassified:{stance.comment_id}")
            if not tick.window_start_utc <= stance.event_time_utc <= tick.window_end_utc:
                failures.append(f"window:{stance.comment_id}")
            if stance.event_time_utc >= tick.data_cutoff_utc:
                failures.append(f"future:{stance.comment_id}")
        actual_measures = tuple(
            observation.metadata["measure_id"] for observation in tick.observations
        )
        if actual_measures != enabled_measures:
            failures.append(f"measures:{tick.observation_time_utc.isoformat()}")
    return {
        "status": "passed" if not failures else "failed",
        "failures": failures,
        "weights_normalized": not any(item.startswith("weights:") for item in failures),
        "support_matches_selected_authors": not any(item.startswith("support:") for item in failures),
        "unique_author_per_tick": not any(item.startswith("authors:") for item in failures),
        "nonclassified_excluded": not any(item.startswith("nonclassified:") for item in failures),
        "future_leak_count": sum(item.startswith("future:") for item in failures),
        "one_distribution_all_enabled_measures": not any(item.startswith("measures:") for item in failures),
    }


async def run_polarization_runtime(
    config: RunConfig, *, output_dir: str | Path, base_dir: str | Path,
    transport: StanceModelTransport | None = None,
) -> PolarizationRuntimeArtifacts:
    """Execute classification, persisted handoff, temporal aggregation and measures."""
    resolved = resolve_run_config(config, base_dir=base_dir)
    actual = resolved.config
    plan = prepare_polarization_runtime(actual)
    root = Path(output_dir).absolute()
    runtime_paths = [root / name for name in (
        DISTRIBUTION_ARTIFACT, SIGNAL_ARTIFACT, RUNTIME_MANIFEST,
    )]
    detection_routes = configured_polarization_routes(actual)
    detection_paths = (
        [
            root / DETECTION_ARTIFACT,
            root / DETECTION_MANIFEST,
            root / DETECTION_ALERT_ARTIFACT,
        ]
        if detection_routes else []
    )
    classifier_paths = [root / name for name in (
        STANCE_ARTIFACT, CLASSIFIER_MANIFEST, CLASSIFIER_RESOLVED_CONFIG,
    )]
    if any(path.exists() for path in runtime_paths + classifier_paths + detection_paths):
        raise FileExistsError("Polarization runtime outputs already exist; choose a fresh directory.")
    artifacts = await run_configured_comment_stance_classification(
        config, plan.comments, output_dir=root, base_dir=base_dir,
        transport=transport, input_ref=str(plan.comments_path),
    )
    stances = _load_stances(artifacts.artifact_path)
    signal = create_configured_polarization_signal(actual)
    temporal = actual.signals.polarization  # type: ignore[union-attr]
    assert temporal is not None
    ticks = signal.advance(stances, data_cutoff_utc=temporal.end_time_utc)
    distribution_rows, observation_rows = _runtime_rows(ticks, stances)
    sanity = _sanity_checks(ticks, stances, temporal.enabled_measures)
    if sanity["status"] != "passed":
        raise RuntimeError("Polarization runtime sanity checks failed.")
    root.mkdir(parents=True, exist_ok=True)
    distribution_hash = _write_jsonl_exclusive(runtime_paths[0], distribution_rows)
    signal_hash = _write_jsonl_exclusive(runtime_paths[1], observation_rows)
    detection_artifacts = None
    if detection_routes:
        detection_artifacts = run_polarization_detection_runtime(
            actual,
            load_polarization_signal_observations(runtime_paths[1]),
            output_dir=root,
            config_hash=resolved.config_hash,
            input_policy_manifest=plan.input_policy_manifest,
            signal_artifact_path=runtime_paths[1],
        )

    classifier_manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))
    counts = {status: int(classifier_manifest["counts_by_status"].get(status, 0))
              for status in ("CLASSIFIED", "NOT_APPLICABLE", "UNCERTAIN", "SKIPPED", "ERROR")}
    count_total = sum(counts.values())
    status_counts = Counter(tick.status for tick in ticks)
    output_counts: dict[str, Counter] = {}
    for row in observation_rows:
        output_counts.setdefault(row["signal_id"], Counter())[row["status"]] += 1
    polarization = actual.polarization
    runtime = polarization.runtime  # type: ignore[union-attr]
    measure_parameters = {
        name: to_json_mapping(polarization.config_for(name))  # type: ignore[union-attr]
        for name in temporal.enabled_measures
    }
    manifest = {
        "artifact_version": "polarization_runtime_v1",
        "execution_status": "executed",
        "methodological_status": "RUNTIME_SMOKE_NOT_CALIBRATED",
        "run_id": actual.identity.run_id,
        "classifier_run_id": temporal.classification_run_id,
        "config_hash": resolved.config_hash,
        "source": {
            "role": runtime.source_role, "input_schema": runtime.input_schema,
            "comments_path": str(plan.comments_path), "comments_sha256": source_sha256(plan.comments_path),
            "videos_path": str(plan.videos_path), "videos_sha256": source_sha256(plan.videos_path),
            "source_row_count": plan.source_row_count,
            "selected_row_count": len(plan.comments),
            "video_ids": list(plan.selected_video_ids),
            **plan.video_scope.to_manifest(),
            "videos": list(plan.selected_video_metadata),
            "selection_start_utc": plan.selection_start_utc.isoformat(),
            "selection_end_utc": plan.selection_end_utc.isoformat(),
            "selected_time_min_utc": plan.selected_time_min_utc.isoformat(),
            "selected_time_max_utc": plan.selected_time_max_utc.isoformat(),
            "experimental_input_policy": plan.input_policy_manifest,
        },
        "classifier": {
            "model": classifier_manifest["model"],
            "prompt_id": classifier_manifest["prompt_id"],
            "prompt_version": classifier_manifest["prompt_version"],
            "prompt_hash": classifier_manifest["prompt_hash"],
            "proposition": classifier_manifest["proposition"],
            "expected_calls": plan.expected_classifier_calls,
            "status_counts": counts,
            "status_fractions": {
                status: count / count_total if count_total else 0.0 for status, count in counts.items()
            },
            "token_usage": classifier_manifest["token_usage"],
        },
        "temporal_signal": {
            "window": temporal.window, "cadence": temporal.cadence,
            "interval_policy": temporal.interval_policy,
            "start_time_utc": temporal.start_time_utc.isoformat(),
            "end_time_utc": temporal.end_time_utc.isoformat(),
            "distribution_policy_id": temporal.distribution_policy_id,
            "min_support_threshold": temporal.min_support_threshold,
            "enabled_measures": list(temporal.enabled_measures),
            "measure_parameters": measure_parameters,
            "tick_count": len(ticks), "tick_status_counts": dict(status_counts),
            "observation_count": len(observation_rows),
            "output_counts": {key: dict(value) for key, value in output_counts.items()},
        },
        "sanity_checks": sanity,
        "artifacts": {
            "comment_stances": STANCE_ARTIFACT,
            "classifier_manifest": CLASSIFIER_MANIFEST,
            "resolved_config": CLASSIFIER_RESOLVED_CONFIG,
            "opinion_distributions": DISTRIBUTION_ARTIFACT,
            "opinion_distributions_sha256": distribution_hash,
            "signal_observations": SIGNAL_ARTIFACT,
            "signal_observations_sha256": signal_hash,
            "polarization_detection_results": (
                DETECTION_ARTIFACT if detection_artifacts is not None else None
            ),
            "polarization_detection_manifest": (
                DETECTION_MANIFEST if detection_artifacts is not None else None
            ),
            "detection_alerts": (
                DETECTION_ALERT_ARTIFACT if detection_artifacts is not None else None
            ),
        },
        "downstream": {
            "detectors_executed": detection_artifacts is not None,
            "detection_results": (
                detection_artifacts.results_produced
                if detection_artifacts is not None else 0
            ),
            "detection_triggers": (
                detection_artifacts.trigger_count
                if detection_artifacts is not None else 0
            ),
            "detection_alerts": (
                detection_artifacts.alerts_produced
                if detection_artifacts is not None else 0
            ),
            "event_candidates_created": False,
            "association_executed": False, "rag_executed": False,
        },
    }
    with runtime_paths[2].open("x", encoding="utf-8") as target:
        target.write(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return PolarizationRuntimeArtifacts(
        actual.identity.run_id, artifacts.artifact_path, runtime_paths[0], runtime_paths[1],
        runtime_paths[2], artifacts.resolved_config_path, len(plan.comments), len(ticks),
        len(observation_rows),
        (
            detection_artifacts.result_path
            if detection_artifacts is not None else None
        ),
        (
            detection_artifacts.manifest_path
            if detection_artifacts is not None else None
        ),
        (
            detection_artifacts.results_produced
            if detection_artifacts is not None else 0
        ),
        (
            detection_artifacts.trigger_count
            if detection_artifacts is not None else 0
        ),
        (
            detection_artifacts.alert_path
            if detection_artifacts is not None else None
        ),
        (
            detection_artifacts.alerts_produced
            if detection_artifacts is not None else 0
        ),
    )


__all__ = [
    "DISTRIBUTION_ARTIFACT", "SIGNAL_ARTIFACT", "RUNTIME_MANIFEST",
    "PolarizationRuntimeArtifacts", "PolarizationRuntimePlan",
    "prepare_polarization_runtime", "run_polarization_runtime",
]
