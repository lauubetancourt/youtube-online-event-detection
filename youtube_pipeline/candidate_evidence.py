"""Causal internal-evidence inventories for neutral associated event candidates."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from youtube_pipeline.association_candidates import (
    READY_FOR_EVIDENCE,
    AssociatedEventCandidate,
)
from youtube_pipeline.experimental_input import (
    CAUSAL_MINIMAL_POLICY_ID,
    CausalInputRecord,
    ExperimentalInputPolicyConfig,
    prepare_causal_minimal_batch,
    read_input_table,
)
from youtube_pipeline.storage import canonicalize_comment_event_time
from youtube_pipeline.rag_sidecars import build_context_units, build_event_thread_map


CANDIDATE_EVIDENCE_POLICY_V1 = "candidate_evidence_v1"
IMMEDIATE_CAUSAL = "immediate_causal"
DELAYED_ENRICHMENT = "delayed_enrichment"
REFERENCE_TIME_ANCHOR = "reference_time_utc"
PARENT_POLICY = "exclude_outside_window_retain_relation_v1"
INTERVAL_POLICY = "closed_window_plus_strict_causal_cutoff_v1"
EVIDENCE_READY = "EVIDENCE_READY"
EVIDENCE_PENDING_HORIZON = "EVIDENCE_PENDING_HORIZON"
EVIDENCE_BLOCKED_MISSING_VIDEO_LINEAGE = "EVIDENCE_BLOCKED_MISSING_VIDEO_LINEAGE"
NO_EVIDENCE_FOUND = "NO_EVIDENCE_FOUND"
CANDIDATE_EVIDENCE_ARTIFACT_VERSION = "candidate_evidence_v1"

CANDIDATE_EVIDENCE_PACKAGES_FILE = "candidate_evidence_packages.jsonl"
CANDIDATE_COMMENT_INVENTORY_FILE = "candidate_comment_inventory.csv"
CANDIDATE_VIDEO_MAP_FILE = "candidate_video_map.csv"
CANDIDATE_THREAD_MAP_FILE = "candidate_thread_map.csv"
CANDIDATE_CONTEXT_UNITS_FILE = "candidate_rag_context_units.jsonl"
CANDIDATE_CONTEXT_COMMENT_MAP_FILE = "candidate_context_unit_comment_map.csv"
CANDIDATE_EVIDENCE_MANIFEST_FILE = "candidate_evidence_manifest.json"
RAG_COMPATIBILITY_DIR = "rag_compatibility"


def _mapping_values(cls: type, payload: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError(f"{cls.__name__} must be an object.")
    unknown = set(payload) - {field.name for field in fields(cls)}
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} fields: {sorted(unknown)}")
    return dict(payload)


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


def _duration(name: str, value: str) -> timedelta:
    _require_text(name, value)
    try:
        parsed = pd.to_timedelta(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a valid duration.") from exc
    if pd.isna(parsed) or parsed < pd.Timedelta(0):
        raise ValueError(f"{name} must be a non-negative duration.")
    return parsed.to_pytimedelta()


def _iso(value: object) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if value is pd.NA:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


@dataclass(frozen=True, slots=True)
class CandidateEvidencePolicyConfig:
    """Experimental temporal and source policy for candidate evidence."""

    source_path: str
    output_dir: str
    lookback: str
    lookahead: str
    mode: str
    parameter_status: str
    policy_id: str = CANDIDATE_EVIDENCE_POLICY_V1
    time_anchor: str = REFERENCE_TIME_ANCHOR
    source_layer: str = "silver"
    input_policy_id: str = CAUSAL_MINIMAL_POLICY_ID
    max_comments_per_context_unit: int = 25

    def __post_init__(self) -> None:
        _require_text("source_path", self.source_path)
        _require_text("output_dir", self.output_dir)
        _duration("lookback", self.lookback)
        _duration("lookahead", self.lookahead)
        if self.mode not in {IMMEDIATE_CAUSAL, DELAYED_ENRICHMENT}:
            raise ValueError(
                "mode must be immediate_causal or delayed_enrichment."
            )
        if self.parameter_status not in {
            "TEST_ONLY",
            "DEVELOPMENT_NOT_CALIBRATED",
        }:
            raise ValueError(
                "parameter_status must be TEST_ONLY or "
                "DEVELOPMENT_NOT_CALIBRATED."
            )
        if self.policy_id != CANDIDATE_EVIDENCE_POLICY_V1:
            raise ValueError(
                f"policy_id must be {CANDIDATE_EVIDENCE_POLICY_V1!r}."
            )
        if self.time_anchor != REFERENCE_TIME_ANCHOR:
            raise ValueError(
                f"time_anchor must be {REFERENCE_TIME_ANCHOR!r} in V1."
            )
        if self.source_layer != "silver":
            raise ValueError("candidate_evidence_v1 requires source_layer='silver'.")
        if self.input_policy_id != CAUSAL_MINIMAL_POLICY_ID:
            raise ValueError(
                "candidate_evidence_v1 requires input_policy_id='causal_minimal_v1'."
            )
        if (
            isinstance(self.max_comments_per_context_unit, bool)
            or not isinstance(self.max_comments_per_context_unit, int)
            or self.max_comments_per_context_unit < 1
        ):
            raise ValueError("max_comments_per_context_unit must be a positive integer.")

    @property
    def lookback_delta(self) -> timedelta:
        return _duration("lookback", self.lookback)

    @property
    def lookahead_delta(self) -> timedelta:
        return _duration("lookahead", self.lookahead)

    @classmethod
    def from_mapping(
        cls, payload: Mapping[str, Any]
    ) -> CandidateEvidencePolicyConfig:
        return cls(**_mapping_values(cls, payload))


@dataclass(frozen=True, slots=True)
class CandidateEvidenceScope:
    candidate_id: str
    status: str
    policy_id: str
    mode: str
    time_anchor: str
    anchor_time_utc: datetime
    requested_window_start_utc: datetime
    requested_window_end_utc: datetime
    evidence_window_start_utc: datetime
    evidence_window_end_utc: datetime
    evidence_cutoff_utc: datetime
    evidence_ready_at_utc: datetime
    window_clipped_by_causality: bool
    associated_video_ids: tuple[str, ...]
    interval_policy: str = INTERVAL_POLICY
    parent_policy: str = PARENT_POLICY

    def to_mapping(self) -> dict[str, Any]:
        payload = asdict(self)
        for name in (
            "anchor_time_utc",
            "requested_window_start_utc",
            "requested_window_end_utc",
            "evidence_window_start_utc",
            "evidence_window_end_utc",
            "evidence_cutoff_utc",
            "evidence_ready_at_utc",
        ):
            payload[name] = getattr(self, name).isoformat()
        payload["associated_video_ids"] = list(self.associated_video_ids)
        return payload


def derive_candidate_evidence_scope(
    candidate: AssociatedEventCandidate,
    config: CandidateEvidencePolicyConfig,
    *,
    current_data_cutoff_utc: datetime | None = None,
) -> CandidateEvidenceScope:
    """Derive evidence bounds without modifying or reinterpreting a candidate."""

    if not isinstance(candidate, AssociatedEventCandidate):
        raise TypeError("candidate must be an AssociatedEventCandidate.")
    if candidate.status != READY_FOR_EVIDENCE:
        raise ValueError("candidate must have status READY_FOR_EVIDENCE.")
    if not isinstance(config, CandidateEvidencePolicyConfig):
        raise TypeError("config must be CandidateEvidencePolicyConfig.")
    if candidate.input_policy_id != config.input_policy_id:
        raise ValueError(
            "Candidate input policy does not match the evidence canonical source."
        )

    anchor = candidate.reference_time_utc
    requested_start = anchor - config.lookback_delta
    requested_end = anchor + config.lookahead_delta
    if config.mode == IMMEDIATE_CAUSAL:
        evidence_cutoff = candidate.created_at_cutoff_utc
        ready_at = candidate.created_at_cutoff_utc
        clipped = requested_end >= evidence_cutoff
        window_end = min(requested_end, evidence_cutoff)
        status = EVIDENCE_READY
    else:
        if current_data_cutoff_utc is None:
            raise ValueError(
                "delayed_enrichment requires current_data_cutoff_utc."
            )
        evidence_cutoff = _require_utc(
            "current_data_cutoff_utc", current_data_cutoff_utc
        )
        if evidence_cutoff < candidate.created_at_cutoff_utc:
            raise ValueError("Evidence cutoff cannot precede candidate creation.")
        ready_at = requested_end
        clipped = False
        window_end = requested_end
        status = (
            EVIDENCE_PENDING_HORIZON
            if evidence_cutoff <= requested_end
            else EVIDENCE_READY
        )
    if not candidate.associated_video_ids:
        status = EVIDENCE_BLOCKED_MISSING_VIDEO_LINEAGE
    return CandidateEvidenceScope(
        candidate_id=candidate.candidate_id,
        status=status,
        policy_id=config.policy_id,
        mode=config.mode,
        time_anchor=config.time_anchor,
        anchor_time_utc=anchor,
        requested_window_start_utc=requested_start,
        requested_window_end_utc=requested_end,
        evidence_window_start_utc=requested_start,
        evidence_window_end_utc=window_end,
        evidence_cutoff_utc=evidence_cutoff,
        evidence_ready_at_utc=ready_at,
        window_clipped_by_causality=clipped,
        associated_video_ids=candidate.associated_video_ids,
    )


_INVENTORY_COLUMNS = [
    "candidate_id",
    "run_id",
    "candidate_type",
    "association_id",
    "config_hash",
    "reference_time_utc",
    "trigger_time",
    "trigger_time_utc",
    "evidence_window_start_utc",
    "evidence_window_end_utc",
    "evidence_cutoff_utc",
    "window_start",
    "window_start_utc",
    "window_end",
    "window_end_utc",
    "order_in_candidate",
    "order_in_candidate_video",
    "event_time_utc",
    "video_id",
    "comment_id",
    "root_comment_id",
    "parent_comment_id",
    "is_reply",
    "parent_in_inventory",
    "author_id",
    "text",
    "original_text",
    "text_clean",
    "classification_status",
    "quality_flags",
    "input_policy_id",
    "input_source_layer",
    "input_source_path",
    "input_source_sha256",
    "input_source_row_index",
    "artifact_version",
]

_VIDEO_MAP_COLUMNS = [
    "candidate_id",
    "run_id",
    "candidate_type",
    "video_id",
    "inventory_comment_count",
    "reply_count",
    "root_comment_count",
    "first_comment_time_utc",
    "last_comment_time_utc",
    "artifact_version",
]

_THREAD_MAP_COLUMNS = [
    "candidate_id",
    "run_id",
    "trigger_time",
    "window_start",
    "window_end",
    "video_id",
    "root_comment_id",
    "root_comment_present",
    "comment_count",
    "reply_count",
    "has_replies",
    "first_comment_time_utc",
    "last_comment_time_utc",
    "comment_ids",
    "artifact_version",
]

_CONTEXT_MAP_COLUMNS = [
    "context_unit_id",
    "candidate_id",
    "run_id",
    "trigger_time",
    "video_id",
    "window_start",
    "window_end",
    "context_type",
    "order_in_context_unit",
    "comment_id",
    "root_comment_id",
    "event_time_utc",
    "is_reply",
    "artifact_version",
]


def _metadata_by_comment(batch: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if batch.empty:
        return {}
    return {
        str(row["comment_id"]): row
        for row in batch.to_dict(orient="records")
    }


def _inventory_for_candidate(
    candidate: AssociatedEventCandidate,
    scope: CandidateEvidenceScope,
    records: Iterable[CausalInputRecord],
    metadata: Mapping[str, Mapping[str, Any]],
) -> pd.DataFrame:
    if scope.status != EVIDENCE_READY:
        return pd.DataFrame(columns=_INVENTORY_COLUMNS)
    videos = set(scope.associated_video_ids)
    selected = [
        record
        for record in records
        if record.video_id in videos
        and scope.evidence_window_start_utc
        <= record.event_time_utc
        <= scope.evidence_window_end_utc
        and record.event_time_utc < scope.evidence_cutoff_utc
    ]
    selected.sort(key=lambda item: (item.event_time_utc, item.comment_id))
    selected_ids = {record.comment_id for record in selected}
    video_orders: dict[str, int] = {}
    rows: list[dict[str, Any]] = []
    for order, record in enumerate(selected, start=1):
        video_orders[record.video_id] = video_orders.get(record.video_id, 0) + 1
        source = metadata.get(record.comment_id, {})
        parent_id = record.parent_comment_id if record.is_reply else None
        rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "run_id": candidate.run_id,
                "candidate_type": candidate.candidate_type,
                "association_id": candidate.association_id,
                "config_hash": candidate.config_hash,
                "reference_time_utc": candidate.reference_time_utc,
                "trigger_time": candidate.reference_time_utc,
                "trigger_time_utc": candidate.reference_time_utc,
                "evidence_window_start_utc": scope.evidence_window_start_utc,
                "evidence_window_end_utc": scope.evidence_window_end_utc,
                "evidence_cutoff_utc": scope.evidence_cutoff_utc,
                "window_start": scope.evidence_window_start_utc,
                "window_start_utc": scope.evidence_window_start_utc,
                "window_end": scope.evidence_window_end_utc,
                "window_end_utc": scope.evidence_window_end_utc,
                "order_in_candidate": order,
                "order_in_candidate_video": video_orders[record.video_id],
                "event_time_utc": record.event_time_utc,
                "video_id": record.video_id,
                "comment_id": record.comment_id,
                "root_comment_id": parent_id or record.comment_id,
                "parent_comment_id": parent_id,
                "is_reply": record.is_reply,
                "parent_in_inventory": bool(parent_id in selected_ids),
                "author_id": record.author_id,
                "text": record.original_text,
                "original_text": record.original_text,
                "text_clean": source.get("text_clean"),
                "classification_status": source.get("classification_status"),
                "quality_flags": json.dumps(
                    list(record.quality_flags), ensure_ascii=False
                ),
                "input_policy_id": record.lineage.input_policy_id,
                "input_source_layer": record.lineage.source_layer,
                "input_source_path": record.lineage.source_path,
                "input_source_sha256": record.lineage.source_sha256,
                "input_source_row_index": record.lineage.source_row_index,
                "artifact_version": CANDIDATE_EVIDENCE_ARTIFACT_VERSION,
            }
        )
    return pd.DataFrame(rows, columns=_INVENTORY_COLUMNS)


def _video_map_for_candidate(
    candidate: AssociatedEventCandidate,
    inventory: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for video_id in candidate.associated_video_ids:
        subset = inventory.loc[inventory["video_id"].astype(str) == video_id]
        rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "run_id": candidate.run_id,
                "candidate_type": candidate.candidate_type,
                "video_id": video_id,
                "inventory_comment_count": int(subset["comment_id"].nunique()),
                "reply_count": int(subset["is_reply"].sum()),
                "root_comment_count": int(
                    subset.loc[~subset["is_reply"], "comment_id"].nunique()
                ),
                "first_comment_time_utc": (
                    subset["event_time_utc"].min() if not subset.empty else None
                ),
                "last_comment_time_utc": (
                    subset["event_time_utc"].max() if not subset.empty else None
                ),
                "artifact_version": CANDIDATE_EVIDENCE_ARTIFACT_VERSION,
            }
        )
    return pd.DataFrame(rows, columns=_VIDEO_MAP_COLUMNS)


def _legacy_inventory_alias(inventory: pd.DataFrame) -> pd.DataFrame:
    alias = inventory.rename(columns={"candidate_id": "event_id"}).copy()
    return alias


def _thread_map(inventory: pd.DataFrame) -> pd.DataFrame:
    if inventory.empty:
        return pd.DataFrame(columns=_THREAD_MAP_COLUMNS)
    result = build_event_thread_map(_legacy_inventory_alias(inventory))
    result = result.rename(columns={"event_id": "candidate_id"})
    result["artifact_version"] = CANDIDATE_EVIDENCE_ARTIFACT_VERSION
    return result.reindex(columns=_THREAD_MAP_COLUMNS)


def _context_artifacts(
    inventory: pd.DataFrame,
    *,
    source_inventory_path: str | Path,
    max_comments_per_context_unit: int,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    if inventory.empty:
        return [], pd.DataFrame(columns=_CONTEXT_MAP_COLUMNS)
    units, comment_map = build_context_units(
        _legacy_inventory_alias(inventory),
        source_inventory_path=source_inventory_path,
        max_comments_per_context_unit=max_comments_per_context_unit,
    )
    candidate_units: list[dict[str, Any]] = []
    for unit in units:
        converted = dict(unit)
        converted["candidate_id"] = converted.pop("event_id")
        converted["artifact_version"] = CANDIDATE_EVIDENCE_ARTIFACT_VERSION
        candidate_units.append(converted)
    converted_map = comment_map.rename(columns={"event_id": "candidate_id"})
    if not converted_map.empty:
        converted_map["artifact_version"] = CANDIDATE_EVIDENCE_ARTIFACT_VERSION
    return candidate_units, converted_map.reindex(columns=_CONTEXT_MAP_COLUMNS)


def _artifact_paths(root: Path) -> dict[str, Path]:
    return {
        "candidate_evidence_packages": root / CANDIDATE_EVIDENCE_PACKAGES_FILE,
        "candidate_comment_inventory": root / CANDIDATE_COMMENT_INVENTORY_FILE,
        "candidate_video_map": root / CANDIDATE_VIDEO_MAP_FILE,
        "candidate_thread_map": root / CANDIDATE_THREAD_MAP_FILE,
        "candidate_context_units": root / CANDIDATE_CONTEXT_UNITS_FILE,
        "candidate_context_comment_map": root / CANDIDATE_CONTEXT_COMMENT_MAP_FILE,
        "candidate_evidence_manifest": root / CANDIDATE_EVIDENCE_MANIFEST_FILE,
    }


def _write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    with path.open("x", encoding="utf-8") as handle:
        for record in records:
            encoded = json.dumps(
                _json_safe(dict(record)),
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            ) + "\n"
            handle.write(encoded)
            digest.update(encoded.encode("utf-8"))
    return digest.hexdigest()


def _write_csv(path: Path, dataframe: pd.DataFrame) -> str:
    serializable = dataframe.copy()
    for column in serializable.columns:
        if pd.api.types.is_datetime64_any_dtype(serializable[column]):
            serializable[column] = serializable[column].map(_iso)
        elif serializable[column].dtype == object:
            serializable[column] = serializable[column].map(
                lambda value: _iso(value)
                if isinstance(value, (datetime, pd.Timestamp))
                else value
            )
    serializable.to_csv(path, index=False)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _package(
    candidate: AssociatedEventCandidate,
    scope: CandidateEvidenceScope,
    inventory: pd.DataFrame,
    video_map: pd.DataFrame,
    thread_map: pd.DataFrame,
    context_units: list[dict[str, Any]],
    *,
    config: CandidateEvidencePolicyConfig,
    source_hash: str,
    canonical_comment_ids_hash: str,
) -> dict[str, Any]:
    status = scope.status
    if status == EVIDENCE_READY and inventory.empty:
        status = NO_EVIDENCE_FOUND
    return {
        "candidate_id": candidate.candidate_id,
        "candidate_type": candidate.candidate_type,
        "run_id": candidate.run_id,
        "candidate_config_hash": candidate.config_hash,
        "association_id": candidate.association_id,
        "association_policy_id": candidate.association_policy_id,
        "association_tolerance": candidate.association_tolerance,
        "candidate_start_time_utc": candidate.start_time_utc,
        "candidate_end_time_utc": candidate.end_time_utc,
        "reference_time_utc": candidate.reference_time_utc,
        "candidate_created_at_cutoff_utc": candidate.created_at_cutoff_utc,
        "evidence_status": status,
        "evidence_policy": {
            "policy_id": config.policy_id,
            "mode": config.mode,
            "time_anchor": config.time_anchor,
            "lookback": config.lookback,
            "lookahead": config.lookahead,
            "parameter_status": config.parameter_status,
            "interval_policy": INTERVAL_POLICY,
            "parent_policy": PARENT_POLICY,
        },
        "evidence_scope": scope.to_mapping(),
        "associated_video_ids": list(candidate.associated_video_ids),
        "comment_count": int(inventory["comment_id"].nunique()),
        "root_count": int((~inventory["is_reply"]).sum()),
        "reply_count": int(inventory["is_reply"].sum()),
        "unique_author_count": int(inventory["author_id"].nunique()),
        "video_count": int(video_map["video_id"].nunique()),
        "thread_count": int(len(thread_map)),
        "context_unit_count": int(len(context_units)),
        "source": {
            "layer": config.source_layer,
            "path": Path(config.source_path).as_posix(),
            "sha256": source_hash,
            "input_policy_id": config.input_policy_id,
            "canonical_comment_ids_sha256": canonical_comment_ids_hash,
        },
        "artifacts": {
            "comment_inventory": CANDIDATE_COMMENT_INVENTORY_FILE,
            "video_map": CANDIDATE_VIDEO_MAP_FILE,
            "thread_map": CANDIDATE_THREAD_MAP_FILE,
            "context_units": CANDIDATE_CONTEXT_UNITS_FILE,
            "context_comment_map": CANDIDATE_CONTEXT_COMMENT_MAP_FILE,
            "manifest": CANDIDATE_EVIDENCE_MANIFEST_FILE,
        },
        "limitations": ["REPLY_COMPLETENESS_OPEN"],
        "rag_executed": False,
        "candidate_validated": False,
        "artifact_version": CANDIDATE_EVIDENCE_ARTIFACT_VERSION,
    }


def _compatibility_frame(dataframe: pd.DataFrame) -> pd.DataFrame:
    return dataframe.rename(columns={"candidate_id": "event_id"}).copy()


def write_rag_compatibility_sidecars(
    *,
    root: str | Path,
    packages: list[dict[str, Any]],
    inventory: pd.DataFrame,
    video_map: pd.DataFrame,
    thread_map: pd.DataFrame,
    context_units: list[dict[str, Any]],
    context_comment_map: pd.DataFrame,
    run_id: str,
) -> Path:
    """Write an explicit event_id alias view for the unchanged RAG consumer."""

    from youtube_pipeline.rag_consumer import SIDECAR_README_FILE
    from youtube_pipeline.rag_sidecars import (
        CONTEXT_SELECTION_MANIFEST_FILE,
        CONTEXT_UNIT_COMMENT_MAP_FILE,
        EVENT_COMMENT_INVENTORY_FILE,
        EVENT_EVIDENCE_PACKAGES_FILE,
        EVENT_THREAD_MAP_FILE,
        EVENT_VIDEO_MAP_FILE,
        RAG_CONTEXT_UNITS_FILE,
    )

    target = Path(root)
    target.mkdir(parents=True, exist_ok=False)
    compatible_packages: list[dict[str, Any]] = []
    for package in packages:
        scope = package["evidence_scope"]
        compatible_packages.append(
            {
                **package,
                "event_id": package["candidate_id"],
                "detector_name": "associated_detection",
                "trigger_time_utc": package["reference_time_utc"],
                "window_start_utc": scope["evidence_window_start_utc"],
                "window_end_utc": scope["evidence_window_end_utc"],
            }
        )
    compatible_inventory = _compatibility_frame(inventory)
    compatible_videos = _compatibility_frame(video_map)
    compatible_threads = _compatibility_frame(thread_map)
    compatible_context_map = _compatibility_frame(context_comment_map)
    compatible_units = []
    for unit in context_units:
        converted = dict(unit)
        converted["event_id"] = converted.pop("candidate_id")
        compatible_units.append(converted)

    _write_jsonl(target / EVENT_EVIDENCE_PACKAGES_FILE, compatible_packages)
    _write_csv(target / EVENT_COMMENT_INVENTORY_FILE, compatible_inventory)
    _write_csv(target / EVENT_VIDEO_MAP_FILE, compatible_videos)
    _write_csv(target / EVENT_THREAD_MAP_FILE, compatible_threads)
    _write_jsonl(target / RAG_CONTEXT_UNITS_FILE, compatible_units)
    _write_csv(target / CONTEXT_UNIT_COMMENT_MAP_FILE, compatible_context_map)
    manifest = {
        "run_id": run_id,
        "artifact_version": "candidate_to_legacy_rag_compatibility_v1",
        "mode": "explicit_candidate_id_to_event_id_alias_no_rag_execution",
        "canonical_join_key": "candidate_id",
        "compatibility_join_key": "event_id",
        "does_not_modify_legacy_sidecars": True,
        "does_not_call_llm": True,
        "counts": {
            "candidate_count": len(packages),
            "inventory_rows": len(inventory),
            "context_unit_count": len(context_units),
        },
    }
    (target / CONTEXT_SELECTION_MANIFEST_FILE).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (target / SIDECAR_README_FILE).write_text(
        "# Candidate evidence RAG compatibility view\n\n"
        "This directory aliases canonical `candidate_id` values to legacy `event_id` "
        "only for the unchanged non-generative RAG consumer.\n",
        encoding="utf-8",
    )
    return target


@dataclass(frozen=True, slots=True)
class CandidateEvidenceArtifacts:
    output_dir: Path
    package_path: Path
    manifest_path: Path
    rag_compatibility_dir: Path
    candidate_count: int
    evidence_ready: int
    no_evidence: int
    blocked: int
    pending: int
    inventory_rows: int


def build_candidate_evidence(
    *,
    candidates: Iterable[AssociatedEventCandidate],
    config: CandidateEvidencePolicyConfig,
    current_data_cutoff_utc: datetime | None = None,
) -> CandidateEvidenceArtifacts:
    """Build causal, complete internal evidence inventories without running RAG."""

    if not isinstance(config, CandidateEvidencePolicyConfig):
        raise TypeError("config must be CandidateEvidencePolicyConfig.")
    candidate_rows = list(candidates)
    for candidate in candidate_rows:
        if not isinstance(candidate, AssociatedEventCandidate):
            raise TypeError("candidates must contain AssociatedEventCandidate values.")
    candidate_ids = [item.candidate_id for item in candidate_rows]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("candidates must not contain duplicate candidate IDs.")
    run_ids = {item.run_id for item in candidate_rows}
    if len(run_ids) > 1:
        raise ValueError("Candidate evidence V1 requires one run_id per execution.")

    scopes = {
        candidate.candidate_id: derive_candidate_evidence_scope(
            candidate,
            config,
            current_data_cutoff_utc=current_data_cutoff_utc,
        )
        for candidate in candidate_rows
    }
    cutoffs = [scope.evidence_cutoff_utc for scope in scopes.values()]
    if current_data_cutoff_utc is not None:
        cutoffs.append(_require_utc("current_data_cutoff_utc", current_data_cutoff_utc))
    if not cutoffs:
        raise ValueError("At least one candidate or current cutoff is required.")
    source_cutoff = max(cutoffs)
    source = canonicalize_comment_event_time(read_input_table(config.source_path))
    prepared = prepare_causal_minimal_batch(
        source,
        ExperimentalInputPolicyConfig(),
        source_path=config.source_path,
        data_cutoff_utc=source_cutoff,
    )
    metadata = _metadata_by_comment(prepared.dataframe)

    all_inventory: list[pd.DataFrame] = []
    all_videos: list[pd.DataFrame] = []
    all_threads: list[pd.DataFrame] = []
    all_context_units: list[dict[str, Any]] = []
    all_context_maps: list[pd.DataFrame] = []
    packages: list[dict[str, Any]] = []
    root = Path(config.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    paths = _artifact_paths(root)
    if any(path.exists() for path in paths.values()) or (root / RAG_COMPATIBILITY_DIR).exists():
        raise FileExistsError("Candidate evidence outputs already exist; use a fresh output.")

    for candidate in sorted(
        candidate_rows,
        key=lambda item: (item.reference_time_utc, item.candidate_id),
    ):
        scope = scopes[candidate.candidate_id]
        inventory = _inventory_for_candidate(
            candidate,
            scope,
            prepared.records,
            metadata,
        )
        video_map = _video_map_for_candidate(candidate, inventory)
        thread_map = _thread_map(inventory)
        context_units, context_map = _context_artifacts(
            inventory,
            source_inventory_path=CANDIDATE_COMMENT_INVENTORY_FILE,
            max_comments_per_context_unit=config.max_comments_per_context_unit,
        )
        all_inventory.append(inventory)
        all_videos.append(video_map)
        all_threads.append(thread_map)
        all_context_units.extend(context_units)
        all_context_maps.append(context_map)
        packages.append(
            _package(
                candidate,
                scope,
                inventory,
                video_map,
                thread_map,
                context_units,
                config=config,
                source_hash=prepared.manifest["source_sha256"],
                canonical_comment_ids_hash=prepared.manifest[
                    "canonical_comment_ids_sha256"
                ],
            )
        )

    inventory = (
        pd.concat(all_inventory, ignore_index=True)
        if all_inventory
        else pd.DataFrame(columns=_INVENTORY_COLUMNS)
    )
    video_map = (
        pd.concat(all_videos, ignore_index=True)
        if all_videos
        else pd.DataFrame(columns=_VIDEO_MAP_COLUMNS)
    )
    thread_map = (
        pd.concat(all_threads, ignore_index=True)
        if all_threads
        else pd.DataFrame(columns=_THREAD_MAP_COLUMNS)
    )
    context_map = (
        pd.concat(all_context_maps, ignore_index=True)
        if all_context_maps
        else pd.DataFrame(columns=_CONTEXT_MAP_COLUMNS)
    )
    cutoff_by_candidate = {
        candidate_id: scope.evidence_cutoff_utc
        for candidate_id, scope in scopes.items()
    }
    future_leak_count = sum(
        pd.Timestamp(row.event_time_utc).to_pydatetime()
        >= cutoff_by_candidate[str(row.candidate_id)]
        for row in inventory.itertuples()
    )
    if future_leak_count:
        raise AssertionError("Candidate evidence inventory contains future records.")
    package_hash = _write_jsonl(paths["candidate_evidence_packages"], packages)
    artifact_hashes = {
        "candidate_evidence_packages_sha256": package_hash,
        "candidate_comment_inventory_sha256": _write_csv(
            paths["candidate_comment_inventory"], inventory
        ),
        "candidate_video_map_sha256": _write_csv(
            paths["candidate_video_map"], video_map
        ),
        "candidate_thread_map_sha256": _write_csv(
            paths["candidate_thread_map"], thread_map
        ),
        "candidate_context_units_sha256": _write_jsonl(
            paths["candidate_context_units"], all_context_units
        ),
        "candidate_context_comment_map_sha256": _write_csv(
            paths["candidate_context_comment_map"], context_map
        ),
    }
    run_id = next(iter(run_ids), "candidate_evidence_empty_run")
    compatibility_dir = write_rag_compatibility_sidecars(
        root=root / RAG_COMPATIBILITY_DIR,
        packages=packages,
        inventory=inventory,
        video_map=video_map,
        thread_map=thread_map,
        context_units=all_context_units,
        context_comment_map=context_map,
        run_id=run_id,
    )
    status_counts = {
        name: sum(package["evidence_status"] == name for package in packages)
        for name in (
            EVIDENCE_READY,
            NO_EVIDENCE_FOUND,
            EVIDENCE_BLOCKED_MISSING_VIDEO_LINEAGE,
            EVIDENCE_PENDING_HORIZON,
        )
    }
    manifest = {
        "artifact_version": CANDIDATE_EVIDENCE_ARTIFACT_VERSION,
        "status": "executed",
        "run_id": run_id,
        "evidence_policy": {
            "policy_id": config.policy_id,
            "mode": config.mode,
            "time_anchor": config.time_anchor,
            "lookback": config.lookback,
            "lookahead": config.lookahead,
            "parameter_status": config.parameter_status,
            "interval_policy": INTERVAL_POLICY,
            "parent_policy": PARENT_POLICY,
            "candidate_type_specific_branching": False,
        },
        "source": prepared.manifest,
        "candidate_lineage": [
            {
                "candidate_id": item.candidate_id,
                "run_id": item.run_id,
                "config_hash": item.config_hash,
                "association_id": item.association_id,
                "association_policy_id": item.association_policy_id,
                "association_tolerance": item.association_tolerance,
            }
            for item in candidate_rows
        ],
        "counts": {
            "candidates": len(candidate_rows),
            "evidence_ready": status_counts[EVIDENCE_READY],
            "no_evidence_found": status_counts[NO_EVIDENCE_FOUND],
            "blocked_missing_video_lineage": status_counts[
                EVIDENCE_BLOCKED_MISSING_VIDEO_LINEAGE
            ],
            "pending_evidence_horizon": status_counts[EVIDENCE_PENDING_HORIZON],
            "inventory_rows": len(inventory),
            "unique_comments": int(inventory["comment_id"].nunique()),
            "video_map_rows": len(video_map),
            "thread_map_rows": len(thread_map),
            "context_units": len(all_context_units),
            "context_comment_rows": len(context_map),
        },
        "causality": {
            "comment_rule": (
                "video_id in candidate videos AND window_start <= event_time <= "
                "window_end AND event_time < evidence_cutoff"
            ),
            "future_leak_count": future_leak_count,
            "causal_clipping_is_explicit": True,
        },
        "artifacts": {
            name: path.name for name, path in paths.items()
            if name != "candidate_evidence_manifest"
        },
        "artifact_hashes": artifact_hashes,
        "rag_compatibility": {
            "path": compatibility_dir.name,
            "strategy": "explicit_candidate_id_to_event_id_alias",
            "legacy_sidecars_modified": False,
        },
        "limitations": {
            "reply_completeness": "OPEN",
            "external_evidence": "NOT_EXECUTED",
        },
        "downstream": {
            "rag_consumer_executed": False,
            "g1_executed": False,
            "g2_executed": False,
            "candidate_validated": False,
        },
    }
    paths["candidate_evidence_manifest"].write_text(
        json.dumps(_json_safe(manifest), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return CandidateEvidenceArtifacts(
        output_dir=root,
        package_path=paths["candidate_evidence_packages"],
        manifest_path=paths["candidate_evidence_manifest"],
        rag_compatibility_dir=compatibility_dir,
        candidate_count=len(candidate_rows),
        evidence_ready=status_counts[EVIDENCE_READY],
        no_evidence=status_counts[NO_EVIDENCE_FOUND],
        blocked=status_counts[EVIDENCE_BLOCKED_MISSING_VIDEO_LINEAGE],
        pending=status_counts[EVIDENCE_PENDING_HORIZON],
        inventory_rows=len(inventory),
    )


__all__ = [
    "CANDIDATE_COMMENT_INVENTORY_FILE",
    "CANDIDATE_CONTEXT_COMMENT_MAP_FILE",
    "CANDIDATE_CONTEXT_UNITS_FILE",
    "CANDIDATE_EVIDENCE_ARTIFACT_VERSION",
    "CANDIDATE_EVIDENCE_MANIFEST_FILE",
    "CANDIDATE_EVIDENCE_PACKAGES_FILE",
    "CANDIDATE_EVIDENCE_POLICY_V1",
    "CANDIDATE_THREAD_MAP_FILE",
    "CANDIDATE_VIDEO_MAP_FILE",
    "CandidateEvidenceArtifacts",
    "CandidateEvidencePolicyConfig",
    "CandidateEvidenceScope",
    "DELAYED_ENRICHMENT",
    "EVIDENCE_BLOCKED_MISSING_VIDEO_LINEAGE",
    "EVIDENCE_PENDING_HORIZON",
    "EVIDENCE_READY",
    "IMMEDIATE_CAUSAL",
    "INTERVAL_POLICY",
    "NO_EVIDENCE_FOUND",
    "PARENT_POLICY",
    "REFERENCE_TIME_ANCHOR",
    "build_candidate_evidence",
    "derive_candidate_evidence_scope",
    "write_rag_compatibility_sidecars",
]
