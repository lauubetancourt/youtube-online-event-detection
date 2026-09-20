"""Opt-in causal input preparation from Silver for new experimental consumers.

This module deliberately performs no semantic filtering.  It provides one
shared canonical record universe plus narrow activity and stance adapters.
"""
from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from youtube_pipeline.stance_classifier import StanceCommentInput


CAUSAL_MINIMAL_POLICY_ID = "causal_minimal_v1"
CAUSAL_MINIMAL_SOURCE = "silver"
ORDERING_POLICY = "event_time_utc_then_comment_id"
TEXT_AUTHORITY = "original"
_SEMANTIC_IDENTITY_FIELDS = (
    "video_id",
    "author_id",
    "event_time_utc",
    "original_text",
    "parent_comment_id",
)


def _mapping_values(cls: type, payload: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError(f"{cls.__name__} must be an object.")
    unknown = set(payload) - {field.name for field in fields(cls)}
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} fields: {sorted(unknown)}")
    return dict(payload)


@dataclass(frozen=True, slots=True)
class ExperimentalInputPolicyConfig:
    """Fixed V1 policy switches, exposed so the effective policy is auditable."""

    policy_id: str = CAUSAL_MINIMAL_POLICY_ID
    source: str = CAUSAL_MINIMAL_SOURCE
    exact_comment_id_dedup: bool = True
    semantic_filtering: bool = False

    def __post_init__(self) -> None:
        if self.policy_id != CAUSAL_MINIMAL_POLICY_ID:
            raise ValueError(
                f"policy_id must be {CAUSAL_MINIMAL_POLICY_ID!r}."
            )
        if self.source != CAUSAL_MINIMAL_SOURCE:
            raise ValueError("causal_minimal_v1 requires source='silver'.")
        if type(self.exact_comment_id_dedup) is not bool:
            raise TypeError("exact_comment_id_dedup must be a boolean.")
        if not self.exact_comment_id_dedup:
            raise ValueError("causal_minimal_v1 requires exact comment_id deduplication.")
        if type(self.semantic_filtering) is not bool:
            raise TypeError("semantic_filtering must be a boolean.")
        if self.semantic_filtering:
            raise ValueError("causal_minimal_v1 does not permit semantic filtering.")

    @classmethod
    def from_mapping(
        cls, payload: Mapping[str, Any]
    ) -> ExperimentalInputPolicyConfig:
        return cls(**_mapping_values(cls, payload))


@dataclass(frozen=True, slots=True)
class InputRecordLineage:
    input_policy_id: str
    source_layer: str
    source_path: str
    source_sha256: str
    source_row_index: int


@dataclass(frozen=True, slots=True)
class CausalInputRecord:
    comment_id: str
    video_id: str
    author_id: str | None
    event_time_utc: datetime
    original_text: str
    parent_comment_id: str | None
    is_reply: bool
    quality_flags: tuple[str, ...]
    lineage: InputRecordLineage


@dataclass(frozen=True, slots=True)
class ActivityInputRecord:
    comment_id: str
    video_id: str
    author_id: str | None
    event_time_utc: datetime
    parent_comment_id: str | None
    is_reply: bool
    quality_flags: tuple[str, ...]
    lineage: InputRecordLineage

    def to_mapping(self) -> dict[str, Any]:
        return {
            "comment_id": self.comment_id,
            "video_id": self.video_id,
            "author_id": self.author_id,
            "event_time_utc": self.event_time_utc,
            "parent_comment_id": self.parent_comment_id,
            "is_reply": self.is_reply,
            "quality_flags": self.quality_flags,
            "input_policy_id": self.lineage.input_policy_id,
            "input_source_layer": self.lineage.source_layer,
            "input_source_path": self.lineage.source_path,
            "input_source_sha256": self.lineage.source_sha256,
            "input_source_row_index": self.lineage.source_row_index,
            "input_quality_flags": self.quality_flags,
        }


@dataclass(frozen=True, slots=True)
class StanceInputRecord:
    classifier_input: StanceCommentInput
    video_id: str
    quality_flags: tuple[str, ...]
    lineage: InputRecordLineage


@dataclass(frozen=True, slots=True)
class CausalPreparedBatch:
    dataframe: pd.DataFrame
    records: tuple[CausalInputRecord, ...]
    manifest: dict[str, Any]


class DuplicateCommentConflictError(ValueError):
    """One comment_id was reused with conflicting semantic source fields."""


def source_sha256(path: str | Path) -> str:
    """Hash a file or partitioned directory deterministically."""

    source = Path(path)
    digest = hashlib.sha256()
    if source.is_file():
        files = (source,)
        root = None
    elif source.is_dir():
        files = tuple(item for item in sorted(source.rglob("*")) if item.is_file())
        root = source
    else:
        raise FileNotFoundError(source)
    for item in files:
        if root is not None:
            relative = item.relative_to(root).as_posix().encode("utf-8")
            digest.update(len(relative).to_bytes(8, "big"))
            digest.update(relative)
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def read_input_table(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if source.is_dir() or source.suffix.lower() == ".parquet":
        return pd.read_parquet(source)
    if source.suffix.lower() == ".csv":
        return pd.read_csv(source)
    raise ValueError(f"Unsupported input dataset format: {source}")


def _is_missing(value: object) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _required_text(name: str, value: object) -> str:
    if _is_missing(value):
        raise ValueError(f"{name} must not be missing.")
    result = str(value)
    if not result.strip():
        raise ValueError(f"{name} must not be empty.")
    return result


def _optional_text(value: object) -> str | None:
    if value is None or _is_missing(value):
        return None
    result = str(value)
    return result if result.strip() else None


def _original_text(row: Mapping[str, Any]) -> str:
    if "original_text" in row:
        value = row["original_text"]
    elif "text" in row:
        value = row["text"]
    else:
        raise ValueError("Silver input requires text or original_text.")
    return "" if value is None or _is_missing(value) else str(value)


def _event_time(row: Mapping[str, Any]) -> datetime:
    if "event_time_utc" not in row:
        raise ValueError("Silver input requires event_time_utc.")
    timestamp = pd.to_datetime(row["event_time_utc"], utc=True, errors="coerce")
    if pd.isna(timestamp):
        raise ValueError("event_time_utc must be a valid timestamp.")
    return timestamp.to_pydatetime().astimezone(timezone.utc)


def _source_flags(row: Mapping[str, Any]) -> tuple[str, ...]:
    flags: list[str] = []
    probable_spam = row.get("is_probable_spam", False)
    if not _is_missing(probable_spam) and _boolean(probable_spam):
        flags.append("spam_like")
    for name in ("acquisition_warning", "temporal_warning"):
        value = _optional_text(row.get(name))
        if value is not None:
            flags.append(f"{name}:{value}")
    existing = row.get("quality_flags")
    if isinstance(existing, str) and existing.strip():
        flags.extend(part.strip() for part in existing.split(",") if part.strip())
    elif isinstance(existing, (list, tuple, set)):
        flags.extend(str(part) for part in existing if str(part).strip())
    return tuple(dict.fromkeys(flags))


def _boolean(value: object) -> bool:
    if value is None or _is_missing(value):
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no", ""}:
            return False
    raise TypeError("is_reply must be boolean-like.")


class CausalMinimalInputPolicy:
    """In-run causal deduplication state shared by batch and replay use."""

    def __init__(
        self,
        config: ExperimentalInputPolicyConfig,
        *,
        source_path: str | Path,
        source_hash: str | None = None,
    ) -> None:
        if not isinstance(config, ExperimentalInputPolicyConfig):
            raise TypeError("config must be ExperimentalInputPolicyConfig.")
        self.config = config
        self.source_path = str(Path(source_path))
        self.source_hash = source_hash or source_sha256(source_path)
        self._seen: dict[str, tuple[Any, ...]] = {}
        self._last_emitted_key: tuple[datetime, str] | None = None
        self.records_seen = 0
        self.records_emitted = 0
        self.exact_duplicates_suppressed = 0
        self.conflicts = 0
        self.future_records_withheld = 0

    @property
    def seen_comment_ids(self) -> frozenset[str]:
        return frozenset(self._seen)

    def _canonicalize(
        self, row: Mapping[str, Any], *, source_row_index: int
    ) -> CausalInputRecord:
        comment_id = _required_text("comment_id", row.get("comment_id"))
        video_id = _required_text("video_id", row.get("video_id"))
        parent_id = _optional_text(row.get("parent_comment_id"))
        if parent_id is None:
            parent_id = _optional_text(row.get("reply_to_comment_id"))
        is_reply = _boolean(row.get("is_reply", parent_id is not None))
        if parent_id == comment_id:
            raise ValueError("A comment cannot be its own parent.")
        return CausalInputRecord(
            comment_id=comment_id,
            video_id=video_id,
            author_id=_optional_text(row.get("author_id")),
            event_time_utc=_event_time(row),
            original_text=_original_text(row),
            parent_comment_id=parent_id,
            is_reply=is_reply,
            quality_flags=_source_flags(row),
            lineage=InputRecordLineage(
                input_policy_id=self.config.policy_id,
                source_layer=self.config.source,
                source_path=self.source_path,
                source_sha256=self.source_hash,
                source_row_index=source_row_index,
            ),
        )

    @staticmethod
    def _fingerprint(record: CausalInputRecord) -> tuple[Any, ...]:
        return tuple(getattr(record, field) for field in _SEMANTIC_IDENTITY_FIELDS)

    def process(
        self,
        row: Mapping[str, Any],
        *,
        source_row_index: int,
        data_cutoff_utc: datetime | None = None,
    ) -> CausalInputRecord | None:
        self.records_seen += 1
        record = self._canonicalize(row, source_row_index=source_row_index)
        if data_cutoff_utc is not None:
            cutoff = pd.Timestamp(data_cutoff_utc)
            if cutoff.tzinfo is None:
                raise ValueError("data_cutoff_utc must be timezone-aware.")
            if record.event_time_utc >= cutoff.tz_convert("UTC").to_pydatetime():
                self.future_records_withheld += 1
                return None

        fingerprint = self._fingerprint(record)
        previous = self._seen.get(record.comment_id)
        if previous is not None:
            if previous != fingerprint:
                self.conflicts += 1
                raise DuplicateCommentConflictError(
                    f"comment_id {record.comment_id!r} has conflicting semantic fields."
                )
            self.exact_duplicates_suppressed += 1
            return None

        ordering_key = (record.event_time_utc, record.comment_id)
        if self._last_emitted_key is not None and ordering_key < self._last_emitted_key:
            raise ValueError(
                "Causal input records must be processed in nondecreasing "
                "(event_time_utc, comment_id) order."
            )
        self._seen[record.comment_id] = fingerprint
        self._last_emitted_key = ordering_key
        self.records_emitted += 1
        return record

    def manifest(self) -> dict[str, Any]:
        joined_ids = "\n".join(sorted(self._seen)).encode("utf-8")
        return {
            "input_policy_id": self.config.policy_id,
            "source_layer": self.config.source,
            "source_path": self.source_path,
            "source_sha256": self.source_hash,
            "exact_comment_id_dedup": self.config.exact_comment_id_dedup,
            "semantic_filtering": self.config.semantic_filtering,
            "ordering_policy": ORDERING_POLICY,
            "text_authority": TEXT_AUTHORITY,
            "records_seen": self.records_seen,
            "records_emitted": self.records_emitted,
            "exact_duplicates_suppressed": self.exact_duplicates_suppressed,
            "conflicts": self.conflicts,
            "future_records_withheld": self.future_records_withheld,
            "canonical_comment_ids_sha256": hashlib.sha256(joined_ids).hexdigest(),
            "incremental_state": "in_run_seen_comment_ids",
        }


def prepare_causal_minimal_batch(
    dataframe: pd.DataFrame,
    config: ExperimentalInputPolicyConfig,
    *,
    source_path: str | Path,
    data_cutoff_utc: datetime | None = None,
) -> CausalPreparedBatch:
    """Apply the same state machine used by replay to one deterministic batch."""

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame.")
    required = {"comment_id", "video_id", "event_time_utc"}
    missing = sorted(required - set(dataframe.columns))
    if missing:
        raise ValueError(f"Silver input lacks required columns: {missing}")
    if not ({"text", "original_text"} & set(dataframe.columns)):
        raise ValueError("Silver input requires text or original_text.")

    working = dataframe.copy()
    working["_policy_source_row_index"] = range(len(working))
    parsed = pd.to_datetime(working["event_time_utc"], utc=True, errors="coerce")
    if parsed.isna().any():
        raise ValueError("Silver input contains invalid event_time_utc values.")
    working["event_time_utc"] = parsed
    if working["comment_id"].isna().any() or working["video_id"].isna().any():
        raise ValueError("Silver input requires non-null comment_id and video_id.")
    working["comment_id"] = working["comment_id"].astype(str)
    working["video_id"] = working["video_id"].astype(str)
    working = working.sort_values(
        ["event_time_utc", "comment_id", "_policy_source_row_index"], kind="stable"
    )

    policy = CausalMinimalInputPolicy(config, source_path=source_path)
    records: list[CausalInputRecord] = []
    emitted_indices: list[int] = []
    for row in working.to_dict(orient="records"):
        index = int(row["_policy_source_row_index"])
        record = policy.process(
            row, source_row_index=index, data_cutoff_utc=data_cutoff_utc
        )
        if record is not None:
            records.append(record)
            emitted_indices.append(index)

    emitted = dataframe.iloc[emitted_indices].copy()
    emitted["event_time_utc"] = [record.event_time_utc for record in records]
    emitted["input_policy_id"] = config.policy_id
    emitted["input_source_layer"] = config.source
    emitted["input_source_path"] = str(Path(source_path))
    emitted["input_source_sha256"] = policy.source_hash
    emitted["input_source_row_index"] = emitted_indices
    emitted["input_quality_flags"] = [record.quality_flags for record in records]
    emitted = emitted.sort_values(
        ["event_time_utc", "comment_id", "input_source_row_index"], kind="stable"
    ).reset_index(drop=True)
    return CausalPreparedBatch(emitted, tuple(records), policy.manifest())


def adapt_activity_records(
    records: Iterable[CausalInputRecord],
) -> tuple[ActivityInputRecord, ...]:
    return tuple(
        ActivityInputRecord(
            comment_id=record.comment_id,
            video_id=record.video_id,
            author_id=record.author_id,
            event_time_utc=record.event_time_utc,
            parent_comment_id=record.parent_comment_id,
            is_reply=record.is_reply,
            quality_flags=record.quality_flags,
            lineage=record.lineage,
        )
        for record in records
    )


def adapt_stance_records(
    records: Iterable[CausalInputRecord],
    *,
    selected_comment_ids: Iterable[str] | None = None,
) -> tuple[StanceInputRecord, ...]:
    canonical = tuple(records)
    lookup = {record.comment_id: record for record in canonical}
    selected = None if selected_comment_ids is None else set(selected_comment_ids)
    output: list[StanceInputRecord] = []
    for record in canonical:
        if selected is not None and record.comment_id not in selected:
            continue
        parent = lookup.get(record.parent_comment_id) if record.is_reply else None
        parent_is_causal = (
            parent is not None
            and parent.video_id == record.video_id
            and parent.event_time_utc <= record.event_time_utc
        )
        quality_flags = list(record.quality_flags)
        if record.is_reply and not parent_is_causal:
            quality_flags.append("missing_parent")
        classifier_input = StanceCommentInput(
            comment_id=record.comment_id,
            author_id=record.author_id,
            event_time_utc=record.event_time_utc,
            text=record.original_text,
            parent_comment_id=record.parent_comment_id if record.is_reply else None,
            parent_text=parent.original_text if parent_is_causal else None,
            parent_event_time_utc=parent.event_time_utc if parent_is_causal else None,
            is_deleted=False,
        )
        output.append(
            StanceInputRecord(
                classifier_input=classifier_input,
                video_id=record.video_id,
                quality_flags=tuple(dict.fromkeys(quality_flags)),
                lineage=record.lineage,
            )
        )
    return tuple(output)


__all__ = [
    "CAUSAL_MINIMAL_POLICY_ID",
    "CAUSAL_MINIMAL_SOURCE",
    "ORDERING_POLICY",
    "TEXT_AUTHORITY",
    "ActivityInputRecord",
    "CausalInputRecord",
    "CausalMinimalInputPolicy",
    "CausalPreparedBatch",
    "DuplicateCommentConflictError",
    "ExperimentalInputPolicyConfig",
    "InputRecordLineage",
    "StanceInputRecord",
    "adapt_activity_records",
    "adapt_stance_records",
    "prepare_causal_minimal_batch",
    "read_input_table",
    "source_sha256",
]
