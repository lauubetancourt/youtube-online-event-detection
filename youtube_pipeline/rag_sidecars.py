from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


RAG_SIDECAR_ARTIFACT_VERSION = "rag_sidecars_v1"

EVENT_EVIDENCE_PACKAGES_FILE = "event_evidence_packages.jsonl"
EVENT_COMMENT_INVENTORY_FILE = "event_comment_inventory.csv"
EVENT_VIDEO_MAP_FILE = "event_video_map.csv"
EVENT_THREAD_MAP_FILE = "event_thread_map.csv"
RAG_CONTEXT_UNITS_FILE = "rag_context_units.jsonl"
CONTEXT_UNIT_COMMENT_MAP_FILE = "context_unit_comment_map.csv"
CONTEXT_SELECTION_MANIFEST_FILE = "context_selection_manifest.json"

TRIGGER_COMMENT_REQUIRED_COLUMNS = {
    "trigger_time",
    "window_start",
    "window_end",
    "trigger_volume",
    "event_time_utc",
    "video_id",
    "comment_id",
    "text",
}

GOLD_COMMENT_REQUIRED_COLUMNS = {
    "event_time_utc",
    "video_id",
    "comment_id",
    "text",
}


@dataclass(frozen=True)
class RagSidecarBuildConfig:
    trigger_comment_map_path: str
    output_dir: str
    comments_path: str = "data/gold/clean_comments.parquet"
    snapshots_path: str | None = None
    detector_name: str = "xiao_ema"
    run_id: str | None = None
    max_comments_per_context_unit: int = 25
    notes: str | None = None
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "RagSidecarBuildConfig":
        missing = [
            key
            for key in ["trigger_comment_map_path", "output_dir"]
            if not payload.get(key)
        ]
        if missing:
            raise ValueError(
                "RAG sidecar config missing required fields: " + ", ".join(missing)
            )
        params = payload.get("params") or {}
        if not isinstance(params, dict):
            raise ValueError("params must be an object.")
        max_comments = int(payload.get("max_comments_per_context_unit", 25))
        if max_comments < 1:
            raise ValueError("max_comments_per_context_unit must be >= 1.")
        return cls(
            trigger_comment_map_path=str(payload["trigger_comment_map_path"]),
            output_dir=str(payload["output_dir"]),
            comments_path=str(
                payload.get("comments_path", "data/gold/clean_comments.parquet")
            ),
            snapshots_path=(
                str(payload["snapshots_path"])
                if payload.get("snapshots_path") is not None
                else None
            ),
            detector_name=str(payload.get("detector_name", "xiao_ema")),
            run_id=payload.get("run_id"),
            max_comments_per_context_unit=max_comments,
            notes=payload.get("notes"),
            params=params,
        )


def _extract_config_payload(payload: dict[str, Any]) -> dict[str, Any]:
    nested = payload.get("rag_sidecars")
    if nested is None:
        return payload
    if not isinstance(nested, dict):
        raise ValueError("rag_sidecars config section must be an object.")
    return nested


def _merge_config_payloads(*payloads: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for payload in payloads:
        for key, value in payload.items():
            if value is None:
                continue
            if key == "params":
                current = merged.get(key, {})
                if not isinstance(current, dict) or not isinstance(value, dict):
                    raise ValueError("params must be an object.")
                merged[key] = {**current, **value}
            else:
                merged[key] = value
    return merged


def load_rag_sidecar_config(
    path: str | Path,
    *,
    overrides: dict[str, Any] | None = None,
) -> RagSidecarBuildConfig:
    p = Path(path)
    payload = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"RAG sidecar config must be an object: {p}")
    base = _extract_config_payload(payload)
    merged = _merge_config_payloads(base, overrides or {})
    return RagSidecarBuildConfig.from_mapping(merged)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_path(path: str | Path | None) -> str | None:
    if path is None:
        return None
    return Path(path).as_posix()


def _short_hash(*parts: Any) -> str:
    payload = "|".join("" if part is None else str(part) for part in parts)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def _require_columns(df: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(df.columns))
    if missing:
        raise KeyError(f"{label} missing required columns: {', '.join(missing)}")


def _to_utc(value: Any) -> pd.Timestamp:
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"Invalid UTC timestamp: {value!r}")
    return pd.Timestamp(ts)


def _normalize_utc_column(df: pd.DataFrame, column: str) -> None:
    df[column] = pd.to_datetime(df[column], utc=True, errors="coerce")
    if df[column].isna().any():
        raise ValueError(f"Column {column!r} contains invalid timestamps.")


def _isoformat(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).isoformat()


def _unix_seconds(value: Any) -> int | None:
    if value is None or pd.isna(value):
        return None
    return int(pd.Timestamp(value).timestamp())


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if value is pd.NA:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(_json_safe(record), ensure_ascii=False) for record in records]
    p.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def read_table(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() == ".parquet" or p.is_dir():
        return pd.read_parquet(p)
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    raise ValueError(f"Unsupported file format: {p}")


def _prepare_trigger_comment_map(trigger_df: pd.DataFrame) -> pd.DataFrame:
    df = trigger_df.copy()
    _require_columns(df, TRIGGER_COMMENT_REQUIRED_COLUMNS, "trigger_comment_map")
    for col in ["trigger_time", "window_start", "window_end", "event_time_utc"]:
        _normalize_utc_column(df, col)
    if "trigger_strength" not in df.columns:
        df["trigger_strength"] = pd.NA
    if "title" not in df.columns:
        df["title"] = pd.NA
    if "channel_title" not in df.columns:
        df["channel_title"] = pd.NA
    return df


def _prepare_comments(comments_df: pd.DataFrame) -> pd.DataFrame:
    df = comments_df.copy()
    _require_columns(df, GOLD_COMMENT_REQUIRED_COLUMNS, "comments")
    _normalize_utc_column(df, "event_time_utc")
    for col in [
        "author_id",
        "text_clean",
        "is_reply",
        "reply_to_comment_id",
        "likes",
        "emoji_count",
        "exclamation_count",
        "question_count",
        "caps_ratio",
        "link_count",
        "token_count",
        "is_probable_spam",
    ]:
        if col not in df.columns:
            df[col] = pd.NA
    return df


def _make_run_id(
    *,
    detector_name: str,
    trigger_comment_map_path: str | Path,
    comments_path: str | Path,
    snapshots_path: str | Path | None,
) -> str:
    digest = _short_hash(
        detector_name,
        _normalize_path(trigger_comment_map_path),
        _normalize_path(comments_path),
        _normalize_path(snapshots_path),
    )
    return f"run_{digest}"


def derive_rag_sidecar_run_id(config: RagSidecarBuildConfig) -> str:
    """Return the existing path-derived stage identity for a sidecar run."""

    return _make_run_id(
        detector_name=config.detector_name,
        trigger_comment_map_path=config.trigger_comment_map_path,
        comments_path=config.comments_path,
        snapshots_path=config.snapshots_path,
    )


def _make_event_id(
    *,
    run_id: str,
    detector_name: str,
    trigger_time: pd.Timestamp,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
) -> str:
    return "evt_" + _short_hash(
        run_id,
        detector_name,
        trigger_time.isoformat(),
        window_start.isoformat(),
        window_end.isoformat(),
    )


def _make_context_unit_id(
    *,
    event_id: str,
    video_id: str,
    context_type: str,
    order: int,
    comment_ids: list[str],
) -> str:
    first_comment_id = comment_ids[0] if comment_ids else ""
    last_comment_id = comment_ids[-1] if comment_ids else ""
    return "ctx_" + _short_hash(
        event_id,
        video_id,
        context_type,
        order,
        first_comment_id,
        last_comment_id,
    )


def _event_rows(trigger_map: pd.DataFrame, *, run_id: str, detector_name: str) -> pd.DataFrame:
    event_cols = [
        "trigger_time",
        "window_start",
        "window_end",
        "trigger_volume",
        "trigger_strength",
    ]
    events = (
        trigger_map[event_cols]
        .drop_duplicates()
        .sort_values(["trigger_time", "window_start", "window_end"])
        .reset_index(drop=True)
    )
    event_records: list[dict[str, Any]] = []
    for _, row in events.iterrows():
        trigger_time = _to_utc(row["trigger_time"])
        window_start = _to_utc(row["window_start"])
        window_end = _to_utc(row["window_end"])
        event_records.append(
            {
                "event_id": _make_event_id(
                    run_id=run_id,
                    detector_name=detector_name,
                    trigger_time=trigger_time,
                    window_start=window_start,
                    window_end=window_end,
                ),
                "run_id": run_id,
                "detector_name": detector_name,
                "trigger_time": trigger_time,
                "trigger_time_utc": trigger_time,
                "trigger_time_unix_s": _unix_seconds(trigger_time),
                "window_start": window_start,
                "window_start_utc": window_start,
                "window_end": window_end,
                "window_end_utc": window_end,
                "trigger_volume": row["trigger_volume"],
                "trigger_strength": row["trigger_strength"],
                "artifact_version": RAG_SIDECAR_ARTIFACT_VERSION,
            }
        )
    return pd.DataFrame(event_records)


def _event_video_source_map(
    trigger_map: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, event in events.iterrows():
        mask = (
            (trigger_map["trigger_time"] == event["trigger_time"])
            & (trigger_map["window_start"] == event["window_start"])
            & (trigger_map["window_end"] == event["window_end"])
        )
        event_trigger_rows = trigger_map.loc[mask].copy()
        video_meta = (
            event_trigger_rows.groupby("video_id", dropna=False)
            .agg(
                title=("title", "first"),
                channel_title=("channel_title", "first"),
                trigger_map_comment_count=("comment_id", "nunique"),
            )
            .reset_index()
            .sort_values(["trigger_map_comment_count", "video_id"], ascending=[False, True])
        )
        for _, video in video_meta.iterrows():
            rows.append(
                {
                    "event_id": event["event_id"],
                    "run_id": event["run_id"],
                    "trigger_time": event["trigger_time"],
                    "window_start": event["window_start"],
                    "window_end": event["window_end"],
                    "video_id": video["video_id"],
                    "title": video["title"],
                    "channel_title": video["channel_title"],
                    "trigger_map_comment_count": int(video["trigger_map_comment_count"]),
                    "poc_join_key": f"{event['trigger_time'].isoformat()}|{video['video_id']}",
                    "artifact_version": RAG_SIDECAR_ARTIFACT_VERSION,
                }
            )
    return pd.DataFrame(rows)


def _append_missing_trigger_comments(
    selected: pd.DataFrame,
    trigger_subset: pd.DataFrame,
    event: pd.Series,
    *,
    comment_source_path: str | Path,
    trigger_comment_map_path: str | Path,
) -> pd.DataFrame:
    selected_ids = set(selected["comment_id"].astype(str))
    missing = trigger_subset.loc[
        ~trigger_subset["comment_id"].astype(str).isin(selected_ids)
    ].copy()
    if missing.empty:
        selected["source_artifact_type"] = "gold_comments"
        return selected

    fallback_cols = {
        "video_id": missing["video_id"],
        "comment_id": missing["comment_id"],
        "text": missing["text"],
        "author_id": missing["author_id"] if "author_id" in missing.columns else pd.NA,
        "event_time_utc": missing["event_time_utc"],
        "text_clean": pd.NA,
        "is_reply": pd.NA,
        "reply_to_comment_id": pd.NA,
        "likes": pd.NA,
        "emoji_count": pd.NA,
        "exclamation_count": pd.NA,
        "question_count": pd.NA,
        "caps_ratio": pd.NA,
        "link_count": pd.NA,
        "token_count": pd.NA,
        "is_probable_spam": pd.NA,
        "comment_source_path": _normalize_path(trigger_comment_map_path),
        "source_artifact_type": "trigger_comment_map_fallback",
    }
    fallback = pd.DataFrame(fallback_cols)
    selected = selected.copy()
    selected["source_artifact_type"] = "gold_comments"
    selected["comment_source_path"] = _normalize_path(comment_source_path)
    return pd.concat([selected, fallback], ignore_index=True)


def build_event_comment_inventory(
    *,
    comments: pd.DataFrame,
    trigger_map: pd.DataFrame,
    events: pd.DataFrame,
    event_video_sources: pd.DataFrame,
    comments_path: str | Path,
    trigger_comment_map_path: str | Path,
) -> pd.DataFrame:
    inventory_parts: list[pd.DataFrame] = []
    for _, event in events.iterrows():
        videos = event_video_sources.loc[
            event_video_sources["event_id"] == event["event_id"],
            "video_id",
        ].astype(str)
        video_ids = set(videos)
        window_start = _to_utc(event["window_start"])
        window_end = _to_utc(event["window_end"])
        event_trigger_mask = (
            (trigger_map["trigger_time"] == event["trigger_time"])
            & (trigger_map["window_start"] == event["window_start"])
            & (trigger_map["window_end"] == event["window_end"])
        )
        trigger_subset = trigger_map.loc[event_trigger_mask].copy()

        selected = comments.loc[
            comments["video_id"].astype(str).isin(video_ids)
            & (comments["event_time_utc"] >= window_start)
            & (comments["event_time_utc"] <= window_end)
        ].copy()
        selected = selected.drop_duplicates(subset=["comment_id"])
        selected["comment_source_path"] = _normalize_path(comments_path)
        selected = _append_missing_trigger_comments(
            selected,
            trigger_subset,
            event,
            comment_source_path=comments_path,
            trigger_comment_map_path=trigger_comment_map_path,
        )
        if selected.empty:
            continue

        meta = event_video_sources.loc[
            event_video_sources["event_id"] == event["event_id"],
            ["video_id", "title", "channel_title"],
        ].drop_duplicates(subset=["video_id"])
        selected = selected.merge(meta, on="video_id", how="left")
        selected = selected.sort_values(["event_time_utc", "video_id", "comment_id"])
        selected.insert(0, "order_in_event", range(1, len(selected) + 1))
        selected["order_in_event_video"] = (
            selected.groupby("video_id").cumcount() + 1
        )
        selected.insert(0, "run_id", event["run_id"])
        selected.insert(0, "event_id", event["event_id"])
        selected["detector_name"] = event["detector_name"]
        selected["trigger_time"] = event["trigger_time"]
        selected["trigger_time_utc"] = event["trigger_time_utc"]
        selected["window_start"] = event["window_start"]
        selected["window_start_utc"] = event["window_start_utc"]
        selected["window_end"] = event["window_end"]
        selected["window_end_utc"] = event["window_end_utc"]
        selected["trigger_volume"] = event["trigger_volume"]
        selected["trigger_strength"] = event["trigger_strength"]
        selected["event_time_unix_s"] = selected["event_time_utc"].map(_unix_seconds)
        inventory_parts.append(selected)

    if not inventory_parts:
        return pd.DataFrame()

    inventory = pd.concat(inventory_parts, ignore_index=True)
    inventory["is_reply"] = inventory["is_reply"].fillna(False).astype(bool)
    reply_parent = inventory["reply_to_comment_id"].where(
        inventory["reply_to_comment_id"].notna(), inventory["comment_id"]
    )
    inventory["root_comment_id"] = reply_parent
    inventory["parent_comment_id"] = inventory["reply_to_comment_id"]

    pair_keys = set(zip(inventory["event_id"], inventory["comment_id"]))
    inventory["parent_in_inventory"] = inventory.apply(
        lambda row: (
            (row["event_id"], row["parent_comment_id"]) in pair_keys
            if pd.notna(row["parent_comment_id"])
            else False
        ),
        axis=1,
    )
    inventory["artifact_version"] = RAG_SIDECAR_ARTIFACT_VERSION

    field_order = [
        "event_id",
        "run_id",
        "detector_name",
        "trigger_time",
        "trigger_time_utc",
        "window_start",
        "window_start_utc",
        "window_end",
        "window_end_utc",
        "trigger_volume",
        "trigger_strength",
        "order_in_event",
        "order_in_event_video",
        "event_time_utc",
        "event_time_unix_s",
        "video_id",
        "title",
        "channel_title",
        "comment_id",
        "root_comment_id",
        "parent_comment_id",
        "is_reply",
        "parent_in_inventory",
        "author_id",
        "text",
        "text_clean",
        "likes",
        "emoji_count",
        "exclamation_count",
        "question_count",
        "caps_ratio",
        "link_count",
        "token_count",
        "is_probable_spam",
        "comment_source_path",
        "source_artifact_type",
        "artifact_version",
    ]
    for col in field_order:
        if col not in inventory.columns:
            inventory[col] = pd.NA
    return inventory[field_order]


def build_event_video_map(event_video_sources: pd.DataFrame, inventory: pd.DataFrame) -> pd.DataFrame:
    if inventory.empty:
        out = event_video_sources.copy()
        out["inventory_comment_count"] = 0
        out["reply_count"] = 0
        out["root_comment_count"] = 0
        out["first_comment_time_utc"] = pd.NA
        out["last_comment_time_utc"] = pd.NA
        return out

    counts = (
        inventory.groupby(["event_id", "video_id"], dropna=False)
        .agg(
            inventory_comment_count=("comment_id", "nunique"),
            reply_count=("is_reply", "sum"),
            root_comment_count=("root_comment_id", "nunique"),
            first_comment_time_utc=("event_time_utc", "min"),
            last_comment_time_utc=("event_time_utc", "max"),
        )
        .reset_index()
    )
    out = event_video_sources.merge(counts, on=["event_id", "video_id"], how="left")
    for col in ["inventory_comment_count", "reply_count", "root_comment_count"]:
        out[col] = out[col].fillna(0).astype(int)
    out["artifact_version"] = RAG_SIDECAR_ARTIFACT_VERSION
    return out


def build_event_thread_map(inventory: pd.DataFrame) -> pd.DataFrame:
    if inventory.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    group_cols = [
        "event_id",
        "run_id",
        "trigger_time",
        "window_start",
        "window_end",
        "video_id",
        "root_comment_id",
    ]
    for keys, group in inventory.groupby(group_cols, dropna=False, sort=False):
        (
            event_id,
            run_id,
            trigger_time,
            window_start,
            window_end,
            video_id,
            root_comment_id,
        ) = keys
        group = group.sort_values(["event_time_utc", "comment_id"])
        comment_ids = [str(value) for value in group["comment_id"].tolist()]
        reply_count = int(group["is_reply"].sum())
        root_rows = group.loc[group["comment_id"].astype(str) == str(root_comment_id)]
        rows.append(
            {
                "event_id": event_id,
                "run_id": run_id,
                "trigger_time": trigger_time,
                "window_start": window_start,
                "window_end": window_end,
                "video_id": video_id,
                "root_comment_id": root_comment_id,
                "root_comment_present": bool(not root_rows.empty),
                "comment_count": int(group["comment_id"].nunique()),
                "reply_count": reply_count,
                "has_replies": bool(reply_count > 0),
                "first_comment_time_utc": group["event_time_utc"].min(),
                "last_comment_time_utc": group["event_time_utc"].max(),
                "comment_ids": "|".join(comment_ids),
                "artifact_version": RAG_SIDECAR_ARTIFACT_VERSION,
            }
        )
    return pd.DataFrame(rows)


def _context_text(group: pd.DataFrame) -> str:
    lines = []
    for _, row in group.iterrows():
        reply_marker = "reply" if bool(row.get("is_reply", False)) else "comment"
        lines.append(
            f"[{row['comment_id']}|{reply_marker}|{row['event_time_utc']}] "
            f"{row.get('text', '')}"
        )
    return "\n".join(lines)


def _chunk_group(group: pd.DataFrame, max_size: int) -> list[pd.DataFrame]:
    return [group.iloc[start : start + max_size].copy() for start in range(0, len(group), max_size)]


def build_context_units(
    inventory: pd.DataFrame,
    *,
    source_inventory_path: str | Path,
    max_comments_per_context_unit: int,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    context_units: list[dict[str, Any]] = []
    comment_map_rows: list[dict[str, Any]] = []
    if inventory.empty:
        return context_units, pd.DataFrame()

    assigned_pairs: set[tuple[str, str]] = set()
    unit_order_by_event: dict[str, int] = {}

    thread_groups = inventory.groupby(
        ["event_id", "video_id", "root_comment_id"], dropna=False, sort=False
    )
    for (event_id, video_id, root_comment_id), group in thread_groups:
        group = group.sort_values(["event_time_utc", "comment_id"])
        is_thread = bool(group["is_reply"].any() or len(group) > 1)
        if not is_thread:
            continue
        chunks = _chunk_group(group, max_comments_per_context_unit)
        for chunk_index, chunk in enumerate(chunks, start=1):
            order = unit_order_by_event.get(str(event_id), 0) + 1
            unit_order_by_event[str(event_id)] = order
            comment_ids = [str(value) for value in chunk["comment_id"].tolist()]
            context_type = "thread" if len(chunks) == 1 else "thread_part"
            context_unit_id = _make_context_unit_id(
                event_id=str(event_id),
                video_id=str(video_id),
                context_type=context_type,
                order=order,
                comment_ids=comment_ids,
            )
            record = _context_unit_record(
                context_unit_id=context_unit_id,
                context_order=order,
                context_type=context_type,
                group=chunk,
                comment_ids=comment_ids,
                source_inventory_path=source_inventory_path,
                selection_reason=(
                    "deterministic_thread_group"
                    if len(chunks) == 1
                    else f"deterministic_thread_group_part_{chunk_index}"
                ),
                root_comment_ids=[str(root_comment_id)],
            )
            context_units.append(record)
            _extend_context_comment_map(comment_map_rows, record, chunk)
            assigned_pairs.update((str(event_id), item) for item in comment_ids)

    remaining = inventory.loc[
        ~inventory.apply(
            lambda row: (str(row["event_id"]), str(row["comment_id"])) in assigned_pairs,
            axis=1,
        )
    ].copy()
    for (event_id, video_id), group in remaining.groupby(
        ["event_id", "video_id"], dropna=False, sort=False
    ):
        group = group.sort_values(["event_time_utc", "comment_id"])
        chunks = _chunk_group(group, max_comments_per_context_unit)
        for chunk_index, chunk in enumerate(chunks, start=1):
            order = unit_order_by_event.get(str(event_id), 0) + 1
            unit_order_by_event[str(event_id)] = order
            comment_ids = [str(value) for value in chunk["comment_id"].tolist()]
            context_type = "video_time_block"
            context_unit_id = _make_context_unit_id(
                event_id=str(event_id),
                video_id=str(video_id),
                context_type=context_type,
                order=order,
                comment_ids=comment_ids,
            )
            record = _context_unit_record(
                context_unit_id=context_unit_id,
                context_order=order,
                context_type=context_type,
                group=chunk,
                comment_ids=comment_ids,
                source_inventory_path=source_inventory_path,
                selection_reason=f"deterministic_video_time_block_{chunk_index}",
                root_comment_ids=[
                    str(value)
                    for value in chunk["root_comment_id"].dropna().drop_duplicates().tolist()
                ],
            )
            context_units.append(record)
            _extend_context_comment_map(comment_map_rows, record, chunk)

    context_comment_map = pd.DataFrame(comment_map_rows)
    return context_units, context_comment_map


def _context_unit_record(
    *,
    context_unit_id: str,
    context_order: int,
    context_type: str,
    group: pd.DataFrame,
    comment_ids: list[str],
    source_inventory_path: str | Path,
    selection_reason: str,
    root_comment_ids: list[str],
) -> dict[str, Any]:
    first = group.iloc[0]
    return {
        "context_unit_id": context_unit_id,
        "context_order_in_event": context_order,
        "event_id": first["event_id"],
        "run_id": first["run_id"],
        "trigger_time": _isoformat(first["trigger_time"]),
        "trigger_time_utc": _isoformat(first["trigger_time_utc"]),
        "video_id": first["video_id"],
        "title": first.get("title"),
        "channel_title": first.get("channel_title"),
        "window_start": _isoformat(first["window_start"]),
        "window_start_utc": _isoformat(first["window_start_utc"]),
        "window_end": _isoformat(first["window_end"]),
        "window_end_utc": _isoformat(first["window_end_utc"]),
        "context_type": context_type,
        "comment_ids": comment_ids,
        "root_comment_ids": root_comment_ids,
        "comment_count": int(len(comment_ids)),
        "time_start_utc": _isoformat(group["event_time_utc"].min()),
        "time_end_utc": _isoformat(group["event_time_utc"].max()),
        "contains_replies": bool(group["is_reply"].any()),
        "selection_reason": selection_reason,
        "source_inventory_path": _normalize_path(source_inventory_path),
        "context_text": _context_text(group),
        "artifact_version": RAG_SIDECAR_ARTIFACT_VERSION,
    }


def _extend_context_comment_map(
    rows: list[dict[str, Any]],
    context_unit: dict[str, Any],
    group: pd.DataFrame,
) -> None:
    for order, (_, row) in enumerate(group.iterrows(), start=1):
        rows.append(
            {
                "context_unit_id": context_unit["context_unit_id"],
                "event_id": row["event_id"],
                "run_id": row["run_id"],
                "trigger_time": row["trigger_time"],
                "video_id": row["video_id"],
                "window_start": row["window_start"],
                "window_end": row["window_end"],
                "context_type": context_unit["context_type"],
                "order_in_context_unit": order,
                "comment_id": row["comment_id"],
                "root_comment_id": row["root_comment_id"],
                "event_time_utc": row["event_time_utc"],
                "is_reply": row["is_reply"],
                "artifact_version": RAG_SIDECAR_ARTIFACT_VERSION,
            }
        )


def _event_signal_counts(
    snapshots: pd.DataFrame | None,
    events: pd.DataFrame,
) -> dict[str, int]:
    if snapshots is None or snapshots.empty:
        return {}
    df = snapshots.copy()
    if {"window_start", "window_end"}.difference(df.columns):
        return {}
    _normalize_utc_column(df, "window_start")
    _normalize_utc_column(df, "window_end")
    counts: dict[str, int] = {}
    for _, event in events.iterrows():
        mask = (
            (df["window_end"] >= event["window_start"])
            & (df["window_end"] <= event["window_end"])
        )
        counts[str(event["event_id"])] = int(mask.sum())
    return counts


def build_event_evidence_packages(
    *,
    events: pd.DataFrame,
    inventory: pd.DataFrame,
    event_video_map: pd.DataFrame,
    context_units: list[dict[str, Any]],
    signal_counts: dict[str, int],
    output_paths: dict[str, str],
    comments_path: str | Path,
    trigger_comment_map_path: str | Path,
    snapshots_path: str | Path | None,
    created_at_utc: str,
) -> list[dict[str, Any]]:
    inventory_counts = (
        inventory.groupby("event_id")
        .agg(
            comment_count=("comment_id", "nunique"),
            reply_count=("is_reply", "sum"),
            unique_video_count=("video_id", "nunique"),
            unique_author_count=("author_id", "nunique"),
        )
        .to_dict("index")
        if not inventory.empty
        else {}
    )
    video_counts = (
        event_video_map.groupby("event_id")["video_id"].nunique().to_dict()
        if not event_video_map.empty
        else {}
    )
    context_counts: dict[str, int] = {}
    for unit in context_units:
        event_id = str(unit["event_id"])
        context_counts[event_id] = context_counts.get(event_id, 0) + 1

    packages: list[dict[str, Any]] = []
    for _, event in events.iterrows():
        event_id = str(event["event_id"])
        counts = inventory_counts.get(event_id, {})
        missing: list[str] = []
        if int(counts.get("comment_count", 0) or 0) == 0:
            missing.append("comments")
        if int(video_counts.get(event_id, 0) or 0) == 0:
            missing.append("videos")
        if snapshots_path and int(signal_counts.get(event_id, 0) or 0) == 0:
            missing.append("signals")
        status = "ready" if not missing else "missing_" + "_and_".join(missing)
        packages.append(
            {
                "event_id": event_id,
                "run_id": event["run_id"],
                "detector_name": event["detector_name"],
                "trigger_time": _isoformat(event["trigger_time"]),
                "trigger_time_utc": _isoformat(event["trigger_time_utc"]),
                "trigger_time_unix_s": event["trigger_time_unix_s"],
                "window_start": _isoformat(event["window_start"]),
                "window_start_utc": _isoformat(event["window_start_utc"]),
                "window_end": _isoformat(event["window_end"]),
                "window_end_utc": _isoformat(event["window_end_utc"]),
                "trigger_volume": event["trigger_volume"],
                "trigger_strength": event["trigger_strength"],
                "source_trigger_comment_map_path": _normalize_path(
                    trigger_comment_map_path
                ),
                "source_comments_path": _normalize_path(comments_path),
                "source_snapshots_path": _normalize_path(snapshots_path),
                "event_comment_inventory_path": output_paths["event_comment_inventory"],
                "event_video_map_path": output_paths["event_video_map"],
                "event_thread_map_path": output_paths["event_thread_map"],
                "rag_context_units_path": output_paths["rag_context_units"],
                "context_unit_comment_map_path": output_paths[
                    "context_unit_comment_map"
                ],
                "context_selection_manifest_path": output_paths[
                    "context_selection_manifest"
                ],
                "comment_count": int(counts.get("comment_count", 0) or 0),
                "reply_count": int(counts.get("reply_count", 0) or 0),
                "unique_video_count": int(counts.get("unique_video_count", 0) or 0),
                "unique_author_count": int(counts.get("unique_author_count", 0) or 0),
                "signal_snapshot_count": int(signal_counts.get(event_id, 0) or 0),
                "context_unit_count": int(context_counts.get(event_id, 0) or 0),
                "rag_readiness_status": status,
                "package_created_at_utc": created_at_utc,
                "package_artifact_version": RAG_SIDECAR_ARTIFACT_VERSION,
            }
        )
    return packages


def _format_timestamp_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = out[col].map(_isoformat)
    return out


def _output_paths(output_dir: str | Path) -> dict[str, str]:
    root = Path(output_dir)
    return {
        "event_evidence_packages": (root / EVENT_EVIDENCE_PACKAGES_FILE).as_posix(),
        "event_comment_inventory": (root / EVENT_COMMENT_INVENTORY_FILE).as_posix(),
        "event_video_map": (root / EVENT_VIDEO_MAP_FILE).as_posix(),
        "event_thread_map": (root / EVENT_THREAD_MAP_FILE).as_posix(),
        "rag_context_units": (root / RAG_CONTEXT_UNITS_FILE).as_posix(),
        "context_unit_comment_map": (root / CONTEXT_UNIT_COMMENT_MAP_FILE).as_posix(),
        "context_selection_manifest": (
            root / CONTEXT_SELECTION_MANIFEST_FILE
        ).as_posix(),
    }


def _coverage_checks(
    inventory: pd.DataFrame,
    context_comment_map: pd.DataFrame,
) -> dict[str, Any]:
    if inventory.empty:
        return {
            "inventory_event_comment_pairs": 0,
            "context_event_comment_pairs": 0,
            "missing_context_pairs": 0,
            "extra_context_pairs": 0,
            "all_inventory_comments_have_context": True,
        }
    inventory_pairs = set(zip(inventory["event_id"], inventory["comment_id"].astype(str)))
    if context_comment_map.empty:
        context_pairs: set[tuple[Any, str]] = set()
    else:
        context_pairs = set(
            zip(context_comment_map["event_id"], context_comment_map["comment_id"].astype(str))
        )
    missing = inventory_pairs.difference(context_pairs)
    extra = context_pairs.difference(inventory_pairs)
    return {
        "inventory_event_comment_pairs": len(inventory_pairs),
        "context_event_comment_pairs": len(context_pairs),
        "missing_context_pairs": len(missing),
        "extra_context_pairs": len(extra),
        "all_inventory_comments_have_context": len(missing) == 0,
    }


def build_context_selection_manifest(
    *,
    config: RagSidecarBuildConfig,
    run_id: str,
    created_at_utc: str,
    output_paths: dict[str, str],
    events: pd.DataFrame,
    inventory: pd.DataFrame,
    event_video_map: pd.DataFrame,
    event_thread_map: pd.DataFrame,
    context_units: list[dict[str, Any]],
    context_comment_map: pd.DataFrame,
) -> dict[str, Any]:
    coverage = _coverage_checks(inventory, context_comment_map)
    missing_fields = {
        "trigger_comment_map": [],
        "comments": [],
    }
    return {
        "run_id": run_id,
        "created_at_utc": created_at_utc,
        "pipeline_stage": "rag_sidecar_evidence_preparation",
        "mode": "sidecar_only_no_generation_no_retrieval_no_validation",
        "artifact_version": RAG_SIDECAR_ARTIFACT_VERSION,
        "source_artifacts": {
            "trigger_comment_map_path": _normalize_path(
                config.trigger_comment_map_path
            ),
            "comments_path": _normalize_path(config.comments_path),
            "snapshots_path": _normalize_path(config.snapshots_path),
        },
        "output_paths": output_paths,
        "compatibility_policy": {
            "preserve_poc_key": "trigger_time + video_id",
            "event_id_role": "internal_retrocompatible_join_key",
            "does_not_modify_existing_outputs": True,
            "does_not_call_llm": True,
            "does_not_create_embeddings": True,
            "does_not_change_detection": True,
        },
        "event_id_formula": (
            "evt_ + sha1(run_id|detector_name|trigger_time_utc|"
            "window_start_utc|window_end_utc)[:12]"
        ),
        "comment_inclusion_rule": (
            "window_start <= event_time_utc <= window_end and video_id in "
            "videos observed for the event in trigger_comment_map"
        ),
        "context_policy": {
            "hierarchy": "event -> video -> thread or video_time_block -> comments",
            "chunking": "deterministic",
            "max_comments_per_context_unit": config.max_comments_per_context_unit,
            "selection": "full_coverage_no_model_ranking",
            "context_retrieval_basis": "metadata_only",
        },
        "counts": {
            "event_count": int(events["event_id"].nunique()) if not events.empty else 0,
            "inventory_rows": int(len(inventory)),
            "inventory_unique_comments": (
                int(inventory["comment_id"].nunique()) if not inventory.empty else 0
            ),
            "event_video_rows": int(len(event_video_map)),
            "event_thread_rows": int(len(event_thread_map)),
            "context_unit_count": int(len(context_units)),
            "context_unit_comment_rows": int(len(context_comment_map)),
        },
        "coverage_checks": coverage,
        "missing_fields": missing_fields,
        "notes": config.notes,
        "params": config.params,
    }


def write_rag_sidecar_artifacts(
    *,
    trigger_comment_map_path: str | Path,
    output_dir: str | Path,
    comments_path: str | Path = "data/gold/clean_comments.parquet",
    snapshots_path: str | Path | None = None,
    detector_name: str = "xiao_ema",
    run_id: str | None = None,
    max_comments_per_context_unit: int = 25,
    notes: str | None = None,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = RagSidecarBuildConfig(
        trigger_comment_map_path=str(trigger_comment_map_path),
        comments_path=str(comments_path),
        snapshots_path=str(snapshots_path) if snapshots_path else None,
        output_dir=str(output_dir),
        detector_name=detector_name,
        run_id=run_id,
        max_comments_per_context_unit=max_comments_per_context_unit,
        notes=notes,
        params=params or {},
    )
    return write_rag_sidecar_artifacts_from_config(config)


def write_rag_sidecar_artifacts_from_config(
    config: RagSidecarBuildConfig,
) -> dict[str, Any]:
    output_root = Path(config.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    output_paths = _output_paths(output_root)
    created_at = _utc_now_iso()
    resolved_run_id = config.run_id or _make_run_id(
        detector_name=config.detector_name,
        trigger_comment_map_path=config.trigger_comment_map_path,
        comments_path=config.comments_path,
        snapshots_path=config.snapshots_path,
    )

    trigger_map = _prepare_trigger_comment_map(read_table(config.trigger_comment_map_path))
    comments = _prepare_comments(read_table(config.comments_path))
    snapshots = read_table(config.snapshots_path) if config.snapshots_path else None

    events = _event_rows(
        trigger_map,
        run_id=resolved_run_id,
        detector_name=config.detector_name,
    )
    event_video_sources = _event_video_source_map(trigger_map, events)
    inventory = build_event_comment_inventory(
        comments=comments,
        trigger_map=trigger_map,
        events=events,
        event_video_sources=event_video_sources,
        comments_path=config.comments_path,
        trigger_comment_map_path=config.trigger_comment_map_path,
    )
    event_video_map = build_event_video_map(event_video_sources, inventory)
    event_thread_map = build_event_thread_map(inventory)
    context_units, context_comment_map = build_context_units(
        inventory,
        source_inventory_path=output_paths["event_comment_inventory"],
        max_comments_per_context_unit=config.max_comments_per_context_unit,
    )
    signal_counts = _event_signal_counts(snapshots, events)
    packages = build_event_evidence_packages(
        events=events,
        inventory=inventory,
        event_video_map=event_video_map,
        context_units=context_units,
        signal_counts=signal_counts,
        output_paths=output_paths,
        comments_path=config.comments_path,
        trigger_comment_map_path=config.trigger_comment_map_path,
        snapshots_path=config.snapshots_path,
        created_at_utc=created_at,
    )
    manifest = build_context_selection_manifest(
        config=config,
        run_id=resolved_run_id,
        created_at_utc=created_at,
        output_paths=output_paths,
        events=events,
        inventory=inventory,
        event_video_map=event_video_map,
        event_thread_map=event_thread_map,
        context_units=context_units,
        context_comment_map=context_comment_map,
    )

    timestamp_cols = [
        "trigger_time",
        "trigger_time_utc",
        "window_start",
        "window_start_utc",
        "window_end",
        "window_end_utc",
        "event_time_utc",
        "first_comment_time_utc",
        "last_comment_time_utc",
    ]
    _format_timestamp_columns(inventory, timestamp_cols).to_csv(
        output_paths["event_comment_inventory"], index=False
    )
    _format_timestamp_columns(event_video_map, timestamp_cols).to_csv(
        output_paths["event_video_map"], index=False
    )
    _format_timestamp_columns(event_thread_map, timestamp_cols).to_csv(
        output_paths["event_thread_map"], index=False
    )
    _write_jsonl(output_paths["rag_context_units"], context_units)
    _format_timestamp_columns(context_comment_map, timestamp_cols).to_csv(
        output_paths["context_unit_comment_map"], index=False
    )
    _write_jsonl(output_paths["event_evidence_packages"], packages)
    _write_json(output_paths["context_selection_manifest"], manifest)

    return {
        "run_id": resolved_run_id,
        "artifact_version": RAG_SIDECAR_ARTIFACT_VERSION,
        "output_paths": output_paths,
        "event_count": manifest["counts"]["event_count"],
        "inventory_rows": manifest["counts"]["inventory_rows"],
        "event_video_rows": manifest["counts"]["event_video_rows"],
        "event_thread_rows": manifest["counts"]["event_thread_rows"],
        "context_unit_count": manifest["counts"]["context_unit_count"],
        "context_unit_comment_rows": manifest["counts"][
            "context_unit_comment_rows"
        ],
        "coverage_checks": manifest["coverage_checks"],
    }


__all__ = [
    "CONTEXT_SELECTION_MANIFEST_FILE",
    "CONTEXT_UNIT_COMMENT_MAP_FILE",
    "EVENT_COMMENT_INVENTORY_FILE",
    "EVENT_EVIDENCE_PACKAGES_FILE",
    "EVENT_THREAD_MAP_FILE",
    "EVENT_VIDEO_MAP_FILE",
    "RAG_CONTEXT_UNITS_FILE",
    "RAG_SIDECAR_ARTIFACT_VERSION",
    "RagSidecarBuildConfig",
    "build_context_selection_manifest",
    "build_context_units",
    "build_event_comment_inventory",
    "build_event_evidence_packages",
    "build_event_thread_map",
    "build_event_video_map",
    "derive_rag_sidecar_run_id",
    "load_rag_sidecar_config",
    "read_table",
    "write_rag_sidecar_artifacts",
    "write_rag_sidecar_artifacts_from_config",
]
