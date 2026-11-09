from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


DAILY_RAG_SIDECAR_ARTIFACT_VERSION = "daily_rag_sidecars_v1"
DAILY_EVENT_EVIDENCE_PACKAGES_FILE = "daily_event_evidence_packages.jsonl"
DAILY_EVENT_COMMENT_INVENTORY_FILE = "daily_event_comment_inventory.csv"
DAILY_EVENT_VIDEO_MAP_FILE = "daily_event_video_map.csv"
DAILY_EVENT_THREAD_MAP_FILE = "daily_event_thread_map.csv"
DAILY_RAG_CONTEXT_UNITS_FILE = "daily_rag_context_units.jsonl"
DAILY_CONTEXT_UNIT_COMMENT_MAP_FILE = "daily_context_unit_comment_map.csv"
DAILY_RAG_SIDECARS_MANIFEST_FILE = "daily_rag_sidecars_manifest.json"
README_FILE = "README.md"

EVENT_REQUIRED_COLUMNS = {
    "daily_event_id",
    "cycle_id",
    "cycle_index",
    "detector_name",
    "signal_name",
    "signal_value",
    "baseline_mean",
    "ratio_to_baseline",
    "delta_value",
    "pct_change_value",
    "threshold_value",
    "trigger_reason",
    "analysis_window_start_utc",
    "analysis_window_end_utc",
    "data_cutoff_utc",
}
WINDOW_REQUIRED_COLUMNS = {
    "cycle_id",
    "cycle_index",
    "comment_id",
    "video_id",
    "event_time_utc",
    "first_seen_cycle_id",
    "analysis_window_start_utc",
    "analysis_window_end_utc",
    "data_cutoff_utc",
    "is_new_in_cycle",
    "is_active_in_window",
}
COMMENT_REQUIRED_COLUMNS = {"comment_id", "video_id", "event_time_utc", "text"}


@dataclass(frozen=True)
class DailyRagSidecarBuildConfig:
    daily_events_path: str | Path
    output_dir: str | Path
    comments_path: str | Path
    cycle_window_inventory_path: str | Path
    daily_scores_path: str | Path | None = None
    daily_detector_manifest_path: str | Path | None = None
    cycle_signal_series_path: str | Path | None = None
    cycle_stateful_context_path: str | Path | None = None
    max_comments_per_context_unit: int = 25
    run_id: str | None = None
    notes: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    run_llm: bool = False
    run_serper: bool = False
    use_embeddings: bool = False
    use_vectorstore: bool = False
    run_g1: bool = False
    run_g2: bool = False

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "DailyRagSidecarBuildConfig":
        config_payload = payload.get("daily_rag_sidecars", payload)
        if not isinstance(config_payload, dict):
            raise ValueError("daily_rag_sidecars config section must be an object.")
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(config_payload) - allowed)
        if unknown:
            raise ValueError(f"Unknown daily RAG sidecar config fields: {unknown}")
        params = config_payload.get("params") or {}
        if not isinstance(params, dict):
            raise ValueError("params must be an object.")
        config_payload = {**config_payload, "params": params}
        return cls(**config_payload)

    def validate(self) -> None:
        if self.max_comments_per_context_unit < 1:
            raise ValueError("max_comments_per_context_unit must be >= 1.")
        forbidden = {
            "run_llm": self.run_llm,
            "run_serper": self.run_serper,
            "use_embeddings": self.use_embeddings,
            "use_vectorstore": self.use_vectorstore,
            "run_g1": self.run_g1,
            "run_g2": self.run_g2,
        }
        enabled = [name for name, value in forbidden.items() if value]
        if enabled:
            raise ValueError(
                "Daily RAG sidecars are non-generative. These flags must remain false: "
                + ", ".join(enabled)
            )


def _read_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Required JSON artifact not found: {p}")
    payload = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must contain an object: {p}")
    return payload


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Required JSONL artifact not found: {p}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"JSONL line {line_number} must contain an object: {p}")
        rows.append(payload)
    return rows


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(_json_safe(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(_json_safe(row), ensure_ascii=False) for row in rows]
    p.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, pd.Timestamp):
        return _isoformat(value)
    if value is pd.NA:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


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


def _normalize_utc_column(df: pd.DataFrame, column: str) -> None:
    df[column] = pd.to_datetime(df[column], utc=True, errors="coerce")
    if df[column].isna().any():
        raise ValueError(f"Column {column!r} contains invalid timestamps.")


def _to_utc(value: Any) -> pd.Timestamp:
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"Invalid UTC timestamp: {value!r}")
    return pd.Timestamp(ts)


def _isoformat(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).isoformat()


def _bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def read_table(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() == ".parquet" or p.is_dir():
        return pd.read_parquet(p)
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    raise ValueError(f"Unsupported file format: {p}")


def _prepare_events(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    _require_columns(df, EVENT_REQUIRED_COLUMNS, "daily frequency events")
    for col in ["analysis_window_start_utc", "analysis_window_end_utc", "data_cutoff_utc"]:
        _normalize_utc_column(df, col)
    return df.sort_values(["cycle_index", "cycle_id", "daily_event_id"]).reset_index(drop=True)


def _prepare_window_inventory(path: str | Path) -> pd.DataFrame:
    df = read_table(path)
    _require_columns(df, WINDOW_REQUIRED_COLUMNS, "cycle_window_inventory")
    for col in ["event_time_utc", "analysis_window_start_utc", "analysis_window_end_utc", "data_cutoff_utc"]:
        _normalize_utc_column(df, col)
    for col in ["is_new_in_cycle", "is_active_in_window"]:
        df[col] = _bool_series(df[col])
    return df


def _prepare_comments(path: str | Path) -> pd.DataFrame:
    df = read_table(path)
    _require_columns(df, COMMENT_REQUIRED_COLUMNS, "Gold comments")
    if df["comment_id"].duplicated().any():
        duplicate_count = int(df["comment_id"].duplicated().sum())
        raise ValueError(f"Gold comments must have unique comment_id. Duplicates: {duplicate_count}")
    _normalize_utc_column(df, "event_time_utc")
    for col in [
        "text_clean",
        "author_id",
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
    df["is_reply"] = _bool_series(df["is_reply"].fillna(False))
    return df


def _make_run_id(config: DailyRagSidecarBuildConfig) -> str:
    return "drun_" + _short_hash(
        config.daily_events_path,
        config.cycle_window_inventory_path,
        config.comments_path,
    )


def _make_daily_rag_event_id(event: pd.Series, *, run_id: str) -> str:
    return "drage_" + _short_hash(
        run_id,
        event["daily_event_id"],
        event["cycle_id"],
        _isoformat(event["analysis_window_start_utc"]),
        _isoformat(event["analysis_window_end_utc"]),
        _isoformat(event["data_cutoff_utc"]),
    )


def _make_context_unit_id(
    *,
    daily_rag_event_id: str,
    video_id: str,
    context_type: str,
    order: int,
    comment_ids: list[str],
) -> str:
    return "dctx_" + _short_hash(
        daily_rag_event_id,
        video_id,
        context_type,
        order,
        comment_ids[0] if comment_ids else "",
        comment_ids[-1] if comment_ids else "",
    )


def _output_paths(output_dir: str | Path) -> dict[str, str]:
    root = Path(output_dir)
    return {
        "daily_event_evidence_packages": (root / DAILY_EVENT_EVIDENCE_PACKAGES_FILE).as_posix(),
        "daily_event_comment_inventory": (root / DAILY_EVENT_COMMENT_INVENTORY_FILE).as_posix(),
        "daily_event_video_map": (root / DAILY_EVENT_VIDEO_MAP_FILE).as_posix(),
        "daily_event_thread_map": (root / DAILY_EVENT_THREAD_MAP_FILE).as_posix(),
        "daily_rag_context_units": (root / DAILY_RAG_CONTEXT_UNITS_FILE).as_posix(),
        "daily_context_unit_comment_map": (root / DAILY_CONTEXT_UNIT_COMMENT_MAP_FILE).as_posix(),
        "daily_rag_sidecars_manifest": (root / DAILY_RAG_SIDECARS_MANIFEST_FILE).as_posix(),
        "readme": (root / README_FILE).as_posix(),
    }


def build_daily_event_comment_inventory(
    *,
    events: pd.DataFrame,
    window_inventory: pd.DataFrame,
    comments: pd.DataFrame,
    run_id: str,
    comments_path: str | Path,
    window_inventory_path: str | Path,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    gold_lookup = comments.drop_duplicates("comment_id").copy()
    for _, event in events.iterrows():
        cycle_rows = window_inventory.loc[window_inventory["cycle_id"] == event["cycle_id"]].copy()
        if cycle_rows.empty:
            continue
        cycle_rows = cycle_rows.loc[
            cycle_rows["is_active_in_window"] | cycle_rows["is_new_in_cycle"]
        ].copy()
        cycle_rows["daily_rag_event_id"] = _make_daily_rag_event_id(event, run_id=run_id)
        cycle_rows["daily_event_id"] = event["daily_event_id"]
        cycle_rows["detector_name"] = event["detector_name"]
        cycle_rows["signal_name"] = event["signal_name"]
        cycle_rows["is_alert_evidence"] = cycle_rows["is_new_in_cycle"]
        cycle_rows["is_validation_context"] = cycle_rows["is_active_in_window"]
        cycle_rows["available_at_cycle"] = cycle_rows["event_time_utc"] < event["data_cutoff_utc"]
        cycle_rows["relative_to_data_cutoff"] = cycle_rows["available_at_cycle"].map(
            {True: "before_data_cutoff", False: "at_or_after_data_cutoff"}
        )
        same_cycle = cycle_rows["first_seen_cycle_id"].astype(str) == str(event["cycle_id"])
        cycle_rows["temporal_role"] = "validation_context_prior"
        cycle_rows.loc[same_cycle & ~cycle_rows["is_alert_evidence"], "temporal_role"] = (
            "validation_context_same_cycle"
        )
        cycle_rows.loc[cycle_rows["is_alert_evidence"], "temporal_role"] = "alert_evidence"
        cycle_rows["source_dataset"] = _normalize_path(comments_path)
        cycle_rows["source_artifact"] = _normalize_path(window_inventory_path)

        merged = cycle_rows.merge(
            gold_lookup,
            on="comment_id",
            how="left",
            suffixes=("_window", "_gold"),
            validate="many_to_one",
        )
        merged["video_id"] = merged["video_id_window"].combine_first(merged["video_id_gold"])
        merged["event_time_utc"] = merged["event_time_utc_window"]
        merged["cycle_index"] = int(event["cycle_index"])
        parts.append(merged)

    if not parts:
        return pd.DataFrame()
    inventory = pd.concat(parts, ignore_index=True)
    inventory = inventory.sort_values(
        ["cycle_index", "daily_event_id", "video_id", "event_time_utc", "comment_id"]
    ).reset_index(drop=True)
    inventory["order_in_daily_event"] = inventory.groupby("daily_event_id").cumcount() + 1
    inventory["order_in_daily_event_video"] = (
        inventory.groupby(["daily_event_id", "video_id"]).cumcount() + 1
    )
    inventory["is_reply"] = _bool_series(inventory["is_reply"].fillna(False))
    inventory["parent_comment_id"] = inventory["reply_to_comment_id"]
    inventory["root_comment_id"] = inventory["reply_to_comment_id"].where(
        inventory["reply_to_comment_id"].notna(),
        inventory["comment_id"],
    )
    event_comment_pairs = set(zip(inventory["daily_event_id"], inventory["comment_id"]))
    inventory["parent_in_inventory"] = inventory.apply(
        lambda row: (
            (row["daily_event_id"], row["parent_comment_id"]) in event_comment_pairs
            if pd.notna(row["parent_comment_id"])
            else False
        ),
        axis=1,
    )
    inventory["artifact_version"] = DAILY_RAG_SIDECAR_ARTIFACT_VERSION
    field_order = [
        "daily_rag_event_id",
        "daily_event_id",
        "cycle_id",
        "cycle_index",
        "comment_id",
        "video_id",
        "event_time_utc",
        "text",
        "text_clean",
        "is_alert_evidence",
        "is_validation_context",
        "temporal_role",
        "available_at_cycle",
        "relative_to_data_cutoff",
        "source_dataset",
        "source_artifact",
        "first_seen_cycle_id",
        "analysis_window_start_utc",
        "analysis_window_end_utc",
        "data_cutoff_utc",
        "order_in_daily_event",
        "order_in_daily_event_video",
        "is_reply",
        "parent_comment_id",
        "root_comment_id",
        "parent_in_inventory",
        "author_id",
        "likes",
        "emoji_count",
        "exclamation_count",
        "question_count",
        "caps_ratio",
        "link_count",
        "token_count",
        "is_probable_spam",
        "artifact_version",
    ]
    for col in field_order:
        if col not in inventory.columns:
            inventory[col] = pd.NA
    return inventory[field_order]


def build_daily_event_video_map(inventory: pd.DataFrame) -> pd.DataFrame:
    if inventory.empty:
        return pd.DataFrame()
    out = (
        inventory.groupby(["daily_rag_event_id", "daily_event_id", "cycle_id", "video_id"], dropna=False)
        .agg(
            alert_evidence_comment_count=("is_alert_evidence", "sum"),
            validation_context_comment_count=("is_validation_context", "sum"),
            first_comment_time_utc=("event_time_utc", "min"),
            last_comment_time_utc=("event_time_utc", "max"),
        )
        .reset_index()
    )
    out["alert_evidence_comment_count"] = out["alert_evidence_comment_count"].astype(int)
    out["validation_context_comment_count"] = out["validation_context_comment_count"].astype(int)
    out["video_context_role"] = "validation_context_only"
    out.loc[out["alert_evidence_comment_count"] > 0, "video_context_role"] = (
        "alert_and_validation_context"
    )
    out["artifact_version"] = DAILY_RAG_SIDECAR_ARTIFACT_VERSION
    return out.sort_values(
        ["daily_event_id", "alert_evidence_comment_count", "validation_context_comment_count", "video_id"],
        ascending=[True, False, False, True],
    )


def build_daily_event_thread_map(inventory: pd.DataFrame) -> pd.DataFrame:
    if inventory.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    group_cols = [
        "daily_rag_event_id",
        "daily_event_id",
        "cycle_id",
        "video_id",
        "root_comment_id",
    ]
    for keys, group in inventory.groupby(group_cols, dropna=False, sort=False):
        daily_rag_event_id, daily_event_id, cycle_id, video_id, root_comment_id = keys
        group = group.sort_values(["event_time_utc", "comment_id"])
        comment_ids = [str(value) for value in group["comment_id"].tolist()]
        reply_count = int(group["is_reply"].sum())
        root_present = str(root_comment_id) in {str(value) for value in group["comment_id"].tolist()}
        if reply_count > 0 and not root_present:
            thread_status = "partial_parent_outside_inventory"
        elif reply_count > 0:
            thread_status = "thread_with_replies"
        else:
            thread_status = "root_or_singleton"
        rows.append(
            {
                "daily_rag_event_id": daily_rag_event_id,
                "daily_event_id": daily_event_id,
                "cycle_id": cycle_id,
                "video_id": video_id,
                "root_comment_id": root_comment_id,
                "root_comment_present": bool(root_present),
                "comment_count": int(group["comment_id"].nunique()),
                "reply_count": reply_count,
                "has_replies": bool(reply_count > 0),
                "first_comment_time_utc": group["event_time_utc"].min(),
                "last_comment_time_utc": group["event_time_utc"].max(),
                "comment_ids": "|".join(comment_ids),
                "thread_status": thread_status,
                "artifact_version": DAILY_RAG_SIDECAR_ARTIFACT_VERSION,
            }
        )
    return pd.DataFrame(rows)


def _context_text(group: pd.DataFrame) -> str:
    lines = []
    for _, row in group.iterrows():
        lines.append(
            f"[{row['comment_id']}|{row['temporal_role']}|{_isoformat(row['event_time_utc'])}] "
            f"{row.get('text', '')}"
        )
    return "\n".join(lines)


def _chunk_group(group: pd.DataFrame, max_size: int) -> list[pd.DataFrame]:
    return [group.iloc[start : start + max_size].copy() for start in range(0, len(group), max_size)]


def _context_unit_record(
    *,
    context_unit_id: str,
    context_order: int,
    context_type: str,
    group: pd.DataFrame,
    comment_ids: list[str],
    selection_reason: str,
    source_inventory_path: str | Path,
) -> dict[str, Any]:
    first = group.iloc[0]
    alert_count = int(group["is_alert_evidence"].sum())
    validation_count = int(group["is_validation_context"].sum())
    contains_alert = alert_count > 0
    contains_validation = validation_count > 0
    non_alert_validation_count = int((group["is_validation_context"] & ~group["is_alert_evidence"]).sum())
    if contains_alert and non_alert_validation_count == 0:
        context_role = "alert_evidence_unit"
    elif not contains_alert and contains_validation:
        context_role = "validation_context_unit"
    else:
        context_role = "mixed_unit"
    temporal_roles = sorted(str(value) for value in group["temporal_role"].dropna().unique())
    if len(temporal_roles) == 1:
        role = temporal_roles[0]
        temporal_scope = {
            "alert_evidence": "alert_cycle",
            "validation_context_prior": "prior_window",
            "validation_context_same_cycle": "same_cycle_context",
        }.get(role, role)
    else:
        temporal_scope = "mixed_temporal_scope"
    return {
        "context_unit_id": context_unit_id,
        "daily_rag_event_id": first["daily_rag_event_id"],
        "daily_event_id": first["daily_event_id"],
        "cycle_id": first["cycle_id"],
        "cycle_index": int(first["cycle_index"]),
        "video_id": first["video_id"],
        "context_order_in_daily_event": context_order,
        "context_type": context_type,
        "temporal_scope": temporal_scope,
        "context_role": context_role,
        "contains_alert_evidence": contains_alert,
        "contains_validation_context": contains_validation,
        "alert_evidence_comment_count": alert_count,
        "validation_context_comment_count": validation_count,
        "comment_ids": comment_ids,
        "comment_count": int(len(comment_ids)),
        "time_start_utc": _isoformat(group["event_time_utc"].min()),
        "time_end_utc": _isoformat(group["event_time_utc"].max()),
        "text_block": _context_text(group),
        "is_alert_evidence_unit": context_role == "alert_evidence_unit",
        "is_validation_context_unit": context_role == "validation_context_unit",
        "contains_replies": bool(group["is_reply"].any()),
        "selection_reason": selection_reason,
        "source_inventory_path": _normalize_path(source_inventory_path),
        "artifact_version": DAILY_RAG_SIDECAR_ARTIFACT_VERSION,
    }


def _extend_context_unit_comment_map(
    rows: list[dict[str, Any]],
    unit: dict[str, Any],
    group: pd.DataFrame,
) -> None:
    for order, (_, row) in enumerate(group.iterrows(), start=1):
        rows.append(
            {
                "context_unit_id": unit["context_unit_id"],
                "daily_rag_event_id": row["daily_rag_event_id"],
                "daily_event_id": row["daily_event_id"],
                "cycle_id": row["cycle_id"],
                "video_id": row["video_id"],
                "comment_id": row["comment_id"],
                "order_in_context_unit": order,
                "event_time_utc": row["event_time_utc"],
                "is_alert_evidence": row["is_alert_evidence"],
                "is_validation_context": row["is_validation_context"],
                "temporal_role": row["temporal_role"],
                "context_type": unit["context_type"],
                "artifact_version": DAILY_RAG_SIDECAR_ARTIFACT_VERSION,
            }
        )


def build_daily_context_units(
    inventory: pd.DataFrame,
    *,
    source_inventory_path: str | Path,
    max_comments_per_context_unit: int,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    units: list[dict[str, Any]] = []
    comment_map_rows: list[dict[str, Any]] = []
    if inventory.empty:
        return units, pd.DataFrame()

    assigned_pairs: set[tuple[str, str]] = set()
    order_by_event: dict[str, int] = {}
    thread_groups = inventory.groupby(
        ["daily_rag_event_id", "video_id", "temporal_role", "root_comment_id"],
        dropna=False,
        sort=False,
    )
    for (daily_rag_event_id, video_id, temporal_role, root_comment_id), group in thread_groups:
        group = group.sort_values(["event_time_utc", "comment_id"])
        is_thread = bool(group["is_reply"].any() or len(group) > 1)
        if not is_thread:
            continue
        chunks = _chunk_group(group, max_comments_per_context_unit)
        for chunk_index, chunk in enumerate(chunks, start=1):
            order = order_by_event.get(str(daily_rag_event_id), 0) + 1
            order_by_event[str(daily_rag_event_id)] = order
            comment_ids = [str(value) for value in chunk["comment_id"].tolist()]
            context_type = "thread" if len(chunks) == 1 else "thread_part"
            unit = _context_unit_record(
                context_unit_id=_make_context_unit_id(
                    daily_rag_event_id=str(daily_rag_event_id),
                    video_id=str(video_id),
                    context_type=context_type,
                    order=order,
                    comment_ids=comment_ids,
                ),
                context_order=order,
                context_type=context_type,
                group=chunk,
                comment_ids=comment_ids,
                selection_reason=(
                    f"deterministic_thread_group_{temporal_role}"
                    if len(chunks) == 1
                    else f"deterministic_thread_group_{temporal_role}_part_{chunk_index}"
                ),
                source_inventory_path=source_inventory_path,
            )
            units.append(unit)
            _extend_context_unit_comment_map(comment_map_rows, unit, chunk)
            assigned_pairs.update((str(daily_rag_event_id), item) for item in comment_ids)

    remaining = inventory.loc[
        ~inventory.apply(
            lambda row: (str(row["daily_rag_event_id"]), str(row["comment_id"])) in assigned_pairs,
            axis=1,
        )
    ].copy()
    for (daily_rag_event_id, video_id, temporal_role), group in remaining.groupby(
        ["daily_rag_event_id", "video_id", "temporal_role"], dropna=False, sort=False
    ):
        group = group.sort_values(["event_time_utc", "comment_id"])
        chunks = _chunk_group(group, max_comments_per_context_unit)
        for chunk_index, chunk in enumerate(chunks, start=1):
            order = order_by_event.get(str(daily_rag_event_id), 0) + 1
            order_by_event[str(daily_rag_event_id)] = order
            comment_ids = [str(value) for value in chunk["comment_id"].tolist()]
            context_type = "video_time_block"
            unit = _context_unit_record(
                context_unit_id=_make_context_unit_id(
                    daily_rag_event_id=str(daily_rag_event_id),
                    video_id=str(video_id),
                    context_type=context_type,
                    order=order,
                    comment_ids=comment_ids,
                ),
                context_order=order,
                context_type=context_type,
                group=chunk,
                comment_ids=comment_ids,
                selection_reason=f"deterministic_video_time_block_{temporal_role}_{chunk_index}",
                source_inventory_path=source_inventory_path,
            )
            units.append(unit)
            _extend_context_unit_comment_map(comment_map_rows, unit, chunk)

    return units, pd.DataFrame(comment_map_rows)


def build_daily_event_evidence_packages(
    *,
    events: pd.DataFrame,
    inventory: pd.DataFrame,
    video_map: pd.DataFrame,
    context_units: list[dict[str, Any]],
    output_paths: dict[str, str],
    source_artifacts: dict[str, str | None],
    run_id: str,
    created_at_utc: str,
) -> list[dict[str, Any]]:
    packages: list[dict[str, Any]] = []
    context_counts: dict[str, int] = {}
    for unit in context_units:
        key = str(unit["daily_event_id"])
        context_counts[key] = context_counts.get(key, 0) + 1

    for _, event in events.iterrows():
        daily_event_id = str(event["daily_event_id"])
        daily_rag_event_id = _make_daily_rag_event_id(event, run_id=run_id)
        event_inventory = inventory.loc[inventory["daily_event_id"] == daily_event_id]
        event_video_map = video_map.loc[video_map["daily_event_id"] == daily_event_id]
        video_ids = sorted(str(value) for value in event_video_map["video_id"].dropna().unique())
        packages.append(
            {
                "daily_rag_event_id": daily_rag_event_id,
                "daily_event_id": daily_event_id,
                "cycle_id": event["cycle_id"],
                "cycle_index": int(event["cycle_index"]),
                "detector_name": event["detector_name"],
                "signal_name": event["signal_name"],
                "signal_value": event["signal_value"],
                "baseline_mean": event["baseline_mean"],
                "ratio_to_baseline": event["ratio_to_baseline"],
                "delta_value": event["delta_value"],
                "pct_change_value": event["pct_change_value"],
                "threshold_value": event["threshold_value"],
                "trigger_reason": event["trigger_reason"],
                "analysis_window_start_utc": _isoformat(event["analysis_window_start_utc"]),
                "analysis_window_end_utc": _isoformat(event["analysis_window_end_utc"]),
                "data_cutoff_utc": _isoformat(event["data_cutoff_utc"]),
                "alert_evidence_comment_count": int(event_inventory["is_alert_evidence"].sum()),
                "validation_context_comment_count": int(event_inventory["is_validation_context"].sum()),
                "video_ids": video_ids,
                "context_unit_count": int(context_counts.get(daily_event_id, 0)),
                "source_artifacts": source_artifacts,
                "output_artifacts": output_paths,
                "run_id": run_id,
                "artifact_version": DAILY_RAG_SIDECAR_ARTIFACT_VERSION,
                "created_at_utc": created_at_utc,
            }
        )
    return packages


def _format_timestamp_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = out[col].map(_isoformat)
    return out


def _coverage_checks(
    *,
    events: pd.DataFrame,
    inventory: pd.DataFrame,
    comments: pd.DataFrame,
    context_comment_map: pd.DataFrame,
) -> dict[str, Any]:
    event_ids = set(events["daily_event_id"].astype(str))
    package_ready_event_ids = set(inventory["daily_event_id"].astype(str)) if not inventory.empty else set()
    gold_ids = set(comments["comment_id"].astype(str))
    inventory_pairs = (
        set(zip(inventory["daily_event_id"], inventory["comment_id"].astype(str)))
        if not inventory.empty
        else set()
    )
    context_pairs = (
        set(zip(context_comment_map["daily_event_id"], context_comment_map["comment_id"].astype(str)))
        if not context_comment_map.empty
        else set()
    )
    duplicate_pairs = (
        int(inventory.duplicated(["daily_event_id", "comment_id"]).sum())
        if not inventory.empty
        else 0
    )
    return {
        "events_detected": int(len(event_ids)),
        "events_with_inventory": int(len(package_ready_event_ids)),
        "events_missing_inventory": sorted(event_ids.difference(package_ready_event_ids)),
        "inventory_rows": int(len(inventory)),
        "inventory_unique_event_comment_pairs": int(len(inventory_pairs)),
        "duplicate_event_comment_pairs": duplicate_pairs,
        "comment_ids_missing_from_gold": int(
            len(set(inventory["comment_id"].astype(str)).difference(gold_ids))
            if not inventory.empty
            else 0
        ),
        "context_event_comment_pairs": int(len(context_pairs)),
        "comments_without_context_unit": int(len(inventory_pairs.difference(context_pairs))),
        "context_comments_not_in_inventory": int(len(context_pairs.difference(inventory_pairs))),
        "all_comments_have_context_unit": len(inventory_pairs.difference(context_pairs)) == 0,
    }


def _temporal_checks(inventory: pd.DataFrame) -> dict[str, Any]:
    if inventory.empty:
        return {
            "future_leak_count": 0,
            "outside_analysis_window_count": 0,
            "alert_not_new_count": 0,
            "validation_not_active_count": 0,
        }
    event_time = pd.to_datetime(inventory["event_time_utc"], utc=True)
    data_cutoff = pd.to_datetime(inventory["data_cutoff_utc"], utc=True)
    window_start = pd.to_datetime(inventory["analysis_window_start_utc"], utc=True)
    window_end = pd.to_datetime(inventory["analysis_window_end_utc"], utc=True)
    future_leak = event_time >= data_cutoff
    outside_window = (event_time < window_start) | (event_time >= window_end)
    alert_not_new = inventory["is_alert_evidence"] & (
        inventory["first_seen_cycle_id"].astype(str) != inventory["cycle_id"].astype(str)
    )
    validation_not_active = ~inventory["is_validation_context"]
    return {
        "future_leak_count": int(future_leak.sum()),
        "outside_analysis_window_count": int(outside_window.sum()),
        "alert_not_new_count": int(alert_not_new.sum()),
        "validation_not_active_count": int(validation_not_active.sum()),
    }


def _unit_checks(context_units: list[dict[str, Any]], context_comment_map: pd.DataFrame) -> dict[str, Any]:
    empty_units = [unit["context_unit_id"] for unit in context_units if int(unit["comment_count"]) < 1]
    mixed_units = [
        unit["context_unit_id"]
        for unit in context_units
        if unit.get("context_role") == "mixed_unit"
    ]
    role_counts = Counter(str(unit.get("context_role")) for unit in context_units)
    temporal_scope_counts = Counter(str(unit.get("temporal_scope")) for unit in context_units)
    return {
        "context_unit_count": int(len(context_units)),
        "empty_context_unit_count": int(len(empty_units)),
        "empty_context_unit_ids": empty_units,
        "mixed_unit_count": int(len(mixed_units)),
        "mixed_context_unit_ids_sample": mixed_units[:10],
        "context_role_counts": dict(role_counts),
        "temporal_scope_counts": dict(temporal_scope_counts),
        "context_unit_comment_rows": int(len(context_comment_map)),
    }


def build_daily_rag_sidecars_manifest(
    *,
    config: DailyRagSidecarBuildConfig,
    run_id: str,
    created_at_utc: str,
    events: pd.DataFrame,
    inventory: pd.DataFrame,
    comments: pd.DataFrame,
    video_map: pd.DataFrame,
    thread_map: pd.DataFrame,
    context_units: list[dict[str, Any]],
    context_comment_map: pd.DataFrame,
    output_paths: dict[str, str],
    source_artifacts: dict[str, str | None],
) -> dict[str, Any]:
    coverage = _coverage_checks(
        events=events,
        inventory=inventory,
        comments=comments,
        context_comment_map=context_comment_map,
    )
    temporal = _temporal_checks(inventory)
    units = _unit_checks(context_units, context_comment_map)
    validation_status = "passed"
    validation_errors: list[str] = []
    for key, expected in {
        "events_missing_inventory": [],
        "duplicate_event_comment_pairs": 0,
        "comment_ids_missing_from_gold": 0,
        "comments_without_context_unit": 0,
        "context_comments_not_in_inventory": 0,
    }.items():
        value = coverage[key]
        if value != expected:
            validation_status = "failed"
            validation_errors.append(f"{key}={value}")
    for key in [
        "future_leak_count",
        "outside_analysis_window_count",
        "alert_not_new_count",
        "validation_not_active_count",
    ]:
        if temporal[key] != 0:
            validation_status = "failed"
            validation_errors.append(f"{key}={temporal[key]}")
    if units["empty_context_unit_count"] != 0:
        validation_status = "failed"
        validation_errors.append(f"empty_context_unit_count={units['empty_context_unit_count']}")

    return {
        "run_id": run_id,
        "created_at_utc": created_at_utc,
        "pipeline_stage": "daily_rag_sidecar_evidence_preparation",
        "mode": "sidecars_only_no_generation_no_external_retrieval",
        "artifact_version": DAILY_RAG_SIDECAR_ARTIFACT_VERSION,
        "source_artifacts": source_artifacts,
        "output_paths": output_paths,
        "daily_rag_event_id_formula": (
            "drage_ + sha1(run_id|daily_event_id|cycle_id|analysis_window_start_utc|"
            "analysis_window_end_utc|data_cutoff_utc)[:12]"
        ),
        "hierarchy": "daily_event_id -> cycle_id -> video_id -> context_unit_id -> comment_id",
        "temporal_rule": {
            "alert_evidence_comments": "comments with is_new_in_cycle=true for detected cycle_id",
            "validation_context_comments": (
                "comments with is_active_in_window=true and "
                "analysis_window_start_utc <= event_time_utc < analysis_window_end_utc"
            ),
            "future_leak_guard": "event_time_utc < data_cutoff_utc",
        },
        "context_policy": {
            "chunking": "deterministic_thread_or_video_time_block_partitioned_by_temporal_role",
            "max_comments_per_context_unit": config.max_comments_per_context_unit,
            "selection": "full_coverage_no_llm_no_ranking",
            "unit_semantics": (
                "context units are manageable traceable fragments inside the available "
                "analysis window; they are not required to span the full 3-day window"
            ),
            "role_partitioning": (
                "alert_evidence and validation_context_prior/same_cycle comments are "
                "partitioned before chunking to avoid mixed units whenever possible"
            ),
        },
        "counts": {
            "events_processed": int(events["daily_event_id"].nunique()),
            "comments_inventoried": int(len(inventory)),
            "alert_evidence_comments": (
                int(inventory["is_alert_evidence"].sum()) if not inventory.empty else 0
            ),
            "validation_context_comments": (
                int(inventory["is_validation_context"].sum()) if not inventory.empty else 0
            ),
            "videos_associated": int(video_map["video_id"].nunique()) if not video_map.empty else 0,
            "event_video_rows": int(len(video_map)),
            "thread_rows": int(len(thread_map)),
            "context_units": int(len(context_units)),
            "context_unit_comment_rows": int(len(context_comment_map)),
        },
        "validations": {
            "status": validation_status,
            "errors": validation_errors,
            "coverage": coverage,
            "temporal": temporal,
            "context_units": units,
            "external_calls": {
                "llm": 0,
                "serper": 0,
                "embeddings": False,
                "vectorstore": False,
                "g1": False,
                "g2": False,
            },
        },
        "limitations": [
            "Estos sidecars preparan evidencia interna; no validan eventos con LLM.",
            "No se ejecuta evidencia externa, Serper, embeddings ni vectorstore.",
            "La evidencia causal corresponde a comentarios nuevos del ciclo detectado.",
            "El contexto de validacion incluye pasado reciente dentro de la ventana analitica.",
        ],
        "notes": config.notes,
        "params": config.params,
    }


def _read_optional_json(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    return _read_json(p)


def _read_optional_jsonl(path: str | None) -> list[dict[str, Any]] | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    return _read_jsonl(p)


def _source_artifacts(config: DailyRagSidecarBuildConfig) -> dict[str, str | None]:
    return {
        "daily_events_path": _normalize_path(config.daily_events_path),
        "daily_scores_path": _normalize_path(config.daily_scores_path),
        "daily_detector_manifest_path": _normalize_path(config.daily_detector_manifest_path),
        "cycle_signal_series_path": _normalize_path(config.cycle_signal_series_path),
        "cycle_window_inventory_path": _normalize_path(config.cycle_window_inventory_path),
        "cycle_stateful_context_path": _normalize_path(config.cycle_stateful_context_path),
        "comments_path": _normalize_path(config.comments_path),
    }


def _write_readme(path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        """# Sidecars RAG diarios

Estos artefactos preparan evidencia interna para eventos detectados por
`daily_frequency_baseline`. No ejecutan validacion generativa, no consultan
fuentes externas, no crean embeddings y no modifican el pipeline base.

## Identificadores

`daily_event_id` identifica la alerta diaria producida por el detector diario.
No se mezcla con los `event_id` usados por los sidecars RAG previos. Cuando se
necesita una llave interna de esta familia de sidecars se usa
`daily_rag_event_id`, preservando siempre `daily_event_id`.

Jerarquia:

```text
daily_event_id
-> cycle_id
-> video_id
-> context_unit_id
-> comment_id
```

## Evidencia causal y contexto de validacion

La evidencia causal de alerta son los comentarios nuevos del ciclo detectado:
`is_new_in_cycle = true`. Estos comentarios explican el incremento de
`new_comment_count`.

El contexto de validacion son los comentarios activos en la ventana analitica:

```text
analysis_window_start_utc <= event_time_utc < analysis_window_end_utc
event_time_utc < data_cutoff_utc
```

Los comentarios de contexto previo ayudan a interpretar la alerta, pero no deben
presentarse como causa directa del pico diario.

## Unidades de contexto

Una unidad de contexto no representa obligatoriamente los 3 dias completos de la
ventana analitica. Es un fragmento manejable, trazable y citable construido
dentro del contexto disponible.

La politica actual crea unidades deterministicas por `video_id`, hilo o bloque
temporal, y particiona por `temporal_role` antes de agrupar. Esto evita mezclar,
cuando es posible, comentarios de evidencia de alerta con comentarios de
contexto previo.

Campos clave de cada unidad:

- `context_role`: `alert_evidence_unit`, `validation_context_unit` o `mixed_unit`.
- `temporal_scope`: `alert_cycle`, `prior_window`, `same_cycle_context` o
  `mixed_temporal_scope`.
- `contains_alert_evidence` y `contains_validation_context`.
- `alert_evidence_comment_count` y `validation_context_comment_count`.

Si una unidad quedara mezclada, debe aparecer como `context_role = "mixed_unit"`.

## Artefactos

- `daily_event_evidence_packages.jsonl`: paquete por evento diario.
- `daily_event_comment_inventory.csv`: inventario completo de comentarios por evento.
- `daily_event_video_map.csv`: relacion evento-video con conteos y tiempos.
- `daily_event_thread_map.csv`: hilos detectables con metadata disponible.
- `daily_rag_context_units.jsonl`: unidades de contexto deterministicas.
- `daily_context_unit_comment_map.csv`: relacion unidad-comentario.
- `daily_rag_sidecars_manifest.json`: configuracion, fuentes, conteos y validaciones.

## Relacion con G-1/G-2

Estos sidecars son la base para una futura G-1/G-2 sobre eventos diarios. La
fase generativa posterior debera citar `comment_id` y `context_unit_id`, y
mantener separadas evidencia causal y contexto de validacion.

## Limitaciones

- No hay validacion con LLM.
- No hay Serper ni evidencia externa.
- No hay embeddings ni vectorstore.
- La familia diaria no reemplaza los sidecars RAG previos.
""",
        encoding="utf-8",
    )


def write_daily_rag_sidecar_artifacts_from_config(
    config: DailyRagSidecarBuildConfig,
) -> dict[str, Any]:
    config.validate()
    output_root = Path(config.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    output_paths = _output_paths(output_root)
    created_at = _utc_now_iso()
    run_id = config.run_id or _make_run_id(config)

    events = _prepare_events(_read_jsonl(config.daily_events_path))
    window_inventory = _prepare_window_inventory(config.cycle_window_inventory_path)
    comments = _prepare_comments(config.comments_path)
    _read_optional_json(config.daily_detector_manifest_path)
    _read_optional_jsonl(config.daily_scores_path)
    _read_optional_jsonl(config.cycle_signal_series_path)
    _read_optional_json(config.cycle_stateful_context_path)

    inventory = build_daily_event_comment_inventory(
        events=events,
        window_inventory=window_inventory,
        comments=comments,
        run_id=run_id,
        comments_path=config.comments_path,
        window_inventory_path=config.cycle_window_inventory_path,
    )
    video_map = build_daily_event_video_map(inventory)
    thread_map = build_daily_event_thread_map(inventory)
    context_units, context_comment_map = build_daily_context_units(
        inventory,
        source_inventory_path=output_paths["daily_event_comment_inventory"],
        max_comments_per_context_unit=config.max_comments_per_context_unit,
    )
    source_artifacts = _source_artifacts(config)
    packages = build_daily_event_evidence_packages(
        events=events,
        inventory=inventory,
        video_map=video_map,
        context_units=context_units,
        output_paths=output_paths,
        source_artifacts=source_artifacts,
        run_id=run_id,
        created_at_utc=created_at,
    )
    manifest = build_daily_rag_sidecars_manifest(
        config=config,
        run_id=run_id,
        created_at_utc=created_at,
        events=events,
        inventory=inventory,
        comments=comments,
        video_map=video_map,
        thread_map=thread_map,
        context_units=context_units,
        context_comment_map=context_comment_map,
        output_paths=output_paths,
        source_artifacts=source_artifacts,
    )

    timestamp_cols = [
        "event_time_utc",
        "analysis_window_start_utc",
        "analysis_window_end_utc",
        "data_cutoff_utc",
        "first_comment_time_utc",
        "last_comment_time_utc",
    ]
    _write_jsonl(output_paths["daily_event_evidence_packages"], packages)
    _format_timestamp_columns(inventory, timestamp_cols).to_csv(
        output_paths["daily_event_comment_inventory"], index=False
    )
    _format_timestamp_columns(video_map, timestamp_cols).to_csv(
        output_paths["daily_event_video_map"], index=False
    )
    _format_timestamp_columns(thread_map, timestamp_cols).to_csv(
        output_paths["daily_event_thread_map"], index=False
    )
    _write_jsonl(output_paths["daily_rag_context_units"], context_units)
    _format_timestamp_columns(context_comment_map, timestamp_cols).to_csv(
        output_paths["daily_context_unit_comment_map"], index=False
    )
    _write_json(output_paths["daily_rag_sidecars_manifest"], manifest)
    _write_readme(output_paths["readme"])

    if manifest["validations"]["status"] != "passed":
        raise ValueError(
            "Daily RAG sidecar validation failed: "
            + "; ".join(manifest["validations"]["errors"])
        )

    return {
        "run_id": run_id,
        "artifact_version": DAILY_RAG_SIDECAR_ARTIFACT_VERSION,
        "output_paths": output_paths,
        **manifest["counts"],
        "validation_status": manifest["validations"]["status"],
        "future_leak_count": manifest["validations"]["temporal"]["future_leak_count"],
    }


def write_daily_rag_sidecar_artifacts(
    **kwargs: Any,
) -> dict[str, Any]:
    return write_daily_rag_sidecar_artifacts_from_config(
        DailyRagSidecarBuildConfig(**kwargs)
    )


__all__ = [
    "DAILY_CONTEXT_UNIT_COMMENT_MAP_FILE",
    "DAILY_EVENT_COMMENT_INVENTORY_FILE",
    "DAILY_EVENT_EVIDENCE_PACKAGES_FILE",
    "DAILY_EVENT_THREAD_MAP_FILE",
    "DAILY_EVENT_VIDEO_MAP_FILE",
    "DAILY_RAG_CONTEXT_UNITS_FILE",
    "DAILY_RAG_SIDECAR_ARTIFACT_VERSION",
    "DAILY_RAG_SIDECARS_MANIFEST_FILE",
    "DailyRagSidecarBuildConfig",
    "build_daily_context_units",
    "build_daily_event_comment_inventory",
    "build_daily_event_evidence_packages",
    "build_daily_event_thread_map",
    "build_daily_event_video_map",
    "read_table",
    "write_daily_rag_sidecar_artifacts",
    "write_daily_rag_sidecar_artifacts_from_config",
]
