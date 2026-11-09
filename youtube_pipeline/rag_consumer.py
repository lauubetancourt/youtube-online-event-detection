from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


RAG_CONSUMER_ARTIFACT_VERSION = "rag_consumer_stub_v1"

EVENT_EVIDENCE_PACKAGES_FILE = "event_evidence_packages.jsonl"
EVENT_COMMENT_INVENTORY_FILE = "event_comment_inventory.csv"
EVENT_VIDEO_MAP_FILE = "event_video_map.csv"
EVENT_THREAD_MAP_FILE = "event_thread_map.csv"
RAG_CONTEXT_UNITS_FILE = "rag_context_units.jsonl"
CONTEXT_UNIT_COMMENT_MAP_FILE = "context_unit_comment_map.csv"
CONTEXT_SELECTION_MANIFEST_FILE = "context_selection_manifest.json"
SIDECAR_README_FILE = "README.md"

RAG_VALIDATION_INPUTS_FILE = "rag_validation_inputs.jsonl"
RAG_CONTEXT_PAYLOADS_FILE = "rag_context_payloads.jsonl"
RAG_CONSUMER_MANIFEST_FILE = "rag_consumer_manifest.json"
RAG_VALIDATION_REPORTS_STUB_FILE = "rag_validation_reports_stub.jsonl"

REQUIRED_PACKAGE_COLUMNS = {
    "event_id",
    "run_id",
    "detector_name",
    "trigger_time_utc",
    "window_start_utc",
    "window_end_utc",
}
REQUIRED_INVENTORY_COLUMNS = {
    "event_id",
    "comment_id",
    "video_id",
    "event_time_utc",
    "trigger_time_utc",
    "window_start_utc",
    "window_end_utc",
    "text",
}
REQUIRED_VIDEO_MAP_COLUMNS = {"event_id", "video_id"}
REQUIRED_CONTEXT_UNIT_COLUMNS = {
    "context_unit_id",
    "event_id",
    "video_id",
    "trigger_time_utc",
    "window_start_utc",
    "window_end_utc",
    "context_type",
    "comment_ids",
}
REQUIRED_CONTEXT_COMMENT_MAP_COLUMNS = {
    "context_unit_id",
    "event_id",
    "comment_id",
    "video_id",
}


@dataclass(frozen=True)
class RagConsumerConfig:
    sidecars_dir: str
    output_dir: str
    run_id: str | None = None
    notes: str | None = None
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "RagConsumerConfig":
        missing = [
            key for key in ["sidecars_dir", "output_dir"] if not payload.get(key)
        ]
        if missing:
            raise ValueError(
                "RAG consumer config missing required fields: " + ", ".join(missing)
            )
        params = payload.get("params") or {}
        if not isinstance(params, dict):
            raise ValueError("params must be an object.")
        return cls(
            sidecars_dir=str(payload["sidecars_dir"]),
            output_dir=str(payload["output_dir"]),
            run_id=payload.get("run_id"),
            notes=payload.get("notes"),
            params=params,
        )


def _extract_config_payload(payload: dict[str, Any]) -> dict[str, Any]:
    nested = payload.get("rag_consumer")
    if nested is None:
        return payload
    if not isinstance(nested, dict):
        raise ValueError("rag_consumer config section must be an object.")
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


def load_rag_consumer_config(
    path: str | Path,
    *,
    overrides: dict[str, Any] | None = None,
) -> RagConsumerConfig:
    p = Path(path)
    payload = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"RAG consumer config must be an object: {p}")
    base = _extract_config_payload(payload)
    merged = _merge_config_payloads(base, overrides or {})
    return RagConsumerConfig.from_mapping(merged)


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


def _isoformat(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).isoformat()


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
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
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


def _read_jsonl(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing JSONL artifact: {p}")
    return pd.read_json(p, lines=True)


def _read_csv(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing CSV artifact: {p}")
    return pd.read_csv(p)


def _read_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing JSON artifact: {p}")
    payload = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must contain an object: {p}")
    return payload


def _sidecar_paths(sidecars_dir: str | Path) -> dict[str, Path]:
    root = Path(sidecars_dir)
    return {
        "event_evidence_packages": root / EVENT_EVIDENCE_PACKAGES_FILE,
        "event_comment_inventory": root / EVENT_COMMENT_INVENTORY_FILE,
        "event_video_map": root / EVENT_VIDEO_MAP_FILE,
        "event_thread_map": root / EVENT_THREAD_MAP_FILE,
        "rag_context_units": root / RAG_CONTEXT_UNITS_FILE,
        "context_unit_comment_map": root / CONTEXT_UNIT_COMMENT_MAP_FILE,
        "context_selection_manifest": root / CONTEXT_SELECTION_MANIFEST_FILE,
        "readme": root / SIDECAR_README_FILE,
    }


def _load_sidecars(config: RagConsumerConfig) -> dict[str, Any]:
    paths = _sidecar_paths(config.sidecars_dir)
    readme_path = paths["readme"]
    if not readme_path.exists():
        raise FileNotFoundError(f"Missing sidecar README: {readme_path}")

    packages = _read_jsonl(paths["event_evidence_packages"])
    inventory = _read_csv(paths["event_comment_inventory"])
    video_map = _read_csv(paths["event_video_map"])
    thread_map = _read_csv(paths["event_thread_map"])
    context_units = _read_jsonl(paths["rag_context_units"])
    context_map = _read_csv(paths["context_unit_comment_map"])
    manifest = _read_json(paths["context_selection_manifest"])

    _require_columns(packages, REQUIRED_PACKAGE_COLUMNS, EVENT_EVIDENCE_PACKAGES_FILE)
    _require_columns(inventory, REQUIRED_INVENTORY_COLUMNS, EVENT_COMMENT_INVENTORY_FILE)
    _require_columns(video_map, REQUIRED_VIDEO_MAP_COLUMNS, EVENT_VIDEO_MAP_FILE)
    _require_columns(context_units, REQUIRED_CONTEXT_UNIT_COLUMNS, RAG_CONTEXT_UNITS_FILE)
    _require_columns(
        context_map, REQUIRED_CONTEXT_COMMENT_MAP_COLUMNS, CONTEXT_UNIT_COMMENT_MAP_FILE
    )

    for column in ["trigger_time_utc", "window_start_utc", "window_end_utc"]:
        _normalize_utc_column(packages, column)
        _normalize_utc_column(inventory, column)
        _normalize_utc_column(context_units, column)
    _normalize_utc_column(inventory, "event_time_utc")
    for column in ["event_time_utc"]:
        if column in context_map.columns:
            _normalize_utc_column(context_map, column)
    for column in ["time_start_utc", "time_end_utc"]:
        if column in context_units.columns:
            _normalize_utc_column(context_units, column)

    return {
        "paths": paths,
        "packages": packages,
        "inventory": inventory,
        "video_map": video_map,
        "thread_map": thread_map,
        "context_units": context_units,
        "context_map": context_map,
        "manifest": manifest,
    }


def _derive_consumer_run_id(config: RagConsumerConfig, sidecar_manifest: dict[str, Any]) -> str:
    if config.run_id:
        return str(config.run_id)
    return "ragconsumer_" + _short_hash(
        Path(config.sidecars_dir).as_posix(),
        Path(config.output_dir).as_posix(),
        sidecar_manifest.get("run_id"),
    )


def derive_rag_consumer_run_id(
    config: RagConsumerConfig,
    sidecar_manifest: dict[str, Any],
) -> str:
    """Return the existing stage identity without changing its formula."""

    return _derive_consumer_run_id(config, sidecar_manifest)


def _infer_temporal_roles(inventory: pd.DataFrame) -> pd.DataFrame:
    df = inventory.copy()
    before = df["event_time_utc"] < df["trigger_time_utc"]
    at = df["event_time_utc"] == df["trigger_time_utc"]
    after = df["event_time_utc"] > df["trigger_time_utc"]

    df["available_at_trigger"] = before | at
    df["relative_to_trigger"] = "unknown"
    df.loc[before, "relative_to_trigger"] = "before"
    df.loc[at, "relative_to_trigger"] = "at"
    df.loc[after, "relative_to_trigger"] = "after"
    df["is_post_trigger_context"] = after
    df["temporal_role"] = "alert_evidence"
    df.loc[after, "temporal_role"] = "post_trigger_validation_context"
    return df


def _ordered_unique(values: pd.Series | list[Any]) -> list[Any]:
    seen: set[str] = set()
    out: list[Any] = []
    for value in values:
        if value is None or pd.isna(value):
            continue
        key = str(value)
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return [_json_safe(record) for record in df.to_dict(orient="records")]


def _comment_record(row: pd.Series, *, context_unit_id: str | None = None) -> dict[str, Any]:
    return {
        "comment_id": row.get("comment_id"),
        "event_id": row.get("event_id"),
        "video_id": row.get("video_id"),
        "event_time_utc": _isoformat(row.get("event_time_utc")),
        "trigger_time_utc": _isoformat(row.get("trigger_time_utc")),
        "temporal_role": row.get("temporal_role"),
        "available_at_trigger": bool(row.get("available_at_trigger")),
        "relative_to_trigger": row.get("relative_to_trigger"),
        "is_post_trigger_context": bool(row.get("is_post_trigger_context")),
        "text": row.get("text"),
        "text_clean": row.get("text_clean"),
        "context_unit_id": context_unit_id,
        "root_comment_id": row.get("root_comment_id"),
        "parent_comment_id": row.get("parent_comment_id"),
        "is_reply": bool(row.get("is_reply")) if not pd.isna(row.get("is_reply")) else None,
    }


def _event_sort_inventory(inventory: pd.DataFrame) -> pd.DataFrame:
    sort_cols = [col for col in ["event_time_utc", "order_in_event", "comment_id"] if col in inventory.columns]
    return inventory.sort_values(sort_cols).reset_index(drop=True)


def _context_unit_metrics(
    context_units: pd.DataFrame,
    context_map: pd.DataFrame,
    inventory: pd.DataFrame,
    video_map: pd.DataFrame,
) -> pd.DataFrame:
    if context_units.empty:
        return context_units.copy()

    role_cols = [
        "event_id",
        "comment_id",
        "available_at_trigger",
        "relative_to_trigger",
        "is_post_trigger_context",
        "temporal_role",
        "event_time_utc",
    ]
    joined = context_map.merge(
        inventory[role_cols],
        on=["event_id", "comment_id"],
        how="left",
        suffixes=("", "_inventory"),
    )
    metrics = (
        joined.groupby("context_unit_id", dropna=False)
        .agg(
            mapped_comment_count=("comment_id", "nunique"),
            alert_evidence_comment_count=("available_at_trigger", "sum"),
            post_trigger_comment_count=("is_post_trigger_context", "sum"),
            first_comment_time_utc=("event_time_utc_inventory", "min"),
            last_comment_time_utc=("event_time_utc_inventory", "max"),
        )
        .reset_index()
    )

    video_order = (
        video_map.reset_index()
        .rename(columns={"index": "video_order"})
        .loc[:, ["event_id", "video_id", "video_order"]]
    )
    out = context_units.merge(metrics, on="context_unit_id", how="left")
    out = out.merge(video_order, on=["event_id", "video_id"], how="left")
    out["mapped_comment_count"] = out["mapped_comment_count"].fillna(0).astype(int)
    out["alert_evidence_comment_count"] = (
        out["alert_evidence_comment_count"].fillna(0).astype(int)
    )
    out["post_trigger_comment_count"] = (
        out["post_trigger_comment_count"].fillna(0).astype(int)
    )
    out["video_order"] = out["video_order"].fillna(10**9).astype(int)
    return out


def _select_context_units(event_context_units: pd.DataFrame) -> pd.DataFrame:
    if event_context_units.empty:
        return event_context_units.copy()
    df = event_context_units.copy()
    df["has_alert_evidence"] = df["alert_evidence_comment_count"] > 0
    sort_cols = [
        "has_alert_evidence",
        "video_order",
        "time_start_utc",
        "context_order_in_event",
        "context_unit_id",
    ]
    existing = [col for col in sort_cols if col in df.columns]
    ascending = [False if col == "has_alert_evidence" else True for col in existing]
    return df.sort_values(existing, ascending=ascending).reset_index(drop=True)


def _context_unit_record(unit: pd.Series, unit_comments: pd.DataFrame) -> dict[str, Any]:
    alert_ids = _ordered_unique(
        unit_comments.loc[unit_comments["available_at_trigger"], "comment_id"]
    )
    post_ids = _ordered_unique(
        unit_comments.loc[unit_comments["is_post_trigger_context"], "comment_id"]
    )
    return {
        "context_unit_id": unit.get("context_unit_id"),
        "event_id": unit.get("event_id"),
        "video_id": unit.get("video_id"),
        "context_type": unit.get("context_type"),
        "trigger_time_utc": _isoformat(unit.get("trigger_time_utc")),
        "window_start_utc": _isoformat(unit.get("window_start_utc")),
        "window_end_utc": _isoformat(unit.get("window_end_utc")),
        "time_start_utc": _isoformat(unit.get("time_start_utc")),
        "time_end_utc": _isoformat(unit.get("time_end_utc")),
        "comment_ids": _ordered_unique(unit_comments["comment_id"]),
        "alert_evidence_comment_ids": alert_ids,
        "post_trigger_comment_ids": post_ids,
        "comment_count": int(unit_comments["comment_id"].nunique()),
        "post_trigger_context_used": bool(post_ids),
        "selection_reason": unit.get("selection_reason"),
        "context_text": unit.get("context_text"),
    }


def _build_event_payloads(
    *,
    packages: pd.DataFrame,
    inventory: pd.DataFrame,
    video_map: pd.DataFrame,
    thread_map: pd.DataFrame,
    context_units: pd.DataFrame,
    context_map: pd.DataFrame,
    consumer_run_id: str,
    created_at_utc: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    validation_inputs: list[dict[str, Any]] = []
    context_payloads: list[dict[str, Any]] = []
    report_stubs: list[dict[str, Any]] = []

    context_units_enriched = _context_unit_metrics(
        context_units, context_map, inventory, video_map
    )

    global_post_trigger_context_used = False
    total_used_context_comments = 0
    total_selected_units = 0

    for _, package in packages.sort_values("trigger_time_utc").iterrows():
        event_id = package["event_id"]
        event_inventory = _event_sort_inventory(
            inventory.loc[inventory["event_id"] == event_id].copy()
        )
        event_video_map = video_map.loc[video_map["event_id"] == event_id].copy()
        event_thread_map = thread_map.loc[thread_map.get("event_id") == event_id].copy() if "event_id" in thread_map.columns else pd.DataFrame()
        event_context_units = context_units_enriched.loc[
            context_units_enriched["event_id"] == event_id
        ].copy()
        selected_units = _select_context_units(event_context_units)
        selected_unit_ids = _ordered_unique(selected_units["context_unit_id"])

        selected_context_map = context_map.loc[
            (context_map["event_id"] == event_id)
            & (context_map["context_unit_id"].isin(selected_unit_ids))
        ].copy()
        if selected_unit_ids:
            selected_context_map["context_unit_id"] = pd.Categorical(
                selected_context_map["context_unit_id"],
                categories=selected_unit_ids,
                ordered=True,
            )
        sort_cols = [
            col
            for col in ["context_unit_id", "order_in_context_unit", "event_time_utc", "comment_id"]
            if col in selected_context_map.columns
        ]
        if sort_cols:
            selected_context_map = selected_context_map.sort_values(sort_cols)

        used_joined = selected_context_map.merge(
            event_inventory,
            on=["event_id", "comment_id", "video_id"],
            how="left",
            suffixes=("_map", ""),
        )
        used_context_comments = [
            _comment_record(row, context_unit_id=str(row.get("context_unit_id")))
            for _, row in used_joined.iterrows()
        ]

        alert_evidence_comment_ids = _ordered_unique(
            event_inventory.loc[event_inventory["available_at_trigger"], "comment_id"]
        )
        validation_context_comment_ids = _ordered_unique(event_inventory["comment_id"])
        post_trigger_comment_ids = _ordered_unique(
            event_inventory.loc[event_inventory["is_post_trigger_context"], "comment_id"]
        )
        post_trigger_context_used = any(
            bool(comment["is_post_trigger_context"]) for comment in used_context_comments
        )
        global_post_trigger_context_used = (
            global_post_trigger_context_used or post_trigger_context_used
        )

        selected_context_units: list[dict[str, Any]] = []
        for _, unit in selected_units.iterrows():
            unit_id = unit["context_unit_id"]
            unit_comments = used_joined.loc[
                used_joined["context_unit_id"].astype(str) == str(unit_id)
            ].copy()
            selected_context_units.append(_context_unit_record(unit, unit_comments))

        validation_id = "valstub_" + _short_hash(consumer_run_id, event_id)
        temporal_scope_note = (
            "All validation context comments are available at or before trigger_time; "
            "alert evidence and validation context coincide for this event."
            if not post_trigger_comment_ids
            else (
                "Post-trigger comments are present only as validation context and must "
                "not be interpreted as causes of the alert."
            )
        )

        validation_config = {
            "consumer_run_id": consumer_run_id,
            "consumer_artifact_version": RAG_CONSUMER_ARTIFACT_VERSION,
            "mode": "non_generative_structural_payloads",
            "does_not_call_llm": True,
            "does_not_create_embeddings": True,
            "context_selection_policy": {
                "selection_unit": "rag_context_units",
                "selection_scope": "all_units_for_event",
                "semantic_ranking": False,
                "ordering": [
                    "units_with_alert_evidence_first",
                    "event_video_map_order",
                    "time_start_utc",
                    "context_order_in_event",
                    "context_unit_id",
                ],
            },
            "temporal_policy": {
                "alert_evidence_rule": "window_start_utc <= event_time_utc <= trigger_time_utc",
                "validation_context_rule": "window_start_utc <= event_time_utc <= window_end_utc",
                "post_trigger_context_is_causal": False,
            },
        }

        event_metadata = {
            "validation_id": validation_id,
            "event_id": event_id,
            "run_id": package.get("run_id"),
            "detector_name": package.get("detector_name"),
            "trigger_time_utc": _isoformat(package.get("trigger_time_utc")),
            "window_start_utc": _isoformat(package.get("window_start_utc")),
            "window_end_utc": _isoformat(package.get("window_end_utc")),
            "trigger_volume": package.get("trigger_volume"),
            "trigger_strength": package.get("trigger_strength"),
        }

        validation_inputs.append(
            {
                **event_metadata,
                "associated_videos": _records(event_video_map),
                "thread_count": int(len(event_thread_map)),
                "alert_evidence_comment_ids": alert_evidence_comment_ids,
                "validation_context_comment_ids": validation_context_comment_ids,
                "post_trigger_comment_ids": post_trigger_comment_ids,
                "used_context_unit_ids": selected_unit_ids,
                "cited_comment_ids": [],
                "temporal_scope_note": temporal_scope_note,
                "validation_config": validation_config,
            }
        )

        context_payloads.append(
            {
                **event_metadata,
                "selected_context_units": selected_context_units,
                "used_context_comments": used_context_comments,
                "used_context_comment_count": len(used_context_comments),
                "post_trigger_context_used": post_trigger_context_used,
                "post_trigger_comment_ids": post_trigger_comment_ids,
                "temporal_scope_note": temporal_scope_note,
            }
        )

        report_stubs.append(
            {
                **event_metadata,
                "alert_evidence_comment_ids": alert_evidence_comment_ids,
                "validation_context_comment_ids": validation_context_comment_ids,
                "used_context_unit_ids": selected_unit_ids,
                "used_context_comments": used_context_comments,
                "cited_comment_ids": [],
                "post_trigger_context_used": post_trigger_context_used,
                "post_trigger_comment_ids": post_trigger_comment_ids,
                "temporal_scope_note": temporal_scope_note,
                "validation_status": "not_evaluated",
                "explanation": (
                    "Structural non-generative report generated from approved RAG "
                    "sidecars. No LLM validation has been run."
                ),
                "limitations": [
                    "No generative validation has been executed.",
                    "No external evidence retrieval, embeddings, or semantic ranking are used.",
                    "cited_comment_ids is intentionally empty until a future validation step cites evidence.",
                ],
                "validation_config": validation_config,
                "validated_at_utc": created_at_utc,
            }
        )

        total_selected_units += len(selected_unit_ids)
        total_used_context_comments += len(used_context_comments)

    counters = {
        "event_count": len(validation_inputs),
        "selected_context_unit_count": total_selected_units,
        "used_context_comment_rows": total_used_context_comments,
        "post_trigger_context_used": global_post_trigger_context_used,
    }
    return validation_inputs, context_payloads, report_stubs, counters


def _coverage_checks(
    inventory: pd.DataFrame,
    context_map: pd.DataFrame,
    report_stubs: list[dict[str, Any]],
) -> dict[str, Any]:
    inventory_pairs = inventory[["event_id", "comment_id"]].drop_duplicates()
    context_pairs = context_map[["event_id", "comment_id"]].drop_duplicates()
    pair_check = inventory_pairs.merge(
        context_pairs, on=["event_id", "comment_id"], how="outer", indicator=True
    )
    report_statuses = {record.get("validation_status") for record in report_stubs}
    cited_counts = [len(record.get("cited_comment_ids") or []) for record in report_stubs]
    return {
        "inventory_event_comment_pairs": int(len(inventory_pairs)),
        "context_event_comment_pairs": int(len(context_pairs)),
        "missing_context_pairs": int((pair_check["_merge"] == "left_only").sum()),
        "extra_context_pairs": int((pair_check["_merge"] == "right_only").sum()),
        "all_inventory_comments_have_context": bool(
            not (pair_check["_merge"] == "left_only").any()
        ),
        "validation_statuses": sorted(report_statuses),
        "all_reports_not_evaluated": report_statuses == {"not_evaluated"},
        "all_cited_comment_ids_empty": all(count == 0 for count in cited_counts),
    }


def _temporal_audit(inventory: pd.DataFrame) -> dict[str, Any]:
    before = int((inventory["relative_to_trigger"] == "before").sum())
    at = int((inventory["relative_to_trigger"] == "at").sum())
    after = int((inventory["relative_to_trigger"] == "after").sum())
    by_event = []
    for event_id, event_df in inventory.groupby("event_id", sort=True):
        by_event.append(
            {
                "event_id": event_id,
                "total_comments": int(len(event_df)),
                "before_trigger": int((event_df["relative_to_trigger"] == "before").sum()),
                "at_trigger": int((event_df["relative_to_trigger"] == "at").sum()),
                "after_trigger": int((event_df["relative_to_trigger"] == "after").sum()),
                "post_trigger_context_present": bool(
                    (event_df["relative_to_trigger"] == "after").any()
                ),
            }
        )
    return {
        "total_comments": int(len(inventory)),
        "before_trigger": before,
        "at_trigger": at,
        "after_trigger": after,
        "post_trigger_context_present": bool(after > 0),
        "by_event": by_event,
    }


def write_rag_consumer_artifacts_from_config(config: RagConsumerConfig) -> dict[str, Any]:
    sidecars = _load_sidecars(config)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    inventory = _infer_temporal_roles(sidecars["inventory"])
    consumer_run_id = _derive_consumer_run_id(config, sidecars["manifest"])
    created_at_utc = _utc_now_iso()

    validation_inputs, context_payloads, report_stubs, counters = _build_event_payloads(
        packages=sidecars["packages"],
        inventory=inventory,
        video_map=sidecars["video_map"],
        thread_map=sidecars["thread_map"],
        context_units=sidecars["context_units"],
        context_map=sidecars["context_map"],
        consumer_run_id=consumer_run_id,
        created_at_utc=created_at_utc,
    )

    output_paths = {
        "rag_validation_inputs": output_dir / RAG_VALIDATION_INPUTS_FILE,
        "rag_context_payloads": output_dir / RAG_CONTEXT_PAYLOADS_FILE,
        "rag_consumer_manifest": output_dir / RAG_CONSUMER_MANIFEST_FILE,
        "rag_validation_reports_stub": output_dir / RAG_VALIDATION_REPORTS_STUB_FILE,
    }

    coverage_checks = _coverage_checks(
        inventory, sidecars["context_map"], report_stubs
    )
    temporal_audit = _temporal_audit(inventory)

    manifest = {
        "run_id": consumer_run_id,
        "created_at_utc": created_at_utc,
        "pipeline_stage": "rag_consumer_non_generative",
        "mode": "structural_payloads_no_llm_no_embeddings_no_validation",
        "artifact_version": RAG_CONSUMER_ARTIFACT_VERSION,
        "source_sidecars_dir": _normalize_path(config.sidecars_dir),
        "source_artifacts": {
            name: _normalize_path(path) for name, path in sidecars["paths"].items()
        },
        "source_sidecar_run_id": sidecars["manifest"].get("run_id"),
        "output_paths": {
            name: _normalize_path(path) for name, path in output_paths.items()
        },
        "compatibility_policy": {
            "does_not_modify_sidecars": True,
            "does_not_modify_pipeline": True,
            "does_not_modify_poc_outputs": True,
            "does_not_call_llm": True,
            "does_not_create_embeddings": True,
            "preserve_event_id": True,
            "preserve_trigger_time_video_id": True,
        },
        "temporal_policy": {
            "alert_evidence_comments": "window_start_utc <= event_time_utc <= trigger_time_utc",
            "validation_context_comments": "window_start_utc <= event_time_utc <= window_end_utc",
            "post_trigger_comments_are_causal": False,
            "current_run_alert_evidence_equals_validation_context": bool(
                temporal_audit["after_trigger"] == 0
            ),
        },
        "context_selection_policy": {
            "strategy": "deterministic_metadata_only",
            "selected_units": "all_context_units_for_event",
            "semantic_ranking": False,
            "embeddings": False,
            "llm_summarization": False,
            "ordering": [
                "units_with_alert_evidence_first",
                "event_video_map_order",
                "time_start_utc",
                "context_order_in_event",
                "context_unit_id",
            ],
        },
        "counts": {
            **counters,
            "inventory_comment_rows": int(len(inventory)),
            "context_unit_comment_rows": int(len(sidecars["context_map"])),
            "context_unit_count": int(len(sidecars["context_units"])),
            "video_map_rows": int(len(sidecars["video_map"])),
            "thread_map_rows": int(len(sidecars["thread_map"])),
        },
        "coverage_checks": coverage_checks,
        "temporal_audit": temporal_audit,
        "notes": config.notes,
        "params": config.params,
    }

    _write_jsonl(output_paths["rag_validation_inputs"], validation_inputs)
    _write_jsonl(output_paths["rag_context_payloads"], context_payloads)
    _write_jsonl(output_paths["rag_validation_reports_stub"], report_stubs)
    _write_json(output_paths["rag_consumer_manifest"], manifest)

    return {
        "run_id": consumer_run_id,
        "output_dir": _normalize_path(output_dir),
        "output_paths": {
            name: _normalize_path(path) for name, path in output_paths.items()
        },
        "counts": manifest["counts"],
        "coverage_checks": coverage_checks,
        "temporal_audit": {
            key: value for key, value in temporal_audit.items() if key != "by_event"
        },
    }


def write_rag_consumer_artifacts(
    *,
    sidecars_dir: str | Path,
    output_dir: str | Path,
    run_id: str | None = None,
    notes: str | None = None,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = RagConsumerConfig(
        sidecars_dir=str(sidecars_dir),
        output_dir=str(output_dir),
        run_id=run_id,
        notes=notes,
        params=params or {},
    )
    return write_rag_consumer_artifacts_from_config(config)


__all__ = [
    "RAG_CONSUMER_ARTIFACT_VERSION",
    "RagConsumerConfig",
    "derive_rag_consumer_run_id",
    "load_rag_consumer_config",
    "write_rag_consumer_artifacts",
    "write_rag_consumer_artifacts_from_config",
]
