from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


DAILY_RAG_CONSUMER_ARTIFACT_VERSION = "daily_rag_consumer_stub_v1"
DAILY_EVENT_EVIDENCE_PACKAGES_FILE = "daily_event_evidence_packages.jsonl"
DAILY_EVENT_COMMENT_INVENTORY_FILE = "daily_event_comment_inventory.csv"
DAILY_EVENT_VIDEO_MAP_FILE = "daily_event_video_map.csv"
DAILY_EVENT_THREAD_MAP_FILE = "daily_event_thread_map.csv"
DAILY_RAG_CONTEXT_UNITS_FILE = "daily_rag_context_units.jsonl"
DAILY_CONTEXT_UNIT_COMMENT_MAP_FILE = "daily_context_unit_comment_map.csv"
DAILY_RAG_SIDECARS_MANIFEST_FILE = "daily_rag_sidecars_manifest.json"
README_FILE = "README.md"

DAILY_RAG_VALIDATION_INPUTS_FILE = "daily_rag_validation_inputs.jsonl"
DAILY_RAG_CONTEXT_PAYLOADS_FILE = "daily_rag_context_payloads.jsonl"
DAILY_RAG_CONSUMER_MANIFEST_FILE = "daily_rag_consumer_manifest.json"
DAILY_RAG_VALIDATION_REPORTS_STUB_FILE = "daily_rag_validation_reports_stub.jsonl"
DAILY_RAG_CONTEXT_SIZE_REPORT_FILE = "daily_rag_context_size_report.jsonl"

REQUIRED_PACKAGE_COLUMNS = {
    "daily_rag_event_id",
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
    "trigger_reason",
    "analysis_window_start_utc",
    "analysis_window_end_utc",
    "data_cutoff_utc",
}
REQUIRED_INVENTORY_COLUMNS = {
    "daily_rag_event_id",
    "daily_event_id",
    "cycle_id",
    "comment_id",
    "video_id",
    "event_time_utc",
    "is_alert_evidence",
    "is_validation_context",
    "temporal_role",
    "data_cutoff_utc",
}
REQUIRED_CONTEXT_UNIT_COLUMNS = {
    "context_unit_id",
    "daily_rag_event_id",
    "daily_event_id",
    "cycle_id",
    "video_id",
    "context_type",
    "temporal_scope",
    "context_role",
    "contains_alert_evidence",
    "contains_validation_context",
    "alert_evidence_comment_count",
    "validation_context_comment_count",
    "comment_ids",
    "comment_count",
    "time_start_utc",
    "time_end_utc",
    "text_block",
}
REQUIRED_CONTEXT_MAP_COLUMNS = {
    "context_unit_id",
    "daily_rag_event_id",
    "daily_event_id",
    "cycle_id",
    "video_id",
    "comment_id",
    "is_alert_evidence",
    "is_validation_context",
    "temporal_role",
}


@dataclass(frozen=True)
class DailyRagConsumerConfig:
    sidecars_dir: str | Path | None = None
    output_dir: str | Path | None = None
    run_id: str | None = None
    max_estimated_input_tokens: int = 16_000
    notes: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    run_llm: bool = False
    run_serper: bool = False
    use_embeddings: bool = False
    use_vectorstore: bool = False
    run_g1: bool = False
    run_g2: bool = False

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "DailyRagConsumerConfig":
        config_payload = payload.get("daily_rag_consumer", payload)
        if not isinstance(config_payload, dict):
            raise ValueError("daily_rag_consumer config section must be an object.")
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(config_payload) - allowed)
        if unknown:
            raise ValueError(f"Unknown daily RAG consumer config fields: {unknown}")
        params = config_payload.get("params") or {}
        if not isinstance(params, dict):
            raise ValueError("params must be an object.")
        return cls(**{**config_payload, "params": params})

    def validate(self) -> None:
        if self.max_estimated_input_tokens < 1:
            raise ValueError("max_estimated_input_tokens must be >= 1.")
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
                "Daily RAG consumer is non-generative. These flags must remain false: "
                + ", ".join(enabled)
            )
        if self.sidecars_dir is None:
            raise ValueError("sidecars_dir is required.")
        if self.output_dir is None:
            raise ValueError("output_dir is required.")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_path(path: str | Path | None) -> str | None:
    if path is None:
        return None
    return Path(path).as_posix()


def _short_hash(*parts: Any) -> str:
    payload = "|".join("" if part is None else str(part) for part in parts)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


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
        json.dumps(_json_safe(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(_json_safe(row), ensure_ascii=False) for row in rows]
    p.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _read_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing JSON artifact: {p}")
    payload = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must contain an object: {p}")
    return payload


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing JSONL artifact: {p}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"JSONL line {line_number} must contain an object: {p}")
        rows.append(payload)
    return rows


def _read_csv(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing CSV artifact: {p}")
    return pd.read_csv(p)


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


def _bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def _ordered_unique(values: list[Any] | pd.Series) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value is None or pd.isna(value):
            continue
        item = str(value)
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return [_json_safe(row) for row in df.to_dict(orient="records")]


def _estimate_tokens(text: Any) -> int:
    if text is None or pd.isna(text):
        return 0
    value = str(text)
    return max(1, len(value) // 4) if value else 0


def _hash_ids(ids: list[str]) -> str:
    payload = "|".join(sorted(str(item) for item in ids))
    return "sha1_" + hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _sidecar_paths(sidecars_dir: str | Path) -> dict[str, Path]:
    root = Path(sidecars_dir)
    return {
        "daily_event_evidence_packages": root / DAILY_EVENT_EVIDENCE_PACKAGES_FILE,
        "daily_event_comment_inventory": root / DAILY_EVENT_COMMENT_INVENTORY_FILE,
        "daily_event_video_map": root / DAILY_EVENT_VIDEO_MAP_FILE,
        "daily_event_thread_map": root / DAILY_EVENT_THREAD_MAP_FILE,
        "daily_rag_context_units": root / DAILY_RAG_CONTEXT_UNITS_FILE,
        "daily_context_unit_comment_map": root / DAILY_CONTEXT_UNIT_COMMENT_MAP_FILE,
        "daily_rag_sidecars_manifest": root / DAILY_RAG_SIDECARS_MANIFEST_FILE,
        "readme": root / README_FILE,
    }


def _output_paths(output_dir: str | Path) -> dict[str, str]:
    root = Path(output_dir)
    return {
        "daily_rag_validation_inputs": (root / DAILY_RAG_VALIDATION_INPUTS_FILE).as_posix(),
        "daily_rag_context_payloads": (root / DAILY_RAG_CONTEXT_PAYLOADS_FILE).as_posix(),
        "daily_rag_consumer_manifest": (root / DAILY_RAG_CONSUMER_MANIFEST_FILE).as_posix(),
        "daily_rag_validation_reports_stub": (
            root / DAILY_RAG_VALIDATION_REPORTS_STUB_FILE
        ).as_posix(),
        "daily_rag_context_size_report": (
            root / DAILY_RAG_CONTEXT_SIZE_REPORT_FILE
        ).as_posix(),
    }


def _load_sidecars(config: DailyRagConsumerConfig) -> dict[str, Any]:
    paths = _sidecar_paths(config.sidecars_dir)
    if not paths["readme"].exists():
        raise FileNotFoundError(f"Missing daily RAG sidecar README: {paths['readme']}")
    packages = pd.DataFrame(_read_jsonl(paths["daily_event_evidence_packages"]))
    inventory = _read_csv(paths["daily_event_comment_inventory"])
    video_map = _read_csv(paths["daily_event_video_map"])
    thread_map = _read_csv(paths["daily_event_thread_map"])
    context_units = pd.DataFrame(_read_jsonl(paths["daily_rag_context_units"]))
    context_map = _read_csv(paths["daily_context_unit_comment_map"])
    manifest = _read_json(paths["daily_rag_sidecars_manifest"])

    _require_columns(packages, REQUIRED_PACKAGE_COLUMNS, DAILY_EVENT_EVIDENCE_PACKAGES_FILE)
    _require_columns(inventory, REQUIRED_INVENTORY_COLUMNS, DAILY_EVENT_COMMENT_INVENTORY_FILE)
    _require_columns(context_units, REQUIRED_CONTEXT_UNIT_COLUMNS, DAILY_RAG_CONTEXT_UNITS_FILE)
    _require_columns(context_map, REQUIRED_CONTEXT_MAP_COLUMNS, DAILY_CONTEXT_UNIT_COMMENT_MAP_FILE)

    for df, columns in [
        (packages, ["analysis_window_start_utc", "analysis_window_end_utc", "data_cutoff_utc"]),
        (
            inventory,
            ["event_time_utc", "analysis_window_start_utc", "analysis_window_end_utc", "data_cutoff_utc"],
        ),
        (context_units, ["time_start_utc", "time_end_utc"]),
    ]:
        for column in columns:
            if column in df.columns:
                _normalize_utc_column(df, column)
    if "event_time_utc" in context_map.columns:
        _normalize_utc_column(context_map, "event_time_utc")

    for df, columns in [
        (
            inventory,
            ["is_alert_evidence", "is_validation_context", "available_at_cycle"],
        ),
        (
            context_units,
            ["contains_alert_evidence", "contains_validation_context"],
        ),
        (
            context_map,
            ["is_alert_evidence", "is_validation_context"],
        ),
    ]:
        for column in columns:
            if column in df.columns:
                df[column] = _bool_series(df[column])

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


def _derive_consumer_run_id(config: DailyRagConsumerConfig, sidecar_manifest: dict[str, Any]) -> str:
    if config.run_id:
        return config.run_id
    return "dragconsumer_" + _short_hash(
        Path(config.sidecars_dir).as_posix(),
        Path(config.output_dir).as_posix(),
        sidecar_manifest.get("run_id"),
    )


def _unit_comment_ids(context_map: pd.DataFrame, context_unit_id: str) -> list[str]:
    return _ordered_unique(
        context_map.loc[
            context_map["context_unit_id"].astype(str) == str(context_unit_id),
            "comment_id",
        ]
    )


def _unit_record(unit: pd.Series, context_map: pd.DataFrame) -> dict[str, Any]:
    unit_id = str(unit["context_unit_id"])
    comment_ids = _unit_comment_ids(context_map, unit_id)
    return {
        "context_unit_id": unit_id,
        "daily_rag_event_id": unit["daily_rag_event_id"],
        "daily_event_id": unit["daily_event_id"],
        "cycle_id": unit["cycle_id"],
        "video_id": unit["video_id"],
        "context_type": unit["context_type"],
        "temporal_scope": unit["temporal_scope"],
        "context_role": unit["context_role"],
        "contains_alert_evidence": bool(unit["contains_alert_evidence"]),
        "contains_validation_context": bool(unit["contains_validation_context"]),
        "alert_evidence_comment_count": int(unit["alert_evidence_comment_count"]),
        "validation_context_comment_count": int(unit["validation_context_comment_count"]),
        "comment_ids": comment_ids,
        "comment_ids_hash": _hash_ids(comment_ids),
        "comment_count": int(unit["comment_count"]),
        "time_start_utc": _isoformat(unit["time_start_utc"]),
        "time_end_utc": _isoformat(unit["time_end_utc"]),
        "estimated_tokens": int(unit["estimated_tokens"]),
        "text_block": unit.get("text_block"),
    }


def _event_units(context_units: pd.DataFrame, daily_event_id: str) -> pd.DataFrame:
    event_units = context_units.loc[context_units["daily_event_id"] == daily_event_id].copy()
    if event_units.empty:
        return event_units
    event_units["estimated_tokens"] = event_units["text_block"].map(_estimate_tokens)
    sort_cols = [
        col
        for col in [
            "context_role",
            "video_id",
            "time_start_utc",
            "context_order_in_daily_event",
            "context_unit_id",
        ]
        if col in event_units.columns
    ]
    return event_units.sort_values(sort_cols, ascending=[True] * len(sort_cols))


def _status_for_tokens(tokens: int, limit: int) -> str:
    if tokens > limit:
        return "requires_context_selection_policy"
    return "fits_initial_context_budget"


def _signal_summary(package: pd.Series) -> dict[str, Any]:
    return {
        "detector_name": package["detector_name"],
        "signal_name": package["signal_name"],
        "signal_value": package["signal_value"],
        "baseline_mean": package["baseline_mean"],
        "ratio_to_baseline": package["ratio_to_baseline"],
        "delta_value": package["delta_value"],
        "pct_change_value": package["pct_change_value"],
        "threshold_value": package.get("threshold_value"),
        "trigger_reason": package["trigger_reason"],
    }


def _build_event_artifacts(
    *,
    config: DailyRagConsumerConfig,
    packages: pd.DataFrame,
    inventory: pd.DataFrame,
    video_map: pd.DataFrame,
    context_units: pd.DataFrame,
    context_map: pd.DataFrame,
    consumer_run_id: str,
    created_at_utc: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    validation_inputs: list[dict[str, Any]] = []
    context_payloads: list[dict[str, Any]] = []
    report_stubs: list[dict[str, Any]] = []
    size_reports: list[dict[str, Any]] = []

    for _, package in packages.sort_values(["cycle_index", "daily_event_id"]).iterrows():
        daily_event_id = str(package["daily_event_id"])
        event_inventory = inventory.loc[inventory["daily_event_id"] == daily_event_id].copy()
        event_video_map = video_map.loc[video_map["daily_event_id"] == daily_event_id].copy()
        event_context_map = context_map.loc[context_map["daily_event_id"] == daily_event_id].copy()
        event_units = _event_units(context_units, daily_event_id)

        alert_units = event_units.loc[event_units["context_role"] == "alert_evidence_unit"]
        validation_units = event_units.loc[event_units["context_role"] == "validation_context_unit"]
        mixed_units = event_units.loc[event_units["context_role"] == "mixed_unit"]
        used_context_units = event_units

        alert_comment_ids = _ordered_unique(
            event_inventory.loc[event_inventory["is_alert_evidence"], "comment_id"]
        )
        validation_comment_ids = _ordered_unique(
            event_inventory.loc[event_inventory["is_validation_context"], "comment_id"]
        )
        used_context_unit_ids = _ordered_unique(used_context_units["context_unit_id"])
        estimated_tokens = int(used_context_units["estimated_tokens"].sum())
        context_size_status = _status_for_tokens(
            estimated_tokens,
            config.max_estimated_input_tokens,
        )
        unit_records = [_unit_record(unit, event_context_map) for _, unit in used_context_units.iterrows()]
        alert_unit_records = [
            _unit_record(unit, event_context_map) for _, unit in alert_units.iterrows()
        ]
        validation_unit_records = [
            _unit_record(unit, event_context_map) for _, unit in validation_units.iterrows()
        ]

        by_video: list[dict[str, Any]] = []
        for _, video in event_video_map.sort_values("video_id").iterrows():
            video_id = str(video["video_id"])
            video_units = used_context_units.loc[used_context_units["video_id"].astype(str) == video_id]
            video_comments = event_inventory.loc[event_inventory["video_id"].astype(str) == video_id]
            by_video.append(
                {
                    "video_id": video_id,
                    "alert_evidence_comment_count": int(video["alert_evidence_comment_count"]),
                    "validation_context_comment_count": int(video["validation_context_comment_count"]),
                    "context_unit_ids": _ordered_unique(video_units["context_unit_id"]),
                    "alert_evidence_unit_count": int((video_units["context_role"] == "alert_evidence_unit").sum()),
                    "validation_context_unit_count": int((video_units["context_role"] == "validation_context_unit").sum()),
                    "comment_ids_hash": _hash_ids(_ordered_unique(video_comments["comment_id"])),
                    "estimated_tokens": int(video_units["estimated_tokens"].sum()) if not video_units.empty else 0,
                }
            )

        common = {
            "daily_rag_event_id": package["daily_rag_event_id"],
            "daily_event_id": daily_event_id,
            "cycle_id": package["cycle_id"],
            "cycle_index": int(package["cycle_index"]),
            "detector_name": package["detector_name"],
            "signal_name": package["signal_name"],
            "signal_value": package["signal_value"],
            "baseline_mean": package["baseline_mean"],
            "ratio_to_baseline": package["ratio_to_baseline"],
            "delta_value": package["delta_value"],
            "pct_change_value": package["pct_change_value"],
            "trigger_reason": package["trigger_reason"],
            "analysis_window_start_utc": _isoformat(package["analysis_window_start_utc"]),
            "analysis_window_end_utc": _isoformat(package["analysis_window_end_utc"]),
            "data_cutoff_utc": _isoformat(package["data_cutoff_utc"]),
            "alert_evidence_comment_count": len(alert_comment_ids),
            "validation_context_comment_count": len(validation_comment_ids),
            "alert_evidence_unit_count": int(len(alert_units)),
            "validation_context_unit_count": int(len(validation_units)),
            "video_ids": _ordered_unique(event_video_map["video_id"]),
            "context_unit_ids": used_context_unit_ids,
            "estimated_input_tokens": estimated_tokens,
            "context_size_status": context_size_status,
        }
        validation_inputs.append(common)

        context_payloads.append(
            {
                **common,
                "consumer_run_id": consumer_run_id,
                "signal_summary": _signal_summary(package),
                "alert_evidence_units": alert_unit_records,
                "validation_context_units": validation_unit_records,
                "used_context_units": unit_records,
                "used_context_unit_count": len(unit_records),
                "grouping_by_video": by_video,
                "alert_evidence_comment_ids_hash": _hash_ids(alert_comment_ids),
                "validation_context_comment_ids_hash": _hash_ids(validation_comment_ids),
                "limitations": [
                    "No semantic ranking or truncation has been applied.",
                    "Large events must receive an approved context selection policy before generative validation.",
                    "Some threads may be split between alert evidence and prior validation context.",
                ],
                "created_at_utc": created_at_utc,
            }
        )

        report_stubs.append(
            {
                "daily_rag_event_id": package["daily_rag_event_id"],
                "daily_event_id": daily_event_id,
                "validation_status": "not_evaluated",
                "event_interpretation": "not_evaluated",
                "confidence_label": None,
                "cited_comment_ids": [],
                "cited_context_unit_ids": [],
                "limitations": [
                    "R-D2 is non-generative and does not evaluate evidence.",
                    "No LLM, Serper, embeddings, vectorstore, G-1, or G-2 execution was performed.",
                    "Context may be too large for future direct prompting without selection.",
                ],
                "created_at_utc": created_at_utc,
            }
        )

        size_reports.append(
            {
                "daily_rag_event_id": package["daily_rag_event_id"],
                "daily_event_id": daily_event_id,
                "cycle_id": package["cycle_id"],
                "cycle_index": int(package["cycle_index"]),
                "alert_evidence_comment_count": len(alert_comment_ids),
                "validation_context_comment_count": len(validation_comment_ids),
                "alert_evidence_unit_count": int(len(alert_units)),
                "validation_context_unit_count": int(len(validation_units)),
                "mixed_unit_count": int(len(mixed_units)),
                "video_count": int(event_video_map["video_id"].nunique()),
                "estimated_input_tokens": estimated_tokens,
                "max_estimated_input_tokens": config.max_estimated_input_tokens,
                "context_size_status": context_size_status,
                "requires_context_selection_policy": (
                    context_size_status == "requires_context_selection_policy"
                ),
                "largest_video_estimated_tokens": max(
                    [int(item["estimated_tokens"]) for item in by_video],
                    default=0,
                ),
                "created_at_utc": created_at_utc,
            }
        )

    return validation_inputs, context_payloads, report_stubs, size_reports


def _validate_outputs(
    *,
    packages: pd.DataFrame,
    inventory: pd.DataFrame,
    context_units: pd.DataFrame,
    context_map: pd.DataFrame,
    validation_inputs: list[dict[str, Any]],
    report_stubs: list[dict[str, Any]],
) -> dict[str, Any]:
    errors: list[str] = []
    package_event_ids = set(packages["daily_event_id"].astype(str))
    output_event_ids = {str(row["daily_event_id"]) for row in validation_inputs}
    if package_event_ids != output_event_ids:
        errors.append(
            f"event_output_mismatch missing={sorted(package_event_ids-output_event_ids)} "
            f"extra={sorted(output_event_ids-package_event_ids)}"
        )

    inventory_pairs = set(zip(inventory["daily_event_id"], inventory["comment_id"].astype(str)))
    context_pairs = set(zip(context_map["daily_event_id"], context_map["comment_id"].astype(str)))
    missing_context = inventory_pairs.difference(context_pairs)
    extra_context = context_pairs.difference(inventory_pairs)
    if missing_context:
        errors.append(f"comments_without_context_unit={len(missing_context)}")
    if extra_context:
        errors.append(f"context_comments_not_in_inventory={len(extra_context)}")

    event_time = pd.to_datetime(inventory["event_time_utc"], utc=True)
    data_cutoff = pd.to_datetime(inventory["data_cutoff_utc"], utc=True)
    future_leak_count = int((event_time >= data_cutoff).sum())
    if future_leak_count:
        errors.append(f"future_leak_count={future_leak_count}")

    units_without_video = int(context_units["video_id"].isna().sum())
    if units_without_video:
        errors.append(f"units_without_video={units_without_video}")

    unit_video_counts = context_map.groupby("context_unit_id")["video_id"].nunique()
    units_mixing_videos = int((unit_video_counts > 1).sum())
    if units_mixing_videos:
        errors.append(f"units_mixing_videos={units_mixing_videos}")

    mixed_unit_count = int((context_units["context_role"] == "mixed_unit").sum())
    if mixed_unit_count:
        errors.append(f"mixed_unit_count={mixed_unit_count}")

    alert_unit_ids = set(
        context_units.loc[
            context_units["context_role"] == "alert_evidence_unit",
            "context_unit_id",
        ].astype(str)
    )
    validation_unit_ids = set(
        context_units.loc[
            context_units["context_role"] == "validation_context_unit",
            "context_unit_id",
        ].astype(str)
    )
    alert_map = context_map.loc[context_map["context_unit_id"].astype(str).isin(alert_unit_ids)]
    validation_map = context_map.loc[
        context_map["context_unit_id"].astype(str).isin(validation_unit_ids)
    ]
    alert_unit_non_alert_comments = int((~alert_map["is_alert_evidence"]).sum())
    validation_unit_alert_comments = int(validation_map["is_alert_evidence"].sum())
    if alert_unit_non_alert_comments:
        errors.append(f"alert_unit_non_alert_comments={alert_unit_non_alert_comments}")
    if validation_unit_alert_comments:
        errors.append(f"validation_unit_alert_comments={validation_unit_alert_comments}")

    subset_failures = 0
    for daily_event_id, group in inventory.groupby("daily_event_id"):
        alert_ids = set(group.loc[group["is_alert_evidence"], "comment_id"].astype(str))
        validation_ids = set(group.loc[group["is_validation_context"], "comment_id"].astype(str))
        if not alert_ids.issubset(validation_ids):
            subset_failures += 1
    if subset_failures:
        errors.append(f"alert_not_subset_of_validation_events={subset_failures}")

    statuses = {row.get("validation_status") for row in report_stubs}
    cited_lengths = [len(row.get("cited_comment_ids") or []) for row in report_stubs]
    context_cited_lengths = [len(row.get("cited_context_unit_ids") or []) for row in report_stubs]
    if statuses != {"not_evaluated"}:
        errors.append(f"unexpected_validation_statuses={sorted(statuses)}")
    if any(cited_lengths) or any(context_cited_lengths):
        errors.append("stub_citations_not_empty")

    return {
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "events_in_packages": int(len(package_event_ids)),
        "events_in_outputs": int(len(output_event_ids)),
        "inventory_event_comment_pairs": int(len(inventory_pairs)),
        "context_event_comment_pairs": int(len(context_pairs)),
        "comments_without_context_unit": int(len(missing_context)),
        "context_comments_not_in_inventory": int(len(extra_context)),
        "future_leak_count": future_leak_count,
        "units_without_video": units_without_video,
        "units_mixing_videos": units_mixing_videos,
        "mixed_unit_count": mixed_unit_count,
        "alert_unit_non_alert_comments": alert_unit_non_alert_comments,
        "validation_unit_alert_comments": validation_unit_alert_comments,
        "alert_evidence_subset_validation_context": subset_failures == 0,
        "all_reports_not_evaluated": statuses == {"not_evaluated"},
        "all_stub_citations_empty": not any(cited_lengths) and not any(context_cited_lengths),
        "external_calls": {
            "llm": 0,
            "serper": 0,
            "embeddings": False,
            "vectorstore": False,
            "g1": False,
            "g2": False,
        },
    }


def write_daily_rag_consumer_artifacts_from_config(
    config: DailyRagConsumerConfig,
) -> dict[str, Any]:
    config.validate()
    sidecars = _load_sidecars(config)
    output_root = Path(config.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    output_paths = _output_paths(output_root)
    created_at = _utc_now_iso()
    consumer_run_id = _derive_consumer_run_id(config, sidecars["manifest"])

    validation_inputs, context_payloads, report_stubs, size_reports = _build_event_artifacts(
        config=config,
        packages=sidecars["packages"],
        inventory=sidecars["inventory"],
        video_map=sidecars["video_map"],
        context_units=sidecars["context_units"],
        context_map=sidecars["context_map"],
        consumer_run_id=consumer_run_id,
        created_at_utc=created_at,
    )
    validations = _validate_outputs(
        packages=sidecars["packages"],
        inventory=sidecars["inventory"],
        context_units=sidecars["context_units"],
        context_map=sidecars["context_map"],
        validation_inputs=validation_inputs,
        report_stubs=report_stubs,
    )
    context_status_counts = Counter(row["context_size_status"] for row in size_reports)
    unit_role_counts = Counter(sidecars["context_units"]["context_role"].astype(str))

    manifest = {
        "run_id": consumer_run_id,
        "created_at_utc": created_at,
        "pipeline_stage": "daily_rag_consumer_non_generative",
        "mode": "structural_payloads_no_llm_no_serper_no_embeddings_no_validation",
        "artifact_version": DAILY_RAG_CONSUMER_ARTIFACT_VERSION,
        "source_sidecars_dir": _normalize_path(config.sidecars_dir),
        "source_artifacts": {
            name: _normalize_path(path) for name, path in sidecars["paths"].items()
        },
        "source_sidecar_run_id": sidecars["manifest"].get("run_id"),
        "output_paths": output_paths,
        "context_policy": {
            "selection_strategy": "none_yet_full_available_context_reported",
            "semantic_ranking": False,
            "truncation": False,
            "future_selection_order": [
                "prioritize_alert_evidence_unit",
                "add_validation_context_unit_by_video",
                "preserve_video_diversity",
                "preserve_relevant_threads_if_approved",
                "preserve_comment_id_traceability",
            ],
            "max_estimated_input_tokens": config.max_estimated_input_tokens,
        },
        "counts": {
            "events_processed": len(validation_inputs),
            "validation_inputs": len(validation_inputs),
            "context_payloads": len(context_payloads),
            "report_stubs": len(report_stubs),
            "size_reports": len(size_reports),
            "context_status_counts": dict(context_status_counts),
            "context_role_counts": dict(unit_role_counts),
            "inventory_comment_rows": int(len(sidecars["inventory"])),
            "context_unit_count": int(len(sidecars["context_units"])),
            "context_unit_comment_rows": int(len(sidecars["context_map"])),
        },
        "validations": validations,
        "compatibility_policy": {
            "does_not_modify_daily_sidecars": True,
            "does_not_modify_previous_rag": True,
            "does_not_modify_xiao": True,
            "does_not_modify_retrospective_replay": True,
            "does_not_modify_g1_g2": True,
            "does_not_modify_canonical_datasets": True,
            "does_not_call_llm": True,
            "does_not_call_serper": True,
            "does_not_create_embeddings": True,
            "does_not_use_vectorstore": True,
        },
        "limitations": [
            "R-D2 does not evaluate events; reports are structural stubs.",
            "Large events are marked for context selection policy instead of being truncated.",
            "Some threads may be split between alert evidence and prior validation context by design.",
        ],
        "notes": config.notes,
        "params": config.params,
    }

    _write_jsonl(output_paths["daily_rag_validation_inputs"], validation_inputs)
    _write_jsonl(output_paths["daily_rag_context_payloads"], context_payloads)
    _write_jsonl(output_paths["daily_rag_validation_reports_stub"], report_stubs)
    _write_jsonl(output_paths["daily_rag_context_size_report"], size_reports)
    _write_json(output_paths["daily_rag_consumer_manifest"], manifest)

    if validations["status"] != "passed":
        raise ValueError("Daily RAG consumer validation failed: " + "; ".join(validations["errors"]))

    return {
        "run_id": consumer_run_id,
        "output_dir": _normalize_path(output_root),
        "output_paths": output_paths,
        "counts": manifest["counts"],
        "validation_status": validations["status"],
        "future_leak_count": validations["future_leak_count"],
    }


def write_daily_rag_consumer_artifacts(**kwargs: Any) -> dict[str, Any]:
    return write_daily_rag_consumer_artifacts_from_config(
        DailyRagConsumerConfig(**kwargs)
    )


__all__ = [
    "DAILY_RAG_CONSUMER_ARTIFACT_VERSION",
    "DailyRagConsumerConfig",
    "write_daily_rag_consumer_artifacts",
    "write_daily_rag_consumer_artifacts_from_config",
]
