from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


DAILY_CONTEXT_SELECTION_ARTIFACT_VERSION = "daily_rag_context_selection_v1"
DAILY_RAG_VALIDATION_INPUTS_FILE = "daily_rag_validation_inputs.jsonl"
DAILY_RAG_CONTEXT_PAYLOADS_FILE = "daily_rag_context_payloads.jsonl"
DAILY_RAG_CONTEXT_SIZE_REPORT_FILE = "daily_rag_context_size_report.jsonl"
DAILY_RAG_CONSUMER_MANIFEST_FILE = "daily_rag_consumer_manifest.json"
DAILY_RAG_VALIDATION_REPORTS_STUB_FILE = "daily_rag_validation_reports_stub.jsonl"

DAILY_EVENT_COMMENT_INVENTORY_FILE = "daily_event_comment_inventory.csv"
DAILY_RAG_CONTEXT_UNITS_FILE = "daily_rag_context_units.jsonl"
DAILY_CONTEXT_UNIT_COMMENT_MAP_FILE = "daily_context_unit_comment_map.csv"

DAILY_RAG_SELECTED_CONTEXT_PAYLOADS_FILE = "daily_rag_selected_context_payloads.jsonl"
DAILY_CONTEXT_SELECTION_MANIFEST_FILE = "daily_context_selection_manifest.json"
DAILY_CONTEXT_SELECTION_COVERAGE_REPORT_FILE = "daily_context_selection_coverage_report.jsonl"
DAILY_CONTEXT_SELECTION_OMISSIONS_FILE = "daily_context_selection_omissions.csv"
DAILY_CONTEXT_SELECTION_UNIT_MAP_FILE = "daily_context_selection_unit_map.csv"


@dataclass(frozen=True)
class DailyContextSelectionConfig:
    consumer_dir: str | Path | None = None
    sidecars_dir: str | Path | None = None
    output_dir: str | Path | None = None
    max_selected_tokens_per_event: int = 16_000
    alert_coverage_target: float = 0.35
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
    def from_mapping(cls, payload: dict[str, Any]) -> "DailyContextSelectionConfig":
        config_payload = payload.get("daily_rag_context_selection", payload)
        if not isinstance(config_payload, dict):
            raise ValueError("daily_rag_context_selection config section must be an object.")
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(config_payload) - allowed)
        if unknown:
            raise ValueError(f"Unknown daily context selection config fields: {unknown}")
        params = config_payload.get("params") or {}
        if not isinstance(params, dict):
            raise ValueError("params must be an object.")
        return cls(**{**config_payload, "params": params})

    def validate(self) -> None:
        if self.max_selected_tokens_per_event < 1:
            raise ValueError("max_selected_tokens_per_event must be >= 1.")
        if not 0 <= self.alert_coverage_target <= 1:
            raise ValueError("alert_coverage_target must be between 0 and 1.")
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
                "Daily context selection is non-generative. These flags must remain false: "
                + ", ".join(enabled)
            )
        if self.consumer_dir is None:
            raise ValueError("consumer_dir is required.")
        if self.sidecars_dir is None:
            raise ValueError("sidecars_dir is required.")
        if self.output_dir is None:
            raise ValueError("output_dir is required.")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _short_hash(*parts: Any) -> str:
    payload = "|".join("" if part is None else str(part) for part in parts)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def _normalize_path(path: str | Path | None) -> str | None:
    if path is None:
        return None
    return Path(path).as_posix()


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


def _hash_ids(ids: list[str]) -> str:
    payload = "|".join(sorted(str(item) for item in ids))
    return "sha1_" + hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _consumer_paths(consumer_dir: str | Path) -> dict[str, Path]:
    root = Path(consumer_dir)
    return {
        "daily_rag_validation_inputs": root / DAILY_RAG_VALIDATION_INPUTS_FILE,
        "daily_rag_context_payloads": root / DAILY_RAG_CONTEXT_PAYLOADS_FILE,
        "daily_rag_context_size_report": root / DAILY_RAG_CONTEXT_SIZE_REPORT_FILE,
        "daily_rag_consumer_manifest": root / DAILY_RAG_CONSUMER_MANIFEST_FILE,
        "daily_rag_validation_reports_stub": root / DAILY_RAG_VALIDATION_REPORTS_STUB_FILE,
    }


def _sidecar_paths(sidecars_dir: str | Path) -> dict[str, Path]:
    root = Path(sidecars_dir)
    return {
        "daily_event_comment_inventory": root / DAILY_EVENT_COMMENT_INVENTORY_FILE,
        "daily_rag_context_units": root / DAILY_RAG_CONTEXT_UNITS_FILE,
        "daily_context_unit_comment_map": root / DAILY_CONTEXT_UNIT_COMMENT_MAP_FILE,
    }


def _output_paths(output_dir: str | Path) -> dict[str, str]:
    root = Path(output_dir)
    return {
        "daily_rag_selected_context_payloads": (
            root / DAILY_RAG_SELECTED_CONTEXT_PAYLOADS_FILE
        ).as_posix(),
        "daily_context_selection_manifest": (
            root / DAILY_CONTEXT_SELECTION_MANIFEST_FILE
        ).as_posix(),
        "daily_context_selection_coverage_report": (
            root / DAILY_CONTEXT_SELECTION_COVERAGE_REPORT_FILE
        ).as_posix(),
        "daily_context_selection_omissions": (
            root / DAILY_CONTEXT_SELECTION_OMISSIONS_FILE
        ).as_posix(),
        "daily_context_selection_unit_map": (
            root / DAILY_CONTEXT_SELECTION_UNIT_MAP_FILE
        ).as_posix(),
    }


def _load_inputs(config: DailyContextSelectionConfig) -> dict[str, Any]:
    consumer_paths = _consumer_paths(config.consumer_dir)
    sidecar_paths = _sidecar_paths(config.sidecars_dir)
    payloads = _read_jsonl(consumer_paths["daily_rag_context_payloads"])
    validation_inputs = _read_jsonl(consumer_paths["daily_rag_validation_inputs"])
    size_report = _read_jsonl(consumer_paths["daily_rag_context_size_report"])
    consumer_manifest = _read_json(consumer_paths["daily_rag_consumer_manifest"])
    report_stubs = _read_jsonl(consumer_paths["daily_rag_validation_reports_stub"])
    inventory = _read_csv(sidecar_paths["daily_event_comment_inventory"])
    units = pd.DataFrame(_read_jsonl(sidecar_paths["daily_rag_context_units"]))
    context_map = _read_csv(sidecar_paths["daily_context_unit_comment_map"])
    for column in ["is_alert_evidence", "is_validation_context"]:
        if column in inventory.columns:
            inventory[column] = _bool_series(inventory[column])
        if column in context_map.columns:
            context_map[column] = _bool_series(context_map[column])
    if "event_time_utc" in inventory.columns:
        inventory["event_time_utc"] = pd.to_datetime(inventory["event_time_utc"], utc=True)
    if "data_cutoff_utc" in inventory.columns:
        inventory["data_cutoff_utc"] = pd.to_datetime(inventory["data_cutoff_utc"], utc=True)
    return {
        "consumer_paths": consumer_paths,
        "sidecar_paths": sidecar_paths,
        "payloads": payloads,
        "validation_inputs": validation_inputs,
        "size_report": size_report,
        "consumer_manifest": consumer_manifest,
        "report_stubs": report_stubs,
        "inventory": inventory,
        "units": units,
        "context_map": context_map,
    }


def _derive_run_id(config: DailyContextSelectionConfig, consumer_manifest: dict[str, Any]) -> str:
    if config.run_id:
        return config.run_id
    return "dragselect_" + _short_hash(
        Path(config.consumer_dir).as_posix(),
        Path(config.sidecars_dir).as_posix(),
        Path(config.output_dir).as_posix(),
        consumer_manifest.get("run_id"),
    )


def _unit_index(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    units = payload.get("used_context_units") or []
    return {str(unit["context_unit_id"]): unit for unit in units}


def _unit_tokens(unit: dict[str, Any]) -> int:
    return int(unit.get("estimated_tokens") or 0)


def _unit_comment_ids(unit: dict[str, Any]) -> list[str]:
    return [str(value) for value in unit.get("comment_ids") or []]


def _video_alert_counts(payload: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in payload.get("grouping_by_video") or []:
        counts[str(item["video_id"])] = int(item.get("alert_evidence_comment_count") or 0)
    return counts


def _alert_units(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return list(payload.get("alert_evidence_units") or [])


def _validation_units(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return list(payload.get("validation_context_units") or [])


def _sort_alert_units(units: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        units,
        key=lambda unit: (
            str(unit.get("video_id")),
            -int(unit.get("comment_count") or 0),
            str(unit.get("time_start_utc") or ""),
            str(unit.get("context_unit_id")),
        ),
    )


def _sort_validation_units(units: list[dict[str, Any]], selected_videos: set[str]) -> list[dict[str, Any]]:
    return sorted(
        units,
        key=lambda unit: (
            0 if str(unit.get("video_id")) in selected_videos else 1,
            str(unit.get("video_id")),
            str(unit.get("time_start_utc") or ""),
            -int(unit.get("comment_count") or 0),
            str(unit.get("context_unit_id")),
        ),
    )


def _add_unit(
    unit: dict[str, Any],
    *,
    selected: dict[str, dict[str, Any]],
    selected_order: list[str],
    selected_tokens: int,
    max_tokens: int,
) -> tuple[bool, int]:
    unit_id = str(unit["context_unit_id"])
    if unit_id in selected:
        return True, selected_tokens
    tokens = _unit_tokens(unit)
    if tokens > max_tokens:
        return False, selected_tokens
    if selected_tokens + tokens > max_tokens:
        return False, selected_tokens
    selected[unit_id] = unit
    selected_order.append(unit_id)
    return True, selected_tokens + tokens


def _select_for_payload(
    payload: dict[str, Any],
    *,
    config: DailyContextSelectionConfig,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    max_tokens = config.max_selected_tokens_per_event
    all_units_by_id = _unit_index(payload)
    alert_units = _alert_units(payload)
    validation_units = _validation_units(payload)
    total_alert_units = len(alert_units)
    total_validation_units = len(validation_units)
    total_alert_comments = int(payload.get("alert_evidence_comment_count") or 0)
    total_validation_comments = int(payload.get("validation_context_comment_count") or 0)
    total_tokens = int(payload.get("estimated_input_tokens") or 0)
    video_counts = _video_alert_counts(payload)
    videos_ordered = [
        video_id
        for video_id, count in sorted(
            video_counts.items(),
            key=lambda item: (-item[1], item[0]),
        )
        if count > 0
    ]

    alert_by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for unit in _sort_alert_units(alert_units):
        alert_by_video[str(unit.get("video_id"))].append(unit)

    selected: dict[str, dict[str, Any]] = {}
    selected_order: list[str] = []
    selected_tokens = 0
    omission_reasons: dict[str, str] = {}

    # Pass 1: one alert unit per active video while budget allows.
    for video_id in videos_ordered:
        units_for_video = alert_by_video.get(video_id, [])
        if not units_for_video:
            continue
        chosen = units_for_video[0]
        ok, selected_tokens = _add_unit(
            chosen,
            selected=selected,
            selected_order=selected_order,
            selected_tokens=selected_tokens,
            max_tokens=max_tokens,
        )
        if not ok:
            omission_reasons[str(chosen["context_unit_id"])] = "token_limit"

    # Pass 2: add more alert evidence until coverage target or budget limit.
    def selected_alert_comment_count() -> int:
        return sum(
            int(unit.get("comment_count") or 0)
            for unit in selected.values()
            if unit.get("context_role") == "alert_evidence_unit"
        )

    for unit in sorted(
        alert_units,
        key=lambda item: (
            -int(item.get("comment_count") or 0),
            str(item.get("video_id")),
            str(item.get("time_start_utc") or ""),
            str(item.get("context_unit_id")),
        ),
    ):
        if total_alert_comments and selected_alert_comment_count() / total_alert_comments >= config.alert_coverage_target:
            break
        ok, selected_tokens = _add_unit(
            unit,
            selected=selected,
            selected_order=selected_order,
            selected_tokens=selected_tokens,
            max_tokens=max_tokens,
        )
        if not ok and str(unit["context_unit_id"]) not in selected:
            omission_reasons.setdefault(str(unit["context_unit_id"]), "token_limit")

    selected_video_ids = {str(unit.get("video_id")) for unit in selected.values()}

    # Pass 3: validation context only after alert evidence, and only for videos
    # that already have selected alert evidence. Context from omitted videos stays
    # available in R-D1/R-D2, but is not promoted to the future model view here.
    validation_candidates: list[dict[str, Any]] = []
    for unit in validation_units:
        unit_id = str(unit["context_unit_id"])
        if str(unit.get("video_id")) not in selected_video_ids:
            omission_reasons.setdefault(unit_id, "video_quota_reached")
            continue
        validation_candidates.append(unit)

    for unit in _sort_validation_units(validation_candidates, selected_video_ids):
        ok, selected_tokens = _add_unit(
            unit,
            selected=selected,
            selected_order=selected_order,
            selected_tokens=selected_tokens,
            max_tokens=max_tokens,
        )
        if not ok and str(unit["context_unit_id"]) not in selected:
            omission_reasons.setdefault(str(unit["context_unit_id"]), "context_after_alert_budget")

    selected_units = [selected[unit_id] for unit_id in selected_order]
    selected_alert_units = [
        unit for unit in selected_units if unit.get("context_role") == "alert_evidence_unit"
    ]
    selected_validation_units = [
        unit for unit in selected_units if unit.get("context_role") == "validation_context_unit"
    ]
    selected_comment_ids: list[str] = []
    for unit in selected_units:
        selected_comment_ids.extend(_unit_comment_ids(unit))
    selected_comment_ids = sorted(set(selected_comment_ids))
    selected_alert_comment_ids: set[str] = set()
    selected_validation_comment_ids: set[str] = set()
    for unit in selected_alert_units:
        selected_alert_comment_ids.update(_unit_comment_ids(unit))
    for unit in selected_validation_units:
        selected_validation_comment_ids.update(_unit_comment_ids(unit))

    all_video_ids = set(str(value) for value in payload.get("video_ids") or [])
    selected_video_ids = set(str(unit.get("video_id")) for unit in selected_units)
    omitted_video_ids = sorted(all_video_ids.difference(selected_video_ids))

    selected_unit_ids = set(selected_order)
    omissions: list[dict[str, Any]] = []
    for unit_id, unit in all_units_by_id.items():
        if unit_id in selected_unit_ids:
            continue
        reason = omission_reasons.get(unit_id)
        if reason is None:
            if unit.get("context_role") == "validation_context_unit":
                reason = "context_after_alert_budget"
            else:
                reason = "lower_priority"
        omissions.append(
            {
                "daily_rag_event_id": payload["daily_rag_event_id"],
                "daily_event_id": payload["daily_event_id"],
                "cycle_id": payload["cycle_id"],
                "cycle_index": payload["cycle_index"],
                "context_unit_id": unit_id,
                "video_id": unit.get("video_id"),
                "context_role": unit.get("context_role"),
                "temporal_scope": unit.get("temporal_scope"),
                "comment_count": int(unit.get("comment_count") or 0),
                "estimated_tokens": _unit_tokens(unit),
                "omission_reason": reason,
            }
        )

    selected_by_video: list[dict[str, Any]] = []
    for video_id in sorted(selected_video_ids):
        video_units = [unit for unit in selected_units if str(unit.get("video_id")) == video_id]
        selected_by_video.append(
            {
                "video_id": video_id,
                "selected_context_unit_ids": [str(unit["context_unit_id"]) for unit in video_units],
                "selected_alert_evidence_unit_count": sum(
                    unit.get("context_role") == "alert_evidence_unit" for unit in video_units
                ),
                "selected_validation_context_unit_count": sum(
                    unit.get("context_role") == "validation_context_unit" for unit in video_units
                ),
                "selected_comment_count": len(
                    set(comment_id for unit in video_units for comment_id in _unit_comment_ids(unit))
                ),
                "selected_token_estimate": sum(_unit_tokens(unit) for unit in video_units),
            }
        )

    alert_coverage = (
        len(selected_alert_comment_ids) / total_alert_comments
        if total_alert_comments
        else 0.0
    )
    validation_coverage = (
        len(selected_validation_comment_ids) / total_validation_comments
        if total_validation_comments
        else 0.0
    )
    video_coverage = len(selected_video_ids) / len(all_video_ids) if all_video_ids else 0.0
    context_selection_status = (
        "complete_within_token_limit"
        if not omissions
        else "partial_due_to_token_limit"
    )
    if selected_tokens > max_tokens:
        context_selection_status = "exceeds_token_limit_recorded"

    selected_payload = {
        "daily_rag_event_id": payload["daily_rag_event_id"],
        "daily_event_id": payload["daily_event_id"],
        "cycle_id": payload["cycle_id"],
        "cycle_index": payload["cycle_index"],
        "detector_name": payload["detector_name"],
        "signal_name": payload["signal_name"],
        "signal_value": payload["signal_value"],
        "baseline_mean": payload["baseline_mean"],
        "ratio_to_baseline": payload["ratio_to_baseline"],
        "delta_value": payload["delta_value"],
        "pct_change_value": payload["pct_change_value"],
        "analysis_window_start_utc": payload["analysis_window_start_utc"],
        "analysis_window_end_utc": payload["analysis_window_end_utc"],
        "data_cutoff_utc": payload["data_cutoff_utc"],
        "context_selection_policy": "deterministic_role_video_token_budget_v1",
        "max_selected_tokens_per_event": max_tokens,
        "selected_token_estimate": selected_tokens,
        "context_selection_status": context_selection_status,
        "selected_alert_evidence_unit_ids": [
            str(unit["context_unit_id"]) for unit in selected_alert_units
        ],
        "selected_validation_context_unit_ids": [
            str(unit["context_unit_id"]) for unit in selected_validation_units
        ],
        "selected_context_unit_ids": selected_order,
        "selected_comment_ids_hash": _hash_ids(selected_comment_ids),
        "selected_video_ids": sorted(selected_video_ids),
        "omitted_video_ids": omitted_video_ids,
        "alert_evidence_coverage": alert_coverage,
        "validation_context_coverage": validation_coverage,
        "video_coverage": video_coverage,
        "selected_context_by_video": selected_by_video,
        "selected_context_units": selected_units,
        "limitations": [
            "Selection is deterministic and non-semantic.",
            "Validation context is added only after alert evidence.",
            "Omitted units remain available in sidecars and R-D2 payloads.",
            "No LLM, Serper, embeddings, or vectorstore were used.",
        ],
    }

    coverage = {
        "daily_rag_event_id": payload["daily_rag_event_id"],
        "daily_event_id": payload["daily_event_id"],
        "total_alert_evidence_units": total_alert_units,
        "selected_alert_evidence_units": len(selected_alert_units),
        "total_validation_context_units": total_validation_units,
        "selected_validation_context_units": len(selected_validation_units),
        "total_alert_evidence_comments": total_alert_comments,
        "selected_alert_evidence_comments": len(selected_alert_comment_ids),
        "total_validation_context_comments": total_validation_comments,
        "selected_validation_context_comments": len(selected_validation_comment_ids),
        "total_videos": len(all_video_ids),
        "selected_videos": len(selected_video_ids),
        "omitted_videos": len(omitted_video_ids),
        "total_estimated_tokens": total_tokens,
        "selected_estimated_tokens": selected_tokens,
        "coverage_reason": (
            "token budget exhausted before full event context could be selected"
            if omissions
            else "all context selected within token budget"
        ),
        "context_selection_status": context_selection_status,
        "alert_evidence_coverage": alert_coverage,
        "validation_context_coverage": validation_coverage,
        "video_coverage": video_coverage,
    }

    unit_map_rows: list[dict[str, Any]] = []
    selected_order_index = {unit_id: index + 1 for index, unit_id in enumerate(selected_order)}
    for unit_id in selected_order:
        unit = selected[unit_id]
        comment_ids = _unit_comment_ids(unit)
        for order_in_selected_unit, comment_id in enumerate(comment_ids, start=1):
            unit_map_rows.append(
                {
                    "daily_rag_event_id": payload["daily_rag_event_id"],
                    "daily_event_id": payload["daily_event_id"],
                    "cycle_id": payload["cycle_id"],
                    "cycle_index": payload["cycle_index"],
                    "context_unit_id": unit_id,
                    "video_id": unit.get("video_id"),
                    "context_role": unit.get("context_role"),
                    "temporal_scope": unit.get("temporal_scope"),
                    "comment_id": comment_id,
                    "order_in_selected_context_unit": order_in_selected_unit,
                    "selection_order": selected_order_index.get(unit_id),
                    "estimated_tokens": _unit_tokens(unit),
                    "unit_comment_count": int(unit.get("comment_count") or 0),
                    "selected": True,
                }
            )

    return selected_payload, coverage, omissions, unit_map_rows


def _validate_selection(
    *,
    payloads: list[dict[str, Any]],
    selected_payloads: list[dict[str, Any]],
    omissions: list[dict[str, Any]],
    unit_map: pd.DataFrame,
    context_units: pd.DataFrame,
    context_map: pd.DataFrame,
    inventory: pd.DataFrame,
    max_tokens: int,
) -> dict[str, Any]:
    errors: list[str] = []
    input_event_ids = {str(payload["daily_event_id"]) for payload in payloads}
    output_event_ids = {str(payload["daily_event_id"]) for payload in selected_payloads}
    if input_event_ids != output_event_ids:
        errors.append(
            f"event_output_mismatch missing={sorted(input_event_ids-output_event_ids)} "
            f"extra={sorted(output_event_ids-input_event_ids)}"
        )

    selected_unit_ids = {
        str(unit_id)
        for payload in selected_payloads
        for unit_id in payload.get("selected_context_unit_ids", [])
    }
    all_payload_unit_ids = {
        str(unit["context_unit_id"])
        for payload in payloads
        for unit in payload.get("used_context_units", [])
    }
    all_unit_ids = set(context_units["context_unit_id"].astype(str))
    missing_units = selected_unit_ids.difference(all_unit_ids)
    if missing_units:
        errors.append(f"selected_units_missing={len(missing_units)}")

    if not unit_map.empty:
        unit_map_ids = set(unit_map["context_unit_id"].astype(str))
        unit_map_unknown = unit_map_ids.difference(selected_unit_ids)
        if unit_map_unknown:
            errors.append(f"unit_map_contains_unselected_units={len(unit_map_unknown)}")
        selected_units_without_map = selected_unit_ids.difference(unit_map_ids)
        if selected_units_without_map:
            errors.append(f"selected_units_without_comment_map={len(selected_units_without_map)}")

    selected_context_map = context_map.loc[
        context_map["context_unit_id"].astype(str).isin(selected_unit_ids)
    ].copy()
    selected_comment_ids = set(selected_context_map["comment_id"].astype(str))
    inventory_comment_ids = set(inventory["comment_id"].astype(str))
    selected_comments_missing = selected_comment_ids.difference(inventory_comment_ids)
    if selected_comments_missing:
        errors.append(f"selected_comments_missing={len(selected_comments_missing)}")
    if not unit_map.empty:
        unit_map_comment_ids = set(unit_map["comment_id"].astype(str))
        unit_map_comments_missing = unit_map_comment_ids.difference(inventory_comment_ids)
        if unit_map_comments_missing:
            errors.append(f"unit_map_comments_missing={len(unit_map_comments_missing)}")
        unit_map_extra_comments = unit_map_comment_ids.difference(selected_comment_ids)
        if unit_map_extra_comments:
            errors.append(f"unit_map_extra_comments={len(unit_map_extra_comments)}")

    inv_selected = inventory.loc[inventory["comment_id"].astype(str).isin(selected_comment_ids)].copy()
    if not inv_selected.empty:
        future_leak_count = int((inv_selected["event_time_utc"] >= inv_selected["data_cutoff_utc"]).sum())
    else:
        future_leak_count = 0
    if future_leak_count:
        errors.append(f"future_leak_count={future_leak_count}")

    selected_units = context_units.loc[
        context_units["context_unit_id"].astype(str).isin(selected_unit_ids)
    ]
    mixed_role_count = (
        int((selected_units["context_role"] == "mixed_unit").sum())
        if "context_role" in selected_units.columns
        else 0
    )
    mixed_type_count = (
        int((selected_units["context_type"] == "mixed_unit").sum())
        if "context_type" in selected_units.columns
        else 0
    )
    mixed_unit_count = mixed_role_count + mixed_type_count
    if mixed_unit_count:
        errors.append(f"selected_mixed_unit_count={mixed_unit_count}")

    unit_video_counts = selected_context_map.groupby("context_unit_id")["video_id"].nunique()
    units_mixing_videos = int((unit_video_counts > 1).sum())
    if units_mixing_videos:
        errors.append(f"selected_units_mixing_videos={units_mixing_videos}")

    over_budget = [
        payload["daily_event_id"]
        for payload in selected_payloads
        if payload["selected_token_estimate"] > max_tokens
        and payload["context_selection_status"] != "exceeds_token_limit_recorded"
    ]
    if over_budget:
        errors.append(f"over_budget_without_status={over_budget}")

    omitted_ids = set(str(row["context_unit_id"]) for row in omissions)
    unselected_ids = all_payload_unit_ids.difference(selected_unit_ids)
    missing_omission_reason = sum(
        1 for row in omissions if not row.get("omission_reason")
    )
    if not unselected_ids.issubset(omitted_ids):
        errors.append("not_all_unselected_units_have_omission_row")
    if missing_omission_reason:
        errors.append(f"missing_omission_reason={missing_omission_reason}")

    role_order_errors = 0
    for payload in selected_payloads:
        ordered_roles = [
            str(unit.get("context_role"))
            for unit in payload.get("selected_context_units", [])
        ]
        first_validation_index = next(
            (
                index
                for index, role in enumerate(ordered_roles)
                if role == "validation_context_unit"
            ),
            None,
        )
        if first_validation_index is None:
            continue
        if any(role == "alert_evidence_unit" for role in ordered_roles[first_validation_index + 1 :]):
            role_order_errors += 1
    if role_order_errors:
        errors.append(f"alert_units_after_validation_units={role_order_errors}")

    return {
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "events_in_inputs": len(input_event_ids),
        "events_in_outputs": len(output_event_ids),
        "selected_unit_count": len(selected_unit_ids),
        "selected_comment_count": len(selected_comment_ids),
        "selected_units_missing": len(missing_units),
        "selected_comments_missing": len(selected_comments_missing),
        "future_leak_count": future_leak_count,
        "selected_mixed_unit_count": mixed_unit_count,
        "selected_units_mixing_videos": units_mixing_videos,
        "unselected_units_with_omission_reason": len(unselected_ids.intersection(omitted_ids)),
        "alert_role_priority_errors": role_order_errors,
        "external_calls": {
            "llm": 0,
            "serper": 0,
            "embeddings": False,
            "vectorstore": False,
            "g1": False,
            "g2": False,
        },
    }


def write_daily_context_selection_artifacts_from_config(
    config: DailyContextSelectionConfig,
) -> dict[str, Any]:
    config.validate()
    loaded = _load_inputs(config)
    output_root = Path(config.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    output_paths = _output_paths(output_root)
    created_at = _utc_now_iso()
    run_id = _derive_run_id(config, loaded["consumer_manifest"])

    selected_payloads: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    omission_rows: list[dict[str, Any]] = []
    unit_map_rows: list[dict[str, Any]] = []
    for payload in loaded["payloads"]:
        selected, coverage, omissions, unit_map = _select_for_payload(payload, config=config)
        selected_payloads.append(selected)
        coverage_rows.append(coverage)
        omission_rows.extend(omissions)
        unit_map_rows.extend(unit_map)

    unit_map_df = pd.DataFrame(unit_map_rows)
    omissions_df = pd.DataFrame(omission_rows)
    validations = _validate_selection(
        payloads=loaded["payloads"],
        selected_payloads=selected_payloads,
        omissions=omission_rows,
        unit_map=unit_map_df,
        context_units=loaded["units"],
        context_map=loaded["context_map"],
        inventory=loaded["inventory"],
        max_tokens=config.max_selected_tokens_per_event,
    )
    status_counts = Counter(row["context_selection_status"] for row in selected_payloads)
    unique_selected_units = {
        str(unit_id): str(unit.get("context_role"))
        for payload in selected_payloads
        for unit in payload.get("selected_context_units", [])
        for unit_id in [unit.get("context_unit_id")]
    }
    selected_unit_role_counts = Counter(unique_selected_units.values())

    manifest = {
        "run_id": run_id,
        "created_at_utc": created_at,
        "pipeline_stage": "daily_rag_context_selection",
        "mode": "deterministic_non_semantic_no_generation",
        "artifact_version": DAILY_CONTEXT_SELECTION_ARTIFACT_VERSION,
        "source_consumer_dir": _normalize_path(config.consumer_dir),
        "source_sidecars_dir": _normalize_path(config.sidecars_dir),
        "source_artifacts": {
            **{name: _normalize_path(path) for name, path in loaded["consumer_paths"].items()},
            **{name: _normalize_path(path) for name, path in loaded["sidecar_paths"].items()},
        },
        "output_paths": output_paths,
        "context_selection_policy": {
            "name": "deterministic_role_video_token_budget_v1",
            "max_selected_tokens_per_event": config.max_selected_tokens_per_event,
            "alert_coverage_target": config.alert_coverage_target,
            "semantic_ranking": False,
            "embeddings": False,
            "llm": False,
            "selection_order": [
                "one_alert_evidence_unit_per_video_by_alert_comment_count",
                "additional_alert_evidence_units_until_target_or_budget",
                "validation_context_units_for_selected_videos_then_other_videos",
            ],
        },
        "counts": {
            "events_processed": len(selected_payloads),
            "selected_payloads": len(selected_payloads),
            "coverage_rows": len(coverage_rows),
            "omission_rows": len(omission_rows),
            "unit_map_rows": len(unit_map_rows),
            "status_counts": dict(status_counts),
            "selected_unit_role_counts": dict(selected_unit_role_counts),
        },
        "validations": validations,
        "compatibility_policy": {
            "does_not_modify_daily_sidecars": True,
            "does_not_modify_daily_consumer": True,
            "does_not_modify_previous_rag": True,
            "does_not_modify_xiao": True,
            "does_not_modify_retrospective_replay": True,
            "does_not_modify_g1_g2": True,
            "does_not_call_llm": True,
            "does_not_call_serper": True,
            "does_not_create_embeddings": True,
            "does_not_use_vectorstore": True,
        },
        "limitations": [
            "Selection is deterministic and does not assess semantic relevance.",
            "Low-volume videos can still be omitted when token budget is exhausted.",
            "Thread continuity may be split by role-based unit construction.",
        ],
        "notes": config.notes,
        "params": config.params,
    }

    _write_jsonl(output_paths["daily_rag_selected_context_payloads"], selected_payloads)
    _write_json(output_paths["daily_context_selection_manifest"], manifest)
    _write_jsonl(output_paths["daily_context_selection_coverage_report"], coverage_rows)
    omissions_df.to_csv(output_paths["daily_context_selection_omissions"], index=False)
    unit_map_df.to_csv(output_paths["daily_context_selection_unit_map"], index=False)

    if validations["status"] != "passed":
        raise ValueError("Daily context selection validation failed: " + "; ".join(validations["errors"]))

    return {
        "run_id": run_id,
        "output_dir": _normalize_path(output_root),
        "output_paths": output_paths,
        "counts": manifest["counts"],
        "validation_status": validations["status"],
    }


def write_daily_context_selection_artifacts(**kwargs: Any) -> dict[str, Any]:
    return write_daily_context_selection_artifacts_from_config(
        DailyContextSelectionConfig(**kwargs)
    )


__all__ = [
    "DAILY_CONTEXT_SELECTION_ARTIFACT_VERSION",
    "DailyContextSelectionConfig",
    "write_daily_context_selection_artifacts",
    "write_daily_context_selection_artifacts_from_config",
]
