from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


SIGNALS_DRY_RUN_MODE = "signals_dry_run"
DEFAULT_XIAO_SIGNAL_NAME = "active_window_comment_count"
INTERVAL_POLICY = "semi_open_daily_bounds_start_inclusive_end_exclusive"


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return _format_utc(value)
    if pd.isna(value) if not isinstance(value, (list, dict, tuple, set)) else False:
        return None
    return value


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required JSON artifact not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Required JSONL artifact not found: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"JSONL line {line_number} must contain an object: {path}")
        rows.append(payload)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(_json_ready(row), ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required CSV artifact not found: {path}")
    return pd.read_csv(path)


def _read_table(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if p.is_dir() or p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    raise ValueError(f"Unsupported canonical dataset format: {p}")


def _bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def _hash_ids(ids: list[str] | set[str]) -> str:
    joined = "\n".join(sorted(str(value) for value in ids))
    return "sha1_" + hashlib.sha1(joined.encode("utf-8")).hexdigest()


def _format_utc(value: Any) -> str | None:
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    return pd.Timestamp(ts).strftime("%Y-%m-%dT%H:%M:%SZ")


def _signal_date(cycle: dict[str, Any]) -> str:
    local_start = cycle.get("collection_window_start_local")
    if local_start:
        return str(local_start)[:10]
    end = pd.Timestamp(cycle["analysis_window_end_utc"])
    return (end - pd.Timedelta(days=1)).date().isoformat()


@dataclass
class CyclicDailySignalConfig:
    simulation_dir: str | Path
    canonical_dataset_path: str | Path
    mode: str = SIGNALS_DRY_RUN_MODE
    output_dir: str | Path | None = None
    xiao_signal_name: str = DEFAULT_XIAO_SIGNAL_NAME
    max_cycles: int | None = None
    run_xiao: bool = False
    run_detection: bool = False
    run_rag: bool = False
    run_llm: bool = False
    run_serper: bool = False
    use_embeddings: bool = False
    use_vectorstore: bool = False

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "CyclicDailySignalConfig":
        config_payload = payload.get("cyclic_daily_signals", payload)
        if not isinstance(config_payload, dict):
            raise ValueError("Cyclic daily signal config must be an object.")
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(config_payload) - allowed)
        if unknown:
            raise ValueError(f"Unknown cyclic daily signal config fields: {unknown}")
        return cls(**config_payload)

    def simulation_path(self) -> Path:
        return Path(self.simulation_dir)

    def output_path(self) -> Path:
        return Path(self.output_dir) if self.output_dir is not None else self.simulation_path()

    def validate_c5_scope(self) -> None:
        if self.mode != SIGNALS_DRY_RUN_MODE:
            raise ValueError("C-5 currently supports only mode='signals_dry_run'.")
        if self.max_cycles is not None and self.max_cycles < 1:
            raise ValueError("max_cycles must be >= 1 when provided.")
        forbidden = {
            "run_xiao": self.run_xiao,
            "run_detection": self.run_detection,
            "run_rag": self.run_rag,
            "run_llm": self.run_llm,
            "run_serper": self.run_serper,
            "use_embeddings": self.use_embeddings,
            "use_vectorstore": self.use_vectorstore,
        }
        enabled = [name for name, value in forbidden.items() if value]
        if enabled:
            raise ValueError(
                "C-5 signals_dry_run only builds daily signals. These flags must remain false: "
                + ", ".join(enabled)
            )


def _load_and_validate_gold(config: CyclicDailySignalConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    gold_path = Path(config.canonical_dataset_path)
    gold = _read_table(gold_path).copy()
    required = ["comment_id", "video_id", "event_time_utc"]
    missing = sorted(set(required) - set(gold.columns))
    if missing:
        raise ValueError(f"Gold canonical dataset missing required fields: {missing}")
    gold["comment_id"] = gold["comment_id"].astype(str)
    duplicates = gold["comment_id"].duplicated(keep=False)
    duplicate_count = int(duplicates.sum())
    if duplicate_count:
        sample = sorted(gold.loc[duplicates, "comment_id"].unique().tolist())[:10]
        raise ValueError(
            "comment_id must be unique in Gold before C-5 signals_dry_run. "
            f"duplicate_row_count={duplicate_count}; sample={sample}"
        )
    gold["event_time_utc"] = pd.to_datetime(gold["event_time_utc"], utc=True, errors="coerce")
    invalid_time_count = int(gold["event_time_utc"].isna().sum())
    if invalid_time_count:
        raise ValueError(
            "Gold canonical dataset has invalid event_time_utc values. "
            f"invalid_time_count={invalid_time_count}"
        )
    optional_signal_columns = [
        "author_id",
        "is_reply",
        "emoji_count",
        "exclamation_count",
        "question_count",
        "caps_ratio",
        "sentiment_score",
    ]
    return gold, {
        "canonical_dataset_path": str(gold_path),
        "canonical_dataset_role": "gold_comments",
        "row_count": int(len(gold)),
        "required_columns": required,
        "optional_signal_columns_available": [
            column for column in optional_signal_columns if column in gold.columns
        ],
        "optional_signal_columns_missing": [
            column for column in optional_signal_columns if column not in gold.columns
        ],
        "all_columns": list(gold.columns),
        "comment_id_unique": True,
        "duplicate_comment_id_count": 0,
        "invalid_event_time_count": 0,
        "schema_status": "passed",
    }


def _validate_input_artifacts(
    *,
    adapter_manifest: dict[str, Any],
    cycle_manifest: list[dict[str, Any]],
    monitoring_inputs: list[dict[str, Any]],
    window_inventory: pd.DataFrame,
) -> list[str]:
    errors: list[str] = []
    if adapter_manifest.get("adapter_stage") != "C-3":
        errors.append("cycle_adapter_manifest.json must come from C-3.")
    if adapter_manifest.get("adapter_mode") != "stateful":
        errors.append("C-5 requires C-3 adapter_mode='stateful'.")
    if adapter_manifest.get("interval_policy") != INTERVAL_POLICY:
        errors.append(f"interval_policy must be {INTERVAL_POLICY}.")
    guards = adapter_manifest.get("execution_guards", {})
    for key in ["run_monitoring", "run_detection", "run_rag"]:
        if guards.get(key) is not False:
            errors.append(f"C-3 execution guard {key} must be false.")
    if not cycle_manifest:
        errors.append("cycle_manifest.jsonl must contain at least one cycle.")
    cycle_ids = {row.get("cycle_id") for row in cycle_manifest}
    monitoring_cycle_ids = {row.get("cycle_id") for row in monitoring_inputs}
    if cycle_ids != monitoring_cycle_ids:
        errors.append("cycle_manifest and cycle_monitoring_inputs cycle_id sets must match.")
    required_window = {
        "cycle_id",
        "cycle_index",
        "comment_id",
        "video_id",
        "event_time_utc",
        "analysis_window_start_utc",
        "analysis_window_end_utc",
        "data_cutoff_utc",
        "is_new_in_cycle",
        "is_active_in_window",
        "exited_window",
    }
    missing_window = sorted(required_window - set(window_inventory.columns))
    if missing_window:
        errors.append(f"cycle_window_inventory.csv missing fields: {missing_window}")
    return errors


def _ordered_cycles(cycles: list[dict[str, Any]], max_cycles: int | None) -> list[dict[str, Any]]:
    ordered = sorted(
        cycles,
        key=lambda row: (
            int(row["cycle_index"]),
            pd.Timestamp(row["cycle_run_at_utc"]),
            str(row["cycle_id"]),
        ),
    )
    return ordered if max_cycles is None else ordered[:max_cycles]


def _rows_for_cycle(window_inventory: pd.DataFrame, cycle_id: str) -> pd.DataFrame:
    return window_inventory.loc[window_inventory["cycle_id"] == cycle_id].copy()


def _active_rows_for_cycle(window_inventory: pd.DataFrame, cycle_id: str) -> pd.DataFrame:
    rows = _rows_for_cycle(window_inventory, cycle_id)
    return rows.loc[_bool_series(rows["is_active_in_window"])].copy()


def _exited_count_for_cycle(window_inventory: pd.DataFrame, cycle_id: str) -> int:
    rows = _rows_for_cycle(window_inventory, cycle_id)
    return int(_bool_series(rows["exited_window"]).sum()) if not rows.empty else 0


def _numeric_mean(view: pd.DataFrame, column: str) -> float | None:
    if column not in view.columns or view.empty:
        return None
    values = pd.to_numeric(view[column], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.mean())


def _numeric_std(view: pd.DataFrame, column: str) -> float | None:
    if column not in view.columns or view.empty:
        return None
    values = pd.to_numeric(view[column], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.std(ddof=0))


def _available_signal_columns(gold_schema: dict[str, Any]) -> set[str]:
    return set(gold_schema["optional_signal_columns_available"])


def _resolve_active_view(
    *,
    cycle: dict[str, Any],
    active_rows: pd.DataFrame,
    gold_by_comment_id: pd.DataFrame,
    gold_schema: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    active_rows = active_rows.copy()
    active_rows["comment_id"] = active_rows["comment_id"].astype(str)
    active_comment_ids = active_rows["comment_id"].tolist()
    active_unique_comment_ids = sorted(set(active_comment_ids))
    duplicate_active_count = int(len(active_comment_ids) - len(active_unique_comment_ids))
    missing_comment_ids = sorted(set(active_unique_comment_ids) - set(gold_by_comment_id.index))

    joined = gold_by_comment_id.reindex(active_comment_ids).reset_index()
    joined_count = int(len(joined.dropna(subset=["event_time_utc"])))
    expected_count = int(len(active_rows))
    extra_joined_count = max(0, joined_count - expected_count)
    context_columns = [
        "comment_id",
        "cycle_id",
        "cycle_index",
        "first_seen_cycle_id",
        "window_membership_role",
        "is_new_in_cycle",
        "is_active_in_window",
        "is_accumulated_by_cycle",
        "analysis_window_start_utc",
        "analysis_window_end_utc",
        "data_cutoff_utc",
    ]
    available_context = [column for column in context_columns if column in active_rows.columns]
    view = joined.merge(active_rows[available_context], on="comment_id", how="left")

    event_time = pd.to_datetime(view["event_time_utc"], utc=True, errors="coerce")
    start = pd.Timestamp(cycle["analysis_window_start_utc"])
    end = pd.Timestamp(cycle["analysis_window_end_utc"])
    cutoff = pd.Timestamp(cycle["data_cutoff_utc"])
    future_leak_count = int((event_time >= cutoff).sum())
    outside_analysis_window_count = int(((event_time < start) | (event_time >= end)).sum())
    checks_pass = (
        duplicate_active_count == 0
        and len(missing_comment_ids) == 0
        and joined_count == expected_count
        and extra_joined_count == 0
        and future_leak_count == 0
        and outside_analysis_window_count == 0
        and gold_schema["comment_id_unique"] is True
    )
    return view, {
        "simulation_run_id": cycle["simulation_run_id"],
        "cycle_id": cycle["cycle_id"],
        "cycle_index": int(cycle["cycle_index"]),
        "canonical_dataset_path": gold_schema["canonical_dataset_path"],
        "comment_ids_hash": _hash_ids(active_comment_ids),
        "active_window_comment_count": expected_count,
        "active_unique_comment_count": int(len(active_unique_comment_ids)),
        "duplicate_active_comment_id_count": duplicate_active_count,
        "joined_comment_count": joined_count,
        "missing_comment_id_count": int(len(missing_comment_ids)),
        "missing_comment_id_sample": missing_comment_ids[:10],
        "extra_joined_comment_count": int(extra_joined_count),
        "future_leak_count": future_leak_count,
        "outside_analysis_window_count": outside_analysis_window_count,
        "gold_comment_id_unique": gold_schema["comment_id_unique"],
        "gold_duplicate_comment_id_count": gold_schema["duplicate_comment_id_count"],
        "schema_status": gold_schema["schema_status"],
        "join_status": "passed" if checks_pass else "failed",
        "temporal_status": "passed"
        if future_leak_count == 0 and outside_analysis_window_count == 0
        else "failed",
        "analysis_window_start_utc": cycle["analysis_window_start_utc"],
        "analysis_window_end_utc": cycle["analysis_window_end_utc"],
        "data_cutoff_utc": cycle["data_cutoff_utc"],
        "materialization_mode": "in_memory",
        "full_rows_written": False,
    }


def _build_signal_row(
    *,
    cycle: dict[str, Any],
    active_rows: pd.DataFrame,
    view: pd.DataFrame,
    exited_window_comment_count: int,
    join_report: dict[str, Any],
    gold_schema: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    available = _available_signal_columns(gold_schema)
    active_comment_ids = active_rows["comment_id"].astype(str).tolist()
    new_rows = active_rows.loc[_bool_series(active_rows["is_new_in_cycle"])].copy()
    new_comment_ids = new_rows["comment_id"].astype(str).tolist()
    unavailable: list[dict[str, str]] = []

    def require_column(signal_name: str, column: str) -> bool:
        if column in available or column in view.columns:
            return True
        unavailable.append(
            {
                "signal": signal_name,
                "reason": f"column_not_available:{column}",
            }
        )
        return False

    active_count = int(len(active_comment_ids))
    active_video_count = int(view["video_id"].dropna().astype(str).nunique()) if "video_id" in view else 0
    unique_author_count = (
        int(view["author_id"].nunique(dropna=True))
        if require_column("unique_author_count", "author_id")
        else None
    )
    reply_count = (
        int(_bool_series(view["is_reply"]).sum())
        if require_column("reply_count", "is_reply")
        else None
    )
    reply_ratio = float(reply_count / active_count) if reply_count is not None and active_count else None
    if reply_count is None:
        unavailable.append(
            {
                "signal": "reply_ratio",
                "reason": "depends_on_unavailable:reply_count",
            }
        )

    signal_specs = {
        "emoji_density": "emoji_count",
        "exclaim_density": "exclamation_count",
        "question_density": "question_count",
        "caps_ratio_mean": "caps_ratio",
    }
    numeric_signals: dict[str, float | None] = {}
    for signal_name, column in signal_specs.items():
        if require_column(signal_name, column):
            numeric_signals[signal_name] = _numeric_mean(view, column)
        else:
            numeric_signals[signal_name] = None

    if require_column("sentiment_mean", "sentiment_score"):
        sentiment_mean = _numeric_mean(view, "sentiment_score")
        sentiment_std = _numeric_std(view, "sentiment_score")
    else:
        sentiment_mean = None
        sentiment_std = None
        unavailable.append(
            {
                "signal": "sentiment_std",
                "reason": "column_not_available:sentiment_score",
            }
        )

    row = {
        "simulation_run_id": cycle["simulation_run_id"],
        "cycle_id": cycle["cycle_id"],
        "cycle_index": int(cycle["cycle_index"]),
        "signal_date": _signal_date(cycle),
        "observation_time_utc": cycle["cycle_run_at_utc"],
        "analysis_window_start_utc": cycle["analysis_window_start_utc"],
        "analysis_window_end_utc": cycle["analysis_window_end_utc"],
        "data_cutoff_utc": cycle["data_cutoff_utc"],
        "active_window_comment_count": active_count,
        "new_comment_count": int(len(new_comment_ids)),
        "exited_window_comment_count": int(exited_window_comment_count),
        "active_video_count": active_video_count,
        "unique_author_count": unique_author_count,
        "reply_count": reply_count,
        "reply_ratio": reply_ratio,
        "emoji_density": numeric_signals["emoji_density"],
        "exclaim_density": numeric_signals["exclaim_density"],
        "question_density": numeric_signals["question_density"],
        "caps_ratio_mean": numeric_signals["caps_ratio_mean"],
        "sentiment_mean": sentiment_mean,
        "sentiment_std": sentiment_std,
        "delta_active_window_comment_count": None,
        "pct_change_active_window_comment_count": None,
        "comment_ids_hash": _hash_ids(active_comment_ids),
        "new_comment_ids_hash": _hash_ids(new_comment_ids),
        "join_status": join_report["join_status"],
        "temporal_status": join_report["temporal_status"],
        "schema_status": gold_schema["schema_status"],
    }
    return row, unavailable


def _add_deltas(signal_rows: list[dict[str, Any]]) -> None:
    previous_count: int | None = None
    for row in signal_rows:
        current = int(row["active_window_comment_count"])
        if previous_count is None:
            row["delta_active_window_comment_count"] = None
            row["pct_change_active_window_comment_count"] = None
        else:
            row["delta_active_window_comment_count"] = int(current - previous_count)
            row["pct_change_active_window_comment_count"] = (
                float((current - previous_count) / previous_count)
                if previous_count
                else None
            )
        previous_count = current


def _build_xiao_input(
    *,
    signal_row: dict[str, Any],
    xiao_signal_name: str,
) -> dict[str, Any]:
    if xiao_signal_name not in signal_row:
        raise ValueError(f"xiao_signal_name is not available in signal row: {xiao_signal_name}")
    xiao_signal_value = signal_row[xiao_signal_name]
    if xiao_signal_value is None:
        raise ValueError(f"xiao_signal_name has null values and cannot feed XIAO: {xiao_signal_name}")
    if xiao_signal_name == "active_window_comment_count":
        delta_signal_value = signal_row["delta_active_window_comment_count"]
        pct_change_signal_value = signal_row["pct_change_active_window_comment_count"]
    else:
        delta_signal_value = signal_row.get(f"delta_{xiao_signal_name}")
        pct_change_signal_value = signal_row.get(f"pct_change_{xiao_signal_name}")
    return {
        "simulation_run_id": signal_row["simulation_run_id"],
        "cycle_id": signal_row["cycle_id"],
        "cycle_index": signal_row["cycle_index"],
        "signal_date": signal_row["signal_date"],
        "observation_time_utc": signal_row["observation_time_utc"],
        "analysis_window_start_utc": signal_row["analysis_window_start_utc"],
        "analysis_window_end_utc": signal_row["analysis_window_end_utc"],
        "data_cutoff_utc": signal_row["data_cutoff_utc"],
        "xiao_signal_name": xiao_signal_name,
        "xiao_signal_value": xiao_signal_value,
        "delta_signal_value": delta_signal_value,
        "pct_change_signal_value": pct_change_signal_value,
        "support_comment_count": signal_row["active_window_comment_count"],
        "active_video_count": signal_row["active_video_count"],
        "comment_ids_hash": signal_row["comment_ids_hash"],
        "join_status": signal_row["join_status"],
        "temporal_status": signal_row["temporal_status"],
        "schema_status": signal_row["schema_status"],
    }


def run_cyclic_daily_signals(config: CyclicDailySignalConfig) -> dict[str, Any]:
    config.validate_c5_scope()
    simulation_dir = config.simulation_path()
    output_dir = config.output_path()
    adapter_manifest = _read_json(simulation_dir / "cycle_adapter_manifest.json")
    cycle_manifest = _read_jsonl(simulation_dir / "cycle_manifest.jsonl")
    monitoring_inputs = _read_jsonl(simulation_dir / "cycle_monitoring_inputs.jsonl")
    window_inventory = _read_csv(simulation_dir / "cycle_window_inventory.csv")
    input_errors = _validate_input_artifacts(
        adapter_manifest=adapter_manifest,
        cycle_manifest=cycle_manifest,
        monitoring_inputs=monitoring_inputs,
        window_inventory=window_inventory,
    )
    if input_errors:
        raise ValueError("C-5 input validation failed: " + "; ".join(input_errors))

    gold, gold_schema = _load_and_validate_gold(config)
    gold_by_comment_id = gold.set_index("comment_id")
    selected_cycles = _ordered_cycles(cycle_manifest, config.max_cycles)

    signal_rows: list[dict[str, Any]] = []
    join_reports: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    unavailable_by_cycle: dict[str, list[dict[str, str]]] = {}

    for cycle in selected_cycles:
        cycle_id = str(cycle["cycle_id"])
        active_rows = _active_rows_for_cycle(window_inventory, cycle_id)
        view, join_report = _resolve_active_view(
            cycle=cycle,
            active_rows=active_rows,
            gold_by_comment_id=gold_by_comment_id,
            gold_schema=gold_schema,
        )
        if join_report["join_status"] != "passed":
            raise ValueError(f"C-5 join/temporal validation failed for cycle_id={cycle_id}")
        exited_count = _exited_count_for_cycle(window_inventory, cycle_id)
        signal_row, unavailable = _build_signal_row(
            cycle=cycle,
            active_rows=active_rows,
            view=view,
            exited_window_comment_count=exited_count,
            join_report=join_report,
            gold_schema=gold_schema,
        )
        unavailable_by_cycle[cycle_id] = unavailable
        signal_rows.append(signal_row)
        join_reports.append(join_report)

    _add_deltas(signal_rows)

    xiao_inputs = [
        _build_xiao_input(signal_row=row, xiao_signal_name=config.xiao_signal_name)
        for row in signal_rows
    ]
    for row in signal_rows:
        unavailable = unavailable_by_cycle[row["cycle_id"]]
        quality_status = (
            "passed"
            if not unavailable
            else "passed_with_unavailable_optional_signals"
        )
        quality_rows.append(
            {
                "simulation_run_id": row["simulation_run_id"],
                "cycle_id": row["cycle_id"],
                "cycle_index": row["cycle_index"],
                "quality_status": quality_status,
                "join_status": row["join_status"],
                "temporal_status": row["temporal_status"],
                "schema_status": row["schema_status"],
                "unavailable_signals": unavailable,
                "double_count_policy": {
                    "individual_comments_sent_to_xiao": False,
                    "one_observation_per_cycle": True,
                    "overlapping_comments_allowed_in_active_windows": True,
                    "windows_are_not_summed_together": True,
                },
                "xiao_execution_status": "not_executed",
                "rag_execution_status": "not_executed",
            }
        )

    simulation_run_id = str(adapter_manifest["simulation_run_id"])
    unavailable_signal_names = sorted(
        {
            item["signal"]
            for unavailable in unavailable_by_cycle.values()
            for item in unavailable
        }
    )
    manifest = {
        "simulation_run_id": simulation_run_id,
        "stage": "C-5",
        "stage_name": "Adaptador de señales diarias para XIAO",
        "mode": SIGNALS_DRY_RUN_MODE,
        "status": "signals_prepared_not_detected",
        "processed_cycle_count": len(signal_rows),
        "canonical_dataset": gold_schema,
        "xiao_signal_name": config.xiao_signal_name,
        "analysis_window_policy": {
            "unit": "daily_cycle",
            "window_kind": "moving_daily_window",
            "initial_window_size_days": selected_cycles[0].get("analysis_window_size_days")
            if selected_cycles
            else None,
            "interval_policy": INTERVAL_POLICY,
        },
        "execution_guards": {
            "run_xiao": False,
            "run_detection": False,
            "run_rag": False,
            "llm_calls": 0,
            "serper_calls": 0,
            "embeddings": False,
            "vectorstore": False,
        },
        "double_count_policy": {
            "individual_comments_sent_to_xiao": False,
            "one_observation_per_cycle": True,
            "overlapping_comments_allowed_in_active_windows": True,
            "windows_are_not_summed_together": True,
            "audit_fields": ["comment_ids_hash", "new_comment_ids_hash"],
        },
        "unavailable_signal_names": unavailable_signal_names,
        "output_artifacts": {
            "cycle_signal_manifest": "cycle_signal_manifest.json",
            "cycle_signal_join_report": "cycle_signal_join_report.jsonl",
            "cycle_signal_series": "cycle_signal_series.jsonl",
            "cycle_signal_quality_report": "cycle_signal_quality_report.jsonl",
            "cycle_xiao_inputs": "cycle_xiao_inputs.jsonl",
        },
        "future_artifacts_not_generated": [
            "cycle_xiao_state.json",
            "cycle_daily_events.jsonl",
        ],
        "compatibility": {
            "retrospective_replay_status": "untouched",
            "xiao_internal_logic_status": "untouched",
            "xiao_thresholds_status": "untouched",
            "rag_status": "not_executed_c5",
            "bronze_silver_gold_status": "read_only_gold_not_modified",
        },
    }

    _write_json(output_dir / "cycle_signal_manifest.json", manifest)
    _write_jsonl(output_dir / "cycle_signal_join_report.jsonl", join_reports)
    _write_jsonl(output_dir / "cycle_signal_series.jsonl", signal_rows)
    _write_jsonl(output_dir / "cycle_signal_quality_report.jsonl", quality_rows)
    _write_jsonl(output_dir / "cycle_xiao_inputs.jsonl", xiao_inputs)

    failed_quality_count = sum(
        1 for row in quality_rows if not str(row["quality_status"]).startswith("passed")
    )
    return {
        "simulation_run_id": simulation_run_id,
        "simulation_dir": str(simulation_dir),
        "output_dir": str(output_dir),
        "mode": SIGNALS_DRY_RUN_MODE,
        "processed_cycle_count": len(signal_rows),
        "failed_quality_count": failed_quality_count,
        "xiao_execution_status": "not_executed",
        "rag_execution_status": "not_executed",
        "xiao_signal_name": config.xiao_signal_name,
        "unavailable_signal_names": unavailable_signal_names,
        "artifacts": {
            "cycle_signal_manifest": str(output_dir / "cycle_signal_manifest.json"),
            "cycle_signal_join_report": str(output_dir / "cycle_signal_join_report.jsonl"),
            "cycle_signal_series": str(output_dir / "cycle_signal_series.jsonl"),
            "cycle_signal_quality_report": str(output_dir / "cycle_signal_quality_report.jsonl"),
            "cycle_xiao_inputs": str(output_dir / "cycle_xiao_inputs.jsonl"),
        },
    }


__all__ = [
    "CyclicDailySignalConfig",
    "run_cyclic_daily_signals",
]
