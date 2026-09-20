from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from youtube_pipeline.experimental_input import (
    ExperimentalInputPolicyConfig,
    adapt_activity_records,
    prepare_causal_minimal_batch,
)


LOCAL_TIME_FORMAT = "%Y-%m-%dT%H:%M:%S%z"
UTC_TIME_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
DEFAULT_SIMULATION_MODE = "cyclic_ingestion_simulation"
DEFAULT_TIMEZONE = "America/Bogota"
DEFAULT_CANONICAL_TIMEZONE = "UTC"
DEFAULT_ANALYSIS_WINDOW_SIZE_DAYS = 3
DEFAULT_RAG_MODE = "sidecars_only"
INTERVAL_POLICY = "semi_open_daily_bounds_start_inclusive_end_exclusive"


def _sha1_short(value: str, length: int = 12) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:length]


def _format_local(ts: pd.Timestamp) -> str:
    return ts.strftime(LOCAL_TIME_FORMAT)


def _format_utc(ts: pd.Timestamp) -> str:
    return ts.tz_convert("UTC").strftime(UTC_TIME_FORMAT)


def _parse_local_date(value: str | date | None) -> date | None:
    if value is None:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    return pd.Timestamp(value).date()


def _date_range(start: date, end: date) -> list[date]:
    if end < start:
        raise ValueError("collection_end_date_local must be >= collection_start_date_local.")
    days = []
    current = start
    while current <= end:
        days.append(current)
        current += timedelta(days=1)
    return days


def _read_table(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if p.is_dir() or p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    raise ValueError(f"Unsupported dataset format: {p}")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        if value.tzinfo is None:
            value = value.tz_localize("UTC")
        return _format_utc(value)
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


@dataclass
class CyclicIngestionConfig:
    input_path: str | Path
    output_dir: str | Path
    simulation_mode: str = DEFAULT_SIMULATION_MODE
    timezone: str = DEFAULT_TIMEZONE
    canonical_timezone: str = DEFAULT_CANONICAL_TIMEZONE
    analysis_window_size_days: int = DEFAULT_ANALYSIS_WINDOW_SIZE_DAYS
    cycle_frequency: str = "daily"
    rag_mode: str = DEFAULT_RAG_MODE
    ts_col: str = "event_time_utc"
    comment_id_col: str = "comment_id"
    video_id_col: str = "video_id"
    collection_start_date_local: str | date | None = None
    collection_end_date_local: str | date | None = None
    simulation_run_id: str | None = None
    dry_run: bool = True
    notes: str | None = None
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "CyclicIngestionConfig":
        config_payload = payload.get("cyclic_ingestion_simulation", payload)
        if not isinstance(config_payload, dict):
            raise ValueError("Cyclic ingestion config must be an object.")
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(config_payload) - allowed)
        if unknown:
            raise ValueError(f"Unknown cyclic ingestion config fields: {unknown}")
        return cls(**config_payload)

    def validate(self) -> None:
        if self.simulation_mode != DEFAULT_SIMULATION_MODE:
            raise ValueError(
                "C-0/C-1 only supports simulation_mode='cyclic_ingestion_simulation'. "
                "The existing retrospective_replay mode is intentionally untouched."
            )
        if self.canonical_timezone != "UTC":
            raise ValueError("canonical_timezone must be UTC for C-0/C-1.")
        if self.analysis_window_size_days < 1:
            raise ValueError("analysis_window_size_days must be >= 1.")
        if self.cycle_frequency != "daily":
            raise ValueError("C-0/C-1 currently supports only daily cycle_frequency.")
        if self.rag_mode not in {
            "disabled",
            "sidecars_only",
            "g1_only",
            "g2_dry_run",
            "g2_real",
        }:
            raise ValueError(f"Unsupported rag_mode: {self.rag_mode}")
        ZoneInfo(self.timezone)

    def output_path(self) -> Path:
        return Path(self.output_dir)


def _prepare_comments(
    raw_df: pd.DataFrame,
    config: CyclicIngestionConfig,
) -> tuple[pd.DataFrame, int]:
    required = [config.ts_col, config.comment_id_col, config.video_id_col]
    missing = [column for column in required if column not in raw_df.columns]
    if missing:
        raise KeyError(f"Missing required columns for cyclic ingestion: {missing}")

    df = raw_df.copy()
    df["source_row_id"] = range(len(df))
    df[config.ts_col] = pd.to_datetime(df[config.ts_col], utc=True, errors="coerce")
    invalid_timestamp_count = int(df[config.ts_col].isna().sum())
    df = df.loc[~df[config.ts_col].isna()].copy()
    df[config.comment_id_col] = df[config.comment_id_col].astype(str)
    df[config.video_id_col] = df[config.video_id_col].astype(str)
    df = df.sort_values(
        [config.ts_col, config.comment_id_col, config.video_id_col, "source_row_id"]
    ).reset_index(drop=True)
    df["duplicate_occurrence_index"] = df.groupby(config.comment_id_col).cumcount()
    df["is_duplicate"] = df["duplicate_occurrence_index"] > 0
    return df, invalid_timestamp_count


def _infer_collection_dates(
    comments_df: pd.DataFrame,
    config: CyclicIngestionConfig,
    tz: ZoneInfo,
) -> tuple[date, date]:
    configured_start = _parse_local_date(config.collection_start_date_local)
    configured_end = _parse_local_date(config.collection_end_date_local)
    if comments_df.empty and (configured_start is None or configured_end is None):
        raise ValueError(
            "The input dataset has no valid timestamps. Provide local collection dates "
            "explicitly to build empty cycle contracts."
        )

    local_times = comments_df[config.ts_col].dt.tz_convert(tz)
    inferred_start = local_times.min().date() if not comments_df.empty else configured_start
    inferred_end = local_times.max().date() if not comments_df.empty else configured_end
    start = configured_start or inferred_start
    end = configured_end or inferred_end
    if start is None or end is None:
        raise ValueError("Could not infer local collection date range.")
    return start, end


def _local_day_interval(day: date, tz: ZoneInfo) -> tuple[pd.Timestamp, pd.Timestamp]:
    start_dt = datetime.combine(day, time(0, 0, 0), tzinfo=tz)
    end_dt = datetime.combine(day + timedelta(days=1), time(0, 0, 0), tzinfo=tz)
    return pd.Timestamp(start_dt), pd.Timestamp(end_dt)


def _build_cycle_records(
    comments_df: pd.DataFrame,
    config: CyclicIngestionConfig,
    simulation_run_id: str,
) -> list[dict[str, Any]]:
    tz = ZoneInfo(config.timezone)
    start_date, end_date = _infer_collection_dates(comments_df, config, tz)
    cycles: list[dict[str, Any]] = []
    for index, collection_day in enumerate(_date_range(start_date, end_date), start=1):
        collection_start_local, collection_end_local = _local_day_interval(
            collection_day, tz
        )
        analysis_start_day = collection_day - timedelta(
            days=config.analysis_window_size_days - 1
        )
        analysis_start_local, _ = _local_day_interval(analysis_start_day, tz)
        _, analysis_end_local = _local_day_interval(collection_day, tz)
        cycle_run_at_local = pd.Timestamp(
            datetime.combine(collection_day + timedelta(days=1), time(0, 0, 0), tzinfo=tz)
        )
        cycle_run_at_utc = cycle_run_at_local.tz_convert("UTC")
        collection_start_utc = collection_start_local.tz_convert("UTC")
        collection_end_utc = collection_end_local.tz_convert("UTC")
        analysis_start_utc = analysis_start_local.tz_convert("UTC")
        analysis_end_utc = analysis_end_local.tz_convert("UTC")
        data_cutoff_local = collection_end_local
        data_cutoff_utc = collection_end_utc
        cycle_id = "cyc_" + _sha1_short(
            "|".join(
                [
                    simulation_run_id,
                    str(index),
                    _format_utc(cycle_run_at_utc),
                    _format_utc(collection_start_utc),
                    _format_utc(collection_end_utc),
                ]
            )
        )
        cycles.append(
            {
                "simulation_run_id": simulation_run_id,
                "cycle_id": cycle_id,
                "cycle_index": index,
                "cycle_run_at_local": _format_local(cycle_run_at_local),
                "cycle_run_at_utc": _format_utc(cycle_run_at_utc),
                "collection_window_start_local": _format_local(collection_start_local),
                "collection_window_end_local": _format_local(collection_end_local),
                "collection_window_start_utc": _format_utc(collection_start_utc),
                "collection_window_end_utc": _format_utc(collection_end_utc),
                "analysis_window_start_local": _format_local(analysis_start_local),
                "analysis_window_end_local": _format_local(analysis_end_local),
                "analysis_window_start_utc": _format_utc(analysis_start_utc),
                "analysis_window_end_utc": _format_utc(analysis_end_utc),
                "analysis_window_size_days": config.analysis_window_size_days,
                "data_cutoff_local": _format_local(data_cutoff_local),
                "data_cutoff_utc": _format_utc(data_cutoff_utc),
                "timezone": config.timezone,
                "canonical_timezone": config.canonical_timezone,
                "simulation_mode": config.simulation_mode,
                "rag_mode": config.rag_mode,
                "_collection_start_utc_ts": collection_start_utc,
                "_collection_end_utc_ts": collection_end_utc,
                "_analysis_start_utc_ts": analysis_start_utc,
                "_analysis_end_utc_ts": analysis_end_utc,
                "_data_cutoff_utc_ts": data_cutoff_utc,
            }
        )
    return cycles


def _deterministic_simulation_run_id(
    comments_df: pd.DataFrame,
    config: CyclicIngestionConfig,
) -> str:
    if config.simulation_run_id:
        return config.simulation_run_id
    if comments_df.empty:
        coverage = "empty"
    else:
        coverage = (
            f"{_format_utc(comments_df[config.ts_col].min())}|"
            f"{_format_utc(comments_df[config.ts_col].max())}|{len(comments_df)}"
        )
    raw = "|".join(
        [
            str(Path(config.input_path)),
            config.simulation_mode,
            config.timezone,
            config.canonical_timezone,
            str(config.analysis_window_size_days),
            str(config.collection_start_date_local),
            str(config.collection_end_date_local),
            INTERVAL_POLICY,
            coverage,
        ]
    )
    return "sim_" + _sha1_short(raw)


def _public_cycle_record(record: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if not key.startswith("_")}


def _assign_first_seen_cycles(
    comments_df: pd.DataFrame,
    cycles: list[dict[str, Any]],
    config: CyclicIngestionConfig,
) -> pd.DataFrame:
    if comments_df.empty:
        return comments_df.copy()
    assigned = comments_df.copy()
    assigned["assigned_cycle_id"] = pd.NA
    assigned["assigned_cycle_index"] = pd.NA
    assigned["event_time_local"] = assigned[config.ts_col].dt.tz_convert(config.timezone)
    assigned["event_date_local"] = assigned["event_time_local"].dt.date.astype(str)

    for cycle in cycles:
        mask = (
            (assigned[config.ts_col] >= cycle["_collection_start_utc_ts"])
            & (assigned[config.ts_col] < cycle["_collection_end_utc_ts"])
        )
        assigned.loc[mask, "assigned_cycle_id"] = cycle["cycle_id"]
        assigned.loc[mask, "assigned_cycle_index"] = cycle["cycle_index"]

    first_seen = (
        assigned.loc[~assigned["is_duplicate"]]
        .groupby(config.comment_id_col)[["assigned_cycle_id", "assigned_cycle_index"]]
        .first()
        .rename(
            columns={
                "assigned_cycle_id": "first_seen_cycle_id",
                "assigned_cycle_index": "first_seen_cycle_index",
            }
        )
    )
    assigned = assigned.join(first_seen, on=config.comment_id_col)
    assigned["is_new_in_cycle"] = (
        ~assigned["is_duplicate"]
        & assigned["assigned_cycle_id"].notna()
        & (assigned["assigned_cycle_id"] == assigned["first_seen_cycle_id"])
    )
    assigned["is_late_arrival"] = False
    assigned["late_arrival_status"] = "not_inferable_missing_ingestion_timestamp"
    return assigned


def _build_cycle_manifests_and_processed_inventory(
    comments_df: pd.DataFrame,
    cycles: list[dict[str, Any]],
    config: CyclicIngestionConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    canonical = comments_df.loc[~comments_df["is_duplicate"]].copy()
    processed_rows: list[pd.DataFrame] = []
    manifest_rows: list[dict[str, Any]] = []

    for cycle in cycles:
        new_mask = canonical["assigned_cycle_id"] == cycle["cycle_id"]
        cumulative_mask = canonical[config.ts_col] < cycle["_data_cutoff_utc_ts"]
        analysis_mask = (
            cumulative_mask
            & (canonical[config.ts_col] >= cycle["_analysis_start_utc_ts"])
            & (canonical[config.ts_col] < cycle["_analysis_end_utc_ts"])
        )
        duplicate_mask = (
            comments_df["is_duplicate"]
            & (comments_df["assigned_cycle_id"] == cycle["cycle_id"])
        )
        processed = canonical.loc[analysis_mask].copy()
        future_leak_count = int(
            (processed[config.ts_col] >= cycle["_data_cutoff_utc_ts"]).sum()
        )
        if not processed.empty:
            processed["simulation_run_id"] = cycle["simulation_run_id"]
            processed["cycle_id"] = cycle["cycle_id"]
            processed["cycle_index"] = cycle["cycle_index"]
            processed["analysis_window_start_utc"] = cycle["analysis_window_start_utc"]
            processed["analysis_window_end_utc"] = cycle["analysis_window_end_utc"]
            processed["data_cutoff_utc"] = cycle["data_cutoff_utc"]
            processed["included_in_analysis"] = True
            processed_rows.append(processed)

        row = _public_cycle_record(cycle)
        row.update(
            {
                "new_comment_count": int(new_mask.sum()),
                "duplicate_row_count": int(duplicate_mask.sum()),
                "cumulative_comment_count": int(cumulative_mask.sum()),
                "analysis_comment_count": int(analysis_mask.sum()),
                "analysis_video_count": int(
                    canonical.loc[analysis_mask, config.video_id_col].nunique()
                ),
                "late_comment_count": 0,
                "future_leak_count": future_leak_count,
                "cycle_status": "dry_run_partitioned",
            }
        )
        manifest_rows.append(row)

    if processed_rows:
        processed_inventory = pd.concat(processed_rows, ignore_index=True)
    else:
        processed_inventory = pd.DataFrame()
    return pd.DataFrame(manifest_rows), processed_inventory


def _input_inventory_columns(
    config: CyclicIngestionConfig, *, include_policy_lineage: bool = False
) -> list[str]:
    columns = [
        "simulation_run_id",
        "source_row_id",
        config.comment_id_col,
        config.video_id_col,
        "author_id",
        config.ts_col,
        "event_time_local",
        "event_date_local",
        "assigned_cycle_id",
        "assigned_cycle_index",
        "first_seen_cycle_id",
        "first_seen_cycle_index",
        "is_new_in_cycle",
        "is_duplicate",
        "duplicate_occurrence_index",
        "is_late_arrival",
        "late_arrival_status",
    ]
    if include_policy_lineage:
        columns.extend(
            [
                "input_policy_id",
                "input_source_layer",
                "input_source_path",
                "input_source_sha256",
                "input_source_row_index",
                "input_quality_flags",
            ]
        )
    return columns


def _processed_inventory_columns(
    config: CyclicIngestionConfig, *, include_policy_lineage: bool = False
) -> list[str]:
    columns = [
        "simulation_run_id",
        "cycle_id",
        "cycle_index",
        config.comment_id_col,
        config.video_id_col,
        "author_id",
        config.ts_col,
        "event_time_local",
        "first_seen_cycle_id",
        "analysis_window_start_utc",
        "analysis_window_end_utc",
        "data_cutoff_utc",
        "included_in_analysis",
        "is_duplicate",
        "is_late_arrival",
    ]
    if include_policy_lineage:
        columns.extend(
            [
                "input_policy_id",
                "input_source_layer",
                "input_source_path",
                "input_source_sha256",
                "input_source_row_index",
                "input_quality_flags",
            ]
        )
    return columns


def _normalize_timestamp_columns(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    for column in output.columns:
        if pd.api.types.is_datetime64_any_dtype(output[column]):
            if getattr(output[column].dt, "tz", None) is None:
                output[column] = output[column].dt.tz_localize("UTC")
            output[column] = output[column].map(
                lambda value: value.isoformat() if pd.notna(value) else ""
            )
    return output


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_csv(path: Path, df: pd.DataFrame, columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output = df.copy()
    if columns is not None:
        for column in columns:
            if column not in output.columns:
                output[column] = pd.NA
        output = output[columns]
    output = _normalize_timestamp_columns(output)
    output.to_csv(path, index=False)


def _quality_report_rows(
    *,
    simulation_run_id: str,
    invalid_timestamp_count: int,
    duplicate_row_count: int,
    future_leak_count: int,
    total_rows: int,
    valid_rows: int,
) -> list[dict[str, Any]]:
    return [
        {
            "simulation_run_id": simulation_run_id,
            "check_name": "missing_or_invalid_event_time_utc",
            "status": "passed" if invalid_timestamp_count == 0 else "warning",
            "count": invalid_timestamp_count,
            "message": "Rows without valid event_time_utc are excluded from cycle partitioning.",
        },
        {
            "simulation_run_id": simulation_run_id,
            "check_name": "duplicate_comment_id",
            "status": "passed" if duplicate_row_count == 0 else "warning",
            "count": duplicate_row_count,
            "message": "Duplicate rows are not counted as new comments after first occurrence.",
        },
        {
            "simulation_run_id": simulation_run_id,
            "check_name": "future_leakage",
            "status": "passed" if future_leak_count == 0 else "failed",
            "count": future_leak_count,
            "message": "No processed comment may have event_time_utc >= data_cutoff_utc.",
        },
        {
            "simulation_run_id": simulation_run_id,
            "check_name": "late_arrival_inference",
            "status": "not_evaluated",
            "count": 0,
            "message": (
                "Late arrivals cannot be inferred because the corpus has no separate "
                "API ingestion timestamp."
            ),
        },
        {
            "simulation_run_id": simulation_run_id,
            "check_name": "partition_input_rows",
            "status": "info",
            "total_rows": total_rows,
            "valid_timestamp_rows": valid_rows,
        },
    ]


def _build_manifest(
    *,
    config: CyclicIngestionConfig,
    simulation_run_id: str,
    cycles_df: pd.DataFrame,
    input_rows: int,
    valid_rows: int,
    unique_comment_count: int,
    duplicate_row_count: int,
    invalid_timestamp_count: int,
    input_policy_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = {
        "simulation_run_id": simulation_run_id,
        "simulation_mode": config.simulation_mode,
        "execution_mode": "dry_run" if config.dry_run else "materialized_partition",
        "timezone": config.timezone,
        "canonical_timezone": config.canonical_timezone,
        "cycle_frequency": config.cycle_frequency,
        "analysis_window_size_days": config.analysis_window_size_days,
        "rag_mode": config.rag_mode,
        "rag_execution_status": "not_executed_c0_c1",
        "detector_execution_status": "not_executed_c0_c1",
        "monitoring_execution_status": "not_executed_c0_c1",
        "interval_policy": INTERVAL_POLICY,
        "temporal_policy": {
            "canonical_filter_field": config.ts_col,
            "canonical_timezone": "UTC",
            "local_cycle_timezone": config.timezone,
            "no_future_leakage_rule": "event_time_utc < data_cutoff_utc",
            "collection_window_rule": (
                "collection_window_start_utc <= event_time_utc < "
                "collection_window_end_utc"
            ),
            "analysis_window_rule": (
                "analysis_window_start_utc <= event_time_utc < analysis_window_end_utc"
            ),
            "filtering_uses_utc": True,
        },
        "input": {
            "path": str(config.input_path),
            "rows": input_rows,
            "valid_timestamp_rows": valid_rows,
            "unique_comment_count": unique_comment_count,
            "duplicate_row_count": duplicate_row_count,
            "invalid_timestamp_count": invalid_timestamp_count,
        },
        "cycles": {
            "count": int(len(cycles_df)),
            "first_cycle_id": None if cycles_df.empty else cycles_df.iloc[0]["cycle_id"],
            "last_cycle_id": None if cycles_df.empty else cycles_df.iloc[-1]["cycle_id"],
            "total_new_comments": int(cycles_df["new_comment_count"].sum())
            if "new_comment_count" in cycles_df
            else 0,
            "max_analysis_comment_count": int(cycles_df["analysis_comment_count"].max())
            if "analysis_comment_count" in cycles_df and not cycles_df.empty
            else 0,
        },
        "contracts": {
            "cycle_manifest": "one record per cycle",
            "cycle_input_inventory": "one record per source comment row with first-seen cycle",
            "cycle_processed_inventory": (
                "one record per canonical comment included in a cycle analysis window"
            ),
            "cycle_quality_report": "JSONL quality checks for C-0/C-1 partitioning",
            "cycle_state": "checkpoint summary for the dry-run partition",
        },
        "compatibility": {
            "retrospective_replay_status": "untouched",
            "event_id_formula_status": "unchanged",
            "detector_logic_status": "unchanged",
            "rag_sidecars_status": "unchanged",
        },
        "notes": config.notes,
        "params": config.params,
    }
    if input_policy_manifest is not None:
        manifest["experimental_input_policy"] = input_policy_manifest
    return manifest


def _build_state(
    *,
    config: CyclicIngestionConfig,
    simulation_run_id: str,
    cycles_df: pd.DataFrame,
    unique_comment_ids: list[str],
    duplicate_row_count: int,
    invalid_timestamp_count: int,
) -> dict[str, Any]:
    joined_ids = "\n".join(sorted(unique_comment_ids))
    return {
        "simulation_run_id": simulation_run_id,
        "simulation_mode": config.simulation_mode,
        "last_completed_cycle_id": None if cycles_df.empty else cycles_df.iloc[-1]["cycle_id"],
        "cycles_total": int(len(cycles_df)),
        "seen_comment_count": len(unique_comment_ids),
        "seen_comment_ids_hash": "sha1_" + hashlib.sha1(joined_ids.encode("utf-8")).hexdigest(),
        "duplicate_row_count": duplicate_row_count,
        "invalid_timestamp_count": invalid_timestamp_count,
        "detector_state_policy": "not_applicable_c0_c1",
        "rag_mode": config.rag_mode,
        "rag_state": "not_executed_c0_c1",
        "late_arrival_policy": "not_inferable_missing_ingestion_timestamp",
    }


def build_cyclic_ingestion_dry_run(
    config: CyclicIngestionConfig,
    *,
    input_policy: ExperimentalInputPolicyConfig | None = None,
) -> dict[str, Any]:
    config.validate()
    raw_df = _read_table(config.input_path)
    source_input_rows = len(raw_df)
    input_policy_manifest = None
    if input_policy is not None:
        prepared_batch = prepare_causal_minimal_batch(
            raw_df,
            input_policy,
            source_path=config.input_path,
        )
        activity_records = adapt_activity_records(prepared_batch.records)
        raw_df = (
            pd.DataFrame([record.to_mapping() for record in activity_records])
            if activity_records
            else prepared_batch.dataframe
        )
        input_policy_manifest = prepared_batch.manifest
    prepared_df, invalid_timestamp_count = _prepare_comments(raw_df, config)
    simulation_run_id = _deterministic_simulation_run_id(prepared_df, config)
    cycles = _build_cycle_records(prepared_df, config, simulation_run_id)
    assigned_df = _assign_first_seen_cycles(prepared_df, cycles, config)
    assigned_df["simulation_run_id"] = simulation_run_id
    cycles_df, processed_df = _build_cycle_manifests_and_processed_inventory(
        assigned_df, cycles, config
    )
    duplicate_row_count = int(assigned_df["is_duplicate"].sum()) if not assigned_df.empty else 0
    if input_policy_manifest is not None:
        duplicate_row_count += int(
            input_policy_manifest["exact_duplicates_suppressed"]
        )
    future_leak_count = (
        int(cycles_df["future_leak_count"].sum())
        if "future_leak_count" in cycles_df and not cycles_df.empty
        else 0
    )
    unique_comment_ids = (
        assigned_df.loc[~assigned_df["is_duplicate"], config.comment_id_col]
        .dropna()
        .astype(str)
        .tolist()
    )

    output_dir = config.output_path()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = _build_manifest(
        config=config,
        simulation_run_id=simulation_run_id,
        cycles_df=cycles_df,
        input_rows=source_input_rows,
        valid_rows=len(prepared_df),
        unique_comment_count=len(unique_comment_ids),
        duplicate_row_count=duplicate_row_count,
        invalid_timestamp_count=invalid_timestamp_count,
        input_policy_manifest=input_policy_manifest,
    )
    state = _build_state(
        config=config,
        simulation_run_id=simulation_run_id,
        cycles_df=cycles_df,
        unique_comment_ids=unique_comment_ids,
        duplicate_row_count=duplicate_row_count,
        invalid_timestamp_count=invalid_timestamp_count,
    )
    quality_rows = _quality_report_rows(
        simulation_run_id=simulation_run_id,
        invalid_timestamp_count=invalid_timestamp_count,
        duplicate_row_count=duplicate_row_count,
        future_leak_count=future_leak_count,
        total_rows=source_input_rows,
        valid_rows=len(prepared_df),
    )

    _write_json(output_dir / "online_simulation_manifest.json", manifest)
    _write_json(output_dir / "cycle_state.json", state)
    _write_jsonl(output_dir / "cycle_quality_report.jsonl", quality_rows)
    _write_jsonl(
        output_dir / "cycle_manifest.jsonl",
        cycles_df.to_dict(orient="records") if not cycles_df.empty else [],
    )
    _write_csv(
        output_dir / "cycle_input_inventory.csv",
        assigned_df,
        columns=_input_inventory_columns(
            config, include_policy_lineage=input_policy_manifest is not None
        ),
    )
    _write_csv(
        output_dir / "cycle_processed_inventory.csv",
        processed_df,
        columns=_processed_inventory_columns(
            config, include_policy_lineage=input_policy_manifest is not None
        ),
    )

    return {
        "simulation_run_id": simulation_run_id,
        "output_dir": str(output_dir),
        "cycles_total": int(len(cycles_df)),
        "input_rows": int(source_input_rows),
        "valid_timestamp_rows": int(len(prepared_df)),
        "unique_comment_count": int(len(unique_comment_ids)),
        "duplicate_row_count": duplicate_row_count,
        "invalid_timestamp_count": invalid_timestamp_count,
        "future_leak_count": future_leak_count,
        "experimental_input_policy": input_policy_manifest,
        "total_new_comments": int(cycles_df["new_comment_count"].sum())
        if "new_comment_count" in cycles_df
        else 0,
        "max_analysis_comment_count": int(cycles_df["analysis_comment_count"].max())
        if "analysis_comment_count" in cycles_df and not cycles_df.empty
        else 0,
        "artifacts": {
            "online_simulation_manifest": str(output_dir / "online_simulation_manifest.json"),
            "cycle_manifest": str(output_dir / "cycle_manifest.jsonl"),
            "cycle_input_inventory": str(output_dir / "cycle_input_inventory.csv"),
            "cycle_processed_inventory": str(output_dir / "cycle_processed_inventory.csv"),
            "cycle_quality_report": str(output_dir / "cycle_quality_report.jsonl"),
            "cycle_state": str(output_dir / "cycle_state.json"),
        },
    }


__all__ = [
    "CyclicIngestionConfig",
    "build_cyclic_ingestion_dry_run",
]
