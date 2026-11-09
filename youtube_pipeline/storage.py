from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class LocalFilesConfig:
    """Input tables and destination for one local-files storage execution."""

    videos_path: str | Path
    comments_path: str | Path
    data_root: str | Path = "data"

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "LocalFilesConfig":
        config_payload = payload.get("local_files", payload)
        if not isinstance(config_payload, dict):
            raise ValueError("Local files config must be an object.")
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(config_payload) - allowed)
        if unknown:
            raise ValueError(f"Unknown local files config fields: {unknown}")
        return cls(**config_payload)


def _read_local_table(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if source.is_dir() or source.suffix.lower() == ".parquet":
        return pd.read_parquet(source)
    if source.suffix.lower() == ".csv":
        return pd.read_csv(source)
    raise ValueError(f"Unsupported file format: {source}")


def _utc_now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _timestamp_error_context(
    dataframe: pd.DataFrame,
    *,
    position: int,
    field: str,
    value: object,
) -> str:
    comment_id = None
    if "comment_id" in dataframe.columns:
        candidate = dataframe.iloc[position]["comment_id"]
        if not pd.isna(candidate):
            comment_id = str(candidate)
    rendered_id = repr(comment_id) if comment_id is not None else "<unknown>"
    return f"comment_id={rendered_id}, field={field!r}, value={value!r}"


def _normalize_timestamp_column(
    dataframe: pd.DataFrame,
    *,
    source_col: str,
    raw_col: str | None = None,
    require_explicit_timezone: bool = False,
) -> pd.Series:
    if source_col not in dataframe.columns:
        raise KeyError(f"Column '{source_col}' not found in dataframe.")
    if raw_col is not None:
        dataframe[raw_col] = dataframe[source_col].astype(str)
    if not require_explicit_timezone:
        timestamps = pd.to_datetime(dataframe[source_col], utc=True, errors="coerce")
        if timestamps.isna().any():
            bad_rows = int(timestamps.isna().sum())
            raise ValueError(
                f"{bad_rows} rows have invalid timestamps in column '{source_col}'."
            )
        return timestamps

    normalized: list[pd.Timestamp] = []
    for position, value in enumerate(dataframe[source_col].tolist()):
        context = _timestamp_error_context(
            dataframe,
            position=position,
            field=source_col,
            value=value,
        )
        try:
            missing = bool(pd.isna(value))
        except (TypeError, ValueError):
            missing = False
        if missing:
            raise ValueError(f"Comment timestamp is missing: {context}.")
        try:
            timestamp = pd.Timestamp(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"Comment timestamp is invalid: {context}.") from exc
        if timestamp.tzinfo is None:
            raise ValueError(
                "Comment timestamp must include an explicit timezone or UTC offset: "
                f"{context}."
            )
        normalized.append(timestamp.tz_convert("UTC"))
    return pd.Series(normalized, index=dataframe.index, dtype="datetime64[ns, UTC]")


def canonicalize_comment_event_time(
    comments_df: pd.DataFrame,
) -> pd.DataFrame:
    """Return comments with one validated, timezone-aware UTC event-time field.

    An existing ``event_time_utc`` column is authoritative.  Otherwise the source
    ``published_at`` column is adapted in memory.  Source columns and row order are
    preserved; this adapter adds no derived calendar, Unix-time, or text fields.
    """

    if not isinstance(comments_df, pd.DataFrame):
        raise TypeError("comments_df must be a pandas DataFrame.")
    if "event_time_utc" in comments_df.columns:
        source_field = "event_time_utc"
    elif "published_at" in comments_df.columns:
        source_field = "published_at"
    else:
        raise ValueError(
            "Comment input requires 'event_time_utc' or source field 'published_at'."
        )

    out = comments_df.copy()
    out["event_time_utc"] = _normalize_timestamp_column(
        out,
        source_col=source_field,
        require_explicit_timezone=True,
    )
    return out


def _unix_seconds(ts: pd.Series) -> pd.Series:
    return ts.map(lambda value: int(value.timestamp())).astype("int64")


def normalize_video_timestamps(
    videos_df: pd.DataFrame, source_col: str = "publishedAt"
) -> pd.DataFrame:
    """Return a copy with stable UTC time fields for simulation."""
    out = videos_df.copy()
    out["published_at_utc"] = _normalize_timestamp_column(
        out, source_col=source_col, raw_col="published_at_raw"
    )
    out["published_at_unix_s"] = _unix_seconds(out["published_at_utc"])
    # Legacy alias retained for compatibility; values are Unix seconds.
    out["published_at_unix_ms"] = out["published_at_unix_s"]
    out["published_date"] = out["published_at_utc"].dt.strftime("%Y-%m-%d")
    return out


def normalize_comment_timestamps(
    comments_df: pd.DataFrame, source_col: str = "published_at"
) -> pd.DataFrame:
    """Return a copy with stable UTC fields used by offline and Streamz phases."""
    out = comments_df.copy()
    out["event_time_utc"] = _normalize_timestamp_column(
        out, source_col=source_col, raw_col="published_at_raw"
    )
    out["event_time_unix_s"] = _unix_seconds(out["event_time_utc"])
    # Legacy alias retained for compatibility; values are Unix seconds.
    out["event_time_unix_ms"] = out["event_time_unix_s"]
    out["event_date"] = out["event_time_utc"].dt.strftime("%Y-%m-%d")
    out["event_year"] = out["event_time_utc"].dt.year.astype("int16")
    out["event_month"] = out["event_time_utc"].dt.month.astype("int8")
    out["event_day"] = out["event_time_utc"].dt.day.astype("int8")
    return out


def write_jsonl(df: pd.DataFrame, output_path: str | Path) -> Path:
    """Write dataframe rows as JSON Lines preserving UTC timestamp strings."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    records = df.to_dict(orient="records")
    with output.open("w", encoding="utf-8") as f:
        for record in records:
            for key, value in list(record.items()):
                if isinstance(value, pd.Timestamp):
                    record[key] = value.isoformat()
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return output


def write_parquet_dataset(
    df: pd.DataFrame,
    dataset_dir: str | Path,
    partition_cols: tuple[str, ...] = ("event_year", "event_month", "event_day"),
    compression: str = "zstd",
) -> Path:
    """Write a partitioned parquet dataset for fast offline and stream reads."""
    output = Path(dataset_dir)
    output.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(
            output,
            index=False,
            partition_cols=list(partition_cols),
            compression=compression,
        )
    except ImportError as exc:
        raise ImportError(
            "Parquet support requires pyarrow or fastparquet. "
            "Install one of them, for example: pip install pyarrow"
        ) from exc
    return output


def persist_batch_snapshot(
    videos_df: pd.DataFrame,
    comments_df: pd.DataFrame,
    data_root: str | Path = "data",
) -> dict[str, Any]:
    """
    Persist extraction outputs in a bronze/silver layout.

    Layout:
      - data/bronze/videos/videos_<stamp>.jsonl
      - data/bronze/comments/comments_<stamp>.jsonl
      - data/silver/videos/ (parquet)
      - data/silver/comments/ (parquet)
    """
    root = Path(data_root)
    stamp = _utc_now_stamp()

    videos_norm = normalize_video_timestamps(videos_df)
    comments_norm = normalize_comment_timestamps(comments_df)

    videos_jsonl = write_jsonl(videos_norm, root / "bronze" / "videos" / f"videos_{stamp}.jsonl")
    comments_jsonl = write_jsonl(
        comments_norm, root / "bronze" / "comments" / f"comments_{stamp}.jsonl"
    )

    videos_parquet = write_parquet_dataset(
        videos_norm.assign(
            event_year=videos_norm["published_at_utc"].dt.year.astype("int16"),
            event_month=videos_norm["published_at_utc"].dt.month.astype("int8"),
            event_day=videos_norm["published_at_utc"].dt.day.astype("int8"),
        ),
        root / "silver" / "videos",
    )
    comments_parquet = write_parquet_dataset(
        comments_norm, root / "silver" / "comments"
    )

    return {
        "videos_rows": len(videos_norm),
        "comments_rows": len(comments_norm),
        "videos_jsonl": str(videos_jsonl),
        "comments_jsonl": str(comments_jsonl),
        "videos_parquet": str(videos_parquet),
        "comments_parquet": str(comments_parquet),
    }


def persist_local_files(config: LocalFilesConfig) -> dict[str, Any]:
    """Load configured local tables and preserve the existing Bronze/Silver contract."""

    if not isinstance(config, LocalFilesConfig):
        raise TypeError("config must be LocalFilesConfig.")
    videos_df = _read_local_table(config.videos_path)
    comments_df = _read_local_table(config.comments_path)
    return persist_batch_snapshot(
        videos_df,
        comments_df,
        data_root=str(config.data_root),
    )
