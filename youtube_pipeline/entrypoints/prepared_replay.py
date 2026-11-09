from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from youtube_pipeline.configuration import (
    ResolvedRunConfig,
    load_run_config,
    resolve_run_config,
    run_config_from_mapping,
)
from youtube_pipeline.detectors import (
    DEFAULT_DETECTOR,
    XiaoEMAConfig,
    normalize_detector_params,
)
from youtube_pipeline.prepared_replay import (
    DEFAULT_PREPARED_TIMESTAMP_COLUMN,
    DEFAULT_REPLAY_MAX_SLEEP_SECONDS,
    DEFAULT_REPLAY_SPEED,
    DEFAULT_REPLAY_WINDOW_SIZE,
    PreparedDatasetConfig,
    ReplayConfig,
)


LEGACY_INPUT_PATH = "data/gold/clean_comments.parquet"
LEGACY_OUTPUT_SNAPSHOTS = "data/gold/snapshots.csv"
_LEGACY_IDENTITY = "legacy_prepared_replay"
_LEGACY_FIELDS = {
    "input_path",
    "output_snapshots",
    "ts_col",
    "window_size",
    "speed",
    "max_sleep_seconds",
    "start",
    "end",
}


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("Prepared replay config must be a JSON object.")
    return payload


def _read_legacy_detector_config(
    config_file: str | Path | None,
) -> tuple[str | None, dict[str, Any]]:
    if config_file is None:
        return None, {}
    payload = _read_json_object(Path(config_file))
    detector_payload = payload.get("detector", payload)
    if isinstance(detector_payload, str):
        return detector_payload, {}
    if not isinstance(detector_payload, Mapping):
        raise ValueError("Detector config must be an object or detector name string.")

    detector = dict(detector_payload)
    detector_name = (
        detector.get("name")
        or detector.get("detector")
        or detector.get("type")
    )
    raw_params = detector.get("params", {})
    if raw_params is None:
        raw_params = {}
    if not isinstance(raw_params, Mapping):
        raise ValueError("Detector config field 'params' must be an object.")
    inline_params = {
        key: value
        for key, value in detector.items()
        if key not in {"name", "detector", "type", "params"}
    }
    return (
        str(detector_name) if detector_name is not None else None,
        {**inline_params, **dict(raw_params)},
    )


def _legacy_defaults() -> dict[str, Any]:
    return {
        "input_path": LEGACY_INPUT_PATH,
        "output_snapshots": LEGACY_OUTPUT_SNAPSHOTS,
        "ts_col": DEFAULT_PREPARED_TIMESTAMP_COLUMN,
        "window_size": DEFAULT_REPLAY_WINDOW_SIZE,
        "speed": DEFAULT_REPLAY_SPEED,
        "max_sleep_seconds": DEFAULT_REPLAY_MAX_SLEEP_SECONDS,
        "start": None,
        "end": None,
    }


def _legacy_component_payload(
    payload: Mapping[str, Any],
    overrides: Mapping[str, Any] | None,
) -> dict[str, Any]:
    unknown = sorted(set(payload) - _LEGACY_FIELDS)
    if unknown:
        raise ValueError(f"Unknown prepared replay config fields: {unknown}")
    component = {**_legacy_defaults(), **dict(payload)}
    if overrides is not None:
        override_unknown = sorted(set(overrides) - _LEGACY_FIELDS)
        if override_unknown:
            raise ValueError(
                f"Unknown prepared replay override fields: {override_unknown}"
            )
        component.update(dict(overrides))
    return component


def _legacy_run_payload(component: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "identity": {"run_id": _LEGACY_IDENTITY},
        "data": {
            "prepared_dataset": {
                "path": component["input_path"],
                "timestamp_column": component["ts_col"],
            }
        },
        "simulation": {
            "replay": {
                "output_snapshots": component["output_snapshots"],
                "window_size": component["window_size"],
                "speed": component["speed"],
                "max_sleep_seconds": component["max_sleep_seconds"],
                "start": component["start"],
                "end": component["end"],
            }
        },
    }


def _profile_overrides(
    overrides: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if overrides is None:
        return None
    unknown = sorted(set(overrides) - _LEGACY_FIELDS)
    if unknown:
        raise ValueError(f"Unknown prepared replay override fields: {unknown}")

    data: dict[str, Any] = {}
    replay: dict[str, Any] = {}
    if "input_path" in overrides:
        data["path"] = overrides["input_path"]
    if "ts_col" in overrides:
        data["timestamp_column"] = overrides["ts_col"]
    for field_name in (
        "output_snapshots",
        "window_size",
        "speed",
        "max_sleep_seconds",
        "start",
        "end",
    ):
        if field_name in overrides:
            replay[field_name] = overrides[field_name]

    result: dict[str, Any] = {}
    if data:
        result["data"] = {"prepared_dataset": data}
    if replay:
        result["simulation"] = {"replay": replay}
    return result or None


def _select_configs(
    run: Any,
) -> tuple[PreparedDatasetConfig, ReplayConfig]:
    if run.data is None or run.data.prepared_dataset is None:
        raise ValueError("RunConfig must include data.prepared_dataset.")
    if run.simulation is None or run.simulation.replay is None:
        raise ValueError("RunConfig must include simulation.replay.")
    return run.data.prepared_dataset, run.simulation.replay


def load_legacy_prepared_replay_configs(
    config_file: str | Path | None,
    *,
    overrides: Mapping[str, Any] | None = None,
) -> tuple[PreparedDatasetConfig, ReplayConfig]:
    """Load the previous playback shape through the common strict loader."""

    payload: dict[str, Any] = {}
    if config_file is not None:
        payload = _read_json_object(Path(config_file))
    component = _legacy_component_payload(payload, overrides)
    return _select_configs(run_config_from_mapping(_legacy_run_payload(component)))


def resolve_prepared_replay_configs(
    *,
    config_file: str | Path | None,
    overrides: Mapping[str, Any] | None = None,
    base_dir: str | Path,
) -> tuple[PreparedDatasetConfig, ReplayConfig]:
    """Resolve either a RunConfig profile or the legacy playback shape."""

    if config_file is not None:
        path = Path(config_file)
        payload = _read_json_object(path)
        is_run_profile = "identity" in payload or any(
            section in payload
            for section in ("data", "simulation", "signals", "detection", "rag")
        )
    else:
        path = None
        payload = {}
        is_run_profile = False

    if is_run_profile:
        run = load_run_config(path, overrides=_profile_overrides(overrides))
    else:
        component = _legacy_component_payload(payload, overrides)
        run = run_config_from_mapping(_legacy_run_payload(component))

    resolved = resolve_run_config(run, base_dir=base_dir).config
    return _select_configs(resolved)


def resolve_legacy_prepared_replay_run(
    *,
    input_path: str | Path | None = None,
    output_snapshots: str | Path | None = None,
    ts_col: str | None = None,
    window_size: str | None = None,
    speed: float | None = None,
    max_sleep_seconds: float | None = None,
    start: str | None = None,
    end: str | None = None,
    detector_name: str | None = None,
    detector_config_file: str | Path | None = None,
    detector_params: Mapping[str, Any] | None = None,
    trigger_threshold: float | None = None,
    trigger_min_volume: int | None = None,
    trigger_window_size: str | None = None,
    trigger_slide_interval: str | None = None,
    trigger_slow_window: str | None = None,
    trigger_cooldown: str | None = None,
    base_dir: str | Path,
) -> tuple[ResolvedRunConfig, str]:
    """Translate the historical playback API into one resolved RunConfig."""

    replay_overrides = {
        key: value
        for key, value in {
            "input_path": input_path,
            "output_snapshots": output_snapshots,
            "ts_col": ts_col,
            "window_size": window_size,
            "speed": speed,
            "max_sleep_seconds": max_sleep_seconds,
            "start": start,
            "end": end,
        }.items()
        if value is not None
    }
    component = _legacy_component_payload({}, replay_overrides)

    file_detector_name, file_params = _read_legacy_detector_config(
        detector_config_file
    )
    effective_detector_name = (
        detector_name or file_detector_name or DEFAULT_DETECTOR
    )
    effective_params = {
        **file_params,
        **dict(detector_params or {}),
    }
    for legacy_name, config_name, value in (
        ("trigger_window_size", "window_size", trigger_window_size),
        ("trigger_slide_interval", "slide_interval", trigger_slide_interval),
        ("trigger_slow_window", "slow_window", trigger_slow_window),
        ("trigger_threshold", "sensitivity_threshold", trigger_threshold),
        ("trigger_min_volume", "v_min", trigger_min_volume),
        ("trigger_cooldown", "cooldown", trigger_cooldown),
    ):
        if value is not None:
            effective_params[legacy_name] = value
        if legacy_name in effective_params and config_name not in effective_params:
            effective_params[config_name] = effective_params.pop(legacy_name)
    effective_params = normalize_detector_params(effective_params)
    effective_params.setdefault("ts_col", component["ts_col"])

    xiao_fields = set(XiaoEMAConfig.__dataclass_fields__)
    unknown = sorted(set(effective_params) - xiao_fields)
    if effective_detector_name == DEFAULT_DETECTOR and unknown:
        raise ValueError(f"Unknown XIAO EMA config fields: {unknown}")
    xiao_payload = (
        {key: value for key, value in effective_params.items() if key in xiao_fields}
        if effective_detector_name == DEFAULT_DETECTOR
        else {}
    )

    payload = _legacy_run_payload(component)
    payload["detection"] = {"xiao_ema": xiao_payload}
    resolved = resolve_run_config(
        run_config_from_mapping(payload),
        base_dir=base_dir,
    )
    return resolved, effective_detector_name


def legacy_replay_detector_params(
    resolved: ResolvedRunConfig,
    detector_name: str,
) -> dict[str, Any]:
    """Adapt resolved detector config to the unchanged replay component API."""

    if detector_name != DEFAULT_DETECTOR:
        return {}
    detection = resolved.config.detection
    if detection is None or detection.xiao_ema is None:
        raise ValueError("Resolved playback RunConfig must include detection.xiao_ema.")
    xiao = detection.xiao_ema
    return {
        "ts_col": xiao.ts_col,
        "text_col": xiao.text_col,
        "window_size": xiao.window_size,
        "slide_interval": xiao.slide_interval,
        "slow_window": xiao.slow_window,
        "sensitivity_threshold": xiao.sensitivity_threshold,
        "v_min": xiao.v_min,
        "cooldown": xiao.cooldown,
    }


__all__ = [
    "LEGACY_INPUT_PATH",
    "LEGACY_OUTPUT_SNAPSHOTS",
    "legacy_replay_detector_params",
    "load_legacy_prepared_replay_configs",
    "resolve_legacy_prepared_replay_run",
    "resolve_prepared_replay_configs",
]
