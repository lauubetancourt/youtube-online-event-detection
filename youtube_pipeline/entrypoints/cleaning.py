from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from youtube_pipeline.cleaning import CleaningConfig
from youtube_pipeline.configuration import (
    load_run_config,
    resolve_run_config,
    run_config_from_mapping,
)


LEGACY_INPUT_PATH = "data/silver/comments"
LEGACY_OUTPUT_PATH = "data/gold/clean_comments.parquet"
_LEGACY_IDENTITY = "legacy_cleaning"


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("Cleaning config must be a JSON object.")
    return payload


def _legacy_component_payload(
    payload: Mapping[str, Any],
    overrides: Mapping[str, Any] | None,
) -> dict[str, Any]:
    component = {
        "input_path": LEGACY_INPUT_PATH,
        "output_path": LEGACY_OUTPUT_PATH,
        **dict(payload),
    }
    if overrides is not None:
        component.update(
            {
                key: value
                for key, value in overrides.items()
                if value is not None
            }
        )
    return component


def _legacy_run_payload(component_payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "identity": {"run_id": _LEGACY_IDENTITY},
        "data": {"cleaning": dict(component_payload)},
    }


def _component_overrides(
    overrides: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if overrides is None:
        return None
    return {"data": {"cleaning": dict(overrides)}}


def load_legacy_cleaning_config(
    config_file: str | Path | None,
    *,
    overrides: Mapping[str, Any] | None = None,
) -> CleaningConfig:
    """Load the previous cleaning shape through the common strict loader."""

    payload: dict[str, Any] = {}
    if config_file is not None:
        payload = _read_json_object(Path(config_file))
    component_payload = payload.get("cleaning", payload)
    if not isinstance(component_payload, Mapping):
        raise TypeError("Cleaning config must be a JSON object.")
    run = run_config_from_mapping(
        _legacy_run_payload(
            _legacy_component_payload(component_payload, overrides)
        )
    )
    if run.data is None or run.data.cleaning is None:
        raise ValueError("Legacy config did not resolve cleaning.")
    return run.data.cleaning


def resolve_cleaning_config(
    *,
    config_file: str | Path | None,
    overrides: Mapping[str, Any] | None = None,
    base_dir: str | Path,
) -> CleaningConfig:
    """Resolve either a RunConfig profile or the legacy component format."""

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
        run = load_run_config(
            path,
            overrides=_component_overrides(overrides),
        )
    else:
        component_payload = payload.get("cleaning", payload)
        if not isinstance(component_payload, Mapping):
            raise TypeError("Cleaning config must be a JSON object.")
        run = run_config_from_mapping(
            _legacy_run_payload(
                _legacy_component_payload(component_payload, overrides)
            )
        )

    resolved = resolve_run_config(run, base_dir=base_dir).config
    if resolved.data is None or resolved.data.cleaning is None:
        raise ValueError("RunConfig must include data.cleaning for this entrypoint.")
    return resolved.data.cleaning


__all__ = [
    "LEGACY_INPUT_PATH",
    "LEGACY_OUTPUT_PATH",
    "load_legacy_cleaning_config",
    "resolve_cleaning_config",
]
