from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from youtube_pipeline.configuration import (
    load_run_config,
    resolve_run_config,
    run_config_from_mapping,
)
from youtube_pipeline.daily_rag_context_selection import (
    DailyContextSelectionConfig,
    write_daily_context_selection_artifacts_from_config,
)


LEGACY_SIMULATION_DIR = "experiments/xiao/media/log_3/cyclic_ingestion_simulation"
LEGACY_CONSUMER_DIR = f"{LEGACY_SIMULATION_DIR}/daily_rag_consumer"
LEGACY_SIDECARS_DIR = f"{LEGACY_SIMULATION_DIR}/daily_rag_sidecars"
LEGACY_OUTPUT_DIR = f"{LEGACY_SIMULATION_DIR}/daily_rag_context_selection"
_LEGACY_IDENTITY_PLACEHOLDER = "legacy_daily_rag_context_selection"


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("Daily context selection config must be a JSON object.")
    return payload


def _legacy_defaults() -> dict[str, Any]:
    return {
        "consumer_dir": LEGACY_CONSUMER_DIR,
        "sidecars_dir": LEGACY_SIDECARS_DIR,
        "output_dir": LEGACY_OUTPUT_DIR,
    }


def _legacy_component_with_overrides(
    component_payload: Mapping[str, Any],
    overrides: Mapping[str, Any] | None,
) -> dict[str, Any]:
    effective = {**_legacy_defaults(), **dict(component_payload)}
    if overrides is not None:
        effective.update(
            {
                key: value
                for key, value in overrides.items()
                if value is not None
            }
        )
    return effective


def _legacy_run_payload(component_payload: Mapping[str, Any]) -> dict[str, Any]:
    component = dict(component_payload)
    explicit_run_id = component.get("run_id")
    return {
        "identity": {
            "run_id": explicit_run_id or _LEGACY_IDENTITY_PLACEHOLDER,
        },
        "rag": {"daily_context_selection": component},
    }


def _legacy_run_id(component_payload: Mapping[str, Any]) -> str | None:
    value = component_payload.get("run_id")
    return None if value is None else str(value)


def _component_overrides(
    overrides: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if overrides is None:
        return None
    component = dict(overrides)
    run_id = component.pop("run_id", None)
    result: dict[str, Any] = {"rag": {"daily_context_selection": component}}
    if run_id is not None:
        result["rag"]["daily_context_selection"]["run_id"] = run_id
    return result


def load_legacy_daily_context_selection_config(
    config_file: str | Path | None,
    *,
    overrides: Mapping[str, Any] | None = None,
) -> DailyContextSelectionConfig:
    """Preserve the legacy component API without changing derived run IDs."""

    payload: dict[str, Any] = {}
    if config_file is not None:
        payload = _read_json_object(Path(config_file))
    component_payload = payload.get("daily_rag_context_selection", payload)
    if not isinstance(component_payload, Mapping):
        raise TypeError("Daily context selection config must be a JSON object.")
    effective = _legacy_component_with_overrides(component_payload, overrides)
    run = run_config_from_mapping(_legacy_run_payload(effective))
    if run.rag is None or run.rag.daily_context_selection is None:
        raise ValueError("Legacy config did not resolve daily context selection.")
    return replace(
        run.rag.daily_context_selection,
        run_id=_legacy_run_id(effective),
    )


def resolve_daily_context_selection_config(
    *,
    config_file: str | Path | None,
    overrides: Mapping[str, Any] | None = None,
    base_dir: str | Path,
) -> DailyContextSelectionConfig:
    """Resolve either a RunConfig profile or the legacy component format."""

    if config_file is not None:
        path = Path(config_file)
        payload = _read_json_object(path)
        is_run_profile = "identity" in payload or any(
            section in payload
            for section in ("simulation", "signals", "detection", "rag")
        )
    else:
        path = None
        payload = {}
        is_run_profile = False

    legacy_run_id: str | None = None
    if is_run_profile:
        run = load_run_config(path, overrides=_component_overrides(overrides))
    else:
        component_payload = payload.get("daily_rag_context_selection", payload)
        if not isinstance(component_payload, Mapping):
            raise TypeError("Daily context selection config must be a JSON object.")
        effective = _legacy_component_with_overrides(component_payload, overrides)
        legacy_run_id = _legacy_run_id(effective)
        run = run_config_from_mapping(_legacy_run_payload(effective))

    resolved = resolve_run_config(run, base_dir=base_dir).config
    if resolved.rag is None or resolved.rag.daily_context_selection is None:
        raise ValueError(
            "RunConfig must include rag.daily_context_selection for this entrypoint."
        )
    component = resolved.rag.daily_context_selection
    if not is_run_profile:
        component = replace(component, run_id=legacy_run_id)
    return component


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Select deterministic non-generative context for daily RAG events. "
            "This does not call LLMs, Serper, embeddings, vectorstores, G-1, or G-2."
        )
    )
    parser.add_argument("--config-file", default=None)
    parser.add_argument("--consumer-dir", default=None)
    parser.add_argument("--sidecars-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--max-selected-tokens-per-event", type=int, default=None)
    parser.add_argument("--alert-coverage-target", type=float, default=None)
    parser.add_argument("--notes", default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    overrides = {
        key: value
        for key, value in {
            "consumer_dir": args.consumer_dir,
            "sidecars_dir": args.sidecars_dir,
            "output_dir": args.output_dir,
            "run_id": args.run_id,
            "max_selected_tokens_per_event": args.max_selected_tokens_per_event,
            "alert_coverage_target": args.alert_coverage_target,
            "notes": args.notes,
        }.items()
        if value is not None
    }
    try:
        config = resolve_daily_context_selection_config(
            config_file=args.config_file,
            overrides=overrides,
            base_dir=Path.cwd(),
        )
        summary = write_daily_context_selection_artifacts_from_config(config)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))


__all__ = [
    "LEGACY_CONSUMER_DIR",
    "LEGACY_OUTPUT_DIR",
    "LEGACY_SIDECARS_DIR",
    "LEGACY_SIMULATION_DIR",
    "load_legacy_daily_context_selection_config",
    "main",
    "resolve_daily_context_selection_config",
]


if __name__ == "__main__":
    main()
