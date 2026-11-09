from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from youtube_pipeline.configuration import (
    load_run_config,
    resolve_run_config,
    run_config_from_mapping,
)
from youtube_pipeline.daily_rag_sidecars import (
    DailyRagSidecarBuildConfig,
    write_daily_rag_sidecar_artifacts_from_config,
)


LEGACY_SIMULATION_DIR = "experiments/xiao/media/log_3/cyclic_ingestion_simulation"
LEGACY_BASELINE_DIR = f"{LEGACY_SIMULATION_DIR}/daily_frequency_baseline_cooldown_0"
LEGACY_DAILY_EVENTS_PATH = f"{LEGACY_BASELINE_DIR}/cycle_daily_frequency_events.jsonl"
LEGACY_DAILY_SCORES_PATH = f"{LEGACY_BASELINE_DIR}/cycle_daily_frequency_scores.jsonl"
LEGACY_DAILY_DETECTOR_MANIFEST_PATH = (
    f"{LEGACY_BASELINE_DIR}/cycle_daily_frequency_detector_manifest.json"
)
LEGACY_SIGNAL_SERIES_PATH = f"{LEGACY_SIMULATION_DIR}/cycle_signal_series.jsonl"
LEGACY_WINDOW_INVENTORY_PATH = f"{LEGACY_SIMULATION_DIR}/cycle_window_inventory.csv"
LEGACY_STATEFUL_CONTEXT_PATH = f"{LEGACY_SIMULATION_DIR}/cycle_stateful_context.json"
LEGACY_COMMENTS_PATH = "data/gold/clean_comments.parquet"
LEGACY_OUTPUT_DIR = f"{LEGACY_SIMULATION_DIR}/daily_rag_sidecars"


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("Daily RAG sidecar config must be a JSON object.")
    return payload


def _legacy_defaults() -> dict[str, Any]:
    return {
        "daily_events_path": LEGACY_DAILY_EVENTS_PATH,
        "output_dir": LEGACY_OUTPUT_DIR,
        "comments_path": LEGACY_COMMENTS_PATH,
        "cycle_window_inventory_path": LEGACY_WINDOW_INVENTORY_PATH,
        "daily_scores_path": LEGACY_DAILY_SCORES_PATH,
        "daily_detector_manifest_path": LEGACY_DAILY_DETECTOR_MANIFEST_PATH,
        "cycle_signal_series_path": LEGACY_SIGNAL_SERIES_PATH,
        "cycle_stateful_context_path": LEGACY_STATEFUL_CONTEXT_PATH,
    }


def _derived_legacy_run_id(component_payload: Mapping[str, Any]) -> str:
    explicit = component_payload.get("run_id")
    if explicit is not None:
        return str(explicit)
    identity = "|".join(
        str(component_payload[field])
        for field in (
            "daily_events_path",
            "cycle_window_inventory_path",
            "comments_path",
        )
    )
    return "drun_" + hashlib.sha1(identity.encode("utf-8")).hexdigest()[:12]


def _legacy_run_payload(component_payload: Mapping[str, Any]) -> dict[str, Any]:
    resolved_component = {**_legacy_defaults(), **dict(component_payload)}
    run_id = _derived_legacy_run_id(resolved_component)
    resolved_component["run_id"] = run_id
    return {
        "identity": {"run_id": run_id},
        "rag": {"daily_sidecars": resolved_component},
    }


def _component_overrides(
    overrides: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if overrides is None:
        return None
    component = dict(overrides)
    run_id = component.pop("run_id", None)
    result: dict[str, Any] = {"rag": {"daily_sidecars": component}}
    if run_id is not None:
        result["rag"]["daily_sidecars"]["run_id"] = run_id
    return result


def _legacy_component_with_overrides(
    component_payload: Mapping[str, Any],
    overrides: Mapping[str, Any] | None,
) -> dict[str, Any]:
    effective = dict(component_payload)
    if overrides is not None:
        effective.update(
            {
                key: value
                for key, value in overrides.items()
                if value is not None
            }
        )
    return effective


def load_legacy_daily_rag_sidecar_config(
    config_file: str | Path | None,
    *,
    overrides: Mapping[str, Any] | None = None,
) -> DailyRagSidecarBuildConfig:
    """Preserve the component-config API through the common RunConfig loader."""

    payload: dict[str, Any] = {}
    if config_file is not None:
        payload = _read_json_object(Path(config_file))
    component_payload = payload.get("daily_rag_sidecars", payload)
    if not isinstance(component_payload, Mapping):
        raise TypeError("Daily RAG sidecar config must be a JSON object.")
    legacy_payload = _legacy_run_payload(
        _legacy_component_with_overrides(component_payload, overrides)
    )
    run = run_config_from_mapping(legacy_payload)
    if run.rag is None or run.rag.daily_sidecars is None:
        raise ValueError("Legacy config did not resolve daily RAG sidecars.")
    return run.rag.daily_sidecars


def resolve_daily_rag_sidecar_config(
    *,
    config_file: str | Path | None,
    overrides: Mapping[str, Any] | None = None,
    base_dir: str | Path,
) -> DailyRagSidecarBuildConfig:
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

    if is_run_profile:
        run = load_run_config(path, overrides=_component_overrides(overrides))
    else:
        component_payload = payload.get("daily_rag_sidecars", payload)
        if not isinstance(component_payload, Mapping):
            raise TypeError("Daily RAG sidecar config must be a JSON object.")
        run = run_config_from_mapping(
            _legacy_run_payload(
                _legacy_component_with_overrides(component_payload, overrides)
            )
        )

    resolved = resolve_run_config(run, base_dir=base_dir).config
    if resolved.rag is None or resolved.rag.daily_sidecars is None:
        raise ValueError("RunConfig must include rag.daily_sidecars for this entrypoint.")
    return resolved.rag.daily_sidecars


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build non-generative RAG sidecars for daily_frequency_baseline events. "
            "This does not call LLMs, Serper, embeddings, vectorstores, G-1, or G-2."
        )
    )
    parser.add_argument("--config-file", default=None)
    parser.add_argument("--daily-events-path", default=None)
    parser.add_argument("--daily-scores-path", default=None)
    parser.add_argument("--daily-detector-manifest-path", default=None)
    parser.add_argument("--cycle-signal-series-path", default=None)
    parser.add_argument("--cycle-window-inventory-path", default=None)
    parser.add_argument("--cycle-stateful-context-path", default=None)
    parser.add_argument("--comments-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-comments-per-context-unit", type=int, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--notes", default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    overrides = {
        key: value
        for key, value in {
            "daily_events_path": args.daily_events_path,
            "daily_scores_path": args.daily_scores_path,
            "daily_detector_manifest_path": args.daily_detector_manifest_path,
            "cycle_signal_series_path": args.cycle_signal_series_path,
            "cycle_window_inventory_path": args.cycle_window_inventory_path,
            "cycle_stateful_context_path": args.cycle_stateful_context_path,
            "comments_path": args.comments_path,
            "output_dir": args.output_dir,
            "max_comments_per_context_unit": args.max_comments_per_context_unit,
            "run_id": args.run_id,
            "notes": args.notes,
        }.items()
        if value is not None
    }
    try:
        config = resolve_daily_rag_sidecar_config(
            config_file=args.config_file,
            overrides=overrides,
            base_dir=Path.cwd(),
        )
        summary = write_daily_rag_sidecar_artifacts_from_config(config)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))


__all__ = [
    "LEGACY_COMMENTS_PATH",
    "LEGACY_DAILY_DETECTOR_MANIFEST_PATH",
    "LEGACY_DAILY_EVENTS_PATH",
    "LEGACY_DAILY_SCORES_PATH",
    "LEGACY_OUTPUT_DIR",
    "LEGACY_SIGNAL_SERIES_PATH",
    "LEGACY_SIMULATION_DIR",
    "LEGACY_STATEFUL_CONTEXT_PATH",
    "LEGACY_WINDOW_INVENTORY_PATH",
    "load_legacy_daily_rag_sidecar_config",
    "main",
    "resolve_daily_rag_sidecar_config",
]


if __name__ == "__main__":
    main()
