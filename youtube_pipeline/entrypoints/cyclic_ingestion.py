from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from youtube_pipeline.configuration import (
    load_run_config,
    resolve_run_config,
    run_config_from_mapping,
)
from youtube_pipeline.cyclic_ingestion import (
    CyclicIngestionConfig,
    build_cyclic_ingestion_dry_run,
)

LEGACY_INPUT_PATH = "data/gold/clean_comments.parquet"
LEGACY_OUTPUT_DIR = "experiments/xiao/media/log_3/cyclic_ingestion_simulation"


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("Cyclic ingestion config must be a JSON object.")
    return payload


def _legacy_run_payload(
    component_payload: Mapping[str, Any],
) -> dict[str, Any]:
    resolved_component = {
        "input_path": LEGACY_INPUT_PATH,
        "output_dir": LEGACY_OUTPUT_DIR,
        **dict(component_payload),
    }
    run_id = resolved_component.get("simulation_run_id") or "legacy_cyclic_ingestion"
    return {
        "identity": {"run_id": str(run_id)},
        "simulation": {"ingestion": resolved_component},
    }


def _component_overrides(overrides: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if overrides is None:
        return None
    return {"simulation": {"ingestion": dict(overrides)}}


def load_legacy_cyclic_ingestion_config(
    config_file: str | Path | None,
    *,
    overrides: Mapping[str, Any] | None = None,
) -> CyclicIngestionConfig:
    """Preserve the old component-config API through the common resolver."""

    payload: dict[str, Any] = {}
    if config_file is not None:
        payload = _read_json_object(Path(config_file))
    component_payload = payload.get("cyclic_ingestion_simulation", payload)
    if not isinstance(component_payload, Mapping):
        raise TypeError("Cyclic ingestion config must be a JSON object.")
    run = run_config_from_mapping(
        _legacy_run_payload(component_payload),
        overrides=_component_overrides(overrides),
    )
    if run.simulation is None or run.simulation.ingestion is None:
        raise ValueError("Legacy cyclic ingestion config did not resolve ingestion.")
    return run.simulation.ingestion


def resolve_cyclic_ingestion_config(
    *,
    config_file: str | Path | None,
    overrides: Mapping[str, Any] | None = None,
    base_dir: str | Path,
) -> CyclicIngestionConfig:
    """Resolve either a RunConfig profile or the legacy component format."""

    if config_file is not None:
        path = Path(config_file)
        payload = _read_json_object(path)
        is_run_profile = "identity" in payload or any(
            section in payload for section in ("simulation", "signals", "detection")
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
        component_payload = payload.get("cyclic_ingestion_simulation", payload)
        if not isinstance(component_payload, Mapping):
            raise TypeError("Cyclic ingestion config must be a JSON object.")
        run = run_config_from_mapping(
            _legacy_run_payload(component_payload),
            overrides=_component_overrides(overrides),
        )

    resolved = resolve_run_config(run, base_dir=base_dir).config
    if resolved.simulation is None or resolved.simulation.ingestion is None:
        raise ValueError("RunConfig must include simulation.ingestion for this entrypoint.")
    return resolved.simulation.ingestion


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build C-0/C-1 dry-run contracts and temporal partitions for "
            "cyclic_ingestion_simulation. This does not run monitoring, detection, "
            "RAG, LLMs, Serper, embeddings, or vectorstores."
        )
    )
    parser.add_argument("--config-file", default=None)
    parser.add_argument("--input-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--timezone", default=None)
    parser.add_argument("--canonical-timezone", default=None)
    parser.add_argument("--analysis-window-size-days", type=int, default=None)
    parser.add_argument("--collection-start-date-local", default=None)
    parser.add_argument("--collection-end-date-local", default=None)
    parser.add_argument("--simulation-run-id", default=None)
    parser.add_argument("--rag-mode", default=None)
    parser.add_argument("--notes", default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    overrides = {
        key: value
        for key, value in {
            "input_path": args.input_path,
            "output_dir": args.output_dir,
            "timezone": args.timezone,
            "canonical_timezone": args.canonical_timezone,
            "analysis_window_size_days": args.analysis_window_size_days,
            "collection_start_date_local": args.collection_start_date_local,
            "collection_end_date_local": args.collection_end_date_local,
            "simulation_run_id": args.simulation_run_id,
            "rag_mode": args.rag_mode,
            "notes": args.notes,
        }.items()
        if value is not None
    }
    config = resolve_cyclic_ingestion_config(
        config_file=args.config_file,
        overrides=overrides,
        base_dir=Path.cwd(),
    )
    summary = build_cyclic_ingestion_dry_run(config)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


__all__ = [
    "LEGACY_INPUT_PATH",
    "LEGACY_OUTPUT_DIR",
    "load_legacy_cyclic_ingestion_config",
    "main",
    "resolve_cyclic_ingestion_config",
]


if __name__ == "__main__":
    main()
