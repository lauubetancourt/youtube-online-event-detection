from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from youtube_pipeline.configuration import (
    DetectionConfig,
    load_run_config,
    resolve_run_config,
    run_config_from_mapping,
)
from youtube_pipeline.cyclic_detection_connector import (
    ACTIVITY_DETECTION_RUNTIME_MODE,
    CyclicDetectionConnectorConfig,
    run_cyclic_detection_connector,
)
from youtube_pipeline.entrypoints.cyclic_ingestion import (
    LEGACY_INPUT_PATH,
    LEGACY_OUTPUT_DIR,
)


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("Cyclic detection connector config must be a JSON object.")
    return payload


def _legacy_run_payload(component_payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "identity": {"run_id": "legacy_cyclic_detection_connector"},
        "detection": {
            "connector": {
                "simulation_dir": LEGACY_OUTPUT_DIR,
                "canonical_dataset_path": LEGACY_INPUT_PATH,
                **dict(component_payload),
            },
            "xiao_ema": {},
        },
    }


def _component_overrides(overrides: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if overrides is None:
        return None
    return {"detection": {"connector": dict(overrides)}}


def load_legacy_detection_connector_config(
    config_file: str | Path | None,
    *,
    overrides: Mapping[str, Any] | None = None,
) -> CyclicDetectionConnectorConfig:
    """Preserve the old component-config API through the common loader."""

    payload: dict[str, Any] = {}
    if config_file is not None:
        payload = _read_json_object(Path(config_file))
    component_payload = payload.get("cyclic_detection_connector", payload)
    if not isinstance(component_payload, Mapping):
        raise TypeError("Cyclic detection connector config must be a JSON object.")
    run = run_config_from_mapping(
        _legacy_run_payload(component_payload),
        overrides=_component_overrides(overrides),
    )
    if run.detection is None or run.detection.connector is None:
        raise ValueError("Legacy config did not resolve the detection connector.")
    return run.detection.connector


def resolve_cyclic_detection_config(
    *,
    config_file: str | Path | None,
    overrides: Mapping[str, Any] | None = None,
    base_dir: str | Path,
) -> DetectionConfig:
    """Resolve the connector and its explicit XIAO subconfiguration."""

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
        run = load_run_config(path, overrides=_component_overrides(overrides))
    else:
        component_payload = payload.get("cyclic_detection_connector", payload)
        if not isinstance(component_payload, Mapping):
            raise TypeError("Cyclic detection connector config must be a JSON object.")
        run = run_config_from_mapping(
            _legacy_run_payload(component_payload),
            overrides=_component_overrides(overrides),
        )

    resolved = resolve_run_config(run, base_dir=base_dir).config
    if resolved.detection is None or resolved.detection.connector is None:
        raise ValueError("RunConfig must include detection.connector for this entrypoint.")
    connector = resolved.detection.connector
    if connector.mode == ACTIVITY_DETECTION_RUNTIME_MODE:
        if resolved.detection.activity_route is None:
            raise ValueError(
                "activity_detection_runtime requires detection.activity_route."
            )
        selected_config = getattr(
            resolved.detection,
            resolved.detection.activity_route.detector_id,
            None,
        )
        if selected_config is None:
            raise ValueError(
                "activity_detection_runtime requires the selected detector config."
            )
    elif resolved.detection.xiao_ema is None:
        raise ValueError("RunConfig must include detection.xiao_ema for this entrypoint.")
    return resolved.detection


def resolve_cyclic_detection_connector_config(
    *,
    config_file: str | Path | None,
    overrides: Mapping[str, Any] | None = None,
    base_dir: str | Path,
) -> CyclicDetectionConnectorConfig:
    """Compatibility view returning only the connector configuration."""

    detection = resolve_cyclic_detection_config(
        config_file=config_file,
        overrides=overrides,
        base_dir=base_dir,
    )
    assert detection.connector is not None
    return detection.connector


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run C-4 cyclic detection connector modes. detection_dry_run prepares "
            "contracts only; detection_smoke_test resolves active comments against "
            "Gold in memory and executes the approved small monitoring/detection "
            "smoke path. Neither mode executes RAG, LLMs, Serper, embeddings, or "
            "vectorstores."
        )
    )
    parser.add_argument("--config-file", default=None)
    parser.add_argument("--simulation-dir", default=None)
    parser.add_argument("--mode", default=None)
    parser.add_argument("--max-cycles", type=int, default=None)
    parser.add_argument("--canonical-dataset-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--debug-full-rows", action="store_true")
    parser.add_argument("--run-monitoring", action="store_true")
    parser.add_argument("--run-detection", action="store_true")
    parser.add_argument("--run-rag", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    overrides = {
        "simulation_dir": args.simulation_dir,
        "mode": args.mode,
        "max_cycles": args.max_cycles,
        "canonical_dataset_path": args.canonical_dataset_path,
        "output_dir": args.output_dir,
        "debug_full_rows": args.debug_full_rows,
        "run_monitoring": args.run_monitoring,
        "run_detection": args.run_detection,
        "run_rag": args.run_rag,
    }
    try:
        detection = resolve_cyclic_detection_config(
            config_file=args.config_file,
            overrides=overrides,
            base_dir=Path.cwd(),
        )
        assert detection.connector is not None
        if detection.connector.mode == ACTIVITY_DETECTION_RUNTIME_MODE:
            raise ValueError(
                "activity_detection_runtime requires the full cyclic-pipeline "
                "entrypoint so the signal definition and run identity remain resolved."
            )
        assert detection.xiao_ema is not None
        summary = run_cyclic_detection_connector(
            detection.connector,
            xiao_config=detection.xiao_ema,
        )
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


__all__ = [
    "load_legacy_detection_connector_config",
    "main",
    "resolve_cyclic_detection_config",
    "resolve_cyclic_detection_connector_config",
]


if __name__ == "__main__":
    main()
