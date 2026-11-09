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
from youtube_pipeline.cyclic_orchestration import (
    CyclicOrchestratorConfig,
    run_cyclic_orchestrator_dry_run,
)
from youtube_pipeline.entrypoints.cyclic_ingestion import LEGACY_OUTPUT_DIR


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("Cyclic orchestrator config must be a JSON object.")
    return payload


def _legacy_run_payload(component_payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "identity": {"run_id": "legacy_cyclic_orchestration"},
        "simulation": {
            "orchestration": {
                "simulation_dir": LEGACY_OUTPUT_DIR,
                **dict(component_payload),
            }
        },
    }


def _component_overrides(overrides: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if overrides is None:
        return None
    return {"simulation": {"orchestration": dict(overrides)}}


def load_legacy_orchestrator_config(
    config_file: str | Path | None,
    *,
    overrides: Mapping[str, Any] | None = None,
) -> CyclicOrchestratorConfig:
    """Preserve the old component-config API through the common loader."""

    payload: dict[str, Any] = {}
    if config_file is not None:
        payload = _read_json_object(Path(config_file))
    component_payload = payload.get("cyclic_ingestion_orchestrator", payload)
    if not isinstance(component_payload, Mapping):
        raise TypeError("Cyclic orchestrator config must be a JSON object.")
    run = run_config_from_mapping(
        _legacy_run_payload(component_payload),
        overrides=_component_overrides(overrides),
    )
    if run.simulation is None or run.simulation.orchestration is None:
        raise ValueError("Legacy config did not resolve cyclic orchestration.")
    return run.simulation.orchestration


def resolve_cyclic_orchestrator_config(
    *,
    config_file: str | Path | None,
    overrides: Mapping[str, Any] | None = None,
    base_dir: str | Path,
) -> CyclicOrchestratorConfig:
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
        run = load_run_config(path, overrides=_component_overrides(overrides))
    else:
        component_payload = payload.get("cyclic_ingestion_orchestrator", payload)
        if not isinstance(component_payload, Mapping):
            raise TypeError("Cyclic orchestrator config must be a JSON object.")
        run = run_config_from_mapping(
            _legacy_run_payload(component_payload),
            overrides=_component_overrides(overrides),
        )

    resolved = resolve_run_config(run, base_dir=base_dir).config
    if resolved.simulation is None or resolved.simulation.orchestration is None:
        raise ValueError("RunConfig must include simulation.orchestration for this entrypoint.")
    return resolved.simulation.orchestration


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run C-2 cyclic ingestion orchestration in dry-run mode. This validates "
            "cycle contracts and records deterministic cycle states without running "
            "monitoring, detection, RAG, LLMs, Serper, embeddings, or vectorstores."
        )
    )
    parser.add_argument("--config-file", default=None)
    parser.add_argument(
        "--simulation-dir",
        default=None,
        help="Directory containing C-0/C-1 cyclic ingestion artifacts.",
    )
    parser.add_argument("--run-monitoring", action="store_true")
    parser.add_argument("--run-detection", action="store_true")
    parser.add_argument("--run-rag", action="store_true")
    parser.add_argument(
        "--no-update-cycle-state",
        dest="update_cycle_state",
        action="store_false",
        help="Write orchestration manifest/plan but leave cycle_state.json untouched.",
    )
    parser.set_defaults(update_cycle_state=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    overrides = {
        "simulation_dir": args.simulation_dir,
        "run_monitoring": args.run_monitoring,
        "run_detection": args.run_detection,
        "run_rag": args.run_rag,
        "update_cycle_state": args.update_cycle_state,
    }
    try:
        config = resolve_cyclic_orchestrator_config(
            config_file=args.config_file,
            overrides=overrides,
            base_dir=Path.cwd(),
        )
        summary = run_cyclic_orchestrator_dry_run(config)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


__all__ = [
    "load_legacy_orchestrator_config",
    "main",
    "resolve_cyclic_orchestrator_config",
]


if __name__ == "__main__":
    main()
