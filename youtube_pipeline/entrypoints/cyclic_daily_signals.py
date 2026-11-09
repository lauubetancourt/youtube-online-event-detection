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
from youtube_pipeline.cyclic_daily_signals import (
    CyclicDailySignalConfig,
    run_cyclic_daily_signals,
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
        raise TypeError("Cyclic daily signal config must be a JSON object.")
    return payload


def _legacy_run_payload(component_payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "identity": {"run_id": "legacy_cyclic_daily_signals"},
        "signals": {
            "daily": {
                "simulation_dir": LEGACY_OUTPUT_DIR,
                "canonical_dataset_path": LEGACY_INPUT_PATH,
                **dict(component_payload),
            }
        },
    }


def _component_overrides(overrides: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if overrides is None:
        return None
    return {"signals": {"daily": dict(overrides)}}


def load_legacy_daily_signal_config(
    config_file: str | Path | None,
    *,
    overrides: Mapping[str, Any] | None = None,
) -> CyclicDailySignalConfig:
    """Preserve the old component-config API through the common loader."""

    payload: dict[str, Any] = {}
    if config_file is not None:
        payload = _read_json_object(Path(config_file))
    component_payload = payload.get("cyclic_daily_signals", payload)
    if not isinstance(component_payload, Mapping):
        raise TypeError("Cyclic daily signal config must be a JSON object.")
    run = run_config_from_mapping(
        _legacy_run_payload(component_payload),
        overrides=_component_overrides(overrides),
    )
    if run.signals is None or run.signals.daily is None:
        raise ValueError("Legacy config did not resolve daily signals.")
    return run.signals.daily


def resolve_cyclic_daily_signal_config(
    *,
    config_file: str | Path | None,
    overrides: Mapping[str, Any] | None = None,
    base_dir: str | Path,
) -> CyclicDailySignalConfig:
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
        component_payload = payload.get("cyclic_daily_signals", payload)
        if not isinstance(component_payload, Mapping):
            raise TypeError("Cyclic daily signal config must be a JSON object.")
        run = run_config_from_mapping(
            _legacy_run_payload(component_payload),
            overrides=_component_overrides(overrides),
        )

    resolved = resolve_run_config(run, base_dir=base_dir).config
    if resolved.signals is None or resolved.signals.daily is None:
        raise ValueError("RunConfig must include signals.daily for this entrypoint.")
    return resolved.signals.daily


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build C-5 daily aggregated signal series for future stateful XIAO execution. "
            "This mode does not execute XIAO, detection, RAG, LLMs, Serper, embeddings, "
            "or vectorstores."
        )
    )
    parser.add_argument("--config-file", default=None)
    parser.add_argument("--simulation-dir", default=None)
    parser.add_argument("--mode", default=None)
    parser.add_argument("--canonical-dataset-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--xiao-signal-name", default=None)
    parser.add_argument("--max-cycles", type=int, default=None)
    parser.add_argument("--run-xiao", action="store_true")
    parser.add_argument("--run-detection", action="store_true")
    parser.add_argument("--run-rag", action="store_true")
    parser.add_argument("--run-llm", action="store_true")
    parser.add_argument("--run-serper", action="store_true")
    parser.add_argument("--use-embeddings", action="store_true")
    parser.add_argument("--use-vectorstore", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    overrides = {
        "simulation_dir": args.simulation_dir,
        "mode": args.mode,
        "canonical_dataset_path": args.canonical_dataset_path,
        "output_dir": args.output_dir,
        "xiao_signal_name": args.xiao_signal_name,
        "max_cycles": args.max_cycles,
        "run_xiao": args.run_xiao,
        "run_detection": args.run_detection,
        "run_rag": args.run_rag,
        "run_llm": args.run_llm,
        "run_serper": args.run_serper,
        "use_embeddings": args.use_embeddings,
        "use_vectorstore": args.use_vectorstore,
    }
    try:
        config = resolve_cyclic_daily_signal_config(
            config_file=args.config_file,
            overrides=overrides,
            base_dir=Path.cwd(),
        )
        summary = run_cyclic_daily_signals(config)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


__all__ = [
    "load_legacy_daily_signal_config",
    "main",
    "resolve_cyclic_daily_signal_config",
]


if __name__ == "__main__":
    main()
