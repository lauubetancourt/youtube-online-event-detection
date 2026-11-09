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
from youtube_pipeline.daily_frequency_baseline import (
    DETECTOR_NAME,
    DailyFrequencyBaselineConfig,
    run_daily_frequency_baseline,
)
from youtube_pipeline.entrypoints.cyclic_ingestion import LEGACY_OUTPUT_DIR


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("Daily frequency baseline config must be a JSON object.")
    return payload


def _legacy_run_payload(component_payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "identity": {"run_id": "legacy_daily_frequency_baseline"},
        "detection": {
            "daily_frequency": {
                "simulation_dir": LEGACY_OUTPUT_DIR,
                **dict(component_payload),
            }
        },
    }


def _component_overrides(overrides: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if overrides is None:
        return None
    return {"detection": {"daily_frequency": dict(overrides)}}


def load_legacy_baseline_config(
    config_file: str | Path | None,
    *,
    overrides: Mapping[str, Any] | None = None,
) -> DailyFrequencyBaselineConfig:
    """Preserve the old component-config API through the common loader."""

    payload: dict[str, Any] = {}
    if config_file is not None:
        payload = _read_json_object(Path(config_file))
    component_payload = payload.get(DETECTOR_NAME, payload)
    if not isinstance(component_payload, Mapping):
        raise TypeError("Daily frequency baseline config must be a JSON object.")
    run = run_config_from_mapping(
        _legacy_run_payload(component_payload),
        overrides=_component_overrides(overrides),
    )
    if run.detection is None or run.detection.daily_frequency is None:
        raise ValueError("Legacy config did not resolve the daily baseline.")
    return run.detection.daily_frequency


def resolve_daily_frequency_baseline_config(
    *,
    config_file: str | Path | None,
    overrides: Mapping[str, Any] | None = None,
    base_dir: str | Path,
) -> DailyFrequencyBaselineConfig:
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
        component_payload = payload.get(DETECTOR_NAME, payload)
        if not isinstance(component_payload, Mapping):
            raise TypeError("Daily frequency baseline config must be a JSON object.")
        run = run_config_from_mapping(
            _legacy_run_payload(component_payload),
            overrides=_component_overrides(overrides),
        )

    resolved = resolve_run_config(run, base_dir=base_dir).config
    if resolved.detection is None or resolved.detection.daily_frequency is None:
        raise ValueError("RunConfig must include detection.daily_frequency for this entrypoint.")
    return resolved.detection.daily_frequency


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the daily_frequency_baseline detector over C-5 daily signal series. "
            "This detector is external to XIAO and does not call RAG, LLMs, Serper, "
            "embeddings, or vectorstores."
        )
    )
    parser.add_argument("--config-file", default=None)
    parser.add_argument("--simulation-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--signal-name", default=None)
    parser.add_argument("--baseline-window-size-cycles", type=int, default=None)
    parser.add_argument("--k-multiplier", type=float, default=None)
    parser.add_argument("--min-count", type=int, default=None)
    parser.add_argument("--min-delta", type=float, default=None)
    parser.add_argument("--min-pct-change", type=float, default=None)
    parser.add_argument("--warmup-cycles", type=int, default=None)
    parser.add_argument("--cooldown-cycles", type=int, default=None)
    parser.add_argument("--cooldown-policy", default=None)
    parser.add_argument("--disable-pct-change", action="store_true")
    parser.add_argument("--disable-delta", action="store_true")
    parser.add_argument("--allow-decreases", action="store_true")
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
        "output_dir": args.output_dir,
        "signal_name": args.signal_name,
        "baseline_window_size_cycles": args.baseline_window_size_cycles,
        "k_multiplier": args.k_multiplier,
        "min_count": args.min_count,
        "min_delta": args.min_delta,
        "min_pct_change": args.min_pct_change,
        "warmup_cycles": args.warmup_cycles,
        "cooldown_cycles": args.cooldown_cycles,
        "cooldown_policy": args.cooldown_policy,
        "use_pct_change": False if args.disable_pct_change else None,
        "use_delta": False if args.disable_delta else None,
        "trigger_on_increase_only": False if args.allow_decreases else None,
        "run_rag": args.run_rag,
        "run_llm": args.run_llm,
        "run_serper": args.run_serper,
        "use_embeddings": args.use_embeddings,
        "use_vectorstore": args.use_vectorstore,
    }
    try:
        config = resolve_daily_frequency_baseline_config(
            config_file=args.config_file,
            overrides=overrides,
            base_dir=Path.cwd(),
        )
        summary = run_daily_frequency_baseline(config)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


__all__ = [
    "load_legacy_baseline_config",
    "main",
    "resolve_daily_frequency_baseline_config",
]


if __name__ == "__main__":
    main()
