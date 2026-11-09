from __future__ import annotations

import json
import logging
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

from youtube_pipeline.configuration import (
    DetectionConfig,
    ResolvedRunConfig,
    RunConfig,
    SignalsConfig,
    SimulationConfig,
    load_run_config,
    resolve_run_config,
)
from youtube_pipeline.cyclic_daily_signals import run_cyclic_daily_signals
from youtube_pipeline.cyclic_detection_connector import (
    ACTIVITY_DETECTION_RUNTIME_MODE,
    DETECTION_CONNECTOR_MODE,
    run_cyclic_detection_connector,
)
from youtube_pipeline.cyclic_ingestion import build_cyclic_ingestion_dry_run
from youtube_pipeline.cyclic_orchestration import run_cyclic_orchestrator_dry_run
from youtube_pipeline.cyclic_stateful_adapter import run_cyclic_stateful_adapter
from youtube_pipeline.daily_frequency_baseline import run_daily_frequency_baseline
from youtube_pipeline.entrypoints.common_cli import (
    CommonRunCliOptions,
    parse_common_run_args,
)
from youtube_pipeline.run_manifest import (
    validate_current_traceability_support,
    write_run_manifest,
)


def _resolved_path(path: str | Path, *, base_dir: Path) -> Path:
    value = Path(path).expanduser()
    if not value.is_absolute():
        value = base_dir / value
    return value.resolve(strict=False)


def _relocate_optional_output(
    path: str | Path | None,
    *,
    previous_root: Path,
    output_root: Path,
    field_name: str,
) -> Path | None:
    if path is None:
        return None
    resolved = Path(path).resolve(strict=False)
    try:
        relative = resolved.relative_to(previous_root)
    except ValueError as exc:
        raise ValueError(
            f"{field_name} must be inside simulation.ingestion.output_dir "
            "when --output-root is used."
        ) from exc
    return output_root / relative


def _with_output_root(
    config: RunConfig,
    *,
    output_root: Path,
) -> RunConfig:
    """Relocate the current cyclic artifact tree without changing its structure."""

    simulation = config.simulation
    signals = config.signals
    detection = config.detection
    if simulation is None or simulation.ingestion is None:
        raise ValueError(
            "--output-root requires simulation.ingestion in the cyclic profile."
        )
    previous_root = Path(simulation.ingestion.output_dir).resolve(strict=False)

    if simulation.orchestration is None or simulation.stateful_adapter is None:
        raise ValueError(
            "The cyclic runner requires simulation.orchestration and "
            "simulation.stateful_adapter."
        )
    if signals is None:
        raise ValueError("The cyclic runner requires signals configuration.")
    if detection is None or detection.connector is None:
        raise ValueError("The cyclic runner requires detection.connector.")

    shared_simulation_paths = {
        "simulation.orchestration.simulation_dir": simulation.orchestration.simulation_dir,
        "simulation.stateful_adapter.simulation_dir": (
            simulation.stateful_adapter.simulation_dir
        ),
        "detection.connector.simulation_dir": detection.connector.simulation_dir,
    }
    if signals.daily is not None:
        shared_simulation_paths["signals.daily.simulation_dir"] = (
            signals.daily.simulation_dir
        )
    if detection.daily_frequency is not None:
        shared_simulation_paths["detection.daily_frequency.simulation_dir"] = (
            detection.daily_frequency.simulation_dir
        )
    mismatched = [
        name
        for name, value in shared_simulation_paths.items()
        if Path(value).resolve(strict=False) != previous_root
    ]
    if mismatched:
        raise ValueError(
            "--output-root requires one shared cyclic simulation directory; "
            "mismatched fields: " + ", ".join(mismatched)
        )

    relocated_simulation = SimulationConfig(
        ingestion=replace(simulation.ingestion, output_dir=output_root),
        orchestration=replace(
            simulation.orchestration,
            simulation_dir=output_root,
        ),
        stateful_adapter=replace(
            simulation.stateful_adapter,
            simulation_dir=output_root,
        ),
    )
    relocated_signals = SignalsConfig(
        daily=(
            replace(
                signals.daily,
                simulation_dir=output_root,
                output_dir=_relocate_optional_output(
                    signals.daily.output_dir,
                    previous_root=previous_root,
                    output_root=output_root,
                    field_name="signals.daily.output_dir",
                ),
            )
            if signals.daily is not None
            else None
        ),
        activity=signals.activity,
        polarization=signals.polarization,
    )
    relocated_detection = DetectionConfig(
        activity_route=detection.activity_route,
        routes=detection.routes,
        gap_policy=detection.gap_policy,
        connector=replace(
            detection.connector,
            simulation_dir=output_root,
            output_dir=_relocate_optional_output(
                detection.connector.output_dir,
                previous_root=previous_root,
                output_root=output_root,
                field_name="detection.connector.output_dir",
            ),
        ),
        xiao_ema=detection.xiao_ema,
        page_hinkley=detection.page_hinkley,
        adwin=detection.adwin,
        kswin=detection.kswin,
        daily_frequency=(
            replace(
                detection.daily_frequency,
                simulation_dir=output_root,
                output_dir=_relocate_optional_output(
                    detection.daily_frequency.output_dir,
                    previous_root=previous_root,
                    output_root=output_root,
                    field_name="detection.daily_frequency.output_dir",
                ),
            )
            if detection.daily_frequency is not None
            else None
        ),
    )
    return RunConfig(
        identity=config.identity,
        data=config.data,
        simulation=relocated_simulation,
        signals=relocated_signals,
        detection=relocated_detection,
        association=config.association,
        polarization=config.polarization,
        rag=config.rag,
        artifacts=config.artifacts,
    )


def _validate_cyclic_profile(config: RunConfig) -> None:
    simulation = config.simulation
    signals = config.signals
    detection = config.detection
    missing: list[str] = []
    if simulation is None or simulation.ingestion is None:
        missing.append("simulation.ingestion")
    if simulation is None or simulation.orchestration is None:
        missing.append("simulation.orchestration")
    if simulation is None or simulation.stateful_adapter is None:
        missing.append("simulation.stateful_adapter")
    if signals is None:
        missing.append("signals")
    if detection is None or detection.connector is None:
        missing.append("detection.connector")
    if missing:
        raise ValueError(
            "Cyclic pipeline profile is missing required sections: "
            + ", ".join(missing)
        )

    assert simulation is not None
    assert simulation.ingestion is not None
    assert detection is not None
    assert detection.connector is not None
    if not simulation.ingestion.dry_run:
        raise ValueError(
            "The current cyclic runner requires simulation.ingestion.dry_run=true."
        )
    if detection.connector.mode not in {
        DETECTION_CONNECTOR_MODE,
        ACTIVITY_DETECTION_RUNTIME_MODE,
    }:
        raise ValueError(
            "The cyclic runner requires detection.connector.mode to be "
            f"{DETECTION_CONNECTOR_MODE!r} or "
            f"{ACTIVITY_DETECTION_RUNTIME_MODE!r}."
        )
    if detection.connector.mode == ACTIVITY_DETECTION_RUNTIME_MODE:
        if signals is None or signals.activity is None:
            missing.append("signals.activity")
        if detection.activity_route is None:
            missing.append("detection.activity_route")
        if (
            signals is not None
            and signals.activity is not None
            and detection.activity_route is not None
            and signals.activity.signal_id != detection.activity_route.signal_id
        ):
            raise ValueError(
                "signals.activity.signal_id must match "
                "detection.activity_route.signal_id."
            )
    else:
        if signals is None or signals.daily is None:
            missing.append("signals.daily")
        if detection.xiao_ema is None:
            missing.append("detection.xiao_ema")
        if detection.daily_frequency is None:
            missing.append("detection.daily_frequency")
    if missing:
        raise ValueError(
            "Cyclic pipeline profile is missing required sections: "
            + ", ".join(missing)
        )
    if (
        config.data is not None
        and config.data.experimental_input is not None
        and config.data.local_files is not None
        and Path(config.data.local_files.comments_path).absolute()
        != Path(simulation.ingestion.input_path).absolute()
    ):
        raise ValueError(
            "Opt-in cyclic activity and stance inputs must reference the same "
            "canonical comments path."
        )


def resolve_cyclic_pipeline_run(
    options: CommonRunCliOptions,
    *,
    base_dir: str | Path,
) -> ResolvedRunConfig:
    """Apply explicit common overrides and resolve one cyclic RunConfig."""

    if options.execution_mode == "execute":
        raise ValueError(
            "The current cyclic pipeline supports only --dry-run; "
            "real execution is not implemented."
        )
    base = Path(base_dir).expanduser().resolve(strict=False)
    loaded = load_run_config(
        options.config_path,
        overrides=options.identity_overrides(),
    )
    resolved = resolve_run_config(loaded, base_dir=base)
    effective = resolved.config
    if options.output_root is not None:
        effective = _with_output_root(
            effective,
            output_root=_resolved_path(options.output_root, base_dir=base),
        )
        resolved = resolve_run_config(effective, base_dir=base)
    _validate_cyclic_profile(resolved.config)
    return resolved


def run_cyclic_pipeline(resolved: ResolvedRunConfig) -> dict[str, Any]:
    """Execute the migrated cyclic vertical slice using component configs only."""

    validate_current_traceability_support(resolved)
    config = resolved.config
    _validate_cyclic_profile(config)
    simulation = config.simulation
    signals = config.signals
    detection = config.detection
    assert simulation is not None
    assert simulation.ingestion is not None
    assert simulation.orchestration is not None
    assert simulation.stateful_adapter is not None
    assert signals is not None
    assert detection is not None
    assert detection.connector is not None

    stages = {
        "ingestion": build_cyclic_ingestion_dry_run(
            simulation.ingestion,
            input_policy=(
                config.data.experimental_input
                if config.data is not None
                else None
            ),
        ),
        "orchestration": run_cyclic_orchestrator_dry_run(
            simulation.orchestration
        ),
        "stateful_adapter": run_cyclic_stateful_adapter(
            simulation.stateful_adapter
        ),
    }
    if detection.connector.mode == ACTIVITY_DETECTION_RUNTIME_MODE:
        assert signals.activity is not None
        assert detection.activity_route is not None
        detector_config = getattr(
            detection,
            detection.activity_route.detector_id,
            None,
        )
        stages["detection_connector"] = run_cyclic_detection_connector(
            detection.connector,
            run_id=config.identity.run_id,
            activity_route=detection.activity_route,
            signal_definition=signals.activity,
            detector_config=detector_config,
            config_hash=resolved.config_hash,
            gap_policy=detection.gap_policy,
        )
    else:
        assert signals.daily is not None
        assert detection.xiao_ema is not None
        assert detection.daily_frequency is not None
        stages["detection_connector"] = run_cyclic_detection_connector(
            detection.connector,
            xiao_config=detection.xiao_ema,
        )
        stages["daily_signals"] = run_cyclic_daily_signals(signals.daily)
        stages["daily_frequency"] = run_daily_frequency_baseline(
            detection.daily_frequency
        )
    run_manifest = write_run_manifest(
        resolved,
        output_dir=simulation.ingestion.output_dir,
        execution_mode="dry_run",
        completed_stages=tuple(stages),
    )
    return {
        "run_id": config.identity.run_id,
        "config_hash": resolved.config_hash,
        "execution_mode": "dry_run",
        "run_manifest": str(run_manifest),
        "stages": stages,
    }


def main(
    argv: Sequence[str] | None = None,
    *,
    base_dir: str | Path | None = None,
) -> None:
    options = parse_common_run_args(
        argv,
        description=(
            "Run the configured cyclic ingestion, state, daily signal, connector, "
            "and daily baseline compatibility flow."
        ),
        prog="cyclic-pipeline",
    )
    logging.basicConfig(level=getattr(logging, options.log_level))
    try:
        resolved = resolve_cyclic_pipeline_run(
            options,
            base_dir=Path.cwd() if base_dir is None else base_dir,
        )
        summary = run_cyclic_pipeline(resolved)
    except (FileNotFoundError, TypeError, ValueError) as exc:
        raise SystemExit(f"cyclic-pipeline: error: {exc}") from exc
    print(json.dumps(summary, indent=2, ensure_ascii=False))


__all__ = [
    "main",
    "resolve_cyclic_pipeline_run",
    "run_cyclic_pipeline",
]


if __name__ == "__main__":
    main()
