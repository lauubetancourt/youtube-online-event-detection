"""Guarded standalone classifier-to-polarization runtime entrypoint."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Sequence

from youtube_pipeline.configuration import load_run_config, resolve_run_config
from youtube_pipeline.entrypoints.common_cli import build_common_run_parser
from youtube_pipeline.polarization_runtime import (
    prepare_polarization_runtime,
    run_polarization_runtime,
)


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_common_run_parser(
        description="Classify an explicit real-comment slice and emit temporal polarization signals."
    )
    args = parser.parse_args(argv)
    config = load_run_config(args.config_path, overrides=(
        {
            "identity": {"run_id": args.run_id},
            "signals": {"polarization": {"classification_run_id": args.run_id}},
        }
        if args.run_id is not None else None
    ))
    resolved = resolve_run_config(config, base_dir=args.config_path.parent)
    plan = prepare_polarization_runtime(resolved.config)
    temporal = resolved.config.signals.polarization  # type: ignore[union-attr]
    classifier = resolved.config.polarization.classifier  # type: ignore[union-attr]
    prepared = {
        "execution_status": "prepared_not_executed",
        "run_id": resolved.config.identity.run_id,
        "config_hash": resolved.config_hash,
        "input_path": str(plan.comments_path),
        "input_rows": len(plan.comments),
        "video_ids": list(plan.selected_video_ids),
        "selected_time_min_utc": plan.selected_time_min_utc.isoformat(),
        "selected_time_max_utc": plan.selected_time_max_utc.isoformat(),
        "expected_classifier_calls": plan.expected_classifier_calls,
        "proposition": {
            "id": classifier.proposition.proposition_id,
            "text": classifier.proposition.proposition_text,
            "context": classifier.proposition.context,
        },
        "model": classifier.model,
        "prompt_id": classifier.prompt_id,
        "prompt_version": classifier.prompt_version,
        "window": temporal.window,
        "cadence": temporal.cadence,
        "enabled_measures": list(temporal.enabled_measures),
        "methodological_status": "RUNTIME_SMOKE_NOT_CALIBRATED",
    }
    if args.execution_mode != "execute":
        print(json.dumps(prepared, ensure_ascii=False))
        return
    if args.output_root is None:
        parser.error("--execute requires an explicit --output-root.")
    artifacts = asyncio.run(run_polarization_runtime(
        config, output_dir=args.output_root, base_dir=args.config_path.parent,
    ))
    print(json.dumps({
        **prepared,
        "execution_status": "executed",
        "stance_artifact": str(artifacts.stance_path),
        "distribution_artifact": str(artifacts.distribution_path),
        "signal_artifact": str(artifacts.signal_path),
        "manifest": str(artifacts.manifest_path),
        "ticks": artifacts.tick_count,
        "measure_observations": artifacts.observation_count,
        "detection_results": artifacts.detection_result_count,
        "detection_triggers": artifacts.detection_trigger_count,
        "detection_alerts": artifacts.detection_alert_count,
        "detection_artifact": (
            str(artifacts.detection_result_path)
            if artifacts.detection_result_path is not None else None
        ),
        "detection_manifest": (
            str(artifacts.detection_manifest_path)
            if artifacts.detection_manifest_path is not None else None
        ),
        "detection_alert_artifact": (
            str(artifacts.detection_alert_path)
            if artifacts.detection_alert_path is not None else None
        ),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
