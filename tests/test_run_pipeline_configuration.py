from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from youtube_pipeline.detectors import DEFAULT_DETECTOR, XiaoEMAConfig
from youtube_pipeline.entrypoints.prepared_replay import (
    LEGACY_INPUT_PATH,
    LEGACY_OUTPUT_SNAPSHOTS,
    legacy_replay_detector_params,
    resolve_legacy_prepared_replay_run,
)
from youtube_pipeline.run_pipeline import (
    DEFAULT_TRIGGER_COOLDOWN,
    DEFAULT_TRIGGER_MIN_VOLUME,
    DEFAULT_TRIGGER_SLIDE_INTERVAL,
    DEFAULT_TRIGGER_SLOW_WINDOW,
    DEFAULT_TRIGGER_THRESHOLD,
    DEFAULT_TRIGGER_WINDOW_SIZE,
    _build_parser,
)


class RunPipelineConfigurationTests(unittest.TestCase):
    def test_legacy_playback_defaults_resolve_through_run_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            resolved, detector_name = resolve_legacy_prepared_replay_run(
                base_dir=base,
            )

        self.assertEqual(detector_name, DEFAULT_DETECTOR)
        self.assertEqual(
            resolved.config.data.prepared_dataset.path,
            base.resolve() / LEGACY_INPUT_PATH,
        )
        self.assertEqual(
            resolved.config.simulation.replay.output_snapshots,
            base.resolve() / LEGACY_OUTPUT_SNAPSHOTS,
        )
        self.assertEqual(
            resolved.config.detection.xiao_ema,
            XiaoEMAConfig(),
        )
        self.assertEqual(
            legacy_replay_detector_params(resolved, detector_name),
            {
                "ts_col": "event_time_utc",
                "text_col": "text",
                "window_size": "120s",
                "slide_interval": "30s",
                "slow_window": "10min",
                "sensitivity_threshold": 1.5,
                "v_min": 46,
                "cooldown": "3min",
            },
        )

    def test_legacy_detector_precedence_is_preserved_in_resolved_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            detector_config = base / "detector.json"
            detector_config.write_text(
                json.dumps(
                    {
                        "detector": {
                            "name": "xiao_ema",
                            "params": {
                                "window_size": "60s",
                                "sensitivity_threshold": 2.0,
                                "v_min": 50,
                            },
                        }
                    }
                ),
                encoding="utf-8",
            )

            resolved, _ = resolve_legacy_prepared_replay_run(
                detector_config_file=detector_config,
                detector_params={"sensitivity_threshold": 2.25},
                trigger_min_volume=99,
                base_dir=base,
            )

        xiao = resolved.config.detection.xiao_ema
        self.assertEqual(xiao.window_size, "60s")
        self.assertEqual(xiao.sensitivity_threshold, 2.25)
        self.assertEqual(xiao.v_min, 99)

    def test_legacy_parser_defers_methodological_defaults_to_resolver(self) -> None:
        parser = _build_parser()

        playback = parser.parse_args(["playback"])
        clean = parser.parse_args(["clean"])

        for field_name in (
            "input_path",
            "output_snapshots",
            "ts_col",
            "window_size",
            "speed",
            "max_sleep_seconds",
            "detector",
        ):
            self.assertIsNone(getattr(playback, field_name))
        for field_name in (
            "input_path",
            "output_path",
            "raw_text_col",
            "timestamp_col",
            "keep_spam",
        ):
            self.assertIsNone(getattr(clean, field_name))

    def test_exported_trigger_defaults_are_aliases_of_xiao_config(self) -> None:
        xiao = XiaoEMAConfig()

        self.assertEqual(DEFAULT_TRIGGER_THRESHOLD, xiao.sensitivity_threshold)
        self.assertEqual(DEFAULT_TRIGGER_MIN_VOLUME, xiao.v_min)
        self.assertEqual(DEFAULT_TRIGGER_WINDOW_SIZE, xiao.window_size)
        self.assertEqual(DEFAULT_TRIGGER_SLIDE_INTERVAL, xiao.slide_interval)
        self.assertEqual(DEFAULT_TRIGGER_SLOW_WINDOW, xiao.slow_window)
        self.assertEqual(DEFAULT_TRIGGER_COOLDOWN, xiao.cooldown)


if __name__ == "__main__":
    unittest.main()
