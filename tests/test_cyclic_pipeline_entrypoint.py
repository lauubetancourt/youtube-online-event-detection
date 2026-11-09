from __future__ import annotations

import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from youtube_pipeline.entrypoints.common_cli import CommonRunCliOptions
from youtube_pipeline.entrypoints.cyclic_pipeline import (
    main,
    resolve_cyclic_pipeline_run,
)


def _write_fixture(base: Path) -> Path:
    comments = []
    for day, count in [(1, 2), (3, 6)]:
        for index in range(count):
            comments.append(
                {
                    "comment_id": f"c{day}_{index}",
                    "video_id": "video_1",
                    "event_time_utc": f"2026-06-{day:02d}T12:{index:02d}:00Z",
                    "text": f"synthetic comment {day}-{index}",
                }
            )
    pd.DataFrame(comments).to_parquet(base / "comments.parquet", index=False)

    config_path = base / "cyclic.json"
    config_path.write_text(
        json.dumps(
            {
                "identity": {"run_id": "profile_run"},
                "simulation": {
                    "ingestion": {
                        "input_path": "comments.parquet",
                        "output_dir": "profile_output",
                        "collection_start_date_local": "2026-06-01",
                        "collection_end_date_local": "2026-06-03",
                        "analysis_window_size_days": 2,
                        "simulation_run_id": "sim_vertical",
                    },
                    "orchestration": {"simulation_dir": "profile_output"},
                    "stateful_adapter": {"simulation_dir": "profile_output"},
                },
                "signals": {
                    "daily": {
                        "simulation_dir": "profile_output",
                        "canonical_dataset_path": "comments.parquet",
                    }
                },
                "detection": {
                    "activity_route": {
                        "signal_id": "comment_count_event_window_120s_step_30s",
                        "detector_id": "xiao_ema",
                    },
                    "connector": {
                        "simulation_dir": "profile_output",
                        "canonical_dataset_path": "comments.parquet",
                        "max_cycles": 3,
                    },
                    "xiao_ema": {},
                    "daily_frequency": {
                        "simulation_dir": "profile_output",
                        "output_dir": "profile_output/baseline",
                        "signal_name": "active_window_comment_count",
                        "baseline_window_size_cycles": 2,
                        "warmup_cycles": 2,
                        "k_multiplier": 2.0,
                        "min_count": 5,
                        "min_delta": 3,
                        "min_pct_change": 1.0,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return config_path


class CyclicPipelineEntrypointTests(unittest.TestCase):
    def test_common_cli_runs_the_vertical_slice_with_explicit_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            config_path = _write_fixture(base)
            stdout = io.StringIO()

            with contextlib.redirect_stdout(stdout):
                main(
                    [
                        "--config",
                        str(config_path),
                        "--run-id",
                        "cli_run",
                        "--output-root",
                        "relocated",
                        "--dry-run",
                    ],
                    base_dir=base,
                )

            summary = json.loads(stdout.getvalue())
            relocated = base / "relocated"
            self.assertEqual(summary["run_id"], "cli_run")
            self.assertEqual(len(summary["config_hash"]), 64)
            self.assertEqual(summary["execution_mode"], "dry_run")
            self.assertEqual(
                summary["run_manifest"],
                str(relocated.resolve() / "run_manifest.json"),
            )
            self.assertEqual(summary["stages"]["ingestion"]["cycles_total"], 3)
            self.assertEqual(
                summary["stages"]["detection_connector"]["events_detected_count"],
                0,
            )
            self.assertEqual(
                summary["stages"]["daily_signals"]["processed_cycle_count"],
                3,
            )
            self.assertEqual(
                summary["stages"]["daily_frequency"]["events_detected"],
                1,
            )
            self.assertTrue((relocated / "cycle_manifest.jsonl").is_file())
            self.assertTrue(
                (relocated / "baseline" / "cycle_daily_frequency_events.jsonl").is_file()
            )
            run_manifest = json.loads(
                (relocated / "run_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(run_manifest["schema_version"], "1")
            self.assertEqual(run_manifest["run_id"], "cli_run")
            self.assertEqual(run_manifest["status"], "completed")
            self.assertEqual(run_manifest["execution_mode"], "dry_run")
            self.assertEqual(run_manifest["run_mode"], "development")
            self.assertEqual(run_manifest["trace_level"], "minimal")
            self.assertEqual(run_manifest["config_hash"], summary["config_hash"])
            self.assertEqual(
                run_manifest["completed_stages"],
                [
                    "ingestion",
                    "orchestration",
                    "stateful_adapter",
                    "detection_connector",
                    "daily_signals",
                    "daily_frequency",
                ],
            )
            resolved_config = run_manifest["resolved_config"]
            self.assertEqual(resolved_config["identity"]["run_id"], "cli_run")
            self.assertEqual(
                resolved_config["simulation"]["ingestion"]["input_path"],
                "comments.parquet",
            )
            self.assertEqual(
                resolved_config["simulation"]["ingestion"]["output_dir"],
                "relocated",
            )
            self.assertEqual(
                resolved_config["detection"]["activity_route"],
                {
                    "signal_id": "comment_count_event_window_120s_step_30s",
                    "detector_id": "xiao_ema",
                },
            )
            canonical = json.dumps(
                resolved_config,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            self.assertEqual(
                hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
                run_manifest["config_hash"],
            )
            self.assertNotIn(str(base), json.dumps(run_manifest))
            self.assertFalse((base / "profile_output").exists())

    def test_run_id_override_changes_only_the_resolved_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            config_path = _write_fixture(base)
            profile = resolve_cyclic_pipeline_run(
                CommonRunCliOptions(config_path=config_path),
                base_dir=base,
            )
            overridden = resolve_cyclic_pipeline_run(
                CommonRunCliOptions(
                    config_path=config_path,
                    run_id="override_run",
                ),
                base_dir=base,
            )

            self.assertEqual(profile.config.identity.run_id, "profile_run")
            self.assertEqual(overridden.config.identity.run_id, "override_run")
            self.assertNotEqual(profile.config_hash, overridden.config_hash)
            self.assertEqual(profile.config.simulation, overridden.config.simulation)
            self.assertEqual(profile.config.signals, overridden.config.signals)
            self.assertEqual(profile.config.detection, overridden.config.detection)

    def test_execute_fails_before_creating_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            config_path = _write_fixture(base)
            options = CommonRunCliOptions(
                config_path=config_path,
                execution_mode="execute",
            )

            with self.assertRaisesRegex(ValueError, "supports only --dry-run"):
                resolve_cyclic_pipeline_run(options, base_dir=base)

            self.assertFalse((base / "profile_output").exists())


if __name__ == "__main__":
    unittest.main()
