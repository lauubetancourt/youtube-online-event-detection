from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import fields
from pathlib import Path

from youtube_pipeline.configuration import (
    DataConfig,
    RunConfig,
    RunIdentityConfig,
    SimulationConfig,
    resolve_run_config,
    run_config_from_mapping,
)
from youtube_pipeline.entrypoints.prepared_replay import (
    LEGACY_INPUT_PATH,
    LEGACY_OUTPUT_SNAPSHOTS,
    load_legacy_prepared_replay_configs,
    resolve_prepared_replay_configs,
)
from youtube_pipeline.prepared_replay import (
    PreparedDatasetConfig,
    ReplayConfig,
)
from youtube_pipeline.storage import LocalFilesConfig


class PreparedReplayConfigurationTests(unittest.TestCase):
    def test_run_config_reuses_prepared_dataset_and_replay_components(self) -> None:
        dataset = PreparedDatasetConfig(path="prepared/comments.parquet")
        replay = ReplayConfig(output_snapshots="outputs/snapshots.csv")

        run = RunConfig(
            identity=RunIdentityConfig(run_id="prepared_replay"),
            data=DataConfig(prepared_dataset=dataset),
            simulation=SimulationConfig(replay=replay),
        )

        self.assertIs(run.data.prepared_dataset, dataset)
        self.assertIs(run.simulation.replay, replay)
        self.assertIsNone(run.data.youtube_api)
        self.assertIsNone(run.data.local_files)

    def test_prepared_dataset_is_an_exclusive_data_source(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly one data source"):
            DataConfig(
                prepared_dataset=PreparedDatasetConfig(
                    path="prepared/comments.parquet"
                ),
                local_files=LocalFilesConfig(
                    videos_path="inputs/videos.csv",
                    comments_path="inputs/comments.csv",
                ),
            )

    def test_loader_is_strict_and_keeps_component_defaults_authoritative(self) -> None:
        run = run_config_from_mapping(
            {
                "identity": {"run_id": "prepared_replay_mapping"},
                "data": {
                    "prepared_dataset": {"path": "prepared/comments.parquet"}
                },
                "simulation": {
                    "replay": {"output_snapshots": "outputs/snapshots.csv"}
                },
            }
        )

        self.assertEqual(
            run.data.prepared_dataset.timestamp_column,
            "event_time_utc",
        )
        self.assertEqual(run.simulation.replay.window_size, "20min")
        self.assertEqual(run.simulation.replay.speed, 120.0)
        self.assertEqual(run.simulation.replay.max_sleep_seconds, 0.2)

        with self.assertRaisesRegex(
            ValueError,
            "Unknown prepared dataset config fields",
        ):
            run_config_from_mapping(
                {
                    "identity": {"run_id": "prepared_unknown"},
                    "data": {
                        "prepared_dataset": {
                            "path": "prepared/comments.parquet",
                            "fingerprint": "out-of-scope",
                        }
                    },
                }
            )

        with self.assertRaisesRegex(ValueError, "Unknown replay config fields"):
            run_config_from_mapping(
                {
                    "identity": {"run_id": "replay_unknown"},
                    "simulation": {
                        "replay": {
                            "output_snapshots": "outputs/snapshots.csv",
                            "change_algorithm": True,
                        }
                    },
                }
            )

    def test_resolves_paths_and_preserves_logical_canonical_paths(self) -> None:
        run = run_config_from_mapping(
            {
                "identity": {"run_id": "prepared_replay_paths"},
                "data": {
                    "prepared_dataset": {"path": "prepared/comments.parquet"}
                },
                "simulation": {
                    "replay": {"output_snapshots": "outputs/snapshots.csv"}
                },
            }
        )

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp).resolve()
            resolved = resolve_run_config(run, base_dir=base)

        self.assertEqual(
            resolved.config.data.prepared_dataset.path,
            base / "prepared/comments.parquet",
        )
        self.assertEqual(
            resolved.config.simulation.replay.output_snapshots,
            base / "outputs/snapshots.csv",
        )
        mapping = json.loads(resolved.canonical_json)
        self.assertEqual(
            mapping["data"]["prepared_dataset"]["path"],
            "prepared/comments.parquet",
        )
        self.assertEqual(
            mapping["simulation"]["replay"]["output_snapshots"],
            "outputs/snapshots.csv",
        )

    def test_common_profile_and_legacy_shape_resolve_equivalently(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            legacy_payload = {
                "input_path": "prepared/comments.parquet",
                "output_snapshots": "outputs/snapshots.csv",
                "ts_col": "observed_at",
                "window_size": "15min",
                "speed": 60.0,
                "max_sleep_seconds": 0.5,
                "start": "2026-01-01T00:00:00Z",
                "end": "2026-01-02T00:00:00Z",
            }
            profile_path = base / "profile.json"
            legacy_path = base / "legacy.json"
            profile_path.write_text(
                json.dumps(
                    {
                        "identity": {"run_id": "prepared_replay_profile"},
                        "data": {
                            "prepared_dataset": {
                                "path": legacy_payload["input_path"],
                                "timestamp_column": legacy_payload["ts_col"],
                            }
                        },
                        "simulation": {
                            "replay": {
                                field_name: legacy_payload[field_name]
                                for field_name in (
                                    "output_snapshots",
                                    "window_size",
                                    "speed",
                                    "max_sleep_seconds",
                                    "start",
                                    "end",
                                )
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            legacy_path.write_text(json.dumps(legacy_payload), encoding="utf-8")

            profile_configs = resolve_prepared_replay_configs(
                config_file=profile_path,
                base_dir=base,
            )
            legacy_configs = resolve_prepared_replay_configs(
                config_file=legacy_path,
                base_dir=base,
            )

        self.assertEqual(profile_configs, legacy_configs)

    def test_composition_does_not_duplicate_component_parameters(self) -> None:
        dataset_fields = {field.name for field in fields(PreparedDatasetConfig)}
        replay_fields = {field.name for field in fields(ReplayConfig)}

        self.assertTrue(
            {field.name for field in fields(DataConfig)}.isdisjoint(dataset_fields)
        )
        self.assertTrue(
            {field.name for field in fields(SimulationConfig)}.isdisjoint(
                replay_fields
            )
        )

    def test_legacy_adapter_preserves_current_paths_and_defaults(self) -> None:
        dataset, replay = load_legacy_prepared_replay_configs(None)

        self.assertEqual(dataset.path, LEGACY_INPUT_PATH)
        self.assertEqual(dataset.timestamp_column, "event_time_utc")
        self.assertEqual(replay.output_snapshots, LEGACY_OUTPUT_SNAPSHOTS)
        self.assertEqual(replay.window_size, "20min")
        self.assertEqual(replay.speed, 120.0)
        self.assertEqual(replay.max_sleep_seconds, 0.2)


if __name__ == "__main__":
    unittest.main()
