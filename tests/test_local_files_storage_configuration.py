from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from youtube_pipeline.configuration import (
    DataConfig,
    RunConfig,
    RunIdentityConfig,
    resolve_run_config,
    run_config_from_mapping,
)
from youtube_pipeline.data_extraction import ExtractionConfig
from youtube_pipeline.entrypoints.local_files_storage import (
    LEGACY_DATA_ROOT,
    load_legacy_local_files_config,
    resolve_local_files_config,
)
from youtube_pipeline.storage import LocalFilesConfig


class LocalFilesStorageConfigurationTests(unittest.TestCase):
    def test_run_config_reuses_local_files_component(self) -> None:
        local_files = LocalFilesConfig(
            videos_path="inputs/videos.csv",
            comments_path="inputs/comments.csv",
            data_root="outputs/data",
        )

        run = RunConfig(
            identity=RunIdentityConfig(run_id="local_files_run"),
            data=DataConfig(local_files=local_files),
        )

        self.assertIs(run.data.local_files, local_files)
        self.assertIsNone(run.data.youtube_api)

    def test_data_config_requires_exactly_one_source(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly one data source"):
            DataConfig()

        with self.assertRaisesRegex(ValueError, "exactly one data source"):
            DataConfig(
                youtube_api=ExtractionConfig(),
                local_files=LocalFilesConfig(
                    videos_path="videos.csv",
                    comments_path="comments.csv",
                ),
            )

    def test_loader_is_strict_for_local_files_contract(self) -> None:
        run = run_config_from_mapping(
            {
                "identity": {"run_id": "local_files_mapping"},
                "data": {
                    "local_files": {
                        "videos_path": "inputs/videos.csv",
                        "comments_path": "inputs/comments.parquet",
                        "data_root": "outputs/data",
                    }
                },
            }
        )

        self.assertIsInstance(run.data.local_files, LocalFilesConfig)
        self.assertEqual(run.data.local_files.data_root, "outputs/data")

        with self.assertRaisesRegex(ValueError, "Unknown local files config fields"):
            run_config_from_mapping(
                {
                    "identity": {"run_id": "local_files_unknown"},
                    "data": {
                        "local_files": {
                            "videos_path": "videos.csv",
                            "comments_path": "comments.csv",
                            "copy_dataset": True,
                        }
                    },
                }
            )

    def test_resolves_all_local_file_paths_without_requiring_outputs(self) -> None:
        run = run_config_from_mapping(
            {
                "identity": {"run_id": "local_files_paths"},
                "data": {
                    "local_files": {
                        "videos_path": "inputs/videos.csv",
                        "comments_path": "inputs/comments.parquet",
                        "data_root": "outputs/data",
                    }
                },
            }
        )

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp).resolve()
            resolved = resolve_run_config(run, base_dir=base)

        config = resolved.config.data.local_files
        self.assertEqual(config.videos_path, base / "inputs/videos.csv")
        self.assertEqual(config.comments_path, base / "inputs/comments.parquet")
        self.assertEqual(config.data_root, base / "outputs/data")
        mapping = json.loads(resolved.canonical_json)
        self.assertEqual(
            mapping["data"]["local_files"]["videos_path"],
            "inputs/videos.csv",
        )
        self.assertEqual(
            mapping["data"]["local_files"]["comments_path"],
            "inputs/comments.parquet",
        )
        self.assertEqual(
            mapping["data"]["local_files"]["data_root"],
            "outputs/data",
        )

    def test_common_profile_and_legacy_shape_resolve_equivalently(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            component = {
                "videos_path": "inputs/videos.csv",
                "comments_path": "inputs/comments.csv",
                "data_root": "outputs/data",
            }
            profile = base / "profile.json"
            legacy = base / "legacy.json"
            profile.write_text(
                json.dumps(
                    {
                        "identity": {"run_id": "local_files_profile"},
                        "data": {"local_files": component},
                    }
                ),
                encoding="utf-8",
            )
            legacy.write_text(json.dumps(component), encoding="utf-8")

            common_config = resolve_local_files_config(
                config_file=profile,
                base_dir=base,
            )
            legacy_config = resolve_local_files_config(
                config_file=legacy,
                base_dir=base,
            )

        self.assertEqual(common_config, legacy_config)

    def test_legacy_adapter_preserves_data_root_default(self) -> None:
        config = load_legacy_local_files_config(
            None,
            overrides={
                "videos_path": "videos.csv",
                "comments_path": "comments.csv",
            },
        )

        self.assertEqual(config.data_root, LEGACY_DATA_ROOT)


if __name__ == "__main__":
    unittest.main()
