from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from youtube_pipeline.cleaning import CleaningConfig
from youtube_pipeline.configuration import (
    DataConfig,
    RunConfig,
    RunIdentityConfig,
    resolve_run_config,
    run_config_from_mapping,
)
from youtube_pipeline.entrypoints.cleaning import (
    LEGACY_INPUT_PATH,
    LEGACY_OUTPUT_PATH,
    load_legacy_cleaning_config,
    resolve_cleaning_config,
)
from youtube_pipeline.storage import LocalFilesConfig


class CleaningConfigurationTests(unittest.TestCase):
    def test_run_config_reuses_cleaning_component_without_source_duplication(self) -> None:
        cleaning = CleaningConfig(
            input_path="data/silver/comments",
            output_path="data/gold/clean_comments.parquet",
        )

        preparation_only = RunConfig(
            identity=RunIdentityConfig(run_id="cleaning_only"),
            data=DataConfig(cleaning=cleaning),
        )
        local_flow = RunConfig(
            identity=RunIdentityConfig(run_id="local_then_clean"),
            data=DataConfig(
                local_files=LocalFilesConfig(
                    videos_path="inputs/videos.csv",
                    comments_path="inputs/comments.csv",
                ),
                cleaning=cleaning,
            ),
        )

        self.assertIs(preparation_only.data.cleaning, cleaning)
        self.assertIs(local_flow.data.cleaning, cleaning)
        self.assertIsNotNone(local_flow.data.local_files)
        self.assertIsNone(local_flow.data.youtube_api)

    def test_loader_is_strict_and_preserves_cleaning_defaults(self) -> None:
        run = run_config_from_mapping(
            {
                "identity": {"run_id": "cleaning_mapping"},
                "data": {
                    "cleaning": {
                        "input_path": "silver/comments",
                        "output_path": "gold/comments.parquet",
                    }
                },
            }
        )

        config = run.data.cleaning
        self.assertIsInstance(config, CleaningConfig)
        self.assertEqual(config.raw_text_col, "text")
        self.assertEqual(config.timestamp_col, "published_at")
        self.assertFalse(config.keep_spam)

        with self.assertRaisesRegex(ValueError, "Unknown cleaning config fields"):
            run_config_from_mapping(
                {
                    "identity": {"run_id": "cleaning_unknown"},
                    "data": {
                        "cleaning": {
                            "input_path": "silver/comments",
                            "output_path": "gold/comments.parquet",
                            "change_schema": True,
                        }
                    },
                }
            )

    def test_resolves_cleaning_paths_and_serializes_logical_paths(self) -> None:
        run = run_config_from_mapping(
            {
                "identity": {"run_id": "cleaning_paths"},
                "data": {
                    "cleaning": {
                        "input_path": "silver/comments",
                        "output_path": "gold/comments.parquet",
                    }
                },
            }
        )

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp).resolve()
            resolved = resolve_run_config(run, base_dir=base)

        config = resolved.config.data.cleaning
        self.assertEqual(config.input_path, base / "silver/comments")
        self.assertEqual(config.output_path, base / "gold/comments.parquet")
        mapping = json.loads(resolved.canonical_json)
        self.assertEqual(
            mapping["data"]["cleaning"]["input_path"],
            "silver/comments",
        )
        self.assertEqual(
            mapping["data"]["cleaning"]["output_path"],
            "gold/comments.parquet",
        )

    def test_common_profile_and_legacy_shape_resolve_equivalently(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            component = {
                "input_path": "silver/comments",
                "output_path": "gold/comments.parquet",
                "raw_text_col": "body",
                "timestamp_col": "created_at",
                "keep_spam": True,
            }
            profile = base / "profile.json"
            legacy = base / "legacy.json"
            profile.write_text(
                json.dumps(
                    {
                        "identity": {"run_id": "cleaning_profile"},
                        "data": {"cleaning": component},
                    }
                ),
                encoding="utf-8",
            )
            legacy.write_text(json.dumps(component), encoding="utf-8")

            common_config = resolve_cleaning_config(
                config_file=profile,
                base_dir=base,
            )
            legacy_config = resolve_cleaning_config(
                config_file=legacy,
                base_dir=base,
            )

        self.assertEqual(common_config, legacy_config)

    def test_legacy_adapter_preserves_current_paths(self) -> None:
        config = load_legacy_cleaning_config(None)

        self.assertEqual(config.input_path, LEGACY_INPUT_PATH)
        self.assertEqual(config.output_path, LEGACY_OUTPUT_PATH)


if __name__ == "__main__":
    unittest.main()
