from __future__ import annotations

import hashlib
import inspect
import json
import logging
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import ANY, patch

import youtube_pipeline.data_extraction as extraction_module
import youtube_pipeline.entrypoints.youtube_extraction as entrypoint_module
import youtube_pipeline.run_pipeline as legacy_pipeline_module
from youtube_pipeline.configuration import run_config_from_mapping
from youtube_pipeline.entrypoints.youtube_extraction import (
    load_legacy_youtube_extraction_config,
    main,
    resolve_youtube_api_key,
    resolve_youtube_extraction_config,
    resolve_youtube_extraction_run,
    run_resolved_youtube_extraction,
    run_youtube_extraction,
)


class YouTubeExtractionEntrypointTests(unittest.TestCase):
    def test_domain_has_no_cli_config_io_or_environment_secret_lookup(self) -> None:
        source = inspect.getsource(extraction_module)

        self.assertNotIn("import argparse", source)
        self.assertNotIn("load_dotenv", source)
        self.assertNotIn("os.getenv", source)
        self.assertNotIn("os.environ", source)

    def test_legacy_adapter_preserves_existing_defaults(self) -> None:
        config = load_legacy_youtube_extraction_config(None)

        self.assertEqual(
            config.query,
            "elecciones 2026 colombia|presidenciales 2026 colombia|"
            "candidatos colombia 2026 -2022",
        )
        self.assertEqual(config.published_after, "2026-01-31T00:00:00Z")
        self.assertEqual(config.published_before, "2026-04-01T00:00:00Z")
        self.assertEqual(config.min_views, 10_000)
        self.assertEqual(config.min_comments, 100)
        self.assertEqual(config.max_comments, 5_000)
        self.assertEqual(config.max_results, 500)
        self.assertTrue(config.save_legacy_csv)
        self.assertEqual(config.data_root, "data")

    def test_run_profile_and_legacy_config_resolve_equivalently(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            component = {
                "query": "synthetic query",
                "published_after": None,
                "published_before": None,
                "min_views": 50,
                "min_comments": 5,
                "max_comments": 20,
                "max_results": 3,
                "save_legacy_csv": False,
                "data_root": "outputs/data",
                "metadata_path": "outputs/metadata/run.json",
            }
            profile = base / "profile.json"
            legacy = base / "legacy.json"
            profile.write_text(
                json.dumps(
                    {
                        "identity": {"run_id": "youtube_profile"},
                        "data": {"youtube_api": component},
                    }
                ),
                encoding="utf-8",
            )
            legacy.write_text(json.dumps(component), encoding="utf-8")

            common_config = resolve_youtube_extraction_config(
                config_file=profile,
                base_dir=base,
            )
            legacy_config = resolve_youtube_extraction_config(
                config_file=legacy,
                base_dir=base,
            )

        self.assertEqual(common_config, legacy_config)
        resolved_base = base.resolve()
        self.assertEqual(
            common_config.data_root,
            str(resolved_base / "outputs/data"),
        )
        self.assertEqual(
            common_config.metadata_path,
            str(resolved_base / "outputs/metadata/run.json"),
        )

    def test_secret_is_resolved_only_in_entrypoint(self) -> None:
        with (
            patch.dict(os.environ, {"YOUTUBE_API_KEY": "infrastructure-secret"}),
            patch.object(entrypoint_module, "load_dotenv") as load_environment,
        ):
            api_key = resolve_youtube_api_key()

        load_environment.assert_called_once_with()
        self.assertEqual(api_key, "infrastructure-secret")

    def test_secret_cannot_be_stored_in_methodological_config(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown data.youtube_api fields"):
            run_config_from_mapping(
                {
                    "identity": {"run_id": "secret_rejected"},
                    "data": {
                        "youtube_api": {
                            "api_key": "must-not-enter-run-config",
                        }
                    },
                }
            )

    def test_entrypoint_injects_secret_into_domain_component(self) -> None:
        config = load_legacy_youtube_extraction_config(None)
        logger = logging.getLogger("test.youtube.entrypoint")
        expected = {"videos_found": 0, "comments_found": 0}

        with (
            patch.object(
                entrypoint_module,
                "resolve_youtube_api_key",
                return_value="infrastructure-secret",
            ),
            patch.object(
                entrypoint_module,
                "run_extraction_pipeline",
                return_value=expected,
            ) as domain_run,
        ):
            result = run_youtube_extraction(config, logger)

        self.assertEqual(result, expected)
        domain_run.assert_called_once_with(
            config,
            logger,
            api_key="infrastructure-secret",
        )

    def test_resolved_run_enriches_primary_metadata_without_replacing_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            profile = base / "profile.json"
            profile.write_text(
                json.dumps(
                    {
                        "identity": {"run_id": "acquisition_trace"},
                        "data": {
                            "youtube_api": {
                                "query": "synthetic query",
                                "data_root": "outputs/data",
                                "metadata_path": "outputs/metadata/run.json",
                                "save_legacy_csv": False,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            resolved = resolve_youtube_extraction_run(
                config_file=profile,
                base_dir=base,
            )
            metadata_path = base / "outputs" / "metadata" / "run.json"
            metadata_path.parent.mkdir(parents=True)
            original_metadata = {
                "run_at_utc": "2026-08-17T12:00:00+00:00",
                "config": {"query": "synthetic query"},
                "quota_hit": False,
                "quota_stage": None,
                "errors": [],
                "persisted": {"comments_jsonl": "outputs/data/comments.jsonl"},
            }
            metadata_path.write_text(
                json.dumps(original_metadata),
                encoding="utf-8",
            )
            summary = {
                "videos_found": 2,
                "comments_found": 3,
                "run_metadata": str(metadata_path),
            }
            logger = logging.getLogger("test.youtube.resolved")

            with patch.object(
                entrypoint_module,
                "run_youtube_extraction",
                return_value=summary,
            ) as acquisition:
                result = run_resolved_youtube_extraction(resolved, logger)

            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        self.assertIs(result, summary)
        acquisition.assert_called_once_with(
            resolved.config.data.youtube_api,
            logger,
        )
        for key, value in original_metadata.items():
            self.assertEqual(metadata[key], value)
        self.assertEqual(metadata["run_id"], "acquisition_trace")
        self.assertEqual(metadata["run_mode"], "development")
        self.assertEqual(metadata["trace_level"], "minimal")
        self.assertEqual(metadata["config_hash"], resolved.config_hash)
        self.assertEqual(
            metadata["resolved_config"]["data"]["youtube_api"]["data_root"],
            "outputs/data",
        )
        self.assertEqual(
            metadata["resolved_config"]["data"]["youtube_api"]["metadata_path"],
            "outputs/metadata/run.json",
        )
        canonical = json.dumps(
            metadata["resolved_config"],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        self.assertEqual(
            hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
            metadata["config_hash"],
        )
        resolved_serialized = json.dumps(metadata["resolved_config"])
        self.assertNotIn(str(base.resolve()), resolved_serialized)
        self.assertNotIn("api_key", json.dumps(metadata).lower())

    def test_dry_run_does_not_resolve_secret_or_create_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            with (
                patch.object(entrypoint_module, "resolve_youtube_api_key") as secret,
                patch.object(entrypoint_module, "run_extraction_pipeline") as domain_run,
            ):
                main(["--dry-run"], base_dir=base)

            secret.assert_not_called()
            domain_run.assert_not_called()
            self.assertFalse((base / "data").exists())

    def test_legacy_pipeline_extract_uses_common_resolver(self) -> None:
        config = load_legacy_youtube_extraction_config(
            None,
            overrides={"query": "legacy wrapper query"},
        )
        expected = {"videos_found": 2, "comments_found": 3}

        with (
            patch.object(
                legacy_pipeline_module,
                "resolve_youtube_extraction_config",
                return_value=config,
            ) as resolver,
            patch.object(
                legacy_pipeline_module,
                "resolve_youtube_api_key",
                return_value="infrastructure-secret",
            ),
            patch.object(
                legacy_pipeline_module,
                "run_extraction_pipeline",
                return_value=expected,
            ) as domain_run,
        ):
            result = legacy_pipeline_module.run_extract(
                query="legacy wrapper query",
                save_legacy_csv=False,
            )

        self.assertEqual(result, expected)
        resolver.assert_called_once_with(
            config_file=None,
            overrides={
                "query": "legacy wrapper query",
                "save_legacy_csv": False,
            },
            base_dir=Path.cwd(),
        )
        domain_run.assert_called_once_with(
            config,
            ANY,
            api_key="infrastructure-secret",
        )


if __name__ == "__main__":
    unittest.main()
