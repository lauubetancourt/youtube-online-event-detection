from __future__ import annotations

import inspect
import json
import tempfile
import unittest
from pathlib import Path

import youtube_pipeline.daily_rag_sidecars as daily_rag_sidecars_module
from youtube_pipeline.configuration import (
    canonical_run_config_json,
    resolve_run_config,
    run_config_from_mapping,
)
from youtube_pipeline.daily_rag_sidecars import DailyRagSidecarBuildConfig
from youtube_pipeline.entrypoints.daily_rag_sidecars import (
    LEGACY_COMMENTS_PATH,
    LEGACY_DAILY_EVENTS_PATH,
    LEGACY_OUTPUT_DIR,
    LEGACY_WINDOW_INVENTORY_PATH,
    load_legacy_daily_rag_sidecar_config,
    resolve_daily_rag_sidecar_config,
)


class DailyRagSidecarConfigurationTests(unittest.TestCase):
    def test_domain_requires_paths_and_contains_no_historical_defaults(self) -> None:
        with self.assertRaises(TypeError):
            DailyRagSidecarBuildConfig()

        source = inspect.getsource(daily_rag_sidecars_module)
        self.assertNotIn(LEGACY_DAILY_EVENTS_PATH, source)
        self.assertNotIn(LEGACY_OUTPUT_DIR, source)
        self.assertNotIn("import argparse", source)

    def test_run_profile_keeps_global_and_sidecar_identities_distinct(self) -> None:
        run = run_config_from_mapping(
            {
                "identity": {"run_id": "run_daily_sidecars"},
                "rag": {
                    "daily_sidecars": {
                        "daily_events_path": "events/events.jsonl",
                        "output_dir": "outputs/sidecars",
                        "comments_path": "prepared/comments.parquet",
                        "cycle_window_inventory_path": "cycles/window.csv",
                        "daily_scores_path": None,
                    }
                },
            }
        )

        self.assertIsNotNone(run.rag)
        self.assertIsInstance(
            run.rag.daily_sidecars,
            DailyRagSidecarBuildConfig,
        )
        self.assertIsNone(run.rag.daily_sidecars.run_id)
        mapping = json.loads(canonical_run_config_json(run))
        self.assertIsNone(mapping["rag"]["daily_sidecars"]["run_id"])

    def test_resolves_all_sidecar_paths_without_requiring_files(self) -> None:
        run = run_config_from_mapping(
            {
                "identity": {"run_id": "run_paths"},
                "rag": {
                    "daily_sidecars": {
                        "daily_events_path": "events.jsonl",
                        "output_dir": "out",
                        "comments_path": "comments.parquet",
                        "cycle_window_inventory_path": "window.csv",
                        "daily_scores_path": "scores.jsonl",
                        "daily_detector_manifest_path": "manifest.json",
                        "cycle_signal_series_path": "signals.jsonl",
                        "cycle_stateful_context_path": "state.json",
                    }
                },
            }
        )

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp).resolve()
            component = resolve_run_config(run, base_dir=base).config.rag.daily_sidecars

        self.assertEqual(component.daily_events_path, base / "events.jsonl")
        self.assertEqual(component.output_dir, base / "out")
        self.assertEqual(component.comments_path, base / "comments.parquet")
        self.assertEqual(
            component.cycle_window_inventory_path,
            base / "window.csv",
        )
        self.assertEqual(component.daily_scores_path, base / "scores.jsonl")
        self.assertEqual(
            component.daily_detector_manifest_path,
            base / "manifest.json",
        )
        self.assertEqual(component.cycle_signal_series_path, base / "signals.jsonl")
        self.assertEqual(
            component.cycle_stateful_context_path,
            base / "state.json",
        )

    def test_common_profile_and_legacy_adapter_resolve_equivalent_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            component = {
                "daily_events_path": "events.jsonl",
                "output_dir": "out",
                "comments_path": "comments.parquet",
                "cycle_window_inventory_path": "window.csv",
                "daily_scores_path": None,
                "daily_detector_manifest_path": None,
                "cycle_signal_series_path": None,
                "cycle_stateful_context_path": None,
                "max_comments_per_context_unit": 7,
            }
            profile_path = base / "profile.json"
            legacy_path = base / "legacy.json"
            profile_path.write_text(
                json.dumps(
                    {
                        "identity": {"run_id": "run_equivalent"},
                        "rag": {
                            "daily_sidecars": {
                                **component,
                                "run_id": "run_equivalent",
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            legacy_path.write_text(
                json.dumps(
                    {
                        "daily_rag_sidecars": {
                            **component,
                            "run_id": "run_equivalent",
                        }
                    }
                ),
                encoding="utf-8",
            )

            common = resolve_daily_rag_sidecar_config(
                config_file=profile_path,
                base_dir=base,
            )
            legacy = resolve_daily_rag_sidecar_config(
                config_file=legacy_path,
                base_dir=base,
            )

        self.assertEqual(common, legacy)

    def test_legacy_adapter_preserves_previous_defaults_and_derived_run_id(self) -> None:
        config = load_legacy_daily_rag_sidecar_config(None)

        self.assertEqual(config.daily_events_path, LEGACY_DAILY_EVENTS_PATH)
        self.assertEqual(config.output_dir, LEGACY_OUTPUT_DIR)
        self.assertEqual(config.comments_path, LEGACY_COMMENTS_PATH)
        self.assertEqual(
            config.cycle_window_inventory_path,
            LEGACY_WINDOW_INVENTORY_PATH,
        )
        self.assertTrue(config.run_id.startswith("drun_"))

    def test_component_run_id_is_independent_from_run_identity(self) -> None:
        run = run_config_from_mapping(
            {
                "identity": {"run_id": "global_run"},
                "rag": {
                    "daily_sidecars": {
                        "daily_events_path": "events.jsonl",
                        "output_dir": "out",
                        "comments_path": "comments.parquet",
                        "cycle_window_inventory_path": "window.csv",
                        "run_id": "drun_stage",
                    }
                },
            }
        )

        self.assertEqual(run.identity.run_id, "global_run")
        self.assertEqual(run.rag.daily_sidecars.run_id, "drun_stage")
        mapping = json.loads(canonical_run_config_json(run))
        self.assertEqual(mapping["rag"]["daily_sidecars"]["run_id"], "drun_stage")


if __name__ == "__main__":
    unittest.main()
