from __future__ import annotations

import inspect
import json
import tempfile
import unittest
from pathlib import Path

import youtube_pipeline.daily_rag_consumer as daily_rag_consumer_module
from tests.test_daily_rag_consumer import _make_sidecars
from youtube_pipeline.configuration import (
    canonical_run_config_json,
    resolve_run_config,
    run_config_from_mapping,
)
from youtube_pipeline.daily_rag_consumer import (
    DailyRagConsumerConfig,
    write_daily_rag_consumer_artifacts_from_config,
)
from youtube_pipeline.entrypoints.daily_rag_consumer import (
    LEGACY_OUTPUT_DIR,
    LEGACY_SIDECARS_DIR,
    load_legacy_daily_rag_consumer_config,
    resolve_daily_rag_consumer_config,
)


class DailyRagConsumerConfigurationTests(unittest.TestCase):
    def test_domain_contains_no_historical_paths_or_argparse(self) -> None:
        config = DailyRagConsumerConfig()
        source = inspect.getsource(daily_rag_consumer_module)

        self.assertIsNone(config.sidecars_dir)
        self.assertIsNone(config.output_dir)
        self.assertNotIn(LEGACY_SIDECARS_DIR, source)
        self.assertNotIn(LEGACY_OUTPUT_DIR, source)
        self.assertNotIn("import argparse", source)

    def test_run_profile_keeps_global_and_consumer_identities_distinct(self) -> None:
        run = run_config_from_mapping(
            {
                "identity": {"run_id": "run_daily_consumer"},
                "rag": {
                    "daily_consumer": {
                        "sidecars_dir": "sidecars",
                        "output_dir": "consumer",
                        "max_estimated_input_tokens": 123,
                    }
                },
            }
        )

        self.assertIsNotNone(run.rag)
        self.assertIsInstance(run.rag.daily_consumer, DailyRagConsumerConfig)
        self.assertIsNone(run.rag.daily_consumer.run_id)
        self.assertEqual(run.rag.daily_consumer.max_estimated_input_tokens, 123)
        mapping = json.loads(canonical_run_config_json(run))
        self.assertIsNone(mapping["rag"]["daily_consumer"]["run_id"])

    def test_resolves_consumer_paths_without_requiring_artifacts(self) -> None:
        run = run_config_from_mapping(
            {
                "identity": {"run_id": "run_paths"},
                "rag": {
                    "daily_consumer": {
                        "sidecars_dir": "sidecars",
                        "output_dir": "consumer",
                    }
                },
            }
        )

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp).resolve()
            component = resolve_run_config(
                run,
                base_dir=base,
            ).config.rag.daily_consumer

        self.assertEqual(component.sidecars_dir, base / "sidecars")
        self.assertEqual(component.output_dir, base / "consumer")

    def test_common_profile_and_legacy_adapter_resolve_equivalent_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            component = {
                "sidecars_dir": "sidecars",
                "output_dir": "consumer",
                "max_estimated_input_tokens": 321,
            }
            profile_path = base / "profile.json"
            legacy_path = base / "legacy.json"
            profile_path.write_text(
                json.dumps(
                    {
                        "identity": {"run_id": "run_equivalent"},
                        "rag": {
                            "daily_consumer": {
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
                        "daily_rag_consumer": {
                            **component,
                            "run_id": "run_equivalent",
                        }
                    }
                ),
                encoding="utf-8",
            )

            common = resolve_daily_rag_consumer_config(
                config_file=profile_path,
                base_dir=base,
            )
            legacy = resolve_daily_rag_consumer_config(
                config_file=legacy_path,
                base_dir=base,
            )

        self.assertEqual(common, legacy)

    def test_legacy_adapter_preserves_defaults_and_deferred_run_id(self) -> None:
        config = load_legacy_daily_rag_consumer_config(None)

        self.assertEqual(config.sidecars_dir, LEGACY_SIDECARS_DIR)
        self.assertEqual(config.output_dir, LEGACY_OUTPUT_DIR)
        self.assertIsNone(config.run_id)

    def test_common_resolver_preserves_legacy_validation_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            sidecars = _make_sidecars(base)
            common_output = base / "common"
            legacy_output = base / "legacy"
            run = run_config_from_mapping(
                {
                    "identity": {"run_id": "run_consumer_compatibility"},
                    "rag": {
                        "daily_consumer": {
                            "sidecars_dir": sidecars,
                            "output_dir": common_output,
                            "run_id": "run_consumer_compatibility",
                            "max_estimated_input_tokens": 16_000,
                        }
                    },
                }
            )
            common = resolve_run_config(
                run,
                base_dir=base,
            ).config.rag.daily_consumer
            legacy = resolve_daily_rag_consumer_config(
                config_file=None,
                overrides={
                    "sidecars_dir": sidecars,
                    "output_dir": legacy_output,
                    "run_id": "run_consumer_compatibility",
                    "max_estimated_input_tokens": 16_000,
                },
                base_dir=base,
            )

            common_summary = write_daily_rag_consumer_artifacts_from_config(common)
            legacy_summary = write_daily_rag_consumer_artifacts_from_config(legacy)

            comparable_keys = {
                "run_id",
                "counts",
                "validation_status",
                "future_leak_count",
            }
            self.assertEqual(
                {key: common_summary[key] for key in comparable_keys},
                {key: legacy_summary[key] for key in comparable_keys},
            )
            self.assertEqual(
                (common_output / "daily_rag_validation_inputs.jsonl").read_text(
                    encoding="utf-8"
                ),
                (legacy_output / "daily_rag_validation_inputs.jsonl").read_text(
                    encoding="utf-8"
                ),
            )

    def test_component_run_id_is_independent_from_run_identity(self) -> None:
        run = run_config_from_mapping(
            {
                "identity": {"run_id": "global_run"},
                "rag": {
                    "daily_consumer": {
                        "sidecars_dir": "sidecars",
                        "output_dir": "consumer",
                        "run_id": "dragconsumer_stage",
                    }
                },
            }
        )

        self.assertEqual(run.identity.run_id, "global_run")
        self.assertEqual(run.rag.daily_consumer.run_id, "dragconsumer_stage")
        mapping = json.loads(canonical_run_config_json(run))
        self.assertEqual(
            mapping["rag"]["daily_consumer"]["run_id"],
            "dragconsumer_stage",
        )


if __name__ == "__main__":
    unittest.main()
