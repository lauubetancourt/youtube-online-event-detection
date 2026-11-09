from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import fields, replace
from pathlib import Path

from youtube_pipeline.entrypoints.non_daily_rag import (
    resolve_rag_consumer_config,
    resolve_rag_evidence_config,
    resolve_rag_g1_config,
    resolve_rag_g2_config,
    resolve_rag_g2_hierarchical_config,
    resolve_rag_sidecar_config,
    resolve_rag_validation_config,
)
from youtube_pipeline.rag_consumer import (
    RagConsumerConfig,
    derive_rag_consumer_run_id,
    load_rag_consumer_config,
)
from youtube_pipeline.rag_evidence import (
    RagEvidenceBuildConfig,
    load_rag_evidence_config,
    make_run_id,
)
from youtube_pipeline.rag_generation_g1 import RagG1Config, load_rag_g1_config
from youtube_pipeline.rag_generation_g2 import RagG2Config, load_rag_g2_config
from youtube_pipeline.rag_generation_g2_hierarchical import (
    RagG2HierarchicalConfig,
    load_rag_g2_hierarchical_config,
)
from youtube_pipeline.rag_sidecars import (
    CONTEXT_SELECTION_MANIFEST_FILE,
    RagSidecarBuildConfig,
    derive_rag_sidecar_run_id,
    load_rag_sidecar_config,
)
from youtube_pipeline.rag_validation import (
    RagValidationPrepareConfig,
    derive_rag_validation_run_id,
    load_rag_validation_config,
)


class NonDailyRagConfigurationTests(unittest.TestCase):
    def test_legacy_loaders_and_common_resolver_preserve_effective_configs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            sidecars_dir = base / "sidecars"
            sidecars_dir.mkdir()
            sidecar_manifest = {"run_id": "sidecar_source"}
            (sidecars_dir / CONTEXT_SELECTION_MANIFEST_FILE).write_text(
                json.dumps(sidecar_manifest), encoding="utf-8"
            )
            cases = [
                (
                    "rag_evidence",
                    {
                        "trigger_comment_map_path": str(base / "trigger.csv"),
                        "comments_path": str(base / "comments.parquet"),
                        "output_dir": str(base / "evidence"),
                    },
                    load_rag_evidence_config,
                    resolve_rag_evidence_config,
                ),
                (
                    "rag_sidecars",
                    {
                        "trigger_comment_map_path": str(base / "trigger.csv"),
                        "comments_path": str(base / "comments.parquet"),
                        "output_dir": str(sidecars_dir),
                    },
                    load_rag_sidecar_config,
                    resolve_rag_sidecar_config,
                ),
                (
                    "rag_consumer",
                    {
                        "sidecars_dir": str(sidecars_dir),
                        "output_dir": str(base / "consumer"),
                    },
                    load_rag_consumer_config,
                    resolve_rag_consumer_config,
                ),
                (
                    "rag_validation",
                    {
                        "evidence_packages_path": str(base / "packages.jsonl"),
                        "output_dir": str(base / "validation"),
                    },
                    load_rag_validation_config,
                    resolve_rag_validation_config,
                ),
                (
                    "rag_generation_g1",
                    {
                        "consumer_dir": str(base / "consumer"),
                        "output_dir": str(base / "g1"),
                        "event_id": "evt_test",
                    },
                    load_rag_g1_config,
                    resolve_rag_g1_config,
                ),
                (
                    "rag_generation_g2",
                    {
                        "consumer_dir": str(base / "consumer"),
                        "g1_dir": str(base / "g1"),
                        "output_dir": str(base / "g2"),
                        "event_id": "evt_34d7999bde8c",
                    },
                    load_rag_g2_config,
                    resolve_rag_g2_config,
                ),
                (
                    "rag_generation_g2_hierarchical",
                    {
                        "consumer_dir": str(base / "consumer"),
                        "output_dir": str(base / "g2h"),
                    },
                    load_rag_g2_hierarchical_config,
                    resolve_rag_g2_hierarchical_config,
                ),
            ]

            for section, payload, legacy_loader, resolver in cases:
                with self.subTest(section=section):
                    path = base / f"{section}.json"
                    path.write_text(json.dumps({section: payload}), encoding="utf-8")
                    legacy = legacy_loader(path)
                    resolved_run, component = resolver(
                        config_file=path,
                        overrides=None,
                        base_dir=base,
                    )

                    if isinstance(legacy, RagEvidenceBuildConfig):
                        legacy = replace(
                            legacy,
                            run_id=make_run_id(
                                detector_name=legacy.detector_name,
                                trigger_comment_map_path=legacy.trigger_comment_map_path,
                                comments_path=legacy.comments_path,
                                snapshots_path=legacy.snapshots_path,
                            ),
                        )
                    elif isinstance(legacy, RagSidecarBuildConfig):
                        legacy = replace(
                            legacy,
                            run_id=derive_rag_sidecar_run_id(legacy),
                        )
                    elif isinstance(legacy, RagConsumerConfig):
                        legacy = replace(
                            legacy,
                            run_id=derive_rag_consumer_run_id(
                                legacy, sidecar_manifest
                            ),
                        )
                    elif isinstance(legacy, RagValidationPrepareConfig):
                        legacy = replace(
                            legacy,
                            validation_run_id=derive_rag_validation_run_id(legacy),
                        )

                    self.assertEqual(component, legacy)
                    self.assertIs(
                        getattr(
                            resolved_run.config.rag,
                            {
                                "rag_evidence": "evidence",
                                "rag_sidecars": "sidecars",
                                "rag_consumer": "consumer",
                                "rag_validation": "validation",
                                "rag_generation_g1": "g1",
                                "rag_generation_g2": "g2",
                                "rag_generation_g2_hierarchical": "g2_hierarchical",
                            }[section],
                        ),
                        component,
                    )

    def test_global_identity_does_not_replace_stage_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            profile = base / "profile.json"
            profile.write_text(
                json.dumps(
                    {
                        "identity": {"run_id": "global_run"},
                        "rag": {
                            "sidecars": {
                                "trigger_comment_map_path": str(base / "trigger.csv"),
                                "output_dir": str(base / "sidecars"),
                                "run_id": "stage_sidecar_run",
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            resolved, component = resolve_rag_sidecar_config(
                config_file=profile,
                overrides=None,
                base_dir=base,
            )

        self.assertEqual(resolved.config.identity.run_id, "global_run")
        self.assertEqual(component.run_id, "stage_sidecar_run")
        canonical = json.loads(resolved.canonical_json)
        self.assertEqual(
            canonical["rag"]["sidecars"]["trigger_comment_map_path"],
            "trigger.csv",
        )
        self.assertEqual(canonical["rag"]["sidecars"]["output_dir"], "sidecars")

    def test_models_and_methodological_defaults_remain_component_authority(self) -> None:
        g1 = RagG1Config("consumer", "g1", "evt")
        g2 = RagG2Config("consumer", "g1", "g2", "evt_34d7999bde8c")
        hierarchical = RagG2HierarchicalConfig("consumer", "g2h")

        self.assertEqual((g1.model, g1.temperature, g1.max_approx_tokens), (
            "gpt-5-mini", 0.0, 16000
        ))
        self.assertEqual(
            (g2.query_model, g2.validation_model, g2.temperature),
            ("gpt-5-mini", "gpt-5-mini", 0.0),
        )
        self.assertEqual(
            (
                hierarchical.query_model,
                hierarchical.validation_model,
                hierarchical.temperature,
                hierarchical.serper_num_results,
            ),
            ("gpt-5-mini", "gpt-5-mini", 0.0, 5),
        )

    def test_hierarchical_legacy_cli_alias_translates_for_common_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            profile = base / "profile.json"
            profile.write_text(
                json.dumps(
                    {
                        "identity": {"run_id": "global_run"},
                        "rag": {
                            "g2_hierarchical": {
                                "consumer_dir": "consumer",
                                "output_dir": "g2h",
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            _, component = resolve_rag_g2_hierarchical_config(
                config_file=profile,
                overrides={"max_videos_per_event": 7},
                base_dir=base,
            )

        self.assertEqual(component.max_videos_per_event_batch, 7)

    def test_common_profile_rejects_unknown_rag_parameters(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            profile = base / "profile.json"
            profile.write_text(
                json.dumps(
                    {
                        "identity": {"run_id": "global_run"},
                        "rag": {
                            "g1": {
                                "consumer_dir": "consumer",
                                "output_dir": "g1",
                                "event_id": "evt",
                                "prompt_override": "not_allowed",
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "Unknown rag.g1 fields"):
                resolve_rag_g1_config(
                    config_file=profile,
                    overrides=None,
                    base_dir=base,
                )

    def test_secrets_are_not_part_of_rag_configuration_contracts(self) -> None:
        config_types = (
            RagEvidenceBuildConfig,
            RagSidecarBuildConfig,
            RagConsumerConfig,
            RagValidationPrepareConfig,
            RagG1Config,
            RagG2Config,
            RagG2HierarchicalConfig,
        )
        forbidden = {"openai_api_key", "serper_api_key", "api_key", "token"}

        for config_type in config_types:
            with self.subTest(config_type=config_type.__name__):
                self.assertTrue(
                    {field.name for field in fields(config_type)}.isdisjoint(forbidden)
                )


if __name__ == "__main__":
    unittest.main()
