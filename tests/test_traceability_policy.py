from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError

from youtube_pipeline.configuration import (
    ArtifactsConfig,
    RunConfig,
    RunIdentityConfig,
    SimulationConfig,
    resolve_run_config,
    run_config_from_mapping,
)
from youtube_pipeline.cyclic_ingestion import CyclicIngestionConfig
from youtube_pipeline.run_manifest import (
    build_resolved_config_metadata,
    validate_current_traceability_support,
)


def _run(*, artifacts: ArtifactsConfig | None = None) -> RunConfig:
    return RunConfig(
        identity=RunIdentityConfig(run_id="traceability_run"),
        simulation=SimulationConfig(
            ingestion=CyclicIngestionConfig(
                input_path="prepared/comments.parquet",
                output_dir="outputs/cyclic",
            )
        ),
        artifacts=artifacts,
    )


class TraceabilityPolicyTests(unittest.TestCase):
    def test_mode_defaults_are_typed_effective_and_immutable(self) -> None:
        development = ArtifactsConfig(run_mode="development")
        reference = ArtifactsConfig(run_mode="reference")
        official = ArtifactsConfig(run_mode="official")

        self.assertEqual(development.trace_level, "minimal")
        self.assertEqual(reference.trace_level, "standard")
        self.assertEqual(official.trace_level, "full")
        with self.assertRaises(FrozenInstanceError):
            development.trace_level = "full"

    def test_allows_more_traceability_but_never_less_than_mode_minimum(self) -> None:
        self.assertEqual(
            ArtifactsConfig(
                run_mode="development",
                trace_level="full",
            ).trace_level,
            "full",
        )
        self.assertEqual(
            ArtifactsConfig(
                run_mode="reference",
                trace_level="full",
            ).trace_level,
            "full",
        )
        with self.assertRaisesRegex(ValueError, "requires trace_level 'standard'"):
            ArtifactsConfig(run_mode="reference", trace_level="minimal")
        with self.assertRaisesRegex(ValueError, "requires trace_level 'full'"):
            ArtifactsConfig(run_mode="official", trace_level="standard")

    def test_rejects_unknown_modes_levels_and_configuration_keys(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported run_mode"):
            ArtifactsConfig(run_mode="debug")
        with self.assertRaisesRegex(ValueError, "Unsupported trace_level"):
            ArtifactsConfig(trace_level="verbose")
        with self.assertRaisesRegex(ValueError, "Unknown artifacts fields"):
            run_config_from_mapping(
                {
                    "identity": {"run_id": "unknown_trace_key"},
                    "simulation": {
                        "ingestion": {
                            "input_path": "prepared/comments.parquet",
                            "output_dir": "outputs/cyclic",
                        }
                    },
                    "artifacts": {
                        "run_mode": "development",
                        "keep_last_n": 3,
                    },
                }
            )

    def test_explicit_policy_is_part_of_resolved_config_and_hash(self) -> None:
        development = resolve_run_config(
            _run(artifacts=ArtifactsConfig(run_mode="development")),
            base_dir=".",
        )
        reference = resolve_run_config(
            _run(artifacts=ArtifactsConfig(run_mode="reference")),
            base_dir=".",
        )

        self.assertNotEqual(development.config_hash, reference.config_hash)
        self.assertIn('"run_mode":"development"', development.canonical_json)
        self.assertIn('"trace_level":"minimal"', development.canonical_json)
        self.assertIn('"run_mode":"reference"', reference.canonical_json)
        self.assertIn('"trace_level":"standard"', reference.canonical_json)

    def test_legacy_config_without_policy_reports_safe_implicit_default(self) -> None:
        resolved = resolve_run_config(_run(), base_dir=".")

        metadata = build_resolved_config_metadata(resolved)

        self.assertEqual(metadata["run_mode"], "development")
        self.assertEqual(metadata["trace_level"], "minimal")
        self.assertNotIn("artifacts", metadata["resolved_config"])

    def test_unimplemented_higher_policy_is_blocked_before_execution(self) -> None:
        reference = resolve_run_config(
            _run(artifacts=ArtifactsConfig(run_mode="reference")),
            base_dir=".",
        )

        with self.assertRaisesRegex(
            ValueError,
            "only implement run_mode='development'",
        ):
            validate_current_traceability_support(reference)


if __name__ == "__main__":
    unittest.main()
