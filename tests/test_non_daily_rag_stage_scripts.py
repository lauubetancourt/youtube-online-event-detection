from __future__ import annotations

import importlib.util
import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from youtube_pipeline.rag_consumer import RagConsumerConfig
from youtube_pipeline.rag_evidence import RagEvidenceBuildConfig
from youtube_pipeline.rag_generation_g1 import RagG1Config
from youtube_pipeline.rag_generation_g2 import RagG2Config
from youtube_pipeline.rag_generation_g2_hierarchical import RagG2HierarchicalConfig
from youtube_pipeline.rag_sidecars import RagSidecarBuildConfig
from youtube_pipeline.rag_validation import RagValidationPrepareConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = PROJECT_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"test_{name}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class NonDailyRagStageScriptTests(unittest.TestCase):
    def test_stage_scripts_delegate_to_common_resolvers(self) -> None:
        cases = [
            (
                "build_rag_event_evidence",
                "resolve_rag_evidence_config",
                "write_rag_evidence_artifacts_from_config",
                ["--trigger-comment-map-path", "trigger.csv", "--output-dir", "out"],
                RagEvidenceBuildConfig("trigger.csv", "out"),
            ),
            (
                "build_rag_sidecars",
                "resolve_rag_sidecar_config",
                "write_rag_sidecar_artifacts_from_config",
                ["--trigger-comment-map-path", "trigger.csv", "--output-dir", "out"],
                RagSidecarBuildConfig("trigger.csv", "out"),
            ),
            (
                "build_rag_consumer_payloads",
                "resolve_rag_consumer_config",
                "write_rag_consumer_artifacts_from_config",
                ["--sidecars-dir", "sidecars", "--output-dir", "out"],
                RagConsumerConfig("sidecars", "out"),
            ),
            (
                "prepare_rag_validation",
                "resolve_rag_validation_config",
                "prepare_rag_validation_artifacts_from_config",
                ["--evidence-packages-path", "packages.jsonl", "--output-dir", "out"],
                RagValidationPrepareConfig("packages.jsonl", "out"),
            ),
            (
                "run_rag_generation_g1",
                "resolve_rag_g1_config",
                "run_rag_g1_validation_from_config",
                ["--consumer-dir", "consumer", "--output-dir", "out", "--event-id", "evt"],
                RagG1Config("consumer", "out", "evt"),
            ),
            (
                "run_rag_generation_g2",
                "resolve_rag_g2_config",
                "run_rag_g2_validation_from_config",
                [
                    "--consumer-dir", "consumer",
                    "--g1-dir", "g1",
                    "--output-dir", "out",
                    "--event-id", "evt_34d7999bde8c",
                ],
                RagG2Config("consumer", "g1", "out", "evt_34d7999bde8c"),
            ),
            (
                "run_rag_generation_g2_hierarchical",
                "resolve_rag_g2_hierarchical_config",
                "plan_rag_g2_hierarchical_dry_run",
                ["--consumer-dir", "consumer", "--output-dir", "out", "--dry-run"],
                RagG2HierarchicalConfig("consumer", "out"),
            ),
        ]

        for script, resolver_name, runner_name, argv, config in cases:
            with self.subTest(script=script):
                module = _load_script(script)
                with (
                    patch.object(
                        module,
                        resolver_name,
                        return_value=(object(), config),
                    ) as resolver,
                    patch.object(
                        module,
                        runner_name,
                        return_value={"status": "ok"},
                    ) as runner,
                    patch.object(sys, "argv", [script, *argv]),
                    redirect_stdout(io.StringIO()),
                ):
                    module.main()

                resolver.assert_called_once()
                self.assertEqual(resolver.call_args.kwargs["config_file"], None)
                self.assertEqual(resolver.call_args.kwargs["base_dir"], Path.cwd())
                runner.assert_called_once_with(config)


if __name__ == "__main__":
    unittest.main()
