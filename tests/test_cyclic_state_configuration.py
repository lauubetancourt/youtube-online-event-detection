from __future__ import annotations

import inspect
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import youtube_pipeline.cyclic_orchestration as orchestration_module
import youtube_pipeline.cyclic_stateful_adapter as adapter_module
from youtube_pipeline.configuration import resolve_run_config, run_config_from_mapping
from youtube_pipeline.cyclic_ingestion import (
    CyclicIngestionConfig,
    build_cyclic_ingestion_dry_run,
)
from youtube_pipeline.cyclic_orchestration import (
    CyclicOrchestratorConfig,
    run_cyclic_orchestrator_dry_run,
)
from youtube_pipeline.cyclic_stateful_adapter import (
    CyclicStatefulAdapterConfig,
    run_cyclic_stateful_adapter,
)
from youtube_pipeline.entrypoints.cyclic_ingestion import LEGACY_OUTPUT_DIR
from youtube_pipeline.entrypoints.cyclic_orchestration import (
    load_legacy_orchestrator_config,
    resolve_cyclic_orchestrator_config,
)
from youtube_pipeline.entrypoints.cyclic_stateful_adapter import (
    load_legacy_stateful_adapter_config,
    resolve_cyclic_stateful_adapter_config,
)


class CyclicStateConfigurationTests(unittest.TestCase):
    def _seed_ingestion(self, dataset: Path, output_dir: Path) -> None:
        build_cyclic_ingestion_dry_run(
            CyclicIngestionConfig(
                input_path=dataset,
                output_dir=output_dir,
                collection_start_date_local="2026-06-01",
                collection_end_date_local="2026-06-02",
                analysis_window_size_days=2,
                simulation_run_id="sim_state_config",
            )
        )

    def test_domain_configs_require_paths_and_hide_historical_defaults(self) -> None:
        with self.assertRaises(TypeError):
            CyclicOrchestratorConfig()
        with self.assertRaises(TypeError):
            CyclicStatefulAdapterConfig()

        for module in (orchestration_module, adapter_module):
            with self.subTest(module=module.__name__):
                source = inspect.getsource(module)
                self.assertNotIn(LEGACY_OUTPUT_DIR, source)
                self.assertNotIn("import argparse", source)

    def test_common_configs_match_legacy_orchestration_and_adapter_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            dataset = base / "comments.parquet"
            common_dir = base / "common"
            legacy_dir = base / "legacy"
            pd.DataFrame(
                [
                    {
                        "comment_id": "c1",
                        "video_id": "v1",
                        "event_time_utc": "2026-06-01T12:00:00Z",
                    },
                    {
                        "comment_id": "c2",
                        "video_id": "v1",
                        "event_time_utc": "2026-06-02T12:00:00Z",
                    },
                ]
            ).to_parquet(dataset, index=False)
            self._seed_ingestion(dataset, common_dir)
            self._seed_ingestion(dataset, legacy_dir)

            run = run_config_from_mapping(
                {
                    "identity": {"run_id": "run_state_config"},
                    "simulation": {
                        "orchestration": {"simulation_dir": "common"},
                        "stateful_adapter": {"simulation_dir": "common"},
                    },
                }
            )
            resolved = resolve_run_config(run, base_dir=base).config
            common_orchestration = resolved.simulation.orchestration
            common_adapter = resolved.simulation.stateful_adapter
            legacy_orchestration = load_legacy_orchestrator_config(
                None,
                overrides={"simulation_dir": legacy_dir},
            )
            legacy_adapter = load_legacy_stateful_adapter_config(
                None,
                overrides={"simulation_dir": legacy_dir},
            )

            common_orchestration_summary = run_cyclic_orchestrator_dry_run(
                common_orchestration
            )
            legacy_orchestration_summary = run_cyclic_orchestrator_dry_run(
                legacy_orchestration
            )
            common_adapter_summary = run_cyclic_stateful_adapter(common_adapter)
            legacy_adapter_summary = run_cyclic_stateful_adapter(legacy_adapter)

            orchestration_keys = {
                "simulation_run_id",
                "orchestration_status",
                "cycles_total",
                "ready_cycle_count",
                "completed_dry_run_cycle_count",
                "skipped_no_comments_cycle_count",
                "failed_contract_validation_cycle_count",
            }
            self.assertEqual(
                {
                    key: common_orchestration_summary[key]
                    for key in orchestration_keys
                },
                {
                    key: legacy_orchestration_summary[key]
                    for key in orchestration_keys
                },
            )
            adapter_keys = {
                "simulation_run_id",
                "adapter_status",
                "cycles_total",
                "readiness_failed",
                "seen_comment_count",
                "active_window_memberships",
                "exited_window_memberships",
                "cycles_with_window_overlap",
            }
            self.assertEqual(
                {key: common_adapter_summary[key] for key in adapter_keys},
                {key: legacy_adapter_summary[key] for key in adapter_keys},
            )
            for artifact in (
                "cycle_orchestration_plan.jsonl",
                "cycle_monitoring_inputs.jsonl",
                "cycle_detection_inputs.jsonl",
                "cycle_window_inventory.csv",
            ):
                with self.subTest(artifact=artifact):
                    self.assertEqual(
                        (common_dir / artifact).read_text(encoding="utf-8"),
                        (legacy_dir / artifact).read_text(encoding="utf-8"),
                    )

    def test_legacy_defaults_and_current_profile_resolve_outside_domain(self) -> None:
        self.assertEqual(
            load_legacy_orchestrator_config(None).simulation_dir,
            LEGACY_OUTPUT_DIR,
        )
        self.assertEqual(
            load_legacy_stateful_adapter_config(None).simulation_dir,
            LEGACY_OUTPUT_DIR,
        )

        repository_root = Path(__file__).resolve().parents[1]
        profile = repository_root / "configs/compatibility/cyclic_current.json"
        orchestration = resolve_cyclic_orchestrator_config(
            config_file=profile,
            base_dir=repository_root,
        )
        adapter = resolve_cyclic_stateful_adapter_config(
            config_file=profile,
            base_dir=repository_root,
        )
        expected = (repository_root / LEGACY_OUTPUT_DIR).resolve(strict=False)
        self.assertEqual(orchestration.simulation_dir, expected)
        self.assertEqual(adapter.simulation_dir, expected)


if __name__ == "__main__":
    unittest.main()
