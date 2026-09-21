from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError

from youtube_pipeline.g2_gate import (
    OFF,
    PERMISSIVE,
    RUN_G2,
    SKIP_G2,
    G2GatePolicyConfig,
    decide_g2,
)
from youtube_pipeline.configuration import run_config_from_mapping, run_config_hash


class G2GateContractTests(unittest.TestCase):
    def test_off_is_legacy_baseline_and_ignores_g1_label(self) -> None:
        config = G2GatePolicyConfig()
        for status in ("confirmed", "not_confirmed", "insufficient_evidence"):
            decision = decide_g2(
                candidate_id="candidate",
                config=config,
                g1_result={
                    "validation_id": "g1-result",
                    "validation_status": status,
                },
            )
            self.assertEqual(decision.decision, RUN_G2)
            self.assertEqual(decision.reason_code, "GATE_OFF_BASELINE")

    def test_non_off_policy_uses_only_explicit_schema_statuses(self) -> None:
        config = G2GatePolicyConfig(
            mode=PERMISSIVE,
            run_g2_for_g1_statuses=("confirmed", "partially_confirmed"),
            parameter_status="TEST_ONLY",
        )
        run = decide_g2(
            candidate_id="candidate",
            config=config,
            g1_result={"validation_status": "confirmed"},
        )
        skip = decide_g2(
            candidate_id="candidate",
            config=config,
            g1_result={"validation_status": "ambiguous"},
        )
        self.assertEqual(run.decision, RUN_G2)
        self.assertEqual(skip.decision, SKIP_G2)
        self.assertEqual(skip.reason_code, "G1_STATUS_CONFIGURED_TO_SKIP")

    def test_config_is_frozen_and_rejects_implicit_non_off_semantics(self) -> None:
        config = G2GatePolicyConfig()
        with self.assertRaises(FrozenInstanceError):
            config.mode = PERMISSIVE  # type: ignore[misc]
        with self.assertRaisesRegex(ValueError, "explicit"):
            G2GatePolicyConfig(mode=PERMISSIVE)
        with self.assertRaisesRegex(ValueError, "Unknown G-1"):
            G2GatePolicyConfig(
                mode=PERMISSIVE,
                run_g2_for_g1_statuses=("invented",),
            )

    def test_gate_configuration_is_resolved_and_hashed(self) -> None:
        base = {
            "identity": {"run_id": "gate-hash"},
            "rag": {
                "candidate_validation": {
                    "candidates_path": "candidates.jsonl",
                    "evidence_dir": "evidence",
                    "output_dir": "validation",
                    "candidate_id": "candidate",
                },
                "g2_gate": {"mode": OFF},
            },
        }
        changed = {
            **base,
            "rag": {
                **base["rag"],
                "g2_gate": {
                    "mode": PERMISSIVE,
                    "run_g2_for_g1_statuses": ["confirmed"],
                    "parameter_status": "TEST_ONLY",
                },
            },
        }
        first = run_config_from_mapping(base)
        second = run_config_from_mapping(changed)
        assert first.rag is not None
        self.assertEqual(first.rag.g2_gate.mode, OFF)
        self.assertNotEqual(run_config_hash(first), run_config_hash(second))


if __name__ == "__main__":
    unittest.main()
