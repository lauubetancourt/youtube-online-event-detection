from __future__ import annotations

import json
import math
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
from measures.metrics.literature import EMDPol, EstebanRay
from measures.metrics.proposed import MEC

from youtube_pipeline.configuration import (
    load_run_config,
    run_config_from_mapping,
    run_config_hash,
    run_config_to_mapping,
)
from youtube_pipeline.polarization_measures import (
    DISTRIBUTION_READY,
    LIKERT5_POSITIONS,
    MEASUREMENT_COMPUTED,
    MEASUREMENT_NOT_COMPUTED,
    NO_CLASSIFIABLE_OPINIONS,
    EMDPolAdapter,
    EMDPolConfig,
    EstebanRayAdapter,
    EstebanRayConfig,
    MECAdapter,
    MECConfig,
    OpinionDistribution,
    PolarizationCalculationError,
    create_polarization_measure,
    get_polarization_measure_names,
)


CANONICAL_WEIGHTS = (
    (1.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0, 0.0),
    (0.5, 0.0, 0.0, 0.0, 0.5),
    (0.2, 0.2, 0.2, 0.2, 0.2),
    (0.1, 0.1, 0.2, 0.2, 0.4),
)


def _distribution(
    weights: tuple[float, ...] = CANONICAL_WEIGHTS[-1],
    *,
    quality: str = "passed",
    quality_reasons: tuple[str, ...] = (),
    support_count: int = 10,
) -> OpinionDistribution:
    bin_counts = tuple(round(weight * support_count) for weight in weights)
    return OpinionDistribution(
        proposition_id="proposition_01",
        distribution_policy_id="likert5_author_latest_classified_v1",
        classification_run_id="classification_run_01",
        positions=LIKERT5_POSITIONS,
        bin_counts=bin_counts,
        weights=weights,
        support_count=support_count,
        unique_author_count=support_count,
        input_count=support_count,
        classified_comment_count=support_count,
        excluded_counts={},
        quality=quality,
        quality_reasons=quality_reasons,
    )


def _empty_distribution() -> OpinionDistribution:
    return OpinionDistribution(
        proposition_id="proposition_01",
        distribution_policy_id="likert5_author_latest_classified_v1",
        classification_run_id="classification_run_01",
        positions=LIKERT5_POSITIONS,
        bin_counts=(0, 0, 0, 0, 0),
        weights=None,
        support_count=0,
        unique_author_count=0,
        input_count=0,
        classified_comment_count=0,
        excluded_counts={},
        quality="degraded",
        quality_reasons=("no_classifiable_opinions",),
        status=NO_CLASSIFIABLE_OPINIONS,
    )


class OpinionDistributionTests(unittest.TestCase):
    def test_ready_distribution_is_immutable_and_canonical(self) -> None:
        distribution = _distribution()

        self.assertEqual(distribution.status, DISTRIBUTION_READY)
        self.assertEqual(distribution.mass_unit, "author")
        self.assertEqual(distribution.positions, LIKERT5_POSITIONS)
        self.assertTrue(math.isclose(math.fsum(distribution.weights), 1.0))
        with self.assertRaises(FrozenInstanceError):
            distribution.quality = "changed"

    def test_empty_distribution_has_no_formal_weights(self) -> None:
        distribution = _empty_distribution()

        self.assertEqual(distribution.status, NO_CLASSIFIABLE_OPINIONS)
        self.assertIsNone(distribution.weights)
        self.assertEqual(distribution.support_count, 0)

    def test_invalid_distributions_are_rejected(self) -> None:
        base = {
            "proposition_id": "proposition_01",
            "distribution_policy_id": "likert5_author_latest_classified_v1",
            "classification_run_id": "classification_run_01",
            "positions": LIKERT5_POSITIONS,
            "bin_counts": (1, 1, 2, 2, 4),
            "weights": CANONICAL_WEIGHTS[-1],
            "support_count": 10,
            "unique_author_count": 10,
            "input_count": 10,
            "classified_comment_count": 10,
            "excluded_counts": {},
            "quality": "passed",
        }
        invalid_cases = (
            ({"positions": (0.0, 0.5, 1.0)}, ValueError),
            ({"weights": (0.1, 0.1, 0.2, 0.2, 0.3)}, ValueError),
            ({"weights": (0.1, 0.1, 0.2, 0.2, float("nan"))}, ValueError),
            ({"weights": (0.1, 0.1, 0.2, 0.2, -0.6)}, ValueError),
            ({"support_count": 0, "unique_author_count": 0}, ValueError),
            ({"support_count": True, "unique_author_count": True}, TypeError),
            ({"unique_author_count": 9}, ValueError),
            ({"bin_counts": (1, 1, 2, 2, 3)}, ValueError),
            ({"input_count": 9}, ValueError),
            ({"classified_comment_count": 9}, ValueError),
            ({"excluded_counts": {"uncertain": -1}}, ValueError),
            ({"mass_unit": "comment"}, ValueError),
        )

        for override, error_type in invalid_cases:
            with self.subTest(override=override):
                payload = dict(base)
                payload.update(override)
                with self.assertRaises(error_type):
                    OpinionDistribution(**payload)

    def test_empty_status_rejects_synthetic_zero_or_neutral_weights(self) -> None:
        for weights in (
            (0.0, 0.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0, 0.0),
        ):
            with self.subTest(weights=weights):
                with self.assertRaisesRegex(ValueError, "weights must be None"):
                    OpinionDistribution(
                        proposition_id="proposition_01",
                        distribution_policy_id=(
                            "likert5_author_latest_classified_v1"
                        ),
                        classification_run_id="classification_run_01",
                        positions=LIKERT5_POSITIONS,
                        bin_counts=(0, 0, 0, 0, 0),
                        weights=weights,
                        support_count=0,
                        unique_author_count=0,
                        input_count=0,
                        classified_comment_count=0,
                        excluded_counts={},
                        quality="degraded",
                        status=NO_CLASSIFIABLE_OPINIONS,
                    )


class PolarizationMeasureConfigTests(unittest.TestCase):
    def test_defaults_match_pinned_library_and_are_immutable(self) -> None:
        esteban_ray = EstebanRayConfig()
        emd_pol = EMDPolConfig()
        mec = MECConfig()

        self.assertEqual(esteban_ray, EstebanRayConfig(alpha=0.8, K=None))
        self.assertEqual(emd_pol, EMDPolConfig())
        self.assertEqual(mec, MECConfig(alpha=2.0, beta=1.15))
        with self.assertRaises(FrozenInstanceError):
            mec.alpha = 1.0

    def test_custom_parameters_are_preserved(self) -> None:
        self.assertEqual(EstebanRayConfig(alpha=1.2, K=2.0).alpha, 1.2)
        self.assertEqual(EstebanRayConfig(alpha=1.2, K=2.0).K, 2.0)
        self.assertEqual(MECConfig(alpha=1.1, beta=0.9).beta, 0.9)

    def test_invalid_parameters_are_rejected(self) -> None:
        invalid_calls = (
            (lambda: EstebanRayConfig(alpha=True), TypeError),
            (lambda: EstebanRayConfig(alpha=0), ValueError),
            (lambda: EstebanRayConfig(alpha=1.7), ValueError),
            (lambda: EstebanRayConfig(K=0), ValueError),
            (lambda: EstebanRayConfig(K=float("inf")), ValueError),
            (lambda: MECConfig(alpha=True), TypeError),
            (lambda: MECConfig(alpha=0), ValueError),
            (lambda: MECConfig(beta=-1), ValueError),
            (lambda: MECConfig(beta=float("nan")), ValueError),
        )

        for call, error_type in invalid_calls:
            with self.subTest(call=call):
                with self.assertRaises(error_type):
                    call()

    def test_unknown_parameter_fields_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown Esteban-Ray"):
            EstebanRayConfig.from_mapping({"threshold": 1})
        with self.assertRaisesRegex(ValueError, "Unknown EMDPol"):
            EMDPolConfig.from_mapping({"normalize": True})
        with self.assertRaisesRegex(ValueError, "Unknown MEC"):
            MECConfig.from_mapping({"gamma": 1})

    def test_json_loading_and_serialization_select_each_measure(self) -> None:
        cases = (
            (
                "esteban_ray",
                {"esteban_ray": {"alpha": 1.0, "K": None}},
            ),
            ("emd_pol", {"emd_pol": {}}),
            ("mec", {"mec": {"alpha": 1.5, "beta": 1.0}}),
        )

        for measure_id, config_block in cases:
            with self.subTest(measure_id=measure_id):
                payload = {
                    "identity": {"run_id": f"{measure_id}_run"},
                    "polarization": {
                        "measure_id": measure_id,
                        **config_block,
                    },
                }
                with tempfile.TemporaryDirectory() as tmp:
                    config_path = Path(tmp) / "polarization.json"
                    config_path.write_text(json.dumps(payload), encoding="utf-8")
                    config = load_run_config(config_path)

                self.assertEqual(config.polarization.measure_id, measure_id)
                self.assertEqual(
                    run_config_to_mapping(config)["polarization"],
                    payload["polarization"],
                )

    def test_selected_measure_requires_matching_config_and_rejects_unknowns(self) -> None:
        with self.assertRaisesRegex(ValueError, "matching configuration block"):
            run_config_from_mapping(
                {
                    "identity": {"run_id": "missing_measure_config"},
                    "polarization": {"measure_id": "mec"},
                }
            )
        with self.assertRaisesRegex(ValueError, "Unknown polarization fields"):
            run_config_from_mapping(
                {
                    "identity": {"run_id": "unknown_measure_config"},
                    "polarization": {
                        "measure_id": "mec",
                        "mec": {},
                        "calibrated": True,
                    },
                }
            )

    def test_parameter_change_changes_config_hash(self) -> None:
        def config(alpha: float):
            return run_config_from_mapping(
                {
                    "identity": {"run_id": "mec_hash"},
                    "polarization": {
                        "measure_id": "mec",
                        "mec": {"alpha": alpha, "beta": 1.15},
                    },
                }
            )

        self.assertNotEqual(run_config_hash(config(1.0)), run_config_hash(config(2.0)))

    def test_resolved_selection_builds_measure_without_python_branching(self) -> None:
        run_config = run_config_from_mapping(
            {
                "identity": {"run_id": "configured_esteban_ray"},
                "polarization": {
                    "measure_id": "esteban_ray",
                    "esteban_ray": {"alpha": 1.1, "K": None},
                },
            }
        )
        polarization = run_config.polarization
        assert polarization is not None

        adapter = create_polarization_measure(
            polarization.measure_id,
            config=polarization.selected_config,
        )
        result = adapter.measure(_distribution())

        self.assertEqual(result.measure_id, "esteban_ray")
        self.assertEqual(dict(result.parameters), {"alpha": 1.1, "K": None})
        self.assertTrue(math.isfinite(result.value))


class PolarizationMeasureAdapterTests(unittest.TestCase):
    def test_registry_is_explicit_and_factory_creates_fresh_adapters(self) -> None:
        self.assertEqual(
            get_polarization_measure_names(),
            ("emd_pol", "esteban_ray", "mec"),
        )
        first = create_polarization_measure(
            "esteban_ray",
            config=EstebanRayConfig(),
        )
        second = create_polarization_measure(
            "esteban_ray",
            config=EstebanRayConfig(),
        )
        self.assertIsInstance(first, EstebanRayAdapter)
        self.assertIsNot(first, second)
        with self.assertRaisesRegex(ValueError, "Unknown polarization measure"):
            create_polarization_measure("unknown")

    def test_result_mapping_quality_and_parameters_are_immutable(self) -> None:
        distribution = _distribution(
            weights=CANONICAL_WEIGHTS[1],
            quality="degraded",
            quality_reasons=("single_mass_unit",),
            support_count=1,
        )
        result = MECAdapter(config=MECConfig(alpha=1.5, beta=1.0)).measure(
            distribution
        )

        self.assertEqual(result.measure_id, "mec")
        self.assertEqual(result.status, MEASUREMENT_COMPUTED)
        self.assertTrue(math.isfinite(result.value))
        self.assertEqual(dict(result.parameters), {"alpha": 1.5, "beta": 1.0})
        self.assertEqual(result.quality, distribution.quality)
        self.assertEqual(result.quality_reasons, distribution.quality_reasons)
        self.assertEqual(result.support_count, distribution.support_count)
        self.assertEqual(
            result.unique_author_count,
            distribution.unique_author_count,
        )
        self.assertEqual(result.proposition_id, distribution.proposition_id)
        self.assertEqual(
            result.distribution_policy_id,
            distribution.distribution_policy_id,
        )
        self.assertEqual(
            result.classification_run_id,
            distribution.classification_run_id,
        )
        with self.assertRaises(FrozenInstanceError):
            result.value = 0
        with self.assertRaises(TypeError):
            result.parameters["alpha"] = 2

    def test_empty_distribution_skips_all_library_calls(self) -> None:
        for adapter in (
            EstebanRayAdapter(),
            EMDPolAdapter(),
            MECAdapter(),
        ):
            with self.subTest(measure_id=adapter.measure_id):
                calls = []

                def fail_if_called(*args, **kwargs):
                    calls.append((args, kwargs))
                    raise AssertionError("library must not be called")

                adapter._library_measure = fail_if_called
                result = adapter.measure(_empty_distribution())

                self.assertEqual(result.status, MEASUREMENT_NOT_COMPUTED)
                self.assertIsNone(result.value)
                self.assertEqual(calls, [])

    def test_all_adapters_receive_identical_unnormalized_inputs(self) -> None:
        distribution = _distribution()
        captured = []
        for adapter in (
            EstebanRayAdapter(),
            EMDPolAdapter(),
            MECAdapter(),
        ):
            def spy(x, weights, **kwargs):
                captured.append((x.copy(), weights.copy(), dict(kwargs)))
                return 0.25

            adapter._library_measure = spy
            adapter.measure(distribution)

        for positions, weights, kwargs in captured:
            np.testing.assert_array_equal(positions, distribution.positions)
            np.testing.assert_array_equal(weights, distribution.weights)
            self.assertEqual(
                kwargs,
                {"normalize_weights": False, "normalize_positions": False},
            )

    def test_non_finite_library_result_is_an_explicit_error(self) -> None:
        adapter = EMDPolAdapter()
        adapter._library_measure = lambda *args, **kwargs: float("nan")

        with self.assertRaisesRegex(
            PolarizationCalculationError,
            "did not return a finite scalar",
        ):
            adapter.measure(_distribution())

    def test_canonical_cases_are_finite_and_match_direct_library_calls(self) -> None:
        cases = (
            (
                EstebanRayAdapter(),
                EstebanRay(),
            ),
            (EMDPolAdapter(), EMDPol()),
            (MECAdapter(), MEC()),
        )

        for weights in CANONICAL_WEIGHTS:
            distribution = _distribution(weights)
            x = np.asarray(distribution.positions, dtype=np.float64)
            pi = np.asarray(distribution.weights, dtype=np.float64)
            for adapter, direct_measure in cases:
                with self.subTest(
                    measure_id=adapter.measure_id,
                    weights=weights,
                ):
                    expected = direct_measure(
                        x,
                        pi,
                        normalize_weights=False,
                        normalize_positions=False,
                    )
                    actual = adapter.measure(distribution).value
                    self.assertTrue(math.isfinite(actual))
                    self.assertAlmostEqual(actual, float(expected), places=12)


if __name__ == "__main__":
    unittest.main()
