from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock, patch

from youtube_pipeline.comment_stance import (
    CLASSIFIED, ERROR, NOT_APPLICABLE, SKIPPED, UNCERTAIN, CommentStance,
)
from youtube_pipeline.configuration import (
    load_run_config, resolve_run_config, run_config_from_mapping,
    run_config_hash, run_config_to_mapping,
)
from youtube_pipeline.polarization_measures import create_polarization_measure
from youtube_pipeline.polarization_signal_composition import create_configured_polarization_signal
from youtube_pipeline.polarization_signals import (
    PolarizationTemporalSignal, PolarizationTemporalSignalConfig,
    create_polarization_temporal_signal,
)
from youtube_pipeline.signal_observations import (
    NO_CLASSIFIABLE_OPINIONS, NO_INPUT, NO_REPRESENTABLE_AUTHORS,
    VALUE_AVAILABLE, SignalObservation,
)


# All temporal choices and thresholds in this module are TEST_ONLY, not calibration.
BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)
MEASURES = ("esteban_ray", "emd_pol", "mec")


def t(seconds: int) -> datetime:
    return BASE + timedelta(seconds=seconds)


def stance(seconds: int, comment: str, author: str | None, category: int | None = 3,
           status: str = CLASSIFIED, proposition: str = "p", quality: str = "passed") -> CommentStance:
    return CommentStance(comment, author, t(seconds), proposition, category, status, quality)


def config(**overrides) -> PolarizationTemporalSignalConfig:
    return replace(PolarizationTemporalSignalConfig(
        proposition_id="p", classification_run_id="synthetic_only",
        window="60s", cadence="30s", start_time_utc=t(0), end_time_utc=t(181),
        enabled_measures=MEASURES,
    ), **overrides)


def producer(definition=None, spies=False):
    definition = definition or config()
    adapters = {name: create_polarization_measure(name) for name in definition.enabled_measures}
    if spies:
        adapters = {name: Mock(wraps=adapter, measure_id=name) for name, adapter in adapters.items()}
    return PolarizationTemporalSignal(definition, adapters), adapters


def synthetic_fixture(reverse=False):
    """Known categories only: transition, update, expiry, excluded statuses, missing ID."""
    initial = (1, 5) if reverse else (3, 3)
    final = (3, 3) if reverse else (1, 5)
    return (
        stance(0, "a0", "A", initial[0]), stance(0, "b0", "B", initial[1]),
        stance(30, "a1", "A", final[0]), stance(30, "b1", "B", final[1]),
        stance(40, "nonopinion", "C", None, NOT_APPLICABLE),
        stance(41, "uncertain", "C", None, UNCERTAIN),
        stance(42, "skipped", "C", None, SKIPPED),
        stance(43, "error", "C", None, ERROR),
        stance(44, "missing", None, 2),
        stance(70, "a2", "A", 5),
    )


def payload():
    return {
        "identity": {"run_id": "synthetic_polarization"},
        "signals": {"polarization": {
            "proposition_id": "p", "classification_run_id": "synthetic_only",
            "window": "60s", "cadence": "30s",
            "start_time_utc": t(0).isoformat(), "end_time_utc": t(181).isoformat(),
            "enabled_measures": list(MEASURES),
        }},
        "polarization": {"esteban_ray": {}, "emd_pol": {}, "mec": {}},
    }


class CommentStanceTests(unittest.TestCase):
    def test_likert_mapping(self):
        for category, position in enumerate((0, .25, .5, .75, 1), 1):
            self.assertEqual(stance(0, "c", "a", category).stance_position, position)

    def test_invalid_categories(self):
        for value in (0, 6, True, 2.5, "3", None):
            with self.subTest(value=value), self.assertRaises((TypeError, ValueError)):
                stance(0, "c", "a", value)

    def test_nonclassified_has_no_position(self):
        for status in (NOT_APPLICABLE, UNCERTAIN, SKIPPED, ERROR):
            self.assertIsNone(stance(0, "c", "a", None, status).stance_position)
            with self.assertRaises(ValueError):
                stance(0, "c", "a", 3, status)

    def test_invalid_identity_time_status(self):
        valid = stance(0, "c", "a")
        for changes in ({"author_id": ""}, {"comment_id": ""},
                        {"proposition_id": ""}, {"classification_status": "other"},
                        {"event_time_utc": datetime(2026, 1, 1)}, {"quality": None}):
            with self.subTest(changes=changes), self.assertRaises((ValueError, TypeError)):
                replace(valid, **changes)

    def test_immutable(self):
        with self.assertRaises(FrozenInstanceError):
            stance(0, "c", "a").stance_category = 1


class SignalObservationTests(unittest.TestCase):
    def observation(self, **changes):
        values = dict(signal_id="polarization_mec", observation_time_utc=t(0),
                      value=1., status=VALUE_AVAILABLE, support_count=2, quality="passed", metadata={})
        return SignalObservation(**(values | changes))

    def test_absence_and_finite_value_validation(self):
        for status in (NO_INPUT, NO_CLASSIFIABLE_OPINIONS, NO_REPRESENTABLE_AUTHORS):
            self.assertIsNone(self.observation(status=status, value=None).value)
            with self.assertRaises(ValueError):
                self.observation(status=status, value=0.)
        for value in (None, True, float("nan"), float("inf"), "1"):
            with self.assertRaises((TypeError, ValueError)):
                self.observation(value=value)
        with self.assertRaises(ValueError):
            self.observation(status="UNKNOWN")
        with self.assertRaises(TypeError):
            self.observation(support_count=True)

    def test_metadata_snapshot_deep_immutable_serializable(self):
        source = {"nested": {"values": [1, 2]}}
        observation = self.observation(metadata=source)
        source["nested"]["values"].append(3)
        self.assertEqual(observation.metadata["nested"]["values"], (1, 2))
        with self.assertRaises(TypeError):
            observation.metadata["nested"]["new"] = 1
        with self.assertRaises(FrozenInstanceError):
            observation.value = 2.
        json.dumps(observation.to_mapping(), allow_nan=False)
        for metadata in ({"bad": object()}, {"bad": float("nan")}, {1: "invalid"}):
            with self.assertRaises((TypeError, ValueError)):
                self.observation(metadata=metadata)


class TemporalPolarizationTests(unittest.TestCase):
    def test_consensus_to_bipolarization_and_adapter_equivalence(self):
        signal, adapters = producer()
        ticks = signal.advance(synthetic_fixture(), data_cutoff_utc=t(181))
        self.assertEqual(ticks[0].distribution.bin_counts, (0, 0, 2, 0, 0))
        self.assertEqual(ticks[1].distribution.bin_counts, (1, 0, 0, 0, 1))
        self.assertEqual(ticks[2].distribution.bin_counts, (1, 0, 0, 0, 1))
        for tick in ticks:
            for observation in tick.observations:
                if tick.status == VALUE_AVAILABLE:
                    direct = adapters[observation.metadata["measure_id"]].measure(tick.distribution)
                    self.assertAlmostEqual(observation.value, direct.value, places=12)
                    self.assertEqual(observation.quality, direct.quality)
                else:
                    self.assertIsNone(observation.value)
        json.dumps([tick.to_mapping() for tick in ticks], allow_nan=False)

    def test_bipolarization_to_consensus(self):
        signal, _ = producer()
        # A complete causal prefix must not include later fixture records.
        ticks = signal.advance(synthetic_fixture(reverse=True)[:4],
                               data_cutoff_utc=t(61))
        self.assertEqual(ticks[0].distribution.bin_counts, (1, 0, 0, 0, 1))
        self.assertEqual(ticks[1].distribution.bin_counts, (0, 0, 2, 0, 0))

    def test_author_change_does_not_add_mass(self):
        signal, _ = producer()
        ticks = signal.advance((stance(0, "a0", "A", 1), stance(20, "a1", "A", 5)),
                               data_cutoff_utc=t(31))
        self.assertEqual(ticks[1].distribution.support_count, 1)
        self.assertEqual(ticks[1].distribution.classified_comment_count, 2)
        self.assertEqual(ticks[1].distribution.bin_counts, (0, 0, 0, 0, 1))
        self.assertEqual(ticks[1].selected_comment_ids, ("a1",))

    def test_timestamp_tiebreak_lexicographic_max(self):
        signal, _ = producer()
        tick = signal.advance((stance(0, "a", "A", 1), stance(0, "z", "A", 5)),
                              data_cutoff_utc=t(1))[0]
        self.assertEqual(tick.selected_comment_ids, ("z",))
        self.assertEqual(tick.distribution.support_count, 1)

    def test_equal_timestamp_group_completed_before_tick(self):
        signal, _ = producer()
        tick = signal.advance((stance(0, "a", "A", 1), stance(0, "b", "B", 5)),
                              data_cutoff_utc=t(1))[0]
        self.assertEqual(tick.distribution.weights, (.5, 0, 0, 0, .5))

    def test_closed_left_boundary_and_author_expiration(self):
        signal, _ = producer()
        ticks = signal.advance((stance(0, "a", "A", 1),), data_cutoff_utc=t(91))
        self.assertEqual(ticks[2].observation_time_utc, t(60))
        self.assertEqual(ticks[2].distribution.support_count, 1)
        self.assertEqual(ticks[3].distribution.support_count, 0)
        self.assertEqual(ticks[3].status, NO_INPUT)

    def test_nonclassified_does_not_erase_last_classified(self):
        signal, _ = producer()
        ticks = signal.advance((stance(0, "a", "A", 1), stance(20, "b", "A", None, UNCERTAIN)),
                               data_cutoff_utc=t(31))
        self.assertEqual(ticks[1].distribution.bin_counts, (1, 0, 0, 0, 0))

    def test_cadence_continues_without_input(self):
        signal, adapters = producer(spies=True)
        ticks = signal.advance(data_cutoff_utc=t(91))
        self.assertEqual([tick.observation_time_utc for tick in ticks], [t(0), t(30), t(60), t(90)])
        self.assertTrue(all(tick.status == NO_INPUT for tick in ticks))
        for adapter in adapters.values():
            adapter.measure.assert_not_called()

    def test_cutoff_tick_deferred_and_future_input_rejected(self):
        signal, _ = producer()
        first = signal.advance((stance(0, "a", "A", 1),), data_cutoff_utc=t(30))
        self.assertEqual([tick.observation_time_utc for tick in first], [t(0)])
        with self.assertRaisesRegex(ValueError, "strictly before"):
            signal.advance((stance(30, "b", "B", 5),), data_cutoff_utc=t(30))
        second = signal.advance((stance(30, "b", "B", 5), stance(30, "c", "C", 5)),
                                data_cutoff_utc=t(31))
        self.assertEqual(second[0].distribution.support_count, 3)
        for tick in first + second:
            self.assertLess(tick.observation_time_utc, tick.data_cutoff_utc)

    def test_global_grid_and_configured_origin(self):
        signal, _ = producer(config(start_time_utc=t(7), end_time_utc=t(91)))
        self.assertEqual([x.observation_time_utc for x in signal.advance(data_cutoff_utc=t(100))],
                         [t(30), t(60), t(90)])
        signal, _ = producer(config(start_time_utc=t(7), tick_origin_utc=t(5), end_time_utc=t(91)))
        self.assertEqual([x.observation_time_utc for x in signal.advance(data_cutoff_utc=t(100))],
                         [t(35), t(65)])

    def test_execution_end_exclusive(self):
        signal, _ = producer(config(end_time_utc=t(60)))
        self.assertEqual([x.observation_time_utc for x in signal.advance(data_cutoff_utc=t(100))], [t(0), t(30)])

    def test_order_regression_rejected_without_state_mutation(self):
        signal, _ = producer()
        for records in ((stance(20, "b", "A"), stance(0, "a", "A")),
                        (stance(0, "z", "A"), stance(0, "a", "A"))):
            with self.assertRaisesRegex(ValueError, "order"):
                signal.advance(records, data_cutoff_utc=t(31))
        self.assertEqual(signal.advance((stance(0, "a", "A"),), data_cutoff_utc=t(1))[0].status, VALUE_AVAILABLE)

    def test_late_input_cutoff_regression_and_duplicate_rejected(self):
        signal, _ = producer()
        signal.advance((stance(0, "a", "A"),), data_cutoff_utc=t(10))
        for records, cutoff in (((stance(5, "b", "B"),), t(11)),
                                ((), t(9)), ((stance(10, "a", "A"),), t(11))):
            with self.assertRaises(ValueError):
                signal.advance(records, data_cutoff_utc=cutoff)
        self.assertEqual(signal.advance(data_cutoff_utc=t(10)), ())

    def test_proposition_filter(self):
        signal, _ = producer()
        tick = signal.advance((stance(0, "a", "A", 1), stance(0, "b", "B", 5, proposition="other")),
                              data_cutoff_utc=t(1))[0]
        self.assertEqual(tick.distribution.input_count, 1)
        self.assertEqual(tick.distribution.support_count, 1)

    def test_activity_without_opinions_never_calls_measures(self):
        signal, adapters = producer(spies=True)
        records = tuple(stance(0, str(i), "A", None, status)
                        for i, status in enumerate((NOT_APPLICABLE, UNCERTAIN, SKIPPED, ERROR)))
        tick = signal.advance(records, data_cutoff_utc=t(1))[0]
        self.assertEqual(tick.status, NO_CLASSIFIABLE_OPINIONS)
        self.assertEqual(tick.distribution.input_count, 4)
        self.assertEqual(tick.distribution.classified_comment_count, 0)
        self.assertEqual(tick.measurements, ())
        for observation in tick.observations:
            self.assertIsNone(observation.value)
        for adapter in adapters.values():
            adapter.measure.assert_not_called()

    def test_no_representable_authors_never_calls_measures(self):
        signal, adapters = producer(spies=True)
        tick = signal.advance((stance(0, "a", None, 1),), data_cutoff_utc=t(1))[0]
        self.assertEqual(tick.status, NO_REPRESENTABLE_AUTHORS)
        self.assertEqual(tick.distribution.classified_comment_count, 1)
        self.assertEqual(tick.distribution.excluded_counts["missing_author"], 1)
        self.assertIn("missing_author", tick.distribution.quality_reasons)
        for adapter in adapters.values():
            adapter.measure.assert_not_called()

    def test_one_distribution_built_and_shared_by_all_measures(self):
        signal, adapters = producer(spies=True)
        with patch.object(signal, "_build_distribution", wraps=signal._build_distribution) as build:
            tick = signal.advance((stance(0, "a", "A"),), data_cutoff_utc=t(1))[0]
        build.assert_called_once()
        for adapter in adapters.values():
            self.assertIs(adapter.measure.call_args.args[0], tick.distribution)
        self.assertEqual([o.signal_id for o in tick.observations], [f"polarization_{m}" for m in MEASURES])

    def test_quality_literal_propagation_and_optional_low_support(self):
        signal, _ = producer(config(min_support_threshold=3))  # TEST_ONLY threshold
        records = (stance(0, "a", "A", 1), stance(0, "b", None, 5),
                   stance(0, "c", "B", None, UNCERTAIN, quality="degraded_context"))
        tick = signal.advance(records, data_cutoff_utc=t(1))[0]
        self.assertEqual(tick.distribution.quality, "low_support")
        self.assertEqual(set(tick.distribution.quality_reasons),
                         {"missing_author", "uncertain_input", "low_support", "input_quality"})
        for measurement, observation in zip(tick.measurements, tick.observations):
            self.assertEqual(measurement.quality, tick.distribution.quality)
            self.assertEqual(observation.quality, measurement.quality)
            self.assertEqual(observation.metadata["quality_reasons"], measurement.quality_reasons)
            expected = create_polarization_measure(measurement.measure_id).measure(tick.distribution)
            self.assertEqual(observation.value, expected.value)

    def test_same_pi_different_support(self):
        small, _ = producer()
        large, _ = producer()
        a = small.advance((stance(0, "a", "A", 1), stance(0, "b", "B", 5)), data_cutoff_utc=t(1))[0]
        b = large.advance(tuple(stance(0, str(i), str(i), 1 if i < 2 else 5) for i in range(4)),
                          data_cutoff_utc=t(1))[0]
        self.assertEqual(a.distribution.weights, b.distribution.weights)
        self.assertNotEqual(a.distribution.support_count, b.distribution.support_count)
        for x, y in zip(a.observations, b.observations):
            self.assertAlmostEqual(x.value, y.value, places=12)

    def test_batching_invariance_and_no_duplicate_ticks(self):
        records = synthetic_fixture()
        once, _ = producer()
        chunks, _ = producer()
        a = once.advance(records, data_cutoff_utc=t(181))
        b = chunks.advance(records[:2], data_cutoff_utc=t(30))
        b += chunks.advance(records[2:4], data_cutoff_utc=t(40))
        b += chunks.advance(records[4:], data_cutoff_utc=t(181))
        self.assertEqual([(x.observation_time_utc, x.distribution.bin_counts) for x in a],
                         [(x.observation_time_utc, x.distribution.bin_counts) for x in b])
        self.assertEqual(chunks.advance(data_cutoff_utc=t(181)), ())

    def test_measure_failure_does_not_commit_state(self):
        signal, adapters = producer(spies=True)
        adapters["mec"].measure.side_effect = RuntimeError("calculation failure")
        with self.assertRaisesRegex(RuntimeError, "calculation failure"):
            signal.advance((stance(0, "a", "A"),), data_cutoff_utc=t(1))
        adapters["mec"].measure.side_effect = None
        self.assertEqual(signal.advance((stance(0, "a", "A"),), data_cutoff_utc=t(1))[0].status, VALUE_AVAILABLE)


class TemporalConfigurationTests(unittest.TestCase):
    def test_json_loading_resolution_serialization_composition(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.json"
            path.write_text(json.dumps(payload()), encoding="utf-8")
            requested = load_run_config(path)
            resolved = resolve_run_config(requested, base_dir=Path(directory))
            actual = resolved.config
            self.assertEqual(actual.signals.polarization, requested.signals.polarization)
            mapping = run_config_to_mapping(actual)
            self.assertNotIn("measure_id", mapping["polarization"])
            self.assertEqual(mapping["signals"]["polarization"]["enabled_measures"], list(MEASURES))
            # Legacy resolved_config retains null optional root sections; the
            # requested-JSON loader expects absent sections instead of nulls.
            self.assertEqual(run_config_from_mapping({k: v for k, v in mapping.items() if v is not None}), actual)
            signal = create_configured_polarization_signal(actual)
            self.assertEqual(len(signal.advance((stance(0, "a", "A"),), data_cutoff_utc=t(1))[0].observations), 3)

    def test_isolated_measure_selector_remains_compatible(self):
        requested = run_config_from_mapping({"identity": {"run_id": "isolated"},
                                             "polarization": {"measure_id": "mec", "mec": {}}})
        self.assertEqual(requested.polarization.selected_config, requested.polarization.mec)
        with self.assertRaisesRegex(ValueError, "measure_id is required"):
            run_config_from_mapping({"identity": {"run_id": "isolated"}, "polarization": {"mec": {}}})

    def test_missing_enabled_parameters_and_unknown_fields(self):
        for change in ("missing", "unknown", "no_parameters"):
            requested = payload()
            if change == "missing":
                del requested["polarization"]["emd_pol"]
            elif change == "unknown":
                requested["signals"]["polarization"]["threshold"] = 1
            else:
                del requested["polarization"]
            with self.assertRaises(ValueError):
                run_config_from_mapping(requested)

    def test_invalid_temporal_config(self):
        cases = ({"window": "0s"}, {"cadence": "-1s"}, {"window": "NaT"},
                 {"cadence": "1ns"}, {"interval_policy": "open"}, {"time_basis": "ingested"},
                 {"timezone": "America/Bogota"}, {"enabled_measures": ()},
                 {"enabled_measures": ("mec", "mec")}, {"enabled_measures": ("unknown",)},
                 {"enabled_measures": "mec"}, {"min_support_threshold": True},
                 {"min_support_threshold": 0}, {"min_support_threshold": float("nan")},
                 {"end_time_utc": t(0)}, {"tick_origin_utc": datetime(2026, 1, 1)},
                 {"distribution_policy_id": "comment_mass"})
        for changes in cases:
            with self.subTest(changes=changes), self.assertRaises((ValueError, TypeError)):
                config(**changes)

    def test_config_immutable_and_parameter_changes_visible_in_hash(self):
        requested = run_config_from_mapping(payload())
        with self.assertRaises(FrozenInstanceError):
            requested.signals.polarization.window = "90s"
        changed = payload()
        changed["signals"]["polarization"]["window"] = "90s"
        self.assertNotEqual(run_config_hash(requested), run_config_hash(run_config_from_mapping(changed)))

    def test_factory_requires_exact_resolved_configs(self):
        for measure_configs in ({}, {name: None for name in MEASURES}):
            with self.assertRaises(ValueError):
                create_polarization_temporal_signal(config(), measure_configs)


if __name__ == "__main__":
    unittest.main()
