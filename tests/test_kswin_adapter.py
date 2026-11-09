from __future__ import annotations

import inspect
import json
import tempfile
import unittest
from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean
from unittest.mock import Mock, patch

from river.drift import KSWIN

from youtube_pipeline.activity_detection import (
    ActivityDetectionRouteConfig, DetectionResult, dispatch_activity_observation,
)
from youtube_pipeline.activity_signals import ActivityObservation, event_window_comment_count_definition
from youtube_pipeline.comment_stance import CommentStance
from youtube_pipeline.configuration import (
    KSWINConfig, load_run_config, resolve_run_config, run_config_from_mapping,
    run_config_hash, run_config_to_mapping,
)
from youtube_pipeline.detectors import (
    ADWINAdapter, KSWINAdapter, PageHinkleyAdapter, create_detector, get_detector_names,
)
from youtube_pipeline.event_candidates import EventCandidateLineage, promote_detection_result
from youtube_pipeline.polarization_signal_composition import create_configured_polarization_signal
from youtube_pipeline.signal_observations import (
    NO_CLASSIFIABLE_OPINIONS, NO_INPUT, NO_REPRESENTABLE_AUTHORS,
    VALUE_AVAILABLE, SignalObservation,
)


START = datetime(2026, 1, 1, tzinfo=timezone.utc)
# TEST_ONLY sensitivity/history/sample sizes: not calibrated event parameters.
TEST_CONFIG = KSWINConfig(alpha=.005, window_size=120, stat_size=40, seed=42)
ACTIVITY_SIGNAL = event_window_comment_count_definition(
    window="120s", cadence="30s", time_basis="event_time_utc",
)


def observation(value=.5, index=0, signal_id="polarization_mec", quality="passed",
                status=VALUE_AVAILABLE):
    return SignalObservation(signal_id, START + timedelta(seconds=index), value, status,
                             2 if status == VALUE_AVAILABLE else 0, quality, {})


def activity(value=.5, index=0):
    timestamp = START + timedelta(seconds=index)
    return ActivityObservation(ACTIVITY_SIGNAL, timestamp, timestamp - timedelta(seconds=120),
                               timestamp, value, 2, "passed")


def streams():
    return {
        "stable": [.5 + .01 * (i % 3 - 1) for i in range(360)],
        "mean": [.2 + .01 * (i % 3 - 1) for i in range(180)]
                + [.7 + .01 * (i % 3 - 1) for i in range(180)],
        "distribution": [.5 + .01 * (i % 3 - 1) for i in range(180)]
                        + [.2 if i % 2 == 0 else .8 for i in range(180)],
    }


def results(adapter, values):
    return [adapter.on_observation(observation(value, i)) for i, value in enumerate(values)]


def payload(seed=42):
    return {
        "identity": {"run_id": "synthetic_kswin"},
        "detection": {
            "activity_route": {"signal_id": "polarization_mec", "detector_id": "kswin"},
            "kswin": {"seed": seed},
        },
    }


class KSWINConfigTests(unittest.TestCase):
    def test_defaults_match_installed_river(self):
        config = KSWINConfig()
        signature = inspect.signature(KSWIN)
        for name in ("alpha", "window_size", "stat_size", "seed"):
            self.assertEqual(getattr(config, name), signature.parameters[name].default)
        self.assertIsNone(config.seed)  # unseeded mode is explicit, not reproducible

    def test_custom_values_immutable_slots(self):
        config = KSWINConfig(alpha=.01, window_size=80, stat_size=25, seed=-7)
        self.assertEqual((config.alpha, config.window_size, config.stat_size, config.seed), (.01, 80, 25, -7))
        self.assertFalse(hasattr(config, "__dict__"))
        with self.assertRaises(FrozenInstanceError):
            config.seed = 1

    def test_invalid_alpha(self):
        for value in (True, "0.01", None, float("nan"), float("inf"), -.1, 1.1):
            with self.subTest(value=value), self.assertRaises((TypeError, ValueError)):
                KSWINConfig(alpha=value)
        # River allows endpoints; do not invent a stricter statistical range.
        self.assertEqual(KSWINConfig(alpha=0).alpha, 0.)
        self.assertEqual(KSWINConfig(alpha=1).alpha, 1.)

    def test_invalid_sizes_and_seed(self):
        for name in ("window_size", "stat_size"):
            for value in (True, 1.5, "100", None, float("inf"), float("nan"), 0, -1):
                with self.subTest(name=name, value=value), self.assertRaises((TypeError, ValueError)):
                    KSWINConfig(**{name: value})
        for seed in (True, 1.5, "42", float("nan"), float("inf"), []):
            with self.subTest(seed=seed), self.assertRaises(TypeError):
                KSWINConfig(seed=seed)

    def test_sampling_precondition_backed_by_public_update(self):
        # River accepts this constructor but cannot draw 30 distinct values from 20.
        direct = KSWIN(window_size=50, stat_size=30, seed=42)
        with self.assertRaises(ValueError):
            for _ in range(50):
                direct.update(.5)
        with self.assertRaisesRegex(ValueError, "sampling without replacement"):
            KSWINConfig(window_size=50, stat_size=30)
        # Equality is operationally valid: sample the entire historical segment.
        config = KSWINConfig(window_size=60, stat_size=30, seed=42)
        self.assertFalse(any(r.triggered for r in results(KSWINAdapter(config=config), [.5] * 61)))

    def test_unknown_fields_and_invalid_config_objects(self):
        for payload in ({"random_state": 1}, {"window": [.5]}, {"cadence": "30s"}):
            with self.assertRaisesRegex(ValueError, "Unknown KSWIN"):
                KSWINConfig.from_mapping(payload)
        with self.assertRaises(TypeError):
            KSWINConfig.from_mapping([])
        with self.assertRaises(TypeError):
            KSWINAdapter(config={})

    def test_json_load_resolve_serialize_and_registry(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "kswin.json"
            path.write_text(json.dumps(payload()), encoding="utf-8")
            config = load_run_config(path)
            resolved = resolve_run_config(config, base_dir=directory)
            self.assertEqual(resolved.config.detection.kswin, KSWINConfig(seed=42))
            serialized = run_config_to_mapping(resolved.config)
            self.assertEqual(serialized["detection"]["kswin"],
                             {"alpha": .005, "window_size": 100, "stat_size": 30, "seed": 42})
            canonical = json.loads(resolved.canonical_json)
            self.assertEqual(canonical["detection"]["kswin"], serialized["detection"]["kswin"])
            self.assertIsInstance(create_detector("kswin", config=config.detection.kswin), KSWINAdapter)

    def test_parameter_and_seed_changes_change_current_hash(self):
        config = run_config_from_mapping(payload())
        for key, value in (("alpha", .01), ("window_size", 120), ("stat_size", 20), ("seed", 7)):
            changed = payload()
            changed["detection"]["kswin"][key] = value
            self.assertNotEqual(run_config_hash(config), run_config_hash(run_config_from_mapping(changed)))
        changed = payload()
        changed["identity"]["run_id"] = "other_run"
        self.assertNotEqual(run_config_hash(config), run_config_hash(run_config_from_mapping(changed)))

    def test_config_requires_matching_route_parameters(self):
        requested = payload()
        requested["detection"].pop("kswin")
        with self.assertRaisesRegex(ValueError, "matching detector configuration"):
            run_config_from_mapping(requested)
        requested = payload()
        requested["detection"]["kswin"]["random_state"] = 42
        with self.assertRaises(ValueError):
            run_config_from_mapping(requested)

    def test_legacy_configs_omit_unselected_kswin(self):
        config = run_config_from_mapping({"identity": {"run_id": "legacy"}, "detection": {"adwin": {}}})
        self.assertNotIn("kswin", run_config_to_mapping(config)["detection"])


class KSWINAdapterTests(unittest.TestCase):
    def test_mapping_metadata_and_result_immutability(self):
        adapter = KSWINAdapter(config=TEST_CONFIG)
        original = observation(-.125, quality="degraded_missing_author")
        result = adapter.on_observation(original)
        self.assertIsInstance(result, DetectionResult)
        self.assertEqual(result.detector_id, "kswin")
        self.assertEqual(result.signal_id, original.signal_id)
        self.assertEqual(result.observation_time_utc, original.observation_time_utc)
        self.assertEqual(result.quality, original.quality)
        self.assertFalse(result.triggered)
        self.assertIsNone(result.score)
        self.assertEqual(dict(result.detector_metadata), {
            "observations_processed": 1, "segment_observations": 1, "window_length": 1,
            "window_size": 120, "stat_size": 40, "alpha": .005, "seed": 42,
        })
        json.dumps(dict(result.detector_metadata), allow_nan=False)
        with self.assertRaises(FrozenInstanceError):
            result.triggered = True
        with self.assertRaises(TypeError):
            result.detector_metadata["seed"] = 7
        adapter.on_observation(observation(.25, 1))
        self.assertEqual(result.detector_metadata["observations_processed"], 1)

    def test_factory_uses_typed_config_only(self):
        with patch("youtube_pipeline.detectors.drift.KSWIN", wraps=KSWIN) as library:
            create_detector("kswin", config=TEST_CONFIG)
        library.assert_called_once_with(alpha=.005, window_size=120, stat_size=40, seed=42)
        with self.assertRaises(TypeError):
            create_detector("kswin", alpha=.01)

    def test_activity_numeric_observation_compatibility(self):
        adapter = create_detector("kswin", config=TEST_CONFIG)
        result = dispatch_activity_observation(
            route=ActivityDetectionRouteConfig(ACTIVITY_SIGNAL.signal_id, "kswin"),
            observation=activity(10.5), detectors={"kswin": adapter},
        )
        self.assertEqual(result.signal_id, ACTIVITY_SIGNAL.signal_id)
        self.assertEqual(result.detector_metadata["observations_processed"], 1)

    def test_stable_stream(self):
        output = results(KSWINAdapter(config=TEST_CONFIG), streams()["stable"])
        self.assertFalse(any(r.triggered for r in output))  # this fixture, not a universal guarantee
        self.assertEqual(output[-1].detector_metadata["window_length"], TEST_CONFIG.window_size)
        self.assertEqual(output[-1].detector_metadata["observations_processed"], 360)

    def test_mean_shift(self):
        output = results(KSWINAdapter(config=TEST_CONFIG), streams()["mean"])
        self.assertFalse(any(r.triggered for r in output[:180]))
        self.assertTrue(any(r.triggered for r in output[180:]))

    def test_distribution_shift_with_same_mean(self):
        values = streams()["distribution"]
        self.assertAlmostEqual(mean(values[:180]), mean(values[180:]), places=12)
        output = results(KSWINAdapter(config=TEST_CONFIG), values)
        self.assertFalse(any(r.triggered for r in output[:180]))
        self.assertTrue(any(r.triggered for r in output[180:]))

    def test_seeded_determinism(self):
        for values in streams().values():
            first = results(KSWINAdapter(config=TEST_CONFIG), values)
            second = results(KSWINAdapter(config=TEST_CONFIG), values)
            self.assertEqual(first, second)

    def test_equivalence_to_installed_river_and_public_state(self):
        direct = KSWIN(alpha=TEST_CONFIG.alpha, window_size=TEST_CONFIG.window_size,
                       stat_size=TEST_CONFIG.stat_size, seed=TEST_CONFIG.seed)
        adapter = KSWINAdapter(config=TEST_CONFIG)
        for i, value in enumerate(streams()["distribution"]):
            direct.update(value)
            result = adapter.on_observation(observation(value, i))
            self.assertEqual(result.triggered, bool(direct.drift_detected))
            self.assertEqual(result.detector_metadata["window_length"], len(direct.window))
            self.assertEqual(result.detector_metadata["segment_observations"], direct.n)

    def test_reset_after_drift_and_gap_preserves_pending_reset(self):
        adapter = KSWINAdapter(config=TEST_CONFIG)
        trigger = next(r for i, value in enumerate(streams()["mean"])
                       if (r := adapter.on_observation(observation(value, i))).triggered)
        self.assertEqual(trigger.detector_metadata["window_length"], TEST_CONFIG.stat_size)
        route = ActivityDetectionRouteConfig("polarization_mec", "kswin")
        with patch.object(adapter, "on_observation", wraps=adapter.on_observation) as update:
            self.assertIsNone(dispatch_activity_observation(
                route=route, observation=observation(None, 1000, status=NO_INPUT), detectors={"kswin": adapter},
            ))
            update.assert_not_called()
        self.assertEqual(adapter.observations_processed, trigger.detector_metadata["observations_processed"])
        after = adapter.on_observation(observation(.7, 1001))
        self.assertFalse(after.triggered)
        self.assertEqual(after.detector_metadata["window_length"], 1)
        self.assertEqual(after.detector_metadata["segment_observations"], 1)
        self.assertEqual(after.detector_metadata["observations_processed"], adapter.observations_processed)

    def test_triggered_activity_result_promotes_without_candidate_changes(self):
        adapter = KSWINAdapter(config=TEST_CONFIG)
        for i, value in enumerate(streams()["mean"]):
            original = activity(value, i)
            result = adapter.on_observation(original)
            if result.triggered:
                break
        self.assertTrue(result.triggered)
        candidate = promote_detection_result(
            candidate_id="synthetic_kswin", observation=original, detection_result=result,
            lineage=EventCandidateLineage("synthetic", "synthetic_hash", "synthetic:test"),
            lifecycle_state="point",
        )
        self.assertEqual(candidate.detector_id, "kswin")


class NeutralNumericDispatchTests(unittest.TestCase):
    def test_registry_contains_all_detectors(self):
        self.assertEqual(set(get_detector_names()), {"xiao_ema", "page_hinkley", "adwin", "kswin"})

    def test_all_absences_bypass_each_numeric_adapter(self):
        for name, adapter in (("page_hinkley", PageHinkleyAdapter()), ("adwin", ADWINAdapter()),
                              ("kswin", KSWINAdapter(config=TEST_CONFIG))):
            route = ActivityDetectionRouteConfig("polarization_mec", name)
            adapter.on_observation(observation(.5))
            processed = adapter.observations_processed
            with patch.object(adapter, "on_observation", wraps=adapter.on_observation) as update:
                for status in (NO_INPUT, NO_CLASSIFIABLE_OPINIONS, NO_REPRESENTABLE_AUTHORS):
                    self.assertIsNone(dispatch_activity_observation(
                        route=route, observation=observation(None, status=status), detectors={name: adapter},
                    ))
                update.assert_not_called()
            self.assertEqual(adapter.observations_processed, processed)

    def test_direct_adapters_reject_absent_values_before_state_change(self):
        for adapter in (PageHinkleyAdapter(), ADWINAdapter(), KSWINAdapter(config=TEST_CONFIG)):
            with self.assertRaisesRegex(ValueError, "VALUE_AVAILABLE"):
                adapter.on_observation(observation(None, status=NO_INPUT))
            self.assertEqual(adapter.observations_processed, 0)
            with self.assertRaises(TypeError):
                adapter.on_observation(object())

    def test_nonfinite_and_none_are_not_numeric_observations(self):
        for value in (None, float("inf"), float("nan"), True):
            with self.assertRaises((TypeError, ValueError)):
                observation(value)

    def test_coexistence_same_route_contract_no_shared_state(self):
        adapters = {"page_hinkley": PageHinkleyAdapter(), "adwin": ADWINAdapter(),
                    "kswin": KSWINAdapter(config=TEST_CONFIG)}
        for name in adapters:
            result = dispatch_activity_observation(
                route=ActivityDetectionRouteConfig("polarization_mec", name),
                observation=observation(.2, quality="degraded"), detectors=adapters,
            )
            self.assertEqual(result.detector_id, name)
            self.assertEqual(result.quality, "degraded")
            self.assertIsNone(result.score)
        adapters["kswin"].on_observation(observation(.3, 1))
        self.assertEqual([a.observations_processed for a in adapters.values()], [1, 1, 2])
        self.assertIsNot(create_detector("kswin", config=TEST_CONFIG), create_detector("kswin", config=TEST_CONFIG))

    def test_absence_still_validates_route_and_detector_availability(self):
        empty = observation(None, status=NO_INPUT)
        for route, detectors in ((ActivityDetectionRouteConfig("wrong", "kswin"), {"kswin": KSWINAdapter()}),
                                 (ActivityDetectionRouteConfig("polarization_mec", "kswin"), {})):
            with self.assertRaises(ValueError):
                dispatch_activity_observation(route=route, observation=empty, detectors=detectors)

    def test_available_values_validate_detector_result_contract(self):
        good = KSWINAdapter(config=TEST_CONFIG).on_observation(observation(.5))
        wrong = (None, replace(good, detector_id="adwin"), replace(good, signal_id="other"),
                 replace(good, observation_time_utc=START + timedelta(seconds=1)))
        for result in wrong:
            detector = Mock(on_observation=Mock(return_value=result))
            with self.assertRaises((TypeError, ValueError)):
                dispatch_activity_observation(
                    route=ActivityDetectionRouteConfig("polarization_mec", "kswin"),
                    observation=observation(.5), detectors={"kswin": detector},
                )

    def test_polarization_temporal_observations_route_without_measure_knowledge(self):
        requested = payload()
        requested["signals"] = {"polarization": {
            "proposition_id": "p", "classification_run_id": "synthetic_only",
            "window": "60s", "cadence": "30s",  # TEST_ONLY
            "start_time_utc": START.isoformat(),
            "end_time_utc": (START + timedelta(seconds=121)).isoformat(),
            "enabled_measures": ["mec", "esteban_ray", "emd_pol"],
        }}
        requested["polarization"] = {"mec": {}, "esteban_ray": {}, "emd_pol": {}}
        config = run_config_from_mapping(requested)
        temporal = create_configured_polarization_signal(config)
        records = (
            CommentStance("a0", "A", START, "p", 3, "CLASSIFIED"),
            CommentStance("b0", "B", START, "p", 3, "CLASSIFIED"),
            CommentStance("a1", "A", START + timedelta(seconds=30), "p", 1, "CLASSIFIED"),
            CommentStance("b1", "B", START + timedelta(seconds=30), "p", 5, "CLASSIFIED"),
        )
        ticks = temporal.advance(records, data_cutoff_utc=START + timedelta(seconds=122))
        for measure in ("mec", "esteban_ray", "emd_pol"):
            adapter = create_detector("kswin", config=config.detection.kswin)
            route = ActivityDetectionRouteConfig(f"polarization_{measure}", "kswin")
            decisions = []
            for tick in ticks:
                original = next(o for o in tick.observations if o.signal_id == route.signal_id)
                result = dispatch_activity_observation(route=route, observation=original, detectors={"kswin": adapter})
                if original.status == VALUE_AVAILABLE:
                    decisions.append(result)
                    self.assertEqual(result.signal_id, original.signal_id)
                    self.assertEqual(result.observation_time_utc, original.observation_time_utc)
                    self.assertEqual(result.quality, original.quality)
                else:
                    self.assertIsNone(result)
            self.assertEqual(adapter.observations_processed, len(decisions))
            self.assertEqual(len(decisions), 4)


if __name__ == "__main__":
    unittest.main()
