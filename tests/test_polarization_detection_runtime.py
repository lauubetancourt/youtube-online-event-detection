from __future__ import annotations

import asyncio
import csv
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from youtube_pipeline.configuration import (
    resolve_run_config,
    run_config_from_mapping,
    run_config_hash,
)
from youtube_pipeline.polarization_detection_runtime import (
    DETECTION_ARTIFACT,
    DETECTION_MANIFEST,
    run_polarization_detection_runtime,
)
from youtube_pipeline.polarization_runtime import run_polarization_runtime
from youtube_pipeline.polarization_runtime_config import PolarizationRuntimeConfig
from youtube_pipeline.signal_observations import (
    NO_CLASSIFIABLE_OPINIONS,
    NO_INPUT,
    NO_REPRESENTABLE_AUTHORS,
    VALUE_AVAILABLE,
    SignalObservation,
)
from youtube_pipeline.stance_classifier import StanceModelResponse, TokenUsage


START = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _payload(
    routes: list[dict[str, str]],
    *,
    page_hinkley: dict | None = None,
    adwin: dict | None = None,
    kswin: dict | None = None,
) -> dict:
    detection: dict = {"routes": routes}
    for name, value in (
        ("page_hinkley", page_hinkley),
        ("adwin", adwin),
        ("kswin", kswin),
    ):
        if value is not None:
            detection[name] = value
    return {
        "identity": {"run_id": "polarization-detection-test"},
        "signals": {"polarization": {
            "proposition_id": "p",
            "classification_run_id": "classifications",
            "window": "60s",
            "cadence": "1s",
            "start_time_utc": START.isoformat(),
            "end_time_utc": (START + timedelta(hours=1)).isoformat(),
            "enabled_measures": ["esteban_ray", "emd_pol", "mec"],
        }},
        "polarization": {"esteban_ray": {}, "emd_pol": {}, "mec": {}},
        "detection": detection,
    }


def _observation(
    signal_id: str,
    value: float | None,
    index: int,
    *,
    status: str = VALUE_AVAILABLE,
    quality: str = "passed",
    support: int = 2,
) -> tuple[SignalObservation, str]:
    timestamp = START + timedelta(seconds=index)
    measure_id = signal_id.removeprefix("polarization_")
    observation = SignalObservation(
        signal_id=signal_id,
        observation_time_utc=timestamp,
        value=value,
        status=status,
        support_count=support if status == VALUE_AVAILABLE else 0,
        quality=quality,
        metadata={
            "measure_id": measure_id,
            "distribution_policy_id": "likert5_author_latest_classified_v1",
            "classification_run_id": "classifications",
            "data_cutoff_utc": (START + timedelta(hours=2)).isoformat(),
        },
    )
    return observation, timestamp.isoformat()


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class FakeAllClassifiedTransport:
    async def complete(self, *, config, messages):
        return StanceModelResponse(
            json.dumps({"status": "CLASSIFIED", "stance": 5}),
            TokenUsage(2, 3, 5),
        )


class PolarizationDetectionRuntimeTests(unittest.TestCase):
    def test_silver_runtime_schema_requires_causal_source_role(self) -> None:
        config = PolarizationRuntimeConfig(
            video_ids=("v1",),
            input_schema="silver_comments_v1",
            source_role="CAUSAL_MINIMAL_V1",
        )
        self.assertEqual(config.input_schema, "silver_comments_v1")
        with self.assertRaisesRegex(ValueError, "source_role"):
            PolarizationRuntimeConfig(
                video_ids=("v1",),
                input_schema="silver_comments_v1",
                source_role="PROVISIONAL_RUNTIME_INPUT",
            )

    def test_routes_three_measures_to_independent_detector_types(self) -> None:
        config = run_config_from_mapping(_payload(
            [
                {"signal_id": "polarization_mec", "detector_id": "kswin"},
                {"signal_id": "polarization_esteban_ray", "detector_id": "adwin"},
                {"signal_id": "polarization_emd_pol", "detector_id": "page_hinkley"},
            ],
            page_hinkley={"min_instances": 5, "threshold": 2, "delta": 0, "alpha": 1},
            adwin={"delta": .01, "clock": 1, "min_window_length": 5, "grace_period": 10},
            kswin={"alpha": .005, "window_size": 120, "stat_size": 40, "seed": 42},
        ))
        observations = []
        signals = (
            "polarization_esteban_ray", "polarization_emd_pol", "polarization_mec"
        )
        for index in range(3):
            for signal in signals:
                observations.append(_observation(
                    signal,
                    None if index == 1 else .2,
                    index,
                    status=NO_INPUT if index == 1 else VALUE_AVAILABLE,
                    quality="degraded" if index == 2 else "passed",
                ))
        with tempfile.TemporaryDirectory() as directory:
            artifacts = run_polarization_detection_runtime(
                config,
                observations,
                output_dir=directory,
                config_hash=run_config_hash(config),
                input_policy_manifest={"input_policy_id": "causal_minimal_v1"},
            )
            rows = _read_jsonl(artifacts.result_path)
            manifest = json.loads(artifacts.manifest_path.read_text())
        self.assertEqual(artifacts.results_produced, 6)
        self.assertEqual(artifacts.observations_skipped_no_value, 3)
        self.assertEqual({row["signal_id"] for row in rows}, set(signals))
        self.assertEqual(
            {row["signal_id"]: row["detector_id"] for row in rows},
            {
                "polarization_esteban_ray": "adwin",
                "polarization_emd_pol": "page_hinkley",
                "polarization_mec": "kswin",
            },
        )
        self.assertEqual([row["quality"] for row in rows[3:]], [
            "degraded", "degraded", "degraded",
        ])
        self.assertEqual(manifest["gap_policy"]["methodological_status"], "OPEN")
        self.assertEqual(manifest["input_policy"]["input_policy_id"], "causal_minimal_v1")
        self.assertEqual(manifest["causality"]["future_leak_count"], 0)

    def test_all_no_value_statuses_skip_updates_and_preserve_state(self) -> None:
        config = run_config_from_mapping(_payload(
            [{"signal_id": "polarization_mec", "detector_id": "kswin"}],
            kswin={"window_size": 60, "stat_size": 30, "seed": 7},
        ))
        observations = [_observation("polarization_mec", .2, 0)]
        for index, status in enumerate((
            NO_INPUT, NO_CLASSIFIABLE_OPINIONS, NO_REPRESENTABLE_AUTHORS
        ), start=1):
            observations.append(_observation(
                "polarization_mec", None, index, status=status
            ))
        observations.append(_observation("polarization_mec", .3, 4, quality="low_support"))
        with tempfile.TemporaryDirectory() as directory:
            artifacts = run_polarization_detection_runtime(
                config, observations, output_dir=directory,
                config_hash=run_config_hash(config),
            )
            rows = _read_jsonl(artifacts.result_path)
        self.assertEqual(artifacts.observations_skipped_no_value, 3)
        self.assertEqual([row["detector_metadata"]["observations_processed"] for row in rows], [1, 2])
        self.assertEqual(rows[-1]["quality"], "low_support")
        self.assertEqual(rows[-1]["support_count"], 2)

    def test_same_detector_type_has_one_fresh_instance_per_signal(self) -> None:
        config = run_config_from_mapping(_payload(
            [
                {"signal_id": "polarization_mec", "detector_id": "kswin"},
                {"signal_id": "polarization_esteban_ray", "detector_id": "kswin"},
                {"signal_id": "polarization_emd_pol", "detector_id": "kswin"},
            ],
            kswin={"window_size": 60, "stat_size": 30, "seed": 42},
        ))
        observations = []
        for index in range(2):
            for signal in (
                "polarization_esteban_ray", "polarization_emd_pol", "polarization_mec"
            ):
                observations.append(_observation(signal, .2 + index / 10, index))
        with tempfile.TemporaryDirectory() as directory:
            artifacts = run_polarization_detection_runtime(
                config, observations, output_dir=directory,
                config_hash=run_config_hash(config),
            )
            rows = _read_jsonl(artifacts.result_path)
        counters: dict[str, list[int]] = {}
        for row in rows:
            counters.setdefault(row["signal_id"], []).append(
                row["detector_metadata"]["observations_processed"]
            )
        self.assertEqual(set(map(tuple, counters.values())), {(1, 2)})

    def test_synthetic_stream_eventually_triggers_each_detector(self) -> None:
        cases = {
            "page_hinkley": (
                {"page_hinkley": {
                    "min_instances": 5, "delta": 0, "threshold": 2,
                    "alpha": 1, "mode": "up",
                }},
                [0.0] * 30 + [10.0] * 30,
            ),
            "adwin": (
                {"adwin": {
                    "delta": .01, "clock": 1, "max_buckets": 5,
                    "min_window_length": 5, "grace_period": 10,
                }},
                [0.0] * 150 + [10.0] * 150,
            ),
            "kswin": (
                {"kswin": {
                    "alpha": .005, "window_size": 120,
                    "stat_size": 40, "seed": 42,
                }},
                [.2 + .01 * (i % 3 - 1) for i in range(180)]
                + [.7 + .01 * (i % 3 - 1) for i in range(180)],
            ),
        }
        for detector_id, (parameters, values) in cases.items():
            with self.subTest(detector_id=detector_id), tempfile.TemporaryDirectory() as directory:
                config = run_config_from_mapping(_payload(
                    [{"signal_id": "polarization_mec", "detector_id": detector_id}],
                    **parameters,
                ))
                observations = [
                    _observation("polarization_mec", value, index)
                    for index, value in enumerate(values)
                ]
                artifacts = run_polarization_detection_runtime(
                    config, observations, output_dir=directory,
                    config_hash=run_config_hash(config),
                )
                self.assertGreater(artifacts.trigger_count, 0)

    def test_routing_and_parameter_changes_change_config_hash(self) -> None:
        base = _payload(
            [{"signal_id": "polarization_mec", "detector_id": "kswin"}],
            kswin={"window_size": 60, "stat_size": 30, "seed": 42},
        )
        changed_route = _payload(
            [{"signal_id": "polarization_mec", "detector_id": "adwin"}],
            adwin={},
        )
        changed_parameter = json.loads(json.dumps(base))
        changed_parameter["detection"]["kswin"]["seed"] = 43
        hashes = {
            run_config_hash(run_config_from_mapping(payload))
            for payload in (base, changed_route, changed_parameter)
        }
        self.assertEqual(len(hashes), 3)

    def test_invalid_route_disabled_signal_duplicate_and_time_regression(self) -> None:
        mapped_signal_override = _payload([], kswin={})
        mapped_signal_override["detection"]["routes"] = {
            "polarization_mec": {
                "signal_id": "polarization_esteban_ray",
                "detector_id": "kswin",
            }
        }
        with self.assertRaisesRegex(ValueError, "Unknown .* fields"):
            run_config_from_mapping(mapped_signal_override)
        duplicate = _payload(
            [
                {"signal_id": "polarization_mec", "detector_id": "kswin"},
                {"signal_id": "polarization_mec", "detector_id": "kswin"},
            ],
            kswin={},
        )
        with self.assertRaisesRegex(ValueError, "unique signal_id"):
            run_config_from_mapping(duplicate)
        disabled = _payload(
            [{"signal_id": "polarization_unknown", "detector_id": "kswin"}],
            kswin={},
        )
        with self.assertRaisesRegex(ValueError, "disabled signals"):
            run_config_from_mapping(disabled)
        config = run_config_from_mapping(_payload(
            [{"signal_id": "polarization_mec", "detector_id": "kswin"}],
            kswin={},
        ))
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(ValueError, "regress"):
            run_polarization_detection_runtime(
                config,
                [_observation("polarization_mec", .2, 2), _observation("polarization_mec", .2, 1)],
                output_dir=directory,
                config_hash=run_config_hash(config),
            )

    def test_integrated_causal_runtime_persists_detection_without_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            comments = root / "silver.parquet"
            videos = root / "videos.csv"
            pd.DataFrame([
                {
                    "comment_id": "c1", "video_id": "v1", "author_id": "a1",
                    "event_time_utc": "2026-01-01T00:00:00Z", "text": "favor",
                    "is_reply": False, "reply_to_comment_id": None,
                },
                {
                    "comment_id": "c2", "video_id": "v1", "author_id": "a2",
                    "event_time_utc": "2026-01-01T00:00:20Z", "text": "favor",
                    "is_reply": False, "reply_to_comment_id": None,
                },
            ]).to_parquet(comments, index=False)
            with videos.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["video_id", "title", "publishedAt"])
                writer.writeheader()
                writer.writerow({"video_id": "v1", "title": "Fixture", "publishedAt": "2026-01-01T00:00:00Z"})
            payload = {
                "identity": {"run_id": "causal-runtime"},
                "data": {
                    "local_files": {"comments_path": str(comments), "videos_path": str(videos)},
                    "experimental_input": {},
                },
                "signals": {"polarization": {
                    "proposition_id": "p", "classification_run_id": "causal-runtime",
                    "window": "60s", "cadence": "30s",
                    "start_time_utc": "2026-01-01T00:00:00Z",
                    "end_time_utc": "2026-01-01T00:01:01Z",
                    "enabled_measures": ["esteban_ray", "emd_pol", "mec"],
                }},
                "polarization": {
                    "classifier": {"proposition": {
                        "proposition_id": "p", "proposition_text": "p?", "context": "c",
                    }},
                    "runtime": {"video_ids": ["v1"]},
                    "esteban_ray": {}, "emd_pol": {}, "mec": {},
                },
                "detection": {
                    "routes": {
                        "polarization_mec": {"detector_id": "kswin"},
                        "polarization_esteban_ray": {"detector_id": "adwin"},
                        "polarization_emd_pol": {"detector_id": "page_hinkley"},
                    },
                    "kswin": {"window_size": 60, "stat_size": 30, "seed": 42},
                    "adwin": {}, "page_hinkley": {},
                },
            }
            config = run_config_from_mapping(payload)
            output = root / "output"
            artifacts = asyncio.run(run_polarization_runtime(
                config,
                output_dir=output,
                base_dir=root,
                transport=FakeAllClassifiedTransport(),
            ))
            manifest = json.loads((output / DETECTION_MANIFEST).read_text())
            runtime_manifest = json.loads(artifacts.manifest_path.read_text())
            result_rows = _read_jsonl(output / DETECTION_ARTIFACT)
        self.assertEqual(artifacts.detection_result_count, 9)
        self.assertEqual(artifacts.detection_trigger_count, 0)
        self.assertEqual(manifest["input_policy"]["input_policy_id"], "causal_minimal_v1")
        self.assertFalse(manifest["downstream"]["event_candidates_created"])
        self.assertTrue(runtime_manifest["downstream"]["detectors_executed"])
        self.assertEqual(len(result_rows), 9)


if __name__ == "__main__":
    unittest.main()
