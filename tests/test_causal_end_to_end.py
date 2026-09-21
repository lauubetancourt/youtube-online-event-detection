from __future__ import annotations

import json
import hashlib
import shutil
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from youtube_pipeline.activity_detection import (
    ActivityDetectionRouteConfig,
    DetectionResult,
    observation_signal_id,
)
from youtube_pipeline.activity_signals import (
    create_activity_signal,
    event_window_comment_count_definition,
)
from youtube_pipeline.candidate_evidence import (
    CandidateEvidencePolicyConfig,
    IMMEDIATE_CAUSAL,
)
from youtube_pipeline.candidate_rag import CandidateRagValidationConfig
from youtube_pipeline.causal_end_to_end import run_causal_end_to_end_replay
from youtube_pipeline.causal_replay_config import CausalEndToEndReplayConfig
from youtube_pipeline.comment_stance import CLASSIFIED, CommentStance
from youtube_pipeline.configuration import (
    run_config_from_mapping,
    run_config_hash,
    run_config_to_mapping,
)
from youtube_pipeline.configuration.models import (
    DataConfig,
    DetectionConfig,
    PolarizationConfig,
    RagConfig,
    RunConfig,
    RunIdentityConfig,
    SignalsConfig,
    SimulationConfig,
)
from youtube_pipeline.detectors import KSWINConfig, PageHinkleyConfig
from youtube_pipeline.detection_alerts import DetectionGapPolicyConfig
from youtube_pipeline.detection_association import DetectionAssociationConfig
from youtube_pipeline.experimental_input import ExperimentalInputPolicyConfig
from youtube_pipeline.g2_gate import G2GatePolicyConfig
from youtube_pipeline.polarization_measures import MECConfig
from youtube_pipeline.polarization_runtime_config import (
    CAUSAL_MINIMAL_V1_INPUT,
    SILVER_COMMENTS_V1,
    PolarizationRuntimeConfig,
)
from youtube_pipeline.polarization_signals import PolarizationTemporalSignalConfig
from youtube_pipeline.rag_generation_g1 import (
    RAG_GENERATION_MANIFEST_FILE as G1_MANIFEST,
    RAG_RAW_MODEL_RESPONSES_FILE as G1_RAW,
    RAG_VALIDATION_REPORTS_FILE as G1_REPORTS,
)
from youtube_pipeline.rag_generation_g2_hierarchical import (
    RAG_EVENT_VALIDATION_SUMMARY_FILE,
    RAG_GENERATION_MANIFEST_FILE as G2_MANIFEST,
    RAG_RAW_MODEL_RESPONSES_FILE as G2_RAW,
    RAG_VIDEO_EXTERNAL_EVIDENCE_FILE,
    RAG_VIDEO_NEWS_QUERIES_FILE,
    RAG_VIDEO_VALIDATION_REPORTS_FILE,
)
from youtube_pipeline.stance_classifier import (
    CommentStanceClassifierConfig,
    StanceProposition,
)
from youtube_pipeline.storage import LocalFilesConfig
from youtube_pipeline.topic_profiles import TopicProfile


UTC = timezone.utc
START = datetime(2026, 1, 1, 10, 0, 30, tzinfo=UTC)
END = datetime(2026, 1, 1, 10, 11, 30, tzinfo=UTC)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


class _FirstUpdateDetector:
    def __init__(self, detector_id: str, *, trigger: bool) -> None:
        self.detector_id = detector_id
        self.trigger = trigger
        self.count = 0

    def on_observation(self, observation) -> DetectionResult:
        self.count += 1
        return DetectionResult(
            detector_id=self.detector_id,
            signal_id=observation_signal_id(observation),
            observation_time_utc=observation.observation_time_utc,
            triggered=self.trigger and self.count == 1,
            quality=observation.quality,
            detector_metadata={"observations_processed": self.count, "test_only": True},
        )


class _MockRag:
    def __init__(self) -> None:
        self.g1_calls = 0
        self.g2_calls = 0
        self.context_comment_ids: list[str] = []

    def g1(self, config) -> dict:
        self.g1_calls += 1
        payloads = _read_jsonl(Path(config.consumer_dir) / "rag_context_payloads.jsonl")
        payload = next(row for row in payloads if row["event_id"] == config.event_id)
        self.context_comment_ids = [
            row["comment_id"] for row in payload["used_context_comments"]
        ]
        report = {
            "event_id": config.event_id,
            "validation_status": "confirmed",
            "event_interpretation": "internal_community_reaction",
        }
        _write_jsonl(Path(config.output_dir) / G1_REPORTS, [report])
        _write_jsonl(
            Path(config.output_dir) / G1_RAW,
            [{"raw_response": {"usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5}}}],
        )
        _write_json(Path(config.output_dir) / G1_MANIFEST, {"model": config.model})
        return {"validation_status": "confirmed"}

    def g2(self, config) -> dict:
        self.g2_calls += 1
        cutoff = datetime.fromisoformat(config.external_evidence_cutoff_utc)
        report = {
            "event_id": config.event_id,
            "video_id": "video-a",
            "validation_status": "partially_confirmed",
        }
        summary = {
            "event_id": config.event_id,
            "overall_validation_status": "partially_confirmed",
            "overall_interpretation": "possible_external_event",
        }
        evidence = {
            "event_id": config.event_id,
            "video_id": "video-a",
            "published_at": (cutoff - timedelta(minutes=1)).isoformat(),
        }
        output = Path(config.output_dir)
        _write_jsonl(output / RAG_VIDEO_VALIDATION_REPORTS_FILE, [report])
        _write_jsonl(output / RAG_EVENT_VALIDATION_SUMMARY_FILE, [summary])
        _write_jsonl(output / RAG_VIDEO_EXTERNAL_EVIDENCE_FILE, [evidence])
        _write_jsonl(output / RAG_VIDEO_NEWS_QUERIES_FILE, [{"event_id": config.event_id}])
        _write_jsonl(
            output / G2_RAW,
            [{"raw_response": {"usage": {"input_tokens": 4, "output_tokens": 2, "total_tokens": 6}}}],
        )
        _write_json(
            output / G2_MANIFEST,
            {"query_call_count": 1, "validation_call_count": 1},
        )
        return {"event_summary": summary}


def _fixture(root: Path, *, run_id: str, output_name: str) -> RunConfig:
    comments = root / "silver_comments.csv"
    videos = root / "silver_videos.csv"
    stances = root / "precomputed_stances.jsonl"
    stance_manifest = root / "classifier_manifest.json"
    rows = [
        {
            "comment_id": "c1", "video_id": "video-a", "author_id": "a1",
            "event_time_utc": (START + timedelta(seconds=1)).isoformat(),
            "text": "Apoyo esta propuesta", "parent_comment_id": None,
            "is_reply": False,
        },
        {
            "comment_id": "c2", "video_id": "video-a", "author_id": "a2",
            "event_time_utc": (START + timedelta(minutes=1)).isoformat(),
            "text": "No la apoyo", "parent_comment_id": None,
            "is_reply": False,
        },
        {
            "comment_id": "future", "video_id": "video-a", "author_id": "a3",
            "event_time_utc": (START + timedelta(minutes=9)).isoformat(),
            "text": "Comentario futuro", "parent_comment_id": None,
            "is_reply": False,
        },
    ]
    pd.DataFrame(rows).to_csv(comments, index=False)
    pd.DataFrame([{"video_id": "video-a", "title": "Video A"}]).to_csv(
        videos, index=False
    )
    stance_rows = []
    for row, category in zip(rows, (5, 1, 4), strict=True):
        stance_rows.append({
            "comment_id": row["comment_id"],
            "author_id": row["author_id"],
            "event_time_utc": row["event_time_utc"],
            "proposition_id": "prop-a",
            "stance_category": category,
            "classification_status": CLASSIFIED,
            "quality": "passed",
        })
    _write_jsonl(stances, stance_rows)
    proposition = StanceProposition(
        proposition_id="prop-a",
        proposition_text="La propuesta debe aprobarse.",
        context="Caso sintético controlado.",
    )
    _write_json(stance_manifest, {
        "artifact_version": "stance_classifier_v1",
        "classifier_run_id": run_id,
        "model": "gpt-4.1-2025-04-14",
        "prompt_id": "comment_stance_likert5",
        "prompt_version": "1",
        "topic_id": "synthetic-topic",
        "proposition": {
            "proposition_id": proposition.proposition_id,
            "proposition_text": proposition.proposition_text,
            "context": proposition.context,
            "context_version": proposition.context_version,
        },
        "context_hash": hashlib.sha256(
            proposition.context.encode("utf-8")
        ).hexdigest(),
        "classification_policy": {
            "policy_id": "comment_stance_likert5_structured_v1",
            "missing_parent_policy": "SKIPPED",
            "thread_context": "immediate_parent_only",
            "text_authority": "original",
        },
        "input_ref": str(comments.absolute()),
        "input_policy_id": "causal_minimal_v1",
        "input_source_sha256": hashlib.sha256(
            comments.read_bytes()
        ).hexdigest(),
        "classified_comment_ids_sha256": hashlib.sha256(
            "\n".join(sorted(row["comment_id"] for row in rows)).encode("utf-8")
        ).hexdigest(),
    })
    activity = event_window_comment_count_definition(
        window="3min", cadence="1min", time_basis="event_time_utc"
    )
    temporal = PolarizationTemporalSignalConfig(
        proposition_id="prop-a",
        classification_run_id=run_id,
        window="4min",
        cadence="2min",
        start_time_utc=START,
        end_time_utc=END,
        enabled_measures=("mec",),
    )
    return RunConfig(
        identity=RunIdentityConfig(run_id),
        data=DataConfig(
            local_files=LocalFilesConfig(
                videos_path=videos, comments_path=comments, data_root=root
            ),
            experimental_input=ExperimentalInputPolicyConfig(),
        ),
        simulation=SimulationConfig(causal_end_to_end=CausalEndToEndReplayConfig(
            start_utc=START, end_utc=END, step="5min",
            output_dir=root / output_name,
            stance_artifact_path=stances,
            stance_manifest_path=stance_manifest,
            parameter_status="TEST_ONLY",
        )),
        signals=SignalsConfig(activity=activity, polarization=temporal),
        detection=DetectionConfig(
            routes=(
                ActivityDetectionRouteConfig(activity.signal_id, "page_hinkley"),
                ActivityDetectionRouteConfig("polarization_mec", "kswin"),
            ),
            gap_policy=DetectionGapPolicyConfig(),
            page_hinkley=PageHinkleyConfig(min_instances=2, threshold=1.0),
            kswin=KSWINConfig(alpha=0.1, window_size=4, stat_size=2, seed=7),
        ),
        association=DetectionAssociationConfig(
            tolerance="2min", parameter_status="TEST_ONLY"
        ),
        polarization=PolarizationConfig(
            mec=MECConfig(alpha=2.0, beta=1.15),
            classifier=CommentStanceClassifierConfig(
                proposition=proposition
            ),
            runtime=PolarizationRuntimeConfig(
                video_ids=("video-a",),
                input_schema=SILVER_COMMENTS_V1,
                source_role=CAUSAL_MINIMAL_V1_INPUT,
            ),
        ),
        rag=RagConfig(
            candidate_evidence=CandidateEvidencePolicyConfig(
                source_path=str(comments), output_dir=str(root / "unused-evidence"),
                lookback="3min", lookahead="0s", mode=IMMEDIATE_CAUSAL,
                parameter_status="TEST_ONLY",
            ),
            candidate_validation=CandidateRagValidationConfig(
                candidates_path=str(root / "unused-candidates.jsonl"),
                evidence_dir=str(root / "unused-evidence"),
                output_dir=str(root / "unused-rag"), candidate_id="runtime-placeholder",
            ),
            g2_gate=G2GatePolicyConfig(),
        ),
        topic_profile=TopicProfile(
            topic_id="synthetic-topic",
            topic_description="Tema sintético para replay causal.",
            proposition=proposition,
        ),
    )


class CausalEndToEndTests(unittest.TestCase):
    def test_stance_reuse_rejects_changed_proposition_or_context(self) -> None:
        for field, value in (
            ("proposition_text", "Una proposición diferente."),
            ("context", "Un contexto diferente."),
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                config = _fixture(root, run_id=f"reuse-{field}", output_name="out")
                manifest_path = root / "classifier_manifest.json"
                manifest = json.loads(manifest_path.read_text())
                manifest["proposition"][field] = value
                if field == "context":
                    manifest["context_hash"] = hashlib.sha256(
                        value.encode("utf-8")
                    ).hexdigest()
                _write_json(manifest_path, manifest)
                with self.assertRaisesRegex(ValueError, "snapshot"):
                    run_causal_end_to_end_replay(config, base_dir=root)

    def test_activity_scheduler_catches_up_all_ticks(self) -> None:
        definition = event_window_comment_count_definition(
            window="2min", cadence="1min", time_basis="event_time_utc"
        )
        signal = create_activity_signal(definition)
        signal.start_at(START)
        observations = signal.advance_to(START + timedelta(minutes=5))
        self.assertEqual(
            [item.observation_time_utc.minute for item in observations],
            [1, 2, 3, 4, 5],
        )
        with self.assertRaisesRegex(ValueError, "must not regress"):
            signal.advance_to(START + timedelta(minutes=4))

    def test_config_serialization_and_hash_include_replay_step(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = _fixture(root, run_id="config-run", output_name="out")
            mapping = run_config_to_mapping(config, path_base=root)
            self.assertEqual(
                mapping["simulation"]["causal_end_to_end"]["step"], "5min"
            )
            def omit_none(value):
                if isinstance(value, dict):
                    return {
                        key: omit_none(item)
                        for key, item in value.items()
                        if item is not None
                    }
                if isinstance(value, list):
                    return [omit_none(item) for item in value]
                return value

            loaded = run_config_from_mapping(omit_none(mapping))
            self.assertEqual(
                loaded.simulation.causal_end_to_end.step, "5min"
            )
            changed = replace(
                config,
                simulation=SimulationConfig(causal_end_to_end=replace(
                    config.simulation.causal_end_to_end, step="30s"
                )),
            )
            self.assertNotEqual(
                run_config_hash(config, path_base=root),
                run_config_hash(changed, path_base=root),
            )

    def _run(self, trigger_ids: set[str], *, expected_type: str) -> tuple[dict, _MockRag]:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        config = _fixture(root, run_id="e2e-run", output_name="out")

        def detector_factory(detector_id: str, _config):
            return _FirstUpdateDetector(detector_id, trigger=detector_id in trigger_ids)

        rag = _MockRag()
        artifacts = run_causal_end_to_end_replay(
            config,
            base_dir=root,
            execute_rag=True,
            detector_factory=detector_factory,
            g1_runner=rag.g1,
            g2_runner=rag.g2,
        )
        manifest = json.loads(artifacts.manifest_path.read_text())
        candidates = _read_jsonl(artifacts.candidate_path)
        self.assertEqual([item["candidate_type"] for item in candidates], [expected_type])
        self.assertEqual(artifacts.future_leak_count, 0)
        self.assertEqual(manifest["causality"]["future_leak_count"], 0)
        self.assertEqual(manifest["scheduling"]["activity_ticks"], 11)
        self.assertEqual(manifest["scheduling"]["polarization_ticks"], 5)
        self.assertEqual(manifest["stance"]["topic_id"], "synthetic-topic")
        self.assertEqual(manifest["rag"]["g2_gate"]["mode"], "OFF")
        self.assertEqual(manifest["counts"]["g2_skipped_by_gate"], 0)
        self.assertEqual(artifacts.validations, 1)
        self.assertNotIn("future", rag.context_comment_ids)
        self.assertEqual(rag.g1_calls, 1)
        self.assertEqual(rag.g2_calls, 1)
        return manifest, rag

    def test_joint_candidate_runs_end_to_end_without_future_leak(self) -> None:
        manifest, _ = self._run(
            {"page_hinkley", "kswin"}, expected_type="ACTIVITY_POLARIZATION"
        )
        self.assertEqual(manifest["counts"]["activity_polarization_candidates"], 1)
        activity_rows = _read_jsonl(Path(manifest["artifacts"]["activity_observations"]))
        early = [
            row for row in activity_rows
            if datetime.fromisoformat(row["observation_time_utc"])
            < START + timedelta(minutes=9)
        ]
        self.assertTrue(early)
        self.assertTrue(all(row["value"] <= 2 for row in early))

    def test_activity_only_candidate_uses_same_pipeline(self) -> None:
        self._run({"page_hinkley"}, expected_type="ACTIVITY")

    def test_polarization_only_candidate_uses_same_pipeline(self) -> None:
        self._run({"kswin"}, expected_type="POLARIZATION")

    def test_zero_candidate_run_is_successful(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = _fixture(root, run_id="zero-run", output_name="out")

            def detector_factory(detector_id: str, _config):
                return _FirstUpdateDetector(detector_id, trigger=False)

            artifacts = run_causal_end_to_end_replay(
                config,
                base_dir=root,
                execute_rag=True,
                detector_factory=detector_factory,
            )
            manifest = json.loads(artifacts.manifest_path.read_text())
            self.assertEqual(artifacts.candidates, 0)
            self.assertEqual(artifacts.validations, 0)
            self.assertEqual(manifest["status"], "executed_successfully")
            self.assertEqual(manifest["counts"]["event_candidates"], 0)

    def test_default_factory_runs_real_configured_detectors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = _fixture(root, run_id="real-detector-run", output_name="out")
            artifacts = run_causal_end_to_end_replay(config, base_dir=root)
            results = _read_jsonl(artifacts.output_dir / "detection_results.jsonl")
            self.assertGreater(len(results), 0)
            self.assertEqual(
                {row["detector_id"] for row in results},
                {"page_hinkley", "kswin"},
            )
            self.assertEqual(artifacts.future_leak_count, 0)

    def test_zero_evidence_candidate_does_not_fail_or_call_rag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = _fixture(root, run_id="no-evidence-run", output_name="out")
            config = replace(
                config,
                rag=RagConfig(
                    candidate_evidence=replace(
                        config.rag.candidate_evidence,
                        lookback="0s",
                        lookahead="0s",
                    ),
                    candidate_validation=config.rag.candidate_validation,
                ),
            )

            def detector_factory(detector_id: str, _config):
                return _FirstUpdateDetector(
                    detector_id, trigger=detector_id == "page_hinkley"
                )

            rag = _MockRag()
            artifacts = run_causal_end_to_end_replay(
                config,
                base_dir=root,
                execute_rag=True,
                detector_factory=detector_factory,
                g1_runner=rag.g1,
                g2_runner=rag.g2,
            )
            result = _read_jsonl(artifacts.validation_results_path)[0]
            self.assertEqual(result["status"], "NO_EVIDENCE_FOUND")
            self.assertEqual(rag.g1_calls, 0)
            self.assertEqual(rag.g2_calls, 0)

    def test_step_smaller_than_both_cadences_does_not_duplicate_ticks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = _fixture(root, run_id="small-step-run", output_name="out")
            config = replace(
                config,
                simulation=SimulationConfig(causal_end_to_end=replace(
                    config.simulation.causal_end_to_end, step="30s"
                )),
            )

            def detector_factory(detector_id: str, _config):
                return _FirstUpdateDetector(detector_id, trigger=False)

            artifacts = run_causal_end_to_end_replay(
                config, base_dir=root, detector_factory=detector_factory
            )
            manifest = json.loads(artifacts.manifest_path.read_text())
            self.assertEqual(manifest["scheduling"]["activity_ticks"], 11)
            self.assertEqual(manifest["scheduling"]["polarization_ticks"], 5)
            self.assertGreater(manifest["replay_clock"]["cutoff_count"], 11)

    def test_end_of_stream_flush_promotes_pending_alert(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = _fixture(root, run_id="flush-run", output_name="out")
            config = replace(
                config,
                association=DetectionAssociationConfig(
                    tolerance="30min", parameter_status="TEST_ONLY"
                ),
            )

            def detector_factory(detector_id: str, _config):
                return _FirstUpdateDetector(
                    detector_id, trigger=detector_id == "page_hinkley"
                )

            artifacts = run_causal_end_to_end_replay(
                config, base_dir=root, detector_factory=detector_factory
            )
            associations = _read_jsonl(
                artifacts.output_dir / "detection_associations.jsonl"
            )
            self.assertEqual(len(associations), 1)
            self.assertEqual(associations[0]["finalization_reason"], "END_OF_STREAM")
            self.assertEqual(artifacts.candidates, 1)

    def test_delayed_evidence_remains_pending_and_skips_rag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = _fixture(root, run_id="pending-run", output_name="out")
            config = replace(
                config,
                rag=RagConfig(
                    candidate_evidence=replace(
                        config.rag.candidate_evidence,
                        mode="delayed_enrichment",
                        lookahead="30min",
                    ),
                    candidate_validation=config.rag.candidate_validation,
                ),
            )

            def detector_factory(detector_id: str, _config):
                return _FirstUpdateDetector(
                    detector_id, trigger=detector_id == "page_hinkley"
                )

            rag = _MockRag()
            artifacts = run_causal_end_to_end_replay(
                config,
                base_dir=root,
                execute_rag=True,
                detector_factory=detector_factory,
                g1_runner=rag.g1,
                g2_runner=rag.g2,
            )
            results = _read_jsonl(artifacts.validation_results_path)
            self.assertEqual(results[0]["status"], "EVIDENCE_PENDING_HORIZON")
            self.assertEqual(rag.g1_calls, 0)
            self.assertEqual(rag.g2_calls, 0)

    def test_delayed_evidence_becomes_ready_only_after_horizon_closes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = _fixture(root, run_id="delayed-ready-run", output_name="out")
            config = replace(
                config,
                rag=RagConfig(
                    candidate_evidence=replace(
                        config.rag.candidate_evidence,
                        mode="delayed_enrichment",
                        lookahead="7min",
                    ),
                    candidate_validation=config.rag.candidate_validation,
                ),
            )

            def detector_factory(detector_id: str, _config):
                return _FirstUpdateDetector(
                    detector_id, trigger=detector_id == "page_hinkley"
                )

            rag = _MockRag()
            artifacts = run_causal_end_to_end_replay(
                config,
                base_dir=root,
                execute_rag=True,
                detector_factory=detector_factory,
                g1_runner=rag.g1,
                g2_runner=rag.g2,
            )
            manifest = json.loads(artifacts.manifest_path.read_text())
            statuses = [
                row["status"]
                for row in manifest["evidence"]["readiness_transitions"]
            ]
            self.assertEqual(
                statuses, ["EVIDENCE_PENDING_HORIZON", "EVIDENCE_READY"]
            )
            self.assertEqual(rag.g1_calls, 1)
            self.assertEqual(rag.g2_calls, 1)

    def test_same_input_config_and_mocks_preserve_deterministic_identities(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = _fixture(root, run_id="deterministic-run", output_name="out")

            def run_once():
                rag = _MockRag()

                def detector_factory(detector_id: str, _config):
                    return _FirstUpdateDetector(detector_id, trigger=True)

                artifacts = run_causal_end_to_end_replay(
                    config,
                    base_dir=root,
                    execute_rag=True,
                    detector_factory=detector_factory,
                    g1_runner=rag.g1,
                    g2_runner=rag.g2,
                )
                return (
                    _read_jsonl(artifacts.output_dir / "detection_alerts.jsonl"),
                    _read_jsonl(artifacts.candidate_path),
                    _read_jsonl(artifacts.validation_results_path),
                )

            first = run_once()
            shutil.rmtree(root / "out")
            second = run_once()
            self.assertEqual(
                [row["alert_id"] for row in first[0]],
                [row["alert_id"] for row in second[0]],
            )
            self.assertEqual(
                [row["candidate_id"] for row in first[1]],
                [row["candidate_id"] for row in second[1]],
            )
            self.assertEqual(
                [row["validation_id"] for row in first[2]],
                [row["validation_id"] for row in second[2]],
            )


if __name__ == "__main__":
    unittest.main()
