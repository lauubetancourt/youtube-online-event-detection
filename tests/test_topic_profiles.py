from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path
from datetime import datetime, timezone

from youtube_pipeline.configuration import (
    load_run_config,
    run_config_from_mapping,
    run_config_hash,
    run_config_to_mapping,
)
from youtube_pipeline.stance_classifier import (
    OpenAICommentStanceClassifier,
    StanceCommentInput,
    StanceModelResponse,
)


def _topic(*, proposition: str = "La medida debe aprobarse.", context: str = "Contexto A.") -> dict:
    return {
        "topic_id": "topic-a",
        "topic_description": "Tema controlado A",
        "stance": {
            "proposition_id": "proposal-a",
            "proposition": proposition,
            "context": context,
            "prompt_id": "comment_stance_likert5",
            "prompt_version": "1",
        },
    }


def _experiment(topic_profile: object = "topic-a") -> dict:
    return {
        "identity": {"run_id": "topic-config-test"},
        "topic_profile": topic_profile,
        "polarization": {
            "classifier": {
                "retry_policy": {
                    "initial_delay_seconds": 0,
                    "max_delay_seconds": 0,
                    "jitter_seconds": 0,
                }
            }
        },
    }


class TopicProfileTests(unittest.TestCase):
    def test_experiment_reference_resolves_into_frozen_classifier_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "configs"
            topics = root / "topics"
            experiments = root / "experiments"
            topics.mkdir(parents=True)
            experiments.mkdir(parents=True)
            (topics / "topic-a.json").write_text(
                json.dumps(_topic()), encoding="utf-8"
            )
            experiment_path = experiments / "experiment.json"
            experiment_path.write_text(
                json.dumps(_experiment()), encoding="utf-8"
            )

            config = load_run_config(experiment_path)

        assert config.topic_profile is not None
        assert config.polarization is not None
        assert config.polarization.classifier is not None
        self.assertEqual(config.topic_profile.topic_id, "topic-a")
        self.assertEqual(
            config.polarization.classifier.proposition.proposition_text,
            "La medida debe aprobarse.",
        )
        self.assertEqual(
            config.polarization.classifier.proposition.context, "Contexto A."
        )
        with self.assertRaises(FrozenInstanceError):
            config.topic_profile.topic_id = "changed"  # type: ignore[misc]

    def test_inline_topic_serializes_and_changes_config_hash(self) -> None:
        first = run_config_from_mapping(_experiment(_topic()))
        second_topic = _topic(context="Contexto B.")
        second = run_config_from_mapping(_experiment(second_topic))
        mapping = run_config_to_mapping(first)
        self.assertEqual(mapping["topic_profile"]["topic_id"], "topic-a")
        self.assertEqual(
            mapping["polarization"]["classifier"]["proposition"]["context"],
            "Contexto A.",
        )
        self.assertNotEqual(run_config_hash(first), run_config_hash(second))

    def test_missing_proposition_context_and_unknown_reference_are_rejected(self) -> None:
        for missing in ("proposition", "context"):
            profile = _topic()
            del profile["stance"][missing]
            with self.subTest(missing=missing), self.assertRaisesRegex(
                ValueError, "Missing TopicProfile stance fields"
            ):
                run_config_from_mapping(_experiment(profile))

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "experiment.json"
            path.write_text(json.dumps(_experiment("unknown")), encoding="utf-8")
            with self.assertRaisesRegex(FileNotFoundError, "Unknown topic profile"):
                load_run_config(path)

    def test_classifier_cannot_conflict_with_topic_authority(self) -> None:
        payload = _experiment(_topic())
        payload["polarization"]["classifier"]["proposition"] = {
            "proposition_id": "other",
            "proposition_text": "Otra.",
            "context": "Otro contexto.",
            "context_version": "provided_v1",
        }
        with self.assertRaisesRegex(ValueError, "conflicts with topic_profile"):
            run_config_from_mapping(payload)


class TopicClassifierIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_different_profiles_reach_classifier_without_code_changes(self) -> None:
        captured: list[dict] = []

        class Transport:
            async def complete(self, *, config, messages):
                captured.append(json.loads(messages[0]["content"].splitlines()[-1]))
                return StanceModelResponse(
                    json.dumps({"status": "CLASSIFIED", "stance": 4})
                )

        for proposition, context in (
            ("Proposición A.", "Contexto A."),
            ("Proposición B.", "Contexto B."),
        ):
            profile = _topic(proposition=proposition, context=context)
            config = run_config_from_mapping(_experiment(profile))
            assert config.polarization is not None
            assert config.polarization.classifier is not None
            classifier = OpenAICommentStanceClassifier(
                config.polarization.classifier,
                transport=Transport(),
            )
            await classifier.classify(
                StanceCommentInput(
                    comment_id=f"comment-{len(captured)}",
                    author_id="author",
                    event_time_utc=datetime(2026, 1, 1, tzinfo=timezone.utc),
                    text="Estoy de acuerdo.",
                )
            )

        self.assertEqual(
            [(item["proposition_text"], item["context"]) for item in captured],
            [("Proposición A.", "Contexto A."), ("Proposición B.", "Contexto B.")],
        )


if __name__ == "__main__":
    unittest.main()
