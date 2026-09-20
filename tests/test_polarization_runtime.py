from __future__ import annotations

import asyncio
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from youtube_pipeline.configuration import load_run_config, run_config_from_mapping
from youtube_pipeline.entrypoints.polarization_runtime import main
from youtube_pipeline.polarization_runtime import (
    DISTRIBUTION_ARTIFACT,
    RUNTIME_MANIFEST,
    SIGNAL_ARTIFACT,
    prepare_polarization_runtime,
    run_polarization_runtime,
)
from youtube_pipeline.stance_classifier import StanceModelResponse, TokenUsage


class FakeTransport:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def complete(self, *, config, messages):
        text = json.loads(messages[-1]["content"])["comment_text"]
        self.calls.append(text)
        if text == "bad output":
            return StanceModelResponse("not-json", TokenUsage(2, 1, 3), response_id="bad")
        status, stance = {
            "against": ("CLASSIFIED", 1),
            "not about it": ("NOT_APPLICABLE", None),
            "unclear": ("UNCERTAIN", None),
            "missing author favor": ("CLASSIFIED", 4),
            "favor": ("CLASSIFIED", 5),
        }[text]
        return StanceModelResponse(
            json.dumps({"status": status, "stance": stance}),
            TokenUsage(2, 3, 5), response_id=f"response-{len(self.calls)}",
        )


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _fixture(root: Path) -> Path:
    videos = root / "videos.csv"
    comments = root / "comments.csv"
    _write_csv(videos, ["video_id", "title", "publishedAt"], [{
        "video_id": "v1", "title": "Controlled real-format fixture",
        "publishedAt": "2026-01-01T00:00:00Z",
    }])
    rows = [
        ("c1", "A", "2026-01-01T00:00:00Z", "against"),
        ("c2", "B", "2026-01-01T00:00:20Z", "not about it"),
        ("c3", "C", "2026-01-01T00:00:40Z", "unclear"),
        ("c4", "D", "2026-01-01T00:00:50Z", ""),
        ("c5", "", "2026-01-01T00:01:00Z", "missing author favor"),
        ("c6", "E", "2026-01-01T00:01:20Z", "bad output"),
        ("c7", "B", "2026-01-01T00:01:40Z", "favor"),
    ]
    _write_csv(comments, [
        "video_id", "comment_id", "text", "author_id", "published_at",
        "is_reply", "reply_to_comment_id",
    ], [{
        "video_id": "v1", "comment_id": comment, "text": text,
        "author_id": author, "published_at": timestamp,
        "is_reply": False, "reply_to_comment_id": "",
    } for comment, author, timestamp, text in rows])
    config = root / "runtime.json"
    config.write_text(json.dumps({
        "identity": {"run_id": "runtime_test"},
        "data": {"local_files": {
            "videos_path": "videos.csv", "comments_path": "comments.csv", "data_root": ".",
        }},
        "signals": {"polarization": {
            "proposition_id": "p", "classification_run_id": "runtime_test",
            "window": "120s", "cadence": "60s",
            "start_time_utc": "2026-01-01T00:00:00Z",
            "end_time_utc": "2026-01-01T00:03:01Z",
            "enabled_measures": ["esteban_ray", "emd_pol", "mec"],
        }},
        "polarization": {
            "classifier": {"proposition": {
                "proposition_id": "p", "proposition_text": "Do you support p?",
                "context": "Controlled context",
            }},
            "runtime": {"video_ids": ["v1"]},
            "esteban_ray": {}, "emd_pol": {}, "mec": {},
        },
    }), encoding="utf-8")
    return config


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


class PolarizationRuntimeTests(unittest.TestCase):
    def test_prepare_is_config_driven_and_makes_no_model_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = _fixture(root)
            config = load_run_config(path)
            from youtube_pipeline.configuration import resolve_run_config
            plan = prepare_polarization_runtime(resolve_run_config(config, base_dir=root).config)
            self.assertEqual(len(plan.comments), 7)
            self.assertEqual(plan.selected_video_ids, ("v1",))
            self.assertEqual(plan.expected_classifier_calls, 6)

    def test_end_to_end_persisted_handoff_statuses_lineage_and_measures(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = _fixture(root)
            output = root / "output"
            transport = FakeTransport()
            artifacts = asyncio.run(run_polarization_runtime(
                load_run_config(path), output_dir=output, base_dir=root, transport=transport,
            ))

            self.assertEqual(len(transport.calls), 6)
            stance_rows = _read_jsonl(artifacts.stance_path)
            self.assertEqual({row["classification_status"] for row in stance_rows}, {
                "CLASSIFIED", "NOT_APPLICABLE", "UNCERTAIN", "SKIPPED", "ERROR",
            })
            distributions = _read_jsonl(output / DISTRIBUTION_ARTIFACT)
            observations = _read_jsonl(output / SIGNAL_ARTIFACT)
            self.assertEqual(len(distributions), 4)
            self.assertEqual(len(observations), 12)
            self.assertEqual({row["signal_id"] for row in observations}, {
                "polarization_esteban_ray", "polarization_emd_pol", "polarization_mec",
            })
            final = distributions[-1]
            selected = final["selected_inputs"]
            self.assertEqual(len({row["author_id"] for row in selected}), len(selected))
            self.assertTrue(all(row["comment_id"] in {"c7"} for row in selected))
            self.assertEqual(final["distribution"]["excluded_counts"]["error"], 1)
            self.assertEqual(final["distribution"]["excluded_counts"]["missing_author"], 1)
            self.assertNotIn("c2", {row["comment_id"] for row in selected})
            self.assertNotIn("c3", {row["comment_id"] for row in selected})
            manifest = json.loads((output / RUNTIME_MANIFEST).read_text(encoding="utf-8"))
            self.assertEqual(manifest["classifier"]["status_counts"], {
                "CLASSIFIED": 3, "NOT_APPLICABLE": 1, "UNCERTAIN": 1,
                "SKIPPED": 1, "ERROR": 1,
            })
            self.assertEqual(manifest["sanity_checks"]["future_leak_count"], 0)
            self.assertFalse(manifest["downstream"]["detectors_executed"])

    def test_downstream_temporal_artifacts_are_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = _fixture(root)
            config = load_run_config(path)
            for name in ("a", "b"):
                asyncio.run(run_polarization_runtime(
                    config, output_dir=root / name, base_dir=root, transport=FakeTransport(),
                ))
            for artifact in (DISTRIBUTION_ARTIFACT, SIGNAL_ARTIFACT):
                self.assertEqual((root / "a" / artifact).read_bytes(), (root / "b" / artifact).read_bytes())

    def test_runtime_configuration_rejects_unknowns_and_cross_section_mismatch(self):
        base = {
            "identity": {"run_id": "r"},
            "data": {"local_files": {"videos_path": "v.csv", "comments_path": "c.csv"}},
            "signals": {"polarization": {
                "proposition_id": "p", "classification_run_id": "r", "window": "1m",
                "cadence": "1m", "start_time_utc": "2026-01-01T00:00:00Z",
                "end_time_utc": "2026-01-01T00:01:01Z", "enabled_measures": ["mec"],
            }},
            "polarization": {
                "classifier": {"proposition": {
                    "proposition_id": "p", "proposition_text": "p?", "context": "c",
                }},
                "runtime": {"video_ids": ["v"]}, "mec": {},
            },
        }
        run_config_from_mapping(base)
        bad = json.loads(json.dumps(base)); bad["polarization"]["runtime"]["limit"] = 2
        with self.assertRaisesRegex(ValueError, "Unknown polarization runtime"):
            run_config_from_mapping(bad)
        bad = json.loads(json.dumps(base)); bad["signals"]["polarization"]["proposition_id"] = "other"
        with self.assertRaisesRegex(ValueError, "proposition_id"):
            run_config_from_mapping(bad)
        bad = json.loads(json.dumps(base)); bad["signals"]["polarization"]["classification_run_id"] = "other"
        with self.assertRaisesRegex(ValueError, "classification_run_id"):
            run_config_from_mapping(bad)

    def test_entrypoint_prepares_by_default_and_requires_output_for_execute(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = _fixture(root)
            with patch("builtins.print") as output:
                main(["--config", str(path)])
            prepared = json.loads(output.call_args.args[0])
            self.assertEqual(prepared["execution_status"], "prepared_not_executed")
            self.assertEqual(prepared["expected_classifier_calls"], 6)
            with patch("builtins.print") as output:
                main(["--config", str(path), "--run-id", "runtime_override"])
            overridden = json.loads(output.call_args.args[0])
            self.assertEqual(overridden["run_id"], "runtime_override")
            with self.assertRaises(SystemExit):
                main(["--config", str(path), "--execute"])


if __name__ == "__main__":
    unittest.main()
