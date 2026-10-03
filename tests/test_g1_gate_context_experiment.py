from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from youtube_pipeline.g1_gate_context_experiment import (
    G1GateContextExperimentConfig,
    pack_context_payload,
    run_g1_gate_context_experiment,
    select_budget_sample,
)
from youtube_pipeline.rag_generation_g1 import PROMPT_VERSION, _approx_tokens, _compact_event_payload


def _json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _jsonl(path: Path, values: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(value) for value in values) + "\n", encoding="utf-8")


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _usage(total: int) -> dict:
    return {"raw_response": {"usage": {"input_tokens": total - 10, "output_tokens": 10, "total_tokens": total}}}


class G1GateContextExperimentTests(unittest.TestCase):
    def test_whole_unit_packing_and_full_dedup(self) -> None:
        validation = {"event_id": "c"}
        payload = {
            "event_id": "c",
            "selected_context_units": [
                {"context_unit_id": "u1", "video_id": "v1", "comment_ids": ["a"], "context_text": "a" * 200},
                {"context_unit_id": "u2", "video_id": "v2", "comment_ids": ["b"], "context_text": "b" * 200},
            ],
            "used_context_comments": [
                {"comment_id": "a", "video_id": "v1", "root_comment_id": "a", "text": "a" * 100},
                {"comment_id": "b", "video_id": "v2", "root_comment_id": "b", "text": "b" * 100},
            ],
            "post_trigger_context_used": False,
        }
        full = _approx_tokens(_compact_event_payload(validation, payload))
        same, metadata = pack_context_payload(validation, payload, full)
        self.assertTrue(metadata["equivalent_to_full"])
        self.assertEqual(same, payload)
        reduced, metadata = pack_context_payload(validation, payload, max(1, full - 80))
        self.assertFalse(metadata["equivalent_to_full"])
        selected_ids = {
            comment_id
            for unit in reduced["selected_context_units"]
            for comment_id in unit["comment_ids"]
        }
        self.assertEqual(
            selected_ids,
            {row["comment_id"] for row in reduced["used_context_comments"]},
        )
        self.assertLessEqual(metadata["estimated_context_tokens"], full - 80)

    def test_size_sample_is_deterministic(self) -> None:
        estimates = {f"c{i}": value for i, value in enumerate([10, 20, 30, 40, 50, 60])}
        first = select_budget_sample(estimates)
        self.assertEqual(first, select_budget_sample(estimates))
        self.assertEqual(len(first), 4)
        self.assertEqual(first[0], "c0")
        self.assertEqual(first[-1], "c5")

    def _fixture(self, root: Path) -> G1GateContextExperimentConfig:
        consumer, g1, g2 = root / "consumer", root / "g1", root / "g2"
        inputs = [{"event_id": candidate, "validation_id": f"v-{candidate}"} for candidate in ("c1", "c2")]
        payloads = []
        for candidate, text in (("c1", "a" * 160), ("c2", "b" * 800)):
            payloads.append(
                {
                    "event_id": candidate,
                    "selected_context_units": [
                        {"context_unit_id": f"u-{candidate}-1", "video_id": "v1", "comment_ids": [f"m-{candidate}-1"], "context_text": text},
                        {"context_unit_id": f"u-{candidate}-2", "video_id": "v2", "comment_ids": [f"m-{candidate}-2"], "context_text": text},
                    ],
                    "used_context_comments": [
                        {"comment_id": f"m-{candidate}-1", "video_id": "v1", "root_comment_id": f"m-{candidate}-1", "text": text},
                        {"comment_id": f"m-{candidate}-2", "video_id": "v2", "root_comment_id": f"m-{candidate}-2", "text": text},
                    ],
                    "post_trigger_context_used": False,
                }
            )
        _jsonl(consumer / "rag_validation_inputs.jsonl", inputs)
        _jsonl(consumer / "rag_context_payloads.jsonl", payloads)
        _json(consumer / "rag_consumer_manifest.json", {"run_id": "consumer"})
        hashes = {
            "rag_validation_inputs": __import__("hashlib").sha256((consumer / "rag_validation_inputs.jsonl").read_bytes()).hexdigest(),
            "rag_context_payloads": __import__("hashlib").sha256((consumer / "rag_context_payloads.jsonl").read_bytes()).hexdigest(),
            "rag_consumer_manifest": __import__("hashlib").sha256((consumer / "rag_consumer_manifest.json").read_bytes()).hexdigest(),
        }
        _jsonl(
            g1 / "rag_validation_reports.jsonl",
            [{"event_id": "c1", "validation_id": "g1-c1", "validation_status": "confirmed", "confidence_label": "high", "event_interpretation": "possible_external_event", "cited_comment_ids": ["m-c1-1"], "cited_context_unit_ids": ["u-c1-1"], "citation_verification": {"valid": True}}],
        )
        _jsonl(g1 / "rag_raw_model_responses.jsonl", [{"event_id": "c1", **_usage(100)}])
        _json(g1 / "rag_generation_manifest.json", {"model_name": "gpt-5-mini", "prompt_version": PROMPT_VERSION, "input_hashes": hashes})
        raw = []
        queries = []
        external = []
        reports = []
        summaries = []
        for candidate in ("c1", "c2"):
            raw.extend([
                {"event_id": candidate, "record_type": "video_query_generation", **_usage(30)},
                {"event_id": candidate, "record_type": "video_validation_generation", **_usage(40)},
            ])
            queries.append({"event_id": candidate, "video_id": "v1"})
            external.append({"event_id": candidate, "external_evidence_id": f"e-{candidate}"})
            reports.append({"event_id": candidate, "external_evidence_assessment": "supports", "used_external_evidence_ids": [f"e-{candidate}"]})
            summaries.append({"event_id": candidate, "validation_status": "confirmed"})
        _jsonl(g2 / "rag_raw_model_responses.jsonl", raw)
        _jsonl(g2 / "rag_video_news_queries.jsonl", queries)
        _jsonl(g2 / "rag_video_external_evidence.jsonl", external)
        _jsonl(g2 / "rag_video_validation_reports.jsonl", reports)
        _jsonl(g2 / "rag_event_validation_summary.jsonl", summaries)
        _json(g2 / "rag_generation_manifest.json", {"run_id": "g2"})
        return G1GateContextExperimentConfig(
            output_dir=(root / "out").as_posix(),
            consumer_dir=consumer.as_posix(),
            existing_g1_dir=g1.as_posix(),
            g2_dir=g2.as_posix(),
            baseline_call_cap=1,
            budget_call_cap=12,
            budgets=(("B50", 500), ("B75", 700), ("B90", 900)),
            sample_size=3,
        )

    def test_completion_gate_savings_stability_and_deterministic_reuse(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._fixture(root)
            calls: list[str] = []

            def mock_runner(run_config):
                calls.append(run_config.event_id)
                consumer = Path(run_config.consumer_dir)
                output = Path(run_config.output_dir)
                payload = json.loads((consumer / "rag_context_payloads.jsonl").read_text().splitlines()[0])
                ids = [row["comment_id"] for row in payload["used_context_comments"]]
                units = [row["context_unit_id"] for row in payload["selected_context_units"]]
                report = {
                    "event_id": run_config.event_id,
                    "validation_id": f"g1-{run_config.event_id}",
                    "validation_status": "not_confirmed" if run_config.event_id == "c2" else "confirmed",
                    "confidence_label": "medium",
                    "event_interpretation": "unclear",
                    "cited_comment_ids": ids[:1],
                    "cited_context_unit_ids": units[:1],
                    "citation_verification": {"valid": True},
                }
                _jsonl(output / "rag_validation_reports.jsonl", [report])
                _jsonl(output / "rag_raw_model_responses.jsonl", [{"event_id": run_config.event_id, **_usage(80)}])
                _json(output / "rag_generation_manifest.json", {"run_id": "mock"})
                return {"event_id": run_config.event_id}

            artifacts = run_g1_gate_context_experiment(config, execute=True, g1_runner=mock_runner)
            baseline = _rows(Path(artifacts.baseline_path))
            self.assertEqual(len(baseline), 2)
            self.assertEqual({row["g1_result_source"] for row in baseline}, {"EXISTING_REUSED", "EXPERIMENT_NEW"})
            gate = _rows(Path(artifacts.gate_path))
            off_c2 = next(row for row in gate if row["policy"] == "OFF" and row["candidate_id"] == "c2")
            strict_c2 = next(row for row in gate if row["policy"] == "STRICT" and row["candidate_id"] == "c2")
            permissive_c2 = next(row for row in gate if row["policy"] == "PERMISSIVE" and row["candidate_id"] == "c2")
            self.assertEqual(off_c2["counterfactual_decision"], "RUN_G2")
            self.assertEqual(strict_c2["counterfactual_decision"], "SKIP_G2")
            self.assertEqual(permissive_c2["counterfactual_decision"], "SKIP_G2")
            self.assertEqual(strict_c2["g2_total_tokens_saved"], "70")
            self.assertEqual(strict_c2["externally_supported_candidate_filtered"], "True")
            stability = _rows(Path(artifacts.stability_path))
            self.assertTrue(any(row["budget_name"] == "FULL" for row in stability))
            evaluated = [row for row in stability if row["result_source"] != "NOT_EXECUTED"]
            self.assertTrue(evaluated)
            self.assertTrue(all(row["status_agreement_with_full"] in {"True", "False"} for row in evaluated))
            self.assertTrue(all(row["confidence_agreement_with_full"] in {"True", "False"} for row in evaluated))
            self.assertTrue(all(row["comment_citation_overlap_with_full"] != "" for row in evaluated))
            self.assertTrue(all(row["video_coverage"] != "" for row in stability))
            first_call_count = len(calls)
            run_g1_gate_context_experiment(config, execute=True, g1_runner=mock_runner)
            self.assertEqual(len(calls), first_call_count)
            manifest = json.loads(Path(artifacts.manifest_path).read_text())
            self.assertEqual(manifest["new_g2_calls"], 0)
            self.assertEqual(manifest["new_serper_requests"], 0)
            summary = json.loads(Path(artifacts.summary_path).read_text())
            self.assertEqual(summary["baseline_completion"]["paired_g1_g2"], 2)
            self.assertGreater(summary["actual_resource_usage"]["new_g1_total_tokens"], 0)

    def test_baseline_call_cap_stops_execution(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = self._fixture(Path(temporary))
            object.__setattr__(config, "baseline_call_cap", 0)
            with self.assertRaisesRegex(RuntimeError, "above cap"):
                run_g1_gate_context_experiment(config, execute=True)


if __name__ == "__main__":
    unittest.main()
