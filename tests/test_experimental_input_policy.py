from __future__ import annotations

import csv
import json
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from youtube_pipeline.configuration import (
    resolve_run_config,
    run_config_from_mapping,
    run_config_hash,
    run_config_to_mapping,
)
from youtube_pipeline.cyclic_ingestion import (
    CyclicIngestionConfig,
    build_cyclic_ingestion_dry_run,
)
from youtube_pipeline.entrypoints.cyclic_pipeline import run_cyclic_pipeline
from youtube_pipeline.experimental_input import (
    CausalMinimalInputPolicy,
    DuplicateCommentConflictError,
    ExperimentalInputPolicyConfig,
    adapt_activity_records,
    adapt_stance_records,
    prepare_causal_minimal_batch,
    read_input_table,
)
from youtube_pipeline.polarization_runtime import prepare_polarization_runtime
from youtube_pipeline.run_pipeline import run_clean
from youtube_pipeline.storage import canonicalize_comment_event_time


UTC = timezone.utc


def _row(
    comment_id: str,
    timestamp: str,
    text: str,
    *,
    author: str = "a",
    video: str = "v1",
    parent: str | None = None,
) -> dict[str, object]:
    return {
        "comment_id": comment_id,
        "video_id": video,
        "author_id": author,
        "event_time_utc": timestamp,
        "text": text,
        "parent_comment_id": parent,
        "reply_to_comment_id": parent,
        "is_reply": parent is not None,
    }


def _policy() -> ExperimentalInputPolicyConfig:
    return ExperimentalInputPolicyConfig()


class ExperimentalInputPolicyTests(unittest.TestCase):
    def test_real_youtube_csv_schema_is_canonicalized_before_causal_policy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "d1_sample.csv"
            pd.DataFrame([{
                "video_id": "v1",
                "comment_id": "c1",
                "text": "Comentario original",
                "author_name": "Autor",
                "author_id": "a1",
                "published_at": "2026-04-20T09:00:00-05:00",
                "likes": 3,
                "is_reply": False,
                "reply_to_comment_id": None,
            }]).to_csv(source, index=False)

            loaded = read_input_table(source)
            canonical = canonicalize_comment_event_time(loaded)
            batch = prepare_causal_minimal_batch(
                canonical,
                _policy(),
                source_path=source,
            )

        self.assertEqual(len(batch.records), 1)
        self.assertEqual(batch.records[0].comment_id, "c1")
        self.assertEqual(
            batch.records[0].event_time_utc,
            datetime(2026, 4, 20, 14, 0, tzinfo=UTC),
        )
        self.assertIn("published_at", batch.dataframe.columns)
        self.assertEqual(
            batch.dataframe.loc[0, "published_at"],
            "2026-04-20T09:00:00-05:00",
        )

    def test_config_is_strict_immutable_serialized_and_hash_visible(self) -> None:
        payload = {
            "identity": {"run_id": "policy-test"},
            "data": {
                "local_files": {
                    "videos_path": "videos.csv",
                    "comments_path": "comments.parquet",
                },
                "experimental_input": {
                    "policy_id": "causal_minimal_v1",
                    "source": "silver",
                    "exact_comment_id_dedup": True,
                    "semantic_filtering": False,
                },
            },
        }
        configured = run_config_from_mapping(payload)
        assert configured.data is not None
        policy = configured.data.experimental_input
        assert policy is not None
        with self.assertRaises(FrozenInstanceError):
            policy.source = "gold"  # type: ignore[misc]
        serialized = run_config_to_mapping(configured)
        self.assertEqual(
            serialized["data"]["experimental_input"]["policy_id"],
            "causal_minimal_v1",
        )
        without = json.loads(json.dumps(payload))
        without["data"].pop("experimental_input")
        self.assertNotEqual(
            run_config_hash(configured),
            run_config_hash(run_config_from_mapping(without)),
        )
        with self.assertRaisesRegex(ValueError, "Unknown ExperimentalInputPolicyConfig"):
            run_config_from_mapping({
                **payload,
                "data": {
                    **payload["data"],
                    "experimental_input": {"unknown": True},
                },
            })

    def test_exact_duplicate_collapses_and_conflict_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "silver.csv"
            source.write_text("fixture", encoding="utf-8")
            duplicate = _row("c1", "2026-01-01T00:00:00Z", "same")
            batch = prepare_causal_minimal_batch(
                pd.DataFrame([duplicate, duplicate]), _policy(), source_path=source
            )
            self.assertEqual([record.comment_id for record in batch.records], ["c1"])
            self.assertEqual(batch.manifest["exact_duplicates_suppressed"], 1)
            conflict = dict(duplicate, text="different")
            with self.assertRaisesRegex(DuplicateCommentConflictError, "conflicting"):
                prepare_causal_minimal_batch(
                    pd.DataFrame([duplicate, conflict]), _policy(), source_path=source
                )

    def test_repeated_text_short_text_and_replies_are_retained(self) -> None:
        rows = [
            _row("c1", "2026-01-01T00:00:00Z", "Sí"),
            _row("c2", "2026-01-01T00:00:01Z", "Sí"),
            _row("c3", "2026-01-01T00:00:02Z", "No"),
            _row("c4", "2026-01-01T00:00:03Z", "🔥"),
            _row("c5", "2026-01-01T00:00:04Z", "Totalmente", parent="c1"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "silver.csv"
            source.write_text("fixture", encoding="utf-8")
            batch = prepare_causal_minimal_batch(
                pd.DataFrame(reversed(rows)), _policy(), source_path=source
            )
        self.assertEqual([record.comment_id for record in batch.records], [
            "c1", "c2", "c3", "c4", "c5",
        ])
        self.assertEqual([record.original_text for record in batch.records], [
            "Sí", "Sí", "No", "🔥", "Totalmente",
        ])
        self.assertTrue(batch.records[-1].is_reply)

    def test_cutoff_order_and_duplicate_across_cycles(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "silver.csv"
            source.write_text("fixture", encoding="utf-8")
            state = CausalMinimalInputPolicy(_policy(), source_path=source)
            first = _row("c1", "2026-01-01T00:00:00Z", "one")
            future = _row("c3", "2026-01-01T00:02:00Z", "future")
            cutoff = datetime(2026, 1, 1, 0, 1, tzinfo=UTC)
            self.assertIsNotNone(state.process(first, source_row_index=0, data_cutoff_utc=cutoff))
            self.assertIsNone(state.process(first, source_row_index=0, data_cutoff_utc=cutoff))
            self.assertIsNone(state.process(future, source_row_index=2, data_cutoff_utc=cutoff))
            self.assertEqual(state.records_emitted, 1)
            self.assertEqual(state.exact_duplicates_suppressed, 1)
            self.assertEqual(state.future_records_withheld, 1)
            self.assertNotIn("c3", state.seen_comment_ids)

            second = _row("c2", "2026-01-01T00:01:00Z", "two")
            self.assertIsNotNone(state.process(
                second,
                source_row_index=1,
                data_cutoff_utc=datetime(2026, 1, 1, 0, 3, tzinfo=UTC),
            ))
            with self.assertRaisesRegex(ValueError, "nondecreasing"):
                state.process(
                    _row("c0", "2025-12-31T23:59:00Z", "late regression"),
                    source_row_index=3,
                )

    def test_activity_and_stance_adapters_share_ids_and_causal_parent(self) -> None:
        rows = [
            _row("root", "2026-01-01T00:00:00Z", "Parent"),
            _row("reply", "2026-01-01T00:00:01Z", "Totalmente", parent="root"),
            _row("orphan", "2026-01-01T00:00:02Z", "Sí", parent="absent"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "silver.csv"
            source.write_text("fixture", encoding="utf-8")
            batch = prepare_causal_minimal_batch(pd.DataFrame(rows), _policy(), source_path=source)
        activity = adapt_activity_records(batch.records)
        stance = adapt_stance_records(batch.records)
        self.assertEqual(
            [item.comment_id for item in activity],
            [item.classifier_input.comment_id for item in stance],
        )
        reply = next(item for item in stance if item.classifier_input.comment_id == "reply")
        self.assertEqual(reply.classifier_input.text, "Totalmente")
        self.assertEqual(reply.classifier_input.parent_text, "Parent")
        orphan = next(item for item in stance if item.classifier_input.comment_id == "orphan")
        self.assertIsNone(orphan.classifier_input.parent_text)
        self.assertIn("missing_parent", orphan.quality_flags)

    def test_cyclic_ingestion_opt_in_records_manifest_and_lineage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "silver.parquet"
            duplicate = _row("c1", "2026-01-01T00:00:00Z", "Sí")
            pd.DataFrame([
                duplicate,
                duplicate,
                _row("c2", "2026-01-01T00:01:00Z", "No"),
            ]).to_parquet(source, index=False)
            result = build_cyclic_ingestion_dry_run(
                CyclicIngestionConfig(
                    input_path=source,
                    output_dir=root / "output",
                    timezone="UTC",
                ),
                input_policy=_policy(),
            )
            self.assertEqual(result["input_rows"], 3)
            self.assertEqual(result["unique_comment_count"], 2)
            self.assertEqual(
                result["experimental_input_policy"]["exact_duplicates_suppressed"], 1
            )
            manifest = json.loads(
                (root / "output" / "online_simulation_manifest.json").read_text()
            )
            self.assertEqual(
                manifest["experimental_input_policy"]["input_policy_id"],
                "causal_minimal_v1",
            )
            inventory = pd.read_csv(root / "output" / "cycle_input_inventory.csv")
            self.assertEqual(inventory["input_policy_id"].unique().tolist(), [
                "causal_minimal_v1"
            ])
            self.assertEqual(inventory["input_source_row_index"].tolist(), [0, 2])
            self.assertEqual(result["future_leak_count"], 0)

    def test_complete_cyclic_activity_route_accepts_opt_in_policy(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        profile = json.loads((
            repository / "configs" / "development" /
            "cyclic_activity_page_hinkley.json"
        ).read_text(encoding="utf-8"))
        comments = repository / "tests" / "fixtures" / "cyclic_runtime_comments.csv"
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "runtime"
            profile["data"] = {
                "local_files": {
                    "comments_path": str(comments),
                    "videos_path": str(repository / "data" / "videos_preliminares.csv"),
                },
                "experimental_input": {},
            }
            profile["simulation"]["ingestion"]["input_path"] = str(comments)
            profile["simulation"]["ingestion"]["output_dir"] = str(output)
            profile["simulation"]["orchestration"]["simulation_dir"] = str(output)
            profile["simulation"]["stateful_adapter"]["simulation_dir"] = str(output)
            profile["detection"]["connector"]["simulation_dir"] = str(output)
            profile["detection"]["connector"]["canonical_dataset_path"] = str(comments)
            resolved = resolve_run_config(
                run_config_from_mapping(profile), base_dir=repository
            )
            summary = run_cyclic_pipeline(resolved)
            self.assertEqual(
                summary["stages"]["ingestion"]["experimental_input_policy"]["input_policy_id"],
                "causal_minimal_v1",
            )
            self.assertEqual(summary["stages"]["ingestion"]["future_leak_count"], 0)
            self.assertEqual(
                summary["stages"]["detection_connector"]["status"], "executed"
            )

    def test_polarization_runtime_opt_in_uses_original_silver_text_and_parent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            comments_path = root / "silver.parquet"
            videos_path = root / "videos.csv"
            pd.DataFrame([
                _row("root", "2026-01-01T00:00:00Z", "Texto ORIGINAL"),
                _row("reply", "2026-01-01T00:00:10Z", "Sí", parent="root"),
            ]).to_parquet(comments_path, index=False)
            with videos_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["video_id", "title", "publishedAt"])
                writer.writeheader()
                writer.writerow({
                    "video_id": "v1", "title": "Fixture",
                    "publishedAt": "2026-01-01T00:00:00Z",
                })
            config = run_config_from_mapping({
                "identity": {"run_id": "r"},
                "data": {
                    "local_files": {
                        "comments_path": str(comments_path),
                        "videos_path": str(videos_path),
                    },
                    "experimental_input": {},
                },
                "signals": {"polarization": {
                    "proposition_id": "p", "classification_run_id": "r",
                    "window": "60s", "cadence": "30s",
                    "start_time_utc": "2026-01-01T00:00:00Z",
                    "end_time_utc": "2026-01-01T00:01:00Z",
                    "enabled_measures": ["mec"],
                }},
                "polarization": {
                    "classifier": {"proposition": {
                        "proposition_id": "p", "proposition_text": "p?", "context": "c",
                    }},
                    "runtime": {"video_ids": ["v1"]},
                    "mec": {},
                },
            })
            resolved = resolve_run_config(config, base_dir=root)
            plan = prepare_polarization_runtime(resolved.config)
            self.assertEqual([item.text for item in plan.comments], ["Texto ORIGINAL", "Sí"])
            self.assertEqual(plan.comments[1].parent_text, "Texto ORIGINAL")
            self.assertEqual(
                plan.input_policy_manifest["input_policy_id"],  # type: ignore[index]
                "causal_minimal_v1",
            )

    def test_gold_cleaning_contract_remains_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "silver.parquet"
            gold = root / "gold.parquet"
            pd.DataFrame([_row(
                "c1", "2026-01-01T00:00:00Z", "HOLA mundo feliz!!!"
            )]).to_parquet(source, index=False)
            run_clean(
                input_path=str(source),
                output_path=str(gold),
                raw_text_col="text",
                timestamp_col="published_at",
                keep_spam=True,
            )
            cleaned = pd.read_parquet(gold)
            self.assertEqual(cleaned.loc[0, "text_clean"], "hola mundo feliz exclaim_token")
            self.assertEqual(cleaned.loc[0, "comment_id"], "c1")


if __name__ == "__main__":
    unittest.main()
