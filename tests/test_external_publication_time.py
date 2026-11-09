from __future__ import annotations

import unittest
import warnings
from datetime import date, datetime, timezone

from youtube_pipeline.external_publication_time import (
    CAUSALLY_AVAILABLE,
    DATE_ONLY,
    EXACT,
    FUTURE_RELATIVE_TO_CUTOFF,
    SAME_DAY_TIME_UNKNOWN,
    UNKNOWN,
    UNKNOWN_TEMPORAL_RELATION,
    external_publication_time_from_record,
    parse_external_publication_time,
    temporal_relation_to_cutoff,
)
from youtube_pipeline.rag_generation_g2 import _normalize_external_evidence
from youtube_pipeline.rag_generation_g2_hierarchical import (
    _filter_external_evidence_by_cutoff,
    _normalize_video_external_evidence,
)


UTC = timezone.utc


class ExternalPublicationTimeTests(unittest.TestCase):
    def test_explicit_day_first_and_ambiguous_day_month(self) -> None:
        explicit = parse_external_publication_time("20/04/2026")
        ambiguous = parse_external_publication_time("04/05/2026")
        self.assertEqual((explicit.precision, explicit.calendar_date), (DATE_ONLY, date(2026, 4, 20)))
        self.assertEqual(ambiguous.calendar_date, date(2026, 5, 4))
        self.assertIsNone(ambiguous.timestamp_utc)

    def test_iso_date_is_date_only(self) -> None:
        parsed = parse_external_publication_time("2026-04-20")
        self.assertEqual((parsed.precision, parsed.calendar_date), (DATE_ONLY, date(2026, 4, 20)))
        self.assertIsNone(parsed.timestamp_utc)

    def test_iso_and_offset_datetimes_are_exact_utc(self) -> None:
        zulu = parse_external_publication_time("2026-04-20T10:15:00Z")
        offset = parse_external_publication_time("2026-04-20T05:15:00-05:00")
        expected = datetime(2026, 4, 20, 10, 15, tzinfo=UTC)
        self.assertEqual((zulu.precision, zulu.timestamp_utc), (EXACT, expected))
        self.assertEqual((offset.precision, offset.timestamp_utc), (EXACT, expected))

    def test_date_only_cutoff_relations_do_not_invent_intraday_time(self) -> None:
        cutoff = "2026-04-21T10:00:00Z"
        self.assertEqual(
            temporal_relation_to_cutoff(parse_external_publication_time("20/04/2026"), cutoff),
            CAUSALLY_AVAILABLE,
        )
        self.assertEqual(
            temporal_relation_to_cutoff(parse_external_publication_time("22/04/2026"), cutoff),
            FUTURE_RELATIVE_TO_CUTOFF,
        )
        self.assertEqual(
            temporal_relation_to_cutoff(parse_external_publication_time("21/04/2026"), cutoff),
            SAME_DAY_TIME_UNKNOWN,
        )

    def test_invalid_missing_and_naive_datetime_are_unknown(self) -> None:
        for value in ("not-a-date", None, "", "2026-04-20T10:15:00"):
            with self.subTest(value=value):
                parsed = parse_external_publication_time(value)
                self.assertEqual(parsed.precision, UNKNOWN)
                self.assertEqual(
                    temporal_relation_to_cutoff(parsed, "2026-04-21T10:00:00Z"),
                    UNKNOWN_TEMPORAL_RELATION,
                )

    def test_persisted_precision_is_authoritative_for_audit_consumers(self) -> None:
        parsed = external_publication_time_from_record({
            "published_at": "not-a-date",
            "raw_publication_date": "provider-original",
            "parsed_publication_time_utc": "2026-04-20T10:15:00+00:00",
            "publication_date": "2026-04-20",
            "publication_time_precision": "EXACT",
        })
        self.assertEqual(parsed.precision, EXACT)
        self.assertEqual(parsed.timestamp_utc, datetime(2026, 4, 20, 10, 15, tzinfo=UTC))

    def test_real_serper_shape_is_shared_by_flat_and_hierarchical_without_warning(self) -> None:
        response = {"body": {"news": [{
            "title": "Fuente",
            "snippet": "Resumen",
            "source": "Medio",
            "link": "https://example.test/news",
            "date": "04/05/2026",
            "position": 1,
        }]}}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            flat = _normalize_external_evidence(
                event_id="candidate-real-shape",
                query_id="query-1",
                query="consulta",
                serper_response=response,
                trigger_time_utc="2026-05-05T10:00:00Z",
                provider="serper_news",
                retrieved_at_utc="2026-05-05T10:01:00Z",
            )
            hierarchical = _normalize_video_external_evidence(
                event_id="candidate-real-shape",
                video_id="video-1",
                query_candidate_id="query-1",
                query="consulta",
                serper_response=response,
                trigger_time_utc="2026-05-05T10:00:00Z",
                retrieved_at_utc="2026-05-05T10:01:00Z",
            )
            kept, audit = _filter_external_evidence_by_cutoff(
                hierarchical,
                "2026-05-05T10:00:00Z",
            )
        self.assertEqual(caught, [])
        for record in (flat[0], hierarchical[0], kept[0]):
            self.assertEqual(record["raw_publication_date"], "04/05/2026")
            self.assertEqual(record["publication_date"], "2026-05-04")
            self.assertEqual(record["publication_time_precision"], DATE_ONLY)
            self.assertIsNone(record["parsed_publication_time_utc"])
        self.assertEqual(kept[0]["temporal_relation_to_cutoff"], CAUSALLY_AVAILABLE)
        self.assertEqual(audit["kept"], 1)


if __name__ == "__main__":
    unittest.main()
