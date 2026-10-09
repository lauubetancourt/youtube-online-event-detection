from __future__ import annotations

import unittest

from youtube_pipeline.polarization_runtime_config import (
    ALL_FROM_COMMENT_STREAM,
    EXPLICIT_FILTER,
    PolarizationRuntimeConfig,
    resolve_video_scope,
)


class PolarizationVideoScopeTests(unittest.TestCase):
    def test_omitted_uses_all_prepared_videos_deduplicated(self) -> None:
        scope = resolve_video_scope(["A", "A", "B", "C"], None)

        self.assertEqual(scope.mode, ALL_FROM_COMMENT_STREAM)
        self.assertEqual(scope.effective_video_ids, ("A", "B", "C"))
        self.assertEqual(scope.effective_video_count, 3)

    def test_null_and_blank_prepared_ids_are_ignored(self) -> None:
        scope = resolve_video_scope(["A", None, "", "   ", float("nan"), "B"], None)

        self.assertEqual(scope.effective_video_ids, ("A", "B"))

    def test_order_does_not_change_effective_ids_or_hash(self) -> None:
        first = resolve_video_scope(["C", "A", "B", "A"], None)
        second = resolve_video_scope(["B", "C", "A"], None)

        self.assertEqual(first.effective_video_ids, second.effective_video_ids)
        self.assertEqual(first.effective_video_ids_hash, second.effective_video_ids_hash)

    def test_explicit_filter_is_intersection_and_records_partial_miss(self) -> None:
        scope = resolve_video_scope(["A", "B", "C"], ("A", "X"))

        self.assertEqual(scope.mode, EXPLICIT_FILTER)
        self.assertEqual(scope.effective_video_ids, ("A",))
        self.assertEqual(scope.configured_video_count, 2)
        self.assertEqual(scope.matched_video_count, 1)
        self.assertEqual(scope.unmatched_video_ids, ("X",))
        self.assertEqual(scope.unmatched_video_count, 1)

    def test_explicit_filter_selects_requested_subset(self) -> None:
        scope = resolve_video_scope(["A", "B", "C"], ("A", "C"))

        self.assertEqual(scope.effective_video_ids, ("A", "C"))

    def test_all_explicit_ids_unmatched_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "do not match any prepared comments"):
            resolve_video_scope(["A", "B"], ("X", "Y"))

    def test_empty_explicit_list_is_configuration_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "empty list"):
            PolarizationRuntimeConfig(video_ids=[])

    def test_omitted_and_null_config_values_are_supported(self) -> None:
        self.assertIsNone(PolarizationRuntimeConfig().video_ids)
        self.assertIsNone(PolarizationRuntimeConfig.from_mapping({"video_ids": None}).video_ids)


if __name__ == "__main__":
    unittest.main()
