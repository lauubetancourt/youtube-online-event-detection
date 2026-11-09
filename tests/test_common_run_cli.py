from __future__ import annotations

import contextlib
import io
import unittest
from pathlib import Path

from youtube_pipeline.entrypoints.common_cli import (
    DEFAULT_LOG_LEVEL,
    CommonRunCliOptions,
    parse_common_run_args,
)


class CommonRunCliTests(unittest.TestCase):
    def test_minimum_cli_requires_only_a_config_profile(self) -> None:
        options = parse_common_run_args(
            ["--config", "configs/compatibility/cyclic_current.json"],
            description="test runner",
        )

        self.assertEqual(
            options.config_path,
            Path("configs/compatibility/cyclic_current.json"),
        )
        self.assertIsNone(options.run_id)
        self.assertIsNone(options.output_root)
        self.assertIsNone(options.execution_mode)
        self.assertEqual(options.log_level, DEFAULT_LOG_LEVEL)
        self.assertIsNone(options.identity_overrides())

    def test_explicit_common_overrides_are_typed_and_preserved(self) -> None:
        options = parse_common_run_args(
            [
                "--config",
                "run.json",
                "--run-id",
                "run_override",
                "--output-root",
                "outputs/reference",
                "--dry-run",
                "--log-level",
                "DEBUG",
            ],
            description="test runner",
        )

        self.assertEqual(options.config_path, Path("run.json"))
        self.assertEqual(options.run_id, "run_override")
        self.assertEqual(options.output_root, Path("outputs/reference"))
        self.assertEqual(options.execution_mode, "dry_run")
        self.assertEqual(options.log_level, "DEBUG")
        self.assertEqual(
            options.identity_overrides(),
            {"identity": {"run_id": "run_override"}},
        )

    def test_common_options_are_immutable(self) -> None:
        options = CommonRunCliOptions(config_path=Path("run.json"))

        with self.assertRaises(AttributeError):
            options.run_id = "changed"  # type: ignore[misc]

    def test_execution_modes_are_mutually_exclusive(self) -> None:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit):
            parse_common_run_args(
                ["--config", "run.json", "--dry-run", "--execute"],
                description="test runner",
            )

        self.assertIn("not allowed with argument", stderr.getvalue())

    def test_methodological_arguments_do_not_proliferate_in_common_cli(self) -> None:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit):
            parse_common_run_args(
                ["--config", "run.json", "--v-min", "46"],
                description="test runner",
            )

        self.assertIn("unrecognized arguments: --v-min 46", stderr.getvalue())

    def test_invalid_empty_run_id_fails_at_the_entrypoint_boundary(self) -> None:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit):
            parse_common_run_args(
                ["--config", "run.json", "--run-id", ""],
                description="test runner",
            )

        self.assertIn("run_id must not be empty", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
