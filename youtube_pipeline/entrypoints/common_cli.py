from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence


ExecutionMode = Literal["dry_run", "execute"]
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
DEFAULT_LOG_LEVEL = "INFO"


@dataclass(frozen=True, slots=True)
class CommonRunCliOptions:
    """Small, shared entrypoint contract for a configured pipeline execution."""

    config_path: Path
    run_id: str | None = None
    output_root: Path | None = None
    execution_mode: ExecutionMode | None = None
    log_level: str = DEFAULT_LOG_LEVEL

    def __post_init__(self) -> None:
        if not isinstance(self.config_path, Path):
            raise TypeError("config_path must be a Path.")
        if self.run_id is not None and not self.run_id.strip():
            raise ValueError("run_id must not be empty when provided.")
        if self.output_root is not None and not isinstance(self.output_root, Path):
            raise TypeError("output_root must be a Path or None.")
        if self.execution_mode not in {None, "dry_run", "execute"}:
            raise ValueError("execution_mode must be dry_run, execute, or None.")
        if self.log_level not in LOG_LEVELS:
            raise ValueError(
                "log_level must be one of: " + ", ".join(LOG_LEVELS) + "."
            )

    def identity_overrides(self) -> dict[str, dict[str, str]] | None:
        """Return only the explicit RunConfig identity override, if supplied."""

        if self.run_id is None:
            return None
        return {"identity": {"run_id": self.run_id}}


def build_common_run_parser(
    *,
    description: str,
    prog: str | None = None,
) -> argparse.ArgumentParser:
    """Build the common, intentionally small CLI shared by future runners."""

    parser = argparse.ArgumentParser(prog=prog, description=description)
    parser.add_argument(
        "--config",
        dest="config_path",
        type=Path,
        required=True,
        help="Versionable JSON profile describing the execution.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Explicit execution identity override.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Explicit root for artifacts produced by the entrypoint.",
    )
    execution = parser.add_mutually_exclusive_group()
    execution.add_argument(
        "--dry-run",
        dest="execution_mode",
        action="store_const",
        const="dry_run",
        help="Select the guarded dry-run execution mode.",
    )
    execution.add_argument(
        "--execute",
        dest="execution_mode",
        action="store_const",
        const="execute",
        help="Select real execution when the target runner supports it.",
    )
    parser.set_defaults(execution_mode=None)
    parser.add_argument(
        "--log-level",
        choices=LOG_LEVELS,
        default=DEFAULT_LOG_LEVEL,
        help="Infrastructure logging level (default: INFO).",
    )
    return parser


def parse_common_run_args(
    argv: Sequence[str] | None = None,
    *,
    description: str,
    prog: str | None = None,
) -> CommonRunCliOptions:
    """Parse the shared CLI without loading or resolving RunConfig yet."""

    parser = build_common_run_parser(description=description, prog=prog)
    namespace = parser.parse_args(argv)
    try:
        return CommonRunCliOptions(
            config_path=namespace.config_path,
            run_id=namespace.run_id,
            output_root=namespace.output_root,
            execution_mode=namespace.execution_mode,
            log_level=namespace.log_level,
        )
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))


__all__ = [
    "build_common_run_parser",
    "CommonRunCliOptions",
    "DEFAULT_LOG_LEVEL",
    "ExecutionMode",
    "LOG_LEVELS",
    "parse_common_run_args",
]
