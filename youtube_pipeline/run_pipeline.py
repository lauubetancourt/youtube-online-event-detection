#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from youtube_pipeline import (  # noqa: E402
    clean_comments_from_config,
    get_detector_names,
    persist_local_files,
    run_extraction_pipeline,
    run_prepared_replay,
)
from youtube_pipeline.entrypoints.cleaning import (  # noqa: E402
    resolve_cleaning_config,
)
from youtube_pipeline.entrypoints.local_files_storage import (  # noqa: E402
    LEGACY_COMMENTS_PATH,
    LEGACY_VIDEOS_PATH,
    resolve_local_files_config,
)
from youtube_pipeline.entrypoints.prepared_replay import (  # noqa: E402
    legacy_replay_detector_params,
    resolve_legacy_prepared_replay_run,
)
from youtube_pipeline.entrypoints.youtube_extraction import (  # noqa: E402
    resolve_youtube_api_key,
    resolve_youtube_extraction_config,
)
from youtube_pipeline.detectors import XiaoEMAConfig  # noqa: E402

_XIAO_COMPATIBILITY_DEFAULTS = XiaoEMAConfig()
DEFAULT_TRIGGER_THRESHOLD = _XIAO_COMPATIBILITY_DEFAULTS.sensitivity_threshold
DEFAULT_TRIGGER_MIN_VOLUME = _XIAO_COMPATIBILITY_DEFAULTS.v_min
DEFAULT_TRIGGER_WINDOW_SIZE = _XIAO_COMPATIBILITY_DEFAULTS.window_size
DEFAULT_TRIGGER_SLIDE_INTERVAL = _XIAO_COMPATIBILITY_DEFAULTS.slide_interval
DEFAULT_TRIGGER_SLOW_WINDOW = _XIAO_COMPATIBILITY_DEFAULTS.slow_window
DEFAULT_TRIGGER_COOLDOWN = _XIAO_COMPATIBILITY_DEFAULTS.cooldown


def run_storage(
    videos_path: str,
    comments_path: str,
    data_root: str | None,
) -> dict[str, Any]:
    config = resolve_local_files_config(
        config_file=None,
        overrides={
            "videos_path": videos_path,
            "comments_path": comments_path,
            "data_root": data_root,
        },
        base_dir=Path.cwd(),
    )
    return persist_local_files(config)


def run_extract(
    *,
    config_file: str | None = None,
    data_root: str | None = None,
    query: str | None = None,
    published_after: str | None = None,
    published_before: str | None = None,
    min_views: int | None = None,
    min_comments: int | None = None,
    max_comments: int | None = None,
    max_results: int | None = None,
    save_legacy_csv: bool | None = None,
    log_level: str = "INFO",
) -> dict[str, Any]:
    overrides = {
        "data_root": data_root,
        "query": query,
        "published_after": published_after,
        "published_before": published_before,
        "min_views": min_views,
        "min_comments": min_comments,
        "max_comments": max_comments,
        "max_results": max_results,
        "save_legacy_csv": save_legacy_csv,
    }
    cfg = resolve_youtube_extraction_config(
        config_file=config_file,
        overrides={
            key: value
            for key, value in overrides.items()
            if value is not None
        },
        base_dir=Path.cwd(),
    )

    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    logger = logging.getLogger("run_pipeline.extract")
    return run_extraction_pipeline(
        cfg,
        logger,
        api_key=resolve_youtube_api_key(),
    )


def run_clean(
    input_path: str | None,
    output_path: str | None,
    raw_text_col: str | None,
    timestamp_col: str | None,
    keep_spam: bool | None,
) -> Path:
    config = resolve_cleaning_config(
        config_file=None,
        overrides={
            "input_path": input_path,
            "output_path": output_path,
            "raw_text_col": raw_text_col,
            "timestamp_col": timestamp_col,
            "keep_spam": keep_spam,
        },
        base_dir=Path.cwd(),
    )
    return clean_comments_from_config(config)


def run_playback(
    input_path: str | None,
    output_snapshots: str | None,
    ts_col: str | None = None,
    window_size: str | None = None,
    trigger_threshold: float | None = None,
    trigger_min_volume: int | None = None,
    trigger_window_size: str | None = None,
    trigger_slide_interval: str | None = None,
    trigger_slow_window: str | None = None,
    trigger_cooldown: str | None = None,
    speed: float | None = None,
    max_sleep_seconds: float | None = None,
    start: str | None = None,
    end: str | None = None,
    detector_name: str | None = None,
    detector_config_file: str | None = None,
    detector_params: dict[str, Any] | None = None,
) -> Path:
    resolved, effective_detector_name = resolve_legacy_prepared_replay_run(
        input_path=input_path,
        output_snapshots=output_snapshots,
        ts_col=ts_col,
        window_size=window_size,
        speed=speed,
        max_sleep_seconds=max_sleep_seconds,
        start=start,
        end=end,
        detector_name=detector_name,
        detector_config_file=detector_config_file,
        detector_params=detector_params,
        trigger_threshold=trigger_threshold,
        trigger_min_volume=trigger_min_volume,
        trigger_window_size=trigger_window_size,
        trigger_slide_interval=trigger_slide_interval,
        trigger_slow_window=trigger_slow_window,
        trigger_cooldown=trigger_cooldown,
        base_dir=Path.cwd(),
    )
    if resolved.config.data is None or resolved.config.data.prepared_dataset is None:
        raise ValueError("Resolved RunConfig must include data.prepared_dataset.")
    if resolved.config.simulation is None or resolved.config.simulation.replay is None:
        raise ValueError("Resolved RunConfig must include simulation.replay.")
    return run_prepared_replay(
        resolved.config.data.prepared_dataset,
        resolved.config.simulation.replay,
        detector_name=effective_detector_name,
        detector_params=legacy_replay_detector_params(
            resolved,
            effective_detector_name,
        ),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run YouTube extraction storage, cleaning, and Streamz playback pipeline."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_extract = subparsers.add_parser("extract", help="Run live extraction from YouTube API.")
    p_extract.add_argument("--config-file", default=None, help="Extraction JSON config file.")
    p_extract.add_argument("--data-root", default=None, help="Output root folder (e.g., data).")
    p_extract.add_argument("--query", default=None)
    p_extract.add_argument("--published-after", default=None)
    p_extract.add_argument("--published-before", default=None)
    p_extract.add_argument("--min-views", type=int, default=None)
    p_extract.add_argument("--min-comments", type=int, default=None)
    p_extract.add_argument("--max-comments", type=int, default=None)
    p_extract.add_argument("--max-results", type=int, default=None)
    p_extract.add_argument("--log-level", default="INFO")
    extract_csv_group = p_extract.add_mutually_exclusive_group()
    extract_csv_group.add_argument(
        "--save-legacy-csv",
        dest="save_legacy_csv",
        action="store_true",
        help="Also write data/videos_preliminares.csv and data/comments.csv.",
    )
    extract_csv_group.add_argument(
        "--no-save-legacy-csv",
        dest="save_legacy_csv",
        action="store_false",
        help="Skip compatibility CSV exports.",
    )
    p_extract.set_defaults(save_legacy_csv=None)

    p_storage = subparsers.add_parser("storage", help="Run storage phase only.")
    p_storage.add_argument("--videos-path", required=True, help="CSV/Parquet of videos.")
    p_storage.add_argument(
        "--comments-path", required=True, help="CSV/Parquet of comments."
    )
    p_storage.add_argument("--data-root", default=None, help="Output root folder.")

    p_clean = subparsers.add_parser("clean", help="Run cleaning phase only.")
    p_clean.add_argument(
        "--input-path",
        default=None,
        help="Input comments (CSV/Parquet file or Parquet dataset dir).",
    )
    p_clean.add_argument(
        "--output-path",
        default=None,
        help="Output cleaned comments (.parquet or .csv).",
    )
    p_clean.add_argument("--raw-text-col", default=None)
    p_clean.add_argument("--timestamp-col", default=None)
    p_clean.add_argument("--keep-spam", action="store_true", default=None)

    p_play = subparsers.add_parser("playback", help="Run playback phase only.")
    p_play.add_argument(
        "--input-path",
        default=None,
        help="Input cleaned dataset (CSV/Parquet).",
    )
    p_play.add_argument(
        "--output-snapshots",
        default=None,
        help="Output snapshots as CSV.",
    )
    p_play.add_argument("--ts-col", default=None)
    p_play.add_argument("--window-size", default=None)
    p_play.add_argument("--trigger-threshold", type=float, default=None)
    p_play.add_argument("--trigger-min-volume", type=int, default=None)
    p_play.add_argument("--trigger-window-size", default=None)
    p_play.add_argument("--trigger-slide-interval", default=None)
    p_play.add_argument("--trigger-slow-window", default=None)
    p_play.add_argument("--trigger-cooldown", default=None)
    p_play.add_argument("--detector", choices=get_detector_names(), default=None)
    p_play.add_argument("--detector-config-file", default=None)
    p_play.add_argument("--speed", type=float, default=None)
    p_play.add_argument(
        "--max-sleep-seconds",
        type=float,
        default=None,
    )
    p_play.add_argument("--start", default=None)
    p_play.add_argument("--end", default=None)

    p_all = subparsers.add_parser("all", help="Run full 3-phase pipeline.")
    p_all.add_argument(
        "--videos-path",
        default=LEGACY_VIDEOS_PATH,
        help="CSV/Parquet generated from extraction phase.",
    )
    p_all.add_argument(
        "--comments-path",
        default=LEGACY_COMMENTS_PATH,
        help="CSV/Parquet generated from extraction phase.",
    )
    p_all.add_argument("--data-root", default=None)
    p_all.add_argument(
        "--clean-output",
        default=None,
        help="Output cleaned comments path.",
    )
    p_all.add_argument(
        "--snapshots-output",
        default=None,
        help="Output snapshots path.",
    )
    p_all.add_argument("--raw-text-col", default=None)
    p_all.add_argument("--timestamp-col", default=None)
    p_all.add_argument("--keep-spam", action="store_true", default=None)
    p_all.add_argument("--ts-col", default=None)
    p_all.add_argument("--window-size", default=None)
    p_all.add_argument("--trigger-threshold", type=float, default=None)
    p_all.add_argument("--trigger-min-volume", type=int, default=None)
    p_all.add_argument("--trigger-window-size", default=None)
    p_all.add_argument("--trigger-slide-interval", default=None)
    p_all.add_argument("--trigger-slow-window", default=None)
    p_all.add_argument("--trigger-cooldown", default=None)
    p_all.add_argument("--detector", choices=get_detector_names(), default=None)
    p_all.add_argument("--detector-config-file", default=None)
    p_all.add_argument("--speed", type=float, default=None)
    p_all.add_argument(
        "--max-sleep-seconds",
        type=float,
        default=None,
    )
    p_all.add_argument("--start", default=None)
    p_all.add_argument("--end", default=None)
    p_all.add_argument(
        "--extract-first",
        action="store_true",
        help="Run API extraction first and use its outputs for clean/playback.",
    )
    p_all.add_argument(
        "--extract-config-file",
        default=None,
        help="Extraction JSON config used when --extract-first is enabled.",
    )
    p_all.add_argument("--extract-log-level", default="INFO")
    p_all.add_argument(
        "--skip-playback",
        action="store_true",
        help="Run storage+clean only.",
    )

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "extract":
        result = run_extract(
            config_file=args.config_file,
            data_root=args.data_root,
            query=args.query,
            published_after=args.published_after,
            published_before=args.published_before,
            min_views=args.min_views,
            min_comments=args.min_comments,
            max_comments=args.max_comments,
            max_results=args.max_results,
            save_legacy_csv=args.save_legacy_csv,
            log_level=args.log_level,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    if args.command == "storage":
        result = run_storage(args.videos_path, args.comments_path, args.data_root)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    if args.command == "clean":
        output = run_clean(
            input_path=args.input_path,
            output_path=args.output_path,
            raw_text_col=args.raw_text_col,
            timestamp_col=args.timestamp_col,
            keep_spam=args.keep_spam,
        )
        print(str(output))
        return

    if args.command == "playback":
        output = run_playback(
            input_path=args.input_path,
            output_snapshots=args.output_snapshots,
            ts_col=args.ts_col,
            window_size=args.window_size,
            speed=args.speed,
            max_sleep_seconds=args.max_sleep_seconds,
            start=args.start,
            end=args.end,
            detector_name=args.detector,
            detector_config_file=args.detector_config_file,
            trigger_threshold=args.trigger_threshold,
            trigger_min_volume=args.trigger_min_volume,
            trigger_window_size=args.trigger_window_size,
            trigger_slide_interval=args.trigger_slide_interval,
            trigger_slow_window=args.trigger_slow_window,
            trigger_cooldown=args.trigger_cooldown,
        )
        print(str(output))
        return

    if args.command == "all":
        if args.extract_first:
            extract_result = run_extract(
                config_file=args.extract_config_file,
                data_root=args.data_root,
                log_level=args.extract_log_level,
            )
            storage_result = extract_result["persisted"]
        else:
            storage_result = run_storage(args.videos_path, args.comments_path, args.data_root)

        clean_output = run_clean(
            input_path=storage_result["comments_parquet"],
            output_path=args.clean_output,
            raw_text_col=args.raw_text_col,
            timestamp_col=args.timestamp_col,
            keep_spam=args.keep_spam,
        )

        summary: dict[str, Any] = {
            "storage": storage_result,
            "clean_output": str(clean_output),
        }
        if args.extract_first:
            summary["extract"] = extract_result

        if not args.skip_playback:
            snapshots_output = run_playback(
                input_path=str(clean_output),
                output_snapshots=args.snapshots_output,
                ts_col=args.ts_col,
                window_size=args.window_size,
                speed=args.speed,
                max_sleep_seconds=args.max_sleep_seconds,
                start=args.start,
                end=args.end,
                detector_name=args.detector,
                detector_config_file=args.detector_config_file,
                trigger_threshold=args.trigger_threshold,
                trigger_min_volume=args.trigger_min_volume,
                trigger_window_size=args.trigger_window_size,
                trigger_slide_interval=args.trigger_slide_interval,
                trigger_slow_window=args.trigger_slow_window,
                trigger_cooldown=args.trigger_cooldown,
            )
            summary["snapshots_output"] = str(snapshots_output)

        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    parser.error("Unknown command.")


if __name__ == "__main__":
    main()
