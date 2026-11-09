from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import requests

from .storage import persist_batch_snapshot

SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
VIDEOS_URL = "https://www.googleapis.com/youtube/v3/videos"
CHANNELS_URL = "https://www.googleapis.com/youtube/v3/channels"
COMMENTS_URL = "https://www.googleapis.com/youtube/v3/commentThreads"

QUOTA_REASONS = {"quotaExceeded", "dailyLimitExceeded", "dailyLimitExceededUnreg"}
RETRYABLE_REASONS = {"userRateLimitExceeded", "rateLimitExceeded", "backendError"}


class QuotaExceededError(RuntimeError):
    """Raised when YouTube API quota is exhausted."""


class YouTubeAPIError(RuntimeError):
    """Raised for non-quota API failures."""


@dataclass
class ExtractionConfig:
    query: str | None = (
    "elecciones 2026 colombia|presidenciales 2026 colombia|candidatos colombia 2026 -2022"
)
    published_after: str | None = "2026-01-31T00:00:00Z"
    published_before: str | None = "2026-04-01T00:00:00Z"
    min_views: int | None = 10000
    min_comments: int | None = 100
    max_comments: int | None = 5000
    max_results: int | None = 500
    request_timeout_seconds: float = 30.0
    request_pause_seconds: float = 0.1
    retry_attempts: int = 3
    retry_backoff_seconds: float = 2.0
    quota_pause_seconds: float = 120.0
    save_legacy_csv: bool = True
    data_root: str = "data"
    metadata_path: str | None = None

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "ExtractionConfig":
        cfg = cls()
        for key, value in payload.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
        return cfg

    def as_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "published_after": self.published_after,
            "published_before": self.published_before,
            "min_views": self.min_views,
            "min_comments": self.min_comments,
            "max_comments": self.max_comments,
            "max_results": self.max_results,
            "request_timeout_seconds": self.request_timeout_seconds,
            "request_pause_seconds": self.request_pause_seconds,
            "retry_attempts": self.retry_attempts,
            "retry_backoff_seconds": self.retry_backoff_seconds,
            "quota_pause_seconds": self.quota_pause_seconds,
            "save_legacy_csv": self.save_legacy_csv,
            "data_root": self.data_root,
            "metadata_path": self.metadata_path,
        }


@dataclass
class ExtractionState:
    quota_hit: bool = False
    quota_stage: str | None = None
    errors: list[str] = field(default_factory=list)


def _is_enabled(value: Any) -> bool:
    return value is not None and (not isinstance(value, str) or value.strip() != "")


def _to_int_or_none(value: Any) -> int | None:
    if not _is_enabled(value):
        return None
    return int(value)


def _chunks(values: Iterable[str], chunk_size: int) -> Iterable[list[str]]:
    items = list(values)
    for i in range(0, len(items), chunk_size):
        yield items[i : i + chunk_size]


def _safe_author_id(snippet: dict[str, Any]) -> str | None:
    channel_data = snippet.get("authorChannelId")
    if isinstance(channel_data, dict):
        return channel_data.get("value")
    return None


def _safe_duration_seconds(value: str | None) -> int:
    if not _is_enabled(value):
        return 0
    raw = str(value)
    # Basic ISO 8601 duration parser for YouTube format (e.g., PT1H2M3S).
    match = re.fullmatch(
        r"P(?:\d+Y)?(?:\d+M)?(?:\d+D)?(?:T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?)?",
        raw,
    )
    if not match:
        return 0
    hours = int(match.group(1) or 0)
    minutes = int(match.group(2) or 0)
    seconds = int(match.group(3) or 0)
    return hours * 3600 + minutes * 60 + seconds


class YouTubeClient:
    def __init__(self, api_key: str, config: ExtractionConfig, logger: logging.Logger) -> None:
        self.api_key = api_key
        self.config = config
        self.logger = logger

    def get(self, url: str, params: dict[str, Any]) -> dict[str, Any]:
        request_params = dict(params)
        request_params["key"] = self.api_key

        for attempt in range(1, self.config.retry_attempts + 1):
            try:
                response = requests.get(
                    url,
                    params=request_params,
                    timeout=self.config.request_timeout_seconds,
                )
            except requests.exceptions.RequestException as exc:
                if attempt < self.config.retry_attempts:
                    sleep_s = self.config.retry_backoff_seconds * attempt
                    self.logger.warning(
                        "Error de red al llamar API (%s). Reintentando en %.1fs (intento %s/%s).",
                        type(exc).__name__,
                        sleep_s,
                        attempt,
                        self.config.retry_attempts,
                    )
                    time.sleep(sleep_s)
                    continue
                raise YouTubeAPIError(
                    f"Network error after {self.config.retry_attempts} attempts: {exc}"
                ) from exc

            try:
                data = response.json()
            except ValueError as exc:
                is_retryable_status = response.status_code >= 500
                if is_retryable_status and attempt < self.config.retry_attempts:
                    sleep_s = self.config.retry_backoff_seconds * attempt
                    self.logger.warning(
                        "Respuesta no-JSON (HTTP %s). Reintentando en %.1fs (intento %s/%s).",
                        response.status_code,
                        sleep_s,
                        attempt,
                        self.config.retry_attempts,
                    )
                    time.sleep(sleep_s)
                    continue
                raise YouTubeAPIError(
                    f"Invalid JSON response (HTTP {response.status_code})"
                ) from exc

            error = data.get("error")
            if error:
                reason = self._extract_reason(error)
                message = self._extract_message(error)
                if reason in QUOTA_REASONS:
                    raise QuotaExceededError(message)

                is_retryable = reason in RETRYABLE_REASONS or response.status_code >= 500
                if is_retryable and attempt < self.config.retry_attempts:
                    sleep_s = self.config.retry_backoff_seconds * attempt
                    self.logger.warning(
                        "API transient error (%s). Reintentando en %.1fs (intento %s/%s).",
                        reason,
                        sleep_s,
                        attempt,
                        self.config.retry_attempts,
                    )
                    time.sleep(sleep_s)
                    continue
                raise YouTubeAPIError(f"{reason}: {message}")

            if response.status_code >= 400:
                raise YouTubeAPIError(f"HTTP {response.status_code}: {response.text}")

            if self.config.request_pause_seconds > 0:
                time.sleep(self.config.request_pause_seconds)
            return data

        raise YouTubeAPIError("Unexpected API error after retries.")

    @staticmethod
    def _extract_reason(error_payload: dict[str, Any]) -> str:
        errors = error_payload.get("errors", [])
        if isinstance(errors, list) and errors:
            return str(errors[0].get("reason", "unknown"))
        return "unknown"

    @staticmethod
    def _extract_message(error_payload: dict[str, Any]) -> str:
        message = error_payload.get("message")
        return str(message) if message else "unknown error"


def search_videos(
    client: YouTubeClient,
    config: ExtractionConfig,
    logger: logging.Logger,
    state: ExtractionState,
) -> list[dict[str, Any]]:
    logger.info("Iniciando busqueda de videos para query: %s", config.query)
    videos: list[dict[str, Any]] = []
    next_page_token: str | None = None
    max_results = _to_int_or_none(config.max_results)

    while True:
        if max_results is not None and len(videos) >= max_results:
            break

        params: dict[str, Any] = {
            "part": "snippet",
            "type": "video",
            "order": "date",
            "maxResults": 50,
        }
        if _is_enabled(config.query):
            params["q"] = config.query
        if _is_enabled(config.published_after):
            params["publishedAfter"] = config.published_after
        if _is_enabled(config.published_before):
            params["publishedBefore"] = config.published_before
        if next_page_token:
            params["pageToken"] = next_page_token

        logger.info(
            "Buscando videos... (%s/%s)",
            len(videos),
            max_results if max_results is not None else "sin limite",
        )
        try:
            data = client.get(SEARCH_URL, params)
        except QuotaExceededError as exc:
            state.quota_hit = True
            state.quota_stage = "search_videos"
            state.errors.append(str(exc))
            logger.warning("Cuota alcanzada durante busqueda. Se guardara avance parcial.")
            break
        except YouTubeAPIError as exc:
            state.errors.append(str(exc))
            logger.error("Error en busqueda de videos: %s", exc)
            break

        video_ids = [
            item.get("id", {}).get("videoId")
            for item in data.get("items", [])
            if item.get("id", {}).get("videoId")
        ]
        if not video_ids:
            break

        details, quota_hit = get_video_details(client, video_ids, logger, state)
        if quota_hit:
            break

        for video in details:
            views = int(video.get("statistics", {}).get("viewCount", 0))
            comments = int(video.get("statistics", {}).get("commentCount", 0))
            if _is_enabled(config.min_views) and views < int(config.min_views):
                continue
            if _is_enabled(config.min_comments) and comments < int(config.min_comments):
                continue
            videos.append(video)
            if max_results is not None and len(videos) >= max_results:
                break

        next_page_token = data.get("nextPageToken")
        if not next_page_token:
            break

    logger.info("Total de videos que cumplen filtros: %s", len(videos))
    return videos


def get_video_details(
    client: YouTubeClient,
    video_ids: list[str],
    logger: logging.Logger,
    state: ExtractionState,
) -> tuple[list[dict[str, Any]], bool]:
    params = {
        "part": "snippet,statistics,contentDetails",
        "id": ",".join(video_ids),
    }
    try:
        data = client.get(VIDEOS_URL, params)
    except QuotaExceededError as exc:
        state.quota_hit = True
        state.quota_stage = "get_video_details"
        state.errors.append(str(exc))
        logger.warning("Cuota alcanzada durante consulta de detalles de videos.")
        return [], True
    except YouTubeAPIError as exc:
        state.errors.append(str(exc))
        logger.error("Error consultando detalles de videos: %s", exc)
        return [], False
    return data.get("items", []), False


def build_videos_dataframe(videos: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for video in videos:
        stats = video.get("statistics", {})
        snippet = video.get("snippet", {})
        content = video.get("contentDetails", {})
        rows.append(
            {
                "video_id": video.get("id"),
                "title": snippet.get("title"),
                "publishedAt": snippet.get("publishedAt"),
                "channel_id": snippet.get("channelId"),
                "channel_title": snippet.get("channelTitle"),
                "views": stats.get("viewCount"),
                "likes": stats.get("likeCount"),
                "comments": stats.get("commentCount"),
                "description": snippet.get("description"),
                "video_tags": ", ".join(snippet.get("tags", [])),
                "duration_seconds": _safe_duration_seconds(content.get("duration")),
                "video_url": f"https://www.youtube.com/watch?v={video.get('id')}",
            }
        )

    columns = [
        "video_id",
        "title",
        "publishedAt",
        "channel_id",
        "channel_title",
        "views",
        "likes",
        "comments",
        "description",
        "video_tags",
        "duration_seconds",
        "video_url",
    ]
    df = pd.DataFrame(rows, columns=columns)
    if not df.empty:
        df = df.drop_duplicates(subset=["video_id"], keep="first")
    return df


def enrich_video_data_with_channels(
    videos_df: pd.DataFrame,
    client: YouTubeClient,
    logger: logging.Logger,
    state: ExtractionState,
) -> pd.DataFrame:
    if videos_df.empty or "channel_id" not in videos_df.columns:
        return videos_df

    logger.info("Obteniendo metadatos de canales...")
    channel_ids = [cid for cid in videos_df["channel_id"].dropna().unique().tolist() if cid]
    channel_info: dict[str, dict[str, Any]] = {}

    for chunk in _chunks(channel_ids, 50):
        params = {"part": "snippet,statistics", "id": ",".join(chunk)}
        try:
            data = client.get(CHANNELS_URL, params)
        except QuotaExceededError as exc:
            state.quota_hit = True
            state.quota_stage = "get_channel_metadata"
            state.errors.append(str(exc))
            logger.warning("Cuota alcanzada al enriquecer canales. Se conserva avance parcial.")
            break
        except YouTubeAPIError as exc:
            state.errors.append(str(exc))
            logger.error("Error obteniendo metadatos de canales: %s", exc)
            continue

        for item in data.get("items", []):
            channel_info[item.get("id")] = {
                "channel_subscribers": item.get("statistics", {}).get("subscriberCount"),
                "channel_country": item.get("snippet", {}).get("country"),
            }

    out = videos_df.copy()
    out["channel_subscribers"] = out["channel_id"].map(
        lambda x: channel_info.get(x, {}).get("channel_subscribers")
    )
    out["channel_country"] = out["channel_id"].map(
        lambda x: channel_info.get(x, {}).get("channel_country")
    )
    return out


def get_comments_for_video(
    client: YouTubeClient,
    video_id: str,
    max_comments: int | None,
    logger: logging.Logger,
    state: ExtractionState,
) -> tuple[list[dict[str, Any]], bool]:
    comments: list[dict[str, Any]] = []
    next_page_token: str | None = None

    while True:
        if max_comments is not None and len(comments) >= max_comments:
            break

        params: dict[str, Any] = {
            "part": "snippet,replies",
            "videoId": video_id,
            "maxResults": 100,
            "textFormat": "plainText",
        }
        if next_page_token:
            params["pageToken"] = next_page_token

        try:
            data = client.get(COMMENTS_URL, params)
        except QuotaExceededError as exc:
            state.quota_hit = True
            state.quota_stage = "get_comments_for_video"
            state.errors.append(str(exc))
            logger.warning(
                "Cuota alcanzada durante extraccion de comentarios del video %s.", video_id
            )
            return comments, True
        except YouTubeAPIError as exc:
            state.errors.append(str(exc))
            logger.error("Error extrayendo comentarios de %s: %s", video_id, exc)
            return comments, False

        for item in data.get("items", []):
            top_comment = (
                item.get("snippet", {})
                .get("topLevelComment", {})
                .get("snippet", {})
            )
            top_comment_id = (
                item.get("snippet", {}).get("topLevelComment", {}).get("id")
            )

            comments.append(
                {
                    "video_id": video_id,
                    "comment_id": top_comment_id,
                    "text": top_comment.get("textDisplay"),
                    "author_name": top_comment.get("authorDisplayName"),
                    "author_id": _safe_author_id(top_comment),
                    "published_at": top_comment.get("publishedAt"),
                    "likes": top_comment.get("likeCount"),
                    "is_reply": False,
                    "reply_to_comment_id": None,
                }
            )

            replies = item.get("replies", {}).get("comments", [])
            for reply in replies:
                reply_data = reply.get("snippet", {})
                comments.append(
                    {
                        "video_id": video_id,
                        "comment_id": reply.get("id"),
                        "text": reply_data.get("textDisplay"),
                        "author_name": reply_data.get("authorDisplayName"),
                        "author_id": _safe_author_id(reply_data),
                        "published_at": reply_data.get("publishedAt"),
                        "likes": reply_data.get("likeCount"),
                        "is_reply": True,
                        "reply_to_comment_id": top_comment_id,
                    }
                )

        next_page_token = data.get("nextPageToken")
        if not next_page_token:
            break

    if max_comments is not None and len(comments) > max_comments:
        comments = comments[:max_comments]
    return comments, False


def get_all_comments(
    client: YouTubeClient,
    video_ids: list[str],
    max_comments: int | None,
    logger: logging.Logger,
    state: ExtractionState,
) -> list[dict[str, Any]]:
    all_comments: list[dict[str, Any]] = []
    for idx, video_id in enumerate(video_ids, start=1):
        logger.info("Extrayendo comentarios para video (%s/%s): %s", idx, len(video_ids), video_id)
        comments, quota_hit = get_comments_for_video(
            client=client,
            video_id=video_id,
            max_comments=max_comments,
            logger=logger,
            state=state,
        )
        all_comments.extend(comments)
        if quota_hit:
            break
    return all_comments


def build_comments_dataframe(comments: list[dict[str, Any]]) -> pd.DataFrame:
    columns = [
        "video_id",
        "comment_id",
        "text",
        "author_name",
        "author_id",
        "published_at",
        "likes",
        "is_reply",
        "reply_to_comment_id",
    ]
    return pd.DataFrame(comments, columns=columns)


def _write_legacy_csv(
    videos_df: pd.DataFrame,
    comments_df: pd.DataFrame,
    data_root: str | Path,
    logger: logging.Logger,
) -> dict[str, str]:
    root = Path(data_root)
    root.mkdir(parents=True, exist_ok=True)
    videos_csv = root / "videos_preliminares.csv"
    comments_csv = root / "comments.csv"
    videos_df.to_csv(videos_csv, index=False)
    comments_df.to_csv(comments_csv, index=False)
    logger.info("Archivos CSV compatibles guardados en %s y %s", videos_csv, comments_csv)
    return {"videos_csv": str(videos_csv), "comments_csv": str(comments_csv)}


def _write_run_metadata(
    config: ExtractionConfig,
    state: ExtractionState,
    persisted: dict[str, Any],
    logger: logging.Logger,
) -> str:
    default_path = (
        Path(config.data_root)
        / "bronze"
        / "runs"
        / f"extraction_run_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    output_path = Path(config.metadata_path) if _is_enabled(config.metadata_path) else default_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": config.as_dict(),
        "quota_hit": state.quota_hit,
        "quota_stage": state.quota_stage,
        "errors": state.errors,
        "persisted": persisted,
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Metadata de corrida guardada en: %s", output_path)
    return str(output_path)


def run_extraction_pipeline(
    config: ExtractionConfig,
    logger: logging.Logger,
    *,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Run extraction with infrastructure credentials supplied by the caller."""

    if api_key is None:
        from .entrypoints.youtube_extraction import resolve_youtube_api_key

        api_key = resolve_youtube_api_key()
    if not _is_enabled(api_key):
        raise ValueError("api_key must not be empty.")

    state = ExtractionState()
    client = YouTubeClient(api_key=str(api_key), config=config, logger=logger)

    logger.info("=== Fase 1: Busqueda y filtrado de videos ===")
    videos_raw = search_videos(client, config, logger, state)
    videos_df = build_videos_dataframe(videos_raw)
    videos_df = enrich_video_data_with_channels(videos_df, client, logger, state)

    logger.info("=== Fase 2: Extraccion de comentarios ===")
    max_comments = _to_int_or_none(config.max_comments)
    video_ids = videos_df["video_id"].dropna().astype(str).tolist()
    comments_raw = get_all_comments(client, video_ids, max_comments, logger, state)
    comments_df = build_comments_dataframe(comments_raw)

    logger.info("=== Fase 3: Persistencia para pipeline reproducible ===")
    persisted = persist_batch_snapshot(videos_df=videos_df, comments_df=comments_df, data_root=config.data_root)

    output: dict[str, Any] = {
        "videos_found": len(videos_df),
        "comments_found": len(comments_df),
        "quota_hit": state.quota_hit,
        "quota_stage": state.quota_stage,
        "persisted": persisted,
    }

    if config.save_legacy_csv:
        output["legacy_csv"] = _write_legacy_csv(videos_df, comments_df, config.data_root, logger)

    output["run_metadata"] = _write_run_metadata(config, state, persisted, logger)
    logger.info("Extraccion finalizada. Videos=%s | Comentarios=%s", len(videos_df), len(comments_df))
    return output


def main() -> None:
    """Compatibility shim; configuration and secrets belong to the entrypoint."""

    from .entrypoints.youtube_extraction import main as entrypoint_main

    entrypoint_main()


if __name__ == "__main__":
    main()
