"""Controlled individual stance classification. No temporal measures or detection."""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import random
from collections import Counter
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass, field, fields, replace
from datetime import datetime, timezone
from numbers import Real
from typing import Any, Protocol, cast

import requests
from dotenv import dotenv_values

from youtube_pipeline.comment_stance import (
    CLASSIFIED, ERROR, NOT_APPLICABLE, SKIPPED, UNCERTAIN, CommentStance,
)
from youtube_pipeline.signal_observations import require_text, require_utc, to_json_mapping
from youtube_pipeline.stance_classifier_prompt import (
    PROMPT_ID, PROMPT_VERSION, build_messages, response_format,
)

REFERENCE_MODEL = "gpt-4.1-2025-04-14"
OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"


def _mapping_values(cls: type, payload: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError(f"{cls.__name__} must be an object.")
    unknown = set(payload) - {f.name for f in fields(cls)}
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} fields: {sorted(unknown)}")
    return dict(payload)


def _positive_integer(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value < 1:
        raise ValueError(f"{name} must be positive.")


def _finite(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


@dataclass(frozen=True, slots=True)
class StanceProposition:
    proposition_id: str
    proposition_text: str
    context: str
    context_version: str = "provided_v1"

    def __post_init__(self) -> None:
        for name in ("proposition_id", "proposition_text", "context_version"):
            require_text(name, getattr(self, name))
        if not isinstance(self.context, str):
            raise TypeError("context must be a string (explicitly empty is allowed).")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> StanceProposition:
        return cls(**_mapping_values(cls, payload))


@dataclass(frozen=True, slots=True)
class StanceCommentInput:
    comment_id: str
    author_id: str | None
    event_time_utc: datetime
    text: str
    parent_comment_id: str | None = None
    parent_text: str | None = None
    parent_event_time_utc: datetime | None = None
    is_deleted: bool = False

    def __post_init__(self) -> None:
        require_text("comment_id", self.comment_id)
        require_utc("event_time_utc", self.event_time_utc)
        if self.author_id is not None:
            require_text("author_id", self.author_id)
        if not isinstance(self.text, str):
            raise TypeError("text must be a string.")
        if type(self.is_deleted) is not bool:
            raise TypeError("is_deleted must be a boolean.")
        if self.parent_comment_id is not None:
            require_text("parent_comment_id", self.parent_comment_id)
            if self.parent_comment_id == self.comment_id:
                raise ValueError("A comment cannot be its own parent.")
        if self.parent_text is not None:
            if not isinstance(self.parent_text, str):
                raise TypeError("parent_text must be a string or None.")
            if self.parent_comment_id is None or self.parent_event_time_utc is None:
                raise ValueError("Explicit parent text requires parent ID and UTC event time.")
        if self.parent_event_time_utc is not None:
            require_utc("parent_event_time_utc", self.parent_event_time_utc)
            if self.parent_comment_id is None:
                raise ValueError("Parent time requires parent_comment_id.")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> StanceCommentInput:
        values = _mapping_values(cls, payload)
        for name in ("event_time_utc", "parent_event_time_utc"):
            if isinstance(values.get(name), str):
                values[name] = datetime.fromisoformat(values[name].replace("Z", "+00:00"))
        return cls(**values)


@dataclass(frozen=True, slots=True)
class ClassifierRetryPolicy:
    max_attempts: int = 3
    initial_delay_seconds: float = 1.0
    max_delay_seconds: float = 60.0
    jitter_seconds: float = 1.0

    def __post_init__(self) -> None:
        _positive_integer("max_attempts", self.max_attempts)
        for name in ("initial_delay_seconds", "max_delay_seconds", "jitter_seconds"):
            value = _finite(name, getattr(self, name))
            if value < 0:
                raise ValueError(f"{name} must be >= 0.")
            object.__setattr__(self, name, value)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> ClassifierRetryPolicy:
        return cls(**_mapping_values(cls, payload))


@dataclass(frozen=True, slots=True)
class CommentStanceClassifierConfig:
    """Notebook model/inference settings; technical execution defaults, not validation.

    Concurrency defaults conservatively to one because account limits are unknown.
    Structured output needs more than the notebook's ten-token numeric CSV pair.
    Character ceilings, when supplied, are input guards, not estimated token counts.
    """

    proposition: StanceProposition
    model: str = REFERENCE_MODEL
    prompt_id: str = PROMPT_ID
    prompt_version: str = PROMPT_VERSION
    max_concurrency: int = 1
    retry_policy: ClassifierRetryPolicy = field(default_factory=ClassifierRetryPolicy)
    temperature: float = 0.0
    top_p: float = 0.9
    max_completion_tokens: int = 128
    request_timeout_seconds: float = 30.0
    max_input_characters: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.proposition, StanceProposition):
            raise TypeError("proposition must be StanceProposition.")
        require_text("model", self.model)
        if (self.prompt_id, self.prompt_version) != (PROMPT_ID, PROMPT_VERSION):
            raise ValueError("Unsupported prompt identity/version; no anonymous override.")
        _positive_integer("max_concurrency", self.max_concurrency)
        _positive_integer("max_completion_tokens", self.max_completion_tokens)
        if not isinstance(self.retry_policy, ClassifierRetryPolicy):
            raise TypeError("retry_policy must be ClassifierRetryPolicy.")
        for name in ("temperature", "top_p", "request_timeout_seconds"):
            object.__setattr__(self, name, _finite(name, getattr(self, name)))
        if not 0 <= self.temperature <= 2:
            raise ValueError("temperature must be in [0,2].")
        if not 0 <= self.top_p <= 1:
            raise ValueError("top_p must be in [0,1].")
        if self.request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be positive.")
        if self.max_input_characters is not None:
            _positive_integer("max_input_characters", self.max_input_characters)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> CommentStanceClassifierConfig:
        values = _mapping_values(cls, payload)
        if "proposition" in values:
            values["proposition"] = StanceProposition.from_mapping(values["proposition"])
        if "retry_policy" in values:
            values["retry_policy"] = ClassifierRetryPolicy.from_mapping(values["retry_policy"])
        return cls(**values)


@dataclass(frozen=True, slots=True)
class TokenUsage:
    input_tokens: int
    output_tokens: int
    total_tokens: int

    def __post_init__(self) -> None:
        for name in ("input_tokens", "output_tokens", "total_tokens"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("Token counts must be nonnegative integers.")
        if self.total_tokens != self.input_tokens + self.output_tokens:
            raise ValueError("total_tokens must equal input_tokens + output_tokens.")


@dataclass(frozen=True, slots=True)
class StanceModelResponse:
    content: str | None
    usage: TokenUsage | None = None
    finish_reason: str = "stop"
    refusal: bool = False
    response_id: str | None = None


class ProviderRequestError(RuntimeError):
    """Sanitized operational failure. Never persist provider body or credentials."""

    def __init__(self, reason: str, *, retryable: bool = False,
                 retry_after_seconds: float | None = None) -> None:
        super().__init__(reason)
        self.reason, self.retryable = reason, retryable
        self.retry_after_seconds = retry_after_seconds


class InvalidStanceOutput(ValueError):
    pass


class StanceModelTransport(Protocol):
    async def complete(self, *, config: CommentStanceClassifierConfig,
                       messages: list[dict[str, str]]) -> StanceModelResponse: ...


class CommentStanceClassifier(Protocol):
    """One explicit proposition/context is fixed for this classifier execution."""

    async def classify(self, comment: StanceCommentInput) -> CommentStance: ...


class OpenAIStanceTransport:
    """Small HTTP boundary using approved requests, not a new SDK dependency."""

    async def complete(self, *, config: CommentStanceClassifierConfig,
                       messages: list[dict[str, str]]) -> StanceModelResponse:
        return await asyncio.to_thread(self._complete_sync, config, messages)

    def _complete_sync(self, config: CommentStanceClassifierConfig,
                       messages: list[dict[str, str]]) -> StanceModelResponse:
        api_key = os.environ.get("OPENAI_API_KEY") or dotenv_values(".env").get("OPENAI_API_KEY")
        if not api_key or not api_key.strip():
            raise ProviderRequestError("missing_api_key")
        payload = {
            "model": config.model, "messages": messages,
            "temperature": config.temperature, "top_p": config.top_p,
            "max_completion_tokens": config.max_completion_tokens,
            "response_format": response_format(), "store": False,
        }
        try:
            response = requests.post(
                OPENAI_CHAT_URL, json=payload,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                timeout=config.request_timeout_seconds, allow_redirects=False,
            )
        except requests.Timeout:
            raise ProviderRequestError("request_timeout", retryable=True) from None
        except requests.ConnectionError:
            raise ProviderRequestError("connection_error", retryable=True) from None
        except requests.RequestException:
            raise ProviderRequestError("request_error") from None
        try:
            if not 200 <= response.status_code < 300:
                retry_after = None
                try:
                    header = float(response.headers.get("Retry-After", "nan"))
                    if math.isfinite(header) and header >= 0:
                        retry_after = header
                except (TypeError, ValueError):
                    pass
                raise ProviderRequestError(
                    f"http_{response.status_code}",
                    retryable=response.status_code in {408, 409, 429} or response.status_code >= 500,
                    retry_after_seconds=retry_after,
                )
            try:
                data = response.json()
                choice = data["choices"][0]
                message = choice["message"]
                content = message.get("content")
                refusal = bool(message.get("refusal"))
                finish = choice["finish_reason"]
                if not isinstance(finish, str) or (content is not None and not isinstance(content, str)):
                    raise ValueError("Invalid response shape")
            except (ValueError, TypeError, KeyError, IndexError, AttributeError):
                raise ProviderRequestError("invalid_api_response") from None
            usage = None
            try:
                raw = data["usage"]
                usage = TokenUsage(raw["prompt_tokens"], raw["completion_tokens"], raw["total_tokens"])
            except (ValueError, TypeError, KeyError):
                pass  # explicitly unreported usage, never invented zero billed tokens
            response_id = data.get("id")
            return StanceModelResponse(content, usage, finish, refusal,
                                      response_id if isinstance(response_id, str) else None)
        finally:
            response.close()


def validate_stance_output(content: str | None) -> tuple[str, int | None]:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise InvalidStanceOutput("duplicate_output_key")
            result[key] = value
        return result

    try:
        parsed = json.loads(content, object_pairs_hook=pairs) if isinstance(content, str) else None
    except (ValueError, TypeError):
        raise InvalidStanceOutput("malformed_structured_output") from None
    if not isinstance(parsed, dict) or set(parsed) != {"status", "stance"}:
        raise InvalidStanceOutput("invalid_output_fields")
    status, category = parsed["status"], parsed["stance"]
    if status not in (CLASSIFIED, NOT_APPLICABLE, UNCERTAIN):
        raise InvalidStanceOutput("invalid_output_status")
    if status == CLASSIFIED:
        if isinstance(category, bool) or not isinstance(category, int) or category not in (1, 2, 3, 4, 5):
            raise InvalidStanceOutput("invalid_output_stance")
    elif category is not None:
        raise InvalidStanceOutput("nonclassified_output_has_stance")
    return status, category


@dataclass(frozen=True, slots=True)
class ClassifiedCommentRecord:
    stance: CommentStance
    parent_comment_id: str | None
    quality_reasons: tuple[str, ...]
    attempts: int
    classification_time_utc: datetime
    input_hash: str
    error_reason: str | None = None
    model_response_id: str | None = None

    def to_mapping(self) -> dict[str, Any]:
        return {
            **to_json_mapping(self.stance), "parent_comment_id": self.parent_comment_id,
            "quality_reasons": list(self.quality_reasons), "attempts": self.attempts,
            "classification_time_utc": self.classification_time_utc.isoformat(),
            "input_hash": self.input_hash, "error_reason": self.error_reason,
            "model_response_id": self.model_response_id,
        }


class OpenAICommentStanceClassifier:
    """Controlled async calls with bounded retries, fixed proposition and ID guards."""

    def __init__(self, config: CommentStanceClassifierConfig, *,
                 transport: StanceModelTransport | None = None,
                 sleep: Callable[[float], Awaitable[None]] = asyncio.sleep) -> None:
        if not isinstance(config, CommentStanceClassifierConfig):
            raise TypeError("config must be CommentStanceClassifierConfig.")
        self.config = config
        self._transport = transport if transport is not None else OpenAIStanceTransport()
        self._sleep = sleep
        self._semaphore = asyncio.Semaphore(config.max_concurrency)
        self._seen: set[str] = set()
        self._records: dict[str, ClassifiedCommentRecord] = {}
        self._attempts = 0
        self._usage_calls = 0
        self._tokens = [0, 0, 0]

    @property
    def records(self) -> tuple[ClassifiedCommentRecord, ...]:
        return tuple(sorted(self._records.values(), key=lambda r: (r.stance.event_time_utc, r.stance.comment_id)))

    @property
    def token_accounting(self) -> dict[str, Any]:
        missing = self._attempts - self._usage_calls
        return dict(input_tokens=self._tokens[0], output_tokens=self._tokens[1],
                    total_tokens=self._tokens[2], reported_attempts=self._usage_calls,
                    unreported_attempts=missing, usage_complete=missing == 0,
                    provider_attempts=self._attempts)

    def _reserve(self, comments: list[StanceCommentInput]) -> None:
        if any(not isinstance(c, StanceCommentInput) for c in comments):
            raise TypeError("All comments must be StanceCommentInput.")
        ids = [c.comment_id for c in comments]
        if len(set(ids)) != len(ids) or self._seen.intersection(ids):
            raise ValueError("Duplicate comment_id within classifier execution.")
        self._seen.update(ids)

    async def classify(self, comment: StanceCommentInput) -> CommentStance:
        self._reserve([comment])
        return await self._classify_reserved(comment)

    async def classify_batch(self, comments: Iterable[StanceCommentInput]) -> tuple[CommentStance, ...]:
        inputs = list(comments)
        self._reserve(inputs)  # validate entire batch before any API call
        lookup = {comment.comment_id: comment for comment in inputs}
        contextualized = []
        for comment in inputs:
            parent = lookup.get(comment.parent_comment_id)
            if comment.parent_text is None and parent is not None and not parent.is_deleted:
                comment = replace(comment, parent_text=parent.text, parent_event_time_utc=parent.event_time_utc)
            contextualized.append(comment)
        output: list[CommentStance | None] = [None] * len(inputs)
        jobs = iter(enumerate(contextualized))

        async def worker() -> None:
            for index, comment in jobs:
                output[index] = await self._classify_reserved(comment)

        await asyncio.gather(*(worker() for _ in range(min(len(inputs), self.config.max_concurrency))))
        assert all(item is not None for item in output)
        return cast(tuple[CommentStance, ...], tuple(output))

    async def _classify_reserved(self, comment: StanceCommentInput) -> CommentStance:
        reasons = set()
        if comment.author_id is None:
            reasons.add("missing_author")
        attempts, response_id, error = 0, None, None
        status, category = SKIPPED, None
        parent = comment.parent_text
        if comment.is_deleted:
            reasons.add("deleted_comment")
        elif not comment.text.strip():
            reasons.add("empty_text")
        elif comment.parent_comment_id is not None and (
            parent is None or not parent.strip() or comment.parent_event_time_utc is None
            or comment.parent_event_time_utc > comment.event_time_utc
        ):
            reasons.add("missing_parent")
            if comment.parent_event_time_utc is not None and comment.parent_event_time_utc > comment.event_time_utc:
                reasons.add("noncausal_parent")
        else:
            proposition = self.config.proposition
            messages = build_messages(
                proposition_id=proposition.proposition_id, proposition_text=proposition.proposition_text,
                context=proposition.context, comment_text=comment.text, parent_text=parent,
            )
            size = sum(len(message["content"]) for message in messages)
            limit = self.config.max_input_characters
            if limit is not None and size > limit:
                reasons.add("input_budget_exceeded")
            else:
                policy = self.config.retry_policy
                for attempt in range(policy.max_attempts):
                    attempts += 1
                    try:
                        async with self._semaphore:
                            self._attempts += 1
                            response = await self._transport.complete(config=self.config, messages=messages)
                        if not isinstance(response, StanceModelResponse):
                            raise InvalidStanceOutput("invalid_transport_response")
                        if response.usage is not None:
                            if not isinstance(response.usage, TokenUsage):
                                raise InvalidStanceOutput("invalid_transport_usage")
                            self._usage_calls += 1
                            for i, value in enumerate((response.usage.input_tokens, response.usage.output_tokens, response.usage.total_tokens)):
                                self._tokens[i] += value
                        response_id = response.response_id
                        if response.refusal or response.finish_reason == "content_filter":
                            reasons.add("model_refusal")
                        elif response.finish_reason != "stop":
                            raise InvalidStanceOutput("incomplete_output")
                        else:
                            status, category = validate_stance_output(response.content)
                        break
                    except InvalidStanceOutput as failure:
                        status, error = ERROR, str(failure)
                        reasons.add(error)
                        break  # invalid output is not a transient API failure
                    except ProviderRequestError as failure:
                        if not failure.retryable or attempt + 1 == policy.max_attempts:
                            status, error = ERROR, failure.reason
                            reasons.add(error)
                            if failure.retryable:
                                reasons.add("retry_exhausted")
                            break
                        delay = policy.initial_delay_seconds * 2 ** attempt + random.random() * policy.jitter_seconds
                        if failure.retry_after_seconds is not None:
                            delay = max(delay, failure.retry_after_seconds)
                        await self._sleep(min(delay, policy.max_delay_seconds))
        quality = ("classification_error" if status == ERROR else
                   "degraded_missing_parent" if "missing_parent" in reasons else
                   "degraded_missing_author" if "missing_author" in reasons else
                   "skipped" if status == SKIPPED else "passed")
        result = CommentStance(comment.comment_id, comment.author_id, comment.event_time_utc,
                               self.config.proposition.proposition_id, category, status, quality)
        input_hash = hashlib.sha256(json.dumps(to_json_mapping(comment), sort_keys=True,
                                              ensure_ascii=False).encode("utf-8")).hexdigest()
        timestamp = datetime.now(timezone.utc)
        self._records[comment.comment_id] = ClassifiedCommentRecord(
            result, comment.parent_comment_id, tuple(sorted(reasons)), attempts,
            timestamp, input_hash, error, response_id,
        )
        return result

    @property
    def counts_by_status(self) -> dict[str, int]:
        return dict(Counter(record.stance.classification_status for record in self.records))
