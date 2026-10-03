from __future__ import annotations

import asyncio
import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import requests

from youtube_pipeline.comment_stance import CLASSIFIED, ERROR, NOT_APPLICABLE, SKIPPED, UNCERTAIN, CommentStance
from youtube_pipeline.configuration import (
    load_run_config, resolve_run_config, run_config_from_mapping, run_config_hash, run_config_to_mapping,
)
from youtube_pipeline.entrypoints.comment_stance_classifier import main
from youtube_pipeline.stance_classifier import (
    ClassifierRetryPolicy, CommentStanceClassifierConfig, InvalidStanceOutput,
    OpenAICommentStanceClassifier, OpenAIStanceTransport, ProviderRequestError,
    REFERENCE_MODEL, StanceCommentInput, StanceModelResponse, StanceProposition,
    TokenUsage, validate_stance_output,
)
from youtube_pipeline.stance_classifier_io import (
    create_configured_stance_classifier, run_configured_comment_stance_classification,
)
from youtube_pipeline.stance_classifier_prompt import PROMPT_ID, PROMPT_VERSION, build_messages, response_format

BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)
PROPOSITION = StanceProposition("p", "La reforma propuesta debe aprobarse", "Contexto técnico sintético.")
NO_WAIT = ClassifierRetryPolicy(initial_delay_seconds=0, max_delay_seconds=0, jitter_seconds=0)


def comment(comment_id="c", text="Estoy de acuerdo", author="A", seconds=0, **kwargs):
    return StanceCommentInput(comment_id, author, BASE + timedelta(seconds=seconds), text, **kwargs)


def config(**kwargs):
    return replace(CommentStanceClassifierConfig(PROPOSITION, retry_policy=NO_WAIT), **kwargs)


def response(status=CLASSIFIED, stance=4, usage=None, **kwargs):
    return StanceModelResponse(json.dumps({"status": status, "stance": stance}), usage=usage, **kwargs)


def payload():
    return {"identity": {"run_id": "synthetic_classifier"}, "polarization": {"classifier": {
        "proposition": {"proposition_id": "p", "proposition_text": PROPOSITION.proposition_text,
                        "context": PROPOSITION.context},
        "retry_policy": {"initial_delay_seconds": 0, "max_delay_seconds": 0, "jitter_seconds": 0},
    }}}


def provider(*items):
    return Mock(complete=AsyncMock(side_effect=list(items)))


class ClassifierContractsTests(unittest.TestCase):
    def test_reference_defaults_and_immutability(self):
        settings = CommentStanceClassifierConfig(PROPOSITION)
        self.assertEqual(settings.model, "gpt-4.1-2025-04-14")
        self.assertEqual((settings.temperature, settings.top_p), (0., .9))
        self.assertEqual(settings.retry_policy.max_attempts, 3)
        self.assertEqual(settings.max_concurrency, 1)
        self.assertEqual(settings.max_completion_tokens, 128)
        self.assertEqual((settings.prompt_id, settings.prompt_version), (PROMPT_ID, PROMPT_VERSION))
        self.assertFalse(hasattr(settings, "__dict__"))
        for obj, field, value in ((settings, "model", "other"), (PROPOSITION, "context", "other"),
                                   (comment(), "text", "other"), (settings.retry_policy, "max_attempts", 7)):
            with self.assertRaises(FrozenInstanceError):
                setattr(obj, field, value)

    def test_config_types_ranges_and_unknown_fields(self):
        cases = ({"max_concurrency": True}, {"max_concurrency": 0}, {"max_completion_tokens": 0},
                 {"max_completion_tokens": 3.5}, {"temperature": True}, {"temperature": float("nan")},
                 {"temperature": 3}, {"top_p": float("inf")}, {"top_p": -1},
                 {"request_timeout_seconds": 0}, {"max_input_characters": False},
                 {"model": ""}, {"prompt_version": "anonymous"}, {"proposition": {}},
                 {"retry_policy": {}})
        for changes in cases:
            with self.subTest(changes=changes), self.assertRaises((ValueError, TypeError)):
                config(**changes)
        for payload in ({"proposition": {}, "api_key": "never-version-secrets"},
                        {"proposition": {"proposition_id": "p", "proposition_text": "x", "context": "", "unknown": 1}},
                        {"proposition": {}, "retry_policy": {"retries_forever": True}}):
            with self.assertRaises((TypeError, ValueError)):
                CommentStanceClassifierConfig.from_mapping(payload)

    def test_retry_validation(self):
        for changes in ({"max_attempts": True}, {"max_attempts": 0}, {"initial_delay_seconds": -1},
                        {"max_delay_seconds": float("inf")}, {"jitter_seconds": float("nan")}):
            with self.assertRaises((TypeError, ValueError)):
                ClassifierRetryPolicy(**changes)

    def test_input_validation_and_serialized_dates(self):
        for changes in ({"text": None}, {"author_id": ""}, {"event_time_utc": BASE.replace(tzinfo=None)},
                        {"parent_comment_id": "c"}, {"parent_text": "orphan"}, {"is_deleted": 1}):
            with self.assertRaises((TypeError, ValueError)):
                replace(comment(), **changes)
        parsed = StanceCommentInput.from_mapping({"comment_id": "c", "author_id": None,
                                                 "event_time_utc": BASE.isoformat(), "text": ""})
        self.assertEqual(parsed.event_time_utc, BASE)

    def test_all_structured_categories_and_nonclassified_states(self):
        for category in range(1, 6):
            self.assertEqual(validate_stance_output(json.dumps({"status": CLASSIFIED, "stance": category})),
                             (CLASSIFIED, category))
        for status in (NOT_APPLICABLE, UNCERTAIN):
            self.assertEqual(validate_stance_output(json.dumps({"status": status, "stance": None})), (status, None))

    def test_invalid_structured_outputs(self):
        pairs = [(CLASSIFIED, category) for category in (0, 6, 3.5, None, True, "3")]
        pairs += [(NOT_APPLICABLE, 3), (UNCERTAIN, 3), ("OTHER", None), (ERROR, None), (SKIPPED, None)]
        for status, category in pairs:
            with self.assertRaises(InvalidStanceOutput):
                validate_stance_output(json.dumps({"status": status, "stance": category}))
        for content in ("4,3", None, "[]", "{}", '{"status":"CLASSIFIED","stance":4,"confidence":.9}',
                        '{"status":"CLASSIFIED","status":"NOT_APPLICABLE","stance":4}'):
            with self.assertRaises(InvalidStanceOutput):
                validate_stance_output(content)

    def test_schema_and_prompt_roles_stance_only(self):
        messages = build_messages(proposition_id="p", proposition_text="Proposición explícita", context="ctx",
                                  comment_text="Totalmente", parent_text="La reforma es terrible")
        self.assertEqual([message["role"] for message in messages], ["system", "user"])
        self.assertIn("Proposición explícita", messages[0]["content"])
        self.assertEqual(json.loads(messages[1]["content"]),
                         {"comment_text": "Totalmente", "parent_comment_text": "La reforma es terrible"})
        schema = response_format()["json_schema"]
        self.assertTrue(schema["strict"])
        self.assertFalse(schema["schema"]["additionalProperties"])
        self.assertEqual(set(schema["schema"]["properties"]), {"status", "stance"})

    def test_json_loading_resolution_serialization_and_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "classifier.json"
            path.write_text(json.dumps(payload()), encoding="utf-8")
            requested = load_run_config(path)
            resolved = resolve_run_config(requested, base_dir=directory)
            self.assertEqual(resolved.config.polarization.classifier, requested.polarization.classifier)
            serialized = run_config_to_mapping(resolved.config)
            block = serialized["polarization"]["classifier"]
            self.assertEqual(block["model"], REFERENCE_MODEL)
            self.assertEqual(block["proposition"]["context"], PROPOSITION.context)
            self.assertEqual(json.loads(resolved.canonical_json)["polarization"]["classifier"], block)
            self.assertIsInstance(create_configured_stance_classifier(requested, transport=provider()), OpenAICommentStanceClassifier)
            changed = payload()
            changed["polarization"]["classifier"]["max_concurrency"] = 2
            self.assertNotEqual(run_config_hash(requested), run_config_hash(run_config_from_mapping(changed)))
            reloaded = run_config_from_mapping({k: v for k, v in serialized.items() if v is not None})
            self.assertEqual(reloaded, resolved.config)

    def test_old_measure_config_unchanged_and_classifier_unknowns_fail(self):
        requested = run_config_from_mapping({"identity": {"run_id": "isolated"},
                                            "polarization": {"measure_id": "mec", "mec": {}}})
        self.assertNotIn("classifier", run_config_to_mapping(requested)["polarization"])
        changed = payload()
        changed["polarization"]["classifier"]["api_key"] = "not-allowed"
        with self.assertRaises(ValueError):
            run_config_from_mapping(changed)

    def test_usage_validation(self):
        for usage in ((True, 1, 2), (-1, 2, 1), (10, 5, 14), (1.5, 2, 3.5)):
            with self.assertRaises(ValueError):
                TokenUsage(*usage)


class ClassifierBehaviorTests(unittest.IsolatedAsyncioTestCase):
    async def test_classifies_all_categories_as_existing_contract(self):
        transport = provider(*(response(stance=category) for category in range(1, 6)))
        classifier = OpenAICommentStanceClassifier(config(), transport=transport)
        outputs = await classifier.classify_batch([comment(str(i)) for i in range(1, 6)])
        self.assertEqual([output.stance_category for output in outputs], list(range(1, 6)))
        self.assertTrue(all(isinstance(output, CommentStance) for output in outputs))
        self.assertEqual(transport.complete.await_count, 5)

    async def test_neutral_not_applicable_and_uncertain_distinct(self):
        transport = provider(response(stance=3), response(NOT_APPLICABLE, None), response(UNCERTAIN, None))
        classifier = OpenAICommentStanceClassifier(config(), transport=transport)
        outputs = await classifier.classify_batch([comment("neutral", "No estoy ni a favor ni en contra"),
                                                   comment("greeting", "Saludos desde Cali"), comment("ambiguous")])
        self.assertEqual([(o.classification_status, o.stance_category) for o in outputs],
                         [(CLASSIFIED, 3), (NOT_APPLICABLE, None), (UNCERTAIN, None)])
        self.assertTrue(all(output.quality == "passed" for output in outputs))

    async def test_root_prompt_and_explicit_reply_context(self):
        transport = provider(response(), response())
        classifier = OpenAICommentStanceClassifier(config(), transport=transport)
        await classifier.classify(comment("root"))
        reply = comment("reply", "Totalmente", seconds=10, parent_comment_id="root",
                        parent_text="La reforma es terrible", parent_event_time_utc=BASE)
        await classifier.classify(reply)
        messages = [call.kwargs["messages"] for call in transport.complete.await_args_list]
        self.assertIsNone(json.loads(messages[0][1]["content"])["parent_comment_text"])
        self.assertEqual(json.loads(messages[1][1]["content"])["parent_comment_text"], "La reforma es terrible")

    async def test_batch_parent_lookup_immediate_only(self):
        transport = provider(response(), response(), response())
        classifier = OpenAICommentStanceClassifier(config(), transport=transport)
        await classifier.classify_batch([comment("root", "Abuelo"),
                                         comment("parent", "Padre", seconds=1, parent_comment_id="root"),
                                         comment("reply", "Totalmente", seconds=2, parent_comment_id="parent")])
        user = json.loads(transport.complete.await_args_list[-1].kwargs["messages"][1]["content"])
        self.assertEqual(user["parent_comment_text"], "Padre")
        self.assertNotIn("Abuelo", json.dumps(user))

    async def test_missing_parent_skipped_without_provider(self):
        transport = provider()
        classifier = OpenAICommentStanceClassifier(config(), transport=transport)
        result = await classifier.classify(comment(parent_comment_id="missing"))
        self.assertEqual(result.classification_status, SKIPPED)
        self.assertIsNone(result.stance_category)
        self.assertEqual(result.quality, "degraded_missing_parent")
        self.assertIn("missing_parent", classifier.records[0].quality_reasons)
        transport.complete.assert_not_awaited()

    async def test_future_parent_never_sent_to_reply(self):
        transport = provider(response())
        classifier = OpenAICommentStanceClassifier(config(), transport=transport)
        outputs = await classifier.classify_batch([comment("reply", "Sí", parent_comment_id="future"),
                                                   comment("future", "Texto futuro", seconds=10)])
        self.assertEqual(outputs[0].classification_status, SKIPPED)
        reply_record = next(r for r in classifier.records if r.stance.comment_id == "reply")
        self.assertIn("noncausal_parent", reply_record.quality_reasons)
        self.assertEqual(transport.complete.await_count, 1)

    async def test_missing_author_quality_not_model_confidence(self):
        classifier = OpenAICommentStanceClassifier(config(), transport=provider(response(stance=5)))
        result = await classifier.classify(comment(author=None))
        self.assertEqual(result.classification_status, CLASSIFIED)
        self.assertEqual(result.stance_category, 5)
        self.assertEqual(result.quality, "degraded_missing_author")
        self.assertIn("missing_author", classifier.records[0].quality_reasons)

    async def test_empty_deleted_and_input_budget_skipped(self):
        transport = provider()
        classifier = OpenAICommentStanceClassifier(config(max_input_characters=1), transport=transport)
        outputs = await classifier.classify_batch([comment("empty", ""), comment("deleted", is_deleted=True), comment("budget")])
        self.assertTrue(all(o.classification_status == SKIPPED for o in outputs))
        reasons = {reason for record in classifier.records for reason in record.quality_reasons}
        self.assertEqual(reasons, {"empty_text", "deleted_comment", "input_budget_exceeded"})
        transport.complete.assert_not_awaited()

    async def test_duplicate_guards_before_any_batch_call_and_across_calls(self):
        transport = provider(response())
        classifier = OpenAICommentStanceClassifier(config(), transport=transport)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            await classifier.classify_batch([comment(), comment()])
        transport.complete.assert_not_awaited()
        await classifier.classify(comment())
        with self.assertRaises(ValueError):
            await classifier.classify(comment())
        self.assertEqual(transport.complete.await_count, 1)

    async def test_malformed_output_error_no_retry_and_usage_retained(self):
        transport = provider(StanceModelResponse("4,3", TokenUsage(10, 5, 15)))
        classifier = OpenAICommentStanceClassifier(config(), transport=transport)
        result = await classifier.classify(comment())
        self.assertEqual(result.classification_status, ERROR)
        self.assertEqual(classifier.records[0].attempts, 1)
        self.assertEqual(classifier.token_accounting["total_tokens"], 15)
        self.assertTrue(classifier.token_accounting["usage_complete"])
        self.assertEqual(transport.complete.await_count, 1)

    async def test_retry_success_and_unknown_failed_attempt_usage(self):
        transport = provider(ProviderRequestError("http_429", retryable=True), response(usage=TokenUsage(10, 5, 15)))
        sleep = AsyncMock()
        classifier = OpenAICommentStanceClassifier(config(), transport=transport, sleep=sleep)
        result = await classifier.classify(comment())
        self.assertEqual(result.classification_status, CLASSIFIED)
        self.assertEqual(classifier.records[0].attempts, 2)
        sleep.assert_awaited_once_with(0.)
        self.assertEqual(classifier.token_accounting["unreported_attempts"], 1)
        self.assertFalse(classifier.token_accounting["usage_complete"])

    async def test_retry_exhaustion_is_error(self):
        transport = provider(*(ProviderRequestError("connection_error", retryable=True) for _ in range(3)))
        classifier = OpenAICommentStanceClassifier(config(), transport=transport, sleep=AsyncMock())
        result = await classifier.classify(comment())
        self.assertEqual(result.classification_status, ERROR)
        self.assertIsNone(result.stance_category)
        self.assertEqual(transport.complete.await_count, 3)
        self.assertIn("retry_exhausted", classifier.records[0].quality_reasons)

    async def test_permanent_failure_no_retry(self):
        transport = provider(ProviderRequestError("http_401"))
        classifier = OpenAICommentStanceClassifier(config(), transport=transport)
        result = await classifier.classify(comment())
        self.assertEqual(result.classification_status, ERROR)
        self.assertEqual(transport.complete.await_count, 1)

    async def test_retry_backoff_and_retry_after_capped(self):
        policy = ClassifierRetryPolicy(initial_delay_seconds=1, max_delay_seconds=4, jitter_seconds=0)
        transport = provider(ProviderRequestError("http_429", retryable=True, retry_after_seconds=100),
                             ProviderRequestError("connection_error", retryable=True), response())
        sleep = AsyncMock()
        await OpenAICommentStanceClassifier(config(retry_policy=policy), transport=transport, sleep=sleep).classify(comment())
        self.assertEqual([call.args[0] for call in sleep.await_args_list], [4., 2.])

    async def test_refusal_and_incomplete_output_not_retried(self):
        for reply, status in ((response(refusal=True), SKIPPED), (response(finish_reason="length"), ERROR),
                              (response(finish_reason="content_filter"), SKIPPED)):
            transport = provider(reply)
            output = await OpenAICommentStanceClassifier(config(), transport=transport).classify(comment())
            self.assertEqual(output.classification_status, status)
            self.assertIsNone(output.stance_category)
            self.assertEqual(transport.complete.await_count, 1)

    async def test_token_usage_aggregation_and_missing_usage(self):
        transport = provider(response(usage=TokenUsage(10, 5, 15)), response(usage=TokenUsage(20, 3, 23)), response())
        classifier = OpenAICommentStanceClassifier(config(), transport=transport)
        await classifier.classify_batch([comment(str(i)) for i in range(3)])
        self.assertEqual(classifier.token_accounting, {"input_tokens": 30, "output_tokens": 8, "total_tokens": 38,
                                                     "reported_attempts": 2, "unreported_attempts": 1,
                                                     "usage_complete": False, "provider_attempts": 3})
        rows = [record.to_mapping() for record in classifier.records]
        self.assertEqual(rows[0]["token_usage"]["total_tokens"], 15)
        self.assertEqual(rows[0]["token_usage_status"], "REPORTED")
        self.assertEqual(rows[2]["token_usage_status"], "UNKNOWN")
        self.assertIsNone(rows[2]["token_usage"])

    async def test_bounded_concurrency_and_deterministic_result_order(self):
        entered, release = asyncio.Event(), asyncio.Event()
        active, peak, completed = 0, 0, []

        async def complete(**kwargs):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            if active == 2:
                entered.set()
            await release.wait()
            text = json.loads(kwargs["messages"][1]["content"])["comment_text"]
            # Yield without elapsed-time sleeps; task completion is not output identity.
            await asyncio.sleep(0)
            completed.append(text)
            active -= 1
            return response()

        classifier = OpenAICommentStanceClassifier(config(max_concurrency=2), transport=Mock(complete=complete))
        task = asyncio.create_task(classifier.classify_batch([comment(str(i), text=str(i)) for i in range(7)]))
        await asyncio.wait_for(entered.wait(), timeout=2)
        release.set()
        output = await asyncio.wait_for(task, timeout=2)
        self.assertEqual(peak, 2)
        self.assertEqual([o.comment_id for o in output], [str(i) for i in range(7)])
        self.assertEqual(len(completed), 7)

    async def test_empty_batch_no_provider(self):
        transport = provider()
        self.assertEqual(await OpenAICommentStanceClassifier(config(), transport=transport).classify_batch([]), ())
        transport.complete.assert_not_awaited()

    async def test_out_of_order_completion_keeps_comment_identity(self):
        second_finished = asyncio.Event()
        completed = []

        async def complete(**kwargs):
            text = json.loads(kwargs["messages"][1]["content"])["comment_text"]
            if text == "first":
                await second_finished.wait()
                category = 1
            else:
                category = 5
                second_finished.set()
            completed.append(text)
            return response(stance=category)

        classifier = OpenAICommentStanceClassifier(
            config(max_concurrency=2), transport=Mock(complete=complete))
        outputs = await asyncio.wait_for(classifier.classify_batch(
            [comment("a", "first"), comment("b", "second")]), timeout=2)
        self.assertEqual(completed, ["second", "first"])
        self.assertEqual([(item.comment_id, item.stance_category) for item in outputs],
                         [("a", 1), ("b", 5)])
        self.assertEqual([record.stance.comment_id for record in classifier.records], ["a", "b"])


class OpenAITransportTests(unittest.IsolatedAsyncioTestCase):
    def http_response(self, *, status=200, content=None, usage=None, finish="stop", refusal=None):
        raw = {"id": "response_synthetic", "choices": [{"message": {
            "content": content if content is not None else '{"status":"CLASSIFIED","stance":4}', "refusal": refusal,
        }, "finish_reason": finish}]}
        if usage is not None:
            raw["usage"] = usage
        return Mock(status_code=status, headers={}, json=Mock(return_value=raw))

    async def test_request_schema_model_parameters_and_no_sdk(self):
        http = self.http_response(usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-only-secret"}), patch(
            "youtube_pipeline.stance_classifier.requests.post", return_value=http,
        ) as post:
            result = await OpenAIStanceTransport().complete(config=config(), messages=[{"role": "user", "content": "test"}])
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["model"], REFERENCE_MODEL)
        self.assertTrue(payload["response_format"]["json_schema"]["strict"])
        self.assertEqual(payload["max_completion_tokens"], 128)
        self.assertFalse(payload["store"])
        self.assertFalse(post.call_args.kwargs["allow_redirects"])
        self.assertEqual(result.usage, TokenUsage(10, 5, 15))
        http.close.assert_called_once()

    async def test_http_errors_retryability_and_redaction(self):
        for status, retryable in ((400, False), (401, False), (408, True), (409, True), (429, True), (503, True)):
            http = self.http_response(status=status)
            http.headers = {"Retry-After": "2"}
            with patch.dict("os.environ", {"OPENAI_API_KEY": "test-only-secret"}), patch(
                "youtube_pipeline.stance_classifier.requests.post", return_value=http,
            ):
                with self.assertRaises(ProviderRequestError) as error:
                    await OpenAIStanceTransport().complete(config=config(), messages=[])
            self.assertEqual(error.exception.retryable, retryable)
            self.assertEqual(error.exception.retry_after_seconds, 2.)
            self.assertNotIn("secret", str(error.exception))
            http.json.assert_not_called()
            http.close.assert_called_once()

    async def test_connection_and_timeout_are_retryable(self):
        for failure in (requests.Timeout("sensitive message"), requests.ConnectionError("sensitive message")):
            with patch.dict("os.environ", {"OPENAI_API_KEY": "test-only-secret"}), patch(
                "youtube_pipeline.stance_classifier.requests.post", side_effect=failure,
            ):
                with self.assertRaises(ProviderRequestError) as error:
                    await OpenAIStanceTransport().complete(config=config(), messages=[])
            self.assertTrue(error.exception.retryable)
            self.assertNotIn("sensitive", str(error.exception))

    async def test_missing_key_no_request(self):
        with patch.dict("os.environ", {}, clear=True), patch("youtube_pipeline.stance_classifier.dotenv_values", return_value={}), patch(
            "youtube_pipeline.stance_classifier.requests.post",
        ) as post:
            with self.assertRaisesRegex(ProviderRequestError, "missing_api_key"):
                await OpenAIStanceTransport().complete(config=config(), messages=[])
            post.assert_not_called()

    async def test_malformed_api_shape_and_usage(self):
        http = self.http_response()
        http.json.return_value = {"choices": []}
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-only-secret"}), patch(
            "youtube_pipeline.stance_classifier.requests.post", return_value=http,
        ):
            with self.assertRaisesRegex(ProviderRequestError, "invalid_api_response"):
                await OpenAIStanceTransport().complete(config=config(), messages=[])
        http = self.http_response(usage={"prompt_tokens": True, "completion_tokens": 2, "total_tokens": 3})
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-only-secret"}), patch(
            "youtube_pipeline.stance_classifier.requests.post", return_value=http,
        ):
            result = await OpenAIStanceTransport().complete(config=config(), messages=[])
        self.assertIsNone(result.usage)


class ClassifierPersistenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_config_to_stances_artifact_manifest_without_downstream(self):
        transport = provider(response(stance=3, usage=TokenUsage(10, 5, 15)), response(NOT_APPLICABLE, None, usage=TokenUsage(8, 4, 12)))
        requested = run_config_from_mapping(payload())
        with tempfile.TemporaryDirectory() as directory:
            output = await run_configured_comment_stance_classification(
                requested, [comment("z", "No estoy ni a favor ni en contra", seconds=10),
                            comment("a", "Saludos desde Cali"), comment("reply", "Totalmente", parent_comment_id="missing")],
                output_dir=directory, base_dir=directory, transport=transport, input_ref="synthetic:test_only",
            )
            rows = [json.loads(line) for line in output.artifact_path.read_text().splitlines()]
            self.assertEqual([row["comment_id"] for row in rows], ["a", "reply", "z"])
            for row in rows:
                self.assertEqual(row["proposition_id"], "p")
                self.assertIn("quality_reasons", row)
                self.assertIn("parent_comment_id", row)
                self.assertIn(row["comment_kind"], {"root", "reply"})
                self.assertIn("token_usage_status", row)
                self.assertNotIn("confidence", row)
                self.assertNotIn("sentiment", row)
                self.assertNotIn("text", row)
                canonical = {key: row[key] for key in ("comment_id", "author_id", "proposition_id", "stance_category", "classification_status", "quality")}
                canonical["event_time_utc"] = datetime.fromisoformat(row["event_time_utc"])
                self.assertIsInstance(CommentStance(**canonical), CommentStance)
            manifest = json.loads(output.manifest_path.read_text())
            self.assertEqual(manifest["classifier_run_id"], "synthetic_classifier")
            self.assertEqual(manifest["model"], REFERENCE_MODEL)
            self.assertEqual(manifest["token_usage"]["total_tokens"], 27)
            self.assertEqual(manifest["counts_by_status"], {CLASSIFIED: 1, NOT_APPLICABLE: 1, SKIPPED: 1})
            self.assertEqual(manifest["proposition"]["proposition_text"], PROPOSITION.proposition_text)
            self.assertEqual(
                manifest["classification_policy"]["policy_id"],
                "comment_stance_likert5_structured_v1",
            )
            self.assertIn("system_instructions", manifest["prompt_artifact"])
            self.assertIsNone(manifest["monetary_cost"])
            self.assertTrue(output.resolved_config_path.exists())
            self.assertEqual(sorted(path.name for path in Path(directory).iterdir()),
                             ["classifier_manifest.json", "comment_stances.jsonl", "resolved_config.json"])

    async def test_existing_outputs_rejected_before_any_call(self):
        transport = provider()
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "comment_stances.jsonl").write_text("preserve", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                await run_configured_comment_stance_classification(
                    run_config_from_mapping(payload()), [comment()], output_dir=directory,
                    base_dir=directory, transport=transport,
                )
            self.assertEqual((Path(directory) / "comment_stances.jsonl").read_text(), "preserve")
        transport.complete.assert_not_awaited()

    async def test_failure_status_persisted_not_minted_classification(self):
        transport = provider(ProviderRequestError("http_401"))
        with tempfile.TemporaryDirectory() as directory:
            output = await run_configured_comment_stance_classification(
                run_config_from_mapping(payload()), [comment()], output_dir=directory,
                base_dir=directory, transport=transport,
            )
            row = json.loads(output.artifact_path.read_text())
            self.assertEqual(row["classification_status"], ERROR)
            self.assertIsNone(row["stance_category"])
            self.assertEqual(row["error_reason"], "http_401")
            self.assertFalse(json.loads(output.manifest_path.read_text())["token_usage"]["usage_complete"])


class ClassifierEntrypointTests(unittest.TestCase):
    def files(self, directory):
        root = Path(directory)
        profile, inputs = root / "profile.json", root / "input.jsonl"
        profile.write_text(json.dumps(payload()), encoding="utf-8")
        inputs.write_text(json.dumps({"comment_id": "c", "author_id": "A", "event_time_utc": BASE.isoformat(),
                                     "text": "Estoy de acuerdo"}) + "\n", encoding="utf-8")
        return profile, inputs

    def test_default_preflight_never_calls_provider(self):
        with tempfile.TemporaryDirectory() as directory:
            profile, inputs = self.files(directory)
            with patch("youtube_pipeline.stance_classifier.requests.post") as post, contextlib.redirect_stdout(io.StringIO()) as out:
                main(["--config", str(profile), "--input-jsonl", str(inputs)])
            self.assertEqual(json.loads(out.getvalue())["execution_status"], "prepared_not_executed")
            post.assert_not_called()

    def test_execute_json_entrypoint_with_mock_http_only(self):
        with tempfile.TemporaryDirectory() as directory:
            profile, inputs = self.files(directory)
            output = Path(directory) / "run"
            http = Mock(status_code=200, headers={}, json=Mock(return_value={
                "choices": [{"message": {"content": '{"status":"CLASSIFIED","stance":4}'}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            }))
            with patch.dict("os.environ", {"OPENAI_API_KEY": "test-only-secret"}), patch(
                "youtube_pipeline.stance_classifier.requests.post", return_value=http,
            ), contextlib.redirect_stdout(io.StringIO()) as out:
                main(["--config", str(profile), "--input-jsonl", str(inputs), "--execute",
                      "--output-root", str(output), "--run-id", "override_synthetic"])
            self.assertEqual(json.loads(out.getvalue())["classifier_run_id"], "override_synthetic")
            manifest = json.loads((output / "classifier_manifest.json").read_text())
            self.assertEqual(manifest["counts_by_status"], {CLASSIFIED: 1})
            self.assertEqual(manifest["classifier_run_id"], "override_synthetic")

    def test_csv_native_preflight_adapts_published_at_without_provider_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile = root / "profile.json"
            source = root / "comments.csv"
            output = root / "stances"
            configured = payload()
            configured["data"] = {
                "local_files": {
                    "videos_path": str(root / "videos"),
                    "comments_path": str(source),
                    "data_root": str(root),
                },
                "experimental_input": {
                    "policy_id": "causal_minimal_v1",
                    "source": "silver",
                    "exact_comment_id_dedup": True,
                    "semantic_filtering": False,
                }
            }
            profile.write_text(json.dumps(configured), encoding="utf-8")
            source.write_text(
                "video_id,comment_id,text,author_id,published_at,is_reply,reply_to_comment_id\n"
                "v,c1,Raíz,a1,2026-04-20T14:00:00Z,false,\n"
                "v,c2,Respuesta,a2,2026-04-20T09:01:00-05:00,true,c1\n",
                encoding="utf-8",
            )
            with patch("youtube_pipeline.stance_classifier.requests.post") as post, contextlib.redirect_stdout(io.StringIO()) as out:
                main([
                    "--config", str(profile),
                    "--comments-path", str(source),
                    "--output-root", str(output),
                    "--run-id", "d1_c1_stance",
                ])

            prepared = json.loads(out.getvalue())
            self.assertEqual(prepared["execution_status"], "prepared_not_executed")
            self.assertEqual(prepared["input_mode"], "comments_csv")
            self.assertEqual(prepared["classifier_run_id"], "d1_c1_stance")
            self.assertEqual(prepared["input_count"], 2)
            self.assertEqual(
                (prepared["roots"], prepared["replies"], prepared["missing_parents"]),
                (1, 1, 0),
            )
            self.assertEqual(prepared["expected_model_calls"], 2)
            self.assertEqual(prepared["authoritative_source_path"], str(source.absolute()))
            self.assertEqual(
                prepared["authoritative_source_sha256"],
                hashlib.sha256(source.read_bytes()).hexdigest(),
            )
            self.assertFalse(output.exists())
            post.assert_not_called()

    def test_csv_native_execute_keeps_csv_as_manifest_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile = root / "profile.json"
            source = root / "comments.csv"
            output = root / "stances"
            configured = payload()
            configured["data"] = {
                "local_files": {
                    "videos_path": str(root / "videos"),
                    "comments_path": str(source),
                    "data_root": str(root),
                },
                "experimental_input": {
                    "policy_id": "causal_minimal_v1",
                    "source": "silver",
                    "exact_comment_id_dedup": True,
                    "semantic_filtering": False,
                }
            }
            profile.write_text(json.dumps(configured), encoding="utf-8")
            source.write_text(
                "video_id,comment_id,text,author_id,published_at,is_reply,reply_to_comment_id\n"
                "v,c1,Estoy de acuerdo,a1,2026-04-20T14:00:00Z,false,\n",
                encoding="utf-8",
            )
            http = Mock(status_code=200, headers={}, json=Mock(return_value={
                "choices": [{"message": {"content": '{"status":"CLASSIFIED","stance":4}'}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            }))
            with patch.dict("os.environ", {"OPENAI_API_KEY": "test-only-secret"}), patch(
                "youtube_pipeline.stance_classifier.requests.post", return_value=http,
            ), contextlib.redirect_stdout(io.StringIO()):
                main([
                    "--config", str(profile),
                    "--comments-path", str(source),
                    "--execute",
                    "--output-root", str(output),
                    "--run-id", "d1_c1_stance",
                ])

            manifest = json.loads((output / "classifier_manifest.json").read_text())
            self.assertEqual(manifest["classifier_run_id"], "d1_c1_stance")
            self.assertEqual(manifest["input_ref"], str(source.absolute()))
            self.assertEqual(
                manifest["input_source_sha256"],
                hashlib.sha256(source.read_bytes()).hexdigest(),
            )
            self.assertEqual(manifest["input_policy_id"], "causal_minimal_v1")
            self.assertNotEqual(manifest["input_ref"], str(output / "comment_stances.jsonl"))
            self.assertEqual(http.json.call_count, 1)


if __name__ == "__main__":
    unittest.main()
