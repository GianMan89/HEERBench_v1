import json
import unittest
from dataclasses import replace
from tempfile import TemporaryDirectory
from pathlib import Path
from unittest.mock import patch

import httpx

from heerbench.providers import (AnthropicProvider, GeminiProvider, GenerationRequest,
                                OpenAICompatibleProvider, OpenAIResponsesProvider,
                                ProviderError)


def request(model="model", request_id="one"):
    return GenerationRequest(request_id, model, "System", "Stable scenario", "Current task", 2048)


def oa_response(model="model", text="Answer"):
    return {"id": "response_1", "model": model, "status": "completed",
            "output": [{"type": "message", "content": [{"type": "output_text", "text": text}]}],
            "usage": {"input_tokens": 12000, "output_tokens": 100,
                      "input_tokens_details": {"cached_tokens": 4000, "cache_write_tokens": 7000,
                                               "cache_write_tokens_details": {"ttl_30m_tokens": 3000,
                                                                              "ttl_12h_tokens": 4000}},
                      "output_tokens_details": {"reasoning_tokens": 30}}}


def gem_response(model="model", text="Answer"):
    return {"modelVersion": model, "candidates": [{"finishReason": "STOP", "content": {
        "parts": [{"text": "Not answer text", "thought": True}, {"text": text}]}}],
        "usageMetadata": {"promptTokenCount": 1000, "candidatesTokenCount": 100,
                          "thoughtsTokenCount": 50, "cachedContentTokenCount": 800,
                          "totalTokenCount": 1150}}


class ProviderContracts(unittest.TestCase):
    def setUp(self):
        self.secret_dir = TemporaryDirectory()
        self.addCleanup(self.secret_dir.cleanup)
        self.key_file = Path(self.secret_dir.name) / "provider.key"
        self.key_file.write_text("test-credential-files-only-123456", encoding="utf-8")

    def provider(self, cls, config, client=None):
        return cls({**config, "api_key_file": str(self.key_file)}, client)

    def client(self, handler):
        value = httpx.Client(transport=httpx.MockTransport(handler))
        self.addCleanup(value.close)
        return value

    def test_openai_prefix_breakpoint_and_usage(self):
        seen = []
        def handler(req):
            seen.append(json.loads(req.content))
            return httpx.Response(200, json=oa_response())
        provider = self.provider(OpenAIResponsesProvider, {"cache": {"mode": "explicit", "ttl": "30m"}}, self.client(handler))
        result = provider.generate(request())
        self.assertEqual(seen[0]["input"][1]["content"][0]["prompt_cache_breakpoint"], {"mode": "explicit"})
        self.assertNotIn("prompt_cache_breakpoint", seen[0]["input"][1]["content"][1])
        self.assertFalse(seen[0]["store"])
        self.assertEqual(result.usage.uncached_input_tokens, 1000)
        self.assertEqual(result.usage.output_tokens, 100)  # reasoning is already included
        self.assertEqual(result.usage.reasoning_tokens, 30)
        self.assertEqual(result.usage.cache_write_long_tokens, 4000)
        self.assertEqual(result.usage.raw, oa_response()["usage"])

    def test_repetitions_are_fresh_http_requests(self):
        calls = []
        def handler(req):
            calls.append(req)
            return httpx.Response(200, json=oa_response(text=f"Answer {len(calls)}"))
        provider = self.provider(OpenAIResponsesProvider, {}, self.client(handler))
        first, second = provider.generate(request()), provider.generate(request(request_id="two"))
        self.assertEqual(len(calls), 2)
        self.assertNotEqual(first.text, second.text)
        self.assertEqual(first.request_payload_hash, second.request_payload_hash)

    def test_model_mismatch_is_not_silently_accepted(self):
        provider = self.provider(OpenAIResponsesProvider, {}, self.client(lambda _: httpx.Response(200, json=oa_response("other"))))
        with self.assertRaises(ProviderError) as error:
            provider.generate(request())
        self.assertEqual(error.exception.kind, "model_identity")

    def test_rate_limit_has_no_hidden_retry(self):
        calls = []
        def handler(req):
            calls.append(req)
            return httpx.Response(429, headers={"retry-after": "2"}, json={"error": "quota"})
        provider = self.provider(OpenAIResponsesProvider, {}, self.client(handler))
        with self.assertRaises(ProviderError) as error:
            provider.generate(request())
        self.assertEqual(len(calls), 1)
        self.assertTrue(error.exception.retryable)
        self.assertEqual(error.exception.retry_after, 2)

    def test_ambiguous_submission_is_not_retryable(self):
        def handler(req):
            raise httpx.ReadTimeout("lost response", request=req)
        provider = self.provider(AnthropicProvider, {}, self.client(handler))
        with self.assertRaises(ProviderError) as error:
            provider.submit_batch([request()])
        self.assertTrue(error.exception.submission_uncertain)
        self.assertFalse(error.exception.retryable)

    def test_lost_synchronous_response_is_also_billing_uncertain(self):
        def handler(req):
            raise httpx.ReadTimeout("lost response", request=req)
        provider = self.provider(OpenAIResponsesProvider, {}, self.client(handler))
        with self.assertRaises(ProviderError) as error:
            provider.generate(request())
        self.assertTrue(error.exception.submission_uncertain)
        self.assertFalse(error.exception.retryable)

    def test_anthropic_usage_adds_disjoint_cache_categories(self):
        raw = {"model": "model", "stop_reason": "end_turn", "content": [{"type": "text", "text": "A"}],
               "usage": {"input_tokens": 100, "output_tokens": 20, "cache_read_input_tokens": 500,
                         "cache_creation_input_tokens": 400,
                         "cache_creation": {"ephemeral_5m_input_tokens": 300, "ephemeral_1h_input_tokens": 100}}}
        provider = self.provider(AnthropicProvider, {"cache": {"mode": "explicit", "ttl": "5m"}},
                                     self.client(lambda _: httpx.Response(200, json=raw)))
        result = provider.generate(request())
        self.assertEqual(result.usage.input_tokens, 1000)
        self.assertEqual(result.usage.uncached_input_tokens, 100)
        self.assertEqual(result.usage.cache_write_short_tokens, 300)
        body = provider.request_body(request())
        self.assertEqual(body["messages"][0]["content"][0]["cache_control"]["ttl"], "5m")
        self.assertNotIn("cache_control", body["messages"][0]["content"][1])

    def test_gemini_thinking_billing_without_exposing_thoughts_as_answer(self):
        provider = self.provider(GeminiProvider, {}, self.client(lambda _: httpx.Response(200, json=gem_response())))
        result = provider.generate(request())
        self.assertEqual(result.text, "Answer")
        self.assertEqual(result.usage.output_tokens, 150)
        self.assertEqual(result.usage.uncached_input_tokens, 200)

    def test_invalid_token_arithmetic_rejected(self):
        broken = gem_response()
        broken["usageMetadata"]["totalTokenCount"] = 1001
        provider = self.provider(GeminiProvider, {}, self.client(lambda _: httpx.Response(200, json=broken)))
        with self.assertRaises(ProviderError):
            provider.generate(request())

    def test_compatible_requires_pinned_openrouter_backend(self):
        provider = self.provider(OpenAICompatibleProvider, {"base_url": "https://openrouter.ai/api/v1"})
        self.addCleanup(provider.close)
        with self.assertRaises(ProviderError):
            provider.request_body(request())
        provider.config["parameters"] = {"provider": {"only": ["deepinfra"], "allow_fallbacks": False}}
        self.assertEqual(provider.request_body(request())["model"], "model")
        with self.assertRaises(ProviderError):
            provider.submit_batch([request()])

    def test_cannot_override_sources_or_tools(self):
        provider = self.provider(OpenAIResponsesProvider, {})
        self.addCleanup(provider.close)
        for parameters in ({"input": "leak"}, {"tools": [{"type": "web_search"}]}, {"model": "other"}):
            with self.assertRaises(ProviderError):
                provider.request_body(replace(request(), parameters=parameters))

    def test_unsupported_seed_rejected_not_ignored(self):
        for cls in (OpenAIResponsesProvider, AnthropicProvider):
            provider = self.provider(cls, {})
            self.addCleanup(provider.close)
            with self.assertRaises(ProviderError):
                provider.request_body(replace(request(), seed=1))

    def test_openai_batch_submit_poll_and_unordered_partial_results(self):
        calls, submitted = [], {}
        def handler(req):
            calls.append((req.method, req.url.path))
            if req.url.path == "/v1/files":
                self.assertIn(b'"url": "/v1/responses"', req.content)
                return httpx.Response(200, json={"id": "file-input"})
            if req.url.path == "/v1/batches":
                submitted.update(json.loads(req.content))
                return httpx.Response(200, json={"id": "batch-one", "status": "validating"})
            if req.url.path == "/v1/batches/batch-one":
                return httpx.Response(200, json={"status": "completed", "output_file_id": "file-out"})
            if req.url.path == "/v1/files/file-out/content":
                rows = [{"custom_id": inverse["two"], "response": {"status_code": 200, "body": oa_response(text="second")}},
                        {"custom_id": inverse["one"], "response": {"status_code": 400, "body": {"error": "bad"}}}]
                return httpx.Response(200, text="\n".join(json.dumps(v) for v in rows))
            self.fail(str(req.url))
        provider = self.provider(OpenAIResponsesProvider, {}, self.client(handler))
        requests = [request(request_id=x) for x in ("one", "two", "three")]
        handle = provider.submit_batch(requests)
        inverse = {value: key for key, value in handle.custom_ids.items()}
        result = provider.collect_batch(handle, {r.request_id: r for r in requests})
        self.assertEqual(submitted["completion_window"], "24h")
        self.assertEqual([row["status"] for row in result], ["failed", "succeeded", "failed"])
        self.assertEqual(result[1]["response"].text, "second")
        self.assertEqual(result[1]["response"].execution_mode, "batch")
        self.assertEqual(result[2]["error"]["kind"], "missing_batch_result")

    def test_anthropic_batch_collects_errors_by_custom_id(self):
        submitted = {}
        raw = {"model": "model", "stop_reason": "end_turn", "content": [{"type": "text", "text": "ok"}],
               "usage": {"input_tokens": 10, "output_tokens": 5}}
        def handler(req):
            if req.method == "POST":
                submitted.update(json.loads(req.content))
                return httpx.Response(200, json={"id": "msgbatch-one"})
            if req.url.path.endswith("/results"):
                cid = submitted["requests"][0]["custom_id"]
                return httpx.Response(200, text=json.dumps({"custom_id": cid, "result": {"type": "succeeded", "message": raw}}))
            return httpx.Response(200, json={"processing_status": "ended"})
        provider = self.provider(AnthropicProvider, {}, self.client(handler))
        req = request(request_id="long:id/not/claude-compatible")
        handle = provider.submit_batch([req])
        self.assertLessEqual(len(submitted["requests"][0]["custom_id"]), 64)
        self.assertEqual(provider.collect_batch(handle, {req.request_id: req})[0]["response"].text, "ok")

    def test_gemini_batch_handles_native_nested_inline_output_and_metadata(self):
        sent = {}
        def handler(req):
            if req.method == "POST":
                sent.update(json.loads(req.content))
                return httpx.Response(200, json={"name": "batches/one"})
            entries = sent["batch"]["inputConfig"]["requests"]["requests"]
            return httpx.Response(200, json={"done": True, "metadata": {"state": "JOB_STATE_SUCCEEDED"},
                "response": {"inlinedResponses": {"inlinedResponses": [
                    {"metadata": entries[1]["metadata"], "error": {"code": 13}},
                    {"metadata": entries[0]["metadata"], "response": gem_response()}]}}})
        provider = self.provider(GeminiProvider, {}, self.client(handler))
        requests = [request(request_id="one"), request(request_id="two")]
        handle = provider.submit_batch(requests)
        result = provider.collect_batch(handle, {r.request_id: r for r in requests})
        self.assertEqual([r["status"] for r in result], ["succeeded", "failed"])
        self.assertEqual(result[0]["response"].execution_mode, "batch")

    def test_batch_payload_change_detected(self):
        sent = {}
        def handler(req):
            if req.method == "POST":
                sent.update(json.loads(req.content))
                return httpx.Response(200, json={"id": "msgbatch-one"})
            if req.url.path.endswith("/results"):
                return httpx.Response(200, text=json.dumps({"custom_id": sent["requests"][0]["custom_id"],
                                                          "result": {"type": "expired"}}))
            return httpx.Response(200, json={"processing_status": "ended"})
        provider = self.provider(AnthropicProvider, {}, self.client(handler))
        req = request()
        handle = provider.submit_batch([req])
        with self.assertRaises(ProviderError) as error:
            provider.collect_batch(handle, {req.request_id: replace(req, task="changed")})
        self.assertEqual(error.exception.kind, "batch_identity")


if __name__ == "__main__":
    unittest.main()
