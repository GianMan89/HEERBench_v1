"""Native Gemini generateContent and inline batch APIs, with implicit prefix caching."""
from __future__ import annotations

import json
from typing import Any, Mapping, Sequence
from urllib.parse import quote

from .base import (BatchHandle, BatchStatus, GenerationRequest, GenerationResponse,
                   Provider, ProviderError, TokenUsage, count)


class GeminiProvider(Provider):
    kind = "gemini"
    default_base_url = "https://generativelanguage.googleapis.com/v1beta"
    default_key_env = "GEMINI_API_KEY"
    supports_batch = True

    def _headers(self) -> dict[str, str]:
        return {"x-goog-api-key": self._key()}

    @staticmethod
    def _model_path(model: str) -> str:
        return "/models/" + quote(model.removeprefix("models/"), safe="-._")

    def generation_path(self, request: GenerationRequest) -> str:
        return self._model_path(request.model) + ":generateContent"

    def request_body(self, request: GenerationRequest) -> dict[str, Any]:
        params = self._parameters(request, {"model", "contents", "systemInstruction", "tools",
                                           "toolConfig", "cachedContent", "continuationToken"})
        if self.config.get("cache", {}).get("mode", "implicit") not in {"implicit", "provider_default"}:
            raise ProviderError("configuration", "This Gemini adapter uses implicit caching only; "
                                "it does not create billed cache-storage resources")
        generation = dict(params.pop("generationConfig", {}))
        if {"candidateCount", "maxOutputTokens"}.intersection(generation):
            raise ProviderError("configuration", "Candidate count and token budget cannot be overridden")
        generation.update({"maxOutputTokens": request.max_output_tokens, "candidateCount": 1})
        if request.temperature is not None:
            generation["temperature"] = request.temperature
        if request.top_p is not None:
            generation["topP"] = request.top_p
        if request.seed is not None:
            generation["seed"] = request.seed
        parts = ([{"text": request.prefix}] if request.prefix else []) + [{"text": request.task}]
        body = {"contents": [{"role": "user", "parts": parts}], "generationConfig": generation, **params}
        if request.system:
            body["systemInstruction"] = {"parts": [{"text": request.system}]}
        return body

    def parse_response(self, request: GenerationRequest, raw: dict[str, Any], body: Any,
                       latency: float | None, mode: str) -> GenerationResponse:
        candidates = raw.get("candidates") or []
        if len(candidates) > 1:
            raise ProviderError("invalid_response", "Expected at most one Gemini candidate")
        candidate = candidates[0] if candidates else {}
        parts = (candidate.get("content") or {}).get("parts", [])
        if any("functionCall" in p for p in parts):
            raise ProviderError("invalid_response", "Unexpected tool invocation")
        text = "\n".join(p["text"] for p in parts if "text" in p and not p.get("thought", False))
        usage = raw.get("usageMetadata") or {}
        visible = count(usage.get("candidatesTokenCount"))
        thinking = count(usage.get("thoughtsTokenCount", 0)) or 0
        total = count(usage.get("totalTokenCount"))
        prompt = count(usage.get("promptTokenCount"))
        tool_tokens = count(usage.get("toolUsePromptTokenCount", 0)) or 0
        if tool_tokens:
            raise ProviderError("invalid_usage", "Unexpected billed tool-use prompt tokens")
        output = visible + thinking if visible is not None else None
        if output is None and total is not None and prompt is not None:
            output = total - prompt
            if output < 0:
                raise ProviderError("invalid_usage", "Gemini output total is negative")
        if total is not None and prompt is not None and output is not None and total != prompt + output:
            raise ProviderError("invalid_usage", "Gemini token accounting does not reconcile")
        normalized = TokenUsage(input_tokens=prompt, output_tokens=output,
                                cached_input_tokens=count(usage.get("cachedContentTokenCount", 0)) or 0,
                                reasoning_tokens=thinking, cache_accounting_known=bool(usage), raw=usage)
        finish = candidate.get("finishReason") or (raw.get("promptFeedback") or {}).get("blockReason")
        return self._make_response(request, raw, text, raw.get("modelVersion"), finish,
                                   normalized, body, latency, mode)

    def _batch_body(self, requests: Sequence[GenerationRequest], label: str) -> dict[str, Any]:
        custom_ids, _ = self._batch_maps(requests)
        inverse = {value: key for key, value in custom_ids.items()}
        if len({r.model for r in requests}) != 1:
            raise ProviderError("configuration", "A Gemini batch must use one model")
        return {"batch": {"displayName": label[:128], "inputConfig": {"requests": {"requests": [
            {"request": self.request_body(r), "metadata": {"key": inverse[r.request_id]}}
            for r in requests]}}}}

    def estimate_batch_bytes(self, requests: Sequence[GenerationRequest]) -> int:
        return len(json.dumps(self._batch_body(requests, "heerbench"), ensure_ascii=False).encode())

    def submit_batch(self, requests: Sequence[GenerationRequest], label: str = "heerbench") -> BatchHandle:
        custom_ids, hashes = self._batch_maps(requests)
        body = self._batch_body(requests, label)
        if len(json.dumps(body, ensure_ascii=False).encode()) > 20_000_000:
            raise ProviderError("configuration", "Gemini inline batch exceeds 20 MB; submit smaller chunks")
        raw = self._json("POST", self._model_path(requests[0].model) + ":batchGenerateContent",
                         json=body, submission=True)
        name = raw.get("name")
        if not isinstance(name, str) or not name.startswith("batches/"):
            raise ProviderError("invalid_response", "Gemini batch response omitted valid name",
                                submission_uncertain=True, raw=raw)
        return BatchHandle(self.kind, name, [r.request_id for r in requests], custom_ids, hashes,
                           raw_submission=raw)

    def poll_batch(self, handle: BatchHandle) -> BatchStatus:
        raw = self._json("GET", "/" + handle.batch_id)
        state = (raw.get("metadata") or {}).get("state", raw.get("state"))
        mapping = {"JOB_STATE_PENDING": "queued", "JOB_STATE_RUNNING": "running",
                   "JOB_STATE_SUCCEEDED": "completed", "JOB_STATE_FAILED": "failed",
                   "JOB_STATE_CANCELLED": "cancelled", "JOB_STATE_EXPIRED": "expired"}
        if state not in mapping:
            if raw.get("done") is False:
                state = "JOB_STATE_RUNNING"
            elif raw.get("done") is True and raw.get("error"):
                state = "JOB_STATE_FAILED"
            else:
                raise ProviderError("invalid_response", f"Unknown Gemini batch state {state!r}")
        return BatchStatus(handle.batch_id, mapping[state], raw)

    def collect_batch(self, handle: BatchHandle,
                      requests_by_id: Mapping[str, GenerationRequest]) -> list[dict[str, Any]]:
        status = self._batch_ready(handle)
        output = status.raw.get("response") or status.raw.get("output") or status.raw.get("dest") or {}
        inline = output.get("inlinedResponses", [])
        if isinstance(inline, dict):
            inline = inline.get("inlinedResponses", [])
        if not isinstance(inline, list):
            raise ProviderError("invalid_response", "Gemini inline responses must be a list")
        if output.get("responsesFile") or output.get("fileName"):
            raise ProviderError("unsupported", "Unexpected file output for an inline Gemini batch")
        records = []
        ordered_keys = list(handle.custom_ids)
        has_keys = [bool((row.get("metadata") or {}).get("key")) for row in inline]
        # Gemini documents that inline results preserve input order. Require exact
        # cardinality before using that guarantee when request metadata is omitted.
        if inline and not all(has_keys):
            if any(has_keys) or len(inline) != len(handle.request_ids):
                raise ProviderError("batch_identity", "Cannot safely align Gemini batch results")
        for i, row in enumerate(inline):
            key = (row.get("metadata") or {}).get("key")
            if key is None:
                key = ordered_keys[i]
            records.append((key, row.get("response"), row.get("error")))
        return self._collect_records(handle, requests_by_id, records)
