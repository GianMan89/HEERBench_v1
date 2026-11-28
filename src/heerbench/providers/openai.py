"""Native Responses API and explicitly configured compatible chat endpoints."""
from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from .base import (BatchHandle, BatchStatus, GenerationRequest, GenerationResponse,
                   Provider, ProviderError, TokenUsage, count, jsonl_records)


class OpenAIResponsesProvider(Provider):
    kind = "openai_responses"
    default_base_url = "https://api.openai.com/v1"
    default_key_env = "OPENAI_API_KEY"
    supports_batch = True

    def generation_path(self, request: GenerationRequest) -> str:
        return "/responses"

    def request_body(self, request: GenerationRequest) -> dict[str, Any]:
        params = self._parameters(request, {"model", "input", "instructions", "previous_response_id",
                                           "conversation", "tools", "tool_choice", "stream", "n",
                                           "max_output_tokens", "store", "prompt_cache_options"})
        messages: list[dict[str, Any]] = []
        if request.system:
            messages.append({"role": "developer", "content": [{"type": "input_text", "text": request.system}]})
        blocks = []
        cache = self.config.get("cache", {})
        cache_mode = cache.get("mode", "provider_default")
        if request.prefix:
            block: dict[str, Any] = {"type": "input_text", "text": request.prefix}
            if cache_mode == "explicit":
                block["prompt_cache_breakpoint"] = {"mode": "explicit"}
            blocks.append(block)
        blocks.append({"type": "input_text", "text": request.task})
        messages.append({"role": "user", "content": blocks})
        body = {"model": request.model, "input": messages,
                "max_output_tokens": request.max_output_tokens, "store": False, **params}
        if request.temperature is not None:
            body["temperature"] = request.temperature
        if request.top_p is not None:
            body["top_p"] = request.top_p
        if request.seed is not None:
            raise ProviderError("configuration", "Responses API has no portable seed parameter")
        if cache_mode in {"explicit", "implicit", "off"}:
            # Explicit mode without breakpoints disables caching on GPT-5.6+.
            body["prompt_cache_options"] = {"mode": "explicit" if cache_mode == "off" else cache_mode,
                                            "ttl": cache.get("ttl", "30m")}
        elif cache_mode != "provider_default":
            raise ProviderError("configuration", "Unsupported OpenAI cache mode")
        return body

    def parse_response(self, request: GenerationRequest, raw: dict[str, Any], body: Any,
                       latency: float | None, mode: str) -> GenerationResponse:
        if raw.get("status") in {"failed", "cancelled"} or raw.get("error"):
            raise ProviderError("generation_failed", "OpenAI did not complete generation", raw=raw)
        if raw.get("status") in {"queued", "in_progress"}:
            raise ProviderError("invalid_response", "Unexpected asynchronous generation response", raw=raw,
                                submission_uncertain=True)
        usage = raw.get("usage") or {}
        details = usage.get("input_tokens_details") or {}
        writes = details.get("cache_write_tokens_details") or {}
        short = count(writes.get("ttl_30m_tokens", 0)) or 0
        long = count(writes.get("ttl_12h_tokens", 0)) or 0
        total_writes = count(details.get("cache_write_tokens", short + long)) or 0
        if total_writes < short + long:
            raise ProviderError("invalid_usage", "OpenAI cache-write components exceed total writes")
        normalized = TokenUsage(
            input_tokens=count(usage.get("input_tokens")), output_tokens=count(usage.get("output_tokens")),
            cached_input_tokens=count(details.get("cached_tokens", 0)) or 0,
            cache_write_short_tokens=short, cache_write_long_tokens=long,
            cache_write_unspecified_tokens=total_writes - short - long,
            reasoning_tokens=count((usage.get("output_tokens_details") or {}).get("reasoning_tokens")),
            cache_accounting_known=bool(usage) and "input_tokens_details" in usage, raw=usage)
        text, refusals = [], []
        for output in raw.get("output", []):
            if output.get("type") != "message":
                continue
            for content in output.get("content", []):
                if content.get("type") == "output_text":
                    text.append(content.get("text", ""))
                elif content.get("type") == "refusal":
                    refusals.append(content.get("refusal", ""))
        finish = raw.get("status")
        if raw.get("incomplete_details"):
            finish = (raw["incomplete_details"].get("reason") or finish)
        if refusals:
            finish = "refusal"
            text.extend(refusals)
        return self._make_response(request, raw, "\n".join(text), raw.get("model"),
                                   finish, normalized, body, latency, mode)

    def submit_batch(self, requests: Sequence[GenerationRequest], label: str = "heerbench") -> BatchHandle:
        custom_ids, hashes = self._batch_maps(requests)
        inverse = {value: key for key, value in custom_ids.items()}
        if len(requests) > 50000:
            raise ProviderError("configuration", "OpenAI batches allow at most 50,000 requests")
        lines = [{"custom_id": inverse[r.request_id], "method": "POST", "url": "/v1/responses",
                  "body": self.request_body(r)} for r in requests]
        content = ("\n".join(json.dumps(line, ensure_ascii=False) for line in lines) + "\n").encode()
        if len(content) > 200_000_000:
            raise ProviderError("configuration", "OpenAI batch file exceeds 200 MB")
        uploaded = self._json("POST", "/files", submission=True, data={"purpose": "batch"},
                              files={"file": ("requests.jsonl", content, "application/jsonl")})
        file_id = uploaded.get("id")
        if not file_id:
            raise ProviderError("invalid_response", "Batch upload omitted file ID", submission_uncertain=True)
        try:
            submitted = self._json("POST", "/batches", submission=True,
                                   json={"input_file_id": file_id, "endpoint": "/v1/responses",
                                         "completion_window": "24h", "metadata": {"label": label[:512]}})
        except ProviderError as exc:
            exc.raw = {"input_file_id": file_id, "label": label[:512], "provider_error": exc.raw}
            raise
        if not submitted.get("id"):
            raise ProviderError("invalid_response", "Batch submission omitted ID", submission_uncertain=True,
                                raw={"input_file_id": file_id, "submission": submitted})
        return BatchHandle(self.kind, submitted["id"], [r.request_id for r in requests], custom_ids,
                           hashes, file_id, submitted)

    def poll_batch(self, handle: BatchHandle) -> BatchStatus:
        raw = self._json("GET", "/batches/" + handle.batch_id)
        statuses = {"validating": "queued", "in_progress": "running", "finalizing": "running",
                    "completed": "completed", "failed": "failed", "expired": "expired",
                    "cancelling": "running", "cancelled": "cancelled"}
        state = raw.get("status")
        if state not in statuses:
            raise ProviderError("invalid_response", f"Unknown OpenAI batch state {state!r}")
        return BatchStatus(handle.batch_id, statuses[state], raw)

    def collect_batch(self, handle: BatchHandle,
                      requests_by_id: Mapping[str, GenerationRequest]) -> list[dict[str, Any]]:
        status = self._batch_ready(handle)
        records = []
        for key in ("output_file_id", "error_file_id"):
            file_id = status.raw.get(key)
            if not file_id:
                continue
            for row in jsonl_records(self._request("GET", "/files/" + file_id + "/content").text):
                response = row.get("response") or {}
                code = response.get("status_code")
                error = row.get("error")
                if code is not None and code != 200:
                    error = {"kind": "http", "http_status": code, "body": response.get("body")}
                records.append((row.get("custom_id"), response.get("body"), error))
        return self._collect_records(handle, requests_by_id, records)


class OpenAICompatibleProvider(Provider):
    """Chat Completions for a pinned host. Native batches are deliberately unavailable."""
    kind = "openai_compatible"
    default_base_url = "http://localhost:8000/v1"
    default_key_env = "COMPATIBLE_API_KEY"

    def generation_path(self, request: GenerationRequest) -> str:
        return "/chat/completions"

    def request_body(self, request: GenerationRequest) -> dict[str, Any]:
        params = self._parameters(request, {"model", "models", "messages", "tools", "tool_choice",
                                           "stream", "n", "max_tokens", "max_completion_tokens"})
        messages = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        # The stable scenario text precedes the task-specific suffix byte-for-byte.
        messages.append({"role": "user", "content": request.prefix + "\n\n" + request.task})
        field = self.config.get("output_budget_field", "max_tokens")
        if field not in {"max_tokens", "max_completion_tokens"}:
            raise ProviderError("configuration", "Unsupported compatible output-budget field")
        body = {"model": request.model, "messages": messages, field: request.max_output_tokens,
                "stream": False, **params}
        for name in ("temperature", "top_p", "seed"):
            value = getattr(request, name)
            if value is not None:
                body[name] = value
        if "openrouter.ai" in self.base_url:
            routing = body.get("provider", {})
            only = routing.get("only", routing.get("order", []))
            if len(only) != 1 or routing.get("allow_fallbacks") is not False:
                raise ProviderError("configuration", "OpenRouter requires one pinned provider and allow_fallbacks=false")
        return body

    def parse_response(self, request: GenerationRequest, raw: dict[str, Any], body: Any,
                       latency: float | None, mode: str) -> GenerationResponse:
        choices = raw.get("choices") or []
        if len(choices) != 1:
            raise ProviderError("invalid_response", "Exactly one completion is required")
        choice = choices[0]
        message = choice.get("message") or {}
        if message.get("tool_calls"):
            raise ProviderError("invalid_response", "Unexpected tool invocation")
        content = message.get("content")
        if content is None:
            content = message.get("refusal") or ""
        if not isinstance(content, str):
            raise ProviderError("invalid_response", "Text completion expected")
        usage = raw.get("usage") or {}
        details = usage.get("prompt_tokens_details") or {}
        cache_write = count(details.get("cache_write_tokens", 0)) or 0
        normalized = TokenUsage(
            input_tokens=count(usage.get("prompt_tokens")), output_tokens=count(usage.get("completion_tokens")),
            cached_input_tokens=count(details.get("cached_tokens", 0)) or 0,
            cache_write_unspecified_tokens=cache_write,
            reasoning_tokens=count((usage.get("completion_tokens_details") or {}).get("reasoning_tokens")),
            cache_accounting_known=("prompt_tokens_details" in usage or self.config.get("cache", {}).get("mode") == "off"),
            raw=usage)
        expected_backend = self.config.get("accepted_backend_names")
        if expected_backend and raw.get("provider") not in expected_backend:
            raise ProviderError("provider_identity", "Returned routing provider differs from pinned provider")
        return self._make_response(request, raw, content, raw.get("model"), choice.get("finish_reason"),
                                   normalized, body, latency, mode)
