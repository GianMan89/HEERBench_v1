"""Native Claude Messages and Message Batches APIs."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from .base import (BatchHandle, BatchStatus, GenerationRequest, GenerationResponse,
                   Provider, ProviderError, TokenUsage, count, jsonl_records)


class AnthropicProvider(Provider):
    kind = "anthropic"
    default_base_url = "https://api.anthropic.com/v1"
    default_key_env = "ANTHROPIC_API_KEY"
    supports_batch = True

    def _headers(self) -> dict[str, str]:
        return {"x-api-key": self._key(),
                "anthropic-version": self.config.get("api_version", "2026-10-21")}

    def generation_path(self, request: GenerationRequest) -> str:
        return "/messages"

    def request_body(self, request: GenerationRequest) -> dict[str, Any]:
        params = self._parameters(request, {"model", "system", "messages", "tools", "tool_choice", "stream",
                                           "max_tokens", "cache_control"})
        if request.seed is not None:
            raise ProviderError("configuration", "Anthropic Messages has no seed parameter")
        cache = self.config.get("cache", {})
        mode = cache.get("mode", "off")
        if mode not in {"off", "explicit"}:
            raise ProviderError("configuration", "Anthropic adapter supports off or explicit prefix caching")
        content = []
        if request.prefix:
            prefix: dict[str, Any] = {"type": "text", "text": request.prefix}
            if mode == "explicit":
                ttl = cache.get("ttl", "5m")
                if ttl not in {"5m", "1h"}:
                    raise ProviderError("configuration", "Anthropic cache TTL must be 5m or 1h")
                prefix["cache_control"] = {"type": "ephemeral", "ttl": ttl}
            content.append(prefix)
        content.append({"type": "text", "text": request.task})
        body: dict[str, Any] = {"model": request.model, "messages": [{"role": "user", "content": content}],
                                "max_tokens": request.max_output_tokens, **params}
        if request.system:
            body["system"] = request.system
        if request.temperature is not None:
            body["temperature"] = request.temperature
        if request.top_p is not None:
            body["top_p"] = request.top_p
        return body

    def parse_response(self, request: GenerationRequest, raw: dict[str, Any], body: Any,
                       latency: float | None, mode: str) -> GenerationResponse:
        usage = raw.get("usage") or {}
        plain = count(usage.get("input_tokens"))
        read = count(usage.get("cache_read_input_tokens", 0)) or 0
        written = count(usage.get("cache_creation_input_tokens", 0)) or 0
        split = usage.get("cache_creation") or {}
        short = count(split.get("ephemeral_5m_input_tokens", 0)) or 0
        long = count(split.get("ephemeral_1h_input_tokens", 0)) or 0
        if written < short + long:
            raise ProviderError("invalid_usage", "Anthropic cache-write components exceed total writes")
        normalized = TokenUsage(input_tokens=(plain + read + written if plain is not None else None),
                                output_tokens=count(usage.get("output_tokens")),
                                cached_input_tokens=read, cache_write_short_tokens=short,
                                cache_write_long_tokens=long,
                                cache_write_unspecified_tokens=written - short - long,
                                reasoning_tokens=None, cache_accounting_known=bool(usage), raw=usage)
        texts = [block.get("text", "") for block in raw.get("content", []) if block.get("type") == "text"]
        if any(block.get("type") == "tool_use" for block in raw.get("content", [])):
            raise ProviderError("invalid_response", "Unexpected tool invocation")
        return self._make_response(request, raw, "\n".join(texts), raw.get("model"), raw.get("stop_reason"),
                                   normalized, body, latency, mode)

    def submit_batch(self, requests: Sequence[GenerationRequest], label: str = "heerbench") -> BatchHandle:
        custom_ids, hashes = self._batch_maps(requests)
        inverse = {value: key for key, value in custom_ids.items()}
        if len(requests) > 100000:
            raise ProviderError("configuration", "Anthropic batch exceeds 100,000 requests")
        raw = self._json("POST", "/messages/batches", submission=True,
                         json={"requests": [{"custom_id": inverse[r.request_id], "params": self.request_body(r)}
                                            for r in requests]})
        if not raw.get("id"):
            raise ProviderError("invalid_response", "Batch response omitted ID", submission_uncertain=True)
        return BatchHandle(self.kind, raw["id"], [r.request_id for r in requests], custom_ids, hashes,
                           raw_submission=raw)

    def poll_batch(self, handle: BatchHandle) -> BatchStatus:
        raw = self._json("GET", "/messages/batches/" + handle.batch_id)
        state = raw.get("processing_status")
        mapping = {"in_progress": "running", "canceling": "running", "ended": "completed"}
        if state not in mapping:
            raise ProviderError("invalid_response", f"Unknown Anthropic batch state {state!r}")
        return BatchStatus(handle.batch_id, mapping[state], raw)

    def collect_batch(self, handle: BatchHandle,
                      requests_by_id: Mapping[str, GenerationRequest]) -> list[dict[str, Any]]:
        self._batch_ready(handle)
        text = self._request("GET", "/messages/batches/" + handle.batch_id + "/results").text
        records = []
        for row in jsonl_records(text):
            result = row.get("result") or {}
            if result.get("type") == "succeeded":
                records.append((row.get("custom_id"), result.get("message"), None))
            else:
                records.append((row.get("custom_id"), None, result))
        return self._collect_records(handle, requests_by_id, records)
