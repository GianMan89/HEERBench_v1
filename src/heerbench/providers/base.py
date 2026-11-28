"""Provider contracts. No SDK retry, response cache, routing or model fallback."""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

import httpx

from ..credentials import CredentialError, read_api_key, redact_sensitive, reject_inline_credentials


def payload_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode()).hexdigest()


def count(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ProviderError("invalid_usage", "Provider returned invalid token accounting")
    return value


@dataclass(frozen=True)
class GenerationRequest:
    request_id: str
    model: str
    system: str
    prefix: str
    task: str
    max_output_tokens: int = 2048
    temperature: float | None = None
    top_p: float | None = None
    seed: int | None = None
    parameters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TokenUsage:
    """Input includes cache reads/writes; output includes billed reasoning tokens."""
    input_tokens: int | None = None
    output_tokens: int | None = None
    cached_input_tokens: int = 0
    cache_write_short_tokens: int = 0
    cache_write_long_tokens: int = 0
    cache_write_unspecified_tokens: int = 0
    reasoning_tokens: int | None = None
    cache_accounting_known: bool = True
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def uncached_input_tokens(self) -> int | None:
        if self.input_tokens is None:
            return None
        n = self.input_tokens - self.cached_input_tokens - self.cache_write_short_tokens \
            - self.cache_write_long_tokens - self.cache_write_unspecified_tokens
        if n < 0:
            raise ProviderError("invalid_usage", "Cache categories exceed total input")
        return n

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GenerationResponse:
    request_id: str
    text: str
    model_requested: str
    model_returned: str | None
    provider: str
    finish_reason: str | None
    usage: TokenUsage
    raw_response: dict[str, Any]
    latency_seconds: float | None
    request_payload_hash: str
    execution_mode: str = "standard"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BatchHandle:
    provider: str
    batch_id: str
    request_ids: list[str]
    custom_ids: dict[str, str]
    payload_hashes: dict[str, str]
    input_file_id: str | None = None
    raw_submission: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BatchHandle":
        return cls(**dict(value))


@dataclass
class BatchStatus:
    batch_id: str
    status: str
    raw: dict[str, Any]

    @property
    def terminal(self) -> bool:
        return self.status in {"completed", "failed", "expired", "cancelled"}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ProviderError(RuntimeError):
    def __init__(self, kind: str, message: str, *, retryable: bool = False,
                 http_status: int | None = None, retry_after: float | None = None,
                 submission_uncertain: bool = False, raw: Any = None):
        super().__init__(redact_sensitive(message))
        self.kind, self.retryable = kind, retryable
        self.http_status, self.retry_after = http_status, retry_after
        self.submission_uncertain, self.raw = submission_uncertain, redact_sensitive(raw)

    def to_dict(self) -> dict[str, Any]:
        return redact_sensitive({"kind": self.kind, "message": str(self), "retryable": self.retryable,
                "http_status": self.http_status, "retry_after": self.retry_after,
                "submission_uncertain": self.submission_uncertain, "raw": self.raw})


class Provider:
    kind = "base"
    default_base_url = ""
    default_key_env = ""
    supports_batch = False

    def __init__(self, config: Mapping[str, Any], client: httpx.Client | None = None):
        reject_inline_credentials(config)
        self.config = dict(config)
        self.base_url = str(config.get("base_url", self.default_base_url)).rstrip("/")
        # Defer networking setup until an actual operation. Offline validation and
        # request planning must not require credentials or a configured proxy.
        self._client = client
        self._owns_client = client is None

    def close(self) -> None:
        if self._owns_client and self._client is not None:
            self._client.close()

    def _key(self) -> str:
        try:
            return read_api_key({**self.config, "base_url": self.base_url})
        except CredentialError as exc:
            raise ProviderError("credentials", str(exc)) from None

    def _headers(self) -> dict[str, str]:
        key = self._key()
        return {"Authorization": f"Bearer {key}"} if key else {}

    def _request(self, method: str, path: str, *, submission: bool = False,
                 **kwargs: Any) -> httpx.Response:
        """Exactly one HTTP attempt. Caller journals and decides whether to retry."""
        headers = {**self._headers(), **kwargs.pop("headers", {})}
        if self._client is None:
            self._client = httpx.Client(timeout=float(self.config.get("timeout_seconds", 180)),
                                        follow_redirects=False)
        try:
            response = self._client.request(method, self.base_url + path, headers=headers, **kwargs)
        except httpx.TransportError as exc:
            raise ProviderError("transport", type(exc).__name__, retryable=not submission,
                                submission_uncertain=submission) from None
        if not response.is_success:
            try:
                detail = response.json()
            except ValueError:
                detail = {"body": response.text[:2000]}
            retry_after = response.headers.get("retry-after")
            try:
                delay = max(0.0, float(retry_after)) if retry_after else None
            except ValueError:
                delay = None
            ambiguous = submission and response.status_code >= 500
            raise ProviderError("http", f"Provider HTTP {response.status_code}",
                                retryable=(response.status_code in {408, 429, 500, 502, 503, 504})
                                and not ambiguous, http_status=response.status_code,
                                retry_after=delay, submission_uncertain=ambiguous, raw=detail)
        # Batch output files use _request(...).text directly. Redact the complete
        # body here as well as JSON values so those paths cannot leak an echoed key.
        clean = redact_sensitive(response.content)
        if clean != response.content:
            clean_headers = {k: v for k, v in response.headers.items()
                             if k.lower() not in {"content-length", "content-encoding"}}
            response = httpx.Response(response.status_code, headers=clean_headers,
                                      content=clean, request=response.request)
        return response

    def _json(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        response = self._request(method, path, **kwargs)
        try:
            value = response.json()
            if not isinstance(value, dict):
                raise ValueError("object expected")
            return redact_sensitive(value)
        except ValueError as exc:
            raise ProviderError("invalid_response", "Expected a JSON response object",
                                submission_uncertain=bool(kwargs.get("submission"))) from exc

    def _parameters(self, request: GenerationRequest, forbidden: set[str]) -> dict[str, Any]:
        parameters = {**self.config.get("parameters", {}), **request.parameters}
        collision = forbidden.intersection(parameters)
        if collision:
            raise ProviderError("configuration", "Reserved request fields cannot be overridden: "
                                + ", ".join(sorted(collision)))
        configured = self.config.get("model", self.config.get("model_id"))
        if configured and request.model != configured:
            raise ProviderError("model_identity", "Request model differs from configured model")
        if not request.request_id or not request.model or request.max_output_tokens < 1:
            raise ProviderError("configuration", "Invalid request identity or output budget")
        return parameters

    def _identity(self, request: GenerationRequest, returned: Any) -> str | None:
        if not returned or not isinstance(returned, str):
            if self.config.get("strict_model_identity", True):
                raise ProviderError("model_identity", "Provider response omitted model identity")
            return None
        accepted = set(self.config.get("accepted_response_models", [request.model]))
        if returned not in accepted:
            raise ProviderError("model_identity", f"Unexpected response model {returned!r}; "
                                "freeze an explicitly accepted version before running")
        return returned

    def _make_response(self, request: GenerationRequest, raw: dict[str, Any], text: str,
                       returned: Any, finish: Any, usage: TokenUsage, body: Any,
                       latency: float | None, mode: str) -> GenerationResponse:
        actual_model = self._identity(request, returned)
        usage.uncached_input_tokens  # validate normalized accounting
        return GenerationResponse(request.request_id, redact_sensitive(text), request.model, actual_model,
                                  self.kind, finish, usage, redact_sensitive(raw), latency, payload_hash(body), mode)

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        body = self.request_body(request)
        started = time.monotonic()
        # A lost generation response may still have consumed paid inference.
        # Treat its completion state as unknown rather than silently retrying.
        raw = self._json("POST", self.generation_path(request), json=body, submission=True)
        try:
            return self.parse_response(request, raw, body, time.monotonic() - started, "standard")
        except ProviderError as exc:
            if exc.raw is None:
                exc.raw = raw
            raise

    def generation_path(self, request: GenerationRequest) -> str:
        raise NotImplementedError

    def request_body(self, request: GenerationRequest) -> dict[str, Any]:
        raise NotImplementedError

    def parse_response(self, request: GenerationRequest, raw: dict[str, Any], body: Any,
                       latency: float | None, mode: str) -> GenerationResponse:
        raise NotImplementedError

    def submit_batch(self, requests: Sequence[GenerationRequest], label: str = "heerbench") -> BatchHandle:
        raise ProviderError("unsupported", f"Native batch is not implemented for {self.kind}")

    def poll_batch(self, handle: BatchHandle) -> BatchStatus:
        raise ProviderError("unsupported", f"Native batch is not implemented for {self.kind}")

    def collect_batch(self, handle: BatchHandle,
                      requests_by_id: Mapping[str, GenerationRequest]) -> list[dict[str, Any]]:
        raise ProviderError("unsupported", f"Native batch is not implemented for {self.kind}")

    def estimate_batch_bytes(self, requests: Sequence[GenerationRequest]) -> int:
        return sum(len(json.dumps(self.request_body(r), ensure_ascii=False).encode()) + 1024
                   for r in requests) + 1024

    def _batch_maps(self, requests: Sequence[GenerationRequest]) -> tuple[dict[str, str], dict[str, str]]:
        if not requests:
            raise ProviderError("configuration", "A batch must contain at least one request")
        ids = [r.request_id for r in requests]
        if len(set(ids)) != len(ids):
            raise ProviderError("configuration", "Duplicate request IDs in batch")
        return ({"r_" + hashlib.sha256(i.encode()).hexdigest()[:60]: i for i in ids},
                {r.request_id: payload_hash(self.request_body(r)) for r in requests})

    def _batch_ready(self, handle: BatchHandle) -> BatchStatus:
        if handle.provider != self.kind:
            raise ProviderError("configuration", "Batch provider does not match adapter")
        status = self.poll_batch(handle)
        if not status.terminal:
            raise ProviderError("batch_pending", f"Batch {handle.batch_id} is {status.status}")
        return status

    def _collect_records(self, handle: BatchHandle, requests_by_id: Mapping[str, GenerationRequest],
                         records: Sequence[tuple[str | None, dict[str, Any] | None, Any]]) -> list[dict[str, Any]]:
        outputs: dict[str, dict[str, Any]] = {}
        for custom_id, raw, error in records:
            raw, error = redact_sensitive(raw), redact_sensitive(error)
            if custom_id not in handle.custom_ids:
                raise ProviderError("batch_identity", "Unknown or absent custom ID in batch output")
            request_id = handle.custom_ids[custom_id]
            if request_id in outputs:
                raise ProviderError("batch_identity", "Duplicate custom ID in batch output")
            request = requests_by_id[request_id]
            body = self.request_body(request)
            if payload_hash(body) != handle.payload_hashes[request_id]:
                raise ProviderError("batch_identity", "Request payload changed after batch submission")
            if raw is not None and error is None:
                try:
                    result = self.parse_response(request, raw, body, None, "batch")
                    outputs[request_id] = {"request_id": request_id, "status": "succeeded",
                                           "response": result, "error": None}
                except ProviderError as exc:
                    if exc.raw is None:
                        exc.raw = raw
                    outputs[request_id] = {"request_id": request_id, "status": "failed",
                                           "response": None, "error": exc.to_dict()}
            else:
                outputs[request_id] = {"request_id": request_id, "status": "failed",
                                       "response": None, "error": error or {"kind": "empty_batch_item"}}
        for request_id in handle.request_ids:
            if request_id not in outputs:
                outputs[request_id] = {"request_id": request_id, "status": "failed", "response": None,
                                       "error": {"kind": "missing_batch_result",
                                                 "message": "No result returned; do not silently regenerate"}}
        return [outputs[r] for r in handle.request_ids]


def jsonl_records(text: str) -> list[dict[str, Any]]:
    try:
        values = [json.loads(line) for line in text.splitlines() if line.strip()]
        if not all(isinstance(v, dict) for v in values):
            raise ValueError("object expected")
        return redact_sensitive(values)
    except ValueError as exc:
        raise ProviderError("invalid_response", "Invalid JSONL result file") from exc
