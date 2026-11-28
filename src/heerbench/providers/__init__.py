"""HTTP adapters with explicit provider identity and resumable native batches."""
from typing import Any, Mapping

import httpx

from .anthropic import AnthropicProvider
from .base import (BatchHandle, BatchStatus, GenerationRequest, GenerationResponse,
                   Provider, ProviderError, TokenUsage)
from .gemini import GeminiProvider
from .openai import OpenAICompatibleProvider, OpenAIResponsesProvider


def build_provider(config: Mapping[str, Any], client: httpx.Client | None = None) -> Provider:
    kinds = {"openai_responses": OpenAIResponsesProvider,
             "openai_compatible": OpenAICompatibleProvider,
             "anthropic": AnthropicProvider, "gemini": GeminiProvider}
    kind = config.get("kind")
    if kind not in kinds:
        raise ProviderError("configuration", f"Unknown provider kind {kind!r}")
    return kinds[kind](config, client=client)


__all__ = ["GenerationRequest", "GenerationResponse", "TokenUsage", "BatchHandle", "BatchStatus",
           "Provider", "ProviderError", "build_provider", "OpenAIResponsesProvider",
           "OpenAICompatibleProvider", "AnthropicProvider", "GeminiProvider"]
