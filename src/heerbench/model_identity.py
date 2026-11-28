"""Pure deployment identity, excluding credential and price metadata."""
from __future__ import annotations
from copy import deepcopy
from typing import Any, Mapping
from .util import digest

_MODEL_DESCRIPTIVE_FIELDS = {
    "id", "aliases", "display_name", "name", "description", "sources", "source_urls",
    "availability", "enabled", "verified_on", "verified_at", "pricing", "currency",
    "api_key", "api_key_file", "api_key_env", "key_file", "token", "credential_root",
    "repository_root", "repo_root", "provision_note", "context_note", "identity_note",
    "reasoning_note", "notes", "availability_reason", "disabled_reason",
    "architecture", "total_parameters_billions", "active_parameters_billions",
    "native_context_tokens", "cache", "timeout_seconds", "tokenizer_file",
    "pricing_valid_until", "pricing_verified_on", "endpoint_snapshot",
    "supported_parameters", "native_batch_supported", "legacy_profile",
}


def model_spec(model: Mapping[str, Any], resolved_base_url: str | None = None) -> dict:
    """Keep deployment and inference settings, excluding labels, prices and secrets.

    Precision, checkpoint revision, actual API model identifier, provider kind,
    backend constraints, output budgets and inference parameters remain relevant.
    Unknown new configuration fields are retained rather than silently ignored.
    """
    result = {k: deepcopy(v) for k, v in model.items()
              if k not in _MODEL_DESCRIPTIVE_FIELDS and not k.endswith(("_note", "_notes"))}
    if resolved_base_url is not None:
        result["base_url"] = resolved_base_url
        result.pop("base_url_env", None)
    if isinstance(result.get("base_url"), str):
        result["base_url"] = result["base_url"].rstrip("/")
    for name in ("accepted_response_models", "accepted_backend_names"):
        if name in result:
            result[name] = sorted(result[name])
    return result


def model_signature(model: Mapping[str, Any], resolved_base_url: str | None = None) -> str:
    return digest(model_spec(model, resolved_base_url))

