"""Local, file-only API credentials with explicit checks and output redaction.

Importing this module, inspecting a model or planning an experiment never reads a
key. Only ``validate_credentials`` and ``read_api_key`` open the configured file.
Credential contents are never returned by validation or stored in configuration.
"""
from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Mapping
from urllib.parse import urlsplit

from .util import project_root


class CredentialError(ValueError):
    """A safe credential diagnostic that never includes the credential value."""


_loaded_secrets: set[str] = set()
_secret_field_names = {
    "api_key", "apikey", "access_token", "refresh_token", "client_secret",
    "secret_key", "password", "authorization", "x_api_key", "x_goog_api_key",
}


def _field_name(name: Any) -> str:
    return str(name).lower().replace("-", "_")


def reject_inline_credentials(value: Any, path: str = "configuration") -> None:
    """Reject plaintext credential fields before a configuration is frozen.

    ``api_key_file`` and legacy variable *names* are not credential values. A
    legacy variable name alone still cannot authenticate an API request.
    """
    if isinstance(value, Mapping):
        for name, item in value.items():
            if _field_name(name) in _secret_field_names and item not in (None, ""):
                raise CredentialError(f"Inline credential field is forbidden at {path}.{name}; configure api_key_file")
            reject_inline_credentials(item, f"{path}.{name}")
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            reject_inline_credentials(item, f"{path}[{index}]")


def credential_path(model: Mapping[str, Any], repository_root: str | Path | None = None) -> Path | None:
    """Resolve a file location without reading it or probing environment keys."""
    value = model.get("api_key_file")
    if value is None or value == "":
        if model.get("allow_unauthenticated_localhost", False):
            url = urlsplit(str(model.get("base_url", "")))
            if url.hostname not in {"localhost", "127.0.0.1", "::1"}:
                raise CredentialError("Unauthenticated access is restricted to an explicitly configured localhost endpoint")
            return None
        if model.get("api_key_env"):
            raise CredentialError("Credential environment variables are unsupported; configure api_key_file")
        raise CredentialError("No api_key_file is configured")
    if not isinstance(value, (str, Path)):
        raise CredentialError("api_key_file must be a filesystem path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        base = repository_root or model.get("credential_root")
        path = (Path(base) if base else project_root()) / path
    return path.resolve()


def read_api_key(model: Mapping[str, Any], repository_root: str | Path | None = None) -> str:
    """Read one plain-text key; never fall back to environment variables."""
    reject_inline_credentials(model)
    path = credential_path(model, repository_root)
    if path is None:
        return ""
    try:
        if path.stat().st_size > 16384:
            raise CredentialError(f"API key file is unexpectedly large: {path}")
        raw = path.read_bytes().decode("utf-8")
    except CredentialError:
        raise
    except (OSError, UnicodeError):
        raise CredentialError(f"API key file is missing, unreadable or not UTF-8: {path}") from None
    # Register even invalid content so a subsequently constructed diagnostic cannot
    # accidentally echo a value that has been seen by this process.
    value = raw.strip()
    if value:
        _loaded_secrets.add(value)
    if raw.startswith("\ufeff"):
        raise CredentialError(f"Save the API key file as UTF-8 without a byte-order mark: {path}")
    if not value:
        raise CredentialError(f"API key file is empty: {path}")
    if any(c.isspace() for c in value):
        raise CredentialError(f"API key file must contain one key on one line: {path}")
    if value[0] in {'"', "'", "{", "["} or re.match(r"^(?:export\s+)?[A-Z][A-Z0-9_]*=", value):
        raise CredentialError(f"API key file must contain the key only, without quotes, JSON or an assignment: {path}")
    return value


def validate_credentials(model: Mapping[str, Any], repository_root: str | Path | None = None) -> dict[str, Any]:
    """Check a file explicitly and return only safe, non-secret metadata."""
    read_api_key(model, repository_root)
    path = credential_path(model, repository_root)
    return {"ready": True, "mode": "file" if path else "unauthenticated_localhost",
            "api_key_file": str(path) if path else None}


def redact_sensitive(value: Any) -> Any:
    """Mask known key values and credential-named fields in provider diagnostics.

    Every loaded key is masked recursively, including when a provider echoes it
    inside an otherwise ordinary message or raw response. Only the in-memory
    authentication header retains the real value.
    """
    if isinstance(value, str):
        for key in sorted(_loaded_secrets, key=len, reverse=True):
            value = value.replace(key, "[REDACTED]")
        return value
    if isinstance(value, bytes):
        for key in sorted(_loaded_secrets, key=len, reverse=True):
            value = value.replace(key.encode("utf-8"), b"[REDACTED]")
        return value
    if isinstance(value, Mapping):
        return {redact_sensitive(str(k)): "[REDACTED]" if _field_name(k) in _secret_field_names
                else redact_sensitive(v) for k, v in value.items()}
    if isinstance(value, list):
        return [redact_sensitive(x) for x in value]
    if isinstance(value, tuple):
        return tuple(redact_sensitive(x) for x in value)
    return value
