"""File-only authentication and credential exclusion from observable artifacts."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

import httpx
import pytest

from heerbench.credentials import (CredentialError, credential_path, read_api_key,
    redact_sensitive, reject_inline_credentials, validate_credentials)
from heerbench.providers import GenerationRequest, OpenAIResponsesProvider, ProviderError


def make_key(tmp_path, value="test-file-only-key-91724365"):
    p = tmp_path / "secrets" / "provider.key"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(value, encoding="utf-8")
    return p


def test_relative_key_location_is_independent_of_working_directory(tmp_path, monkeypatch):
    path = make_key(tmp_path)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    model = {"api_key_file": "secrets/provider.key", "credential_root": str(tmp_path)}
    assert credential_path(model) == path
    assert read_api_key(model) == "test-file-only-key-91724365"
    metadata = validate_credentials(model)
    assert metadata == {"ready": True, "mode": "file", "api_key_file": str(path)}
    assert "test-file-only-key-91724365" not in json.dumps(metadata)


def test_environment_is_never_a_fallback(tmp_path, monkeypatch):
    secret = "test-env-key-that-must-not-be-read-12345"
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    with pytest.raises(CredentialError, match="environment variables are unsupported") as error:
        read_api_key({"api_key_env": "OPENAI_API_KEY"})
    assert secret not in str(error.value)
    with pytest.raises(CredentialError, match="missing, unreadable"):
        read_api_key({"api_key_file": str(tmp_path / "absent.key"), "api_key_env": "OPENAI_API_KEY"})


@pytest.mark.parametrize("content", ["", "  \n", "\ufefftest-valid-key-123", "one\ntwo",
    '"test-key-quoted-123"', "{'api_key':'test-json-key'}", "API_KEY=test-assignment-key", "test key with spaces"])
def test_invalid_file_formats_fail_without_exposing_content(tmp_path, content):
    path = make_key(tmp_path, content)
    with pytest.raises(CredentialError) as error:
        read_api_key({"api_key_file": str(path)})
    assert str(path) in str(error.value)
    if len(content.strip()) > 4:
        assert content.strip() not in str(error.value)


def test_one_trailing_windows_line_ending_is_accepted(tmp_path):
    path = make_key(tmp_path, "test-windows-line-ending-key-456\r\n")
    assert read_api_key({"api_key_file": str(path)}) == "test-windows-line-ending-key-456"


def test_model_construction_and_payload_planning_do_not_read_keys(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Offline request construction must not read a credential file")
    monkeypatch.setattr(Path, "read_bytes", forbidden)
    provider = OpenAIResponsesProvider({"api_key_file": str(tmp_path / "does-not-exist.key")})
    body = provider.request_body(GenerationRequest("offline", "model", "system", "packet", "question"))
    assert body["model"] == "model"


def test_inline_credentials_are_rejected_in_nested_target_or_judge_config():
    for config in ({"models": [{"api_key": "sensitive-inline-a-123"}]},
                   {"judge_models": [{"headers": {"Authorization": "sensitive-inline-b-234"}}]}):
        with pytest.raises(CredentialError, match="Inline credential") as error:
            reject_inline_credentials(config)
        assert "sensitive-inline" not in str(error.value)


def test_only_explicit_unauthenticated_localhost_is_allowed():
    assert read_api_key({"base_url": "http://127.0.0.1:8000/v1", "allow_unauthenticated_localhost": True}) == ""
    with pytest.raises(CredentialError, match="localhost"):
        read_api_key({"base_url": "https://example.org/v1", "allow_unauthenticated_localhost": True})


def test_http_error_echoes_are_redacted_and_file_wins_over_environment(tmp_path, monkeypatch):
    key = "test-request-header-key-732462138"
    path = make_key(tmp_path, key)
    monkeypatch.setenv("OPENAI_API_KEY", "environment-value-must-not-be-used")
    def handler(request):
        assert request.headers["authorization"] == "Bearer " + key
        return httpx.Response(401, json={"error": {"message": "Rejected " + key,
            "headers": {"Authorization": "Bearer " + key}}, "access_token": "another-credential"})
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAIResponsesProvider({"api_key_file": str(path)}, client)
        with pytest.raises(ProviderError) as error:
            provider.generate(GenerationRequest("one", "model", "s", "p", "t"))
    dumped = json.dumps(error.value.to_dict())
    assert key not in dumped
    assert "another-credential" not in dumped
    assert "[REDACTED]" in dumped
    assert error.value.kind == "http"


def test_successful_raw_response_and_answer_cannot_echo_key(tmp_path):
    key = "test-provider-success-echo-key-771235"
    path = make_key(tmp_path, key)
    raw = {"id": "r1", "model": "model", "status": "completed", "metadata": {"detail": key},
           "output": [{"type": "message", "content": [{"type": "output_text", "text": "Echo " + key}]}],
           "usage": {"input_tokens": 3, "output_tokens": 2, "input_tokens_details": {"cached_tokens": 0}}}
    with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=raw))) as client:
        response = OpenAIResponsesProvider({"api_key_file": str(path)}, client).generate(
            GenerationRequest("one", "model", "s", "p", "t"))
    assert key not in json.dumps(response.to_dict())
    assert response.text == "Echo [REDACTED]"
    assert response.usage.input_tokens == 3


def test_git_ignores_actual_key_files_but_keeps_setup_guide(tmp_path):
    if shutil.which("git") is None:
        pytest.skip("Git is not installed")
    root = Path(__file__).resolve().parents[1]
    shutil.copyfile(root / ".gitignore", tmp_path / ".gitignore")
    (tmp_path / "secrets").mkdir()
    for file in ("openai.key", "anthropic.key", "google.key", "openrouter.key", "custom.key", "README.md"):
        (tmp_path / "secrets" / file).touch()
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True, capture_output=True)
    for name in ("secrets/openai.key", "secrets/anthropic.key", "secrets/google.key",
                 "secrets/openrouter.key", "secrets/custom.key", ".env", "another.key"):
        check = subprocess.run(["git", "-C", str(tmp_path), "check-ignore", "-q", name])
        assert check.returncode == 0, name
    check = subprocess.run(["git", "-C", str(tmp_path), "check-ignore", "-q", "secrets/README.md"])
    assert check.returncode == 1


def test_shipped_key_locations_are_documented_without_reading_live_files():
    root = Path(__file__).resolve().parents[1]
    guide = (root / "secrets" / "README.md").read_text()
    for name in ("openai.key", "anthropic.key", "google.key", "openrouter.key"):
        assert name in guide
