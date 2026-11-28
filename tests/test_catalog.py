"""Catalogue contract tests use archived public metadata; no network or paid calls."""
from decimal import Decimal
import json
from pathlib import Path

import pytest

from heerbench.config import load_catalog, resolve_model_ids
from heerbench.costs import select_rates


ROOT = Path(__file__).resolve().parents[1]
CATALOG = load_catalog(ROOT / "configs/provider_catalog.yaml")
MODELS = CATALOG["models"]


def test_agreed_pools_and_separate_judges():
    defaults = resolve_model_ids(CATALOG, group="default")
    extended = resolve_model_ids(CATALOG, group="extended")
    assert len(defaults) == len(set(defaults)) == 11
    assert len(extended) == len(set(extended)) == 16
    assert len(set(defaults + extended)) == 26
    assert "grok46" not in defaults
    assert "grok46" not in CATALOG["groups"]["pilot"]
    assert "grok46" in extended
    assert CATALOG["groups"]["defaults"] == defaults
    assert CATALOG["groups"]["frontier"] == extended
    assert CATALOG["default_judges"] == {
        "primary": "gpt61_terra", "secondary": "claude_sonnet55"}
    assert set(CATALOG["default_judges"].values()).isdisjoint(defaults + extended)


@pytest.mark.parametrize("display,expected", [
    ("GPT-6 Luna", "gpt6_luna"),
    ("Claude Haiku 5.5", "claude_haiku55"),
    ("Gemini 3.5 Flash-Lite", "gemini35_flash_lite"),
    ("Muse Spark 1.3", "muse_spark13"),
    ("Grok 4.6", "grok46"),
    ("Claude Opus 5", "claude_opus5"),
    ("GLM-5.3", "glm53"),
    ("GPT-6 Astra", "gpt6_astra"),
    ("Kimi K3", "kimi_k3"),
    ("GPT-5.6 Sol", "gpt56_sol"),
    ("GPT-5.6 Luna", "gpt56_luna"),
    ("Nemotron 3 Ultra", "nemotron3_ultra"),
    ("MiniMax M3", "minimax_m3"),
    ("Claude Fable 5.1", "claude_fable51"),
    ("Qwen3.8-Max", "qwen38_max"),
    ("DeepSeek-V4-Pro", "deepseek_v4_pro"),
    ("Gemini 3.1 Pro", "gemini31_pro"),
    ("Mistral Medium 3.5", "mistral_medium35"),
    ("Gemma 4 31", "gemma4_31b"),
])
def test_named_models_resolve_without_successor_substitution(display, expected):
    assert resolve_model_ids(CATALOG, [display]) == [expected]


def test_default_and_extended_credentials_are_file_only():
    keys = set(CATALOG["groups"]["default"] + CATALOG["groups"]["extended"])
    files = set()
    for model in {**MODELS, **CATALOG["judges"]}.values():
        assert "api_key_env" not in model
        assert model["api_key_file"].startswith("secrets/")
        assert model["api_key_file"].endswith(".key")
        assert not Path(model["api_key_file"]).is_absolute()
    for key in keys:
        files.add(MODELS[key]["api_key_file"])
    assert files == {"secrets/openai.key", "secrets/anthropic.key",
                     "secrets/google.key", "secrets/openrouter.key"}


@pytest.mark.parametrize("key", [key for key, m in MODELS.items() if m.get("endpoint_snapshot")])
def test_router_contract_matches_single_archived_endpoint(key):
    model = MODELS[key]
    snapshot = json.loads((ROOT / model["endpoint_snapshot"]).read_text())
    endpoint = next(e for e in snapshot["data"]["endpoints"]
                    if e["tag"] == model["endpoint_tag"])
    route = model["parameters"]["provider"]
    assert route["only"] == [endpoint["tag"]]
    assert route["allow_fallbacks"] is False
    assert route["require_parameters"] is True
    assert model["model"] == endpoint["model_id"]
    assert model["accepted_response_models"] == [endpoint["model_id"]]
    assert model["accepted_backend_names"] == [endpoint["provider_name"]]
    assert model["precision"] == endpoint["quantization"]
    assert model["max_context_tokens"] == endpoint["context_length"]
    assert model["max_output_tokens"] == endpoint["max_completion_tokens"]
    rates = select_rates(model["pricing"], "standard", 1)
    assert Decimal(rates["input"]) == Decimal(endpoint["pricing"]["prompt"]) * 1_000_000
    assert Decimal(rates["output"]) == Decimal(endpoint["pricing"]["completion"]) * 1_000_000
    assert "batch" not in model["pricing"]
    assert model["native_batch_supported"] is False
    assert endpoint.get("service_tier") is None


def test_haiku_long_input_tier_applies_to_all_channels():
    model = MODELS["claude_haiku55"]
    for mode in ["standard", "batch"]:
        below = select_rates(model["pricing"], mode, 100000)
        above = select_rates(model["pricing"], mode, 100001)
        for channel in ["input", "output", "cache_read", "cache_write_short", "cache_write_long"]:
            assert Decimal(above[channel]) == Decimal(below[channel]) * 5
    assert select_rates(model["pricing"], "standard", 100001)["input"] == "0.50"


def test_native_and_router_tiers_preserve_their_own_boundaries():
    openai = MODELS["gpt6_luna"]["pricing"]
    assert Decimal(select_rates(openai, "standard", 272000)["input"]) == Decimal(".1")
    assert Decimal(select_rates(openai, "standard", 272001)["input"]) == Decimal(".2")
    grok = MODELS["grok46"]["pricing"]
    assert Decimal(select_rates(grok, "standard", 199999)["input"]) == Decimal("2")
    assert Decimal(select_rates(grok, "standard", 200000)["input"]) == Decimal("4")
    assert MODELS["gpt56_sol"]["pricing_valid_until"] == "2026-11-30"


def test_model_specific_batch_cache_rates_are_not_interchanged():
    flash = MODELS["gemini35_flash_lite"]["pricing"]
    pro = MODELS["gemini31_pro"]["pricing"]
    assert Decimal(select_rates(flash, "batch", 1000)["cache_read"]) == Decimal(".015")
    assert Decimal(select_rates(pro, "batch", 1000)["cache_read"]) == Decimal(".20")


def test_endpoint_and_legacy_provenance_is_explicit():
    assert MODELS["qwen38_max"]["model"] == "qwen/qwen3.8-max-0902"
    assert MODELS["gemma4_31b"]["endpoint_tag"] == "coreweave/fp4"
    assert MODELS["nemotron35_lightning"]["max_context_tokens"] == 262144
    assert (ROOT / "docs/provider_sources/catalog_v01_original.yaml").is_file()
    assert MODELS["qwen3_235b_a22b"]["enabled"] is False

