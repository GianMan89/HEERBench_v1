from __future__ import annotations

import copy
import os
from pathlib import Path
import re

import yaml

from .util import project_root
from .credentials import reject_inline_credentials


def load_catalog(path: str | Path | None = None) -> dict:
    path = Path(path or project_root() / "configs/provider_catalog.yaml").resolve()
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("models"), dict):
        raise ValueError("Model catalogue must contain a models mapping")
    reject_inline_credentials(value)
    return value


def _normalise(name: str) -> str:
    return re.sub(r"[-_\s]+", "-", name.strip().casefold())


def resolve_model_ids(catalog: dict, names: list[str] | None = None, group: str | None = None) -> list[str]:
    if names and group:
        raise ValueError("Choose explicit models or a model group, not both")
    if group:
        groups = catalog.get("groups", {})
        if group not in groups:
            raise ValueError("Unknown model group " + group + ". Available: " + ", ".join(sorted(groups)))
        names = groups[group]
    if not names:
        raise ValueError("Choose at least one model")
    known = {**catalog["models"], **catalog.get("judges", {})}
    aliases = {}
    for key, model in known.items():
        for label in [key, model.get("model"), model.get("display_name"), model.get("name"), *model.get("aliases", [])]:
            if not label:
                continue
            normal = _normalise(label)
            if normal in aliases and aliases[normal] != key:
                raise ValueError("Ambiguous model alias in catalogue: " + label)
            aliases[normal] = key
    selected = []
    for name in names:
        key = aliases.get(_normalise(name))
        if key is None:
            raise ValueError("Unknown model " + name + ". Use heerbench models to list exact choices.")
        if key not in selected:
            selected.append(key)
    return selected


def _reject_embedded_secrets(model: dict):
    forbidden = {"api_key", "token", "authorization", "bearer_token", "access_token", "secret", "password"}
    def visit(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if str(key).lower() in forbidden and item:
                    raise ValueError("Plaintext credentials are prohibited in configuration. Use api_key_file.")
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)
    visit(model)


def load_config(path: str | Path, *, models: list[str] | None = None,
                model_group: str | None = None, pilot: bool | None = None,
                repetitions: int | None = None) -> dict:
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError("Configuration must be a mapping")
    reject_inline_credentials(cfg)
    base = path.parent
    catalogue_path = (base / cfg["model_catalog"]).resolve() if cfg.get("model_catalog") else None
    credential_root = ((base / Path(cfg["credential_root"]).expanduser()).resolve() if cfg.get("credential_root")
                       else (catalogue_path.parent.parent if catalogue_path else base))
    if catalogue_path:
        catalog = load_catalog(catalogue_path)
        all_models = {**catalog["models"], **catalog.get("judges", {})}
        if models or model_group:
            selected = resolve_model_ids(catalog, models, model_group)
        elif cfg.get("model_group"):
            selected = resolve_model_ids(catalog, group=cfg["model_group"])
        else:
            selected = cfg.get("models", [])
        cfg["models"] = [{"id": x, **copy.deepcopy(all_models[x])} if isinstance(x, str) else copy.deepcopy(x)
                         for x in selected]
        judge_settings = cfg.setdefault("judge", {})
        judge_settings.setdefault("primary", catalog.get("default_judges", {}).get("primary", "gpt61_terra"))
        judge_settings.setdefault("secondary", catalog.get("default_judges", {}).get("secondary", "claude_sonnet55"))
        judge_settings.setdefault("secondary_fraction", 0.2)
        judge_settings.setdefault("secondary_sampling_unit", "model_repetition")
        ids = resolve_model_ids(catalog, [judge_settings["primary"], judge_settings["secondary"]])
        judge_settings["primary"] = resolve_model_ids(catalog, [judge_settings["primary"]])[0]
        judge_settings["secondary"] = resolve_model_ids(catalog, [judge_settings["secondary"]])[0]
        cfg["judge_models"] = [{"id": key, **copy.deepcopy(all_models[key])} for key in ids]
        cfg["model_catalog"] = str(catalogue_path)
    elif models or model_group:
        raise ValueError("CLI model selection requires a model catalogue")
    if not cfg.get("models") or len({m["id"] for m in cfg["models"]}) != len(cfg["models"]):
        raise ValueError("Choose nonempty, uniquely identified models")
    cfg.setdefault("schema_version", 2)
    cfg.setdefault("protocol_version", "heerbench-abf-v1")
    cfg.setdefault("mode", "pilot")
    if pilot is True:
        cfg["mode"] = "pilot"
    elif pilot is False:
        cfg["mode"] = "full"
    cfg.setdefault("seed", 20261127)
    if repetitions is not None:
        cfg["repetitions"] = repetitions
    elif pilot is True:
        cfg["repetitions"] = 1
    cfg.setdefault("repetitions", 1 if cfg["mode"] != "full" else 3)
    cfg.setdefault("conditions", ["E1", "E2", "E3"])
    if set(cfg["conditions"]) != {"E1", "E2", "E3"} or len(cfg["conditions"]) != 3:
        raise ValueError("This protocol requires E1, E2 and E3 exactly once")
    if type(cfg["repetitions"]) is not int or cfg["repetitions"] < 1:
        raise ValueError("repetitions must be a positive integer")
    if cfg["mode"] not in {"full", "pilot", "demo"}:
        raise ValueError("mode must be full, pilot or demo")
    cfg.setdefault("chains_per_scenario", 2)
    cfg.setdefault("generation", {})
    cfg["generation"].setdefault("max_output_tokens", 4096)
    cfg["generation"].setdefault("temperature", None)
    cfg.setdefault("execution", {})
    cfg["execution"].setdefault("transport", "sync")
    cfg["execution"].setdefault("max_attempts", 3)
    cfg["execution"].setdefault("batch_max_requests", 100)
    cfg["execution"].setdefault("batch_max_bytes", 15_000_000)
    cfg["execution"].setdefault("max_cost_usd", None)
    if cfg["execution"]["transport"] not in {"sync", "batch"}:
        raise ValueError("transport must be sync or batch")
    cfg.setdefault("judge", {"primary": "gpt61_terra", "secondary": "claude_sonnet55", "secondary_fraction": 0.2})
    cfg["dataset"] = str((base / cfg.get("dataset", "../data/heerbench_v05")).resolve())
    cfg["config_source"] = str(path)
    cfg["prompt_file"] = str((base / cfg.get("prompt_file", "../prompts/generator_system.txt")).resolve())
    cfg["judge_prompt_file"] = str((base / cfg.get("judge_prompt_file", "../prompts/judge_system.txt")).resolve())
    for m in cfg["models"] + cfg.get("judge_models", []):
        _reject_embedded_secrets(m)
        m.setdefault("model", m["id"])
        if m.get("api_key_file"):
            m["api_key_file"] = str((credential_root / m["api_key_file"]).resolve())
        # Legacy environment names are descriptive provenance only; there is no runtime fallback.
        m.pop("api_key_env", None)
        m["credential_root"] = str(credential_root)
    for m in cfg["models"]:
        if "max_output_tokens" in m:
            m["provider_max_output_tokens"] = m.pop("max_output_tokens")
        m["max_output_tokens"] = cfg["generation"]["max_output_tokens"]
        m.setdefault("temperature", cfg["generation"]["temperature"])
    reject_inline_credentials(cfg)
    return cfg


def resolve_endpoint(model: dict) -> dict:
    """Resolve endpoint locations only. Credentials are read from files by the adapter."""
    result = copy.deepcopy(model)
    if result.get("base_url_env"):
        name = result["base_url_env"]
        value = os.environ.get(name)
        if not value:
            raise ValueError(f"Set {name} to the provisioned endpoint URL for {result['id']}")
        result["base_url"] = value
    return result
