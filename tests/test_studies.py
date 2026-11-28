"""Incremental selections preserve scientific identity and never dispatch extras."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import httpx
import pytest
import yaml

from heerbench.aggregation import read_run_snapshot
from heerbench.cli import main
from heerbench.config import load_config
from heerbench.dataset import Dataset
from heerbench.providers.base import GenerationResponse, TokenUsage
from heerbench.runner import Runner, load_run
from heerbench.studies import execute_study, prepare_study
from heerbench.util import digest, read_jsonl


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def data():
    return Dataset(ROOT / "data/heerbench_v05")


def config(tmp_path, data, models=("A",), repetitions=1, mode="demo", **changes):
    chain = min(data.chains, key=lambda c: len(data.chain_nodes(c["chain_id"])))
    raw = {
        "mode": mode, "dataset": str(data.root), "chain_ids": [chain["chain_id"]],
        "prompt_file": str(ROOT / "prompts/generator_system.txt"),
        "judge_prompt_file": str(ROOT / "prompts/judge_system.txt"),
        "repetitions": repetitions, "chains_per_scenario": 1,
        "generation": {"max_output_tokens": 1024},
        "execution": {"transport": "sync"},
        "models": [{"id": mid, "model": "mock-" + mid, "kind": "openai_compatible",
                    "base_url": "https://test.invalid", "max_context_tokens": 1_000_000,
                    "api_key_file": str(tmp_path / "secrets" / (mid + ".key")),
                    "pricing": {"standard": {"input": "0", "output": "0"}}}
                   for mid in models],
        **changes,
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return load_config(path)


def frozen_bytes(path):
    return {name: (Path(path) / name).read_bytes()
            for name in ("config.frozen.json", "manifest.json", "plan.jsonl")}


def no_network(*args, **kwargs):
    raise AssertionError("This test must not make a network request or read a credential")


def test_incremental_models_total_repetitions_and_exact_parent_handoffs(tmp_path, data, monkeypatch):
    monkeypatch.setattr(httpx.Client, "request", no_network)
    monkeypatch.setattr("heerbench.credentials.read_api_key", no_network)
    monkeypatch.setattr("heerbench.runner.validate_credentials", no_network)
    runs = tmp_path / "runs"
    a1 = prepare_study(config(tmp_path, data), runs, "experiment")
    assert a1["new_model_repetitions"] == {"A": [1]}
    original = Path(a1["new_segment"])
    before = frozen_bytes(original)
    executed = execute_study(a1)
    assert sum(x["new_attempts"] for x in executed["segments_status"]) == 7
    first_results = read_run_snapshot(original)["generation"]

    again = prepare_study(config(tmp_path, data), runs, "experiment")
    assert again["new_segment"] is None and again["new_generation_requests"] == 0
    assert again["new_model_repetitions"] == {}
    assert sum(x["new_attempts"] for x in execute_study(again)["segments_status"]) == 0
    assert read_run_snapshot(original)["generation"] == first_results

    b1 = prepare_study(config(tmp_path, data, models=("B",)), runs, "experiment")
    assert b1["new_model_repetitions"] == {"B": [1]}
    assert b1["segments"] == [b1["new_segment"]]
    execute_study(b1)

    a3 = prepare_study(config(tmp_path, data, repetitions=3), runs, "experiment")
    assert a3["new_model_repetitions"] == {"A": [2, 3]}
    assert a3["requested_repetitions"] == {"A": [1, 2, 3]}
    added = read_jsonl(Path(a3["new_segment"]) / "plan.jsonl")
    assert {(r["model_id"], r["repetition"]) for r in added} == {("A", 2), ("A", 3)}
    added_by_id = {r["request_id"]: r for r in added}
    for row in added:
        if row["condition"] == "E3":
            for pid, qid in zip(row["parent_request_ids"], row["parent_question_ids"]):
                parent = added_by_id[pid]
                assert parent["question_id"] == qid
                assert parent["model_id"] == row["model_id"]
                assert parent["repetition"] == row["repetition"]
                assert parent["condition"] == ("E1" if parent["abf_stage"] == "A" else "E3")
    assert sum(x["new_attempts"] for x in execute_study(a3)["segments_status"]) == 14
    assert frozen_bytes(original) == before
    assert read_run_snapshot(original)["generation"] == first_results
    assert len(list((runs / "experiment/segments").glob("*/manifest.json"))) == 3

    # Requesting fewer totals later neither creates repetitions nor dispatches
    # those outside the requested range.
    a1_again = prepare_study(config(tmp_path, data), runs, "experiment")
    assert a1_again["segments"] == [str(original)]
    assert sum(x["new_attempts"] for x in execute_study(a1_again)["segments_status"]) == 0


def test_model_and_repetition_subset_in_existing_multimodel_segment(tmp_path, data, monkeypatch):
    monkeypatch.setattr(httpx.Client, "request", no_network)
    combined = prepare_study(config(tmp_path, data, models=("A", "B"), repetitions=3),
                             tmp_path / "runs", "experiment")
    path = Path(combined["new_segment"])
    before = frozen_bytes(path)
    selected = prepare_study(config(tmp_path, data, repetitions=1), tmp_path / "runs", "experiment")
    assert selected["new_segment"] is None
    assert selected["segments"] == [str(path)]
    result = execute_study(selected, max_requests=2)
    assert sum(x["new_attempts"] for x in result["segments_status"]) == 2
    result = execute_study(selected)
    assert sum(x["new_attempts"] for x in result["segments_status"]) == 5
    snap = read_run_snapshot(path)
    for row in snap["generation"]:
        expected = row["plan"]["model_id"] == "A" and row["plan"]["repetition"] == 1
        assert row["state"] == ("succeeded" if expected else "pending")
        assert row["attempts"] == int(expected)
    assert frozen_bytes(path) == before


@pytest.mark.parametrize("change", ["prompt", "model", "temperature", "task_scope"])
def test_incompatible_additions_leave_the_accepted_study_unchanged(tmp_path, data, change):
    cfg = config(tmp_path, data)
    first = prepare_study(cfg, tmp_path / "runs", "experiment")
    index = Path(first["study_path"]) / "study.json"
    before_index, before_segment = index.read_bytes(), frozen_bytes(first["new_segment"])
    changed = copy.deepcopy(cfg)
    if change == "prompt":
        changed["generator_system"] = "A changed scientific prompt"
    elif change == "model":
        changed["models"][0]["model"] = "different-checkpoint"
    elif change == "temperature":
        changed["models"][0]["temperature"] = 0.8
    else:
        changed["chain_ids"] = [c["chain_id"] for c in data.chains
                                if c["chain_id"] not in cfg["chain_ids"]][:1]
    with pytest.raises(ValueError, match="different|changed|Incompatible"):
        prepare_study(changed, tmp_path / "runs", "experiment")
    assert index.read_bytes() == before_index
    assert frozen_bytes(first["new_segment"]) == before_segment
    assert len(list((Path(first["study_path"]) / "segments").glob("*"))) == 1


def test_credentials_are_not_frozen_as_values_or_read_during_plan_only(tmp_path, data, monkeypatch, capsys):
    cfg = config(tmp_path, data, mode="pilot")
    secret_path = Path(cfg["models"][0]["api_key_file"])
    secret_path.parent.mkdir()
    sentinel = "LOCAL_ONLY_TEST_SECRET_8d594cb2"
    secret_path.write_text(sentinel)
    monkeypatch.setattr(httpx.Client, "request", no_network)
    monkeypatch.setattr("heerbench.credentials.read_api_key", no_network)
    monkeypatch.setattr("heerbench.runner.validate_credentials", no_network)
    main(["run", "--config", str(tmp_path / "config.yaml"), "--study", "pilot",
          "--runs-root", str(tmp_path / "runs"), "--plan-only"])
    result = json.loads(capsys.readouterr().out)
    assert result["plan_only"] is True
    assert result["new_model_repetitions"] == {"A": [1]}
    segment = Path(result["new_segment"])
    assert not (segment / "journal.sqlite").exists()
    frozen = json.loads((segment / "config.frozen.json").read_text())
    assert frozen["models"][0]["api_key_file"] == str(secret_path)
    for name in ("config.frozen.json", "manifest.json", "plan.jsonl"):
        assert sentinel not in (segment / name).read_text()


def test_selected_dispatch_checks_only_selected_credentials(tmp_path, data, monkeypatch):
    # Exercise non-demo admission and orchestration with an in-process provider.
    # Model B deliberately has no key. Selecting A must neither require B's
    # credentials nor instantiate its provider nor dispatch its requests.
    cfg = config(tmp_path, data, models=("A", "B"), mode="pilot")
    key = Path(cfg["models"][0]["api_key_file"])
    key.parent.mkdir()
    key.write_text("TEMPORARY_AUTH_TOKEN_FOR_A_ONLY")
    combined = prepare_study(cfg, tmp_path / "runs", "experiment")
    selected = prepare_study(config(tmp_path, data, mode="pilot"), tmp_path / "runs", "experiment")
    calls = []
    providers = []

    class RecordingProvider:
        def __init__(self, model):
            providers.append(model["id"])
            assert model["id"] == "A"

        def generate(self, request):
            calls.append(request)
            return GenerationResponse(request.request_id, "MOCK:" + request.request_id,
                                      request.model, request.model, "in_process_mock", "stop",
                                      TokenUsage(input_tokens=10, output_tokens=5), {"mock": True},
                                      0.0, digest(request.to_dict()))

        def close(self):
            pass

    monkeypatch.setattr(httpx.Client, "request", no_network)
    monkeypatch.setattr("heerbench.studies.Runner",
                        lambda path: Runner(path, provider_factory=RecordingProvider))
    result = execute_study(selected)
    assert result["dispatched"] is True
    assert providers == ["A"]
    assert len(calls) == 7
    assert {call.model for call in calls} == {"mock-A"}
    rows = read_run_snapshot(combined["new_segment"])["generation"]
    assert all(row["state"] == "pending" and row["attempts"] == 0
               for row in rows if row["plan"]["model_id"] == "B")


def test_explicit_relative_credential_root_uses_config_directory_not_cwd(tmp_path, data, monkeypatch):
    config(tmp_path, data)
    path = tmp_path / "config.yaml"
    raw = yaml.safe_load(path.read_text())
    raw["credential_root"] = "credential-home"
    raw["models"][0]["api_key_file"] = "secrets/A.key"
    path.write_text(yaml.safe_dump(raw))
    other = tmp_path / "unrelated-cwd"
    other.mkdir()
    monkeypatch.chdir(other)
    cfg = load_config(path)
    assert cfg["models"][0]["api_key_file"] == str(tmp_path / "credential-home/secrets/A.key")


def test_updated_key_locations_do_not_mutate_existing_frozen_segments(tmp_path, data):
    cfg = config(tmp_path, data)
    first = prepare_study(cfg, tmp_path / "runs", "experiment")
    before = frozen_bytes(first["new_segment"])
    cfg["models"][0]["api_key_file"] = str(tmp_path / "different-key-location.key")
    selected = prepare_study(cfg, tmp_path / "runs", "experiment")
    assert selected["new_segment"] is None
    assert frozen_bytes(first["new_segment"]) == before
    _, frozen, _, _ = load_run(first["new_segment"])
    assert frozen["models"][0]["api_key_file"] != cfg["models"][0]["api_key_file"]


@pytest.mark.parametrize("field", ["client_secret", "x-api-key", "x-goog-api-key"])
@pytest.mark.parametrize("role", ["generation", "judge"])
def test_catalog_cannot_freeze_nested_plaintext_credentials(tmp_path, data, field, role):
    config(tmp_path, data)
    path = tmp_path / "config.yaml"
    raw = yaml.safe_load(path.read_text())
    model = raw["models"][0]
    judge = {**copy.deepcopy(model), "model": "separate-judge"}
    if role == "generation":
        model["parameters"] = {"headers": {field: "NEVER_PERSIST_THIS_VALUE"}}
    else:
        judge["parameters"] = {"headers": {field: "NEVER_PERSIST_THIS_VALUE"}}
    catalog = {"models": {"A": model}, "judges": {"J": judge},
               "default_judges": {"primary": "J", "secondary": "J"}}
    (tmp_path / "catalog.yaml").write_text(yaml.safe_dump(catalog))
    raw["models"] = ["A"]
    raw["model_catalog"] = "catalog.yaml"
    raw["judge"] = {"primary": "J", "secondary": "J"}
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="credential") as failure:
        load_config(path)
    assert "NEVER_PERSIST_THIS_VALUE" not in str(failure.value)
