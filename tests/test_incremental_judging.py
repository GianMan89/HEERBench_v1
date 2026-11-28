"""Incremental scoring and sampling contracts, using only temporary fixtures.

Synthetic successful responses below are UNIT TEST data. No provider is built,
no key is read, no judge call is made, and no human approval is manufactured.
"""
from __future__ import annotations

from collections import Counter
import copy
import json
import math
from pathlib import Path

import httpx
import pytest
import yaml

from heerbench.config import load_config
from heerbench.cost_estimation import forecast_judges
from heerbench.dataset import Dataset
from heerbench.evaluation import generation_records, prepare_judges
from heerbench.judge_validation import judge_configuration_digest, select_secondary_sample
from heerbench.planning import create_plan
from heerbench.runner import Runner


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def forbid_network_and_credentials(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Judging-preparation tests must not contact an API or read a key")
    monkeypatch.setattr(httpx.Client, "request", forbidden)
    monkeypatch.setattr("heerbench.credentials.read_api_key", forbidden)
    monkeypatch.setattr("heerbench.runner.validate_credentials", forbidden)


@pytest.fixture
def runner(tmp_path):
    data = Dataset(ROOT / "data/heerbench_v05")
    chain = min(data.chains, key=lambda c: len(data.chain_nodes(c["chain_id"])))
    raw = {
        "mode": "pilot", "dataset": str(data.root), "chain_ids": [chain["chain_id"]],
        "prompt_file": str(ROOT / "prompts/generator_system.txt"),
        "judge_prompt_file": str(ROOT / "prompts/judge_system.txt"),
        "repetitions": 7, "model_repetition_ids": {"A": [2, 7], "B": [2, 7]},
        "models": [{"id": mid, "model": "fixture-" + mid, "kind": "openai_compatible",
                    "max_context_tokens": 1_000_000} for mid in ("A", "B")],
        "judge": {"primary": "J1", "secondary": "J2", "secondary_fraction": .2,
                  "secondary_sampling_unit": "model_repetition"},
        "judge_models": [{"id": mid, "model": "fixture-" + mid,
                          "kind": "openai_compatible", "max_output_tokens": 16384,
                          "max_context_tokens": 1_000_000,
                          "api_key_file": str(tmp_path / (mid + ".key")),
                          "pricing": {"standard": {"input": "1", "output": "2"}}}
                         for mid in ("J1", "J2")],
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    cfg = load_config(path)
    dest = tmp_path / "run"
    create_plan(cfg, data, dest)
    result = Runner(dest, provider_factory=lambda _: pytest.fail("No provider may be constructed"))
    try:
        yield result
    finally:
        result.close()


def complete_fixture(runner, model=None, rep=None):
    for row in runner.store.rows("generation"):
        if model is not None and row["plan"]["model_id"] != model:
            continue
        if rep is not None and row["plan"]["repetition"] != rep:
            continue
        runner.store.update(row["id"], "succeeded", result={
            "text": "SYNTHETIC UNIT TEST ANSWER " + row["id"],
            "provider": "UNIT_TEST_FIXTURE_ONLY", "fixture_not_research_data": True,
        })


def secondary_counts(runner):
    return Counter((row["plan"]["model_id"], row["plan"]["repetition"])
                   for row in runner.store.rows("judge") if row["plan"]["judge_role"] == "secondary")


def test_incremental_judging_preserves_prior_judge_ids_and_sample(runner):
    complete_fixture(runner, "A", 2)
    with pytest.raises(ValueError, match="Complete"):
        prepare_judges(runner, model_ids={"A"})  # A repetition 7 is still pending.
    assert runner.store.rows("judge") == []
    first = prepare_judges(runner, model_ids={"A"}, repetitions={"A": [2]})
    assert first["new_judge_requests"] == 9
    original = {r["id"]: copy.deepcopy(r) for r in runner.store.rows("judge")}
    original_sample = json.loads((runner.root / "judge_sample.json").read_text())["secondary_answer_ids"]
    assert len(original_sample) == 2
    assert secondary_counts(runner) == {("A", 2): 2}
    assert all(r["state"] == "pending" for r in runner.store.rows("generation")
               if r["plan"]["model_id"] == "B")
    assert prepare_judges(runner, model_ids={"A"}, repetitions={"A": [2]})["new_judge_requests"] == 0

    complete_fixture(runner, "B", 2)
    later = prepare_judges(runner, model_ids={"B"}, repetitions={"B": [2]})
    assert later["new_judge_requests"] == 9
    assert {rid: runner.store.get(rid) for rid in original} == original
    current_sample = json.loads((runner.root / "judge_sample.json").read_text())["secondary_answer_ids"]
    assert set(original_sample) < set(current_sample)
    assert secondary_counts(runner) == {("A", 2): 2, ("B", 2): 2}

    complete_fixture(runner)
    final = prepare_judges(runner, model_ids={"A", "B"}, repetitions={"A": [7], "B": [7]})
    assert final["new_judge_requests"] == 18
    assert secondary_counts(runner) == {("A", 2): 2, ("B", 2): 2, ("A", 7): 2, ("B", 7): 2}
    assert {rid: runner.store.get(rid) for rid in original} == original
    assert prepare_judges(runner)["new_judge_requests"] == 0
    assert len(runner.store.rows("judge")) == 36


def test_secondary_sampling_ignores_scores_and_answer_quality(runner):
    complete_fixture(runner)
    records = [r for r in generation_records(runner)
               if r["model_id"] == "A" and r["repetition"] == 2]
    original = select_secondary_sample(records, runner.dataset.by_id, fraction=.2, seed=57)
    manipulated = copy.deepcopy(records)
    for i, row in enumerate(manipulated):
        row["score"] = {"accuracy": 1000 - i, "correctness": int(i % 2 == 0)}
        row["judge_verdict"] = "satisfied" if i % 2 else "not_satisfied"
        row["text"] = "" if i % 2 else "An incorrect fixture response."
    selected = select_secondary_sample(list(reversed(manipulated)), runner.dataset.by_id,
                                       fraction=.2, seed=57)
    assert [r["task_id"] for r in selected] == [r["task_id"] for r in original]
    assert len(selected) == math.ceil(.2 * len(records))


def test_judge_forecast_rounding_matches_all_successful_per_model_repetition_samples(runner):
    planned = Counter((r["model_id"], r["repetition"]) for r in runner.plan)
    assert set(planned.values()) == {7}
    forecast = forecast_judges(runner.cfg, runner.dataset, runner.plan,
                               expected_candidate_tokens=100, expected_judge_output_tokens=200)
    expected_secondary = sum(math.ceil(.2 * count) for count in planned.values())
    assert expected_secondary == 8
    assert expected_secondary != math.ceil(.2 * len(runner.plan))
    complete_fixture(runner)
    prepare_judges(runner)
    actual = Counter(r["plan"]["judge_role"] for r in runner.store.rows("judge"))
    assert forecast["primary"]["calls"] == actual["primary"] == 28
    assert forecast["secondary"]["calls"] == actual["secondary"] == expected_secondary
    assert secondary_counts(runner) == {key: math.ceil(.2 * n) for key, n in planned.items()}


BASE_JUDGE = {
    "id": "fixture-judge", "model": "fixture-pinned-version", "kind": "openai_compatible",
    "base_url": "https://fixture.invalid", "max_output_tokens": 8192,
    "parameters": {"reasoning_effort": "medium"},
    "api_key_file": "/old/location.key", "credential_root": "/old",
    "pricing": {"standard": {"input": "1", "output": "2"}},
    "pricing_valid_until": "2026-12-31",
}


@pytest.mark.parametrize("field,value", [
    ("api_key_file", "/new/location.key"), ("credential_root", "/new"),
    ("pricing", {"standard": {"input": "100", "output": "200"}}),
    ("pricing_valid_until", "2027-12-31"),
])
def test_approval_digest_survives_credential_and_price_metadata_changes(field, value):
    original = judge_configuration_digest("fixed prompt", BASE_JUDGE, "fixed corpus")
    changed = {**copy.deepcopy(BASE_JUDGE), field: value}
    assert judge_configuration_digest("fixed prompt", changed, "fixed corpus") == original


@pytest.mark.parametrize("change", ["model", "effort", "prompt"])
def test_approval_digest_binds_actual_judge_model_effort_and_prompt(change):
    changed = copy.deepcopy(BASE_JUDGE)
    prompt = "fixed prompt"
    if change == "model":
        changed["model"] = "a-different-version"
    elif change == "effort":
        changed["parameters"]["reasoning_effort"] = "high"
    else:
        prompt = "a changed prompt"
    assert judge_configuration_digest(prompt, changed, "fixed corpus") != judge_configuration_digest(
        "fixed prompt", BASE_JUDGE, "fixed corpus")
