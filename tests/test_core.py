"""Offline integration checks for frozen inputs, DAG handoffs and durable resume."""
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from heerbench.config import load_config
from heerbench.dataset import Dataset
from heerbench.planning import create_plan
from heerbench.prompts import build_generation_request
from heerbench.runner import Runner, load_run
from heerbench.store import Store
from heerbench.util import read_jsonl, write_json, write_jsonl


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def dataset():
    return Dataset(ROOT / "data/heerbench_v05")


def config(tmp_path, *, mode="demo", repetitions=1, n_models=1, chain_ids=None):
    raw = {
        "mode": mode,
        "dataset": str(ROOT / "data/heerbench_v05"),
        "prompt_file": str(ROOT / "prompts/generator_system.txt"),
        "judge_prompt_file": str(ROOT / "prompts/judge_system.txt"),
        "repetitions": repetitions,
        "chains_per_scenario": 1,
        "generation": {"max_output_tokens": 1024, "temperature": 0.2},
        "models": [{"id": f"mock-{i}", "model": f"mock-{i}",
                    "kind": "openai_compatible", "max_context_tokens": 1_000_000,
                    "pricing": {"input_per_million": 0, "output_per_million": 0}}
                   for i in range(n_models)],
    }
    if chain_ids is not None:
        raw["chain_ids"] = chain_ids
    path = tmp_path / "test_config.yaml"
    path.write_text(yaml.safe_dump(raw))
    return load_config(path)


def make_run(tmp_path, dataset, **kwargs):
    cfg = config(tmp_path, **kwargs)
    dest = tmp_path / "run"
    manifest = create_plan(cfg, dataset, dest)
    return dest, manifest


def test_accepted_dataset_integrity_and_dependency_context_counts(dataset):
    result = dataset.validate()
    assert result["questions"] == 488
    assert result["chains"] == 150
    assert result["scenarios"] == 10
    assert result["stages"] == {"A": 175, "B": 147, "F": 166}
    assert result["dependency_sets"] == 313
    assert len({e["child_id"] for e in dataset.dependencies}) == 313
    assert all(e["logic"] == "ALL" for e in dataset.dependencies)
    assert sum(len(q["chain_ids"]) > 1 for q in dataset.questions) == 64


def test_full_plan_counts_shared_nodes_once_and_separates_repetitions(tmp_path, dataset):
    dest, manifest = make_run(tmp_path, dataset, mode="full", repetitions=3, n_models=8)
    rows = read_jsonl(dest / "plan.jsonl")
    assert manifest["n_requests"] == len(rows) == 26_736
    assert manifest["n_questions"] == 488
    assert manifest["n_chains"] == 150
    assert Counter(r["condition"] for r in rows) == {"E1": 11_712, "E2": 7_512, "E3": 7_512}
    assert len({r["request_id"] for r in rows}) == len(rows)
    by_id = {r["request_id"]: r for r in rows}
    for row in rows:
        if row["abf_stage"] == "A":
            assert row["condition"] == "E1"
            assert row["parent_request_ids"] == []
        if row["condition"] == "E3":
            assert len(row["parent_request_ids"]) == len(row["parent_question_ids"])
            for pid, qid in zip(row["parent_request_ids"], row["parent_question_ids"]):
                parent = by_id[pid]
                assert parent["question_id"] == qid
                assert parent["model_id"] == row["model_id"]
                assert parent["repetition"] == row["repetition"]
                assert parent["condition"] == ("E1" if parent["abf_stage"] == "A" else "E3")
    a_id = next(q["question_id"] for q in dataset.questions if q["abf_stage"] == "A")
    requests = [r for r in rows if r["question_id"] == a_id and r["model_id"] == "mock-0"]
    assert len(requests) == 3
    assert len({r["request_id"] for r in requests}) == 3


def test_pilot_selection_is_reproducible_complete_and_dependency_closed(dataset):
    selected = dataset.select_chains("pilot", per_scenario=2, seed=17)
    assert selected == dataset.select_chains("pilot", per_scenario=2, seed=17)
    assert len(selected) == 20
    assert Counter(dataset.by_chain[c]["case_id"] for c in selected) == {f"C{i}": 2 for i in range(1, 11)}
    qids = set().union(*(dataset.chain_nodes(c) for c in selected))
    assert all(set(dataset.parents.get(q, ())) <= qids for q in qids)


def test_prompts_keep_packet_fixed_and_use_only_defined_parent_text(tmp_path, dataset):
    # Choose a multi-parent task so omitting a jointly required premise is detected.
    edge = next(e for e in dataset.dependencies if len(e["parent_ids"]) > 1)
    qid = edge["child_id"]
    chain = dataset.by_id[qid]["chain_ids"][0]
    dest, _ = make_run(tmp_path, dataset, chain_ids=[chain])
    _, cfg, ds, rows = load_run(dest)
    selected = {r["condition"]: r for r in rows if r["question_id"] == qid}
    sentinels = {p: f"GENERATED ANSWER WITH ERROR {i}" for i, p in enumerate(ds.parents[qid])}
    requests = {c: build_generation_request(r, cfg, ds, sentinels) for c, r in selected.items()}
    assert len({r.prefix for r in requests.values()}) == 1
    assert len({r.system for r in requests.values()}) == 1
    assert len({r.max_output_tokens for r in requests.values()}) == 1
    assert ds.packets[ds.by_id[qid]["case_id"]] in requests["E3"].prefix
    payloads = {c: json.loads(r.task) for c, r in requests.items()}
    for p in payloads.values():
        assert set(p) == {"current_question", "question_id", "preceding_analytical_outputs"}
        assert p["current_question"] == ds.by_id[qid]["question"]
        assert "criteria" not in p and "expected_answer" not in p
    assert payloads["E1"]["preceding_analytical_outputs"] == []
    assert {x["question_id"] for x in payloads["E3"]["preceding_analytical_outputs"]} == set(ds.parents[qid])
    for p in payloads["E3"]["preceding_analytical_outputs"]:
        assert p["answer"] == sentinels[p["question_id"]]
        assert set(p) == {"question_id", "question", "answer"}
    for p in payloads["E2"]["preceding_analytical_outputs"]:
        assert p["answer"] == ds.by_id[p["question_id"]]["expected_answer"]
        assert set(p) == {"question_id", "question", "answer"}


def test_frozen_config_and_plan_edits_are_rejected(tmp_path, dataset):
    dest, _ = make_run(tmp_path, dataset, chain_ids=[dataset.chains[0]["chain_id"]])
    _, cfg, _, rows = load_run(dest)
    rows[0]["condition"] = "E9"
    write_jsonl(dest / "plan.jsonl", rows)
    with pytest.raises(ValueError, match="request plan was edited"):
        load_run(dest)
    cfg["repetitions"] = 9
    write_json(dest / "config.frozen.json", cfg)
    with pytest.raises(ValueError, match="configuration was edited"):
        load_run(dest)


def test_store_locks_preserves_plan_and_checks_artifact_hash(tmp_path):
    store = Store(tmp_path / "journal")
    another = Store(tmp_path / "journal")
    try:
        with store.lock():
            with pytest.raises(RuntimeError, match="Another orchestrator"):
                with another.lock():
                    pass
        with another.lock():
            pass
        row = {"request_id": "request-1", "model_id": "mock"}
        store.add(row)
        store.add(row)
        assert len(store.rows()) == 1
        with pytest.raises(ValueError, match="different frozen plan"):
            store.add({**row, "model_id": "different"})
        raw = {"system": "instruction", "prefix": "packet", "task": "question", "model": "mock"}
        packed = store.pack_request(raw)
        assert store.unpack_request(packed) == raw
        key = packed["prefix_blob"]
        path = store.root / "blobs" / key[:2] / (key + ".txt")
        path.write_text("edited packet")
        with pytest.raises(ValueError, match="Artifact hash mismatch"):
            store.unpack_request(packed)
    finally:
        another.close()
        store.close()


def test_offline_runner_resume_preserves_outputs_and_actual_parent_links(tmp_path, dataset):
    dest, manifest = make_run(tmp_path, dataset, repetitions=2, chain_ids=[dataset.chains[0]["chain_id"]])

    def no_network(_):
        raise AssertionError("An offline mock must not build an API provider")

    first = Runner(dest, provider_factory=no_network)
    try:
        first.step(max_requests=3)
        successful = {r["id"]: r["result"] for r in first.store.rows() if r["state"] == "succeeded"}
        assert len(successful) == 3
    finally:
        first.close()
    resumed = Runner(dest, provider_factory=no_network)
    try:
        status = resumed.step()
        assert status["generation"] == {"succeeded": manifest["n_requests"]}
        rows = resumed.store.rows("generation")
        assert all(r["attempts"] == 1 for r in rows)
        assert all(resumed.store.get(rid)["result"] == value for rid, value in successful.items())
        for row in rows:
            request = resumed.store.unpack_request(row["request"])
            task = json.loads(request["task"])
            if row["plan"]["condition"] == "E3":
                for pid, parent in zip(row["plan"]["parent_request_ids"], task["preceding_analytical_outputs"]):
                    assert parent["answer"] == resumed.store.get(pid)["result"]["text"]
            assert row["result"]["provider"] == "mock"
            assert row["result"]["raw_response"]["is_mock"] is True
        assert resumed.step() == status
        assert all(r["attempts"] == 1 for r in resumed.store.rows("generation"))
    finally:
        resumed.close()


def test_failed_upstream_blocks_descendants_without_silent_reference_substitution(tmp_path, dataset):
    chain = dataset.chains[0]["chain_id"]
    dest, _ = make_run(tmp_path, dataset, chain_ids=[chain])
    runner = Runner(dest)
    try:
        a = next(r for r in runner.plan if r["abf_stage"] == "A")
        runner.store.update(a["request_id"], "failed", error={"kind": "test_transport_failure"})
        runner.step()
        assert runner.store.get(a["request_id"])["result"] is None
        blocked = [r for r in runner.store.rows() if r["state"] == "blocked"]
        assert blocked
        assert all(r["plan"]["condition"] == "E3" and r["result"] is None and r["attempts"] == 0 for r in blocked)
        assert all(r["state"] == "succeeded" for r in runner.store.rows() if r["plan"]["condition"] == "E2")
        with pytest.raises(ValueError, match="reason is required"):
            runner.recover([a["request_id"]], "")
        runner.recover([a["request_id"]], "Synthetic transport failure resolved in integration test")
        runner.recover([r["id"] for r in blocked], "Re-enable synthetic descendants after explicit parent recovery")
        runner.step()
        assert runner.status()["generation"] == {"succeeded": len(runner.plan)}
    finally:
        runner.close()


def cli(*args):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    # A demo must not depend on live credentials or reachable API endpoints.
    for name in tuple(env):
        if name.endswith("_API_KEY"):
            env.pop(name)
    result = subprocess.run([sys.executable, "-m", "heerbench", *map(str, args)],
                            cwd=ROOT, env=env, text=True, capture_output=True, timeout=45)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_cli_offline_demo_writes_blinded_judgements_and_graph_reports(tmp_path):
    dest = tmp_path / "demo"
    result = cli("demo", "--out", dest)
    assert result["is_mock"] is True
    plan = read_jsonl(dest / "plan.jsonl")
    generations = read_jsonl(dest / "generation.jsonl")
    judgements = read_jsonl(dest / "judgements.jsonl")
    assert len(plan) == len(generations) == len(judgements)
    assert all(r["generation_status"] == "succeeded" for r in generations)
    assert all(r["judge_status"] == "unscorable" for r in judgements)
    assert {r["generation_request_id"] for r in judgements} == {r["request_id"] for r in plan}
    assert all(r["judge_role"] == "primary" for r in judgements)
    report = json.loads((dest / "reports/analysis.json").read_text())
    assert report["metadata"]["is_mock"] is True
    assert report["metadata"]["selected_chain_ids"] == json.loads((dest / "config.frozen.json").read_text())["selected_chain_ids"]
    assert all(r["scored"] == 0 for r in report["coverage"])
    assert all(r["criterion_score_observed"] is None for r in report["stage_scores"])
    assert "not measurements" in (dest / "reports/report.md").read_text()
    assert cli("status", dest) == result["status"]
    assert cli("run", dest) == result["status"]
    assert cli("judge", dest) == result["status"]
    assert all(r["attempts"] == 1 for r in read_jsonl(dest / "generation.jsonl"))
    assert all(r["attempts"] == 1 for r in read_jsonl(dest / "judgements.jsonl"))


def test_cli_pilot_plan_and_preflight_make_no_network_requests(tmp_path):
    dest = tmp_path / "pilot"
    manifest = cli("plan", "--config", ROOT / "configs/pilot.yaml", "--out", dest)
    assert manifest["n_chains"] == 20
    assert manifest["is_mock"] is False
    cfg = json.loads((dest / "config.frozen.json").read_text())
    assert cfg["repetitions"] == 1
    assert len(cfg["models"]) == 2
    pre = cli("preflight", dest)
    assert pre["network_requests"] == 0
    assert pre["credentials_checked"] is False
    # Missing provisioned endpoints can make admission false without triggering calls.
    assert len(pre["models"]) == 2
    assert not (dest / "generation.jsonl").exists()
