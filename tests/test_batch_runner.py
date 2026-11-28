"""Exercise real Runner dispatch through an HTTP-mocked provider contract.

The run is deliberately not marked demo, so the normal batch/cost/journal code
executes. HTTPX MockTransport handles every HTTP operation in process.
"""
from collections import Counter
import json
from pathlib import Path

import httpx
import pytest
import yaml

from heerbench.config import load_config
from heerbench.dataset import Dataset
from heerbench.planning import create_plan
from heerbench.providers.base import Provider, BatchHandle, BatchStatus, TokenUsage
from heerbench.runner import Runner
from heerbench.util import digest


ROOT = Path(__file__).resolve().parents[1]


def test_batch_budget_admission_does_not_leave_partial_reservations(tmp_path, data, monkeypatch):
    dest, backend, factory = run_fixture(tmp_path, data, monkeypatch)
    runner = Runner(dest, provider_factory=factory)
    try:
        chosen = runner.ready("generation")[:2]
        first = runner._reservation(chosen[0], runner.request(chosen[0]), "batch")[1]
        # In-memory cap adjustment is a test fixture; production config remains frozen.
        runner.cfg["execution"]["max_cost_usd"] = float(first["total_usd"])
        with pytest.raises(ValueError, match="max_cost_usd"):
            runner._submit(chosen, "generation")
        assert runner.store.db.execute("SELECT COUNT(*) FROM events WHERE event='cost_reserved'").fetchone()[0] == 0
        assert not backend.calls
        assert all(r["state"] == "pending" for r in chosen)
    finally:
        runner.close()


class FakeHTTPBackend:
    def __init__(self):
        self.calls = []
        self.batches = {}
        self.completed = set()
        self.omit_ids = set()
        self.ambiguous_submission = False

    @staticmethod
    def response(body):
        return {"model": body["model"], "text": "GENERATED:" + body["request_id"],
                "input_tokens": 10, "output_tokens": 5}

    def __call__(self, request):
        self.calls.append((request.method, request.url.path))
        if request.method == "POST" and request.url.path == "/responses":
            return httpx.Response(200, json=self.response(json.loads(request.content)))
        if request.method == "POST" and request.url.path == "/batches":
            batch_id = "batch-" + str(len(self.batches) + 1)
            self.batches[batch_id] = json.loads(request.content)["requests"]
            if self.ambiguous_submission:
                # Simulate remote acceptance followed by a lost response.
                self.ambiguous_submission = False
                raise httpx.ReadTimeout("Simulated response lost after acceptance", request=request)
            return httpx.Response(200, json={"id": batch_id})
        if request.method == "GET" and request.url.path.startswith("/batches/"):
            parts = request.url.path.strip("/").split("/")
            bid = parts[1]
            if len(parts) == 2:
                return httpx.Response(200, json={"id": bid, "status": "completed" if bid in self.completed else "in_progress"})
            if parts[2] == "output":
                items = [{"custom_id": r["custom_id"], "response": self.response(r["body"])}
                         for r in self.batches[bid] if r["body"]["request_id"] not in self.omit_ids]
                return httpx.Response(200, json={"items": items})
        raise AssertionError(f"Unexpected in-process mock HTTP call {request.method} {request.url}")


class FakeProvider(Provider):
    def __init__(self, config, backend):
        self.kind = config["kind"]
        self.mock_client = httpx.Client(transport=httpx.MockTransport(backend))
        super().__init__(config, client=self.mock_client)

    def close(self):
        self.mock_client.close()

    def _headers(self):
        return {}

    def request_body(self, request):
        return request.to_dict()

    def generation_path(self, request):
        return "/responses"

    def parse_response(self, request, raw, body, latency, mode):
        return self._make_response(request, raw, raw["text"], raw["model"], "stop",
            TokenUsage(input_tokens=raw["input_tokens"], output_tokens=raw["output_tokens"]),
            body, latency, mode)

    def submit_batch(self, requests, label="test"):
        custom, hashes = self._batch_maps(requests)
        by_id = {r.request_id: r for r in requests}
        body = {"requests": [{"custom_id": cid, "body": self.request_body(by_id[rid])}
                             for cid, rid in custom.items()]}
        raw = self._json("POST", "/batches", json=body, submission=True)
        return BatchHandle(self.kind, raw["id"], [r.request_id for r in requests], custom, hashes)

    def poll_batch(self, handle):
        raw = self._json("GET", "/batches/" + handle.batch_id)
        return BatchStatus(handle.batch_id, raw["status"], raw)

    def collect_batch(self, handle, requests_by_id):
        raw = self._json("GET", "/batches/" + handle.batch_id + "/output")
        return self._collect_records(handle, requests_by_id,
            [(item["custom_id"], item["response"], None) for item in raw["items"]])


@pytest.fixture(scope="module")
def data():
    return Dataset(ROOT / "data/heerbench_v05")


def run_fixture(tmp_path, data, monkeypatch, *, kind="openai_responses", cap=None):
    chain = min(data.chains, key=lambda c: len(data.chain_nodes(c["chain_id"])))
    assert len(data.chain_nodes(chain["chain_id"])) == 3
    raw = {
        "mode": "pilot", "dataset": str(data.root), "chain_ids": [chain["chain_id"]],
        "prompt_file": str(ROOT / "prompts/generator_system.txt"), "repetitions": 1,
        "judge_prompt_file": str(ROOT / "prompts/judge_system.txt"),
        "models": [{"id": "fake", "model": "fake", "kind": kind,
                    "base_url": "https://in-process-test.invalid", "max_context_tokens": 1_000_000,
                    "pricing": {"standard": {"input": "1", "output": "2"},
                                "batch": {"input": "0.5", "output": "1"}}}],
        "generation": {"max_output_tokens": 1024},
        "execution": {"transport": "batch", "batch_max_requests": 100,
                      "batch_max_bytes": 15_000_000, "max_cost_usd": cap},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(raw))
    dest = tmp_path / "run"
    create_plan(load_config(config_path), data, dest)
    monkeypatch.setattr("heerbench.runner.preflight", lambda *args, **kwargs: {"ready": True})
    backend = FakeHTTPBackend()
    factory = lambda cfg: FakeProvider(cfg, backend)
    return dest, backend, factory


def test_batch_waves_resume_and_collect_each_completed_batch_once(tmp_path, data, monkeypatch):
    dest, backend, factory = run_fixture(tmp_path, data, monkeypatch)
    runner = Runner(dest, provider_factory=factory)
    try:
        assert runner.cfg["is_mock"] is False
        first = runner.step()
        assert first["generation"] == {"submitted": 5, "pending": 2}
        assert len(backend.batches) == 1
        initial_ids = {x["body"]["request_id"] for x in backend.batches["batch-1"]}
        assert all(r["condition"] != "E3" for r in runner.plan if r["request_id"] in initial_ids)
        assert runner.step() == first  # Pending provider batch does not unlock children.
        assert len(backend.batches) == 1
    finally:
        runner.close()

    # A new orchestrator resumes the same durable run and remote batch state.
    resumed = Runner(dest, provider_factory=factory)
    try:
        backend.completed.add("batch-1")
        assert resumed.step()["generation"] == {"succeeded": 5, "submitted": 1, "pending": 1}
        second = backend.batches["batch-2"]
        assert len(second) == 1
        b_row = resumed.store.get(second[0]["body"]["request_id"])["plan"]
        assert (b_row["condition"], b_row["abf_stage"]) == ("E3", "B")
        preceding = json.loads(second[0]["body"]["task"])["preceding_analytical_outputs"]
        assert [x["answer"] for x in preceding] == ["GENERATED:" + p for p in b_row["parent_request_ids"]]

        backend.completed.add("batch-2")
        assert resumed.step()["generation"] == {"succeeded": 6, "submitted": 1}
        third = backend.batches["batch-3"]
        f_row = resumed.store.get(third[0]["body"]["request_id"])["plan"]
        assert (f_row["condition"], f_row["abf_stage"]) == ("E3", "F")
        preceding = json.loads(third[0]["body"]["task"])["preceding_analytical_outputs"]
        assert [x["answer"] for x in preceding] == ["GENERATED:" + p for p in f_row["parent_request_ids"]]

        backend.completed.add("batch-3")
        assert resumed.step()["generation"] == {"succeeded": 7}
        calls = list(backend.calls)
        assert resumed.step()["generation"] == {"succeeded": 7}
        assert backend.calls == calls
        assert Counter(path for method, path in backend.calls if path.endswith("/output")) == {
            "/batches/batch-1/output": 1, "/batches/batch-2/output": 1, "/batches/batch-3/output": 1}
        assert all(r["attempts"] == 1 and r["result"]["execution_mode"] == "batch" for r in resumed.store.rows())
        assert all(r["result"]["cost"]["execution_mode"] == "batch" for r in resumed.store.rows())
    finally:
        resumed.close()


def test_ambiguous_submission_is_journalled_and_never_automatically_resent(tmp_path, data, monkeypatch):
    dest, backend, factory = run_fixture(tmp_path, data, monkeypatch)
    backend.ambiguous_submission = True
    runner = Runner(dest, provider_factory=factory)
    try:
        state = runner.step()
        assert state["generation"] == {"uncertain": 5, "pending": 2}
        assert runner.store.batches("generation")[0]["state"] == "uncertain"
        assert len(backend.batches) == 1  # Remote acceptance happened despite lost response.
    finally:
        runner.close()
    resumed = Runner(dest, provider_factory=factory)
    try:
        calls = list(backend.calls)
        assert resumed.step() == state
        assert backend.calls == calls
        uncertain = [r for r in resumed.store.rows() if r["state"] == "uncertain"]
        assert all(r["attempts"] == 1 and r["error"]["submission_uncertain"] for r in uncertain)
    finally:
        resumed.close()


def test_missing_batch_parent_result_fails_it_and_blocks_e3_descendants(tmp_path, data, monkeypatch):
    dest, backend, factory = run_fixture(tmp_path, data, monkeypatch)
    runner = Runner(dest, provider_factory=factory)
    try:
        a = next(r for r in runner.plan if r["abf_stage"] == "A")
        backend.omit_ids.add(a["request_id"])
        runner.step()
        backend.completed.add("batch-1")
        status = runner.step()
        assert status["generation"] == {"failed": 1, "succeeded": 4, "blocked": 2}
        assert runner.store.get(a["request_id"])["error"]["kind"] == "missing_batch_result"
        assert all(r["plan"]["condition"] == "E3" for r in runner.store.rows() if r["state"] == "blocked")
        assert len(backend.batches) == 1
        assert runner.step() == status
        assert len(backend.batches) == 1
    finally:
        runner.close()


def test_non_native_endpoint_uses_standard_requests_and_actual_standard_prices(tmp_path, data, monkeypatch):
    dest, backend, factory = run_fixture(tmp_path, data, monkeypatch, kind="openai_compatible")
    runner = Runner(dest, provider_factory=factory)
    try:
        assert runner.step()["generation"] == {"succeeded": 7}
        assert not backend.batches
        assert Counter(backend.calls) == {("POST", "/responses"): 7}
        for row in runner.store.rows():
            assert row["result"]["execution_mode"] == "standard"
            assert row["result"]["cost"]["execution_mode"] == "standard"
            assert row["result"]["cost"]["rates_usd_per_million"] == {"input": "1", "output": "2"}
    finally:
        runner.close()


@pytest.mark.parametrize("kind", ["openai_responses", "openai_compatible"])
def test_cost_cap_stops_before_any_http_submission(tmp_path, data, monkeypatch, kind):
    dest, backend, factory = run_fixture(tmp_path, data, monkeypatch, kind=kind, cap=0)
    runner = Runner(dest, provider_factory=factory)
    try:
        with pytest.raises(ValueError, match="max_cost_usd"):
            runner.step()
        assert backend.calls == []
        assert runner.status()["generation"] == {"pending": 7}
        assert all(r["attempts"] == 0 for r in runner.store.rows())
    finally:
        runner.close()
