"""Offline pooling tests. Every generated response here is an explicit test fixture."""
import json

import pytest

from heerbench.aggregation import ConsolidationError, consolidate_runs, read_run_snapshot
from heerbench.config import load_config
from heerbench.dataset import Dataset
from heerbench.evaluation import generation_records, prepare_validation
from heerbench.planning import create_plan
from heerbench.runner import Runner, load_run
from heerbench.store import Store
from heerbench.study_calibration import prepare_study_calibration
from heerbench.util import canonical, digest, project_root, read_jsonl, software_fingerprint, write_json, write_jsonl


def _segment(path, model="gpt6_luna", repetition=1, *, mutate=None, failed=0):
    cfg = load_config(project_root() / "configs/pilot.yaml", models=[model])
    cfg["model_repetition_ids"] = {model: [repetition]}
    cfg["study_id"] = "UNIT_TEST_PILOT_STUDY"
    if mutate:
        mutate(cfg)
    ds = Dataset(cfg["dataset"])
    create_plan(cfg, ds, path)
    store = Store(path)
    try:
        for index, plan in enumerate(read_jsonl(path / "plan.jsonl")):
            store.add(plan)
            request = {"request_id": plan["request_id"], "model": cfg["models"][0]["model"],
                       "system": "UNIT TEST SYSTEM ONLY", "prefix": "UNIT TEST PACKET " + plan["case_id"],
                       "task": "UNIT TEST QUESTION " + plan["question_id"], "max_output_tokens": 64}
            if index < failed:
                store.update(plan["request_id"], "failed", request=store.pack_request(request),
                             error={"kind": "UNIT_TEST_TRANSPORT_FAILURE"}, attempt=True)
                continue
            store.update(plan["request_id"], "succeeded", request=store.pack_request(request),
                         result={"request_id": plan["request_id"],
                                 "text": "UNIT TEST FIXTURE ONLY. " + ds.by_id[plan["question_id"]]["expected_answer"],
                                 "model_requested": cfg["models"][0]["model"],
                                 "model_returned": cfg["models"][0]["model"],
                                 "provider": "UNIT_TEST_FIXTURE_ONLY", "finish_reason": "stop",
                                 "raw_response": {"fixture_only": True, "id": plan["request_id"]},
                                 "usage": {"input_tokens": 123, "output_tokens": 12},
                                 "cost": {"total_usd": "0.123"}}, attempt=True)
        store.export()
    finally:
        store.close()
    return path


def _file_snapshot(path):
    return {p.relative_to(path).as_posix(): p.read_bytes() for p in path.rglob("*") if p.is_file()
            and not p.name.endswith(("-wal", "-shm"))}


def test_two_completed_individual_pilots_yield_one_200_answer_panel(tmp_path, monkeypatch):
    def no_network(*args, **kwargs):
        raise AssertionError("Calibration pooling must make no API request")
    monkeypatch.setattr("httpx.Client.send", no_network)
    first = _segment(tmp_path / "one", "gpt6_luna", repetition=3, failed=1)
    second = _segment(tmp_path / "two", "claude_haiku55", repetition=7, failed=2)
    before = {p: _file_snapshot(p) for p in [first, second]}
    source_rows = {r["id"]: r for p in [first, second] for r in read_run_snapshot(p)["generation"]}
    source_plans = {r["request_id"]: r for p in [first, second] for r in read_jsonl(p / "plan.jsonl")}
    assert all(len(read_jsonl(p / "plan.jsonl")) < 200 for p in [first, second])
    out = tmp_path / "calibration"
    result = prepare_study_calibration([first, second], out)
    assert result["run"] == str(out)
    assert result["records_copied"] == len(source_rows)
    assert result["available_answers"] == len(source_rows) - 3
    for path, snapshot in before.items():
        assert _file_snapshot(path) == snapshot
    _, cfg, _, plan = load_run(out)
    assert cfg["model_repetition_ids"] == {"claude_haiku55": [7], "gpt6_luna": [3]}
    assert {r["request_id"]: r for r in plan} == source_plans
    assert cfg["purpose"] == "study_calibration"
    assert cfg["reused_generation_only"] is True and cfg["calibration_only"] is True
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["config_sha256"] == digest(cfg)
    assert manifest["plan_sha256"] == digest(plan)
    assert manifest["software_sha256"] == software_fingerprint()
    assert manifest["dataset_sha256"] == cfg["dataset_fingerprint"]
    assert len(manifest["source_segments"]) == 2
    runner = Runner(out)
    try:
        assert {r["id"]: r for r in runner.store.rows("generation")} == source_rows
        assert runner.store.rows("judge") == []
        for record in runner.store.rows("generation"):
            request = runner.store.unpack_request(record["request"])
            assert request["system"] == "UNIT TEST SYSTEM ONLY"
        panel = tmp_path / "panel"
        prepare_validation(runner, panel, n=200, development_n=50)
        selection = json.loads((panel / "selection.private.json").read_text())
        assert len(selection["development"]) == 50
        assert len(selection["heldout"]) == 150
        development_groups = {a["group_id"] for a in selection["assignment"] if a["split"] == "development"}
        heldout_groups = {a["group_id"] for a in selection["assignment"] if a["split"] == "heldout"}
        assert development_groups.isdisjoint(heldout_groups)
        assert {r["model_id"] for r in selection["development"] + selection["heldout"]} == {"gpt6_luna", "claude_haiku55"}
        assert len(generation_records(runner)) == len(plan)
        with pytest.raises(ValueError, match="generation is disabled"):
            runner.step("generation")
        with pytest.raises(ValueError, match="cannot regenerate"):
            runner.recover([plan[0]["request_id"]], "UNIT TEST MUST NOT GENERATE")
    finally:
        runner.close()


def test_old_judges_panels_approvals_and_credentials_are_not_inherited(tmp_path):
    source = _segment(tmp_path / "source")
    store = Store(source)
    try:
        target = store.rows("generation")[0]["plan"]
        old_blob = store.put_blob("UNIT TEST OLD JUDGE CONTRACT MUST NOT BE COPIED")
        old_plan = {**{k: target[k] for k in ["model_id", "condition", "repetition", "question_id", "case_id", "abf_stage"]},
                    "request_id": "unit-test-old-judge", "generation_request_id": target["request_id"],
                    "judge_role": "primary", "executor_id": "gpt61_terra", "judge_contract_blob": old_blob}
        store.add(old_plan, "judge")
        store.update(old_plan["request_id"], "failed", error={"kind": "UNIT_TEST_OLD_JUDGE"})
    finally:
        store.close()
    (source / "secrets").mkdir()
    (source / "secrets/openai.key").write_text("UNIT_TEST_SECRET_DO_NOT_COPY")
    write_json(source / "validation_selection.json", {"sentinel": "OLD_PANEL"})
    write_json(source / "judge_approval.template.json", {"sentinel": "OLD_APPROVAL_TEMPLATE"})
    out = tmp_path / "calibration"
    prepare_study_calibration([source], out)
    assert not (out / "secrets").exists()
    assert not (out / "validation_selection.json").exists()
    assert not (out / "judge_approval.json").exists()
    assert not (out / "blobs" / old_blob[:2] / (old_blob + ".txt")).exists()
    assert "OLD_APPROVAL_TEMPLATE" not in (out / "judge_approval.template.json").read_text()
    assert read_jsonl(out / "judgements.jsonl") == []
    reuse = json.loads((out / "generation_reuse.json").read_text())
    assert reuse["judge_results_copied"] == reuse["approvals_copied"] == 0
    assert reuse["new_target_generation_cost_usd"] == "0"
    ledger = read_jsonl(out / "generation_reuse_ledger.jsonl")
    assert all(not row["is_new_target_observation"] and row["new_target_cost_usd"] == "0" for row in ledger)
    assert all(row["historical_cost"]["total_usd"] == "0.123" for row in ledger)


@pytest.mark.parametrize("state", ["missing", "pending", "running", "submitted", "uncertain"])
def test_incomplete_source_cannot_be_silently_dropped_or_reconstructed(tmp_path, state):
    source = _segment(tmp_path / "source")
    store = Store(source)
    try:
        task_id = store.rows("generation")[0]["id"]
        if state == "missing":
            store.db.execute("DELETE FROM tasks WHERE id=?", (task_id,))
            store.db.commit()
        else:
            store.update(task_id, state)
    finally:
        store.close()
    out = tmp_path / "calibration"
    with pytest.raises(ConsolidationError, match="Finish the selected pilot segments first"):
        prepare_study_calibration([source], out)
    assert not out.exists()


def test_duplicate_logical_runs_are_rejected_even_with_distinct_physical_ids(tmp_path):
    first = _segment(tmp_path / "one")
    second = _segment(tmp_path / "two", mutate=lambda cfg: cfg.update(study_id="ANOTHER_TEST_ID"))
    assert read_jsonl(first / "plan.jsonl")[0]["request_id"] != read_jsonl(second / "plan.jsonl")[0]["request_id"]
    with pytest.raises(ConsolidationError, match="Duplicate logical"):
        prepare_study_calibration([first, second], tmp_path / "out")
    with pytest.raises(ConsolidationError, match="Duplicate source segment"):
        prepare_study_calibration([first, first], tmp_path / "out")


def test_disjoint_global_repetitions_of_same_model_keep_their_ids(tmp_path):
    first = _segment(tmp_path / "one", repetition=2)
    second = _segment(tmp_path / "two", repetition=6)
    out = tmp_path / "out"
    prepare_study_calibration([first, second], out)
    _, cfg, _, plan = load_run(out)
    assert cfg["model_repetition_ids"] == {"gpt6_luna": [2, 6]}
    assert {row["repetition"] for row in plan} == {2, 6}


def test_conflicting_model_deployments_and_protocols_are_rejected(tmp_path):
    first = _segment(tmp_path / "one", repetition=1)
    second = _segment(tmp_path / "two", repetition=2,
                      mutate=lambda cfg: cfg["models"][0].update(precision="TEST_DIFFERENT_PRECISION"))
    with pytest.raises(ConsolidationError, match="conflicting deployment"):
        prepare_study_calibration([first, second], tmp_path / "out")
    third = _segment(tmp_path / "three", "claude_haiku55",
                     mutate=lambda cfg: cfg["generation"].update(max_output_tokens=8192))
    with pytest.raises(ConsolidationError, match="incompatible corpus"):
        prepare_study_calibration([first, third], tmp_path / "out")


def test_corrupt_referenced_blob_rejects_pooling_before_output(tmp_path):
    source = _segment(tmp_path / "source")
    row = read_run_snapshot(source)["generation"][0]
    blob = row["request"]["prefix_blob"]
    (source / "blobs" / blob[:2] / (blob + ".txt")).write_text("CORRUPTED TEST BLOB")
    with pytest.raises(ConsolidationError, match="blob is missing, corrupted"):
        prepare_study_calibration([source], tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_sequence_cannot_use_another_models_parent_even_with_valid_source_hashes(tmp_path):
    first = _segment(tmp_path / "one", "gpt6_luna")
    second = _segment(tmp_path / "two", "claude_haiku55")
    first_plan = read_jsonl(first / "plan.jsonl")
    second_plan = read_jsonl(second / "plan.jsonl")
    child = next(p for p in second_plan if p["condition"] == "E3" and p["parent_request_ids"])
    wrong_parent = next(p for p in first_plan if p["question_id"] == child["parent_question_ids"][0]
                        and p["condition"] in {"E1", "E3"})
    child["parent_request_ids"][0] = wrong_parent["request_id"]
    write_jsonl(second / "plan.jsonl", second_plan)
    manifest = json.loads((second / "manifest.json").read_text())
    manifest["plan_sha256"] = digest(second_plan)
    write_json(second / "manifest.json", manifest)
    store = Store(second)
    try:
        store.db.execute("UPDATE tasks SET plan=? WHERE id=?", (canonical(child), child["request_id"]))
        store.db.commit()
    finally:
        store.close()
    with pytest.raises(ConsolidationError, match="Sequential dependency crosses"):
        prepare_study_calibration([first, second], tmp_path / "out")


def test_mock_or_already_reused_observations_cannot_become_a_new_panel_population(tmp_path):
    source = _segment(tmp_path / "source")
    bundle = tmp_path / "bundle"
    prepare_study_calibration([source], bundle)
    with pytest.raises(ConsolidationError, match="original generation segments"):
        prepare_study_calibration([bundle], tmp_path / "again")
    store = Store(source)
    try:
        record = store.rows("generation")[0]
        response = record["result"]
        response["provider"] = "mock"
        store.update(record["id"], record["state"], result=response)
    finally:
        store.close()
    with pytest.raises(ConsolidationError, match="real target generations"):
        prepare_study_calibration([source], tmp_path / "mock_pool")


def test_normal_reporting_excludes_calibration_copy(tmp_path):
    source = _segment(tmp_path / "source")
    out = tmp_path / "calibration"
    prepare_study_calibration([source], out)
    report = consolidate_runs(tmp_path, tmp_path / "reports", bootstrap=0)
    assert sum(len(group["models"]) for group in report["groups"]) == 1
    assert report["groups"][0]["source_runs"] == 1
    assert any(row["source_directory"] == str(out) for row in report["excluded_runs"])


def test_existing_or_overlapping_destinations_remain_untouched(tmp_path):
    source = _segment(tmp_path / "source")
    out = tmp_path / "occupied"
    out.mkdir()
    (out / "keep.txt").write_text("KEEP THIS TEST FILE")
    for destination in [source, source / "nested", tmp_path, out]:
        with pytest.raises(ConsolidationError):
            prepare_study_calibration([source], destination)
    assert (out / "keep.txt").read_text() == "KEEP THIS TEST FILE"

