"""Offline tests of scientific pooling boundaries and frozen-plan denominators.

All response records are explicit synthetic test fixtures in pytest directories.
No provider is instantiated and none of these records are experiment results.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil

import pytest

from heerbench.aggregation import (
    ConsolidationError, compatibility_key, consolidate_runs, model_signature,
    read_run_snapshot,
)
from heerbench.dataset import Dataset
from heerbench.judge_validation import human_review_protocol, judge_configuration_digest
from heerbench.planning import create_plan
from heerbench.store import Store
from heerbench.util import digest, read_jsonl, write_json, write_jsonl


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def dataset():
    ds = Dataset(ROOT / "data/heerbench_v05")
    ds.validate()
    return ds


def config(ds, model="fixture-m1", *, rep_ids=(1,), mode="pilot", study="paper", **updates):
    chain = min(ds.chains, key=lambda c: (len(ds.chain_nodes(c["chain_id"])), c["chain_id"]))
    cfg = {
        "mode": mode, "dataset": str(ds.root), "chain_ids": [chain["chain_id"]],
        "chains_per_scenario": 1, "seed": 42, "repetitions": max(rep_ids),
        "model_repetition_ids": {model: list(rep_ids)}, "study_id": study,
        "conditions": ["E1", "E2", "E3"],
        "generator_system": "Offline test fixture generator prompt.",
        "judge_system": "Offline test fixture judge prompt.",
        "generation": {"max_output_tokens": 1024, "temperature": 0.2},
        "models": [{"id": model, "model": model + "-pinned-1", "kind": "openai_compatible",
                    "base_url": "https://fixture.invalid/v1", "parameters": {"temperature": 0.2},
                    "api_key_file": "secrets/fixture.key", "pricing": {"input": "1"}}],
        "judge_models": [{"id": "fixture-judge", "model": "fixture-judge-pinned-1",
                          "kind": "openai_compatible", "base_url": "https://fixture.invalid/v1",
                          "parameters": {"temperature": 0}}],
        "judge": {"primary": "fixture-judge", "secondary": None, "secondary_fraction": 0.1,
                  "secondary_sampling_unit": "question"},
    }
    cfg.update(updates)
    return cfg


def make_run(root, name, ds, cfg, *, complete=0, fail=0, returned=None, judge_returned=None):
    out = root / name
    create_plan(cfg, ds, out)
    rows = read_jsonl(out / "plan.jsonl")
    if complete or fail:
        store = Store(out)
        try:
            for i, p in enumerate(rows[:complete + fail]):
                store.add(p)
                if i >= complete:
                    store.update(p["request_id"], "failed", error={"kind": "synthetic_transport_failure"})
                    continue
                source_model = next(m for m in cfg["models"] if m["id"] == p["model_id"])
                store.update(p["request_id"], "succeeded", result={
                    "text": "SYNTHETIC PYTEST FIXTURE", "provider": "pytest_fixture",
                    "model_returned": returned or source_model["model"], "raw_response": {},
                })
                judge = {**p, "request_id": "judge-" + p["request_id"],
                         "generation_request_id": p["request_id"], "judge_role": "primary",
                         "executor_id": "fixture-judge"}
                store.add(judge, "judge")
                store.update(judge["request_id"], "succeeded", result={
                    "model_returned": judge_returned or "fixture-judge-pinned-1",
                    "score": {"criterion_score": 1.0, "fully_correct": True,
                              "has_error": False, "negative_error_rate": None, "scoreable": True},
                })
        finally:
            store.close()
    return out, rows


def report_for(result, group=0):
    return json.loads(Path(result["groups"][group]["files"]["analysis_json"]).read_text())


def accept_fixture_judge(run, ds, reviewers=None):
    """Synthetic provenance fixture only; never a real expert decision."""
    cfg = json.loads((run / "config.frozen.json").read_text())
    configuration = judge_configuration_digest(cfg["judge_system"], cfg["judge_models"][0], ds.fingerprint)
    heldout = {"split": "heldout", "n_answers": 1, "is_test_fixture": True,
               "judge_configuration_sha256": configuration}
    basis = human_review_protocol(reviewers) if reviewers is not None else None
    if basis is not None:
        heldout["human_review"] = basis
    approval = {"decision": "approved", "approved_by": "SYNTHETIC PYTEST FIXTURE",
                "approved_at_utc": "2026-01-01T00:00:00Z", "scope": "Offline test only",
                "decision_rationale": "Synthetic provenance test, not a scientific assessment",
                "configuration_sha256": configuration, "heldout_report_sha256": digest(heldout)}
    # Metadata must be recoverable from the hash-bound report even if a generic
    # older approval template did not duplicate that information.
    write_json(run / "judge_validation_heldout.json", heldout)
    write_json(run / "judge_approval.json", approval)
    return basis


@pytest.mark.parametrize("reviewers", [None, 1, 2])
def test_accepted_calibration_preserves_reference_basis_without_inventing_legacy_counts(tmp_path, dataset, reviewers):
    run, _ = make_run(tmp_path, "run", dataset, config(dataset), complete=1)
    basis = accept_fixture_judge(run, dataset, reviewers)
    snapshot = read_run_snapshot(run, dataset)
    assert snapshot["judge_validated"] is True
    assert snapshot["judge_validation_basis"] == basis
    result = consolidate_runs(run, tmp_path / "reports", bootstrap=0)
    report = report_for(result)
    assert report["judge_validated_for_publication"] is True
    assert report["judge_validation_basis"] == basis
    ledger = json.loads((Path(result["groups"][0]["directory"]) / "source_run_ledger.json").read_text())
    assert ledger[0]["judge_validation_basis"] == basis
    if reviewers is None:
        text = Path(result["groups"][0]["files"]["report_markdown"]).read_text()
        assert "no reviewer count inferred" in text


def test_mixed_reference_designs_remain_per_source_without_changing_scientific_grouping(tmp_path, dataset):
    runs = tmp_path / "runs"
    first, _ = make_run(runs, "first", dataset, config(dataset), complete=1)
    second, _ = make_run(runs, "second", dataset, config(dataset, "fixture-m2"), complete=1)
    one = accept_fixture_judge(first, dataset, 1)
    two = accept_fixture_judge(second, dataset, 2)
    result = consolidate_runs(runs, tmp_path / "reports", bootstrap=0)
    assert result["protocol_groups"] == 1
    report = report_for(result)
    assert report["judge_validated_for_publication"] is True
    bases = report["judge_validation_basis"]
    assert {row["source_directory"]: row["human_review"] for row in bases} == {"first": one, "second": two}
    assert all(row["calibration_accepted"] is True for row in bases)
    text = Path(result["groups"][0]["files"]["report_markdown"]).read_text()
    assert "One military expert" in text and "Two experts, adjudicated reference" in text
    assert "Reviewer counts are not added across runs" in text


def test_new_models_merge_without_losing_failed_or_unstarted_plan_rows(tmp_path, dataset):
    runs = tmp_path / "runs"
    first, p1 = make_run(runs, "first", dataset, config(dataset), complete=1, fail=1)
    second, p2 = make_run(runs, "second", dataset, config(dataset, "fixture-m2"))
    before = hashlib.sha256((first / "journal.sqlite").read_bytes()).hexdigest()
    result = consolidate_runs(runs, tmp_path / "reports", bootstrap=0)
    assert result["protocol_groups"] == 1
    report = report_for(result)
    assert report["metadata"]["models"] == ["fixture-m1", "fixture-m2"]
    assert report["metadata"]["planned_generation_requests"] == len(p1) + len(p2)
    assert len(report["question_scores"]) == len(p1) + len(p2)
    assert sum(r["generation_status"] == "failed" for r in report["question_scores"]) == 1
    assert sum(r["generation_status"] == "pending" for r in report["question_scores"]) == len(p1) + len(p2) - 2
    assert sum(r["scoreable"] for r in report["question_scores"]) == 1
    assert all(r["condition"] == "E1" for r in report["question_scores"] if r["abf_stage"] == "A")
    assert report["judge_validated_for_publication"] is False
    assert "provisional" in " ".join(report["metadata"]["warnings"]).lower()
    assert not (second / "journal.sqlite").exists()
    assert hashlib.sha256((first / "journal.sqlite").read_bytes()).hexdigest() == before


def test_incremental_repetition_ids_are_preserved_and_not_counted_as_new_scenarios(tmp_path, dataset):
    runs = tmp_path / "runs"
    make_run(runs, "rep1", dataset, config(dataset, rep_ids=(1,)), complete=100)
    make_run(runs, "rep2to3", dataset, config(dataset, rep_ids=(2, 3)), complete=100)
    result = consolidate_runs(runs, tmp_path / "reports", bootstrap=0)
    report = report_for(result)
    assert report["metadata"]["repetitions_by_model"] == {"fixture-m1": 3}
    assert {r["repetition"] for r in report["question_scores"]} == {1, 2, 3}
    assert len(report["metadata"]["scenarios"]) == 1
    ledger = json.loads((Path(result["groups"][0]["directory"]) / "source_run_ledger.json").read_text())
    assert sorted(r["repetition_ids_by_model"]["fixture-m1"] for r in ledger) == [[1], [2, 3]]


def test_overlap_is_rejected_even_when_source_runs_have_different_ids(tmp_path, dataset):
    runs = tmp_path / "runs"
    make_run(runs, "one", dataset, config(dataset, study="one"), complete=1)
    make_run(runs, "two", dataset, config(dataset, study="two"), complete=1)
    with pytest.raises(ConsolidationError, match="Overlapping logical observation"):
        consolidate_runs(runs, tmp_path / "reports", bootstrap=0)
    assert not (tmp_path / "reports" / "index.json").exists()
    selected = consolidate_runs(runs, tmp_path / "selected", study="one", bootstrap=0)
    assert selected["groups"][0]["source_runs"] == 1
    assert selected["excluded_runs"][0]["reason"] == "different_study"


def test_exact_copied_physical_snapshots_are_deduplicated_but_divergent_copies_fail(tmp_path, dataset):
    runs = tmp_path / "runs"
    first, plan = make_run(runs, "one", dataset, config(dataset), complete=1)
    copy = runs / "copy"
    shutil.copytree(first, copy)
    result = consolidate_runs(runs, tmp_path / "reports", bootstrap=0)
    assert result["groups"][0]["source_runs"] == 1
    assert result["groups"][0]["duplicate_snapshots_excluded"] == 1
    assert report_for(result)["metadata"]["planned_generation_requests"] == len(plan)
    store = Store(copy)
    try:
        store.update(plan[0]["request_id"], "succeeded", result={
            "text": "DIFFERENT SYNTHETIC RECORD", "provider": "pytest_fixture",
            "model_returned": "fixture-m1-pinned-1", "raw_response": {},
        })
    finally:
        store.close()
    with pytest.raises(ConsolidationError, match="Overlapping logical observation"):
        consolidate_runs(runs, tmp_path / "conflict", bootstrap=0)


@pytest.mark.parametrize("change", ["prompt", "judge", "scope", "sampling"])
def test_changed_scientific_protocols_get_separate_reports(tmp_path, dataset, change):
    runs = tmp_path / "runs"
    base = config(dataset)
    make_run(runs, "one", dataset, base)
    changed = config(dataset, "fixture-m2")
    if change == "prompt":
        changed["generator_system"] += " A substantive instruction changed."
    elif change == "judge":
        changed["judge_system"] += " Different evaluation criterion instruction."
    elif change == "scope":
        changed["mode"] = "full"
    else:
        changed["judge"]["secondary_sampling_unit"] = "answer"
    make_run(runs, "two", dataset, changed)
    result = consolidate_runs(runs, tmp_path / "reports", bootstrap=0)
    assert result["protocol_groups"] == 2
    assert all(group["source_runs"] == 1 for group in result["groups"])
    index = json.loads(Path(result["index_json"]).read_text())
    assert index["groups"][1]["protocol_fields_differing_from_first_group"]


def test_same_model_label_cannot_hide_deployment_parameter_or_version_changes(tmp_path, dataset):
    runs = tmp_path / "runs"
    make_run(runs, "one", dataset, config(dataset))
    changed = config(dataset, rep_ids=(2,))
    changed["models"][0]["parameters"]["temperature"] = 0.7
    make_run(runs, "two", dataset, changed)
    with pytest.raises(ConsolidationError, match="incompatible deployment or inference settings"):
        consolidate_runs(runs, tmp_path / "reports", bootstrap=0)


@pytest.mark.parametrize("role", ["target", "judge"])
def test_observed_version_drift_is_not_silently_merged(tmp_path, dataset, role):
    runs = tmp_path / "runs"
    make_run(runs, "one", dataset, config(dataset), complete=1)
    kwargs = {"returned": "fixture-m1-pinned-2"} if role == "target" else {"judge_returned": "fixture-judge-pinned-2"}
    make_run(runs, "two", dataset, config(dataset, rep_ids=(2,)), complete=1, **kwargs)
    with pytest.raises(ConsolidationError, match="different version identifiers"):
        consolidate_runs(runs, tmp_path / "reports", bootstrap=0)


def test_prices_key_paths_and_display_metadata_do_not_change_scientific_identity(dataset):
    a = config(dataset)
    b = deepcopy(a)
    b["models"][0].update(pricing={"input": "10"}, api_key_file="secrets/relocated.key",
                           display_name="Updated display label", availability="account_verified",
                           pricing_valid_until="2027-12-31")
    assert model_signature(a["models"][0]) == model_signature(b["models"][0])
    b["judge_models"][0].update(pricing={"output": "100"}, api_key_file="secrets/other.key")
    b["repetitions"] = 9
    b["models"].append({"id": "additional-model"})
    assert compatibility_key(a, dataset) == compatibility_key(b, dataset)
    b["models"][0]["parameters"]["temperature"] = 0.9
    assert model_signature(a["models"][0]) != model_signature(b["models"][0])


def test_mock_and_judge_only_forks_are_excluded_explicitly(tmp_path, dataset):
    runs = tmp_path / "runs"
    make_run(runs, "original", dataset, config(dataset))
    make_run(runs, "mock", dataset, config(dataset, mode="demo"))
    make_run(runs, "fork", dataset, config(dataset, reused_generation_only=True, calibration_only=True))
    result = consolidate_runs(runs, tmp_path / "reports", bootstrap=0)
    assert result["groups"][0]["source_runs"] == 1
    assert {r["reason"] for r in result["excluded_runs"]} == {
        "mock_pipeline_run", "judge_development_fork_not_new_generation"}


def test_integrity_and_complete_declared_plan_are_checked(tmp_path, dataset):
    run, plan = make_run(tmp_path, "run", dataset, config(dataset))
    manifest = json.loads((run / "manifest.json").read_text())
    write_jsonl(run / "plan.jsonl", plan[1:])
    with pytest.raises(ConsolidationError, match="request-plan hash mismatch"):
        read_run_snapshot(run)
    manifest["plan_sha256"] = digest(plan[1:])
    write_json(run / "manifest.json", manifest)
    with pytest.raises(ConsolidationError, match="does not cover exactly"):
        read_run_snapshot(run)


def test_sequential_parent_cannot_cross_repetitions(tmp_path, dataset):
    run, plan = make_run(tmp_path, "run", dataset, config(dataset, rep_ids=(1, 2)))
    victim = next(p for p in plan if p["condition"] == "E3" and p["repetition"] == 1)
    parent_question = victim["parent_question_ids"][0]
    wrong = next(p for p in plan if p["question_id"] == parent_question and p["repetition"] == 2
                 and p["condition"] == ("E1" if p["abf_stage"] == "A" else "E3"))
    victim["parent_request_ids"][0] = wrong["request_id"]
    manifest = json.loads((run / "manifest.json").read_text())
    manifest["plan_sha256"] = digest(plan)
    write_jsonl(run / "plan.jsonl", plan)
    write_json(run / "manifest.json", manifest)
    with pytest.raises(ConsolidationError, match="crosses an absent model or repetition"):
        consolidate_runs(run, tmp_path / "reports", bootstrap=0)


def test_historical_source_code_is_not_executed_or_required_for_readonly_reporting(tmp_path, dataset):
    run, _ = make_run(tmp_path, "run", dataset, config(dataset))
    manifest = json.loads((run / "manifest.json").read_text())
    manifest["software_sha256"] = "historical-software-fingerprint"
    write_json(run / "manifest.json", manifest)
    result = consolidate_runs(run, tmp_path / "reports", bootstrap=0)
    assert report_for(result)["metadata"]["source_software_hashes"] == ["historical-software-fingerprint"]


def test_unaccepted_heldout_metrics_remain_provisional_and_unsupported_approval_fails(tmp_path, dataset):
    run, _ = make_run(tmp_path, "run", dataset, config(dataset))
    write_json(run / "judge_validation_heldout.json", {"fixture": "Not an accepted approval"})
    result = consolidate_runs(run, tmp_path / "provisional", bootstrap=0)
    assert report_for(result)["judge_validated_for_publication"] is False
    (run / "judge_validation_heldout.json").unlink()
    write_json(run / "judge_approval.json", {"fixture": "Unsupported claim"})
    with pytest.raises(ConsolidationError, match="Incomplete judge acceptance"):
        consolidate_runs(run, tmp_path / "invalid", bootstrap=0)
