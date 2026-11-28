"""Single-expert calibration regressions using synthetic, offline fixtures only.

Every answer, annotation and approval below exists in a temporary test directory.
These deliberately artificial labels test software contracts, not military truth,
judge accuracy in the benchmark, or a real expert's acceptance of a judge.
"""
import csv
import json
from types import SimpleNamespace

import httpx
import pytest

from heerbench.cli import parser
from heerbench.evaluation import calibration_report, prepare_judges, prepare_validation
from heerbench.judge import JudgeRequest, SYSTEM_PROMPT, score_judgement
from heerbench.judge_validation import validate_judge_approval
from heerbench.store import Store
from heerbench.util import digest, write_json

from test_judge import panel_fixture, valid_pass


SINGLE_EXPERT = {
    "reviewers": 1,
    "reference_type": "single_expert",
    "reference_file": "reviewer_1.csv",
}


@pytest.fixture
def runner(tmp_path, monkeypatch):
    """Make synthetic completed generations without constructing any provider."""
    def reject_network(*args, **kwargs):
        raise AssertionError("Calibration regression fixtures must not contact an API")

    monkeypatch.setattr(httpx.Client, "request", reject_network)
    monkeypatch.setattr(httpx.AsyncClient, "request", reject_network)
    questions, sources, errors, chains, records = panel_fixture()
    local = [{"question_id": qid, "local_error_id": qid + "-E1",
              "text": q["error_conditions"][0]} for qid, q in questions.items()]
    dataset = SimpleNamespace(
        by_id=questions, sources=sources, errors=errors, local_errors=local,
        chains=chains, fingerprint="SYNTHETIC_SINGLE_EXPERT_TEST_CORPUS",
        for_question=lambda rows, qid: [r for r in rows if r["question_id"] == qid],
    )
    cfg = {
        "mode": "pilot", "is_mock": False, "seed": 20261127,
        "judge_system": SYSTEM_PROMPT + "\nUNIT TEST CONFIGURATION ONLY.\n",
        "judge": {"primary": "test-primary", "secondary": "test-secondary",
                  "secondary_fraction": .2},
        "judge_models": [
            {"id": "test-primary", "model": "fixture-primary-version", "max_output_tokens": 8192},
            {"id": "test-secondary", "model": "fixture-secondary-version", "max_output_tokens": 8192},
        ],
    }
    instance = SimpleNamespace(root=tmp_path, cfg=cfg, dataset=dataset, store=Store(tmp_path))
    for record in records:
        plan = {k: v for k, v in record.items() if k not in {"task_id", "generation_status", "text"}}
        q = questions[record["question_id"]]
        plan.update(request_id=record["task_id"], case_id=q["case_id"], abf_stage=q["abf_stage"],
                    parent_request_ids=[], parent_question_ids=[])
        instance.store.add(plan, "generation")
        instance.store.update(plan["request_id"], "succeeded", result={
            "text": record["text"], "provider": "UNIT_TEST_FIXTURE_ONLY",
            "fixture_not_research_data": True,
        })
    yield instance
    instance.store.close()


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        return list(reader), reader.fieldnames


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def panel(runner, *, n=8, development_n=2, reviewers=1):
    directory = runner.root / "panel"
    prepare_validation(runner, directory, n=n, development_n=development_n, reviewers=reviewers)
    return directory, json.loads((directory / "selection.private.json").read_text())


def fill_test_labels(directory, selection, *, filenames=("reviewer_1.csv",),
                     splits=("development", "heldout"), one_failure_in=None):
    """Synthetic labels only. Real production worksheets stay blank on export."""
    split_by_id = {a["blind_answer_id"]: a["split"] for a in selection["assignment"]}
    changed = False
    for filename in filenames:
        path = directory / filename
        rows, fields = read_csv(path)
        for row in rows:
            if split_by_id[row["blind_answer_id"]] not in splits:
                continue
            row["verdict"] = "satisfied" if row["kind"] == "positive" else "error_absent"
            row["comment"] = "SYNTHETIC UNIT TEST LABEL, NOT A HUMAN ANNOTATION"
            if (not changed and filename == one_failure_in and row["kind"] == "positive"
                    and split_by_id[row["blind_answer_id"]] == "heldout"):
                row["verdict"] = "not_satisfied"
                changed = True
        write_csv(path, rows, fields)


def complete_test_judges(runner, *, skip_generation=None):
    for row in runner.store.rows("judge"):
        if row["plan"]["generation_request_id"] == skip_generation:
            continue
        contract = JudgeRequest(**json.loads(runner.store.get_blob(row["plan"]["judge_contract_blob"])))
        judgement = valid_pass(contract)
        runner.store.update(row["id"], "succeeded", result={
            "judgement": judgement, "score": score_judgement(judgement, contract),
            "fixture_not_research_data": True,
        })


def prepare_test_judges(runner, directory, *, repetitions=None):
    kwargs = {} if repetitions is None else {"validation_repetitions": repetitions}
    return prepare_judges(runner, validation_selection=directory / "selection.private.json", **kwargs)


def assert_single_expert_metadata(report):
    assert {key: report["human_review"][key] for key in SINGLE_EXPERT} == SINGLE_EXPERT


def test_cli_defaults_and_explicit_alternatives():
    cli = parser()
    args = cli.parse_args(["calibration-prepare", "fixture-run", "--out", "fixture-panel"])
    assert (args.n, args.development_n, args.reviewers, args.group_by) == (60, 10, 1, "component")
    args = cli.parse_args(["judge-prepare", "fixture-run", "--selection", "selection.private.json"])
    assert args.repetitions == 1
    args = cli.parse_args(["calibration-report", "fixture-run", "--panel", "fixture-panel"])
    assert args.split == "all"
    args = cli.parse_args(["calibration-prepare", "fixture-run", "--out", "fixture-panel", "--reviewers", "2"])
    assert args.reviewers == 2
    args = cli.parse_args(["calibration-report", "fixture-run", "--panel", "fixture-panel", "--split", "development"])
    assert args.split == "development"


def test_default_panel_is_60_answers_10_development_with_one_reviewer(runner):
    directory = runner.root / "panel"
    prepare_validation(runner, directory)
    selection = json.loads((directory / "selection.private.json").read_text())
    assert (len(selection["development"]), len(selection["heldout"])) == (10, 50)
    assert_single_expert_metadata(selection)
    assert (directory / "reviewer_1.csv").is_file()
    assert not (directory / "reviewer_2.csv").exists()
    assert not (directory / "adjudicated.csv").exists()
    groups = {split: {a["group_id"] for a in selection["assignment"] if a["split"] == split}
              for split in ("development", "heldout")}
    assert groups["development"].isdisjoint(groups["heldout"])
    assert all(not row["verdict"] for row in read_csv(directory / "reviewer_1.csv")[0])


def test_default_single_call_report_uses_actual_reference_and_preserves_undefined_metrics(runner):
    directory, selection = panel(runner, n=60, development_n=10)
    queued = prepare_test_judges(runner, directory)
    assert queued["new_judge_requests"] == 60
    assert {r["plan"]["judge_repetition"] for r in runner.store.rows("judge")} == {1}
    assert prepare_test_judges(runner, directory)["new_judge_requests"] == 0
    fill_test_labels(directory, selection, one_failure_in="reviewer_1.csv")
    complete_test_judges(runner)
    report = calibration_report(runner, directory)
    assert_single_expert_metadata(report)
    assert_single_expert_metadata(report["heldout"])
    assert report["development"]["n_answers"] == 10
    assert report["heldout"]["n_answers"] == 50
    assert report["expert_agreement"] is None
    assert report["repeatability"] is None
    positive = report["heldout"]["by_kind"]["positive"]
    assert 0 < positive["agreement"] < 1
    assert positive["cohen_kappa"] == pytest.approx(0)
    assert positive["failure_f1"] == 0
    assert positive["coverage"] == 1
    for kind in ("local_error", "inherited_error"):
        assert report["heldout"]["by_kind"][kind]["cohen_kappa"] is None
        assert report["heldout"]["by_kind"][kind]["failure_f1"] is None
    stored = json.loads((directory / "judge_heldout_report.json").read_text())
    hashes = json.loads((directory / "judge_approval.hashes.json").read_text())
    assert hashes["heldout_report_sha256"] == digest(stored) == digest(report["heldout"])
    assert not (directory / "reviewer_2.csv").exists()
    assert not (directory / "adjudicated.csv").exists()


def test_development_report_does_not_require_or_score_heldout_labels(runner):
    directory, selection = panel(runner)
    prepare_test_judges(runner, directory)
    complete_test_judges(runner)
    fill_test_labels(directory, selection, splits=("development",))
    report = calibration_report(runner, directory, split="development")
    assert report["development"]["n_answers"] == 2
    assert "heldout" not in report
    assert not (directory / "judge_heldout_report.json").exists()
    assert not (directory / "judge_approval.hashes.json").exists()
    assert not (runner.root / "judge_approval.json").exists()
    with pytest.raises(ValueError):
        calibration_report(runner, directory, split="all")


def test_heldout_report_does_not_require_or_score_development_labels(runner):
    directory, selection = panel(runner)
    prepare_test_judges(runner, directory)
    complete_test_judges(runner)
    fill_test_labels(directory, selection, splits=("heldout",))
    report = calibration_report(runner, directory, split="heldout")
    assert report["heldout"]["n_answers"] == 6
    assert "development" not in report
    assert not (directory / "judge_development_report.json").exists()
    assert (directory / "judge_approval.hashes.json").is_file()
    with pytest.raises(ValueError):
        calibration_report(runner, directory, split="all")


@pytest.mark.parametrize("corruption", ["missing_positive", "missing_local_error", "missing_inherited_error",
                                        "duplicate", "unknown", "blank_required"])
def test_reference_requires_exact_complete_criterion_coverage(runner, corruption):
    directory, selection = panel(runner)
    prepare_test_judges(runner, directory)
    complete_test_judges(runner)
    fill_test_labels(directory, selection)
    path = directory / "reviewer_1.csv"
    rows, fields = read_csv(path)
    if corruption.startswith("missing_"):
        kind = corruption.removeprefix("missing_")
        rows.pop(next(i for i, row in enumerate(rows) if row["kind"] == kind))
    elif corruption == "duplicate":
        rows.append(dict(rows[0]))
    elif corruption == "unknown":
        extra = dict(rows[0], item_id="NONEXISTENT_TEST_CRITERION")
        rows.append(extra)
    else:
        rows[0]["verdict"] = ""
    write_csv(path, rows, fields)
    with pytest.raises(ValueError):
        calibration_report(runner, directory)


@pytest.mark.parametrize("legacy", [False, True], ids=["explicit_two_experts", "legacy_panel"])
def test_two_expert_adjudicated_workflow_remains_available(runner, legacy):
    directory, selection = panel(runner, reviewers=2)
    filenames = ("reviewer_1.csv", "reviewer_2.csv", "adjudicated.csv")
    assert all((directory / name).exists() for name in filenames)
    if legacy:
        for name in ("selection.private.json", "selection.json"):
            path = directory / name
            if path.exists():
                data = json.loads(path.read_text())
                data.pop("human_review", None)
                write_json(path, data)
        selection.pop("human_review", None)
    assert prepare_test_judges(runner, directory, repetitions=3)["new_judge_requests"] == 24
    fill_test_labels(directory, selection, filenames=filenames, one_failure_in="reviewer_2.csv")
    complete_test_judges(runner)
    report = calibration_report(runner, directory)
    assert report["human_review"]["reviewers"] == 2
    assert report["human_review"]["reference_file"] == "adjudicated.csv"
    assert report["heldout"]["by_kind"]["positive"]["agreement"] == 1
    assert report["expert_agreement"]["by_kind"]["positive"]["agreement"] < 1
    assert report["repeatability"]["repeats"] == 3


def test_one_expert_labels_do_not_implicitly_approve_production_judging(runner):
    directory, selection = panel(runner)
    prepare_test_judges(runner, directory)
    fill_test_labels(directory, selection)
    complete_test_judges(runner)
    calibration_report(runner, directory)
    runner.cfg["mode"] = "full"
    with pytest.raises(ValueError, match="acceptance|approval|requires"):
        prepare_judges(runner)
    template = json.loads((directory / "judge_approval.template.json").read_text())
    template.update(
        decision="approved", approved_by="UNIT TEST FIXTURE, NOT A PERSON",
        approved_at_utc="2000-01-01T00:00:00Z", scope="UNIT TEST ONLY",
        decision_rationale="Exercise approval binding, not scientific acceptance.",
        heldout_report_sha256=json.loads((directory / "judge_approval.hashes.json").read_text())["heldout_report_sha256"],
    )
    approval_path = runner.root / "fixture_approval.json"
    write_json(approval_path, template)
    kwargs = {"approval_file": approval_path,
              "heldout_report_file": directory / "judge_heldout_report.json"}
    assert prepare_judges(runner, **kwargs)["new_judge_requests"] == 224 + 45
    runner.cfg["judge_models"][0]["model"] = "changed-primary-version"
    with pytest.raises(ValueError, match="configuration"):
        prepare_judges(runner, **kwargs)


def test_missing_judge_answer_reduces_coverage_without_removing_reference_rows(runner):
    directory, selection = panel(runner)
    prepare_test_judges(runner, directory)
    fill_test_labels(directory, selection)
    heldout_task = selection["heldout"][0]["task_id"]
    complete_test_judges(runner, skip_generation=heldout_task)
    report = calibration_report(runner, directory, split="heldout")
    assert report["heldout"]["n_answers"] == 6
    assert report["heldout"]["n_missing_prediction_rows"] > 0
    assert report["heldout"]["by_kind"]["positive"]["coverage"] < 1
    assert report["repeatability"] is None


def test_mock_answers_cannot_populate_human_calibration_panel(runner):
    runner.cfg["is_mock"] = True
    with pytest.raises(ValueError, match="real answers|eligible|mock"):
        panel(runner)
    assert not (runner.root / "panel" / "reviewer_1.csv").exists()


def test_unknown_report_split_is_rejected(runner):
    directory, _ = panel(runner)
    with pytest.raises(ValueError):
        calibration_report(runner, directory, split="typo")


@pytest.mark.parametrize("field,value", [
    ("model", "different-primary-model-version"),
    ("max_output_tokens", 16384),
])
def test_report_cannot_relabel_completed_judgements_with_a_changed_judge_configuration(runner, field, value):
    """Identical executor IDs do not imply identical frozen model configurations."""
    directory, selection = panel(runner)
    prepare_test_judges(runner, directory)
    fill_test_labels(directory, selection)
    complete_test_judges(runner)
    runner.cfg["judge_models"][0][field] = value
    # Full config contributes to each deterministic judge request ID. Even a
    # setting that does not change this fixture's capped request length changes
    # the declared frozen configuration and must not relabel the old results.
    with pytest.raises(ValueError):
        calibration_report(runner, directory)
    assert not (directory / "judge_approval.hashes.json").exists()


def test_report_rejects_a_stored_request_for_a_different_model(runner):
    """The submitted request must agree with the recorded plan/configuration."""
    directory, selection = panel(runner)
    prepare_test_judges(runner, directory)
    fill_test_labels(directory, selection)
    complete_test_judges(runner)
    row = runner.store.rows("judge")[0]
    request = runner.store.unpack_request(row["request"])
    request["model"] = "different-submitted-model-version"
    runner.store.update(row["id"], "succeeded", request=runner.store.pack_request(request))
    with pytest.raises(ValueError):
        calibration_report(runner, directory)
    assert not (directory / "judge_approval.hashes.json").exists()


def test_approval_rejects_report_configuration_mismatch_even_when_report_hash_matches(runner):
    """Approval must bind the report's judge basis as well as its file digest."""
    directory, selection = panel(runner)
    prepare_test_judges(runner, directory)
    fill_test_labels(directory, selection)
    complete_test_judges(runner)
    heldout = dict(calibration_report(runner, directory)["heldout"])
    heldout["judge_configuration_sha256"] = "a" * 64
    approval = json.loads((directory / "judge_approval.template.json").read_text())
    approval.update(
        decision="approved", approved_by="UNIT TEST FIXTURE, NOT A PERSON",
        approved_at_utc="2000-01-01T00:00:00Z", scope="UNIT TEST ONLY",
        decision_rationale="Exercise configuration binding, not scientific acceptance.",
        heldout_report_sha256=digest(heldout),
    )
    with pytest.raises(ValueError):
        validate_judge_approval(
            approval, system_prompt=runner.cfg["judge_system"],
            judge_config=runner.cfg["judge_models"][0],
            corpus_digest=runner.dataset.fingerprint, heldout_report=heldout,
        )


def test_approval_cannot_describe_single_expert_reference_as_two_expert_adjudication(runner):
    directory, selection = panel(runner)
    prepare_test_judges(runner, directory)
    fill_test_labels(directory, selection)
    complete_test_judges(runner)
    heldout = calibration_report(runner, directory)["heldout"]
    approval = json.loads((directory / "judge_approval.template.json").read_text())
    approval.update(
        decision="approved", approved_by="UNIT TEST FIXTURE, NOT A PERSON",
        approved_at_utc="2000-01-01T00:00:00Z", scope="UNIT TEST ONLY",
        decision_rationale="Exercise reference-basis binding, not scientific acceptance.",
        heldout_report_sha256=digest(heldout),
        human_review={"reviewers": 2, "reference_type": "two_expert_adjudicated",
                      "reference_file": "adjudicated.csv"},
    )
    with pytest.raises(ValueError):
        validate_judge_approval(
            approval, system_prompt=runner.cfg["judge_system"],
            judge_config=runner.cfg["judge_models"][0],
            corpus_digest=runner.dataset.fingerprint, heldout_report=heldout,
        )
