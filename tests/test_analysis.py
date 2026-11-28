"""Tests target denominator, graph-dependency and statistical-unit failures."""
import json
import math
from pathlib import Path

import pytest

from heerbench.analysis import AnalysisError, analyse, join_records, paired_cluster_bootstrap
from heerbench.reporting import markdown_report, write_report


def fixture():
    questions = [
        {"question_id": "a1", "case_id": "C1", "abf_stage": "A"},
        {"question_id": "a2", "case_id": "C1", "abf_stage": "A"},
        {"question_id": "b1", "case_id": "C1", "abf_stage": "B"},
        {"question_id": "f1", "case_id": "C1", "abf_stage": "F"},
        {"question_id": "f2", "case_id": "C1", "abf_stage": "F"},
    ]
    edges = [
        {"dependency_id": "e1", "case_id": "C1", "parent_ids": ["a1", "a2"], "child_id": "b1", "logic": "ALL"},
        {"dependency_id": "e2", "case_id": "C1", "parent_ids": ["b1"], "child_id": "f1", "logic": "ALL"},
        {"dependency_id": "e3", "case_id": "C1", "parent_ids": ["b1"], "child_id": "f2", "logic": "ALL"},
    ]
    chains = [
        {"chain_id": "ch1", "case_id": "C1", "a_ids": ["a1", "a2"], "b_ids": ["b1"], "f_ids": ["f1"]},
        {"chain_id": "ch2", "case_id": "C1", "a_ids": ["a1", "a2"], "b_ids": ["b1"], "f_ids": ["f2"]},
    ]
    plan, results = [], []
    for condition in ("E1", "E2", "E3"):
        for q in questions:
            if q["abf_stage"] == "A" and condition != "E1":
                continue
            rid = condition + "_" + q["question_id"]
            p = {**q, "request_id": rid, "model_id": "test-model", "condition": condition, "repetition": 1}
            plan.append(p)
            results.append({"request_id": rid, "generation_status": "succeeded", "judge_status": "succeeded",
                            "criterion_score": 1.0, "fully_correct": True, "has_error": False,
                            "negative_error_rate": None, "scoreable": True})
    return questions, edges, chains, plan, results


def test_shared_questions_count_once_and_joint_parent_error_is_not_causal_propagation():
    q, edges, chains, plan, results = fixture()
    by = {r["request_id"]: r for r in results}
    by["E1_a2"].update(criterion_score=0.0, fully_correct=False)
    by["E3_b1"].update(criterion_score=0.0, fully_correct=False)
    report = analyse(plan, results, q, chains, edges, bootstrap=10)
    assert report["metadata"]["planned_generation_requests"] == 11
    assert len(report["question_scores"]) == 11
    assert len(report["chain_scores"]) == 6
    e = next(x for x in report["transitions"] if x["dependency_id"] == "e1")
    assert e["parent_count"] == 2
    assert e["parent_set_correct"] is False
    assert e["transition"] == "incorrect_to_incorrect_association"
    assert next(x for x in report["transitions"] if x["dependency_id"] == "e2")["transition"] == "incorrect_to_correct"
    # Oracle F uses the expert B, but its joint-task score still contains the model A.
    oracle = next(x for x in report["chain_scores"] if x["condition"] == "E2")
    assert oracle["success_lower_bound"] == 0
    assert oracle["metric_label"] == "oracle_assisted_joint_task_success"
    assert "not an end-to-end" in oracle["oracle_caveat"]


def test_execution_failure_remains_in_denominator_and_is_not_content_error():
    q, edges, chains, plan, results = fixture()
    results = [r for r in results if r["request_id"] != "E3_f2"]
    report = analyse(plan, results, q, chains, edges, bootstrap=0)
    coverage = next(x for x in report["coverage"] if x["condition"] == "E3")
    assert coverage["planned"] == 3
    assert coverage["scored"] == 2
    assert coverage["criterion_score_observed"] == 1.0
    assert coverage["criterion_score_lower_bound"] == pytest.approx(2 / 3)
    assert coverage["any_error_observed"] == 0.0
    assert coverage["generation_pending"] == 1
    chain = next(x for x in report["chain_scores"] if x["condition"] == "E3" and x["chain_id"] == "ch2")
    assert chain["success_observed"] is None
    assert chain["success_lower_bound"] == 0


def test_scenario_macro_does_not_give_large_scenario_extra_weight():
    questions, plan, results = [], [], []
    for case, n, score in [("C1", 1, 1.0), ("C2", 9, 0.0)]:
        for j in range(n):
            q = {"question_id": f"{case}a{j}", "case_id": case, "abf_stage": "A"}
            questions.append(q)
            rid = q["question_id"]
            plan.append({**q, "request_id": rid, "model_id": "m", "condition": "E1", "repetition": 1})
            results.append({"request_id": rid, "generation_status": "succeeded", "judge_status": "succeeded", "criterion_score": score, "fully_correct": bool(score), "has_error": False})
    report = analyse(plan, results, questions, [], [], bootstrap=0)
    assert report["stage_scores"][0]["criterion_score_lower_bound"] == 0.5
    assert report["coverage"][0]["criterion_score_lower_bound"] == 0.1


def test_unscorable_judgement_has_no_observed_score():
    q, _, _, plan, results = fixture()
    r = next(r for r in results if r["request_id"] == "E1_a1")
    r.update(judge_status="unscorable", scoreable=False, criterion_score=None, fully_correct=None, has_error=None)
    joined, _ = join_records(plan, results, q)
    r = next(r for r in joined if r["request_id"] == "E1_a1")
    assert r["criterion_score"] is None and r["criterion_score_lower_bound"] == 0
    assert r["has_error"] is None


def test_secondary_judge_is_not_pooled_as_another_run():
    q, _, _, plan, results = fixture()
    results.append({**results[0], "judge_role": "secondary"})
    joined, ignored = join_records(plan, results, q)
    assert len(joined) == len(plan)
    assert ignored == 1


def test_duplicates_invalid_scoring_and_missing_parent_fail_loudly():
    q, edges, chains, plan, results = fixture()
    with pytest.raises(AnalysisError, match="Duplicate primary"):
        join_records(plan, results + [results[0]], q)
    broken = [dict(r) for r in results]
    broken[0]["criterion_score"] = 1.5
    with pytest.raises(AnalysisError, match="finite and in"):
        join_records(plan, broken, q)
    broken[0].update(criterion_score=0.5, fully_correct=True)
    with pytest.raises(AnalysisError, match="Contradictory"):
        join_records(plan, broken, q)
    pp = [p for p in plan if p["request_id"] != "E1_a2"]
    rr = [r for r in results if r["request_id"] != "E1_a2"]
    with pytest.raises(AnalysisError, match="lacks a planned parent"):
        analyse(pp, rr, q, [], edges, bootstrap=0)


def test_paired_bootstrap_is_reproducible_and_keeps_pairing():
    values = {"C1": {1: 0.2, 2: 0.2, 3: 0.2}, "C2": {1: 0.2, 2: 0.2, 3: 0.2}}
    a = paired_cluster_bootstrap(values, n_resamples=200, seed=42)
    b = paired_cluster_bootstrap(values, n_resamples=200, seed=42)
    assert a == b
    assert a["estimate"] == pytest.approx(0.2)
    assert a["ci_low"] == pytest.approx(0.2) and a["ci_high"] == pytest.approx(0.2)
    assert paired_cluster_bootstrap({"C1": {1: 0.4}}, n_resamples=200)["ci_low"] is None
    with pytest.raises(AnalysisError, match="nonnegative integer"):
        paired_cluster_bootstrap(values, n_resamples=2.5)


def test_complete_case_bf_requires_both_stages_and_counts_only_used_pairs():
    q, edges, chains, plan, results = fixture()
    results = [r for r in results if r["request_id"] != "E3_b1"]
    report = analyse(plan, results, q, chains, edges, bootstrap=0)
    conditional = next(r for r in report["paired_effects"] if r["contrast"] == "oracle_minus_sequential" and r["stage"] == "BF" and r["metric"] == "criterion_score")
    assert conditional["planned_pairs"] == 3
    assert conditional["used_pairs"] == 0
    assert conditional["estimate"] is None
    conservative = next(r for r in report["paired_effects"] if r["contrast"] == "oracle_minus_sequential" and r["stage"] == "BF" and r["metric"] == "criterion_score_lower_bound")
    assert conservative["used_pairs"] == 3
    assert conservative["estimate"] == 0.5


def test_real_repetition_variation_and_bf_pairing():
    q, edges, chains, base_plan, base_results = fixture()
    plan, results = [], []
    for rep in (1, 2, 3):
        for p in base_plan:
            plan.append({**p, "request_id": p["request_id"] + f"_r{rep}", "repetition": rep})
        for r in base_results:
            rr = {**r, "request_id": r["request_id"] + f"_r{rep}"}
            if r["request_id"].startswith("E3_"):
                score = [0.0, 0.5, 1.0][rep - 1]
                rr.update(criterion_score=score, fully_correct=score == 1)
            results.append(rr)
    report = analyse(plan, results, q, chains, edges, bootstrap=200, seed=12)
    stage = next(r for r in report["stage_scores"] if r["condition"] == "E3" and r["stage"] == "BF")
    assert stage["criterion_score_lower_bound"] == 0.5
    assert stage["criterion_score_lower_bound_run_sd"] == 0.5
    paired = next(r for r in report["paired_effects"] if r["contrast"] == "oracle_minus_sequential" and r["stage"] == "BF" and r["metric"] == "criterion_score_lower_bound")
    assert paired["estimate"] == 0.5
    assert paired["used_pairs"] == 9


def test_report_labels_mock_data_and_preserves_nulls(tmp_path: Path):
    q, edges, chains, plan, results = fixture()
    report = analyse(plan, results, q, chains, edges, bootstrap=0, is_mock=True)
    files = write_report(report, tmp_path)
    assert "MOCK" in Path(files["report_markdown"]).read_text()
    assert "not measurements" in Path(files["report_markdown"]).read_text()
    stored = json.loads(Path(files["analysis_json"]).read_text())
    assert stored["metadata"]["is_mock"] is True
    assert all(Path(f).exists() for f in files.values())
    assert "NaN" not in Path(files["analysis_json"]).read_text()


def test_real_pilot_report_does_not_claim_validated_judge_results():
    q, edges, chains, plan, results = fixture()
    report = analyse(plan, results, q, chains, edges, bootstrap=0)
    report["judge_validated_for_publication"] = False
    text = markdown_report(report)
    assert "JUDGE CALIBRATION NOT YET ACCEPTED" in text
    assert "full unchanged scenario packet" in text
    assert "not itself a bound" in text
    report["judge_validated_for_publication"] = True
    text = markdown_report(report)
    assert "JUDGE CALIBRATION NOT YET ACCEPTED" not in text
    assert "does not assert a human performance baseline" in text
    assert "no reviewer count inferred" in text


@pytest.mark.parametrize("reviewers,reference_type,expected", [
    (1, "single_expert", "One military expert, single-expert reference"),
    (2, "two_expert_adjudicated", "Two experts, adjudicated reference"),
])
def test_report_discloses_recorded_human_reference_basis(reviewers, reference_type, expected):
    q, edges, chains, plan, results = fixture()
    report = analyse(plan, results, q, chains, edges, bootstrap=0)
    report["judge_validated_for_publication"] = True
    report["judge_validation_basis"] = {"reviewers": reviewers, "reference_type": reference_type,
                                         "reference_file": "reviewer_1.csv" if reviewers == 1 else "adjudicated.csv"}
    text = markdown_report(report)
    assert expected in text
    assert "Accepted calibration records the expert decision" in text
    assert "does not certify scientific adequacy" in text
    if reviewers == 1:
        assert "no inter-human agreement or adjudicated consensus" in text
