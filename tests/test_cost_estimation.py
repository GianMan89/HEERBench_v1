"""Cost forecasts must include rubric/source tokens and never price unknown calls at zero."""
from copy import deepcopy
from decimal import Decimal
from types import SimpleNamespace

import pytest

from heerbench.cost_estimation import (estimate_generation_request_cost, forecast_judges,
                                       measured_charge_completeness)
from heerbench.providers import GenerationRequest


def fixture():
    qid = "C1-A1"
    question = {"question_id": qid, "case_id": "C1", "abf_stage": "A",
                "question": "Which temperature is forecast?", "expected_answer": "Minus 20 C.",
                "acceptable_alternatives": ["Equivalent numerical notation."],
                "criteria": [{"criterion_id": "C1-A1-R1", "text": "States minus 20 C.", "required": True}],
                "error_conditions": ["Asserts plus 20 C."]}
    ds = SimpleNamespace(by_id={qid: question}, sources=[{
        "question_id": qid, "source_link_id": "C1-A1-S1", "doc_id": "D1",
        "document_title": "Exercise forecast", "line_start": 1, "line_end": 1,
        "source_excerpt": "Forecast: minus 20 C."}], errors=[{
            "question_id": qid, "error_link_id": "C1-A1-N1", "trigger": "Reverses temperature sign.",
            "applicability": "Forecast reporting."}], local_errors=[{
                "question_id": qid, "local_error_id": "C1-A1-E1", "text": "Asserts plus 20 C."}])
    ds.for_question = lambda records, key: [r for r in records if r["question_id"] == key]
    rates = {"input": "2", "output": "10", "cache_write_short": "2.5", "cache_write_long": "5"}
    cfg = {"seed": 4, "execution": {"transport": "batch"},
           "judge": {"primary": "primary", "secondary": "secondary", "secondary_fraction": .2},
           "judge_models": [
               {"id": "primary", "model": "fixed-primary", "kind": "openai_responses", "max_output_tokens": 10000,
                "cache": {"mode": "explicit", "ttl": "30m"},
                "pricing": {"standard": rates, "batch": {k: str(Decimal(v)/2) for k, v in rates.items()}}},
               {"id": "secondary", "model": "fixed-secondary", "kind": "anthropic", "max_output_tokens": 10000,
                "cache": {"mode": "explicit", "ttl": "5m"}, "pricing": {"standard": rates, "batch": rates}},
           ]}
    plan = [{"request_id": f"r{i}", "question_id": qid, "model_id": "generator",
             "case_id": "C1", "abf_stage": "A", "condition": "E1", "repetition": 1} for i in range(6)]
    return cfg, ds, plan


def test_complete_forecast_counts_batch_rates_and_separate_calibration():
    cfg, ds, plan = fixture()
    result = forecast_judges(cfg, ds, plan)
    assert result["primary"]["calls"] == 6
    assert result["secondary"]["calls"] == 2  # ceil(6 * .2)
    assert result["optional_calibration"]["calls"] == 60
    assert result["optional_calibration"]["answers"] == 60
    assert result["optional_calibration"]["judge_repetitions"] == 1
    assert result["optional_calibration"]["enough_planned_answers"] is False
    assert result["primary"]["transport"] == "batch"
    # Judge requests have no stable prefix breakpoint even though their model
    # configuration supports explicit caching. They pay ordinary input prices.
    assert result["primary"]["components_usd"]["cache_write_short"] == "0"
    expected = Decimal(result["primary"]["total_usd"]) + Decimal(result["secondary"]["total_usd"])
    assert Decimal(result["scoring_total_usd"]) == expected
    assert Decimal(result["optional_calibration"]["total_usd"]) == 10 * Decimal(result["primary"]["total_usd"])
    cfg["execution"]["transport"] = "sync"
    standard = forecast_judges(cfg, ds, plan)
    assert Decimal(standard["primary"]["total_usd"]) == 2 * Decimal(result["primary"]["total_usd"])


def test_custom_calibration_counts_preserve_ordinary_costs_and_legacy_forecast():
    cfg, ds, plan = fixture()
    # The two existing optional token counts remain accepted positionally.
    default = forecast_judges(cfg, ds, plan, 1000, 1200)
    legacy = forecast_judges(cfg, ds, plan, 1000, 1200,
                             calibration_answers=200, calibration_repetitions=3)
    for key in ("primary", "secondary", "scoring_total_usd", "scoring_known_subtotal_usd",
                "scoring_pricing_complete", "per_generation_repetition"):
        assert legacy[key] == default[key]
    calibration = legacy["optional_calibration"]
    assert calibration["calls"] == 600
    assert calibration["answers"] == 200
    assert calibration["judge_repetitions"] == 3
    assert calibration["included_in_scoring_total"] is False
    assert Decimal(calibration["total_usd"]) == 10 * Decimal(default["optional_calibration"]["total_usd"])
    custom = forecast_judges(cfg, ds, plan, calibration_answers=6, calibration_repetitions=2)
    assert custom["optional_calibration"]["calls"] == 12
    assert custom["optional_calibration"]["enough_planned_answers"] is True
    assert custom["optional_calibration"]["group_separation_validated"] is False


@pytest.mark.parametrize("name", ["calibration_answers", "calibration_repetitions"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5, "2"])
def test_invalid_calibration_counts_are_rejected(name, value):
    cfg, ds, plan = fixture()
    with pytest.raises(ValueError, match=name + " must be a positive integer"):
        forecast_judges(cfg, ds, plan, **{name: value})


@pytest.mark.parametrize("flags, expected_answers, expected_repetitions", [
    ([], 60, 1),
    (["--calibration-answers", "200", "--calibration-repetitions", "3"], 200, 3),
])
def test_estimate_cli_forwards_calibration_protocol(monkeypatch, tmp_path,
                                                   flags, expected_answers, expected_repetitions):
    from heerbench import cli
    cfg, ds, plan = fixture()
    monkeypatch.setattr(cli, "estimate_run", lambda path, expected: {"generation_marker": expected})
    monkeypatch.setattr(cli, "load_run", lambda path: (None, cfg, ds, plan))
    args = cli.parser().parse_args(["estimate", str(tmp_path), *flags])
    result = cli._estimate(tmp_path, args)
    calibration = result["judging"]["optional_calibration"]
    assert calibration["answers"] == expected_answers
    assert calibration["judge_repetitions"] == expected_repetitions
    assert calibration["calls"] == expected_answers * expected_repetitions
    assert result["generation_marker"] == 1000
    assert (tmp_path / "cost_estimate.json").exists()


def test_source_rubric_and_candidate_length_affect_judge_forecast():
    cfg, ds, plan = fixture()
    base = forecast_judges(cfg, ds, plan, expected_candidate_tokens=100)
    longer_candidate = forecast_judges(cfg, ds, plan, expected_candidate_tokens=200)
    assert Decimal(longer_candidate["primary"]["estimated_input_tokens"]) > Decimal(base["primary"]["estimated_input_tokens"])
    ds.sources[0]["source_excerpt"] += " Additional validated source material." * 100
    ds.by_id["C1-A1"]["criteria"].append({"criterion_id": "C1-A1-R2", "text": "Preserves forecast qualification.", "required": True})
    expanded = forecast_judges(cfg, ds, plan, expected_candidate_tokens=100)
    assert Decimal(expanded["primary"]["total_usd"]) > Decimal(base["primary"]["total_usd"])


def test_unknown_rates_remain_unknown_and_no_batch_discount_for_compatible_api():
    cfg, ds, plan = fixture()
    cfg["judge_models"][0]["pricing"] = {}
    result = forecast_judges(cfg, ds, plan)
    assert result["primary"]["total_usd"] is None
    assert result["scoring_total_usd"] is None
    assert result["optional_calibration"]["total_usd"] is None
    assert "input_rate_unknown" in result["primary"]["unresolved"]
    cfg, ds, plan = fixture()
    cfg["judge_models"][0].update(kind="openai_compatible", cache={"mode": "off"})
    result = forecast_judges(cfg, ds, plan)
    assert result["primary"]["transport"] == "standard"


def test_generation_cache_misses_pay_write_rate_instead_of_ordinary_input():
    cfg, _, _ = fixture()
    model = cfg["judge_models"][0]
    req = GenerationRequest("r", model["model"], "s", "scenario" * 1000, "question", max_output_tokens=2000)
    write = estimate_generation_request_cost(req, model, "standard", 3000, 1000)
    off = deepcopy(model)
    off["cache"]["mode"] = "off"
    plain = estimate_generation_request_cost(req, off, "standard", 3000, 1000)
    assert Decimal(write.components_usd["cache_write_short"]) > 0
    assert Decimal(write.total_usd) > Decimal(plain.total_usd)
    model["cache"]["ttl"] = "12h"
    long_write = estimate_generation_request_cost(req, model, "standard", 3000, 1000)
    assert Decimal(long_write.components_usd["cache_write_long"]) > 0
    assert Decimal(long_write.total_usd) > Decimal(write.total_usd)
    model["pricing"]["standard"].pop("cache_write_long")
    assert estimate_generation_request_cost(req, model, "standard", 3000, 1000).total_usd is None


def test_observed_subtotal_does_not_hide_uncertain_or_retried_attempts():
    cost = {"known_subtotal_usd": ".03", "total_usd": ".03", "complete": True}
    succeeded = {"id": "ok", "state": "succeeded", "attempts": 1, "result": {"cost": cost}}
    assert measured_charge_completeness([succeeded])["total_usd"] == "0.03"
    uncertain = {"id": "unknown", "state": "uncertain", "attempts": 1, "result": None}
    result = measured_charge_completeness([succeeded, uncertain])
    assert result["total_usd"] is None
    assert result["known_subtotal_usd"] == "0.03"
    assert "unknown" in result["unquantified_attempt_request_ids"]
    retried = {**succeeded, "attempts": 2}
    assert measured_charge_completeness([retried])["complete_for_recorded_attempts"] is False
    pending = {"id": "pending", "state": "pending", "attempts": 0, "result": None}
    assert measured_charge_completeness([succeeded, pending])["total_usd"] == "0.03"


def test_invalid_counts_are_rejected():
    cfg, ds, plan = fixture()
    with pytest.raises(ValueError):
        forecast_judges(cfg, ds, plan, expected_candidate_tokens=0)
    with pytest.raises(ValueError):
        forecast_judges(cfg, ds, [])
