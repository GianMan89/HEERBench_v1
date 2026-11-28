"""Reproducible, offline cost scenarios for generation and blinded judging.

These are planning estimates, not billed usage or model outputs. Every judge
prompt uses the same source passages, rubric and JSON schema as live scoring.
Candidate text is a length placeholder and is never sent to an API or scored.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from decimal import Decimal
import math
from typing import Any, Iterable, Mapping

from .costs import CostResult, estimate_request_cost
from .judge import build_judge_request
from .providers import GenerationRequest
from .util import canonical, estimated_tokens


def execution_mode(cfg: Mapping[str, Any], model: Mapping[str, Any]) -> str:
    """Match the runner's native-batch capability, without inventing discounts."""
    native = model.get("kind") in {"openai_responses", "anthropic", "gemini"}
    return "batch" if cfg.get("execution", {}).get("transport") == "batch" and native else "standard"


def _cache_writes(request: GenerationRequest, model: Mapping[str, Any],
                  input_tokens: int) -> tuple[int, int]:
    """Assume a miss at every actual prefix breakpoint, including its preceding system text.

    Tiny prefixes may not satisfy a provider's cache minimum. Counting them as
    writes is deliberately conservative. Implicit Gemini caching has no write fee.
    The shipped compatible endpoints do not receive explicit cache directives.
    """
    kind = model.get("kind")
    cache = model.get("cache", {})
    mode = cache.get("mode", "provider_default" if kind == "openai_responses" else "off")
    amount = 0
    if kind in {"openai_responses", "anthropic"} and mode == "explicit" and request.prefix:
        amount = min(input_tokens, estimated_tokens(request.system + request.prefix) + 128)
    elif kind == "openai_responses" and mode in {"implicit", "provider_default"}:
        # Current native OpenAI adapter uses the GPT-5.6+ caching API. All input
        # could be written under its implicit final-message breakpoint.
        amount = input_tokens
    if not amount:
        return 0, 0
    ttl = cache.get("ttl", "30m" if kind == "openai_responses" else "5m")
    short, long = (("30m", "12h") if kind == "openai_responses" else ("5m", "1h"))
    if ttl not in {short, long}:
        raise ValueError(f"Unsupported cache TTL for cost estimate: {kind}/{ttl}")
    return (amount, 0) if ttl == short else (0, amount)


def estimate_generation_request_cost(
    request: GenerationRequest, model: Mapping[str, Any], mode: str,
    input_tokens: int, expected_output_tokens: int,
) -> CostResult:
    """Estimate one request with zero reads and all configured prefix writes.

    The name also covers judge requests. Their current empty prefix means
    explicit caching has no breakpoint and therefore incurs no write category.
    """
    if type(input_tokens) is not int or input_tokens < 0:
        raise ValueError("input_tokens must be a nonnegative integer")
    if type(expected_output_tokens) is not int or expected_output_tokens < 0:
        raise ValueError("expected_output_tokens must be a nonnegative integer")
    short, long = _cache_writes(request, model, input_tokens)
    return estimate_request_cost(
        input_tokens, min(expected_output_tokens, request.max_output_tokens),
        model.get("pricing", {}), mode,
        cache_write_short_tokens=short, cache_write_long_tokens=long,
    )


def _aggregate(unit: Mapping[str, dict], counts: Counter, model: Mapping[str, Any],
               mode: str, calls: int | None = None) -> dict:
    """Exact population sum or a stated-size forecast at its mean prompt cost."""
    population = sum(counts.values())
    if not population:
        raise ValueError("A cost forecast requires at least one planned answer")
    requested = population if calls is None else calls
    scale = Decimal(requested) / Decimal(population)
    known = sum((Decimal(unit[q]["cost"].known_subtotal_usd) * n for q, n in counts.items()), Decimal(0))
    complete = all(unit[q]["cost"].complete for q in counts)
    components = defaultdict(Decimal)
    for q, n in counts.items():
        for name, amount in unit[q]["cost"].components_usd.items():
            components[name] += Decimal(amount) * n
    issues = sorted({issue for q in counts for issue in unit[q]["cost"].unresolved})
    estimated_input = sum(unit[q]["input_tokens"] * n for q, n in counts.items())
    estimated_output = sum(unit[q]["output_tokens"] * n for q, n in counts.items())
    costs = [Decimal(unit[q]["cost"].total_usd) for q in counts if unit[q]["cost"].complete]
    return {
        "model_id": model["id"], "model": model["model"], "calls": requested,
        "transport": mode, "total_usd": str(known * scale) if complete else None,
        "known_subtotal_usd": str(known * scale), "pricing_complete": complete,
        "components_usd": {k: str(v * scale) for k, v in components.items()},
        "unresolved": issues,
        "estimated_input_tokens": str(Decimal(estimated_input) * scale),
        "estimated_output_tokens": str(Decimal(estimated_output) * scale),
        "input_tokens_per_call_range": [min(unit[q]["input_tokens"] for q in counts),
                                        max(unit[q]["input_tokens"] for q in counts)],
        "cost_envelope_usd": ([str(min(costs) * requested), str(max(costs) * requested)]
                              if complete else [None, None]),
        "basis": ("all_planned_prompts" if calls is None else "planned_prompt_mix_mean"),
        "output_clipped_to_cap_for_question_count": sum(unit[q]["output_clipped"] for q in counts),
    }


def forecast_judges(cfg, dataset, plan, expected_candidate_tokens=1000,
                    expected_judge_output_tokens=1200, *,
                    calibration_answers=60, calibration_repetitions=1) -> dict:
    """Forecast primary, secondary, and optional configurable calibration calls.

    The forecast assumes every planned generation returns an answer. The real
    secondary sample is drawn only from successful, non-mock answers. Its exact
    prompt mix is unknown in advance, so its cost uses the planned population
    mean and reports a min/max envelope. Calibration defaults to 60 answers
    with one primary-judge call each and is forecast separately; it is not
    included in the main scoring total. Explicit calibration counts support
    legacy or custom protocols. Neither calculation claims a validation panel
    already exists or that group separation is feasible.
    """
    for name, value in (("expected_candidate_tokens", expected_candidate_tokens),
                        ("expected_judge_output_tokens", expected_judge_output_tokens),
                        ("calibration_answers", calibration_answers),
                        ("calibration_repetitions", calibration_repetitions)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not plan:
        raise ValueError("Cannot forecast judges for an empty generation plan")
    counts = Counter(row["question_id"] for row in plan)
    models = {m["id"]: m for m in cfg["judge_models"]}
    primary_id = cfg["judge"]["primary"]
    secondary_id = cfg["judge"].get("secondary")
    fraction = cfg["judge"].get("secondary_fraction", 0.2)
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("secondary_fraction must be finite and in [0,1]")
    if cfg["judge"].get("secondary_sampling_unit") == "model_repetition":
        group_counts = Counter((r["model_id"], r["repetition"]) for r in plan)
        secondary_calls = sum(math.ceil(fraction * n) for n in group_counts.values())
    else:
        secondary_calls = math.ceil(fraction * len(plan))
    if secondary_calls and secondary_id is None:
        raise ValueError("A positive secondary fraction requires a configured secondary judge")
    for model_id in [primary_id] + ([secondary_id] if secondary_calls else []):
        if model_id not in models:
            raise ValueError(f"Judge configuration is missing: {model_id}")

    # Byte-count surrogate only. This content is never returned as a generated
    # answer, used for judging or persisted in an experiment's response journal.
    candidate = "x" * (3 * expected_candidate_tokens)
    contracts = {}
    for qid in counts:
        contracts[qid] = build_judge_request(
            dataset.by_id[qid], candidate, dataset.for_question(dataset.sources, qid),
            dataset.for_question(dataset.errors, qid), dataset.for_question(dataset.local_errors, qid),
            system_prompt=cfg.get("judge_system"),
        )

    units = {}
    for model_id in [primary_id] + ([secondary_id] if secondary_calls and secondary_id != primary_id else []):
        model = models[model_id]
        mode = execution_mode(cfg, model)
        units[model_id] = {}
        for qid, contract in contracts.items():
            cap = min(model.get("max_output_tokens", 16384),
                      max(4096, 256 * (len(contract.criterion_ids) + len(contract.error_ids)) + 512))
            request = GenerationRequest(
                "forecast-only-" + qid, model["model"], contract.system, "",
                canonical({"evaluation": contract.payload, "required_response_schema": contract.response_schema}),
                max_output_tokens=cap,
            )
            tokens = estimated_tokens(request.system + request.task) + 256
            units[model_id][qid] = {
                "cost": estimate_generation_request_cost(request, model, mode, tokens, expected_judge_output_tokens),
                "input_tokens": tokens, "output_tokens": min(expected_judge_output_tokens, cap),
                "output_clipped": expected_judge_output_tokens > cap,
            }
    primary = _aggregate(units[primary_id], counts, models[primary_id], execution_mode(cfg, models[primary_id]))
    secondary = (_aggregate(units[secondary_id], counts, models[secondary_id],
                            execution_mode(cfg, models[secondary_id]), secondary_calls)
                 if secondary_calls else {"calls": 0, "total_usd": "0", "known_subtotal_usd": "0", "pricing_complete": True,
                                          "basis": "disabled", "model_id": secondary_id})
    calibration = _aggregate(units[primary_id], counts, models[primary_id],
                             execution_mode(cfg, models[primary_id]),
                             calibration_answers * calibration_repetitions)
    calibration.update({"optional": True, "included_in_scoring_total": False,
                        "answers": calibration_answers, "judge_repetitions": calibration_repetitions,
                        "enough_planned_answers": len(plan) >= calibration_answers,
                        "group_separation_validated": False})
    per_repetition = {}
    for repetition in sorted({row["repetition"] for row in plan}):
        subset = Counter(row["question_id"] for row in plan if row["repetition"] == repetition)
        part = _aggregate(units[primary_id], subset, models[primary_id], execution_mode(cfg, models[primary_id]))
        # Global secondary sampling need not allocate exactly 20% to each repeat.
        secondary_weight = Decimal(sum(subset.values())) / Decimal(len(plan))
        secondary_cost = (Decimal(secondary["total_usd"]) * secondary_weight
                          if secondary["pricing_complete"] else None)
        per_repetition[str(repetition)] = {
            "primary_calls": sum(subset.values()), "primary_usd": part["total_usd"],
            "secondary_usd_proportional_allocation": str(secondary_cost) if secondary_cost is not None else None,
            "total_usd": (str(Decimal(part["total_usd"]) + secondary_cost)
                          if part["pricing_complete"] and secondary_cost is not None else None),
        }
    complete = primary["pricing_complete"] and secondary["pricing_complete"]
    total = (Decimal(primary["total_usd"]) + Decimal(secondary["total_usd"])) if complete else None
    return {
        "primary": primary, "secondary": secondary, "optional_calibration": calibration,
        "scoring_total_usd": str(total) if total is not None else None,
        "scoring_known_subtotal_usd": str(Decimal(primary["known_subtotal_usd"]) + Decimal(secondary["known_subtotal_usd"])),
        "scoring_pricing_complete": complete, "per_generation_repetition": per_repetition,
        "expected_candidate_tokens": expected_candidate_tokens,
        "expected_judge_output_tokens": expected_judge_output_tokens,
        "assumptions": [
            "Offline UTF-8 bytes/3 estimates with 256 tokens of request overhead; no tokenizer claim.",
            "Real question, reference, alternatives, all criterion/error checks, validated source excerpts and response schema are included.",
            "Every planned generation is assumed successful; actual scoring excludes transport failures and mock answers.",
            "Candidate text is a length surrogate only; requested output tokens include billed reasoning.",
            "No cache hits are assumed. Explicit writes are included only when the actual request contains a cache breakpoint.",
            "Secondary and optional calibration costs use the planned prompt mix; min/max envelopes do not bound future answer lengths.",
            f"Optional calibration is {calibration_answers} answers times {calibration_repetitions} primary-judge call(s), additional to scoring; group feasibility is not asserted.",
            "Pilot measurements replace estimates. Rates, retries, tax, hosting, tools and any storage charges require reconciliation.",
        ],
    }


def measured_charge_completeness(rows: Iterable[Mapping[str, Any]]) -> dict:
    """Sum observed list-price estimates while exposing unaccounted paid attempts.

    Accepts the store's decoded rows (result.cost, attempts, state). The result is
    not an invoice. A failed or uncertain attempted call without usage, and any
    earlier retry not costed in the final response, remains unquantified.
    Pending/unattempted blocked work is not claimed to incur a charge.
    """
    known = Decimal(0)
    accounted = 0
    incomplete_ids = []
    unresolved_attempt_ids = []
    attempt_count = 0
    for row in rows:
        attempts = int(row.get("attempts", 0))
        attempt_count += attempts
        result = row.get("result") or {}
        cost = result.get("cost")
        identity = row.get("id", row.get("request_id", "unknown"))
        if cost:
            known += Decimal(cost.get("known_subtotal_usd", "0"))
            accounted += 1
            if not cost.get("complete", False):
                incomplete_ids.append(identity)
        elif attempts:
            incomplete_ids.append(identity)
        if attempts > (1 if cost else 0) or row.get("state") == "uncertain":
            unresolved_attempt_ids.append(identity)
    complete = not incomplete_ids and not unresolved_attempt_ids
    return {"known_subtotal_usd": str(known), "accounted_response_count": accounted,
            "attempt_count": attempt_count, "complete_for_recorded_attempts": complete,
            "total_usd": str(known) if complete else None,
            "incomplete_usage_or_price_request_ids": incomplete_ids,
            "unquantified_attempt_request_ids": unresolved_attempt_ids,
            "basis": "observed_tokens_list_prices_not_invoice",
            "note": "No cost is fabricated for failed, retried or uncertain calls without usable usage; invoice reconciliation may add charges."}
