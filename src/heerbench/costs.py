"""Decimal token-cost accounting. Rates are explicit for each mode and cache channel.

All monetary inputs are USD per million tokens except storage, which is USD per
million token-hours. No generic batch multiplier or assumed cache-hit fraction
is applied. Missing usage/rates produce an incomplete cost, never a free call.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping

from .providers.base import TokenUsage

MILLION = Decimal("1000000")


def _decimal(value: Any) -> Decimal:
    try:
        result = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError("Invalid cost value") from exc
    if not result.is_finite() or result < 0:
        raise ValueError("Cost values must be finite and nonnegative")
    return result


@dataclass
class CostResult:
    total_usd: str | None
    known_subtotal_usd: str
    complete: bool
    basis: str
    execution_mode: str
    components_usd: dict[str, str] = field(default_factory=dict)
    unresolved: list[str] = field(default_factory=list)
    rates_usd_per_million: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def select_rates(pricing: Mapping[str, Any], execution_mode: str,
                 input_tokens: int | None) -> dict[str, Any]:
    """Select a separately listed mode/tier, never synthesize a batch discount."""
    if execution_mode not in {"standard", "batch"}:
        raise ValueError("execution_mode must be standard or batch")
    mode_rates = pricing.get(execution_mode)
    if mode_rates is None:
        return {}
    if not isinstance(mode_rates, Mapping):
        raise ValueError("Pricing mode must contain a mapping")
    rates = dict(mode_rates)
    tiers = rates.pop("tiers", None)
    if tiers:
        if input_tokens is None:
            return {}
        for tier in tiers:
            upper = tier.get("max_input_tokens")
            if upper is None or input_tokens <= upper:
                return {**rates, **{key: value for key, value in tier.items() if key != "max_input_tokens"}}
        return {}
    return rates


def cost_from_usage(usage: TokenUsage | Mapping[str, Any], pricing: Mapping[str, Any],
                    execution_mode: str = "standard", *, measured: bool = True) -> CostResult:
    """Compute a list-price estimate from observed usage, not an invoice amount.

    The provider-reported cost (when present) remains in usage.raw and should be
    reconciled separately against the final invoice. Unknown cache accounting
    prevents claiming a complete observed cost even when plain-token totals exist.
    """
    if not isinstance(usage, TokenUsage):
        usage = TokenUsage(**{k: v for k, v in dict(usage).items() if k in TokenUsage.__dataclass_fields__})
    rates = select_rates(pricing, execution_mode, usage.input_tokens)
    channels = {"input": usage.uncached_input_tokens, "output": usage.output_tokens,
                "cache_read": usage.cached_input_tokens,
                "cache_write_short": usage.cache_write_short_tokens,
                "cache_write_long": usage.cache_write_long_tokens,
                "cache_write_unspecified": usage.cache_write_unspecified_tokens}
    components: dict[str, str] = {}
    unresolved = []
    total = Decimal(0)
    for name, tokens in channels.items():
        if name != "output" and measured and not usage.cache_accounting_known:
            # The aggregate prompt count cannot be assigned a definite input
            # price if unreported cache reads/writes may use different rates.
            continue
        if tokens is None:
            unresolved.append(name + "_tokens_unknown")
            continue
        if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 0:
            raise ValueError("Token counts must be nonnegative integers")
        if tokens == 0:
            components[name] = "0"
            continue
        rate = rates.get(name)
        if rate is None:
            unresolved.append(name + "_rate_unknown")
            continue
        amount = Decimal(tokens) * _decimal(rate) / MILLION
        components[name] = str(amount)
        total += amount
    if measured and not usage.cache_accounting_known:
        unresolved.append("cache_accounting_unknown")
    return CostResult(total_usd=str(total) if not unresolved else None,
                      known_subtotal_usd=str(total), complete=not unresolved,
                      basis="observed_tokens_list_prices" if measured else "token_estimate_list_prices",
                      execution_mode=execution_mode, components_usd=components,
                      unresolved=unresolved, rates_usd_per_million=rates)


def estimate_request_cost(input_tokens: int, output_tokens: int, pricing: Mapping[str, Any],
                          execution_mode: str = "standard", *, cached_input_tokens: int = 0,
                          cache_write_short_tokens: int = 0, cache_write_long_tokens: int = 0) -> CostResult:
    """Scenario estimate. Caller must state estimated counts and cache assumptions."""
    usage = TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens,
                       cached_input_tokens=cached_input_tokens,
                       cache_write_short_tokens=cache_write_short_tokens,
                       cache_write_long_tokens=cache_write_long_tokens,
                       cache_accounting_known=True)
    return cost_from_usage(usage, pricing, execution_mode, measured=False)


def conservative_request_reservation(input_tokens_upper: int, output_tokens_cap: int,
                                     pricing: Mapping[str, Any], execution_mode: str = "standard") -> CostResult:
    """Reserve all input at the highest listed input/cache-write rate.

    This is conditional on the supplied token upper bound and verified rate card.
    It does not assume a cache hit, or account for separately purchased tools,
    regional multipliers, storage, retries or provider charges outside this run.
    """
    if input_tokens_upper < 0 or output_tokens_cap < 0:
        raise ValueError("Negative token reservation")
    rates = select_rates(pricing, execution_mode, input_tokens_upper)
    available = [rates[k] for k in ("input", "cache_write_short", "cache_write_long",
                                   "cache_write_unspecified") if rates.get(k) is not None]
    if not available or rates.get("output") is None:
        return CostResult(None, "0", False, "conservative_reservation", execution_mode,
                          unresolved=["reservation_rates_unknown"], rates_usd_per_million=rates)
    upper_rates = {execution_mode: {"input": str(max(_decimal(value) for value in available)),
                                    "output": rates["output"]}}
    result = estimate_request_cost(input_tokens_upper, output_tokens_cap, upper_rates, execution_mode)
    result.basis = "conservative_reservation"
    return result


def cache_storage_cost(token_count: int, hours: str | Decimal, rate_per_million_token_hour: Any) -> str:
    if isinstance(token_count, bool) or not isinstance(token_count, int) or token_count < 0:
        raise ValueError("Invalid stored token count")
    return str(Decimal(token_count) * _decimal(hours) * _decimal(rate_per_million_token_hour) / MILLION)


def sum_costs(results: list[CostResult]) -> CostResult:
    total = sum((_decimal(r.known_subtotal_usd) for r in results), Decimal(0))
    unresolved = [f"request_{i}:{issue}" for i, r in enumerate(results) for issue in r.unresolved]
    complete = all(r.complete for r in results)
    return CostResult(str(total) if complete else None, str(total), complete, "aggregate", "mixed",
                      unresolved=unresolved)
