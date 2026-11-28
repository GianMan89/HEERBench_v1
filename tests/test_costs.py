import unittest
from decimal import Decimal

from heerbench.costs import (cache_storage_cost, conservative_request_reservation,
                            cost_from_usage, estimate_request_cost, sum_costs)
from heerbench.providers import TokenUsage


class CostAccounting(unittest.TestCase):
    def test_disjoint_cache_write_accounting_not_double_billed(self):
        usage = TokenUsage(12000, 1000, cached_input_tokens=4000,
                           cache_write_short_tokens=3000, cache_write_long_tokens=4000,
                           reasoning_tokens=500)
        pricing = {"standard": {"input": "1", "output": "5", "cache_read": "0.1",
                                 "cache_write_short": "1.25", "cache_write_long": "2.5"}}
        result = cost_from_usage(usage, pricing)
        self.assertEqual(Decimal(result.total_usd), Decimal("0.02015"))
        self.assertEqual(Decimal(result.components_usd["output"]), Decimal("0.005"))

    def test_gemini_batch_cache_read_is_not_halved_again(self):
        pricing = {"standard": {"input": "2", "output": "12", "cache_read": "0.2"},
                   "batch": {"input": "1", "output": "6", "cache_read": "0.2"}}
        result = estimate_request_cost(10000, 1000, pricing, "batch", cached_input_tokens=9000)
        self.assertEqual(Decimal(result.total_usd), Decimal("0.0088"))

    def test_no_batch_rate_means_unknown_not_discounted(self):
        result = estimate_request_cost(100, 100, {"standard": {"input": "2", "output": "10"}}, "batch")
        self.assertIsNone(result.total_usd)
        self.assertFalse(result.complete)

    def test_long_context_price_tier_applies_to_entire_request(self):
        pricing = {"standard": {"tiers": [
            {"max_input_tokens": 200000, "input": "2", "output": "12"},
            {"max_input_tokens": None, "input": "4", "output": "18"}]}}
        result = estimate_request_cost(200001, 1000, pricing)
        self.assertEqual(Decimal(result.total_usd), Decimal("0.818004"))

    def test_missing_usage_never_becomes_zero_cost(self):
        result = cost_from_usage(TokenUsage(input_tokens=None, output_tokens=10),
                                 {"standard": {"input": "1", "output": "1"}})
        self.assertFalse(result.complete)
        self.assertIsNone(result.total_usd)
        self.assertIn("input_tokens_unknown", result.unresolved)

    def test_unspecified_cache_lifetime_requires_a_rate(self):
        usage = TokenUsage(1000, 100, cache_write_unspecified_tokens=500)
        result = cost_from_usage(usage, {"standard": {"input": "1", "output": "5", "cache_write_short": "1.25"}})
        self.assertFalse(result.complete)
        self.assertIn("cache_write_unspecified_rate_unknown", result.unresolved)

    def test_reservation_is_conservative_with_cache_writes(self):
        pricing = {"standard": {"input": "2", "output": "10", "cache_write_short": "2.5", "cache_write_long": "5"}}
        result = conservative_request_reservation(10000, 1000, pricing)
        self.assertEqual(Decimal(result.total_usd), Decimal("0.06"))
        self.assertEqual(result.basis, "conservative_reservation")

    def test_unknown_cache_accounting_stays_incomplete(self):
        result = cost_from_usage(TokenUsage(10, 10, cache_accounting_known=False),
                                 {"standard": {"input": "1", "output": "1"}})
        self.assertFalse(result.complete)

    def test_storage_cost_and_aggregate_preserve_incompleteness(self):
        self.assertEqual(Decimal(cache_storage_cost(20000, "2", "4.5")), Decimal("0.18"))
        known = estimate_request_cost(1000, 100, {"standard": {"input": "1", "output": "1"}})
        unknown = estimate_request_cost(1000, 100, {})
        combined = sum_costs([known, unknown])
        self.assertFalse(combined.complete)
        self.assertIsNone(combined.total_usd)
        self.assertEqual(Decimal(combined.known_subtotal_usd), Decimal("0.0011"))

    def test_impossible_token_counts_rejected(self):
        with self.assertRaises(Exception):
            estimate_request_cost(10, 100, {"standard": {"input": "1", "output": "1"}}, cached_input_tokens=11)


if __name__ == "__main__":
    unittest.main()
