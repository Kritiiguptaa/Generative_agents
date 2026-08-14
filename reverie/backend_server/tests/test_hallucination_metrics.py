"""
Tests for the hallucination-measurement changes.

Covers:
  Unit        — the budget calculation, the state-grounding detector, the
                episode collapser and the corrected persona watchlist check
  Regression  — the specific defects observed in runs 01-03, each of which
                made a metric report a number that could not be produced by
                the thing it claimed to measure

Run from backend_server:
    python -m pytest tests/test_hallucination_metrics.py
"""

import datetime
import os
import sys
import unittest

BACKEND = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from trading_reverie import (refresh_currently, check_state_grounding,
                             _count_episodes, execute_trading_action)
from middleware.action_filtering import (get_legal_actions, build_prompt,
                                         _get_spendable_budget)
from middleware.persona_anchoring import (score_persona_consistency,
                                          PERSONA_DRIFT_THRESHOLD)


class _Scratch:
    def __init__(self, name, cash, positions, risk, tol, watchlist):
        self.name = name
        self.cash_balance = cash
        self.positions = positions
        self.risk_limit_per_trade = risk
        self.risk_tolerance = tol
        self.watchlist = watchlist
        self.trader_type = "retail"
        self.innate = self.learned = ""
        self.currently = "placeholder"


class _Agent:
    def __init__(self, name="Marcus Webb", cash=25000.0, positions=None,
                 risk=0.2, tol="aggressive", watchlist=("TSLA", "NVDA", "AMD")):
        self.scratch = _Scratch(name, cash,
                                dict(positions or {}), risk, tol, list(watchlist))

    def portfolio_value(self, prices):
        return self.scratch.cash_balance + sum(
            p["qty"] * prices.get(s, 0) for s, p in self.scratch.positions.items())


class _Market:
    SYMBOLS = ["NVDA", "AAPL", "TSLA", "AMD", "GOOGL"]
    SCRIPTED_NEWS = {}
    is_open = True
    current_time = datetime.datetime(2026, 8, 14, 10, 0)

    def __init__(self, prices=None):
        self.current_prices = dict(prices or {
            "NVDA": 203.0, "AAPL": 190.0, "TSLA": 250.0,
            "AMD": 150.0, "GOOGL": 140.0})

    def execute_order(self, name, order):
        qty = int(order.get("quantity", 0))
        px = self.current_prices[order["symbol"]]
        return {"status": "filled", "symbol": order["symbol"], "quantity": qty,
                "fill_price": px, "total_value": px * qty, "type": order["type"]}


class BudgetDisclosure(unittest.TestCase):
    """
    Regression: the prompt advertised the risk cap while get_legal_actions()
    applied min(cash, risk_cap). Marcus Webb was shown "Max single-trade size:
    $5,017" holding $76.41, with no BUY on the menu, and requested `buy NVDA
    x10` on 20 near-consecutive steps.
    """

    def test_budget_is_capped_by_cash(self):
        agent = _Agent(cash=76.41, positions={"NVDA": {"qty": 10, "avg_price": 180.0}})
        self.assertAlmostEqual(_get_spendable_budget(agent, _Market()), 76.41, places=2)

    def test_risk_cap_binds_when_cash_is_ample(self):
        agent = _Agent(cash=25000.0, positions={})
        # PV 25000 * 0.2 = 5000, well under cash
        self.assertAlmostEqual(_get_spendable_budget(agent, _Market()), 5000.0, places=2)

    def test_prompt_never_advertises_an_unaffordable_budget(self):
        agent = _Agent(cash=76.41, positions={"NVDA": {"qty": 10, "avg_price": 180.0}})
        prompt, legal = build_prompt(agent, _Market(), "ctx")
        self.assertFalse([a for a in legal if a.type == "BUY"])
        self.assertIn("cannot afford to buy any share", prompt)
        self.assertNotIn("Max single-trade size", prompt)

    def test_prompt_states_the_constraints_as_rules(self):
        agent = _Agent(cash=25000.0, positions={})
        prompt, _ = build_prompt(agent, _Market(), "ctx")
        self.assertIn("cannot borrow", prompt.lower())
        self.assertIn("cannot sell shares you do not hold", prompt.lower())

    def test_currently_reports_weights_not_just_share_counts(self):
        alex = _Agent(name="Alex Chen", cash=402600.0,
                      positions={"NVDA": {"qty": 200, "avg_price": 192.20}},
                      risk=0.03, tol="moderate")
        text = refresh_currently(alex, _Market())
        self.assertIn("% of portfolio", text)
        self.assertIn("9.2% of portfolio", text)      # 200*203 / 443200
        for intent in ("watching", "waiting", "considering", "wants"):
            self.assertNotIn(intent, text.lower())


class StateGrounding(unittest.TestCase):
    """
    Regression: HOLD returns early from execute_trading_action(), so an agent
    that never trades could not register a hallucination. Alex Chen held on
    200/200 steps in both arms claiming his NVDA position was "already
    significant" while it was ~10% of his book and $402,600 sat in cash.
    """

    def setUp(self):
        self.market = _Market()
        self.alex = _Agent(name="Alex Chen", cash=402600.0,
                           positions={"NVDA": {"qty": 200, "avg_price": 192.20}},
                           risk=0.03, tol="moderate",
                           watchlist=("NVDA", "AAPL", "AMD", "GOOGL"))

    def test_flags_false_concentration_claim(self):
        out = check_state_grounding(
            "Position in NVDA is already significant; holding.", self.alex, self.market)
        self.assertTrue(out)
        self.assertIn("9.2%", out)

    def test_silent_on_neutral_reasoning(self):
        self.assertEqual("", check_state_grounding(
            "Signals are mixed, waiting for confirmation.", self.alex, self.market))

    def test_no_false_positive_when_genuinely_concentrated(self):
        conc = _Agent(cash=100.0, positions={"NVDA": {"qty": 100, "avg_price": 180.0}})
        self.assertEqual("", check_state_grounding(
            "my position is already significant", conc, self.market))

    def test_flags_false_capacity_claim(self):
        broke = _Agent(cash=76.41, positions={"NVDA": {"qty": 10, "avg_price": 180.0}})
        self.assertTrue(check_state_grounding(
            "I have plenty of cash to deploy here.", broke, self.market))

    def test_flags_phantom_position(self):
        out = check_state_grounding("my tsla position is up nicely",
                                    self.alex, self.market)
        self.assertIn("TSLA", out)

    def test_empty_reasoning_is_not_a_contradiction(self):
        self.assertEqual("", check_state_grounding("", self.alex, self.market))


class EpisodeCollapsing(unittest.TestCase):
    """
    Regression: a blocked order leaves state untouched, so the same request is
    reissued every step. 27 of run 03's 44 hallucinations were one agent
    repeating `buy NVDA x10`.
    """

    def test_consecutive_identical_requests_are_one_episode(self):
        self.assertEqual(1, _count_episodes(
            [(i, ("buy", "NVDA", 10)) for i in range(36, 60)]))

    def test_distinct_requests_are_distinct_episodes(self):
        self.assertEqual(2, _count_episodes(
            [(1, ("buy", "NVDA", 10)), (2, ("buy", "TSLA", 5))]))

    def test_a_long_gap_starts_a_new_episode(self):
        self.assertEqual(2, _count_episodes(
            [(1, ("buy", "NVDA", 10)), (40, ("buy", "NVDA", 10))]))

    def test_empty(self):
        self.assertEqual(0, _count_episodes([]))


class PersonaWatchlistCheck(unittest.TestCase):
    """
    Regression: the check penalised reasoning that named NO watchlist symbol,
    which measures brevity rather than character. Because the compressed arm
    writes shorter reasoning, Marcus Webb's score fell 0.98 -> 0.82 and was
    misread as the anchoring layer degrading identity.
    """

    def setUp(self):
        self.agent = _Agent(tol="aggressive", watchlist=("TSLA", "NVDA", "AMD"))

    def test_naming_no_symbol_is_not_a_violation(self):
        self.assertEqual(1.0, score_persona_consistency(
            "Current positions are within the risk limit; no new trades.", self.agent))

    def test_company_name_counts_as_the_ticker(self):
        self.assertEqual(1.0, score_persona_consistency(
            "NVIDIA smashes Q4 estimates, buying more.", self.agent))

    def test_naming_only_off_watchlist_symbols_is_a_violation(self):
        score = score_persona_consistency(
            "Rotating into GOOGL for stability.", self.agent)
        self.assertIsNotNone(score)
        self.assertLess(score, 1.0)

    def test_out_of_character_language_is_penalised(self):
        score = score_persona_consistency(
            "NVDA looks too risky, I will play it safe.", self.agent)
        self.assertLessEqual(score, PERSONA_DRIFT_THRESHOLD)

    def test_empty_reasoning_is_unscoreable_not_perfect(self):
        self.assertIsNone(score_persona_consistency("   ", self.agent))


class VenueRejection(unittest.TestCase):
    """
    Regression: execute_order() returns {"status": "rejected"} with no
    "total_value", and the caller read fill["total_value"] unguarded -- a
    KeyError that killed the run mid-step.
    """

    class _RejectingMarket(_Market):
        def execute_order(self, name, order):
            return {"status": "rejected", "reason": "invalid symbol or quantity"}

    def test_buy_rejection_does_not_raise(self):
        agent = _Agent(cash=25000.0, positions={"NVDA": {"qty": 10, "avg_price": 180.0}})
        result = execute_trading_action(
            agent, self._RejectingMarket(),
            {"action": "buy", "symbol": "NVDA", "quantity": 5, "reasoning": ""})
        self.assertTrue(result["filtered"])
        self.assertEqual("venue_rejected", result["halluc_kind"])

    def test_sell_rejection_does_not_raise(self):
        agent = _Agent(cash=25000.0, positions={"NVDA": {"qty": 10, "avg_price": 180.0}})
        result = execute_trading_action(
            agent, self._RejectingMarket(),
            {"action": "sell", "symbol": "NVDA", "quantity": 5, "reasoning": ""})
        self.assertTrue(result["filtered"])


class ClampingIsStillAHallucination(unittest.TestCase):
    """The baseline arm's only observable failure mode: an oversized request
    that executes at a corrected size."""

    def test_overselling_is_flagged_and_clamped(self):
        agent = _Agent(cash=1000.0, positions={"NVDA": {"qty": 2, "avg_price": 180.0}})
        result = execute_trading_action(
            agent, _Market(),
            {"action": "sell", "symbol": "NVDA", "quantity": 3, "reasoning": ""})
        self.assertTrue(result["hallucination"])
        self.assertEqual("clamped_holdings", result["halluc_kind"])
        self.assertEqual(3, result["requested_quantity"])
        self.assertEqual(2, result["filled_quantity"])


if __name__ == "__main__":
    unittest.main()
