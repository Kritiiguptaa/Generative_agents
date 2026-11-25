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
from middleware.persona_anchoring import (score_persona_consistency, hold_rate_drift,
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

    def test_buying_off_watchlist_is_drift_even_without_reasoning(self):
        score = score_persona_consistency(
            "", self.agent, {"action": "buy", "symbol": "GOOGL", "quantity": 5})
        self.assertLessEqual(score, PERSONA_DRIFT_THRESHOLD)

    def test_buying_on_watchlist_is_scored_in_character(self):
        self.assertEqual(1.0, score_persona_consistency(
            "", self.agent, {"action": "buy", "symbol": "TSLA", "quantity": 5}))

    def test_selling_off_watchlist_holding_is_not_drift(self):
        # Exiting a position is in character; only the reasoning is scored.
        self.assertIsNone(score_persona_consistency(
            "", self.agent, {"action": "sell", "symbol": "GOOGL", "quantity": 5}))

    def test_bought_symbol_is_not_scored_twice_via_reasoning(self):
        # One off-watchlist buy is one violation, not two: the language check
        # also runs for an aggressive agent and passes, so 1 of 2 -> 0.5.
        score = score_persona_consistency(
            "Buying GOOGL.", self.agent, {"action": "buy", "symbol": "GOOGL"})
        self.assertEqual(0.5, score)

    def test_aggressive_agent_holding_most_of_run_is_drift(self):
        self.assertTrue(hold_rate_drift(0.7, self.agent))
        self.assertFalse(hold_rate_drift(0.4, self.agent))

    def test_moderate_agent_holding_all_run_is_not_drift(self):
        self.assertFalse(hold_rate_drift(1.0, _Agent(tol="moderate", watchlist=("NVDA",))))


class ConservativeLimitLanguage(unittest.TestCase):
    """
    Regression: "maximum position" is on the aggressive-word list, so a
    conservative agent citing its own limit ("within the maximum position
    limit") was scored as drifting. The first nine-agent smoke run (smoke9_mw) scored
    Daniel Reyes 0.50 on exactly that sentence.
    """

    def setUp(self):
        self.agent = _Agent(name="Daniel Reyes", tol="conservative",
                            watchlist=("AAPL", "GOOGL", "NVDA"))

    def test_citing_the_position_limit_is_not_drift(self):
        for text in (
            "AAPL and GOOGL are already within the maximum position limit.",
            "I am at the maximum position limit for GOOGL, so I hold.",
            "NVDA is under the maximum position allowed by my policy.",
            "That would exceed the maximum leverage cap, so I will wait.",
        ):
            self.assertEqual(1.0, score_persona_consistency(text, self.agent), text)

    def test_the_smoke9_sentence_scores_in_character(self):
        text = ("Maintain current positions as per the strategy. NVDA's recent "
                "fall requires volume confirmation before considering a purchase. "
                "AAPL and GOOGL are already within the maximum position limit, "
                "and the current portfolio value is already close to the maximum "
                "value for a single trade.")
        self.assertEqual(1.0, score_persona_consistency(text, self.agent))

    def test_real_aggressive_intent_still_flags(self):
        for text in (
            "Time to go to maximum position in NVDA.",
            "Going all-in on GOOGL.",
            "This is a massive opportunity, go big on AAPL.",
            "I will take maximum leverage here.",
        ):
            self.assertLessEqual(score_persona_consistency(text, self.agent),
                                 PERSONA_DRIFT_THRESHOLD, text)


class HoldCommentaryIsNotDrift(unittest.TestCase):
    """
    Regression: the reasoning-symbol check scored a HOLD 0.50 when every stock
    it named was off the watchlist. In the first 100-step nine-agent run, a
    conservative agent that never traded (watchlist AAPL, GOOGL) hit this five
    times for sentences like "Elon Musk selling a significant amount of Tesla
    shares might indicate some market volatility". Naming a stock while holding
    is commentary; only acting on it can contradict a watchlist.
    """

    HOLD = {"action": "hold", "symbol": None, "quantity": 0}

    def setUp(self):
        self.agent = _Agent(name="Margaret Okafor", tol="conservative",
                            watchlist=("AAPL", "GOOGL"))

    def test_the_run_sentences_score_in_character_on_a_hold(self):
        for text in (
            "I am cautious. The recent news about Tesla recalls and NHTSA "
            "autopilot safety probe increases my concern.",
            "Given the recent gains in NVDA and the potential supply "
            "bottleneck, it would be prudent to wait for a mo",
            "The CPI print raising fears of resumed hikes and Elon Musk selling "
            "a significant amount of Tesla shares might indicate volatility.",
        ):
            self.assertEqual(1.0, score_persona_consistency(text, self.agent, self.HOLD), text)

    def test_a_sell_that_mentions_an_off_watchlist_stock_is_not_drift(self):
        sell = {"action": "sell", "symbol": "TSLA", "quantity": 2}
        self.assertEqual(1.0, score_persona_consistency(
            "Trimming TSLA because of the recall news.", self.agent, sell))

    def test_buying_an_off_watchlist_stock_still_flags(self):
        buy = {"action": "buy", "symbol": "TSLA", "quantity": 2}
        self.assertLessEqual(score_persona_consistency(
            "Buying TSLA on the dip.", self.agent, buy), PERSONA_DRIFT_THRESHOLD)

    def test_without_an_order_the_reasoning_check_still_runs(self):
        # Callers that cannot say what the agent did keep the old behaviour.
        self.assertLessEqual(score_persona_consistency(
            "Tesla recalls worry me.", self.agent), PERSONA_DRIFT_THRESHOLD)
        self.assertLessEqual(score_persona_consistency(
            "Tesla recalls worry me.", self.agent, {}), PERSONA_DRIFT_THRESHOLD)

    def test_aggressive_wording_on_a_hold_still_flags(self):
        self.assertLessEqual(score_persona_consistency(
            "Not buying today, but I am going all-in tomorrow.", self.agent, self.HOLD),
            PERSONA_DRIFT_THRESHOLD)


class WholeWordLanguageMatch(unittest.TestCase):
    """
    Regression: the language lists were matched as substrings, so "all in"
    matched inside "fall in" and "fall indicates", and "panic" matched a
    description of the market. Three false drift events in the second 100-step
    nine-agent run, all on HOLDs: Daniel and Priya (conservative) for the word
    "fall", Kofi (aggressive) for "short-term panic".
    """

    HOLD = {"action": "hold", "symbol": None, "quantity": 0}

    def test_the_word_fall_is_not_going_all_in(self):
        cons = _Agent(name="Daniel Reyes", tol="conservative",
                      watchlist=("AAPL", "GOOGL", "NVDA"))
        for text in (
            "NVDA's critical supply bottleneck and deliveries at risk, Tesla's "
            "vehicle recalls, and the recent fall in GOOGL's price warrant "
            "careful consideration before making any new trade.",
            "AMD's recent fall indicates potential weakness, further "
            "confirmation needed before entering a new position.",
            "A small investment in AAPL would be tall in stature for this account.",
        ):
            self.assertEqual(1.0, score_persona_consistency(text, cons, self.HOLD), text)

    def test_going_all_in_still_flags_with_either_spelling(self):
        cons = _Agent(name="Daniel Reyes", tol="conservative", watchlist=("AAPL",))
        for text in ("Going all in on AAPL.", "I am all-in on this one."):
            self.assertLessEqual(score_persona_consistency(text, cons, self.HOLD),
                                 PERSONA_DRIFT_THRESHOLD, text)

    def test_describing_market_panic_is_not_the_agents_fear(self):
        aggr = _Agent(name="Kofi Mensah", tol="aggressive", watchlist=("TSLA", "AMD", "GOOGL"))
        text = ("Current market volatility presents an opportunity for profits. "
                "The recent SEC filing of Elon Musk selling Tesla shares may "
                "cause short-term panic, but Goldman Sachs is bullish.")
        self.assertEqual(1.0, score_persona_consistency(text, aggr, self.HOLD))

    def test_the_agents_own_fear_still_flags(self):
        aggr = _Agent(name="Kofi Mensah", tol="aggressive", watchlist=("TSLA",))
        for text in ("I am terrified of this drop.", "I panic when it falls.",
                     "TSLA is too risky, I will play it safe."):
            self.assertLessEqual(score_persona_consistency(text, aggr, self.HOLD),
                                 PERSONA_DRIFT_THRESHOLD, text)


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
