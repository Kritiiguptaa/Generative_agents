"""
Regression tests for the three defects found in the run-06 sweep.

Each pins something run 06 actually produced and could not have produced if
the code were right:

  1. --seed reached HistoricalMarketEnvironment and nothing else, while the
     decision call ran at temperature=0 (greedy). Four seeds therefore bought
     one middleware trajectory: Marcus Webb was buy=7/hold=193 with PnL
     +$608.62 on ALL four seeds, and mw_s44 / mw_s45 were identical for all
     three agents. A sweep whose whole purpose was to escape n=1 returned n~=1.
  2. paired_eval.py never updated the ID-RAG identity graph, so its middleware
     arm anchored on a frozen bootstrap graph while the runner read a live one.
     Same seed, same advancing arm, same step count gave 175 trade attempts at
     ~2,036 context chars (paired) vs 50 at ~800 (independent), diverging at
     step 1 -- Marcus: buy TSLA x13 independent, buy AMD x9 paired.
  3. The report let an arm abstain its way to a clean rate. Middleware
     reported "0.0% of 21 trade attempts" against baseline's 66.9% of 184,
     with Alex Chen requesting zero trades in three of four seeds, and nothing
     in the output distinguished that from a genuinely clean arm.

Run from backend_server:
    python -m pytest tests/test_run06_defects.py
"""

import inspect
import os
import sys
import unittest
from unittest import mock

BACKEND = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from persona.prompt_template import gpt_structure


class _FakeResponse:
    status_code = 200

    def raise_for_status(self):
        pass

    def json(self):
        return {"response": '{"action": "hold"}', "done_reason": "stop"}


class SeedReachesTheSampler(unittest.TestCase):
    """Defect 1: the seed never reached the LLM, so seeds were cosmetic."""

    def _capture_payload(self, **kwargs):
        with mock.patch.object(gpt_structure.requests, "post",
                               return_value=_FakeResponse()) as post:
            gpt_structure.ollama_request("prompt", **kwargs)
        return post.call_args[1]["json"]

    def test_seed_is_sent_to_ollama(self):
        payload = self._capture_payload(seed=43, temperature=0.7)
        self.assertEqual(payload["options"]["seed"], 43)

    def test_seed_is_omitted_when_not_given(self):
        # Absent, not 0 -- a default of 0 would silently pin every unseeded
        # call site to one sampler state.
        payload = self._capture_payload()
        self.assertNotIn("seed", payload["options"])

    def test_temperature_is_forwarded(self):
        payload = self._capture_payload(seed=42, temperature=0.7)
        self.assertEqual(payload["options"]["temperature"], 0.7)

    def test_distinct_seeds_produce_distinct_payloads(self):
        a = self._capture_payload(seed=42, temperature=0.7)
        b = self._capture_payload(seed=43, temperature=0.7)
        self.assertNotEqual(a["options"]["seed"], b["options"]["seed"])


class DecisionCallIsSeeded(unittest.TestCase):
    """The seed and temperature must actually reach the decision call site."""

    def test_make_trading_decision_accepts_seed_and_temperature(self):
        import trading_reverie
        params = inspect.signature(
            trading_reverie.make_trading_decision).parameters
        self.assertIn("seed", params)
        self.assertIn("temperature", params)

    def test_decision_call_forwards_both(self):
        import trading_reverie
        src = inspect.getsource(trading_reverie.make_trading_decision)
        self.assertIn("seed=seed", src)
        self.assertIn("temperature=temperature", src)

    def test_runner_default_temperature_is_above_zero(self):
        # At temperature=0 the sampler is greedy and the seed is inert. A
        # default of 0 would leave the run-06 defect in place.
        import trading_reverie
        default = inspect.signature(
            trading_reverie.TradingReverie.__init__
        ).parameters["temperature"].default
        self.assertGreater(default, 0.0)


class PairedHarnessMatchesTheRunner(unittest.TestCase):
    """Defect 2: paired_eval skipped the ID-RAG graph update entirely."""

    def test_paired_eval_updates_the_identity_graph(self):
        import paired_eval
        src = inspect.getsource(paired_eval.PairedEvaluation._step_agent)
        self.assertIn("update_graph_current_situation", src)
        self.assertIn("update_graph_belief", src)

    def test_both_harnesses_update_the_same_graph_functions(self):
        import paired_eval
        import trading_reverie
        paired = inspect.getsource(paired_eval.PairedEvaluation._step_agent)
        runner = inspect.getsource(trading_reverie.TradingReverie._step_agent)
        for fn in ("update_graph_current_situation", "update_graph_belief"):
            self.assertIn(fn, runner, fn + " missing from the runner")
            self.assertIn(fn, paired, fn + " missing from the paired harness")

    def test_paired_arms_share_seed_and_temperature(self):
        # Both arms must be sampled identically; the arm switch changes the
        # prompt, never the sampler.
        import paired_eval
        src = inspect.getsource(paired_eval.PairedEvaluation._decide_both)
        self.assertIn("temperature=self.temperature", src)
        self.assertIn("seed=self.seed", src)
        # One shared _call_llm, so neither arm can drift from the other.
        self.assertEqual(src.count("def _call_llm"), 1)


class AbstentionIsReported(unittest.TestCase):
    """Defect 3: HOLD is legal by construction, so 0% can mean 'did nothing'."""

    def _report(self, per_agent_actions, attempts):
        import trading_reverie
        holds = sum(a.get("hold", 0) for a in per_agent_actions.values())
        rep = {
            "middleware_enabled": True,
            "simulation_steps": 200,
            "total_log_entries": 600,
            "total_decisions": 600,
            "total_trade_attempts": sum(attempts.values()),
            "total_hallucinations": 0,
            "overall_hallucination_rate_pct": 0.0,
            "total_caught_pre_execution": 0,
            "total_executed_malformed": 0,
            "total_hallucination_episodes": 0,
            "overall_episode_rate_pct": 0.0,
            "total_state_contradictions": 9,
            "overall_state_contradiction_rate_pct": 1.5,
            "total_abstention_rate_pct": round(holds / 600 * 100, 1),
            "participating_agents": sum(1 for v in attempts.values() if v),
            "total_agents": len(attempts),
            "per_agent": {},
        }
        for name, dist in per_agent_actions.items():
            rep["per_agent"][name] = {
                "total_decisions": 200,
                "active_decisions": attempts[name],
                "trade_attempts": attempts[name],
                "action_hallucinations": 0,
                "action_hallucination_rate_pct": (
                    None if not attempts[name] else 0.0),
                "hallucination_kinds": {},
                "caught_pre_execution": 0,
                "executed_malformed": 0,
                "hallucination_episodes": 0,
                "episode_rate_pct": None,
                "fallback_decisions": 0,
                "fallback_rate_pct": 0.0,
                "validation_errors": {},
                "reasoned_decisions": 200,
                "stale_context_hits": 0,
                "stale_context_rate_pct": 0.0,
                "state_contradictions": 3,
                "state_contradiction_rate_pct": 1.5,
                "avg_persona_consistency": 1.0,
                "persona_drift_events": 0,
                "scored_decisions": 200,
                "memory_compression": {
                    "nodes_in": 0, "nodes_out": 0, "compression_ratio": 1.0,
                    "dropped_duplicate": 0, "dropped_stale": 0,
                    "dropped_budget": 0, "annotated_stale": 0,
                    "chars_in": 0, "chars_out": 0, "avg_context_chars": 0,
                },
                "action_distribution": dist,
                "abstention_rate_pct": round(dist.get("hold", 0) / 200 * 100, 1),
                "participated": bool(attempts[name]),
                "start_portfolio_usd": 1000.0,
                "end_portfolio_usd": 1000.0,
                "pnl_usd": 0.0,
            }
        return trading_reverie, rep

    def _render(self, per_agent_actions, attempts):
        import io as _io
        from contextlib import redirect_stdout
        tr, rep = self._report(per_agent_actions, attempts)
        buf = _io.StringIO()
        with redirect_stdout(buf):
            tr.TradingReverie._print_report(None, rep)
        return buf.getvalue()

    # Run 06 middleware arm, seed 44: Alex 0 attempts, Marcus 7, Sara 14.
    RUN06_MW = ({"Alex": {"buy": 0, "sell": 0, "hold": 200},
                 "Marcus": {"buy": 7, "sell": 0, "hold": 193},
                 "Sara": {"buy": 14, "sell": 0, "hold": 186}},
                {"Alex": 0, "Marcus": 7, "Sara": 14})

    ACTIVE = ({"Alex": {"buy": 20, "sell": 20, "hold": 160},
               "Marcus": {"buy": 30, "sell": 30, "hold": 140},
               "Sara": {"buy": 25, "sell": 25, "hold": 150}},
              {"Alex": 40, "Marcus": 60, "Sara": 50})

    def test_abstention_is_printed_next_to_the_rate(self):
        out = self._render(*self.RUN06_MW)
        self.assertIn("Abstention", out)
        self.assertIn("96.5%", out)   # 579 holds of 600

    def test_non_participant_is_named(self):
        # Run 06: Alex Chen requested zero trades in seeds 43, 44 and 45.
        out = self._render(*self.RUN06_MW)
        self.assertIn("NON-PARTICIPANT", out)
        self.assertIn("1 of 3 agents requested NO trades", out)

    def test_a_fully_participating_arm_gets_no_warning(self):
        out = self._render(*self.ACTIVE)
        self.assertNotIn("NON-PARTICIPANT", out)
        self.assertNotIn("requested NO trades", out)

    def test_abstaining_and_active_arms_do_not_read_alike(self):
        abstaining = self._render(*self.RUN06_MW)
        active = self._render(*self.ACTIVE)
        # Both report 0 hallucinations; only the abstention line separates them.
        self.assertIn("Total hallucinations : 0", abstaining)
        self.assertIn("Total hallucinations : 0", active)
        self.assertNotEqual(
            [l for l in abstaining.splitlines() if "Abstention (HOLD)" in l],
            [l for l in active.splitlines() if "Abstention (HOLD)" in l])


if __name__ == "__main__":
    unittest.main()
