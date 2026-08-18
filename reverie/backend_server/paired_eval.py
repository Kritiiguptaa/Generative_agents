"""
Paired evaluation: both arms decide from IDENTICAL state, every step.

Why this exists
---------------
The two-independent-runs design cannot support the claim it is being used to
make. `trading_reverie.py --sim A` and `--sim B --no-middleware` start from the
same fork, then diverge the moment either places a different trade. By step 15
of run 03 the arms were no longer comparable:

    MW-Marcus    cash $76.41   (fully deployed, could not afford any purchase)
    BASE-Marcus  cash ~$21,000 (comfortable, still trading)

Those are not one agent under two conditions -- they are two agents in
different financial situations. Every downstream difference (hallucination
counts, action mix, PnL) is confounded by that divergence, and no counting rule
applied afterwards can separate the middleware effect from the state effect.
Run 03's headline gap (50.6% vs 14.3%) is mostly this.

What this does instead
----------------------
One canonical simulation advances the world. At each step, for each agent, BOTH
decision paths are invoked against the SAME persona state, the SAME market
prices and the SAME retrieved memories. The two requests are recorded as a
matched pair. Exactly one arm (--advance-with) is then allowed to actually
execute and move the world forward, so the state trajectory stays coherent.

That yields N matched pairs rather than two unpaired trajectories, which is
what makes a within-subject comparison possible: for each pair the only thing
that differs is the middleware.

What this deliberately does NOT do
----------------------------------
It does not claim the non-advancing arm's requests are what that arm would have
produced in a world it had been steering all along -- they are counterfactual
one-step-ahead requests from the advancing arm's trajectory. That is a real
limitation and it is why `--advance-with` should be run both ways and the
results compared; if the conclusion flips depending on which arm drives, the
effect is not robust and should not be reported as one.

Usage
-----
    python paired_eval.py --sim paired_01 --steps 200 --seed 42
    python paired_eval.py --sim paired_02 --steps 200 --seed 42 \
        --advance-with baseline
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from trading_reverie import (
    TradingReverie,
    build_memory_context,
    free_form_decision,
    refresh_currently,
    _count_episodes,
    check_state_grounding,
    _check_stale_reasoning,
    execute_trading_action,
)
from middleware.action_filtering import run_action_filtering_step, is_trade_request
from persona.cognitive_modules.retrieve import new_retrieve
from persona.prompt_template.gpt_structure import ollama_request
from market_perceive import market_perceive, record_trade_fill, record_order_feedback
# The independent runner updates the ID-RAG identity graph every step
# (trading_reverie._step_agent). This harness did not, so id_rag_anchor() here
# was reading a frozen bootstrap graph while the independent middleware arm
# read a live one -- the two harnesses were running different middleware.
from middleware.id_rag import update_graph_belief, update_graph_current_situation


def _snapshot(persona):
    """Deep-enough copy of the mutable decision inputs."""
    s = persona.scratch
    return {
        "cash": float(s.cash_balance),
        "positions": {k: dict(v) for k, v in s.positions.items()},
        "currently": s.currently,
    }


def _restore(persona, snap):
    s = persona.scratch
    s.cash_balance = snap["cash"]
    s.positions = {k: dict(v) for k, v in snap["positions"].items()}
    s.currently = snap["currently"]


def _classify(persona, market, decision, filter_stats):
    """
    Score a request WITHOUT executing it.

    Mirrors the disposition logic in TradingReverie._step_agent: an infeasible
    request is one the ground-truth validator would reject or clamp. Evaluated
    against a restored snapshot so neither arm's probe mutates the world.
    """
    snap = _snapshot(persona)
    # execute_trading_action() reaches market.execute_order(), which appends a
    # fill to market.order_log. Scoring BOTH arms every step therefore wrote
    # phantom fills for trades that never happened -- including the arm that
    # is not advancing the world. Nothing reads order_log today, so this had no
    # effect on agent behaviour, but it left the order log an invalid record of
    # a paired run. Truncate it back to its pre-probe length.
    n_orders = len(market.order_log)
    try:
        result = execute_trading_action(persona, market, decision)
    finally:
        _restore(persona, snap)
        del market.order_log[n_orders:]

    illegal = bool(filter_stats.get("illegal_request"))
    infeasible = bool(result["hallucination"]) or illegal
    kind = result["halluc_kind"] or ("illegal_request" if illegal else "")
    if not infeasible:
        disposition = ""
    elif illegal or result["filtered"]:
        disposition = "caught_pre_execution"
    else:
        disposition = "executed_malformed"
    return {
        "infeasible": infeasible,
        "kind": kind,
        "disposition": disposition,
        "requested_quantity": result["requested_quantity"],
        "filled_quantity": result["filled_quantity"],
    }


class PairedEvaluation(TradingReverie):

    def __init__(self, *args, advance_with="middleware", **kwargs):
        super().__init__(*args, **kwargs)
        self.advance_with = advance_with
        self.pairs = []

    def _decide_both(self, persona, market, retrieved):
        """
        Run both decision paths against identical inputs.

        Each path is given a fresh snapshot so the first cannot perturb the
        second -- build_memory_context() and refresh_currently() both write to
        scratch.
        """
        def _call_llm(prompt):
            # Same sampling settings for BOTH arms -- the arm switch changes
            # what goes into the prompt, never how the model is sampled.
            # seed/temperature are inherited from TradingReverie; at
            # temperature=0 the seed is inert and every seed replays the same
            # decisions (run 06).
            return ollama_request(prompt, max_tokens=300, format="json",
                                  temperature=self.temperature, seed=self.seed)

        base_snap = _snapshot(persona)

        # --- middleware arm -------------------------------------------------
        refresh_currently(persona, market)
        mw_context, mc_stats = build_memory_context(
            retrieved, persona, market, use_middleware=True)
        mw_decision, mw_stats = run_action_filtering_step(
            persona, market, mw_context, _call_llm, self.action_filter_log_path)
        mw_score = _classify(persona, market, mw_decision, mw_stats)
        _restore(persona, base_snap)

        # --- baseline arm ---------------------------------------------------
        refresh_currently(persona, market)
        base_context, _ = build_memory_context(
            retrieved, persona, market, use_middleware=False)
        base_decision, base_stats = free_form_decision(
            persona, market, base_context, _call_llm,
            self.action_filter_log_path)
        base_score = _classify(persona, market, base_decision, base_stats)
        _restore(persona, base_snap)

        return {
            "middleware": {"decision": mw_decision, "stats": mw_stats,
                           "score": mw_score, "context_chars": len(mw_context),
                           "compression": mc_stats},
            "baseline":   {"decision": base_decision, "stats": base_stats,
                           "score": base_score,
                           "context_chars": len(base_context)},
        }

    def _step_agent(self, name, persona, events, step, log):
        persona.scratch.curr_time = self.market.current_time
        pv_before = persona.portfolio_value(self.market.current_prices)

        market_perceive(persona, self.market, events)
        refresh_currently(persona, self.market)

        from persona.cognitive_modules.reflect import reflect
        thoughts_before = len(persona.a_mem.seq_thought)
        reflect(persona)
        new_thoughts = persona.a_mem.seq_thought[thoughts_before:]

        # ID-RAG dynamic updates after reflect. This block mirrors
        # trading_reverie._step_agent and its absence was a real defect, not a
        # simplification: in run 06, seed 42, advancing with middleware, this
        # harness reported 175 middleware trade attempts at ~2,036 context
        # chars where the independent runner on the same seed reported 50 at
        # ~800. The arms diverged at STEP 1 (Marcus: buy TSLA x13 independent
        # vs buy AMD x9 paired), which rules out drift -- the paired
        # middleware arm was anchoring on a stale identity graph from the
        # first decision onward. Paired run-06 numbers predate this fix.
        currently = getattr(persona.scratch, "currently", "") or ""
        update_graph_current_situation(name, currently)
        for thought_node in new_thoughts:
            desc = getattr(thought_node, "description", "") or ""
            if desc:
                update_graph_belief(name, desc)

        from trading_reverie import current_focus_str
        focus = current_focus_str(persona, self.market)
        focal_points = [
            f"What should {name} trade right now?",
            f"Recent price action for {', '.join(persona.scratch.watchlist[:2])}",
            f"Current plan focus: {focus}",
        ]
        retrieved = new_retrieve(persona, focal_points, n_count=15)

        both = self._decide_both(persona, self.market, retrieved)

        mw_d = both["middleware"]["decision"]
        bs_d = both["baseline"]["decision"]
        print(f"  [{name}] MW  : {mw_d.get('action')} {mw_d.get('symbol')} "
              f"x{mw_d.get('quantity')}"
              f"{'  INFEASIBLE' if both['middleware']['score']['infeasible'] else ''}")
        print(f"  [{name}] BASE: {bs_d.get('action')} {bs_d.get('symbol')} "
              f"x{bs_d.get('quantity')}"
              f"{'  INFEASIBLE' if both['baseline']['score']['infeasible'] else ''}")

        # Only the driving arm mutates the world.
        driver = both["middleware"] if self.advance_with == "middleware" else both["baseline"]
        decision = driver["decision"]
        result = execute_trading_action(persona, self.market, decision)
        if result["fill"] and not result["filtered"]:
            record_trade_fill(persona, self.market, result["fill"])

        req = driver["stats"].get("requested") or {}
        if driver["score"]["infeasible"]:
            record_order_feedback(
                persona, self.market,
                f"{name}'s order to {req.get('action')} {req.get('quantity')} "
                f"{req.get('symbol')} was REJECTED/ADJUSTED. "
                f"Cash ${persona.scratch.cash_balance:,.0f}. "
                f"Do not repeat this order unless the position or cash changes.")

        pair = {
            "step": step,
            "agent": name,
            "advanced_with": self.advance_with,
            "portfolio_value_before": pv_before,
            "cash": round(persona.scratch.cash_balance, 2),
            "market_prices": dict(self.market.current_prices),
        }
        for arm in ("middleware", "baseline"):
            side = both[arm]
            d = side["decision"]
            reasoning = d.get("reasoning", "")
            pair[arm] = {
                "action": d.get("action"),
                "symbol": d.get("symbol"),
                "quantity": d.get("quantity"),
                "requested": side["stats"].get("requested") or {},
                "filter_status": side["stats"].get("status", "success"),
                "infeasible": side["score"]["infeasible"],
                "kind": side["score"]["kind"],
                "disposition": side["score"]["disposition"],
                "has_reasoning": bool((reasoning or "").strip()),
                "context_chars": side["context_chars"],
                "stale_context": _check_stale_reasoning(
                    reasoning, step, self.market.SCRIPTED_NEWS),
                "state_contradiction": check_state_grounding(
                    reasoning, persona, self.market),
            }
        self.pairs.append(pair)
        log.append(pair)

    def _generate_report(self, log, n_steps):
        agents = sorted({p["agent"] for p in self.pairs})
        report = {
            "design": "paired",
            "advanced_with": self.advance_with,
            "seed": self.seed,
            "simulation_steps": n_steps,
            "total_pairs": len(self.pairs),
            "per_agent": {},
        }
        for arm in ("middleware", "baseline"):
            # Same denominator definition the independent report uses -- see
            # is_trade_request(). An invalid verb is a trade request; a null
            # action (unparseable) is not.
            attempts = sum(
                1 for p in self.pairs
                if is_trade_request(
                    p[arm]["requested"]
                    if p[arm]["requested"].get("action")
                    else {"action": p[arm]["action"]}))
            infeasible = sum(1 for p in self.pairs if p[arm]["infeasible"])
            caught = sum(1 for p in self.pairs
                         if p[arm]["disposition"] == "caught_pre_execution")
            executed = sum(1 for p in self.pairs
                           if p[arm]["disposition"] == "executed_malformed")
            reasoned = sum(1 for p in self.pairs if p[arm]["has_reasoning"])
            stale = sum(1 for p in self.pairs if p[arm]["stale_context"])
            contra = sum(1 for p in self.pairs if p[arm]["state_contradiction"])
            per_agent_eps = 0
            for a in agents:
                reqs = [(p["step"], (p[arm]["requested"].get("action"),
                                     p[arm]["requested"].get("symbol"),
                                     p[arm]["requested"].get("quantity")))
                        for p in self.pairs
                        if p["agent"] == a and p[arm]["infeasible"]]
                per_agent_eps += _count_episodes(reqs)
            report[arm] = {
                "trade_attempts": attempts,
                "infeasible_requests": infeasible,
                "infeasible_rate_pct": (round(infeasible / attempts * 100, 1)
                                        if attempts else None),
                "episodes": per_agent_eps,
                "episode_rate_pct": (round(per_agent_eps / attempts * 100, 1)
                                     if attempts else None),
                "caught_pre_execution": caught,
                "executed_malformed": executed,
                "reasoned_decisions": reasoned,
                "stale_context_hits": stale,
                "stale_rate_pct": (round(stale / reasoned * 100, 1)
                                   if reasoned else None),
                "state_contradictions": contra,
                "state_contradiction_rate_pct": (round(contra / reasoned * 100, 1)
                                                 if reasoned else None),
                "kinds": dict(Counter(p[arm]["kind"] for p in self.pairs
                                      if p[arm]["kind"])),
                "avg_context_chars": (round(sum(p[arm]["context_chars"]
                                                for p in self.pairs) / len(self.pairs))
                                      if self.pairs else 0),
            }

        # The paired statistic: pairs where the arms disagreed on feasibility.
        # McNemar's b/c cells -- the only cells that carry information about a
        # within-subject difference.
        both_bad = sum(1 for p in self.pairs
                       if p["middleware"]["infeasible"] and p["baseline"]["infeasible"])
        mw_only = sum(1 for p in self.pairs
                      if p["middleware"]["infeasible"] and not p["baseline"]["infeasible"])
        base_only = sum(1 for p in self.pairs
                        if p["baseline"]["infeasible"] and not p["middleware"]["infeasible"])
        neither = len(self.pairs) - both_bad - mw_only - base_only
        report["paired_contingency"] = {
            "both_infeasible": both_bad,
            "middleware_only": mw_only,
            "baseline_only": base_only,
            "neither": neither,
            "note": ("middleware_only and baseline_only are the discordant "
                     "pairs; a McNemar test uses only those two cells. "
                     "baseline_only > middleware_only means middleware "
                     "prevented infeasible requests on this seed."),
        }
        return report

    def _print_report(self, report):
        sep = "=" * 68
        print(f"\n{sep}\nPAIRED EVALUATION REPORT\n{sep}")
        print(f"Seed            : {report['seed']}")
        print(f"Advanced with   : {report['advanced_with']}")
        print(f"Matched pairs   : {report['total_pairs']}")
        print()
        hdr = f"{'':<28}{'middleware':>14}{'baseline':>14}"
        print(hdr)
        print("-" * len(hdr))
        rows = [
            ("trade attempts", "trade_attempts"),
            ("infeasible requests", "infeasible_requests"),
            ("infeasible rate %", "infeasible_rate_pct"),
            ("distinct episodes", "episodes"),
            ("caught pre-execution", "caught_pre_execution"),
            ("EXECUTED malformed", "executed_malformed"),
            ("stale context hits", "stale_context_hits"),
            ("state contradictions", "state_contradictions"),
            ("avg context chars", "avg_context_chars"),
        ]
        for label, key in rows:
            mw = report["middleware"][key]
            bs = report["baseline"][key]
            print(f"{label:<28}{str(mw):>14}{str(bs):>14}")
        c = report["paired_contingency"]
        print(f"\nDiscordant pairs (the informative ones):")
        print(f"  middleware infeasible, baseline fine : {c['middleware_only']}")
        print(f"  baseline infeasible, middleware fine : {c['baseline_only']}")
        print(f"  both infeasible                      : {c['both_infeasible']}")
        print(f"  neither                              : {c['neither']}")
        print(f"\n{c['note']}")
        print(f"\nSingle seed. Repeat across seeds and with "
              f"--advance-with {'baseline' if self.advance_with == 'middleware' else 'middleware'} "
              f"before drawing a conclusion.")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--fork", default="base_trading")
    ap.add_argument("--sim", default="paired_01")
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--advance-with", choices=("middleware", "baseline"),
                    default="middleware",
                    help="Which arm's decisions actually move the world. Run "
                         "both ways; a conclusion that depends on this is not "
                         "robust.")
    ap.add_argument("--temperature", type=float, default=0.7,
                    help="Decision-call sampling temperature. Must be > 0 for "
                         "--seed to change agent behaviour. Keep it identical "
                         "across the paired arms and across the sweep.")
    args = ap.parse_args()

    sim = PairedEvaluation(args.sim, args.fork, use_middleware=True,
                           fresh=args.fresh, seed=args.seed,
                           temperature=args.temperature,
                           advance_with=args.advance_with)
    sim.run(args.steps)


if __name__ == "__main__":
    main()
