"""
File: trading_reverie.py
Description: Headless trading simulation loop.
  Replaces the tile-based ReverieServer for the trading domain.
  No frontend / browser required — runs entirely from the command line.

Usage:
    python trading_reverie.py --fork base_trading --sim my_run_001 --steps 200

Each step:
  1. Market ticks (prices move, scripted news fires)
  2. Each agent perceives relevant events → stored in associative memory
  3. reflect() fires if the importance budget is exhausted (memory compression)
  4. new_retrieve() selects the most relevant memories for the decision
  5. LLM decides: buy / sell / hold / analyze
  6. Action filtering validates the decision against hard portfolio constraints
  7. Approved order is executed; fill is stored in memory
  8. Checkpoint saved every 100 steps; final log written at end

Hallucination without middleware:
  - Agents receive ALL retrieved memories verbatim in the prompt (no compression)
  - LLM may act on contradictory old news (e.g. NVDA up at step 5, NVDA down at
    step 30 — which does the agent believe at step 50?)
  - Marcus Webb can request trades larger than his cash balance
  - Without persona anchoring the LLM may drift from the agent's risk profile

Your middleware layers (to be added) slot in between steps 4 and 5.
"""

import argparse
import json
import os
import shutil
import sys
import traceback
from pathlib import Path

# ── path setup ──────────────────────────────────────────────────────────────
THIS_FILE   = Path(__file__).resolve()
BACKEND_DIR = THIS_FILE.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
os.chdir(BACKEND_DIR)
# ────────────────────────────────────────────────────────────────────────────

from market_environment import MarketEnvironment

# Suppress the noisy debug prints inside reflection_trigger()
# without touching reflect.py.  We monkey-patch after import.
import persona.cognitive_modules.reflect as _reflect_mod
_orig_reflection_trigger = _reflect_mod.reflection_trigger
def _quiet_reflection_trigger(persona):
    import builtins
    _real_print = builtins.print
    builtins.print = lambda *a, **k: None   # silence during trigger check
    try:
        result = _orig_reflection_trigger(persona)
    finally:
        builtins.print = _real_print
    return result
_reflect_mod.reflection_trigger = _quiet_reflection_trigger
from market_perceive    import market_perceive, record_trade_fill
from trading_persona    import TradingPersona
from trading_daily_plan import ensure_daily_plan, apply_interaction_to_plan, current_focus_str
from trading_interactions import maybe_interaction

from persona.cognitive_modules.retrieve import new_retrieve
from persona.cognitive_modules.reflect  import reflect
from persona.prompt_template.gpt_structure import ollama_request, get_embedding
from utils import fs_storage


# ===========================================================================
# LLM Decision
# ===========================================================================

def make_trading_decision(persona: TradingPersona,
                          market:  MarketEnvironment,
                          retrieved: dict) -> dict:
    """
    Ask the LLM to decide what to do next.
    Returns {"action", "symbol", "quantity", "reasoning"}.

    NOTE: retrieved memories are passed raw here (no middleware compression).
    This is intentional — it exposes the hallucination problem that your
    middleware is meant to solve.
    """
    # Build memory context from retrieved nodes
    memory_lines = []
    for focal_pt, nodes in retrieved.items():
        for node in nodes[:6]:          # cap at 6 per focal point
            memory_lines.append(f"- {node.description}")
    memory_context = "\n".join(memory_lines) if memory_lines else "No relevant memories."

    prompt = f"""You are {persona.scratch.name}, a {persona.scratch.trader_type} trader.
Traits: {persona.scratch.innate}
Background: {persona.scratch.learned}
Current situation: {persona.scratch.currently}
Risk tolerance: {persona.scratch.risk_tolerance}
Max single-trade size: {persona.scratch.risk_limit_per_trade*100:.0f}% of portfolio

MARKET — Step {market.step} | {market.current_time.strftime('%Y-%m-%d %H:%M')}
{market.prices_str()}

YOUR PORTFOLIO
{persona.position_summary(market.current_prices)}

RELEVANT MEMORIES (most recent first)
{memory_context}

Based on your analysis, decide your next action.
Respond ONLY with valid JSON, no markdown, no extra text:
{{"action": "buy" or "sell" or "hold" or "analyze",
  "symbol": "TICKER or null",
  "quantity": integer or null,
  "reasoning": "one sentence"}}"""

    raw = ollama_request(prompt, max_tokens=120, stop=["\n\n"], timeout=300)

    # Parse — fall back to hold if LLM output is malformed
    try:
        # Strip markdown fences if the model wraps the JSON
        cleaned = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        decision = json.loads(cleaned)
        if decision.get("action") not in ("buy", "sell", "hold", "analyze"):
            raise ValueError("unknown action")
        return decision
    except Exception:
        print(f"  [LLM parse error for {persona.name}] raw: {raw[:120]!r}")
        return {"action": "hold", "symbol": None, "quantity": None,
                "reasoning": "Could not parse LLM response — defaulting to hold."}


# ===========================================================================
# Action Filtering + Execution
# ===========================================================================

def execute_trading_action(persona: TradingPersona,
                           market:  MarketEnvironment,
                           decision: dict) -> dict:
    """
    Validate decision against hard portfolio constraints, then execute.
    Returns a result dict with keys: outcome_str, fill (or None), filtered (bool).

    Action filtering rules (your middleware will enforce these; here they act
    as the ground-truth validator that reveals hallucinations):
      1. Buy cost must not exceed available cash.
      2. Agent cannot sell more shares than it owns.
      3. Single trade value must not exceed risk_limit_per_trade × portfolio.
    """
    action   = decision.get("action", "hold")
    symbol   = decision.get("symbol")
    quantity = decision.get("quantity") or 0
    reason   = decision.get("reasoning", "")

    name = persona.scratch.name

    if action in ("hold", "analyze") or not symbol:
        verb = "holds" if action == "hold" else "analyzes market"
        return {"outcome_str": f"{name} {verb}. {reason}",
                "fill": None, "filtered": False}

    price = market.current_prices.get(symbol)
    if not price:
        return {"outcome_str": f"[FILTERED] {name}: unknown symbol {symbol!r}",
                "fill": None, "filtered": True}

    # ── portfolio value for risk-limit check ───────────────────────────────
    pv = persona.portfolio_value(market.current_prices)

    if action == "buy":
        requested_cost = price * quantity
        max_by_cash    = int(persona.scratch.cash_balance / price)
        max_by_risk    = int(pv * persona.scratch.risk_limit_per_trade / price)
        max_qty        = min(max_by_cash, max_by_risk)

        # ── HALLUCINATION TRIGGER: agent asks for more than it can afford ──
        if quantity > max_by_cash:
            print(f"  [ACTION FILTER] {name} tried to buy {quantity} {symbol} "
                  f"(${requested_cost:,.0f}) but cash=${persona.scratch.cash_balance:,.0f}. "
                  f"Adjusted to {max_qty}.")

        if max_qty <= 0:
            return {"outcome_str":
                        f"[FILTERED] {name} tried to buy {symbol} but "
                        f"insufficient cash (${persona.scratch.cash_balance:,.0f}) "
                        f"or risk limit reached.",
                    "fill": None, "filtered": True}

        quantity = max_qty
        fill = market.execute_order(name, {"type": "buy", "symbol": symbol,
                                           "quantity": quantity})
        # Update portfolio
        s = persona.scratch
        s.cash_balance -= fill["total_value"]
        if symbol in s.positions:
            old   = s.positions[symbol]
            total = old["qty"] + quantity
            avg   = (old["qty"] * old["avg_price"] + quantity * price) / total
            s.positions[symbol] = {"qty": total, "avg_price": round(avg, 2)}
        else:
            s.positions[symbol] = {"qty": quantity, "avg_price": price}

        return {"outcome_str":
                    f"{name} BUYS {quantity} {symbol} @ ${price:.2f} "
                    f"(total ${fill['total_value']:,.0f}). {reason}",
                "fill": fill, "filtered": False}

    elif action == "sell":
        owned = persona.scratch.positions.get(symbol, {}).get("qty", 0)

        # ── HALLUCINATION TRIGGER: agent tries to sell shares it doesn't own ─
        if owned == 0:
            return {"outcome_str":
                        f"[FILTERED] {name} tried to SELL {quantity} {symbol} "
                        f"but owns 0 shares.",
                    "fill": None, "filtered": True}

        quantity = min(quantity, owned)
        fill = market.execute_order(name, {"type": "sell", "symbol": symbol,
                                           "quantity": quantity})
        s = persona.scratch
        s.cash_balance += fill["total_value"]
        s.positions[symbol]["qty"] -= quantity
        if s.positions[symbol]["qty"] == 0:
            del s.positions[symbol]

        return {"outcome_str":
                    f"{name} SELLS {quantity} {symbol} @ ${price:.2f} "
                    f"(total ${fill['total_value']:,.0f}). {reason}",
                "fill": fill, "filtered": False}

    return {"outcome_str": f"{name} holds (unknown action).",
            "fill": None, "filtered": False}


# ===========================================================================
# Stale-context hallucination detector
# ===========================================================================

def _check_stale_reasoning(reasoning: str, current_step: int,
                            scripted_news: dict) -> str:
    """
    Cheap heuristic: scan the LLM's reasoning for keywords from news headlines
    that are more than 20 steps old AND have a contradicting headline at a
    later step.  Returns a warning string if stale context is detected.

    This is intentionally simple — it catches obvious cases like an agent
    referencing 'earnings beat' (step 5 news) at step 50 when step 30
    already fired contradicting supply-chain bad news.
    """
    if not reasoning:
        return ""

    reasoning_lower = reasoning.lower()

    # Build a map: keyword → [(step, is_positive)]
    kw_map: dict = {}
    for step, (symbol, headline, impact) in scripted_news.items():
        is_positive = impact > 0
        for word in headline.lower().split():
            if len(word) > 5 and word.isalpha():
                kw_map.setdefault(word, []).append((step, is_positive))

    staleness_warnings = []
    for word, appearances in kw_map.items():
        if word not in reasoning_lower:
            continue
        for news_step, is_positive in appearances:
            age = current_step - news_step
            if age < 20:
                continue
            # Check if a contradicting event for the same keyword exists
            # at a more recent step
            for other_step, other_positive in appearances:
                if other_step > news_step and other_positive != is_positive:
                    if current_step - other_step < age:
                        staleness_warnings.append(
                            f"'{word}' (from step {news_step}, "
                            f"{age} steps ago; contradicted at step {other_step})"
                        )
                        break

    return "; ".join(staleness_warnings[:2]) if staleness_warnings else ""


# ===========================================================================
# Simulation class
# ===========================================================================

class TradingReverie:

    def __init__(self, sim_code: str, fork_sim_code: str = "base_trading"):
        self.sim_code      = sim_code
        self.fork_sim_code = fork_sim_code
        self.sim_folder    = f"{fs_storage}/{sim_code}"
        self.market        = MarketEnvironment(seed=42)

        # Copy base simulation folder if target doesn't exist yet
        fork_path = Path(f"{fs_storage}/{fork_sim_code}")
        sim_path  = Path(self.sim_folder)
        if not sim_path.exists():
            shutil.copytree(fork_path, sim_path)
            print(f"[TradingReverie] Forked '{fork_sim_code}' -> '{sim_code}'")

        # Load meta
        meta_path = sim_path / "reverie" / "meta.json"
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        self.sec_per_step          = meta.get("sec_per_step", 300)
        self.market.sec_per_step   = self.sec_per_step

        # Load agents
        self.personas: dict[str, TradingPersona] = {}
        for name in meta["persona_names"]:
            folder = f"{self.sim_folder}/personas/{name}"
            p = TradingPersona(name, folder)
            # Introduce each agent to the others
            for other in meta["persona_names"]:
                if other != name:
                    p.s_mem.add_known_agent(other)
            self.personas[name] = p

        # Initialize daily plans for each agent
        for persona in self.personas.values():
            ensure_daily_plan(persona, self.market, force=True)

        print(f"[TradingReverie] Loaded {len(self.personas)} agents: "
              f"{list(self.personas.keys())}")
        for name, p in self.personas.items():
            pv = p.portfolio_value(self.market.current_prices)
            print(f"  {name}: ${pv:,.0f} portfolio | "
                  f"cash=${p.scratch.cash_balance:,.0f}")

    # -----------------------------------------------------------------------

    def run(self, n_steps: int) -> list:
        log = []
        interactions = []
        sim_path = Path(self.sim_folder)

        for _ in range(n_steps):
            step = self.market.step      # capture before tick increments it
            print(f"\n{'─'*60}")
            print(f"STEP {step} | {self.market.current_time.strftime('%Y-%m-%d %H:%M')}")

            # ── 1. Market tick ──────────────────────────────────────────────
            events = self.market.tick()

            if events:
                print(f"  Market events: "
                      + " | ".join(e.description[:50] for e in events))

            # ── 1.5. Daily plans (refresh on day rollover) ─────────────────
            for persona in self.personas.values():
                ensure_daily_plan(persona, self.market)

            # ── 1.6. Peer interaction (cadence-based) ─────────────────────
            interaction = maybe_interaction(self.personas, self.market, step)
            if interaction:
                for agent_name in interaction.get("agents", []):
                    apply_interaction_to_plan(
                        self.personas[agent_name], interaction.get("summary", "")
                    )
                interactions.append({"step": step, **interaction})
                print(
                    f"  [Interaction] {interaction['agents'][0]} + "
                    f"{interaction['agents'][1]}: {interaction['summary']}"
                )

            for name, persona in self.personas.items():
                try:
                    self._step_agent(name, persona, events, step, log)
                except Exception as exc:
                    print(f"  [{name}] UNHANDLED ERROR: {exc}")
                    traceback.print_exc()

            # ── Checkpoint ─────────────────────────────────────────────────
            if self.market.step % 100 == 0:
                self._save()
                print(f"  [Checkpoint] saved at market step {self.market.step}")

        # Final save + log
        self._save()
        log_path = sim_path / "reverie" / "trading_log.json"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log, f, indent=2)

        interactions_path = sim_path / "reverie" / "trading_interactions.json"
        with open(interactions_path, "w", encoding="utf-8") as f:
            json.dump(interactions, f, indent=2)

        report = self._generate_report(log, n_steps)
        report_path = sim_path / "reverie" / "hallucination_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        self._print_report(report)
        return log

    # -----------------------------------------------------------------------

    def _step_agent(self, name: str, persona: TradingPersona,
                    events: list, step: int, log: list):
        """Run one cognitive cycle for a single agent."""

        # Keep persona's internal clock in sync with market time.
        # reflect() and memory nodes use this for timestamps.
        persona.scratch.curr_time = self.market.current_time

        # 2. Perceive
        market_perceive(persona, self.market, events)

        # 3. Reflect (compresses memory when budget exhausted)
        reflect(persona)

        # 4. Retrieve — two focal points per agent per step
        focus = current_focus_str(persona, self.market)
        focal_points = [
            f"What should {name} trade right now?",
            f"Recent price action for {', '.join(persona.scratch.watchlist[:2])}",
            f"Current plan focus: {focus}",
        ]
        retrieved = new_retrieve(persona, focal_points, n_count=15)

        # 5. Decide  ← middleware compression / anchoring hooks go here
        decision = make_trading_decision(persona, self.market, retrieved)
        print(f"  [{name}] decision: {decision.get('action','?')} "
              f"{decision.get('symbol','')} x{decision.get('quantity','')}")

        # 6. Execute (with action filtering)
        result = execute_trading_action(persona, self.market, decision)
        outcome = result["outcome_str"]

        # Print with clear hallucination markers
        if result["filtered"]:
            print(f"  [{name}] *** HALLUCINATION CAUGHT *** {outcome}")
        else:
            print(f"  [{name}] {outcome}")

        # 7. Record fill in memory
        if result["fill"] and not result["filtered"]:
            record_trade_fill(persona, self.market, result["fill"])

        # 8. Detect stale-news hallucination:
        # If agent cites reasoning that references a news headline older than
        # 20 steps while a contradicting headline exists in memory, flag it.
        reasoning = decision.get("reasoning", "")
        stale_flag = _check_stale_reasoning(
            reasoning, step, self.market.SCRIPTED_NEWS
        )
        if stale_flag:
            print(f"  [{name}] *** STALE CONTEXT *** reasoning may reference "
                  f"outdated news: {stale_flag}")

        # 9. Log
        pv = persona.portfolio_value(self.market.current_prices)
        log.append({
            "step":             step,
            "agent":            name,
            "decision":         decision,
            "outcome":          outcome,
            "hallucination":    result["filtered"],
            "stale_context":    stale_flag,
            "cash":             round(persona.scratch.cash_balance, 2),
            "portfolio_value":  pv,
            "positions":        {s: dict(p)
                                 for s, p in persona.scratch.positions.items()},
            "market_prices":    dict(self.market.current_prices),
        })

    # -----------------------------------------------------------------------

    def _generate_report(self, log: list, n_steps: int) -> dict:
        """
        Compute hallucination metrics from the trading log.

        Hallucination types tracked:
          1. action_hallucination  — LLM requested an impossible trade
             (insufficient cash, selling unowned stock, over risk-limit)
          2. stale_context         — LLM reasoning cited outdated contradicted news

        Returns a dict suitable for JSON serialisation.
        """
        from collections import defaultdict

        agents = list(self.personas.keys())
        per_agent = {
            name: {
                "total_decisions":       0,
                "active_decisions":      0,  # buy or sell (not hold/analyze)
                "action_hallucinations": 0,
                "stale_context_hits":    0,
                "action_counts":         {"buy": 0, "sell": 0,
                                          "hold": 0, "analyze": 0},
                "start_portfolio":       0.0,
                "end_portfolio":         0.0,
            }
            for name in agents
        }

        # Capture start portfolio from first log entry per agent
        seen_start = set()
        for entry in log:
            name = entry["agent"]
            if name not in seen_start:
                per_agent[name]["start_portfolio"] = entry["portfolio_value"]
                seen_start.add(name)

        for entry in log:
            name   = entry["agent"]
            action = entry["decision"].get("action", "hold")
            ag     = per_agent[name]

            ag["total_decisions"] += 1
            ag["action_counts"][action] = ag["action_counts"].get(action, 0) + 1

            if action in ("buy", "sell"):
                ag["active_decisions"] += 1

            if entry.get("hallucination"):
                ag["action_hallucinations"] += 1

            if entry.get("stale_context"):
                ag["stale_context_hits"] += 1

            # Last seen entry = end state
            ag["end_portfolio"] = entry["portfolio_value"]

        # Compute rates
        summary = {}
        total_hallucinations = 0
        for name, ag in per_agent.items():
            active = ag["active_decisions"] or 1   # avoid /0
            total  = ag["total_decisions"]  or 1

            action_h_rate  = round(ag["action_hallucinations"] / active * 100, 1)
            stale_rate     = round(ag["stale_context_hits"]    / total  * 100, 1)
            pnl            = round(ag["end_portfolio"] - ag["start_portfolio"], 2)

            summary[name] = {
                "total_decisions":          ag["total_decisions"],
                "active_decisions":         ag["active_decisions"],
                "action_hallucinations":    ag["action_hallucinations"],
                "action_hallucination_rate_pct": action_h_rate,
                "stale_context_hits":       ag["stale_context_hits"],
                "stale_context_rate_pct":   stale_rate,
                "action_distribution":      ag["action_counts"],
                "start_portfolio_usd":      ag["start_portfolio"],
                "end_portfolio_usd":        ag["end_portfolio"],
                "pnl_usd":                  pnl,
            }
            total_hallucinations += ag["action_hallucinations"]

        total_decisions = sum(a["total_decisions"] for a in per_agent.values()) or 1
        return {
            "simulation_steps":            n_steps,
            "total_log_entries":           len(log),
            "total_hallucinations":        total_hallucinations,
            "overall_hallucination_rate_pct":
                round(total_hallucinations / total_decisions * 100, 1),
            "per_agent": summary,
        }

    def _print_report(self, report: dict):
        sep = "=" * 60
        print(f"\n{sep}")
        print("HALLUCINATION REPORT")
        print(sep)
        print(f"Steps simulated : {report['simulation_steps']}")
        print(f"Total decisions : {report['total_log_entries']}")
        print(f"Total hallucinations : {report['total_hallucinations']}  "
              f"({report['overall_hallucination_rate_pct']}% of all decisions)")
        print()
        for name, ag in report["per_agent"].items():
            print(f"  {name}")
            print(f"    Decisions      : {ag['total_decisions']}  "
                  f"(active={ag['active_decisions']})")
            print(f"    Action hallucinations : {ag['action_hallucinations']}  "
                  f"({ag['action_hallucination_rate_pct']}% of active)")
            print(f"    Stale context hits    : {ag['stale_context_hits']}  "
                  f"({ag['stale_context_rate_pct']}% of decisions)")
            dist = ag["action_distribution"]
            print(f"    Actions        : buy={dist.get('buy',0)}  "
                  f"sell={dist.get('sell',0)}  hold={dist.get('hold',0)}  "
                  f"analyze={dist.get('analyze',0)}")
            print(f"    Portfolio PnL  : ${ag['pnl_usd']:+,.2f}  "
                  f"(${ag['start_portfolio_usd']:,.0f} -> ${ag['end_portfolio_usd']:,.0f})")
            print()
        print(sep)

    # -----------------------------------------------------------------------

    def _save(self):
        for name, persona in self.personas.items():
            save_folder = f"{self.sim_folder}/personas/{name}/bootstrap_memory"
            persona.save(save_folder)
        print(f"  [Saved] market step {self.market.step}")


# ===========================================================================
# CLI entry point
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run the trading agent simulation."
    )
    parser.add_argument("--fork",  default="base_trading",
                        help="Source simulation to fork from")
    parser.add_argument("--sim",   default="trading_sim_001",
                        help="Name for the new simulation run")
    parser.add_argument("--steps", type=int, default=120,
                        help="Number of market steps to simulate")
    args = parser.parse_args()

    sim = TradingReverie(args.sim, args.fork)
    sim.run(args.steps)


if __name__ == "__main__":
    main()
