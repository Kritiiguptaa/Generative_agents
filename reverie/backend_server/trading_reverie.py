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
from market_perceive    import market_perceive, record_trade_fill
from trading_persona    import TradingPersona

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
            print(f"[TradingReverie] Forked '{fork_sim_code}' → '{sim_code}'")

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

        print(f"[TradingReverie] Loaded {len(self.personas)} agents: "
              f"{list(self.personas.keys())}")
        for name, p in self.personas.items():
            pv = p.portfolio_value(self.market.current_prices)
            print(f"  {name}: ${pv:,.0f} portfolio | "
                  f"cash=${p.scratch.cash_balance:,.0f}")

    # -----------------------------------------------------------------------

    def run(self, n_steps: int) -> list:
        log = []
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
        print(f"\n[Done] {n_steps} steps complete. Log → {log_path}")
        return log

    # -----------------------------------------------------------------------

    def _step_agent(self, name: str, persona: TradingPersona,
                    events: list, step: int, log: list):
        """Run one cognitive cycle for a single agent."""

        # 2. Perceive
        market_perceive(persona, self.market, events)

        # 3. Reflect (compresses memory when budget exhausted)
        reflect(persona)

        # 4. Retrieve — two focal points per agent per step
        focal_points = [
            f"What should {name} trade right now?",
            f"Recent price action for {', '.join(persona.scratch.watchlist[:2])}",
        ]
        retrieved = new_retrieve(persona, focal_points, n_count=15)

        # 5. Decide  ← middleware compression / anchoring hooks go here
        decision = make_trading_decision(persona, self.market, retrieved)
        print(f"  [{name}] decision: {decision.get('action','?')} "
              f"{decision.get('symbol','')} x{decision.get('quantity','')}")

        # 6. Execute (with action filtering)
        result = execute_trading_action(persona, self.market, decision)
        outcome = result["outcome_str"]
        print(f"  [{name}] {outcome}")

        # 7. Record fill in memory
        if result["fill"] and not result["filtered"]:
            record_trade_fill(persona, self.market, result["fill"])

        # 8. Log
        pv = persona.portfolio_value(self.market.current_prices)
        log.append({
            "step":            step,
            "agent":           name,
            "decision":        decision,
            "outcome":         outcome,
            "filtered":        result["filtered"],
            "cash":            round(persona.scratch.cash_balance, 2),
            "portfolio_value": pv,
            "positions":       {s: dict(p) for s, p in persona.scratch.positions.items()},
            "market_prices":   dict(self.market.current_prices),
        })

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
