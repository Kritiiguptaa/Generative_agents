"""
File: trading_persona.py
Description: Extends the base Persona class for a trading simulation.
  - Replaces spatial memory (s_mem) with MarketKnowledge
  - Adds trading-specific fields onto the existing Scratch object
  - Overrides save() to:
      a) patch the Scratch.save() crash that occurs when act_start_time is None
      b) persist trading fields back into scratch.json
      c) save market knowledge to market_knowledge.json

No existing files are modified.
"""
import json
import os
import sys
sys.path.append(os.path.dirname(__file__))

from persona.persona import Persona
from market_knowledge import MarketKnowledge


class TradingPersona(Persona):
    """Persona extended for the trading simulation."""

    def __init__(self, name, folder_mem_saved):
        # Load standard memory (a_mem, scratch, s_mem).
        # s_mem loads spatial_memory.json if it exists; we overwrite it next.
        super().__init__(name, folder_mem_saved)

        # Replace tile-based spatial memory with market knowledge.
        mk_path = f"{folder_mem_saved}/bootstrap_memory/market_knowledge.json"
        self.s_mem = MarketKnowledge(mk_path)

        # Attach trading fields onto the existing scratch object.
        # Uses .get() so a plain scratch.json (no trading fields) gets safe defaults.
        scratch_path = f"{folder_mem_saved}/bootstrap_memory/scratch.json"
        self._load_trading_fields(scratch_path)

    # ------------------------------------------------------------------
    # Private: load trading fields
    # ------------------------------------------------------------------

    def _load_trading_fields(self, scratch_path):
        try:
            with open(scratch_path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}

        s = self.scratch
        s.cash_balance         = float(data.get("cash_balance", 0.0))
        s.positions            = data.get("positions", {})
        # positions: {"NVDA": {"qty": 100, "avg_price": 487.0}, ...}

        s.trader_type          = data.get("trader_type", None)
        s.specialization       = data.get("specialization", None)
        s.risk_tolerance       = data.get("risk_tolerance", "moderate")
        s.risk_limit_per_trade = float(data.get("risk_limit_per_trade", 0.05))
        s.watchlist            = data.get("watchlist", [])
        s.trading_strategy     = data.get("trading_strategy", None)
        s.pnl_today            = float(data.get("pnl_today", 0.0))
        s.pnl_total            = float(data.get("pnl_total", 0.0))

    # ------------------------------------------------------------------
    # Persistence  (overrides Persona.save)
    # ------------------------------------------------------------------

    def save(self, save_folder):
        """
        Save all memory structures.

        Scratch.save() crashes when act_start_time is None (it calls
        .strftime() unconditionally).  In the trading simulation act_start_time
        is never set by plan.py, so we temporarily patch it to curr_time
        before delegating to the parent, then restore it.
        """
        s = self.scratch

        # ── patch: prevent Scratch.save() crash ──────────────────────────
        _patched_start = False
        if s.act_start_time is None:
            s.act_start_time = s.curr_time   # curr_time is always set
            _patched_start = True

        try:
            # Ensure associative_memory/ subdirectory exists —
            # AssociativeMemory.save() doesn't create it.
            os.makedirs(os.path.join(save_folder, "associative_memory"),
                        exist_ok=True)
            # Saves: a_mem → associative_memory/
            #        scratch → scratch.json  (now safe)
            #        s_mem   → spatial_memory.json  (writes MarketKnowledge JSON)
            super().save(save_folder)
        finally:
            if _patched_start:
                s.act_start_time = None      # restore

        # ── append trading fields to the scratch.json just written ───────
        scratch_path = os.path.join(save_folder, "scratch.json")
        try:
            with open(scratch_path, encoding="utf-8") as f:
                scratch_data = json.load(f)
        except Exception:
            scratch_data = {}

        scratch_data["cash_balance"]         = s.cash_balance
        scratch_data["positions"]            = s.positions
        scratch_data["trader_type"]          = s.trader_type
        scratch_data["specialization"]       = s.specialization
        scratch_data["risk_tolerance"]       = s.risk_tolerance
        scratch_data["risk_limit_per_trade"] = s.risk_limit_per_trade
        scratch_data["watchlist"]            = s.watchlist
        scratch_data["trading_strategy"]     = s.trading_strategy
        scratch_data["pnl_today"]            = s.pnl_today
        scratch_data["pnl_total"]            = s.pnl_total

        with open(scratch_path, "w", encoding="utf-8") as f:
            json.dump(scratch_data, f, indent=2)

        # ── save market knowledge to its own file ────────────────────────
        mk_path = os.path.join(save_folder, "market_knowledge.json")
        self.s_mem.save(mk_path)

    # ------------------------------------------------------------------
    # Convenience helpers (used in trading_reverie.py)
    # ------------------------------------------------------------------

    def portfolio_value(self, current_prices: dict) -> float:
        """Total mark-to-market portfolio value."""
        value = self.scratch.cash_balance
        for symbol, pos in self.scratch.positions.items():
            price = current_prices.get(symbol, pos["avg_price"])
            value += pos["qty"] * price
        return round(value, 2)

    def position_summary(self, current_prices: dict) -> str:
        """Human-readable portfolio string for LLM prompts."""
        lines = [f"Cash: ${self.scratch.cash_balance:,.2f}"]
        if not self.scratch.positions:
            lines.append("Positions: none")
        else:
            for sym, pos in self.scratch.positions.items():
                price   = current_prices.get(sym, pos["avg_price"])
                pnl_pct = (price - pos["avg_price"]) / pos["avg_price"] * 100
                lines.append(
                    f"  {sym}: {pos['qty']} shares @ ${pos['avg_price']:.2f} avg, "
                    f"now ${price:.2f} ({pnl_pct:+.1f}%)"
                )
        return "\n".join(lines)
