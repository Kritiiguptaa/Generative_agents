"""
File: trading_persona.py
Description: Extends the base Persona class for a trading simulation.
  - Replaces spatial memory (s_mem) with MarketKnowledge
  - Adds trading-specific fields to scratch (cash, positions, risk params)
  - Overrides save() to persist those extra fields

No existing files are modified. All original Persona behaviour is preserved
because this class simply adds on top via subclassing.
"""
import json
import sys
import os
sys.path.append(os.path.dirname(__file__))

from persona.persona import Persona
from market_knowledge import MarketKnowledge


class TradingPersona(Persona):
    """Persona extended for trading simulations."""

    def __init__(self, name, folder_mem_saved):
        # Load all standard memory structures (a_mem, scratch, s_mem)
        super().__init__(name, folder_mem_saved)

        # Replace spatial tree with market knowledge
        mk_path = f"{folder_mem_saved}/bootstrap_memory/market_knowledge.json"
        self.s_mem = MarketKnowledge(mk_path)

        # Attach trading fields onto the existing scratch object.
        # We read them from scratch.json using .get() so the file stays
        # backward-compatible — a normal scratch.json without these keys
        # silently gets safe defaults.
        scratch_path = f"{folder_mem_saved}/bootstrap_memory/scratch.json"
        self._load_trading_fields(scratch_path)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_trading_fields(self, scratch_path):
        try:
            with open(scratch_path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}

        s = self.scratch
        s.cash_balance        = float(data.get("cash_balance", 0.0))
        s.positions           = data.get("positions", {})
        # positions format: {"NVDA": {"qty": 100, "avg_price": 487.0}, ...}

        s.trader_type         = data.get("trader_type", None)
        s.specialization      = data.get("specialization", None)
        s.risk_tolerance      = data.get("risk_tolerance", "moderate")
        # risk_limit_per_trade: max fraction of total portfolio per single trade
        s.risk_limit_per_trade = float(data.get("risk_limit_per_trade", 0.05))
        s.watchlist           = data.get("watchlist", [])
        s.trading_strategy    = data.get("trading_strategy", None)
        s.pnl_today           = float(data.get("pnl_today", 0.0))
        s.pnl_total           = float(data.get("pnl_total", 0.0))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, save_folder):
        """
        Save everything the base Persona saves, then additionally write
        the trading fields back into scratch.json and persist market knowledge.
        """
        # 1. Run the original save (writes associative_memory/, scratch.json, spatial)
        super().save(save_folder)

        # 2. Re-open the scratch.json the parent just wrote and append trading fields
        scratch_path = f"{save_folder}/scratch.json"
        try:
            with open(scratch_path, encoding="utf-8") as f:
                scratch_data = json.load(f)
        except Exception:
            scratch_data = {}

        s = self.scratch
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

        # 3. Save market knowledge (replaces spatial_memory.json)
        mk_path = f"{save_folder}/market_knowledge.json"
        self.s_mem.save(mk_path)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def portfolio_value(self, current_prices: dict) -> float:
        """Total portfolio value = cash + mark-to-market positions."""
        value = self.scratch.cash_balance
        for symbol, pos in self.scratch.positions.items():
            price = current_prices.get(symbol, pos["avg_price"])
            value += pos["qty"] * price
        return round(value, 2)

    def position_summary(self, current_prices: dict) -> str:
        """Human-readable portfolio for LLM prompts."""
        lines = [f"Cash: ${self.scratch.cash_balance:,.2f}"]
        if not self.scratch.positions:
            lines.append("Positions: none")
        else:
            for sym, pos in self.scratch.positions.items():
                price = current_prices.get(sym, pos["avg_price"])
                pnl_pct = (price - pos["avg_price"]) / pos["avg_price"] * 100
                lines.append(
                    f"  {sym}: {pos['qty']} shares @ ${pos['avg_price']:.2f} avg, "
                    f"now ${price:.2f} ({pnl_pct:+.1f}%)"
                )
        return "\n".join(lines)
