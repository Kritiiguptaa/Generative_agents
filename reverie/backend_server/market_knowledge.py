"""
File: market_knowledge.py
Description: Stores what a trading agent knows about the market.
  Replaces spatial_memory.py (MemoryTree) for the trading simulation.
  The interface mirrors MemoryTree just enough so TradingPersona can
  swap it in without any other file needing to change.
"""
import json
import os


class MarketKnowledge:
    """
    Persistent knowledge store about the market and its participants.

    Analogous to SpatialMemory / MemoryTree in the original system —
    it holds what the agent *knows* about the environment structure,
    but the environment here is a market instead of a physical map.
    """

    def __init__(self, f_saved: str):
        # Known symbols: symbol -> {sector, last_known_price, notes}
        self.known_stocks: dict = {}
        # Other traders this agent is aware of
        self.known_agents: list = []
        # General belief about macro / market regime
        self.market_context: str = ""

        if os.path.exists(f_saved):
            try:
                with open(f_saved, encoding="utf-8") as fh:
                    data = json.load(fh)
                self.known_stocks   = data.get("known_stocks", {})
                self.known_agents   = data.get("known_agents", [])
                self.market_context = data.get("market_context", "")
            except Exception as e:
                print(f"[MarketKnowledge] WARNING: could not load {f_saved}: {e}")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, out_path: str):
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump({
                "known_stocks":   self.known_stocks,
                "known_agents":   self.known_agents,
                "market_context": self.market_context,
            }, fh, indent=2)

    # ------------------------------------------------------------------
    # Update helpers (called from market_perceive.py)
    # ------------------------------------------------------------------

    def update_price(self, symbol: str, price: float):
        """Record the latest observed price for a symbol."""
        if symbol not in self.known_stocks:
            self.known_stocks[symbol] = {"sector": "unknown", "notes": []}
        self.known_stocks[symbol]["last_known_price"] = round(price, 2)

    def add_note(self, symbol: str, note: str):
        """Append an analyst note for a symbol (max 5 kept, oldest dropped)."""
        if symbol not in self.known_stocks:
            self.known_stocks[symbol] = {"sector": "unknown", "notes": []}
        notes = self.known_stocks[symbol].setdefault("notes", [])
        notes.append(note)
        self.known_stocks[symbol]["notes"] = notes[-5:]

    def add_known_agent(self, agent_name: str):
        if agent_name not in self.known_agents:
            self.known_agents.append(agent_name)

    def set_market_context(self, context: str):
        self.market_context = context

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_last_price(self, symbol: str) -> float:
        """Return last known price, or 0 if unknown."""
        return self.known_stocks.get(symbol, {}).get("last_known_price", 0.0)

    def get_notes(self, symbol: str) -> list:
        return self.known_stocks.get(symbol, {}).get("notes", [])

    def summary_str(self) -> str:
        """Compact string for LLM prompts."""
        lines = []
        if self.market_context:
            lines.append(f"Market context: {self.market_context}")
        for sym, info in self.known_stocks.items():
            price = info.get("last_known_price", "?")
            lines.append(f"  {sym}: last known ${price}")
        return "\n".join(lines) if lines else "No market knowledge yet."

    # ------------------------------------------------------------------
    # MemoryTree compatibility shim
    # (so code that calls persona.s_mem.tree doesn't crash outright)
    # ------------------------------------------------------------------

    @property
    def tree(self):
        """Compatibility shim — returns known_stocks as a flat dict."""
        return self.known_stocks

    def print_tree(self):
        print(self.summary_str())
