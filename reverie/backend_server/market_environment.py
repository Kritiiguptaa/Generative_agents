"""
File: market_environment.py
Description: Simulates a stock market for the trading agent simulation.
  Replaces maze.py — instead of a 2D tile map this provides a price feed,
  an order book, and a scripted news calendar.

Hallucination design:
  The scripted news events are intentionally contradictory over time so that
  agents who rely on stale memories (without proper compression/anchoring)
  will make wrong decisions.  The middleware layers are expected to reduce
  this by compressing memories and anchoring the agent to its current context.

  Example contradiction pairs baked in:
    Step   5 — NVDA: blowout earnings, stock +8%      (positive thesis)
    Step  30 — NVDA: supply-chain bottleneck, -6%     (contradicts step 5)
    Step  70 — NVDA: analyst says concerns overblown  (contradicts step 30)

    Step  15 — TSLA: safety recall, -7%
    Step  40 — TSLA: Q1 delivery beat, upgraded, +9%  (contradicts step 15)
    Step  80 — TSLA: CEO sells $2B of stock, -5%      (contradicts step 40)
"""

import random
import datetime
import math
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

class MarketEvent:
    """A single observable market event (price move, news, trade fill)."""

    def __init__(self, event_type: str, symbol: Optional[str],
                 description: str, magnitude: float,
                 timestamp: datetime.datetime):
        self.event_type  = event_type   # "price_move" | "news" | "trade_fill"
        self.symbol      = symbol       # None for market-wide events
        self.description = description
        self.magnitude   = magnitude    # % change (signed)
        self.timestamp   = timestamp

    def to_spo(self) -> Tuple[str, str, str]:
        """Return (subject, predicate, object) triple for ConceptNode."""
        if self.event_type == "price_move":
            verb = "rises" if self.magnitude > 0 else "falls"
            return (self.symbol, verb, f"{abs(self.magnitude):.1f}%")
        elif self.event_type == "news":
            subj = self.symbol if self.symbol else "market"
            return (subj, "reports", self.description[:60])
        elif self.event_type == "trade_fill":
            return (self.symbol, "fills", self.description[:60])
        else:
            return ("market", "is", self.description[:60])

    def __repr__(self):
        return f"<MarketEvent {self.event_type} {self.symbol} {self.magnitude:+.1f}%>"


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class MarketEnvironment:
    """
    Simulates 5 stocks with a regime-based random-walk price model
    plus scripted news.

    Usage:
        market = MarketEnvironment(seed=42)
        events = market.tick()          # advance one step, get events
        fill   = market.execute_order(agent_name, order_dict)
        snap   = market.get_snapshot()
    """

    SYMBOLS = ["NVDA", "AAPL", "TSLA", "AMD", "GOOGL"]

    # Daily volatility targets (fraction). Used to derive per-step vol.
    DAILY_VOLATILITY = {
        "NVDA":  0.045,   # high
        "AAPL":  0.020,   # low-moderate
        "TSLA":  0.060,   # very high
        "AMD":   0.040,
        "GOOGL": 0.025,
    }

    # Daily drift targets (fraction).
    DAILY_DRIFT = {
        "NVDA":  0.0006,
        "AAPL":  0.0003,
        "TSLA":  0.0006,
        "AMD":   0.0005,
        "GOOGL": 0.0003,
    }

    BASE_PRICES = {
        "NVDA":  487.0,
        "AAPL":  172.0,
        "TSLA":  245.0,
        "AMD":   178.0,
        "GOOGL": 141.0,
    }

    # Regime multipliers for volatility clustering.
    REGIME_MULT = {
        "calm": 0.7,
        "normal": 1.0,
        "turbulent": 1.6,
    }

    # Execution model (offline microstructure)
    BASE_SPREAD_BPS = {
        "NVDA": 3.0,
        "AAPL": 2.0,
        "TSLA": 5.0,
        "AMD":  4.0,
        "GOOGL": 3.0,
    }
    LIQUIDITY = {
        "NVDA": 20000,
        "AAPL": 30000,
        "TSLA": 18000,
        "AMD":  15000,
        "GOOGL": 22000,
    }
    BASE_SLIPPAGE_BPS = 1.0
    IMPACT_BPS_AT_FULL_LIQ = 15.0
    COMMISSION_PER_SHARE = 0.005
    MIN_COMMISSION = 1.0

    # -----------------------------------------------------------------------
    # Scripted news: step -> (symbol, headline, price_impact_fraction)
    # Designed so that contradictory events for the same symbol appear at
    # different steps, creating a "stale memory" hallucination risk.
    # -----------------------------------------------------------------------
    SCRIPTED_NEWS: Dict[int, Tuple[str, str, float]] = {
        # ---- NVDA arc: positive → negative → positive ----
        5:   ("NVDA",  "NVIDIA smashes Q4 estimates — AI chip revenue up 265% YoY",          +0.08),
        30:  ("NVDA",  "NVDA Taiwan fab reports critical supply bottleneck; deliveries at risk", -0.06),
        70:  ("NVDA",  "Goldman Sachs: NVDA supply fears overblown, maintains Buy $650 target", +0.04),
        100: ("NVDA",  "NVIDIA announces 10-for-1 stock split effective next quarter",          +0.07),

        # ---- TSLA arc: negative → positive → negative ----
        15:  ("TSLA",  "Tesla recalls 200k vehicles; NHTSA opens autopilot safety probe",       -0.07),
        40:  ("TSLA",  "Tesla Q1 deliveries beat street by 18%; Wedbush upgrades to Outperform",+0.09),
        80:  ("TSLA",  "SEC filing: Elon Musk sells $2.1B of Tesla shares",                     -0.05),

        # ---- AAPL: single negative ----
        60:  ("AAPL",  "DOJ files antitrust suit against Apple App Store; $3B fine possible",   -0.04),

        # ---- AMD: positive mid-sim ----
        50:  ("AMD",   "AMD gains enterprise GPU share as NVDA supply tightens; Q2 outlook raised", +0.06),

        # ---- GOOGL: positive late ----
        90:  ("GOOGL", "Google DeepMind AlphaFold 3 licensed to Pfizer — landmark AI deal",     +0.06),

        # ---- Market-wide macro shock ----
        45:  (None,    "Fed signals pause in rate hikes — markets rally broadly",               +0.02),
        85:  (None,    "Stronger-than-expected CPI print raises fears of resumed hikes",        -0.02),
    }

    def __init__(self, seed: int = 42, sec_per_step: int = 60):
        random.seed(seed)
        self.sec_per_step   = sec_per_step
        self.step           = 0
        self.current_time   = datetime.datetime(2024, 1, 15, 9, 30, 0)
        self.current_prices = dict(self.BASE_PRICES)
        self.price_history  = {s: [self.BASE_PRICES[s]] for s in self.SYMBOLS}
        self.order_log: List[dict]         = []
        self.all_events:  List[MarketEvent] = []

        # Regime and volatility state
        self.regime = "normal"
        self.vol_scale = {s: 1.0 for s in self.SYMBOLS}
        self.steps_per_day = int((6.5 * 60 * 60) / self.sec_per_step)

    def _base_step_vol(self, symbol: str) -> float:
        daily_vol = self.DAILY_VOLATILITY.get(symbol, 0.03)
        return daily_vol / math.sqrt(max(self.steps_per_day, 1))

    def _current_sigma(self, symbol: str) -> float:
        base = self._base_step_vol(symbol)
        scale = min(3.0, max(0.5, self.vol_scale.get(symbol, 1.0)))
        return base * self.REGIME_MULT[self.regime] * scale

    def _maybe_shift_regime(self) -> None:
        r = random.random()
        if self.regime == "calm":
            if r < 0.80:
                return
            if r < 0.99:
                self.regime = "normal"
            else:
                self.regime = "turbulent"
        elif self.regime == "normal":
            if r < 0.10:
                self.regime = "calm"
            elif r < 0.95:
                return
            else:
                self.regime = "turbulent"
        else:
            if r < 0.05:
                self.regime = "calm"
            elif r < 0.30:
                self.regime = "normal"
            else:
                return

    # -----------------------------------------------------------------------
    # Core simulation step
    # -----------------------------------------------------------------------

    def tick(self) -> List[MarketEvent]:
        """
        Advance the market by one step.
        Returns the list of MarketEvent objects that occurred this step.
        Events with magnitude < 1% are suppressed — they are the equivalent
        of 'idle' tiles in the original simulation.
        """
        events: List[MarketEvent] = []

        # 1. Random price moves
        self._maybe_shift_regime()
        for symbol in self.SYMBOLS:
            sigma = self._current_sigma(symbol)
            mu = self.DAILY_DRIFT.get(symbol, 0.0003) / max(self.steps_per_day, 1)
            pct = random.gauss(mu, sigma)
            old  = self.current_prices[symbol]
            new  = max(1.0, round(old * (1.0 + pct), 2))
            self.current_prices[symbol] = new
            self.price_history[symbol].append(new)

            base_vol = self._base_step_vol(symbol)
            if base_vol > 0:
                ratio = abs(pct) / base_vol
                self.vol_scale[symbol] = 0.9 * self.vol_scale[symbol] + 0.1 * ratio

            # Only emit event for moves above a small threshold
            threshold = max(0.003, base_vol * 1.5)
            if abs(pct) >= threshold:
                verb = "rises" if pct > 0 else "falls"
                desc = (f"{symbol} {verb} {abs(pct)*100:.1f}% "
                        f"to ${new:.2f}")
                events.append(MarketEvent(
                    event_type="price_move",
                    symbol=symbol,
                    description=desc,
                    magnitude=pct * 100,
                    timestamp=self.current_time,
                ))

        # 2. Scripted news (overrides random price move for that symbol)
        if self.step in self.SCRIPTED_NEWS:
            symbol, headline, impact = self.SCRIPTED_NEWS[self.step]
            if symbol:
                old  = self.current_prices[symbol]
                new  = max(1.0, round(old * (1.0 + impact), 2))
                self.current_prices[symbol] = new
                self.price_history[symbol][-1] = new
                base_vol = self._base_step_vol(symbol)
                if base_vol > 0:
                    self.vol_scale[symbol] = min(3.0, abs(impact) / base_vol)
            else:
                # Market-wide: apply to all symbols
                for sym in self.SYMBOLS:
                    old = self.current_prices[sym]
                    self.current_prices[sym] = max(1.0, round(old * (1.0 + impact), 2))
                    self.price_history[sym][-1] = self.current_prices[sym]
                    base_vol = self._base_step_vol(sym)
                    if base_vol > 0:
                        self.vol_scale[sym] = min(3.0, abs(impact) / base_vol)

            events.append(MarketEvent(
                event_type="news",
                symbol=symbol,
                description=headline,
                magnitude=impact * 100,
                timestamp=self.current_time,
            ))

        # 3. Advance clock
        self.current_time += datetime.timedelta(seconds=self.sec_per_step)
        self.step += 1
        self.all_events.extend(events)

        return events

    # -----------------------------------------------------------------------
    # Order execution
    # -----------------------------------------------------------------------

    def execute_order(self, agent_name: str, order: dict) -> dict:
        """
        Fill a trading order at current market price.

        order keys: type ("buy"/"sell"), symbol, quantity
        Returns a fill dict with status, fill_price, total_value.
        """
        symbol = order.get("symbol")
        qty    = int(order.get("quantity", 0))

        if symbol not in self.current_prices or qty <= 0:
            return {"status": "rejected", "reason": "invalid symbol or quantity",
                    "agent": agent_name, "symbol": symbol, "quantity": qty}

        mid_price = self.current_prices[symbol]
        spread_bps = self.BASE_SPREAD_BPS.get(symbol, 3.0)
        liq = max(1, self.LIQUIDITY.get(symbol, 10000))
        impact_bps = (qty / liq) * self.IMPACT_BPS_AT_FULL_LIQ
        slippage_bps = self.BASE_SLIPPAGE_BPS + impact_bps

        half_spread = (spread_bps / 10000.0) / 2.0
        slip = slippage_bps / 10000.0

        side = order.get("type")
        if side == "buy":
            fill_price = mid_price * (1.0 + half_spread + slip)
        else:
            fill_price = mid_price * (1.0 - half_spread - slip)

        fill_price = round(fill_price, 2)
        gross_value = round(fill_price * qty, 2)
        commission = max(self.MIN_COMMISSION,
                         round(self.COMMISSION_PER_SHARE * qty, 2))
        if side == "buy":
            total_value = round(gross_value + commission, 2)
        else:
            total_value = round(gross_value - commission, 2)

        fill = {
            "status":      "filled",
            "agent":       agent_name,
            "type":        order["type"],
            "symbol":      symbol,
            "quantity":    qty,
            "fill_price":  fill_price,
            "gross_value": gross_value,
            "commission":  commission,
            "total_value": total_value,
            "timestamp":   self.current_time.strftime("%Y-%m-%d %H:%M"),
            "step":        self.step,
        }
        self.order_log.append(fill)
        return fill

    # -----------------------------------------------------------------------
    # Query helpers
    # -----------------------------------------------------------------------

    def get_snapshot(self) -> dict:
        """Return a lightweight market state dict suitable for LLM prompts."""
        return {
            "step":      self.step,
            "timestamp": self.current_time.strftime("%Y-%m-%d %H:%M"),
            "prices":    dict(self.current_prices),
        }

    def get_price_change_pct(self, symbol: str, lookback: int = 1) -> float:
        """Return the % price change over the last `lookback` steps."""
        hist = self.price_history.get(symbol, [])
        if len(hist) < lookback + 1:
            return 0.0
        old = hist[-(lookback + 1)]
        new = hist[-1]
        if old == 0:
            return 0.0
        return round((new - old) / old * 100, 2)

    def prices_str(self) -> str:
        """One-line price summary for prompts."""
        return ", ".join(
            f"{s}=${p:.2f}" for s, p in self.current_prices.items()
        )
