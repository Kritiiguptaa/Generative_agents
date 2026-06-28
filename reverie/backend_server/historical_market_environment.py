"""
File: historical_market_environment.py
Description: Variant of MarketEnvironment that replays real, cached 1-minute
  price bars (produced by fetch_historical_data.py) instead of a synthetic
  Gaussian/regime random walk for the price *path*.

  Everything else is inherited unchanged from market_environment.py:
  the scripted contradictory-news calendar (SCRIPTED_NEWS), the order
  execution microstructure model (spread/slippage/commission), and the
  query helpers. Only step 1 of tick() -- how prices move absent news --
  is replaced with a lookup into the cached real session(s).

  The cache can hold multiple trading days back-to-back. current_time
  advances by sec_per_step every tick regardless of market hours (same as
  the base class), so a multi-day run will pass through a simulated
  overnight close. Real price bars only exist for market-open minutes, so
  a separate bar index -- advanced only while the market is open -- is used
  to walk the cached array; during closed hours prices hold flat and no
  price_move events fire. This also makes is_open available as a plain
  attribute, which action_filtering._market_is_open() reads directly
  instead of re-deriving it from current_time.

  Once the cache is exhausted (e.g. a --steps run longer than the cached
  days), the last known price is held flat with no event, so longer runs
  degrade gracefully instead of crashing.
"""

import datetime
import json
from datetime import time
from pathlib import Path
from typing import List

from market_environment import MarketEnvironment, MarketEvent

DEFAULT_CACHE_PATH = Path(__file__).resolve().parent / "market_data" / "historical_prices.json"

MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)


class HistoricalMarketEnvironment(MarketEnvironment):

    def __init__(self, seed: int = 42, sec_per_step: int = 60,
                 cache_path: Path = DEFAULT_CACHE_PATH):
        super().__init__(seed=seed, sec_per_step=sec_per_step)

        with open(cache_path, encoding="utf-8") as f:
            cached = json.load(f)
        self.historical_dates = cached["dates"]
        self.historical_prices = cached["symbols"]

        # Advances only on ticks where the market is open -- NOT the same
        # as self.step, which also counts simulated overnight/closed ticks.
        self._bar_index = 0
        self.is_open = True

        # Start the sim at the real session's opening prices instead of the
        # synthetic BASE_PRICES.
        for symbol in self.SYMBOLS:
            series = self.historical_prices.get(symbol, [])
            if series:
                self.current_prices[symbol] = series[0]
                self.price_history[symbol] = [series[0]]

    def tick(self) -> List[MarketEvent]:
        events: List[MarketEvent] = []

        self.is_open = MARKET_OPEN <= self.current_time.time() < MARKET_CLOSE

        # 1. Real price moves (replayed from the cached session), only
        #    while the market is open. Closed hours hold prices flat.
        if self.is_open:
            idx = self._bar_index
            for symbol in self.SYMBOLS:
                series = self.historical_prices.get(symbol, [])
                old = self.current_prices[symbol]
                new = series[idx] if idx < len(series) else old
                self.current_prices[symbol] = new
                self.price_history[symbol].append(new)

                pct = (new - old) / old if old else 0.0
                if abs(pct) >= 0.003:
                    verb = "rises" if pct > 0 else "falls"
                    desc = f"{symbol} {verb} {abs(pct)*100:.1f}% to ${new:.2f}"
                    events.append(MarketEvent(
                        event_type="price_move",
                        symbol=symbol,
                        description=desc,
                        magnitude=pct * 100,
                        timestamp=self.current_time,
                    ))
            self._bar_index += 1
        else:
            for symbol in self.SYMBOLS:
                self.price_history[symbol].append(self.current_prices[symbol])

        # 2. Scripted news (same overlay logic as MarketEnvironment.tick)
        if self.step in self.SCRIPTED_NEWS:
            symbol, headline, impact = self.SCRIPTED_NEWS[self.step]
            if symbol:
                old = self.current_prices[symbol]
                new = max(1.0, round(old * (1.0 + impact), 2))
                self.current_prices[symbol] = new
                self.price_history[symbol][-1] = new
            else:
                for sym in self.SYMBOLS:
                    old = self.current_prices[sym]
                    self.current_prices[sym] = max(1.0, round(old * (1.0 + impact), 2))
                    self.price_history[sym][-1] = self.current_prices[sym]

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
