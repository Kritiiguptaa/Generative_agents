"""
File: fetch_historical_data.py
Description: One-time fetch of real 1-minute OHLC bars for the trading-sim
  symbols, cached to market_data/historical_prices.json so the simulation
  can replay real trading sessions deterministically
  (see historical_market_environment.py).

  Run this manually whenever you want to refresh the cached days. The
  simulation itself never calls yfinance at runtime -- it only reads the
  cached JSON, which keeps every sim run reproducible.

Usage:
    pip install yfinance
    python fetch_historical_data.py
"""

import json
from pathlib import Path

import yfinance as yf

SYMBOLS = ["NVDA", "AAPL", "TSLA", "AMD", "GOOGL"]
OUTPUT_PATH = Path(__file__).resolve().parent / "market_data" / "historical_prices.json"

# Yahoo's 1-minute bars only go back ~7 calendar days, so this is the
# practical ceiling -- it is not an arbitrary choice.
N_DAYS = 3


def fetch_and_cache(n_days: int = N_DAYS) -> None:
    raw = yf.download(
        tickers=SYMBOLS,
        period="7d",
        interval="1m",
        group_by="ticker",
        auto_adjust=False,
        progress=False,
    )

    if raw.empty:
        raise RuntimeError("yfinance returned no data -- check tickers/network.")

    # Not every symbol has bars on every minute/day. Find dates where all
    # five symbols have data, so the cached series stay aligned.
    common_dates = None
    per_symbol_close = {}
    for symbol in SYMBOLS:
        closes = raw[symbol]["Close"].dropna()
        per_symbol_close[symbol] = closes
        dates = {ts.date() for ts in closes.index}
        common_dates = dates if common_dates is None else (common_dates & dates)

    if not common_dates:
        raise RuntimeError("No trading day has data for all symbols.")

    target_dates = sorted(common_dates)[-n_days:]
    if len(target_dates) < n_days:
        print(f"Warning: only {len(target_dates)} trading day(s) available in the "
              f"7-day window (requested {n_days}). Yahoo's 1-minute history is capped "
              f"at ~7 calendar days -- this is the most you can get for free at this "
              f"granularity.")

    series = {symbol: [] for symbol in SYMBOLS}
    day_lengths = []
    for day in target_dates:
        day_series = {}
        for symbol in SYMBOLS:
            closes = per_symbol_close[symbol]
            day_closes = closes[[ts.date() == day for ts in closes.index]]
            day_closes = day_closes.sort_index().ffill()
            day_series[symbol] = [round(float(v), 2) for v in day_closes.values]

        # Trim this day to the shortest series so symbols stay aligned even
        # if one symbol is missing a few bars that day.
        day_min_len = min(len(v) for v in day_series.values())
        for symbol in SYMBOLS:
            series[symbol].extend(day_series[symbol][:day_min_len])
        day_lengths.append(day_min_len)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "dates": [str(d) for d in target_dates],
            "day_lengths": day_lengths,
            "symbols": series,
        }, f, indent=2)

    total = sum(day_lengths)
    print(f"Cached {len(target_dates)} day(s), {day_lengths} bars/day "
          f"({total} total minute bars/symbol) for {[str(d) for d in target_dates]} "
          f"-> {OUTPUT_PATH}")


if __name__ == "__main__":
    fetch_and_cache()
