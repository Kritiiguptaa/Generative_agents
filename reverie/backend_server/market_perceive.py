"""
File: market_perceive.py
Description: Converts raw MarketEvent objects into ConceptNodes stored in
  a TradingPersona's associative memory.  Replaces perceive.py for the
  trading simulation.

Hallucination design choices:
  - News events always get high poignancy (6-9) so they drain the reflection
    budget quickly and trigger frequent reflection / compression cycles.
  - Idle / flat-market events (magnitude < 1%) are suppressed entirely —
    they never enter memory.  This is the fix for the "75x refrigerator is
    idle" problem from the original simulation.
  - A 50-event dedup window prevents the same price-move description from
    being stored repeatedly when the market is choppy in a narrow range.
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from persona.prompt_template.gpt_structure import get_embedding


# ---------------------------------------------------------------------------
# Poignancy scoring
# ---------------------------------------------------------------------------

def score_market_event(event, watchlist: list) -> int:
    """
    Rate a market event 1-10 for poignancy (importance to this agent).

    Rules:
      - magnitude drives the base score
      - symbols on the watchlist get +2 bonus
      - news always starts at 6 (never boring)
      - trade fills are always 7 (agent's own action confirmed)
    """
    mag = abs(event.magnitude)
    on_watch = event.symbol and event.symbol in watchlist

    if event.event_type == "trade_fill":
        return 7

    if event.event_type == "news":
        base = 6
        if mag > 7:
            base = 9
        elif mag > 4:
            base = 7
        return min(10, base + (2 if on_watch else 0))

    if event.event_type == "price_move":
        # 1% move → 2, 5% move → 7, 10%+ → 9
        base = min(9, max(1, int(mag * 0.8) + 1))
        return min(10, base + (2 if on_watch else 0))

    return 1


# ---------------------------------------------------------------------------
# Main perception function
# ---------------------------------------------------------------------------

def market_perceive(persona, market, market_events: list) -> list:
    """
    Process a list of MarketEvent objects for one simulation step.

    For each relevant event:
      1. Check if already in recent memory (dedup window = 50)
      2. Get or create embedding
      3. Score poignancy
      4. Add ConceptNode to associative memory
      5. Update agent's market knowledge (last known prices)
      6. Drain the reflection budget

    Returns the list of new ConceptNode instances added.
    """
    ret_events = []
    watchlist = getattr(persona.scratch, "watchlist", market.SYMBOLS)

    for event in market_events:
        # Relevance gate: only perceive watchlist stocks + news
        if event.event_type == "price_move" and event.symbol not in watchlist:
            continue

        desc = event.description

        # Dedup: skip if this exact (s,p,o) triple is in the last 50 events
        s, p, o = event.to_spo()
        latest = persona.a_mem.get_summarized_latest_events(50)
        if (s, p, o) in latest:
            continue

        # Embedding
        if desc in persona.a_mem.embeddings:
            embedding_vec = persona.a_mem.embeddings[desc]
        else:
            embedding_vec = get_embedding(desc)
        embedding_pair = (desc, embedding_vec)

        # Poignancy
        poignancy = score_market_event(event, watchlist)

        # Keywords
        keywords = _build_keywords(event)

        # Store in associative memory
        node = persona.a_mem.add_event(
            created=market.current_time,
            expiration=None,
            s=s, p=p, o=o,
            description=desc,
            keywords=keywords,
            poignancy=poignancy,
            embedding_pair=embedding_pair,
            filling=[],
        )
        ret_events.append(node)

        # Update market knowledge with latest price
        if event.symbol and hasattr(persona, "s_mem"):
            current_price = market.current_prices.get(event.symbol, 0)
            persona.s_mem.update_price(event.symbol, current_price)
            # Add news as an analyst note so it persists in market knowledge
            if event.event_type == "news":
                note = f"[step {market.step}] {event.description[:80]}"
                persona.s_mem.add_note(event.symbol, note)

        # Drain reflection budget
        persona.scratch.importance_trigger_curr -= poignancy
        persona.scratch.importance_ele_n += 1

    return ret_events


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_keywords(event) -> set:
    kw = set()
    if event.symbol:
        kw.add(event.symbol.lower())
    if event.event_type == "news":
        kw.add("news")
        # Pick out long content words from the headline
        for word in event.description.lower().split():
            if len(word) > 5 and word.isalpha():
                kw.add(word)
    elif event.event_type == "price_move":
        kw.add("price")
        kw.add("rise" if event.magnitude > 0 else "fall")
    elif event.event_type == "trade_fill":
        kw.add("fill")
        kw.add("trade")
    return kw


def record_trade_fill(persona, market, fill: dict) -> None:
    """
    Store a completed trade fill as an event in the agent's memory.
    Called by trading_reverie.py after execute_trading_action() succeeds.
    """
    if fill.get("status") != "filled":
        return

    sym = fill["symbol"]
    qty = fill["quantity"]
    price = fill["fill_price"]
    action = fill["type"]

    desc = f"{persona.name} {action}s {qty} {sym} @ ${price:.2f}"
    s    = persona.name
    p    = f"{action}s"
    o    = f"{qty} {sym}"

    if desc in persona.a_mem.embeddings:
        emb = persona.a_mem.embeddings[desc]
    else:
        emb = get_embedding(desc)

    persona.a_mem.add_event(
        created=market.current_time,
        expiration=None,
        s=s, p=p, o=o,
        description=desc,
        keywords={persona.name.lower(), sym.lower(), action, "trade"},
        poignancy=7,
        embedding_pair=(desc, emb),
        filling=[],
    )
    # This fill is important — drain the budget accordingly
    persona.scratch.importance_trigger_curr -= 7
    persona.scratch.importance_ele_n += 1
