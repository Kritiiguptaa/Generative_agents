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

# Windows' default console codepage (cp1252) can't encode the box-drawing
# characters used in this file's progress output -- force UTF-8 so prints
# don't crash mid-run.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

# ── path setup ──────────────────────────────────────────────────────────────
THIS_FILE   = Path(__file__).resolve()
BACKEND_DIR = THIS_FILE.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
os.chdir(BACKEND_DIR)
# ────────────────────────────────────────────────────────────────────────────

from market_environment import MarketEnvironment
from historical_market_environment import HistoricalMarketEnvironment

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
from market_perceive    import (market_perceive, record_trade_fill,
                                record_order_feedback)
from trading_persona    import TradingPersona
from trading_daily_plan import ensure_daily_plan, apply_interaction_to_plan, current_focus_str
from trading_interactions import maybe_interaction
from middleware.action_filtering    import (
    run_action_filtering_step,
    log_action,
    _parse_json_response,
    _split_combined_action,
    _market_is_open,
    _get_tradeable_symbols,
    coerce_quantity,
)
from middleware.persona_anchoring   import (
    score_persona_consistency,
    PERSONA_DRIFT_THRESHOLD,
)
from middleware.memory_compression  import compress_memories
from middleware.id_rag import (
    id_rag_anchor,
    update_graph_belief,
    update_graph_relationship,
    update_graph_current_situation,
)

from persona.cognitive_modules.retrieve import new_retrieve
from persona.cognitive_modules.reflect  import reflect
from persona.prompt_template.gpt_structure import ollama_request, get_embedding
from utils import fs_storage


# ===========================================================================
# LLM Decision
# ===========================================================================

def build_memory_context(retrieved: dict,
                         persona: TradingPersona,
                         market:  MarketEnvironment,
                         use_middleware: bool = True):
    """
    Turn new_retrieve() output into the memory block the prompt carries.

    Split out of make_trading_decision() so paired_eval.py can build BOTH arms'
    contexts from one retrieval result without also invoking a decision.

    Returns (memory_context, compression_stats).
    """
    if use_middleware:
        # Middleware layer: compress the retrieval output before it reaches the
        # LLM. Replaces the old arbitrary nodes[:6] truncation.
        memory_context, mc_stats = compress_memories(retrieved, persona, market)

        # ID-RAG: retrieve only the identity facts relevant to this context
        # and prepend them. Replaces the static full-block anchor.
        memory_context = id_rag_anchor(persona, memory_context)
        return memory_context, mc_stats

    # Baseline: hand the LLM everything new_retrieve() returned, verbatim.
    # No dedup, no supersession pruning, no budget -- this is the raw
    # context the middleware exists to clean up.
    raw_nodes = []
    for nodes in retrieved.values():
        raw_nodes.extend(nodes)
    memory_context = "\n".join(n.description for n in raw_nodes)
    # Report the size of the context even though nothing compressed it. This
    # arm was observed choosing "analyze" on 100% of decisions, and the
    # leading hypothesis is that the uncompressed block is large enough to
    # crowd out the instruction -- untestable while the stats said only
    # {"enabled": False}. nodes_out == nodes_in because nothing was dropped.
    mc_stats = {
        "enabled":           False,
        "nodes_in":          len(raw_nodes),
        "nodes_out":         len(raw_nodes),
        "dropped_duplicate": 0,
        "dropped_stale":     0,
        "dropped_budget":    0,
        "annotated_stale":   0,
        "compression_ratio": 1.0,
        "chars_in":          len(memory_context),
        "chars_out":         len(memory_context),
    }
    return memory_context, mc_stats


def make_trading_decision(persona: TradingPersona,
                          market:  MarketEnvironment,
                          retrieved: dict,
                          action_log_path: str,
                          use_middleware: bool = True) -> dict:
    """
    Ask the LLM to decide what to do next.
    Returns (decision, compression_stats, filter_stats), where decision is
    {"action", "symbol", "quantity", "reasoning"} and filter_stats carries the
    pre-filter request -- see run_action_filtering_step() for why that has to be
    threaded out of here rather than recovered downstream.

    Middleware arm (use_middleware=True), applied in order:
      1. compress_memories()  — dedup, prune superseded news, enforce budget
      2. id_rag_anchor()      — prepend context-relevant identity facts
      3. action filtering     — constrain the LLM to legal actions

    Baseline arm (use_middleware=False) — the ablation comparison:
      every retrieved memory is concatenated verbatim and handed to the LLM
      with no compression, no identity retrieval, and no legal-action menu.
      Return shape is identical so both arms are scored by the same
      downstream code (execute_trading_action, _check_stale_reasoning,
      score_persona_consistency, _generate_report).
    """
    memory_context, mc_stats = build_memory_context(
        retrieved, persona, market, use_middleware)

    def _call_llm(prompt: str) -> str:
        # format="json": grammar-constrained decoding. phi3:mini otherwise
        #   intermittently emits invalid JSON syntax (bare `end="..."` keys,
        #   // comments, trailing commas) that json.loads() rejects, which
        #   showed up as "response was not valid JSON" fallbacks.
        # max_tokens=300: 120 truncated verbose reasoning mid-string
        #   (done_reason="length"), producing unterminated JSON. Measured
        #   completions run ~70-135 tokens, so 300 is real headroom.
        # No stop=["\n\n"]: in JSON mode the object ends when it closes, and a
        #   blank line inside pretty-printed JSON would cut it off early.
        # Identical settings in both arms -- the switch changes what goes into
        # the prompt, never how the model is sampled.
        return ollama_request(prompt, max_tokens=300, timeout=300, format="json")

    if use_middleware:
        decision, filter_stats = run_action_filtering_step(
            persona,
            market,
            memory_context,
            _call_llm,
            action_log_path,
        )
    else:
        decision, filter_stats = free_form_decision(
            persona,
            market,
            memory_context,
            _call_llm,
            action_log_path,
        )
    return decision, mc_stats, filter_stats


def free_form_decision(persona: TradingPersona,
                       market:  MarketEnvironment,
                       memory_context: str,
                       llm_call,
                       action_log_path: str) -> dict:
    """
    Baseline (no-middleware) decision path.

    Gives the LLM the persona, the portfolio, the prices and the full memory
    context, then takes whatever it returns -- no legal-action menu, no
    legality validation, no retry-on-illegal. An unaffordable buy or a sell of
    unowned stock is passed straight through to execute_trading_action(), which
    is what flags it as a hallucination. That is the whole point of this arm.

    The JSON *parsing* helpers from action_filtering are reused deliberately:
    stripping markdown fences and splitting a combined {"action": "BUY NVDA"}
    field is small-model formatting cleanup, not decision constraint. Doing it
    identically in both arms keeps the comparison about decision quality rather
    than about which arm had a fussier parser.

    For the same reason this prompt states the same tradeable universe, market
    status and action vocabulary that the filtered arm's legal-action menu
    states. The ablation is "is the menu enforced?", not "did the two arms even
    know about the same symbols?" -- previously the filtered arm was restricted
    to the watchlist and offered no ANALYZE, so the arms' action distributions
    were not comparable.

    Returns (decision, stats) with the same shape run_action_filtering_step()
    returns. illegal_request is always False here: this arm has no legality
    check, so illegality is detected downstream by execute_trading_action().
    """
    s = persona.scratch
    tradeable = _get_tradeable_symbols(persona, market)
    market_open = "OPEN" if _market_is_open(market) else "CLOSED"
    prompt = f"""You are {s.name}, a {getattr(s, 'trader_type', None) or 'generalist'} trader.
Traits: {getattr(s, 'innate', '')}
Background: {getattr(s, 'learned', '')}
Current situation: {getattr(s, 'currently', '')}
Risk tolerance: {getattr(s, 'risk_tolerance', '')}

Current state:
- Cash: ${s.cash_balance:,.2f}
- Holdings: {s.positions}

Market:
- Status: {market_open}
- Prices: {market.current_prices}
- Tradeable symbols: {', '.join(tradeable)}

Memory:
{memory_context}

Rules:
- You cannot borrow money or trade on margin. A purchase's total cost
  (price x quantity) must not exceed your cash.
- You cannot sell shares you do not hold.

Decide your next action: buy, sell, or hold.
Respond ONLY in this JSON format:
{{"action": "buy/sell/hold", "symbol": "TICKER or null", "quantity": number, "reasoning": "why"}}
"""

    name = s.name
    fallback = {"action": "hold", "symbol": None, "quantity": 0, "reasoning": ""}
    stats = {"status": "success", "illegal_request": False,
             "error_reason": "", "requested": {}}

    def _fail(reason: str):
        log_action(action_log_path, name, fallback, "fallback", reason)
        stats["status"] = "fallback"
        stats["error_reason"] = reason
        return fallback, stats

    raw = llm_call(prompt)
    response, error = _parse_json_response(raw)
    if response is None:
        return _fail(error or "unparseable")

    # ── normalise shape only (same normalisation the filtered arm applies) ──
    action_type, symbol_hint = _split_combined_action(response.get("action"))
    if action_type is None:
        return _fail("missing action")
    action = action_type.lower()
    # Same normalisation the filtered arm applies: ANALYZE is not offered, but
    # if the model emits it anyway that means "no trade", not an illegal one.
    if action == "analyze":
        action = "hold"
    if action not in ("buy", "sell", "hold"):
        return _fail(f"unknown action {action!r}")

    symbol = response.get("symbol")
    if symbol is None:
        symbol = response.get("ticker")
    if symbol is None:
        symbol = symbol_hint
    if isinstance(symbol, str):
        symbol = symbol.strip().upper()
        # phi3:mini emits the *string* "null"/"none" rather than JSON null.
        if symbol in ("NULL", "NONE", ""):
            symbol = None

    # Same coercion the filtered arm applies -- see coerce_quantity().
    quantity = coerce_quantity(response.get("quantity"))
    if quantity is None:
        quantity = 0

    decision = {
        "action":    action,
        "symbol":    symbol,
        "quantity":  quantity,
        "reasoning": response.get("reasoning", ""),
    }
    # "success" here means "parsed", not "legal" -- nothing was checked against
    # cash, holdings or risk limits on this path. The requested action is
    # therefore identical to the executed one; it is still logged so both arms
    # produce the same CSV columns and can be diffed directly.
    stats["requested"] = {"action": action, "symbol": symbol, "quantity": quantity}
    log_action(action_log_path, name, decision, "success",
               requested=stats["requested"])
    return decision, stats


# ===========================================================================
# Live agent state
# ===========================================================================

def refresh_currently(persona: TradingPersona, market: MarketEnvironment) -> str:
    """
    Rewrite scratch.currently from the agent's ACTUAL portfolio state.

    The bootstrap `currently` strings describe every agent as poised to enter a
    position but not yet in one -- "wants confirmation of supply tightening
    before entering", "watching for a dip entry", "considering adding 50 more
    shares if price holds above $191". Nothing in the simulation ever updated
    the field: the only writer is plan.py's revise_identity(), which belongs to
    the village sim and is never called from the trading loop. So the text the
    agent read at step 199 was the text it read at step 0.

    Measured consequence: all three agents restated their own `currently` text
    back as reasoning and chose not to trade on 600/600 decisions, even though
    their stated conditions were satisfied -- Sara's "price holds above $191"
    was true on 272 of 272 observations, and Marcus saw 122 down-ticks while
    waiting for "a dip". A static intention keeps an agent permanently
    pre-entry because no prompt ever tells it the condition has been met.

    `currently` now carries facts (cash, holdings, mark-to-market, headroom).
    Intent still lives in trading_strategy / innate / learned, which is where
    persona differentiation belongs -- those are stable traits, not a status
    that goes stale.
    """
    s = persona.scratch
    prices = market.current_prices
    pv = persona.portfolio_value(prices)
    cash = float(s.cash_balance)
    cash_pct = (cash / pv * 100.0) if pv else 0.0

    if s.positions:
        holdings = []
        for sym, pos in s.positions.items():
            px = prices.get(sym, pos["avg_price"])
            avg = pos["avg_price"] or px
            pnl_pct = ((px - avg) / avg * 100.0) if avg else 0.0
            # Position weight. The bootstrap `currently` stated this ("roughly
            # 8.7% of his $441,040 portfolio") and the first version of this
            # function dropped it, leaving only raw share counts. Measured
            # consequence: Alex Chen held for 200/200 steps justifying it with
            # "current position in NVDA is already significant" while NVDA was
            # 10% of his book and $402,600 sat idle. Absolute quantities do not
            # tell an agent whether it is concentrated or under-deployed.
            weight = (pos["qty"] * px / pv * 100.0) if pv else 0.0
            holdings.append(f"{pos['qty']} {sym} at avg ${avg:.2f} "
                            f"(now ${px:.2f}, {pnl_pct:+.1f}%, "
                            f"{weight:.1f}% of portfolio)")
        holdings_str = "holds " + "; ".join(holdings)
    else:
        holdings_str = "holds no open positions"

    # The spendable budget is the risk cap AND the cash, whichever binds first.
    # Reporting the uncapped risk cap told Marcus Webb "maximum single trade
    # $5,017" while his cash was $76.41 and get_legal_actions() -- which does
    # apply min(cash, risk_cap) -- offered him no BUY at all. He then requested
    # `buy NVDA x10` ($2,030) on 20 near-consecutive steps. The prompt was
    # self-contradicting, so those requests are only partly the model's fault.
    risk_cap  = pv * float(getattr(s, "risk_limit_per_trade", 0.05) or 0.05)
    max_trade = max(0.0, min(risk_cap, cash))

    # A nonzero budget that still buys no share reads as an invitation to
    # purchase. Compare against the cheapest symbol the agent may trade so the
    # wording matches what is actually possible.
    tradeable = _get_tradeable_symbols(persona, market)
    affordable = [prices[sym] for sym in tradeable
                  if prices.get(sym) and prices[sym] <= max_trade]

    if not affordable:
        budget_str = (f"You cannot afford to buy any share right now "
                      f"(spendable budget ${max_trade:,.0f}). Your only "
                      f"options are HOLD or SELL. You cannot borrow.")
    else:
        budget_str = (f"Maximum value for a single trade is ${max_trade:,.0f} "
                      f"(risk cap ${risk_cap:,.0f}, cash ${cash:,.0f}, "
                      f"whichever is lower). You cannot borrow.")

    s.currently = (
        f"{s.name} has ${cash:,.0f} in cash ({cash_pct:.1f}% of portfolio) "
        f"and {holdings_str}. Total portfolio value ${pv:,.0f}. {budget_str}"
    )
    return s.currently


# ===========================================================================
# Action Filtering + Execution
# ===========================================================================

def execute_trading_action(persona: TradingPersona,
                           market:  MarketEnvironment,
                           decision: dict) -> dict:
    """
    Validate decision against hard portfolio constraints, then execute.

    Returns a result dict with keys:
      outcome_str, fill (or None), filtered (bool), hallucination (bool),
      halluc_kind (str), requested_quantity (int), filled_quantity (int).

    `filtered` means the order was rejected outright. `hallucination` means the
    LLM asked for something it was not entitled to -- which includes the
    rejected cases AND the *clamped* ones. Those used to be scored clean: an
    agent asking to buy 5000 shares it could not afford was silently reduced to
    the affordable quantity, printed as an informational "[ACTION FILTER]" line
    and logged with filtered=False. That is the exact hallucination this module
    was built to count (see the module docstring's Marcus Webb example), so
    repairing it and reporting success made the headline metric unable to
    observe its own primary trigger.

    Rules enforced (this is the ground-truth validator that reveals
    hallucinations; the middleware arm additionally prevents them upstream):
      0. No trading while the market is closed.
      0b. Symbol must be in the agent's tradeable universe.
      1. Buy cost must not exceed available cash.
      2. Agent cannot sell more shares than it owns.
      3. Single trade value must not exceed risk_limit_per_trade × portfolio.
    """
    action   = decision.get("action", "hold")
    symbol   = decision.get("symbol")
    reason   = decision.get("reasoning", "")
    try:
        requested = int(decision.get("quantity") or 0)
    except (TypeError, ValueError):
        requested = 0

    name = persona.scratch.name

    def _result(outcome_str, fill=None, filtered=False,
                hallucination=False, kind="", filled=0):
        return {"outcome_str": outcome_str, "fill": fill, "filtered": filtered,
                "hallucination": hallucination, "halluc_kind": kind,
                "requested_quantity": requested, "filled_quantity": filled}

    if action in ("hold", "analyze") or not symbol:
        verb = "holds" if action == "hold" else "analyzes market"
        return _result(f"{name} {verb}. {reason}")

    # ── Rule 0: market hours. Previously only the middleware arm checked this
    # (via get_legal_actions), so the baseline arm could trade at 3am while the
    # filtered arm was frozen at HOLD -- an asymmetry that shows up as an
    # action-distribution difference on any run longer than one session.
    if not _market_is_open(market):
        return _result(
            f"[FILTERED] {name} tried to {action.upper()} {requested} {symbol} "
            f"but the market is closed.",
            filtered=True, hallucination=True, kind="market_closed")

    # ── Rule 0b: symbol must be in the agent's tradeable universe ──────────
    tradeable = _get_tradeable_symbols(persona, market)
    if symbol not in tradeable:
        return _result(
            f"[FILTERED] {name} tried to {action.upper()} {symbol!r}, which is "
            f"not in its tradeable universe ({', '.join(tradeable)}).",
            filtered=True, hallucination=True, kind="untradeable_symbol")

    price = market.current_prices.get(symbol)
    if not price:
        return _result(f"[FILTERED] {name}: unknown symbol {symbol!r}",
                       filtered=True, hallucination=True, kind="unknown_symbol")

    # ── portfolio value for risk-limit check ───────────────────────────────
    pv = persona.portfolio_value(market.current_prices)

    if action == "buy":
        requested_cost = price * requested
        max_by_cash    = int(persona.scratch.cash_balance / price)
        max_by_risk    = int(pv * persona.scratch.risk_limit_per_trade / price)
        max_qty        = min(max_by_cash, max_by_risk)

        if max_qty <= 0:
            return _result(
                f"[FILTERED] {name} tried to buy {requested} {symbol} but "
                f"insufficient cash (${persona.scratch.cash_balance:,.0f}) "
                f"or risk limit reached.",
                filtered=True, hallucination=True,
                kind="unaffordable" if requested > max_by_cash else "over_risk_limit")

        # ── HALLUCINATION: agent asked for more than it was entitled to. The
        # order still executes at the clamped size (that is the middleware's
        # job), but the *request* was illegal and is counted as such.
        quantity = min(requested, max_qty)
        kind = ""
        if requested > max_qty:
            kind = "clamped_cash" if requested > max_by_cash else "clamped_risk"
            print(f"  [ACTION FILTER] {name} tried to buy {requested} {symbol} "
                  f"(${requested_cost:,.0f}) but cash="
                  f"${persona.scratch.cash_balance:,.0f}, risk cap={max_by_risk}. "
                  f"Adjusted to {quantity}.")

        fill = market.execute_order(name, {"type": "buy", "symbol": symbol,
                                           "quantity": quantity})
        # execute_order() returns {"status": "rejected", "reason": ...} with no
        # "total_value" key, so reading it unguarded raised KeyError and killed
        # the whole run mid-step. Treat a venue rejection as a filtered order.
        if fill.get("status") != "filled":
            return _result(
                f"[FILTERED] {name}'s {action.upper()} of {quantity} {symbol} "
                f"was rejected by the venue: "
                f"{fill.get('reason', 'unknown reason')}.",
                filtered=True, hallucination=False, kind="venue_rejected")

        # Update portfolio. Cost basis uses the actual fill price, not the mid:
        # cash is debited fill["total_value"], so booking avg_price at `price`
        # understated the basis by the spread, slippage and commission.
        s = persona.scratch
        s.cash_balance -= fill["total_value"]
        unit_cost = fill["total_value"] / quantity
        if symbol in s.positions:
            old   = s.positions[symbol]
            total = old["qty"] + quantity
            avg   = (old["qty"] * old["avg_price"] + quantity * unit_cost) / total
            s.positions[symbol] = {"qty": total, "avg_price": round(avg, 2)}
        else:
            s.positions[symbol] = {"qty": quantity, "avg_price": round(unit_cost, 2)}

        return _result(
            f"{name} BUYS {quantity} {symbol} @ ${price:.2f} "
            f"(total ${fill['total_value']:,.0f}). {reason}",
            fill=fill, hallucination=bool(kind), kind=kind, filled=quantity)

    elif action == "sell":
        owned = persona.scratch.positions.get(symbol, {}).get("qty", 0)

        # ── HALLUCINATION: agent tries to sell shares it doesn't own ────────
        if owned == 0:
            return _result(
                f"[FILTERED] {name} tried to SELL {requested} {symbol} "
                f"but owns 0 shares.",
                filtered=True, hallucination=True, kind="unowned")

        # ── HALLUCINATION: asking to sell 500 when you hold 10 is an illegal
        # request even though it is trivially satisfiable at a smaller size.
        quantity = min(requested, owned)
        kind = ""
        if requested > owned:
            kind = "clamped_holdings"
            print(f"  [ACTION FILTER] {name} tried to sell {requested} {symbol} "
                  f"but owns {owned}. Adjusted to {quantity}.")

        fill = market.execute_order(name, {"type": "sell", "symbol": symbol,
                                           "quantity": quantity})
        if fill.get("status") != "filled":
            return _result(
                f"[FILTERED] {name}'s SELL of {quantity} {symbol} was rejected "
                f"by the venue: {fill.get('reason', 'unknown reason')}.",
                filtered=True, hallucination=False, kind="venue_rejected")

        s = persona.scratch
        s.cash_balance += fill["total_value"]
        s.positions[symbol]["qty"] -= quantity
        if s.positions[symbol]["qty"] == 0:
            del s.positions[symbol]

        return _result(
            f"{name} SELLS {quantity} {symbol} @ ${price:.2f} "
            f"(total ${fill['total_value']:,.0f}). {reason}",
            fill=fill, hallucination=bool(kind), kind=kind, filled=quantity)

    return _result(f"{name} holds (unknown action).")


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
# Reasoning-vs-state grounding detector
# ===========================================================================

# Claims of the form "I already hold a lot of this" / "I have no room left".
_CONCENTRATION_CLAIMS = (
    "already significant", "already large", "already substantial",
    "already sizable", "already sizeable", "fully invested",
    "fully deployed", "no cash", "out of cash", "insufficient funds",
    "maximum allowed", "max allowed", "at my limit", "at the limit",
    "position limit", "no capital", "capital is tied up",
    "heavily weighted", "overexposed", "over-exposed", "concentrated",
)

# Claims of the form "I have plenty of room".
_CAPACITY_CLAIMS = (
    "plenty of cash", "plenty of capital", "ample cash", "significant cash",
    "cash on hand", "dry powder", "under-invested", "underinvested",
)

# A position at or below this share of portfolio value is not "significant";
# at or above the upper bound it genuinely is.
CONCENTRATION_FLOOR_PCT = 20.0
CAPACITY_FLOOR_PCT      = 20.0


def check_state_grounding(reasoning: str, persona: TradingPersona,
                          market: MarketEnvironment) -> str:
    """
    Detect reasoning that contradicts the agent's own portfolio.

    Everything else in this file validates the *action*. A HOLD is legal by
    construction (execute_trading_action returns early on it), so an agent that
    never trades cannot register a hallucination no matter what it claims. That
    is not a hypothetical gap: Alex Chen held on 200/200 steps in both arms
    while justifying it with "current position in NVDA is already significant"
    -- NVDA was ~10% of his book and $402,600 sat in cash. Both arms scored him
    perfectly clean, and middleware-Marcus held *more* than baseline-Marcus
    (193 vs 181), so an arm can improve its hallucination rate purely by
    trading less.

    This check is deliberately restricted to claims that can be settled
    arithmetically against cash / holdings / prices -- no keyword sentiment, no
    judgement about whether a trade was wise. Returns a description of the
    contradiction, or "" when the reasoning makes no checkable claim.
    """
    if not reasoning:
        return ""

    text = reasoning.lower()
    s = persona.scratch
    prices = market.current_prices
    pv = persona.portfolio_value(prices)
    if not pv:
        return ""

    cash     = float(s.cash_balance)
    cash_pct = cash / pv * 100.0
    findings = []

    # 1. "I'm concentrated / out of room" while sitting on idle cash.
    if any(c in text for c in _CONCENTRATION_CLAIMS):
        largest_pct = 0.0
        for sym, pos in (s.positions or {}).items():
            px = prices.get(sym, pos.get("avg_price") or 0.0)
            largest_pct = max(largest_pct, pos.get("qty", 0) * px / pv * 100.0)
        if largest_pct <= CONCENTRATION_FLOOR_PCT and cash_pct >= 25.0:
            findings.append(
                f"claims concentration//no-capacity but largest position is "
                f"{largest_pct:.1f}% of portfolio and cash is {cash_pct:.1f}% "
                f"(${cash:,.0f})")

    # 2. "I have plenty of cash" while effectively broke.
    if any(c in text for c in _CAPACITY_CLAIMS) and cash_pct < 5.0:
        findings.append(
            f"claims spare capital but cash is {cash_pct:.1f}% of portfolio "
            f"(${cash:,.0f})")

    # 3. Names a symbol as held that it does not hold.
    held = {sym.upper() for sym in (s.positions or {})}
    for sym in prices:
        if sym.upper() in held:
            continue
        for phrase in (f"my {sym.lower()} position", f"my position in {sym.lower()}",
                       f"holding {sym.lower()}", f"my {sym.lower()} shares"):
            if phrase in text:
                findings.append(f"refers to a {sym} position it does not hold")
                break

    return "; ".join(findings[:2])


# Steps within which an identical repeated request counts as the same episode.
EPISODE_GAP_STEPS = 3


def _count_episodes(halluc_requests: list) -> int:
    """
    Collapse consecutive identical hallucinated requests into one episode.

    `halluc_requests` is [(step, (action, symbol, quantity)), ...] for a single
    agent. A run of the same request with no more than EPISODE_GAP_STEPS
    between consecutive occurrences is one episode; a different request, or a
    longer gap, starts a new one.

    The gap is a reporting parameter, not a measurement: changing it changes
    the episode count, so the raw count is always reported next to it.
    """
    episodes = 0
    prev_key = None
    prev_step = None
    for step, key in sorted(halluc_requests, key=lambda r: (r[0] is None, r[0])):
        if (prev_key is None or key != prev_key or step is None
                or prev_step is None or step - prev_step > EPISODE_GAP_STEPS):
            episodes += 1
        prev_key, prev_step = key, step
    return episodes


# ===========================================================================
# Simulation class
# ===========================================================================

class TradingReverie:

    def __init__(self, sim_code: str, fork_sim_code: str = "base_trading",
                 use_middleware: bool = True, fresh: bool = False,
                 seed: int = 42):
        self.sim_code       = sim_code
        self.fork_sim_code  = fork_sim_code
        self.use_middleware = use_middleware
        self.seed           = seed
        self.sim_folder    = f"{fs_storage}/{sim_code}"
        # Seed is a parameter so an arm can be replicated. A single run of each
        # arm cannot support a claim about a rate difference -- run 03 produced
        # 3 baseline hallucination events in total -- and the two arms must
        # share a seed to be paired.
        self.market        = HistoricalMarketEnvironment(seed=seed)
        self.action_filter_log_path = str(
            Path(self.sim_folder) / "reverie" / "action_filter_log.csv"
        )

        # Copy base simulation folder if target doesn't exist yet.
        #
        # Reusing an existing --sim name used to silently RESUME it: personas
        # are persisted by _save(), the market is not, so the agents came back
        # holding positions they had bought at step-N prices while the market
        # was reconstructed at step 0. Two arms run under recycled sim names
        # therefore started from different portfolios -- which is how a baseline
        # and a middleware run of the same fork reported different starting
        # portfolio values for the same agent, making every cross-arm PnL and
        # rate comparison meaningless. Resuming is refused rather than repaired
        # because there is no market checkpoint to resume against.
        fork_path = Path(f"{fs_storage}/{fork_sim_code}")
        sim_path  = Path(self.sim_folder)
        if sim_path.exists() and fresh:
            shutil.rmtree(sim_path)
            print(f"[TradingReverie] --fresh: removed existing '{sim_code}'")
        if sim_path.exists():
            raise SystemExit(
                f"[TradingReverie] Simulation '{sim_code}' already exists at\n"
                f"  {sim_path}\n"
                f"Resuming is not supported: persona state is saved but market "
                f"state is not, so a resumed run starts agents with existing "
                f"positions against a step-0 market and is not comparable to a "
                f"fresh run.\n"
                f"Use a new --sim name, or pass --fresh to delete and re-fork it."
            )
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

        # Initialize daily plans for each agent. Refresh `currently` first --
        # generate_daily_plan() feeds it into the planning prompt, so the plan
        # would otherwise be built from the stale bootstrap "waiting to enter"
        # text this run is trying to get rid of.
        for persona in self.personas.values():
            refresh_currently(persona, self.market)
            ensure_daily_plan(persona, self.market, force=True)

        print(f"[TradingReverie] Middleware: "
              f"{'ENABLED' if self.use_middleware else 'DISABLED (baseline arm)'}")
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
        log_path = sim_path / "reverie" / "trading_log.json"
        interactions_path = sim_path / "reverie" / "trading_interactions.json"
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Mark the run live immediately, before the first step even
        # finishes, so a frontend poll landing right after startup sees
        # sim_running=True instead of a stale/absent flag.
        self._write_run_status(self.market.step, running=True)

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
                # ID-RAG: update relationship nodes for both agents
                agents_in = interaction.get("agents", [])
                summary   = interaction.get("summary", "")
                if len(agents_in) == 2 and summary:
                    update_graph_relationship(agents_in[0], agents_in[1], summary)
                    update_graph_relationship(agents_in[1], agents_in[0], summary)

            for name, persona in self.personas.items():
                try:
                    self._step_agent(name, persona, events, step, log)
                except Exception as exc:
                    print(f"  [{name}] UNHANDLED ERROR: {exc}")
                    traceback.print_exc()

            # ── Live flush ─────────────────────────────────────────────────
            # Write what we have after every step, not just at the end, so a
            # frontend poll mid-run sees this step's decisions immediately --
            # this is what makes the map view live instead of replay-only.
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(log, f, indent=2)
            # Interactions too -- the map renders these as the two agents
            # meeting in the aisle, so they have to land live alongside the
            # decisions rather than only at end-of-run.
            with open(interactions_path, "w", encoding="utf-8") as f:
                json.dump(interactions, f, indent=2)
            self._write_run_status(step, running=True)

            # ── Checkpoint ─────────────────────────────────────────────────
            if self.market.step % 100 == 0:
                self._save()
                print(f"  [Checkpoint] saved at market step {self.market.step}")

        # Final save + log
        self._save()
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log, f, indent=2)

        with open(interactions_path, "w", encoding="utf-8") as f:
            json.dump(interactions, f, indent=2)

        report = self._generate_report(log, n_steps)
        report_path = sim_path / "reverie" / "hallucination_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        self._print_report(report)

        # Run is done -- flip the live flag off so frontend pollers stop.
        last_step = log[-1]["step"] if log else self.market.step
        self._write_run_status(last_step, running=False)
        return log

    # -----------------------------------------------------------------------

    def _step_agent(self, name: str, persona: TradingPersona,
                    events: list, step: int, log: list):
        """Run one cognitive cycle for a single agent."""

        # Keep persona's internal clock in sync with market time.
        # reflect() and memory nodes use this for timestamps.
        persona.scratch.curr_time = self.market.current_time

        # Portfolio value BEFORE this step's trade. The log's portfolio_value is
        # recorded after execution, so taking the first entry as the starting
        # value silently excluded the first trade from PnL -- an agent that
        # bought on step 0 had that purchase's cost baked into its "start".
        pv_before = persona.portfolio_value(self.market.current_prices)

        # 2. Perceive
        market_perceive(persona, self.market, events)

        # 2b. Refresh the agent's self-description from live portfolio state,
        # before anything reads scratch.currently (the decision prompt in both
        # arms, the ID-RAG graph update below, and the daily planner all do).
        refresh_currently(persona, self.market)

        # 3. Reflect (compresses memory when budget exhausted)
        # Capture thought count before reflect so we can detect new insights.
        thoughts_before = len(persona.a_mem.seq_thought)
        reflect(persona)
        new_thoughts = persona.a_mem.seq_thought[thoughts_before:]

        # ID-RAG dynamic updates after reflect
        currently = getattr(persona.scratch, "currently", "") or ""
        update_graph_current_situation(name, currently)
        for thought_node in new_thoughts:
            desc = getattr(thought_node, "description", "") or ""
            if desc:
                update_graph_belief(name, desc)

        # 4. Retrieve — two focal points per agent per step
        focus = current_focus_str(persona, self.market)
        focal_points = [
            f"What should {name} trade right now?",
            f"Recent price action for {', '.join(persona.scratch.watchlist[:2])}",
            f"Current plan focus: {focus}",
        ]
        retrieved = new_retrieve(persona, focal_points, n_count=15)

        # 5. Decide  ← memory compression + ID-RAG anchoring applied inside
        decision, mc_stats, filter_stats = make_trading_decision(
            persona,
            self.market,
            retrieved,
            self.action_filter_log_path,
            use_middleware=self.use_middleware,
        )
        print(f"  [{name}] decision: {decision.get('action','?')} "
              f"{decision.get('symbol','')} x{decision.get('quantity','')}")
        if mc_stats.get("enabled") and mc_stats.get("nodes_in"):
            print(f"  [{name}] compression: {mc_stats['nodes_in']}→"
                  f"{mc_stats['nodes_out']} nodes "
                  f"(dup={mc_stats['dropped_duplicate']} "
                  f"stale={mc_stats['dropped_stale']} "
                  f"budget={mc_stats['dropped_budget']})")

        # 6. Execute (with action filtering)
        result = execute_trading_action(persona, self.market, decision)
        outcome = result["outcome_str"]

        # An action hallucination is observable in two different places
        # depending on the arm, and counting only one of them is what made this
        # metric read 0 in both arms:
        #   - middleware arm: the illegal request is rejected by
        #     run_action_filtering_step() and never reaches execution, so it is
        #     only visible in filter_stats.
        #   - baseline arm: there is no filter, so it surfaces at execution.
        # The union is the real count.
        illegal_request = bool(filter_stats.get("illegal_request"))
        hallucinated = bool(result["hallucination"]) or illegal_request
        halluc_kind = result["halluc_kind"] or (
            "illegal_request" if illegal_request else "")

        # Where the infeasible request ended up. Raw counts of `hallucinated`
        # are not comparable across arms on their own: the filtered arm catches
        # requests before execution, the baseline arm can only catch them at
        # execution, and the baseline's clamped orders *execute anyway* at a
        # corrected size. Splitting the disposition is what lets the report say
        # "44 caught, 0 executed" against "3 caught, 3 executed" instead of
        # implying the filtered arm hallucinates more.
        if not hallucinated:
            halluc_disposition = ""
        elif illegal_request or result["filtered"]:
            halluc_disposition = "caught_pre_execution"
        else:
            halluc_disposition = "executed_malformed"

        # Print with clear hallucination markers
        if hallucinated:
            print(f"  [{name}] *** HALLUCINATION CAUGHT *** "
                  f"[{halluc_kind}/{halluc_disposition}] {outcome}")
        else:
            print(f"  [{name}] {outcome}")

        # 7. Record fill in memory
        if result["fill"] and not result["filtered"]:
            record_trade_fill(persona, self.market, result["fill"])

        # 7b. Feed the failure back so the agent can see it happened. Both arms,
        # for the reason documented on record_order_feedback().
        feedback = ""
        req = filter_stats.get("requested") or {}
        if illegal_request and req.get("action") in ("buy", "sell"):
            feedback = (
                f"{name}'s order to {req['action']} {req.get('quantity')} "
                f"{req.get('symbol')} was REJECTED: "
                f"{filter_stats.get('error_reason') or 'not a permitted action'}. "
                f"Cash ${persona.scratch.cash_balance:,.0f}. "
                f"Do not repeat this order unless the position or cash changes.")
        elif result["filtered"]:
            feedback = (
                f"{name}'s order was REJECTED: {outcome} "
                f"Do not repeat this order unless the position or cash changes.")
        elif result["hallucination"] and result["filled_quantity"]:
            feedback = (
                f"{name} requested {result['requested_quantity']} shares but "
                f"only {result['filled_quantity']} were permitted; the order "
                f"was reduced. Cash ${persona.scratch.cash_balance:,.0f}.")
        if feedback:
            record_order_feedback(persona, self.market, feedback)

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

        # 8a2. Reasoning-vs-state grounding. Runs on every decision including
        # HOLD, which is the only detector here that does -- see
        # check_state_grounding() for why that matters.
        grounding_flag = check_state_grounding(reasoning, persona, self.market)
        if grounding_flag:
            print(f"  [{name}] *** STATE CONTRADICTION *** {grounding_flag}")

        # 8b. Persona consistency — did the LLM stay in character?
        # None means "not scoreable" (no reasoning was produced at all), which
        # is distinct from "scored badly" and must not be averaged in as either.
        drift_score = score_persona_consistency(reasoning, persona)
        if drift_score is not None and drift_score <= PERSONA_DRIFT_THRESHOLD:
            print(f"  [{name}] *** PERSONA DRIFT *** consistency score "
                  f"{drift_score:.2f} (reasoning may contradict agent profile)")

        # 9. Log
        pv = persona.portfolio_value(self.market.current_prices)
        log.append({
            "step":                step,
            "agent":               name,
            "decision":            decision,
            "requested":           filter_stats.get("requested") or {},
            "filter_status":       filter_stats.get("status", "success"),
            "outcome":             outcome,
            "hallucination":       hallucinated,
            "halluc_kind":         halluc_kind,
            "halluc_disposition":  halluc_disposition,
            "illegal_request":     illegal_request,
            "order_feedback":      feedback,
            "requested_quantity":  result["requested_quantity"],
            "filled_quantity":     result["filled_quantity"],
            "filtered":            result["filtered"],
            # Both the stale-context scan and the persona-consistency score read
            # `reasoning`. An empty one scores clean on both for free, so a run
            # that fails to parse often looks like a run that reasons well.
            # Recorded per step so the report can use it as the denominator
            # instead of counting unscoreable decisions as passes.
            "has_reasoning":       bool((reasoning or "").strip()),
            "stale_context":       stale_flag,
            "state_contradiction": grounding_flag,
            "persona_drift_score": drift_score,
            "compression":         mc_stats,
            "cash":                round(persona.scratch.cash_balance, 2),
            "portfolio_value":        pv,
            "portfolio_value_before": pv_before,
            "positions":           {s: dict(p)
                                    for s, p in persona.scratch.positions.items()},
            "market_prices":       dict(self.market.current_prices),
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

        from collections import Counter

        agents = list(self.personas.keys())
        per_agent = {
            name: {
                "total_decisions":       0,
                "active_decisions":      0,  # buy or sell (not hold/analyze)
                "trade_attempts":        0,  # buy/sell REQUESTED, pre-filter
                "action_hallucinations": 0,
                "halluc_kinds":          Counter(),
                # Disposition split -- see halluc_disposition in _step_agent().
                "caught_pre_execution":  0,
                "executed_malformed":    0,
                # (step, request) of every hallucination, for episode collapsing
                "halluc_requests":       [],
                "fallback_decisions":    0,
                "reasoned_decisions":    0,  # non-empty reasoning: the only
                                             # decisions stale/drift can score
                "stale_context_hits":    0,
                "state_contradictions":  0,
                "drift_scores":          [],  # persona_drift_score per step
                "mc_nodes_in":           0,
                "mc_nodes_out":          0,
                "mc_dropped_duplicate":  0,
                "mc_dropped_stale":      0,
                "mc_dropped_budget":     0,
                "mc_annotated_stale":    0,
                "mc_chars_in":           0,
                "mc_chars_out":          0,
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
                per_agent[name]["start_portfolio"] = entry.get(
                    "portfolio_value_before", entry["portfolio_value"])
                seen_start.add(name)

        for entry in log:
            name   = entry["agent"]
            action = entry["decision"].get("action", "hold")
            ag     = per_agent[name]

            ag["total_decisions"] += 1
            ag["action_counts"][action] = ag["action_counts"].get(action, 0) + 1

            if action in ("buy", "sell"):
                ag["active_decisions"] += 1

            # The denominator for a hallucination rate has to be what the model
            # *asked* to do, not what it was allowed to do. Using executed
            # buy/sell counts meant a run where every illegal trade was blocked
            # divided by ~0 and reported 0.0%.
            requested_action = (entry.get("requested") or {}).get("action") or action
            if requested_action in ("buy", "sell"):
                ag["trade_attempts"] += 1

            if entry.get("filter_status") == "fallback":
                ag["fallback_decisions"] += 1

            if entry.get("has_reasoning"):
                ag["reasoned_decisions"] += 1

            if entry.get("hallucination"):
                ag["action_hallucinations"] += 1
                ag["halluc_kinds"][entry.get("halluc_kind") or "unspecified"] += 1
                disp = entry.get("halluc_disposition")
                if disp == "caught_pre_execution":
                    ag["caught_pre_execution"] += 1
                elif disp == "executed_malformed":
                    ag["executed_malformed"] += 1
                req = entry.get("requested") or {}
                ag["halluc_requests"].append((
                    entry.get("step"),
                    (req.get("action"), req.get("symbol"), req.get("quantity")),
                ))

            if entry.get("stale_context"):
                ag["stale_context_hits"] += 1

            if entry.get("state_contradiction"):
                ag["state_contradictions"] += 1

            drift = entry.get("persona_drift_score")
            if drift is not None:
                ag["drift_scores"].append(drift)

            mc = entry.get("compression") or {}
            ag["mc_nodes_in"]          += mc.get("nodes_in", 0)
            ag["mc_nodes_out"]         += mc.get("nodes_out", 0)
            ag["mc_dropped_duplicate"] += mc.get("dropped_duplicate", 0)
            ag["mc_dropped_stale"]     += mc.get("dropped_stale", 0)
            ag["mc_dropped_budget"]    += mc.get("dropped_budget", 0)
            ag["mc_annotated_stale"]   += mc.get("annotated_stale", 0)
            ag["mc_chars_in"]          += mc.get("chars_in", 0)
            ag["mc_chars_out"]         += mc.get("chars_out", 0)

            # Last seen entry = end state
            ag["end_portfolio"] = entry["portfolio_value"]

        # Compute rates
        summary = {}
        total_hallucinations = 0
        for name, ag in per_agent.items():
            total  = ag["total_decisions"]  or 1

            # None, not 0.0, when the agent never attempted a trade: there is
            # nothing to have hallucinated about, and printing "0.0%" for a 0/0
            # made a vacuous result look like a measured one.
            attempts = ag["trade_attempts"]
            action_h_rate = (round(ag["action_hallucinations"] / attempts * 100, 1)
                             if attempts else None)
            # Stale context is only observable in reasoning text, so an empty
            # reasoning is not a clean decision -- it is an unscored one.
            # Dividing by all decisions let a high fallback rate masquerade as a
            # low stale-context rate, which is the single most likely
            # explanation for a large baseline-vs-middleware gap on this metric.
            reasoned = ag["reasoned_decisions"]
            stale_rate = (round(ag["stale_context_hits"] / reasoned * 100, 1)
                          if reasoned else None)
            contradiction_rate = (
                round(ag["state_contradictions"] / reasoned * 100, 1)
                if reasoned else None)
            fallback_rate  = round(ag["fallback_decisions"]    / total  * 100, 1)
            pnl            = round(ag["end_portfolio"] - ag["start_portfolio"], 2)

            # Distinct episodes. A blocked order leaves cash and holdings
            # untouched, so the next step rebuilds nearly the same prompt and
            # the model reissues the same request; 27 of run 03's 44 recorded
            # hallucinations were a single agent repeating `buy NVDA x10`.
            # Counting those as 27 independent errors lets one livelocked agent
            # set the headline number. Reported ALONGSIDE the raw count, never
            # instead of it -- collapsing is a way to read the data, not a
            # correction to it.
            episodes = _count_episodes(ag["halluc_requests"])
            episode_rate = (round(episodes / attempts * 100, 1)
                            if attempts else None)

            # drift_scores only ever collects non-None values, so this averages
            # over decisions that were actually scoreable.
            scores = ag["drift_scores"]
            avg_drift  = round(sum(scores) / len(scores), 2) if scores else None
            drift_events = sum(1 for s in scores if s <= PERSONA_DRIFT_THRESHOLD)

            summary[name] = {
                "total_decisions":              ag["total_decisions"],
                "active_decisions":             ag["active_decisions"],
                "trade_attempts":               ag["trade_attempts"],
                "action_hallucinations":        ag["action_hallucinations"],
                "action_hallucination_rate_pct": action_h_rate,
                "hallucination_kinds":          dict(ag["halluc_kinds"]),
                # Disposition: caught before execution vs actually executed in
                # a malformed (clamped) form. The filtered arm can only produce
                # the former, the baseline arm mostly the latter, so the raw
                # totals are not like-for-like without this split.
                "caught_pre_execution":         ag["caught_pre_execution"],
                "executed_malformed":           ag["executed_malformed"],
                "hallucination_episodes":       episodes,
                "hallucination_episode_rate_pct": episode_rate,
                "fallback_decisions":           ag["fallback_decisions"],
                "fallback_rate_pct":            fallback_rate,
                "reasoned_decisions":           ag["reasoned_decisions"],
                "stale_context_hits":           ag["stale_context_hits"],
                "stale_context_rate_pct":       stale_rate,
                # Reasoning that contradicts the agent's own book. The only
                # metric here that can score a HOLD.
                "state_contradictions":         ag["state_contradictions"],
                "state_contradiction_rate_pct": contradiction_rate,
                "avg_persona_consistency":      avg_drift,
                "persona_drift_events":         drift_events,
                "scored_decisions":             len(scores),
                "memory_compression": {
                    "nodes_in":          ag["mc_nodes_in"],
                    "nodes_out":         ag["mc_nodes_out"],
                    "compression_ratio": (round(ag["mc_nodes_out"] / ag["mc_nodes_in"], 3)
                                          if ag["mc_nodes_in"] else 1.0),
                    "dropped_duplicate": ag["mc_dropped_duplicate"],
                    "dropped_stale":     ag["mc_dropped_stale"],
                    "dropped_budget":    ag["mc_dropped_budget"],
                    "annotated_stale":   ag["mc_annotated_stale"],
                    "chars_in":          ag["mc_chars_in"],
                    "chars_out":         ag["mc_chars_out"],
                    "avg_context_chars": (round(ag["mc_chars_out"] / total)
                                          if total else 0),
                },
                "action_distribution":          ag["action_counts"],
                "start_portfolio_usd":          ag["start_portfolio"],
                "end_portfolio_usd":            ag["end_portfolio"],
                "pnl_usd":                      pnl,
            }
            total_hallucinations += ag["action_hallucinations"]

        total_decisions = sum(a["total_decisions"] for a in per_agent.values())
        total_attempts  = sum(a["trade_attempts"]  for a in per_agent.values())
        total_caught    = sum(a["caught_pre_execution"] for a in per_agent.values())
        total_executed  = sum(a["executed_malformed"]   for a in per_agent.values())
        total_episodes  = sum(s["hallucination_episodes"] for s in summary.values())
        total_contra    = sum(a["state_contradictions"] for a in per_agent.values())
        total_reasoned  = sum(a["reasoned_decisions"]   for a in per_agent.values())
        return {
            "middleware_enabled":          self.use_middleware,
            "simulation_steps":            n_steps,
            "total_log_entries":           len(log),
            "total_decisions":             total_decisions,
            "total_trade_attempts":        total_attempts,
            "total_hallucinations":        total_hallucinations,
            # Same denominator the per-agent rate uses -- these two were
            # previously divided by different things (total vs active), so the
            # headline and the per-agent numbers were not comparable.
            "overall_hallucination_rate_pct":
                (round(total_hallucinations / total_attempts * 100, 1)
                 if total_attempts else None),
            "total_caught_pre_execution":  total_caught,
            "total_executed_malformed":    total_executed,
            "total_hallucination_episodes": total_episodes,
            "overall_episode_rate_pct":
                (round(total_episodes / total_attempts * 100, 1)
                 if total_attempts else None),
            "total_state_contradictions":  total_contra,
            "overall_state_contradiction_rate_pct":
                (round(total_contra / total_reasoned * 100, 1)
                 if total_reasoned else None),
            "per_agent": summary,
        }

    def _print_report(self, report: dict):
        sep = "=" * 60
        print(f"\n{sep}")
        print("HALLUCINATION REPORT")
        print(sep)
        print(f"Middleware      : "
              f"{'ENABLED' if report.get('middleware_enabled') else 'DISABLED (baseline)'}")
        print(f"Steps simulated : {report['simulation_steps']}")
        print(f"Total decisions : {report['total_log_entries']}")
        overall = report["overall_hallucination_rate_pct"]
        if overall is None:
            print(f"Total hallucinations : {report['total_hallucinations']}  "
                  f"(rate N/A -- no agent ever requested a trade)")
        else:
            print(f"Total hallucinations : {report['total_hallucinations']}  "
                  f"({overall}% of {report['total_trade_attempts']} trade attempts)")
        # The number that is actually comparable across arms. Raw counts are
        # not: the filtered arm intercepts requests before execution, so it
        # reports many caught / none executed, while the baseline arm silently
        # clamps and executes them. "Caught" without "executed" reads as the
        # filtered arm being worse when it is doing its job.
        print(f"  caught before execution : {report.get('total_caught_pre_execution', 0)}")
        print(f"  EXECUTED malformed      : {report.get('total_executed_malformed', 0)}"
              f"   <- reached the market")
        ep_rate = report.get("overall_episode_rate_pct")
        print(f"  distinct episodes       : "
              f"{report.get('total_hallucination_episodes', 0)}"
              + (f"  ({ep_rate}% of attempts)" if ep_rate is not None else ""))
        c_rate = report.get("overall_state_contradiction_rate_pct")
        print(f"State contradictions : {report.get('total_state_contradictions', 0)}"
              + (f"  ({c_rate}% of reasoned decisions)" if c_rate is not None
                 else "  (rate N/A)")
              + "   <- includes HOLDs")
        print()
        for name, ag in report["per_agent"].items():
            print(f"  {name}")
            print(f"    Decisions      : {ag['total_decisions']}  "
                  f"(active={ag['active_decisions']}, "
                  f"requested={ag['trade_attempts']})")
            rate = ag["action_hallucination_rate_pct"]
            rate_str = ("N/A (no trades requested)" if rate is None
                        else f"{rate}% of {ag['trade_attempts']} attempts")
            print(f"    Action hallucinations : {ag['action_hallucinations']}  "
                  f"({rate_str})")
            if ag["hallucination_kinds"]:
                kinds = "  ".join(f"{k}={v}"
                                  for k, v in sorted(ag["hallucination_kinds"].items()))
                print(f"      by kind: {kinds}")
            ep = ag.get("hallucination_episodes", 0)
            ep_r = ag.get("hallucination_episode_rate_pct")
            print(f"      caught={ag.get('caught_pre_execution', 0)}  "
                  f"executed={ag.get('executed_malformed', 0)}  "
                  f"episodes={ep}"
                  + (f" ({ep_r}%)" if ep_r is not None else ""))
            # Without this you cannot tell "the middleware improved the metrics"
            # from "the middleware produced empty reasoning strings", since a
            # fallback decision has reasoning="" and so scores clean on both
            # the stale-context and persona-consistency checks for free.
            print(f"    Parser fallbacks      : {ag['fallback_decisions']}  "
                  f"({ag['fallback_rate_pct']}% of decisions)")
            print(f"    Decisions w/ reasoning: {ag['reasoned_decisions']}  "
                  f"(the only ones stale/drift can score)")
            stale_pct = ag["stale_context_rate_pct"]
            stale_str = ("N/A (no reasoning produced)" if stale_pct is None
                         else f"{stale_pct}% of {ag['reasoned_decisions']} reasoned")
            print(f"    Stale context hits    : {ag['stale_context_hits']}  "
                  f"({stale_str})")
            c_pct = ag.get("state_contradiction_rate_pct")
            c_str = ("N/A (no reasoning produced)" if c_pct is None
                     else f"{c_pct}% of {ag['reasoned_decisions']} reasoned")
            print(f"    State contradictions  : {ag.get('state_contradictions', 0)}  "
                  f"({c_str})")
            avg_c = ag.get("avg_persona_consistency")
            drift_e = ag.get("persona_drift_events", 0)
            scored = ag.get("scored_decisions", 0)
            avg_str = f"{avg_c:.2f}" if avg_c is not None else "n/a"
            print(f"    Persona consistency   : avg={avg_str}  "
                  f"drift events={drift_e}  (scored {scored})")
            mc = ag.get("memory_compression", {})
            if mc.get("nodes_in"):
                if report.get("middleware_enabled"):
                    print(f"    Memory compression    : {mc['nodes_in']}→{mc['nodes_out']} nodes "
                          f"(ratio={mc['compression_ratio']})")
                    print(f"      dropped: dup={mc['dropped_duplicate']} "
                          f"stale={mc['dropped_stale']} budget={mc['dropped_budget']}  "
                          f"annotated={mc['annotated_stale']}")
                else:
                    print(f"    Memory context        : {mc['nodes_in']} nodes "
                          f"(uncompressed)")
                print(f"      avg prompt context    : "
                      f"{mc['avg_context_chars']:,} chars/decision")
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

    # -----------------------------------------------------------------------

    def _write_run_status(self, step: int, running: bool):
        """
        Read-modify-write meta.json's step/sim_running fields. This is the
        signal translator/sim_data.py + the map view's live poller use to
        decide whether to keep polling for new steps or treat the run as a
        finished replay -- written every step, so keep it cheap.
        """
        meta_path = Path(self.sim_folder) / "reverie" / "meta.json"
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        meta["step"] = step
        meta["sim_running"] = running
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)


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
    parser.add_argument("--fresh", action="store_true",
                        help="Delete an existing --sim folder and re-fork it. "
                             "Without this, reusing a --sim name is refused: "
                             "resuming carries persona positions into a "
                             "step-0 market and makes arms incomparable.")
    parser.add_argument("--no-middleware", action="store_true",
                        help="Baseline arm: disable memory compression, ID-RAG "
                             "anchoring and the action-filtering legal-action "
                             "menu. Scoring is unchanged, so the resulting "
                             "hallucination_report.json is directly comparable "
                             "to a normal run.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Market seed. Paired arms MUST share a seed, and "
                             "several seeds per arm are needed before a rate "
                             "difference means anything.")
    args = parser.parse_args()

    sim = TradingReverie(args.sim, args.fork,
                         use_middleware=not args.no_middleware,
                         fresh=args.fresh, seed=args.seed)
    sim.run(args.steps)


if __name__ == "__main__":
    main()
