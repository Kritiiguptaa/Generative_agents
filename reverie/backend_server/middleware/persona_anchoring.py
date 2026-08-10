"""
File: middleware/persona_anchoring.py
Description: Persona anchoring middleware for the trading simulation.

  Without anchoring, LLMs drift from an agent's defined character under
  market pressure — a conservative agent starts making aggressive bets after
  reading exciting news, or an aggressive trader freezes and holds everything
  out of fear.

  This module does two things:
    1. create_persona_anchor(persona) -> str
       Builds a structured identity block that is prepended to every LLM
       prompt so the model commits to who this agent is *before* it sees
       any market data or memories.

    2. score_persona_consistency(reasoning, persona) -> float
       Heuristic 0.0–1.0 score for how in-character the LLM's returned
       reasoning is. Lower = more drift detected.
       Logged per step alongside the hallucination and stale-context flags.
"""

from typing import Optional


# ---------------------------------------------------------------------------
# Identity block builder
# ---------------------------------------------------------------------------

def create_persona_anchor(persona) -> str:
    """
    Build a structured identity block from the persona's trading fields.
    Returns a multi-line string meant to be prepended to the LLM prompt.
    """
    s = persona.scratch

    name            = getattr(s, "name", "Unknown")
    trader_type     = getattr(s, "trader_type",     None) or "generalist"
    specialization  = getattr(s, "specialization",  None) or ""
    risk_tolerance  = getattr(s, "risk_tolerance",  "moderate")
    risk_limit      = float(getattr(s, "risk_limit_per_trade", 0.05)) * 100
    strategy        = getattr(s, "trading_strategy", None) or ""
    innate          = getattr(s, "innate",           None) or ""
    watchlist       = getattr(s, "watchlist",        [])

    lines = [
        f"=== IDENTITY: {name} ===",
        f"You are {name}, a {trader_type} trader.",
    ]

    if specialization:
        lines.append(f"Specialization : {specialization}.")

    if innate:
        lines.append(f"Personality    : {innate}.")

    if strategy:
        lines.append(f"Strategy       : {strategy}.")

    if watchlist:
        lines.append(f"Watchlist      : {', '.join(watchlist)}.")

    # Derive explicit behavioural rules from risk profile so the LLM has
    # unambiguous instructions rather than vague adjectives.
    lines.append(f"Risk tolerance : {risk_tolerance}.")
    lines.append(
        f"Hard rule      : Never risk more than {risk_limit:.1f}% of your "
        f"total portfolio on a single trade."
    )

    risk_rules = _risk_rules(risk_tolerance, risk_limit)
    for rule in risk_rules:
        lines.append(f"Guideline      : {rule}")

    lines += [
        "Every decision you make MUST be consistent with this identity.",
        "=== END IDENTITY ===",
        "",
    ]

    return "\n".join(lines)


def _risk_rules(risk_tolerance: str, risk_limit_pct: float) -> list:
    """Return concrete behavioural guidelines derived from the risk profile."""
    tol = risk_tolerance.lower()

    if tol in ("low", "conservative"):
        return [
            "Prefer HOLD or small positions when uncertain.",
            "Do not chase momentum — wait for clear confirmation.",
            "Take profits early rather than holding for maximum gain.",
            f"If a single trade would cost more than {risk_limit_pct:.0f}% of "
            f"portfolio value, choose HOLD instead.",
        ]
    elif tol in ("high", "aggressive"):
        return [
            "Act decisively on strong signals — hesitation costs opportunity.",
            "Size positions near the maximum allowed limit when conviction is high.",
            "Accept short-term volatility in pursuit of larger gains.",
        ]
    else:  # moderate
        return [
            "Balance conviction against position size — neither too large nor too small.",
            "Hold when signals are mixed; act when at least two signals align.",
            f"Keep individual trades below {risk_limit_pct:.0f}% of portfolio value.",
        ]


# ---------------------------------------------------------------------------
# Consistency scorer
# ---------------------------------------------------------------------------

_AGGRESSIVE_WORDS = frozenset({
    "all-in", "all in", "massive", "huge bet", "yolo", "double down",
    "aggressive", "maximum position", "go big", "bet everything",
    "full position", "maximum leverage",
})

_FEARFUL_WORDS = frozenset({
    "too risky", "too scared", "avoid everything", "play it safe",
    "too uncertain", "sit this out", "stay on sidelines", "do nothing",
    "terrified", "panic",
})

_STALE_SIGNAL_WORDS = frozenset({
    "as mentioned earlier", "as stated before", "previously noted",
    "earlier report", "as we discussed",
})


def score_persona_consistency(reasoning: str, persona) -> float:
    """
    Heuristic consistency score in [0.0, 1.0].

    Checks three independent dimensions; each violation subtracts from 1.0:
      - Language-risk mismatch  (aggressive words for conservative agent, etc.)
      - Watchlist blindness      (agent ignores stocks it's supposed to track)
      - Circular / stale signal  (agent references prior context as justification)

    Returns 1.0 (fully consistent) down to 0.0 (all three violated).
    """
    s = persona.scratch
    risk_tolerance  = (getattr(s, "risk_tolerance", "moderate") or "moderate").lower()
    watchlist       = [sym.upper() for sym in getattr(s, "watchlist", [])]
    reasoning_lower = reasoning.lower()
    reasoning_upper = reasoning.upper()

    violations = 0
    checks     = 3  # keep denominator fixed

    # 1. Language-risk mismatch
    if risk_tolerance in ("low", "conservative"):
        if any(w in reasoning_lower for w in _AGGRESSIVE_WORDS):
            violations += 1
    elif risk_tolerance in ("high", "aggressive"):
        if any(w in reasoning_lower for w in _FEARFUL_WORDS):
            violations += 1
    # moderate: no language violation check (too noisy)

    # 2. Watchlist blindness — acting on a symbol not in the watchlist
    #    when the agent has a non-empty watchlist defined.
    if watchlist:
        mentioned_watchlist = any(sym in reasoning_upper for sym in watchlist)
        if not mentioned_watchlist:
            violations += 1

    # 3. Stale / circular signal
    if any(w in reasoning_lower for w in _STALE_SIGNAL_WORDS):
        violations += 1

    return round(1.0 - violations / checks, 2)


# ---------------------------------------------------------------------------
# Convenience: inject anchor into an existing memory-context string
# ---------------------------------------------------------------------------

def anchor_memory_context(persona, memory_context: str) -> str:
    """
    Prepend the persona anchor to an existing memory context string.
    This is the single call site used by make_trading_decision().
    """
    anchor = create_persona_anchor(persona)
    return anchor + memory_context
