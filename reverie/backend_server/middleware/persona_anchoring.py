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

    2. score_persona_consistency(reasoning, persona) -> float | None
       Heuristic 0.0–1.0 score for how in-character the LLM's returned
       reasoning is. Lower = more drift detected. None when there is no
       reasoning to score, which is distinct from scoring badly.
       Logged per step alongside the hallucination and stale-context flags.
"""

from typing import Optional


# A decision is counted as persona drift when strictly more than half of the
# checks that actually ran were violated.
#
# This used to be a bare `drift_score < 0.67` at the call site, which was
# unreachable for the case that dominated both arms: a fallback decision has
# reasoning="", which violated only the watchlist check and scored exactly
# round(1 - 1/3, 2) == 0.67 -- and `0.67 < 0.67` is False. Every empty-reasoning
# fallback was therefore recorded as perfectly in character, which is why drift
# events read 0 everywhere. Empty reasoning now scores None (see below) and is
# excluded from the average rather than silently counted as consistent.
PERSONA_DRIFT_THRESHOLD = 0.5


def _normalise_risk_tolerance(raw: Optional[str]) -> str:
    """
    Map free-text risk_tolerance onto {conservative, moderate, aggressive}.

    The persona files contain values like "moderate-high" that matched none of
    the literal branches this module used to test, so those agents silently
    skipped the language check while still being divided by a fixed denominator
    of 3 -- structurally scoring higher than an agent whose tolerance happened
    to be spelled "aggressive".
    """
    tol = (raw or "moderate").strip().lower().replace("_", "-")
    if "conserv" in tol or tol in ("low", "very-low"):
        return "conservative"
    if "aggress" in tol or tol in ("high", "very-high"):
        return "aggressive"
    return "moderate"


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
    tol = _normalise_risk_tolerance(risk_tolerance)

    if tol == "conservative":
        return [
            "Prefer HOLD or small positions when uncertain.",
            "Do not chase momentum — wait for clear confirmation.",
            "Take profits early rather than holding for maximum gain.",
            f"If a single trade would cost more than {risk_limit_pct:.0f}% of "
            f"portfolio value, choose HOLD instead.",
        ]
    elif tol == "aggressive":
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


def score_persona_consistency(reasoning: str, persona) -> Optional[float]:
    """
    Heuristic consistency score in [0.0, 1.0], or None when not scoreable.

    Checks up to three independent dimensions; each violation subtracts from 1.0:
      - Language-risk mismatch  (aggressive words for conservative agent, etc.)
      - Watchlist blindness      (agent ignores stocks it's supposed to track)
      - Circular / stale signal  (agent references prior context as justification)

    Returns None when there is nothing to score -- either the model produced no
    reasoning at all (every parser fallback has reasoning="") or the persona
    defines none of the fields the checks read. Scoring an empty string is not a
    measurement of character: it violates the watchlist check by construction and
    passes the other two for free, which is exactly how a decision the model
    never actually made ended up recorded as in-character.

    The denominator counts only the checks that ran. It used to be hard-coded to
    3 even though the language check is skipped for moderate agents, so those
    agents could never score below 0.33 while an aggressive agent could reach 0.0.
    """
    s = persona.scratch
    reasoning = (reasoning or "").strip()
    if not reasoning:
        return None

    risk_tolerance  = _normalise_risk_tolerance(getattr(s, "risk_tolerance", None))
    watchlist       = [sym.upper() for sym in getattr(s, "watchlist", [])]
    reasoning_lower = reasoning.lower()
    reasoning_upper = reasoning.upper()

    violations = 0
    checks     = 0

    # 1. Language-risk mismatch — only meaningful when the profile commits to a
    #    direction. Moderate has no characteristic vocabulary to violate, so the
    #    check does not run and must not be counted in the denominator.
    if risk_tolerance == "conservative":
        checks += 1
        if any(w in reasoning_lower for w in _AGGRESSIVE_WORDS):
            violations += 1
    elif risk_tolerance == "aggressive":
        checks += 1
        if any(w in reasoning_lower for w in _FEARFUL_WORDS):
            violations += 1

    # 2. Watchlist blindness — acting on a symbol not in the watchlist
    #    when the agent has a non-empty watchlist defined.
    if watchlist:
        checks += 1
        mentioned_watchlist = any(sym in reasoning_upper for sym in watchlist)
        if not mentioned_watchlist:
            violations += 1

    # 3. Stale / circular signal — always runs, needs no persona fields.
    checks += 1
    if any(w in reasoning_lower for w in _STALE_SIGNAL_WORDS):
        violations += 1

    if checks == 0:
        return None
    return round(1.0 - violations / checks, 2)


# ---------------------------------------------------------------------------
# Convenience: inject anchor into an existing memory-context string
# ---------------------------------------------------------------------------

def anchor_memory_context(persona, memory_context: str) -> str:
    """
    Prepend the persona anchor to an existing memory context string.

    NOTE: make_trading_decision() calls id_rag.id_rag_anchor() instead, which
    retrieves only the identity facts relevant to the current context rather
    than prepending this full static block. This function is kept for callers
    that want the unconditional anchor (input_stabilizer.py) -- it is no longer
    on the main decision path, whatever this docstring used to claim.
    """
    anchor = create_persona_anchor(persona)
    return anchor + memory_context
