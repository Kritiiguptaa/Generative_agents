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

import re
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
# Drift fires when the score is at or below this, i.e. when at least half of
# the checks that ran failed.
#
# The comparison used to be strict (`score < 0.5`), which combined badly with
# the small number of checks to make drift almost unreachable. An agent running
# two checks scores 1.0 / 0.5 / 0.0, so a single violation landed on exactly
# 0.50 and passed; two simultaneous violations were required to register
# anything. Run 03 recorded 0 drift events across all six agent-runs with a
# minimum observed score of 0.71 -- not a finding, just a metric that could not
# fire. Use `score <= PERSONA_DRIFT_THRESHOLD` at call sites.
PERSONA_DRIFT_THRESHOLD = 0.5

# An aggressive agent that HOLDs on more than this fraction of its decisions is
# out of character for the run, whatever its reasoning says. The per-decision
# score cannot see this: a HOLD that names no symbol and uses no fearful phrase
# passes every check, which is how Marcus Webb ("buy on breakouts and dips, go
# big") held 70% of a smoke run while scoring 1.00.
# shortcut: 0.5 is a judgement call, not calibrated; replace with the baseline
# arm's measured hold rate once the run 07 sweep gives one.
AGGRESSIVE_MAX_HOLD_RATE = 0.5


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

# "maximum position" / "maximum leverage" are aggressive intent when an agent
# says it will take one ("go to maximum position") and plain risk-management
# vocabulary when it cites its own limit ("within the maximum position limit").
# Matched as bare substrings they flagged the second use, and a conservative
# agent cites its limit all the time: the first nine-agent smoke run (smoke9_mw) scored
# a conservative HOLD 0.50 for "AAPL and GOOGL are already within the maximum
# position limit". Phrases that refer to the limit are blanked out before the
# aggressive-word match, so only an unqualified use still counts.
# shortcut: "maximum position size" is still flagged -- it is as likely to be
# intent ("take maximum position size") as a limit, and a wrong guess either
# way is worse than leaving the phrase out of this list.
_LIMIT_REFERENCE_RE = re.compile(
    r"\bmaximum (?:position|leverage)\s+(?:limit|cap|threshold|allowed|allowance)\b"
    r"|\b(?:within|under|below|near|exceeds?|exceeding|over)\s+"
    r"(?:the |my |our |its )?maximum (?:position|leverage)\b")


def _has_aggressive_language(reasoning_lower: str) -> bool:
    """True when the reasoning uses an aggressive phrase, ignoring references
    to the agent's own position/leverage limit."""
    text = _LIMIT_REFERENCE_RE.sub(" ", reasoning_lower)
    return any(w in text for w in _AGGRESSIVE_WORDS)


_FEARFUL_WORDS = frozenset({
    "too risky", "too scared", "avoid everything", "play it safe",
    "too uncertain", "sit this out", "stay on sidelines", "do nothing",
    "terrified", "panic",
})

_STALE_SIGNAL_WORDS = frozenset({
    "as mentioned earlier", "as stated before", "previously noted",
    "earlier report", "as we discussed",
})

# Company names that satisfy the watchlist check for a ticker.
#
# The check matched tickers as literal substrings, so an agent writing "NVIDIA
# smashes Q4 estimates ... an opportunity to capitalize" was scored as ignoring
# its own watchlist -- "NVIDIA" does not contain "NVDA". Measured effect on
# run 03: Marcus Webb named a watchlist symbol in 93% of baseline reasonings but
# only 46% under middleware, and his consistency score fell 0.98 -> 0.82. That
# gap was read as the anchoring layer degrading identity. It was not: the
# compressed arm simply produced shorter reasoning that used company names or
# no name at all, and the check counted brevity as drift.
_SYMBOL_ALIASES = {
    "NVDA":  ("NVIDIA",),
    "AAPL":  ("APPLE",),
    "TSLA":  ("TESLA",),
    "AMD":   ("ADVANCED MICRO",),
    "GOOGL": ("GOOGLE", "ALPHABET"),
    "MSFT":  ("MICROSOFT",),
    "AMZN":  ("AMAZON",),
    "META":  ("FACEBOOK",),
}


# Every symbol the scorer can recognise in free text. Used to tell "named an
# off-watchlist stock" (a real character violation) apart from "named nothing"
# (no evidence either way).
_ALL_KNOWN_SYMBOLS = frozenset(_SYMBOL_ALIASES)


def _mentions_symbol(reasoning_upper: str, symbol: str) -> bool:
    """True if the text names `symbol` by ticker or by company name."""
    if symbol in reasoning_upper:
        return True
    return any(alias in reasoning_upper
               for alias in _SYMBOL_ALIASES.get(symbol, ()))


def score_persona_consistency(reasoning: str, persona,
                              requested: Optional[dict] = None) -> Optional[float]:
    """
    Heuristic consistency score in [0.0, 1.0], or None when not scoreable.

    Checks up to two independent dimensions; each violation subtracts from 1.0:
      - Language-risk mismatch  (aggressive words for conservative agent, etc.)
      - Watchlist               (BUYING an off-watchlist symbol; when the order
                                 is unknown, naming only off-watchlist symbols
                                 in the reasoning. A HOLD or SELL that merely
                                 mentions an off-watchlist stock is not scored
                                 on this dimension)

    `requested` is the order the model asked for, BEFORE the filter touched it.
    What the agent tried to do is better evidence of character than what it
    wrote, and it is scoreable even when the reasoning is empty.

    Returns None when there is nothing to score -- no reasoning and no trade
    request, or a persona defining none of the fields the checks read. Scoring
    an empty string is not a measurement of character: it violates the watchlist
    check by construction and passes the others for free, which is exactly how
    a decision the model never actually made ended up recorded as in-character.

    The denominator counts only the checks that ran. It used to be hard-coded to
    3 even though the language check is skipped for moderate agents, so those
    agents could never score below 0.33 while an aggressive agent could reach 0.0.
    """
    s = persona.scratch
    reasoning = (reasoning or "").strip()
    requested = requested or {}

    risk_tolerance  = _normalise_risk_tolerance(getattr(s, "risk_tolerance", None))
    watchlist       = [sym.upper() for sym in getattr(s, "watchlist", [])]
    reasoning_lower = reasoning.lower()
    reasoning_upper = reasoning.upper()

    violations = 0
    checks     = 0

    # 0. Buying off the watchlist. Only BUY: selling an off-watchlist holding is
    #    exiting a position, which is in character for any profile. Not a
    #    hallucination either -- the symbol exists and the order is legal -- so
    #    this does not double-count anything the filter already reports.
    #    When it runs it replaces the reasoning-symbol check below, so one act
    #    is not scored twice for naming the stock it bought.
    bought = str(requested.get("symbol") or "").upper()
    buy_checked = (str(requested.get("action") or "").lower() == "buy"
                   and bool(bought) and bool(watchlist))
    if buy_checked:
        checks += 1
        if bought not in watchlist:
            violations += 1

    # 1. Language-risk mismatch — only meaningful when the profile commits to a
    #    direction. Moderate has no characteristic vocabulary to violate, so the
    #    check does not run and must not be counted in the denominator.
    if not reasoning:
        pass
    elif risk_tolerance == "conservative":
        checks += 1
        if _has_aggressive_language(reasoning_lower):
            violations += 1
    elif risk_tolerance == "aggressive":
        checks += 1
        if any(w in reasoning_lower for w in _FEARFUL_WORDS):
            violations += 1

    # 2. Watchlist blindness — acting on a symbol not in the watchlist
    #    when the agent has a non-empty watchlist defined.
    # The violation is naming an OFF-watchlist symbol, not failing to name one.
    #
    # This check used to fire whenever the reasoning contained no watchlist
    # ticker, which penalises brevity rather than character: "current positions
    # are still within the risk limit, no new trades" is perfectly in-character
    # for a trader who simply did not name an instrument. Because the compressed
    # arm produces shorter reasoning, that turned a length difference into an
    # identity difference -- Marcus Webb named a symbol in 93% of baseline
    # reasonings vs 46% under middleware, and the resulting 0.98 -> 0.82 score
    # gap looked like the anchoring layer degrading the persona when nothing
    # about his character had changed.
    #
    # Reasoning that names no symbol at all carries no evidence either way, so
    # the check does not run and is not counted in the denominator.
    #
    # NOT run when the order is an explicit HOLD or SELL. Naming an off-watchlist
    # stock in the reasoning of a HOLD is market commentary, not acting on it: in
    # the first 100-step nine-agent run a conservative agent that never traded
    # (watchlist AAPL, GOOGL) was scored 0.50 five times for writing that Tesla
    # recalls and an NVDA rally were reasons to wait. That is what a cautious
    # agent says. The act that can contradict a watchlist is a BUY, which the
    # check at the top of this function already scores from the order itself.
    # Without an explicit order (requested has no action) the reasoning is all
    # there is, so the check still runs for those callers.
    holds_or_exits = str(requested.get("action") or "").lower() in ("hold", "sell")
    if watchlist and reasoning and not buy_checked and not holds_or_exits:
        watch_up = {s.upper() for s in watchlist}
        mentioned = {sym for sym in _ALL_KNOWN_SYMBOLS
                     if _mentions_symbol(reasoning_upper, sym)}
        if mentioned:
            checks += 1
            if not (mentioned & watch_up):
                violations += 1

    # NOTE: a third check used to run here, flagging "as mentioned earlier" /
    # "as we discussed" as a stale/circular signal. It was removed because
    # _check_stale_reasoning() in trading_reverie.py already scans the same
    # reasoning string for stale references and reports them as
    # stale_context_hits. Counting one utterance in two independent metrics
    # makes them covary for a reason unrelated to what either claims to
    # measure, and stale_context is the better of the two: it checks the
    # reference against the scripted-news timeline rather than matching a
    # fixed phrase list. _STALE_SIGNAL_WORDS is kept for callers that import
    # it, but no longer contributes to this score.

    if checks == 0:
        return None
    return round(1.0 - violations / checks, 2)


def hold_rate_drift(hold_rate: float, persona) -> bool:
    """
    True when a run's HOLD rate is out of character for the agent's profile.

    Run-level, not per-decision: one HOLD is in character for anyone, holding
    nearly all run is not for an aggressive trader. Only aggressive commits to
    a direction here -- a moderate "high conviction only" agent (Alex Chen)
    holding all run is following his own strategy.
    """
    tol = _normalise_risk_tolerance(getattr(persona.scratch, "risk_tolerance", None))
    return tol == "aggressive" and hold_rate > AGGRESSIVE_MAX_HOLD_RATE


# ---------------------------------------------------------------------------
# Convenience: inject anchor into an existing memory-context string
# ---------------------------------------------------------------------------

def anchor_memory_context(persona, memory_context: str) -> str:
    """
    Prepend the persona anchor to an existing memory context string.

    LEGACY. NOT ON ANY CODE PATH, AND HAS NO CALLER AT ALL.

    The pipeline calls id_rag.id_rag_anchor() instead, which retrieves only the
    identity facts relevant to the current context rather than prepending this
    full static block. This function's only caller was input_stabilizer.py,
    which has been deleted: it was the original four-layer orchestrator, it
    imported four classes that never existed, and it could not be imported.

    Kept rather than deleted because the paper's capstone report describes a
    "layer 3: persona anchoring", and this is that layer. It is worth being
    able to point at what the description refers to. What is measured is
    score_persona_consistency() further up, a post-hoc metric, so layer 3 is a
    measurement and not an intervention. See middleware/README.md.
    """
    anchor = create_persona_anchor(persona)
    return anchor + memory_context
