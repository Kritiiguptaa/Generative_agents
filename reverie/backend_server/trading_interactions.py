"""
File: trading_interactions.py
Description: Lightweight peer interactions for trading agents.
"""

import json
from itertools import combinations

from persona.prompt_template.gpt_structure import get_embedding, ollama_request


def maybe_interaction(personas: dict, market, step: int, interval_steps: int = 12):
    """
    Create a peer interaction on a fixed cadence.
    Returns an interaction dict or None.
    """
    if step <= 0 or step % interval_steps != 0:
        return None

    names = sorted(personas.keys())
    if len(names) < 2:
        return None

    pairs = list(combinations(names, 2))
    idx = (step // interval_steps) % len(pairs)
    a_name, b_name = pairs[idx]

    a = personas[a_name]
    b = personas[b_name]

    interaction = _run_interaction(a, b, market)
    _record_interaction(a, b, interaction, market)
    _record_interaction(b, a, interaction, market)

    return interaction


def _run_interaction(a, b, market) -> dict:
    """Use the LLM to generate a short peer exchange summary."""
    prompt = f"""Two trading agents briefly exchange insights.

Agent A: {a.scratch.name}
Traits: {a.scratch.innate}
Focus: {a.scratch.currently}
Watchlist: {', '.join(a.scratch.watchlist)}

Agent B: {b.scratch.name}
Traits: {b.scratch.innate}
Focus: {b.scratch.currently}
Watchlist: {', '.join(b.scratch.watchlist)}

Market snapshot
{market.prices_str()}

Return JSON only (no markdown):
{{
  "topic": "short topic",
  "summary": "one sentence summary",
  "follow_up": "one follow-up action",
  "symbol": "TICKER or null"
}}
"""

    raw = ollama_request(prompt, max_tokens=120, stop=["\n\n"], timeout=120)
    try:
        cleaned = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        data = json.loads(cleaned)
        if data.get("summary"):
            return {
                "topic": str(data.get("topic", "peer sync")).strip(),
                "summary": str(data.get("summary", "")).strip(),
                "follow_up": str(data.get("follow_up", "review idea"))[:120],
                "symbol": data.get("symbol"),
                "agents": [a.scratch.name, b.scratch.name],
            }
    except Exception:
        pass

    # Fallback
    shared = list(set(a.scratch.watchlist) & set(b.scratch.watchlist))
    symbol = shared[0] if shared else market.SYMBOLS[0]
    return {
        "topic": f"watchlist focus {symbol}",
        "summary": f"{a.scratch.name} and {b.scratch.name} align on {symbol} watchlist priorities.",
        "follow_up": f"review {symbol} setup after next news tick",
        "symbol": symbol,
        "agents": [a.scratch.name, b.scratch.name],
    }


def _record_interaction(persona, other, interaction: dict, market) -> None:
    """Store the interaction as a memory event."""
    summary = interaction.get("summary", "peer interaction")
    topic = interaction.get("topic", "market sync")

    desc = f"{persona.scratch.name} discussed {topic} with {other.scratch.name}. {summary}"
    if desc in persona.a_mem.embeddings:
        emb = persona.a_mem.embeddings[desc]
    else:
        emb = get_embedding(desc)

    persona.a_mem.add_event(
        created=market.current_time,
        expiration=None,
        s=persona.scratch.name,
        p="discusses",
        o=topic,
        description=desc,
        keywords={"chat", "peer", "trade", topic.lower()},
        poignancy=6,
        embedding_pair=(desc, emb),
        filling=[],
    )

    persona.scratch.importance_trigger_curr -= 6
    persona.scratch.importance_ele_n += 1
