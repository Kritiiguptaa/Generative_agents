"""
File: middleware/memory_compression.py
Description: Prompt-level memory compression for the trading simulation.

  Distinct from reflect() (persona/cognitive_modules/reflect.py), which
  compresses the *memory store* by synthesising raw events into thought
  nodes. This module compresses the *retrieval output* -- what actually
  reaches the LLM on this particular step. Storage is left untouched.

  Four deterministic stages, no LLM call:

    1. Flatten + dedup     new_retrieve() scores the same node pool against
                           every focal point, so a high-poignancy headline is
                           returned once per focal point. Exact duplicates are
                           removed by node_id; near-duplicates by cosine
                           similarity over the embeddings already stored in
                           persona.a_mem.embeddings. When two nodes are
                           near-identical the NEWER one survives.

    2. Symbol-scoped       Nodes are grouped by ticker (read from node.keywords,
       recency pruning     which market_perceive stores lowercased) and only the
                           newest K per group are kept. This is what removes
                           superseded news: a step-5 NVDA headline is dropped
                           because step-30 and step-70 NVDA events are newer --
                           without the compressor ever being told they
                           contradict it.

    3. Supersession        Surviving nodes are annotated with their age and how
       annotation          many newer events exist for the same symbol, e.g.
                           "(50 steps ago, 3 newer NVDA events since)". Marking
                           rather than deleting keeps the information available
                           while cueing a weak model to discount it.

    4. Token budget        If the block still exceeds the budget, the
                           lowest-scoring nodes (recency x poignancy) are
                           dropped until it fits.

  METHODOLOGICAL CONSTRAINT -- do not remove:
    This module must NEVER read market.SCRIPTED_NEWS. That table is the
    contradiction oracle used by _check_stale_reasoning() in trading_reverie.py
    to SCORE stale-context hallucinations. A compressor that reads the same
    table would be graded against its own answer key and the measured
    reduction would be meaningless. Only signals a deployed system would
    genuinely have are used here: timestamps, embeddings, poignancy, symbol
    keywords, and current prices.
"""

import datetime
import traceback
from typing import Optional

# Sort floor for nodes with no timestamp -- keeps every sort key datetime-typed
# so a missing `created` can never raise TypeError mid-sort.
_EPOCH = datetime.datetime.min

# ---------------------------------------------------------------------------
# Defaults (mirrored in MiddlewareConfig; kept here so the module is importable
# standalone for tests)
# ---------------------------------------------------------------------------

_DEFAULTS = {
    "memory_compression_enabled":        True,
    "mc_keep_newest_per_symbol":         2,     # stage 2: K per symbol group
    "mc_near_duplicate_threshold":       0.95,  # stage 1: cosine cutoff
    "mc_annotate_superseded":            True,  # stage 3
    "mc_stale_age_steps":                20,    # stage 3: age before annotating
    "mc_max_memory_tokens":              350,   # stage 4 budget
    "mc_chars_per_token":                4,     # token estimator
}


def _cfg(config, key: str):
    """Read a config value from a MiddlewareConfig object, a dict, or defaults."""
    if config is None:
        return _DEFAULTS[key]
    if isinstance(config, dict):
        return config.get(key, _DEFAULTS[key])
    return getattr(config, key, _DEFAULTS[key])


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _estimate_tokens(text: str, config=None) -> int:
    """Cheap token estimate. Ollama has no local tokeniser exposed here, and a
    chars/token heuristic is accurate enough for budget enforcement."""
    cpt = _cfg(config, "mc_chars_per_token") or 4
    return max(0, len(text) // cpt)


def _cosine(v1, v2) -> float:
    """Cosine similarity over two float sequences. 0.0 on bad/missing input."""
    if not v1 or not v2:
        return 0.0
    try:
        import numpy as np
        a = np.asarray(v1, dtype=float)
        b = np.asarray(v2, dtype=float)
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0.0 or nb == 0.0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))
    except Exception:
        return 0.0


def _node_symbol(node, symbols) -> Optional[str]:
    """
    Resolve the ticker a node refers to, or None for general/non-symbol nodes.

    market_perceive._build_keywords() stores the symbol lowercased, and
    record_trade_fill() does the same, so keywords are checked first.
    Falls back to scanning the description for a known ticker, which covers
    reflection thoughts and peer-interaction nodes that never went through
    _build_keywords().
    """
    if not symbols:
        return None

    kw = getattr(node, "keywords", None) or set()
    try:
        lowered = {str(k).lower() for k in kw}
    except Exception:
        lowered = set()

    for sym in symbols:
        if sym.lower() in lowered:
            return sym

    desc = (getattr(node, "description", "") or "").upper()
    for sym in symbols:
        if sym.upper() in desc:
            return sym

    return None


def _age_in_steps(node, market) -> float:
    """Age of a node in simulation steps. 0.0 if it cannot be determined."""
    try:
        created = getattr(node, "created", None)
        now = getattr(market, "current_time", None)
        sec_per_step = float(getattr(market, "sec_per_step", 0) or 0)
        if created is None or now is None or sec_per_step <= 0:
            return 0.0
        delta = (now - created).total_seconds()
        return max(0.0, delta / sec_per_step)
    except Exception:
        return 0.0


def _created_key(node):
    """Sort key for chronological ordering; always returns a datetime."""
    created = getattr(node, "created", None)
    return created if isinstance(created, datetime.datetime) else _EPOCH


def _embedding_for(node, persona):
    """Look up a node's stored embedding via its embedding_key."""
    try:
        key = getattr(node, "embedding_key", None) or getattr(node, "description", "")
        return persona.a_mem.embeddings.get(key)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Stage 1 -- flatten + dedup
# ---------------------------------------------------------------------------

def _flatten_and_dedup(retrieved: dict, persona, config=None):
    """
    Flatten the focal-point dict into a single node list, removing exact
    duplicates (same node_id) and near-duplicates (cosine >= threshold).
    Newer nodes win ties. Returns (nodes, dropped_count).
    """
    threshold = float(_cfg(config, "mc_near_duplicate_threshold"))

    flat = []
    seen_ids = set()
    exact_dropped = 0

    for _focal_pt, nodes in (retrieved or {}).items():
        for node in (nodes or []):
            nid = getattr(node, "node_id", None)
            if nid is not None and nid in seen_ids:
                exact_dropped += 1
                continue
            if nid is not None:
                seen_ids.add(nid)
            flat.append(node)

    # Newest first so the survivor of a near-duplicate pair is the newer node.
    flat.sort(key=_created_key, reverse=True)

    kept = []
    near_dropped = 0
    for node in flat:
        emb = _embedding_for(node, persona)
        is_dup = False
        if emb:
            for existing in kept:
                existing_emb = _embedding_for(existing, persona)
                if existing_emb and _cosine(emb, existing_emb) >= threshold:
                    is_dup = True
                    break
        if is_dup:
            near_dropped += 1
        else:
            kept.append(node)

    return kept, exact_dropped + near_dropped


# ---------------------------------------------------------------------------
# Stage 2 -- symbol-scoped recency pruning
# ---------------------------------------------------------------------------

def _group_by_symbol(nodes, symbols):
    """Group nodes by ticker. Non-symbol nodes land under the None key."""
    groups = {}
    for node in nodes:
        sym = _node_symbol(node, symbols)
        groups.setdefault(sym, []).append(node)
    return groups


def _prune_by_symbol_recency(nodes, symbols, config=None):
    """
    Keep only the newest K nodes per symbol group. Returns
    (kept_nodes, dropped_count, newer_counts) where newer_counts maps
    node_id -> how many newer nodes existed for that same symbol in the input.
    """
    keep_k = int(_cfg(config, "mc_keep_newest_per_symbol"))
    groups = _group_by_symbol(nodes, symbols)

    kept = []
    dropped = 0
    newer_counts = {}

    for _sym, group in groups.items():
        group.sort(key=_created_key, reverse=True)
        for rank, node in enumerate(group):
            # rank 0 is newest, so rank == number of newer nodes in this group
            nid = getattr(node, "node_id", None)
            if nid is not None:
                newer_counts[nid] = rank
        kept.extend(group[:keep_k])
        dropped += max(0, len(group) - keep_k)

    return kept, dropped, newer_counts


# ---------------------------------------------------------------------------
# Stage 3 -- supersession annotation
# ---------------------------------------------------------------------------

def _annotate(node, market, symbols, newer_counts, config=None):
    """
    Render one node as a prompt line, appending an age/supersession marker
    when the node is old AND newer events exist for the same symbol.
    Returns (line, was_annotated).
    """
    desc = (getattr(node, "description", "") or "").strip()
    if not _cfg(config, "mc_annotate_superseded"):
        return f"- {desc}", False

    stale_after = float(_cfg(config, "mc_stale_age_steps"))
    age = _age_in_steps(node, market)
    nid = getattr(node, "node_id", None)
    newer = newer_counts.get(nid, 0)

    if age >= stale_after and newer > 0:
        sym = _node_symbol(node, symbols)
        label = f"{sym} " if sym else ""
        plural = "s" if newer != 1 else ""
        return (f"- {desc} ({int(age)} steps ago, "
                f"{newer} newer {label}event{plural} since)"), True

    if age >= stale_after:
        return f"- {desc} ({int(age)} steps ago)", False

    return f"- {desc}", False


# ---------------------------------------------------------------------------
# Stage 4 -- token budget
# ---------------------------------------------------------------------------

def _node_score(node, market) -> float:
    """Value score for budget eviction: poignancy decayed by age."""
    poignancy = float(getattr(node, "poignancy", 1) or 1)
    age = _age_in_steps(node, market)
    return poignancy / (1.0 + age)


def _enforce_budget(lines_with_nodes, market, config=None):
    """
    Drop the lowest-scoring lines until the block fits the token budget.
    lines_with_nodes: list of (line, node, was_annotated).
    Returns (kept_list, dropped_count).
    """
    budget = int(_cfg(config, "mc_max_memory_tokens"))
    if budget <= 0:
        return lines_with_nodes, 0

    kept = list(lines_with_nodes)
    dropped = 0

    def total_tokens(items):
        return _estimate_tokens("\n".join(line for line, _n, _a in items), config)

    while kept and total_tokens(kept) > budget:
        worst_idx = min(
            range(len(kept)),
            key=lambda i: _node_score(kept[i][1], market),
        )
        kept.pop(worst_idx)
        dropped += 1

    return kept, dropped


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def compress_memories(retrieved: dict, persona, market, config=None):
    """
    Compress the output of new_retrieve() into a prompt-ready memory block.

    ARGS:
      retrieved: dict mapping focal_point -> list[ConceptNode] (from new_retrieve)
      persona:   TradingPersona (read-only; a_mem.embeddings used for dedup)
      market:    MarketEnvironment (SYMBOLS, current_time, sec_per_step only --
                 never SCRIPTED_NEWS, see module docstring)
      config:    MiddlewareConfig, dict, or None for defaults

    RETURNS:
      (memory_context: str, stats: dict)

      stats keys: nodes_in, nodes_out, dropped_duplicate, dropped_stale,
                  dropped_budget, annotated_stale, compression_ratio,
                  chars_in, chars_out, enabled
    """
    raw_nodes = [n for nodes in (retrieved or {}).values() for n in (nodes or [])]
    nodes_in = len(raw_nodes)
    chars_in = sum(len(getattr(n, "description", "") or "") for n in raw_nodes)

    def _stats(nodes_out, dup, stale, budget, annotated, chars_out, enabled=True):
        return {
            "enabled":            enabled,
            "nodes_in":           nodes_in,
            "nodes_out":          nodes_out,
            "dropped_duplicate":  dup,
            "dropped_stale":      stale,
            "dropped_budget":     budget,
            "annotated_stale":    annotated,
            "compression_ratio":  round(nodes_out / nodes_in, 3) if nodes_in else 1.0,
            "chars_in":           chars_in,
            "chars_out":          chars_out,
        }

    # Disabled -> passthrough matching the original uncompressed behaviour,
    # so the ablation arm is a clean baseline.
    if not _cfg(config, "memory_compression_enabled"):
        lines = [f"- {getattr(n, 'description', '')}" for n in raw_nodes]
        text = "\n".join(lines) if lines else "No relevant memories."
        return text, _stats(nodes_in, 0, 0, 0, 0, len(text), enabled=False)

    if not raw_nodes:
        return "No relevant memories.", _stats(0, 0, 0, 0, 0, 0)

    try:
        symbols = list(getattr(market, "SYMBOLS", []) or [])

        # 1. flatten + dedup
        nodes, dropped_dup = _flatten_and_dedup(retrieved, persona, config)

        # 2. symbol-scoped recency pruning
        nodes, dropped_stale, newer_counts = _prune_by_symbol_recency(
            nodes, symbols, config
        )

        # 3. supersession annotation (newest first for readability)
        nodes.sort(key=_created_key, reverse=True)
        annotated = []
        for node in nodes:
            line, was_marked = _annotate(node, market, symbols, newer_counts, config)
            annotated.append((line, node, was_marked))

        # 4. token budget
        kept, dropped_budget = _enforce_budget(annotated, market, config)

        text = "\n".join(line for line, _n, _a in kept)
        if not text:
            text = "No relevant memories."

        return text, _stats(
            nodes_out=len(kept),
            dup=dropped_dup,
            stale=dropped_stale,
            budget=dropped_budget,
            annotated=sum(1 for _l, _n, marked in kept if marked),
            chars_out=len(text),
        )

    except Exception as exc:
        # Never break the simulation: fall back to the uncompressed block.
        #
        # But say so. This used to swallow the exception and return stats that
        # were indistinguishable from a genuine no-op compression, while the
        # caller still recorded enabled=True. A middleware arm whose
        # compression crashed on every step would report as a working
        # middleware arm that happened to find nothing to compress, and the
        # ablation would silently be comparing baseline against baseline.
        traceback.print_exc()
        print(f"  [memory_compression] FAILED for "
              f"{getattr(getattr(persona, 'scratch', None), 'name', '?')}: "
              f"{type(exc).__name__}: {exc} -- falling back to uncompressed "
              f"context. This step's compression stats are not meaningful.")
        lines = [f"- {getattr(n, 'description', '')}" for n in raw_nodes]
        text = "\n".join(lines) if lines else "No relevant memories."
        stats = _stats(nodes_in, 0, 0, 0, 0, len(text))
        stats["failed"] = True
        stats["error"] = f"{type(exc).__name__}: {exc}"
        return text, stats
