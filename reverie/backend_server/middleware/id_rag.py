"""
File: middleware/id_rag.py
Description: ID-RAG (Identity Retrieval-Augmented Generation) for the trading simulation.

Instead of prepending the full persona block every step (static anchoring),
ID-RAG stores the agent's identity as a knowledge graph and retrieves only
the nodes most relevant to the current trading context via cosine similarity.

Flow per LLM call:
  1. On first call for an agent, build a knowledge graph from scratch fields.
  2. Embed each node using nomic-embed-text (same model as associative memory).
  3. Embed the current context (first N chars of memory context).
  4. Retrieve top-k most similar nodes.
  5. Always append the forbidden node regardless of similarity score.
  6. Format retrieved nodes and prepend to the prompt.

Dynamic updates (called from trading_reverie.py):
  - update_graph_belief()          after reflect() fires
  - update_graph_relationship()    after maybe_interaction() returns a result
  - update_graph_current_situation() every step (cheap — skips re-embed if unchanged)
"""

import hashlib
import sys
import os
from typing import Optional

import numpy as np

try:
    import networkx as nx
    _NX_AVAILABLE = True
except ImportError:
    _NX_AVAILABLE = False

# Try to reuse get_embedding from gpt_structure (same Ollama model, avoids
# duplicating the retry/fallback logic already written there).
try:
    from persona.prompt_template.gpt_structure import get_embedding as _gpt_get_embedding
    _GPT_EMBED_AVAILABLE = True
except ImportError:
    _GPT_EMBED_AVAILABLE = False

import requests

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_agent_graphs: dict = {}       # persona_name -> nx.DiGraph
_embedding_cache: dict = {}    # sha256(text) -> list[float]

# ---------------------------------------------------------------------------
# Config dataclass (plain dict of defaults — no dependency on MiddlewareConfig
# so this module can be imported standalone for tests)
# ---------------------------------------------------------------------------

_DEFAULTS = {
    "id_rag_enabled":                   True,
    "id_rag_top_k":                     4,
    "id_rag_always_include_forbidden":  True,
    "id_rag_max_context_chars":         300,
    "id_rag_embed_cache_enabled":       True,
    "id_rag_dynamic_updates":           True,
    "ollama_base_url":                  "http://localhost:11434",
    "ollama_embed_model":               "nomic-embed-text",
}


def _cfg(config, key: str):
    """Read a config value from a MiddlewareConfig object, dict, or fall back to default."""
    if config is None:
        return _DEFAULTS[key]
    if isinstance(config, dict):
        return config.get(key, _DEFAULTS[key])
    return getattr(config, key, _DEFAULTS[key])


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def _embed_text(text: str, config=None) -> list:
    """
    Return a float embedding for text.
    Tries get_embedding() from gpt_structure first (reuses its retry logic).
    Falls back to a direct Ollama HTTP call if the import failed.
    Returns [] on any failure — callers must handle this.
    """
    text = text.strip().replace("\n", " ")
    if not text:
        return []

    cache_enabled = _cfg(config, "id_rag_embed_cache_enabled")
    key = hashlib.sha256(text.encode()).hexdigest()

    if cache_enabled and key in _embedding_cache:
        return _embedding_cache[key]

    embedding = []

    if _GPT_EMBED_AVAILABLE:
        try:
            result = _gpt_get_embedding(text)
            if isinstance(result, list) and result:
                embedding = result
        except Exception:
            pass

    if not embedding:
        # Direct Ollama call (fallback when gpt_structure not importable)
        base_url = _cfg(config, "ollama_base_url")
        model    = _cfg(config, "ollama_embed_model")
        for endpoint in ["/api/embeddings", "/api/embed"]:
            try:
                resp = requests.post(
                    base_url + endpoint,
                    json={"model": model, "prompt": text},
                    timeout=30,
                )
                resp.raise_for_status()
                data = resp.json()
                if isinstance(data, dict) and "embedding" in data:
                    embedding = data["embedding"]
                    break
                if isinstance(data, dict) and "data" in data:
                    embedding = data["data"][0]["embedding"]
                    break
            except Exception:
                continue

    if cache_enabled and embedding:
        _embedding_cache[key] = embedding

    return embedding


def _cosine_similarity(v1: list, v2: list) -> float:
    """Cosine similarity between two float lists. Returns 0.0 on bad input."""
    if not v1 or not v2:
        return 0.0
    try:
        a = np.array(v1, dtype=float)
        b = np.array(v2, dtype=float)
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0.0 or nb == 0.0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Graph building
# ---------------------------------------------------------------------------

def _node_id(node_type: str, index: int) -> str:
    return f"{node_type}_{index}"


def build_graph(persona, config=None):
    """
    Build a knowledge graph from persona.scratch trading fields.
    All fields are read dynamically — nothing is hardcoded.
    Stores result in _agent_graphs[persona.scratch.name].
    """
    if not _NX_AVAILABLE:
        return None

    s = persona.scratch
    name = getattr(s, "name", "unknown")
    G = nx.DiGraph()

    node_counter = {"n": 0}

    def add_node(ntype: str, content: str, **attrs) -> str:
        nid = _node_id(ntype, node_counter["n"])
        node_counter["n"] += 1
        G.add_node(nid, type=ntype, content=content, embedding=[], **attrs)
        return nid

    # ── Trait nodes (split innate by comma) ──────────────────────────────────
    innate = getattr(s, "innate", "") or ""
    trait_ids = []
    for trait in innate.split(","):
        trait = trait.strip()
        if trait:
            nid = add_node("trait", trait)
            trait_ids.append(nid)

    # ── Risk profile node ─────────────────────────────────────────────────────
    risk_tolerance  = getattr(s, "risk_tolerance", "moderate") or "moderate"
    risk_limit      = float(getattr(s, "risk_limit_per_trade", 0.05)) * 100
    risk_content    = (
        f"Risk tolerance: {risk_tolerance}. "
        f"Max single trade: {risk_limit:.1f}% of portfolio."
    )
    risk_id = add_node("risk_profile", risk_content)

    # ── Strategy node ─────────────────────────────────────────────────────────
    strategy = getattr(s, "trading_strategy", "") or ""
    strategy_id = None
    if strategy:
        strategy_id = add_node("strategy", strategy)

    # ── Specialization node ───────────────────────────────────────────────────
    specialization = getattr(s, "specialization", "") or ""
    spec_id = None
    if specialization:
        spec_id = add_node("specialization", specialization)

    # ── Watchlist nodes (one per symbol) ─────────────────────────────────────
    watchlist = getattr(s, "watchlist", []) or []
    watchlist_ids = []
    for symbol in watchlist:
        nid = add_node("watchlist", f"Monitors {symbol}", symbol=symbol)
        watchlist_ids.append(nid)

    # ── Background node (from learned field) ─────────────────────────────────
    learned = getattr(s, "learned", "") or ""
    bg_id = None
    if learned:
        bg_id = add_node("background", learned[:300])  # cap length

    # ── Current situation node ────────────────────────────────────────────────
    currently = getattr(s, "currently", "") or ""
    curr_id = None
    if currently:
        curr_id = add_node("current_situation", currently)

    # ── Trader type node ──────────────────────────────────────────────────────
    trader_type = getattr(s, "trader_type", "") or ""
    ttype_id = None
    if trader_type:
        ttype_id = add_node("trader_type", f"Trader type: {trader_type}")

    # ── Forbidden node (always injected) ─────────────────────────────────────
    forbidden_content = (
        "Do not adopt other agents' views or follow market panic. "
        "Always act within your defined risk limits. "
        "Never request a trade that exceeds your cash balance or current holdings."
    )
    forbidden_id = add_node("forbidden", forbidden_content)

    # ── Edges (semantic relationships) ────────────────────────────────────────
    for tid in trait_ids:
        G.add_edge(tid, risk_id, relation="influences")
    if strategy_id:
        for wid in watchlist_ids:
            G.add_edge(strategy_id, wid, relation="targets")
        if ttype_id:
            G.add_edge(ttype_id, strategy_id, relation="defines")
    if bg_id and curr_id:
        G.add_edge(bg_id, curr_id, relation="leads_to")
    if spec_id and strategy_id:
        G.add_edge(spec_id, strategy_id, relation="shapes")

    _agent_graphs[name] = G
    return G


def _get_or_build_graph(persona, config=None):
    name = getattr(persona.scratch, "name", "unknown")
    if name not in _agent_graphs:
        build_graph(persona, config)
    return _agent_graphs.get(name)


# ---------------------------------------------------------------------------
# Embedding lazy-load
# ---------------------------------------------------------------------------

def _ensure_embeddings(G, config=None):
    """Embed any node that doesn't have an embedding yet."""
    if G is None:
        return
    for nid, data in G.nodes(data=True):
        if not data.get("embedding"):
            emb = _embed_text(data.get("content", ""), config)
            G.nodes[nid]["embedding"] = emb


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def retrieve_relevant_nodes(persona, context_text: str, config=None) -> list:
    """
    Embed context_text and return top-k most similar non-forbidden nodes,
    plus the forbidden node(s) appended unconditionally.

    Returns list of dicts: [{"type", "content", "score"}, ...]
    """
    if not _NX_AVAILABLE:
        return []

    top_k               = _cfg(config, "id_rag_top_k")
    always_forbidden    = _cfg(config, "id_rag_always_include_forbidden")
    max_ctx_chars       = _cfg(config, "id_rag_max_context_chars")

    G = _get_or_build_graph(persona, config)
    if G is None or len(G.nodes) == 0:
        return []

    _ensure_embeddings(G, config)

    ctx = context_text[:max_ctx_chars].strip()
    ctx_emb = _embed_text(ctx, config)
    if not ctx_emb:
        # Can't embed context — return all non-forbidden nodes up to top_k
        results = []
        for nid, data in G.nodes(data=True):
            if data.get("type") != "forbidden":
                results.append({"type": data["type"], "content": data["content"], "score": 0.0})
        results = results[:top_k]
        if always_forbidden:
            for nid, data in G.nodes(data=True):
                if data.get("type") == "forbidden":
                    results.append({"type": "forbidden", "content": data["content"], "score": 1.0})
        return results

    scored = []
    forbidden_nodes = []

    for nid, data in G.nodes(data=True):
        node_type = data.get("type", "")
        content   = data.get("content", "")
        emb       = data.get("embedding", [])

        if node_type == "forbidden":
            forbidden_nodes.append({"type": node_type, "content": content, "score": 1.0})
            continue

        score = _cosine_similarity(ctx_emb, emb)
        scored.append({"type": node_type, "content": content, "score": score})

    scored.sort(key=lambda x: x["score"], reverse=True)
    results = scored[:top_k]

    if always_forbidden:
        results.extend(forbidden_nodes)

    return results


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def format_retrieved_nodes(nodes: list, persona_name: str) -> str:
    """
    Format retrieved node list into a prompt-ready identity block.
    """
    if not nodes:
        return ""

    lines = [f"[IDENTITY — {persona_name}]"]
    for node in nodes:
        ntype   = node.get("type", "info").upper()
        content = node.get("content", "")
        lines.append(f"{ntype}: {content}")
    lines.append("[END IDENTITY]")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def id_rag_anchor(persona, memory_context: str, config=None) -> str:
    """
    Build a context-relevant identity block and prepend it to memory_context.
    Drop-in replacement for anchor_memory_context() in persona_anchoring.py.

    Silent fail: returns memory_context unchanged on any error.
    """
    if not _cfg(config, "id_rag_enabled"):
        return memory_context

    if not _NX_AVAILABLE:
        return memory_context

    try:
        name    = getattr(persona.scratch, "name", "unknown")
        nodes   = retrieve_relevant_nodes(persona, memory_context, config)
        block   = format_retrieved_nodes(nodes, name)
        if not block:
            return memory_context
        return block + "\n\n" + memory_context
    except Exception:
        return memory_context


# ---------------------------------------------------------------------------
# Dynamic graph updates
# ---------------------------------------------------------------------------

def update_graph_belief(persona_name: str, belief_text: str, config=None) -> None:
    """
    Add a new belief node after a reflection event.
    Immediately embeds the belief so it's ready for retrieval.
    No-op if graph doesn't exist or dynamic updates are disabled.
    """
    if not _cfg(config, "id_rag_dynamic_updates"):
        return
    if not _NX_AVAILABLE:
        return

    G = _agent_graphs.get(persona_name)
    if G is None:
        return

    try:
        existing_beliefs = [
            nid for nid, d in G.nodes(data=True)
            if d.get("type") == "belief" and d.get("content") == belief_text
        ]
        if existing_beliefs:
            return  # deduplicate

        count = sum(1 for _, d in G.nodes(data=True) if d.get("type") == "belief")
        nid   = f"belief_{count}"
        emb   = _embed_text(belief_text, config)
        G.add_node(nid, type="belief", content=belief_text, embedding=emb)
    except Exception:
        pass


def update_graph_relationship(
    persona_name: str,
    other_agent_name: str,
    relationship_desc: str,
    config=None,
) -> None:
    """
    Add or update a relationship node after a peer interaction.
    """
    if not _cfg(config, "id_rag_dynamic_updates"):
        return
    if not _NX_AVAILABLE:
        return

    G = _agent_graphs.get(persona_name)
    if G is None:
        return

    try:
        content = f"Relationship with {other_agent_name}: {relationship_desc}"

        # Update existing node if one already exists for this pair
        for nid, data in G.nodes(data=True):
            if (data.get("type") == "relationship"
                    and other_agent_name in data.get("content", "")):
                G.nodes[nid]["content"]   = content
                G.nodes[nid]["embedding"] = _embed_text(content, config)
                return

        count = sum(1 for _, d in G.nodes(data=True) if d.get("type") == "relationship")
        nid   = f"relationship_{count}"
        emb   = _embed_text(content, config)
        G.add_node(nid, type="relationship", content=content, embedding=emb)
    except Exception:
        pass


def update_graph_current_situation(
    persona_name: str,
    new_situation: str,
    config=None,
) -> None:
    """
    Replace the current_situation node with updated text.
    Skips re-embedding if the text hasn't changed (cache hit).
    """
    if not _cfg(config, "id_rag_dynamic_updates"):
        return
    if not _NX_AVAILABLE:
        return

    G = _agent_graphs.get(persona_name)
    if G is None:
        return

    try:
        for nid, data in G.nodes(data=True):
            if data.get("type") == "current_situation":
                if data.get("content") == new_situation:
                    return  # unchanged — skip
                G.nodes[nid]["content"]   = new_situation
                G.nodes[nid]["embedding"] = _embed_text(new_situation, config)
                return

        # No existing node — create one
        count = sum(1 for _, d in G.nodes(data=True) if d.get("type") == "current_situation")
        nid   = f"current_situation_{count}"
        emb   = _embed_text(new_situation, config)
        G.add_node(nid, type="current_situation", content=new_situation, embedding=emb)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------

def id_rag_cache_clear() -> None:
    """Clear all in-memory graph and embedding state."""
    _agent_graphs.clear()
    _embedding_cache.clear()


def id_rag_graph_for(persona_name: str):
    """Return the graph for a given agent name, or None. Used in tests."""
    return _agent_graphs.get(persona_name)
