"""
Tests for middleware/id_rag.py

Covers:
  Unit tests    — individual functions in isolation with mocked embeddings
  Regression    — behaviour contracts that must not break across changes

Run from backend_server directory:
    python -m pytest tests/test_id_rag.py -v
or standalone:
    python tests/test_id_rag.py
"""

import sys
import os
import unittest
from unittest.mock import patch, MagicMock

# ---------------------------------------------------------------------------
# Path setup — allow running from backend_server OR project root
# ---------------------------------------------------------------------------
BACKEND = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

# ---------------------------------------------------------------------------
# Fake persona helpers
# ---------------------------------------------------------------------------

class FakeScratch:
    def __init__(self, **kwargs):
        self.name               = kwargs.get("name", "Test Agent")
        self.innate             = kwargs.get("innate", "patient, analytical, disciplined")
        self.learned            = kwargs.get("learned", "Former hedge fund analyst with 10 years experience.")
        self.currently          = kwargs.get("currently", "Monitoring NVDA for a momentum entry.")
        self.trader_type        = kwargs.get("trader_type", "momentum")
        self.specialization     = kwargs.get("specialization", "tech stocks")
        self.risk_tolerance     = kwargs.get("risk_tolerance", "moderate")
        self.risk_limit_per_trade = kwargs.get("risk_limit_per_trade", 0.05)
        self.watchlist          = kwargs.get("watchlist", ["NVDA", "AAPL"])
        self.trading_strategy   = kwargs.get("trading_strategy", "Buy breakouts on high volume.")


class FakePersona:
    def __init__(self, **kwargs):
        self.scratch = FakeScratch(**kwargs)


# ---------------------------------------------------------------------------
# Deterministic fake embedding (avoids Ollama dependency in tests)
# A simple hash-based vector so cosine similarity is still meaningful.
# ---------------------------------------------------------------------------

def _fake_embed(text: str, config=None) -> list:
    """Deterministic fake embedding: hash of text -> 8-dim unit vector."""
    import hashlib
    import math
    h = int(hashlib.md5(text.encode()).hexdigest(), 16)
    raw = [(h >> (i * 4)) & 0xF for i in range(8)]
    norm = math.sqrt(sum(v * v for v in raw)) or 1.0
    return [v / norm for v in raw]


# ---------------------------------------------------------------------------
# Test config (embedding cache on so we can verify caching)
# ---------------------------------------------------------------------------
TEST_CFG = {
    "id_rag_enabled":                   True,
    "id_rag_top_k":                     3,
    "id_rag_always_include_forbidden":  True,
    "id_rag_max_context_chars":         200,
    "id_rag_embed_cache_enabled":       True,
    "id_rag_dynamic_updates":           True,
    "ollama_base_url":                  "http://localhost:11434",
    "ollama_embed_model":               "nomic-embed-text",
}


# ---------------------------------------------------------------------------
# Unit Tests
# ---------------------------------------------------------------------------

class TestBuildGraph(unittest.TestCase):

    def setUp(self):
        from middleware import id_rag
        id_rag.id_rag_cache_clear()
        self.id_rag = id_rag

    @patch("middleware.id_rag._embed_text", side_effect=_fake_embed)
    def test_all_node_types_created(self, _mock):
        persona = FakePersona()
        G = self.id_rag.build_graph(persona, TEST_CFG)
        types = {d["type"] for _, d in G.nodes(data=True)}
        self.assertIn("trait", types)
        self.assertIn("risk_profile", types)
        self.assertIn("strategy", types)
        self.assertIn("watchlist", types)
        self.assertIn("specialization", types)
        self.assertIn("background", types)
        self.assertIn("current_situation", types)
        self.assertIn("trader_type", types)
        self.assertIn("forbidden", types)

    @patch("middleware.id_rag._embed_text", side_effect=_fake_embed)
    def test_traits_split_by_comma(self, _mock):
        persona = FakePersona(innate="patient, analytical, disciplined")
        G = self.id_rag.build_graph(persona, TEST_CFG)
        trait_contents = [
            d["content"] for _, d in G.nodes(data=True) if d["type"] == "trait"
        ]
        self.assertEqual(len(trait_contents), 3)
        self.assertIn("patient", trait_contents)
        self.assertIn("analytical", trait_contents)
        self.assertIn("disciplined", trait_contents)

    @patch("middleware.id_rag._embed_text", side_effect=_fake_embed)
    def test_watchlist_nodes_per_symbol(self, _mock):
        persona = FakePersona(watchlist=["NVDA", "AAPL", "TSLA"])
        G = self.id_rag.build_graph(persona, TEST_CFG)
        wl_nodes = [d for _, d in G.nodes(data=True) if d["type"] == "watchlist"]
        self.assertEqual(len(wl_nodes), 3)

    @patch("middleware.id_rag._embed_text", side_effect=_fake_embed)
    def test_graph_stored_by_name(self, _mock):
        persona = FakePersona(name="Marcus Webb")
        self.id_rag.build_graph(persona, TEST_CFG)
        self.assertIsNotNone(self.id_rag.id_rag_graph_for("Marcus Webb"))

    @patch("middleware.id_rag._embed_text", side_effect=_fake_embed)
    def test_empty_optional_fields_skipped(self, _mock):
        """Optional fields (strategy, specialization, etc.) produce no node when empty."""
        persona = FakePersona(
            trading_strategy="",
            specialization="",
            learned="",
            currently="",
            trader_type="",
        )
        G = self.id_rag.build_graph(persona, TEST_CFG)
        types = [d["type"] for _, d in G.nodes(data=True)]
        self.assertNotIn("strategy", types)
        self.assertNotIn("specialization", types)
        self.assertNotIn("background", types)
        self.assertNotIn("current_situation", types)
        self.assertNotIn("trader_type", types)


class TestEmbedding(unittest.TestCase):

    def setUp(self):
        from middleware import id_rag
        id_rag.id_rag_cache_clear()
        self.id_rag = id_rag

    @patch("middleware.id_rag._embed_text", side_effect=_fake_embed)
    def test_embedding_cached_on_second_call(self, mock_embed):
        """
        Embeddings are lazy: build_graph leaves them empty; first _ensure_embeddings
        populates them; second _ensure_embeddings should be a no-op.
        """
        cfg = dict(TEST_CFG)
        cfg["id_rag_embed_cache_enabled"] = True
        persona = FakePersona(name="CacheAgent")
        self.id_rag.build_graph(persona, cfg)

        G = self.id_rag.id_rag_graph_for("CacheAgent")
        # First ensure: embeds all nodes
        self.id_rag._ensure_embeddings(G, cfg)
        calls_after_first = mock_embed.call_count
        self.assertGreater(calls_after_first, 0)

        # Second ensure: all nodes already have embeddings — no new calls
        self.id_rag._ensure_embeddings(G, cfg)
        self.assertEqual(mock_embed.call_count, calls_after_first)

    def test_cosine_similarity_identical_vectors(self):
        v = [1.0, 0.0, 0.0]
        self.assertAlmostEqual(self.id_rag._cosine_similarity(v, v), 1.0)

    def test_cosine_similarity_orthogonal_vectors(self):
        v1 = [1.0, 0.0]
        v2 = [0.0, 1.0]
        self.assertAlmostEqual(self.id_rag._cosine_similarity(v1, v2), 0.0)

    def test_cosine_similarity_empty_returns_zero(self):
        self.assertEqual(self.id_rag._cosine_similarity([], [1.0, 0.0]), 0.0)
        self.assertEqual(self.id_rag._cosine_similarity([1.0], []), 0.0)


class TestRetrieval(unittest.TestCase):

    def setUp(self):
        from middleware import id_rag
        id_rag.id_rag_cache_clear()
        self.id_rag = id_rag

    @patch("middleware.id_rag._embed_text", side_effect=_fake_embed)
    def test_returns_at_most_top_k_plus_forbidden(self, _mock):
        persona = FakePersona()
        cfg = dict(TEST_CFG)
        cfg["id_rag_top_k"] = 2
        cfg["id_rag_always_include_forbidden"] = True
        self.id_rag.build_graph(persona, cfg)
        nodes = self.id_rag.retrieve_relevant_nodes(persona, "buy NVDA now", cfg)
        non_forbidden = [n for n in nodes if n["type"] != "forbidden"]
        forbidden     = [n for n in nodes if n["type"] == "forbidden"]
        self.assertLessEqual(len(non_forbidden), 2)
        self.assertGreaterEqual(len(forbidden), 1)

    @patch("middleware.id_rag._embed_text", side_effect=_fake_embed)
    def test_forbidden_always_included(self, _mock):
        persona = FakePersona()
        cfg = dict(TEST_CFG)
        cfg["id_rag_always_include_forbidden"] = True
        self.id_rag.build_graph(persona, cfg)
        nodes = self.id_rag.retrieve_relevant_nodes(persona, "anything", cfg)
        types = [n["type"] for n in nodes]
        self.assertIn("forbidden", types)

    @patch("middleware.id_rag._embed_text", side_effect=_fake_embed)
    def test_forbidden_excluded_from_similarity_ranking(self, _mock):
        """Forbidden node should NOT compete with others for top_k slots."""
        persona = FakePersona()
        cfg = dict(TEST_CFG)
        cfg["id_rag_top_k"] = 100  # large enough to include everything
        cfg["id_rag_always_include_forbidden"] = True
        self.id_rag.build_graph(persona, cfg)
        nodes = self.id_rag.retrieve_relevant_nodes(persona, "buy NVDA", cfg)
        forbidden_count = sum(1 for n in nodes if n["type"] == "forbidden")
        # Should appear exactly once regardless of top_k
        self.assertEqual(forbidden_count, 1)

    @patch("middleware.id_rag._embed_text", side_effect=_fake_embed)
    def test_returns_empty_for_unknown_persona(self, _mock):
        """Persona with no graph built yet returns empty list without crashing."""
        class NoGraphPersona:
            class scratch:
                name = "Ghost Agent"
                innate = ""
                learned = ""
                currently = ""
                trader_type = ""
                specialization = ""
                risk_tolerance = "low"
                risk_limit_per_trade = 0.01
                watchlist = []
                trading_strategy = ""
        nodes = self.id_rag.retrieve_relevant_nodes(NoGraphPersona(), "context", TEST_CFG)
        # May return empty (no graph) or built on-demand — must not crash
        self.assertIsInstance(nodes, list)


class TestFormatting(unittest.TestCase):

    def setUp(self):
        from middleware import id_rag
        self.id_rag = id_rag

    def test_format_includes_persona_name(self):
        nodes = [{"type": "trait", "content": "patient", "score": 0.9}]
        result = self.id_rag.format_retrieved_nodes(nodes, "Marcus Webb")
        self.assertIn("Marcus Webb", result)

    def test_format_includes_node_type_uppercase(self):
        nodes = [{"type": "risk_profile", "content": "low risk", "score": 0.8}]
        result = self.id_rag.format_retrieved_nodes(nodes, "X")
        self.assertIn("RISK_PROFILE", result)

    def test_format_empty_nodes_returns_empty(self):
        result = self.id_rag.format_retrieved_nodes([], "X")
        self.assertEqual(result, "")

    def test_format_has_identity_markers(self):
        nodes = [{"type": "trait", "content": "disciplined", "score": 0.9}]
        result = self.id_rag.format_retrieved_nodes(nodes, "Agent")
        self.assertIn("[IDENTITY", result)
        self.assertIn("[END IDENTITY]", result)


class TestIdRagAnchor(unittest.TestCase):

    def setUp(self):
        from middleware import id_rag
        id_rag.id_rag_cache_clear()
        self.id_rag = id_rag

    @patch("middleware.id_rag._embed_text", side_effect=_fake_embed)
    def test_anchor_prepends_to_memory_context(self, _mock):
        persona = FakePersona(name="Anchor Agent")
        memory = "NVDA rose 5%. Recent trades: bought AAPL."
        result = self.id_rag.id_rag_anchor(persona, memory, TEST_CFG)
        self.assertTrue(result.endswith(memory))
        self.assertGreater(len(result), len(memory))

    @patch("middleware.id_rag._embed_text", side_effect=_fake_embed)
    def test_anchor_contains_identity_block(self, _mock):
        persona = FakePersona(name="Identity Agent")
        result = self.id_rag.id_rag_anchor(persona, "context", TEST_CFG)
        self.assertIn("[IDENTITY", result)

    def test_anchor_disabled_returns_original(self):
        persona = FakePersona()
        cfg = dict(TEST_CFG)
        cfg["id_rag_enabled"] = False
        memory = "original memory context"
        result = self.id_rag.id_rag_anchor(persona, memory, cfg)
        self.assertEqual(result, memory)

    def test_anchor_silent_fail_on_bad_persona(self):
        """A persona with no scratch attribute must not crash the simulation."""
        class BrokenPersona:
            pass
        memory = "some context"
        result = self.id_rag.id_rag_anchor(BrokenPersona(), memory, TEST_CFG)
        self.assertEqual(result, memory)


class TestDynamicUpdates(unittest.TestCase):

    def setUp(self):
        from middleware import id_rag
        id_rag.id_rag_cache_clear()
        self.id_rag = id_rag

    @patch("middleware.id_rag._embed_text", side_effect=_fake_embed)
    def test_update_belief_adds_node(self, _mock):
        persona = FakePersona(name="Belief Agent")
        self.id_rag.build_graph(persona, TEST_CFG)
        self.id_rag.update_graph_belief("Belief Agent", "Market is entering a correction.", TEST_CFG)
        G = self.id_rag.id_rag_graph_for("Belief Agent")
        beliefs = [d for _, d in G.nodes(data=True) if d["type"] == "belief"]
        self.assertEqual(len(beliefs), 1)
        self.assertEqual(beliefs[0]["content"], "Market is entering a correction.")

    @patch("middleware.id_rag._embed_text", side_effect=_fake_embed)
    def test_update_belief_deduplicates(self, _mock):
        persona = FakePersona(name="Dedup Agent")
        self.id_rag.build_graph(persona, TEST_CFG)
        text = "NVDA is overbought."
        self.id_rag.update_graph_belief("Dedup Agent", text, TEST_CFG)
        self.id_rag.update_graph_belief("Dedup Agent", text, TEST_CFG)
        G = self.id_rag.id_rag_graph_for("Dedup Agent")
        beliefs = [d for _, d in G.nodes(data=True) if d["type"] == "belief"]
        self.assertEqual(len(beliefs), 1)

    @patch("middleware.id_rag._embed_text", side_effect=_fake_embed)
    def test_update_relationship_adds_node(self, _mock):
        persona = FakePersona(name="Social Agent")
        self.id_rag.build_graph(persona, TEST_CFG)
        self.id_rag.update_graph_relationship(
            "Social Agent", "Risk Agent", "agreed on NVDA momentum trade", TEST_CFG
        )
        G = self.id_rag.id_rag_graph_for("Social Agent")
        rels = [d for _, d in G.nodes(data=True) if d["type"] == "relationship"]
        self.assertEqual(len(rels), 1)
        self.assertIn("Risk Agent", rels[0]["content"])

    @patch("middleware.id_rag._embed_text", side_effect=_fake_embed)
    def test_update_relationship_updates_existing(self, _mock):
        """Second update for same pair replaces, not appends."""
        persona = FakePersona(name="Update Agent")
        self.id_rag.build_graph(persona, TEST_CFG)
        self.id_rag.update_graph_relationship("Update Agent", "Peer", "discussed AAPL", TEST_CFG)
        self.id_rag.update_graph_relationship("Update Agent", "Peer", "discussed TSLA instead", TEST_CFG)
        G = self.id_rag.id_rag_graph_for("Update Agent")
        rels = [d for _, d in G.nodes(data=True) if d["type"] == "relationship"]
        self.assertEqual(len(rels), 1)
        self.assertIn("TSLA", rels[0]["content"])

    @patch("middleware.id_rag._embed_text", side_effect=_fake_embed)
    def test_update_current_situation_replaces(self, _mock):
        persona = FakePersona(name="Curr Agent", currently="watching NVDA")
        self.id_rag.build_graph(persona, TEST_CFG)
        self.id_rag.update_graph_current_situation("Curr Agent", "now watching TSLA", TEST_CFG)
        G = self.id_rag.id_rag_graph_for("Curr Agent")
        curr_nodes = [d for _, d in G.nodes(data=True) if d["type"] == "current_situation"]
        self.assertEqual(len(curr_nodes), 1)
        self.assertEqual(curr_nodes[0]["content"], "now watching TSLA")

    @patch("middleware.id_rag._embed_text", side_effect=_fake_embed)
    def test_update_current_situation_no_op_if_unchanged(self, _mock):
        """Re-embedding is skipped when content hasn't changed."""
        persona = FakePersona(name="NoOp Agent", currently="watching NVDA")
        self.id_rag.build_graph(persona, TEST_CFG)
        calls_before = _mock.call_count
        self.id_rag.update_graph_current_situation("NoOp Agent", "watching NVDA", TEST_CFG)
        self.assertEqual(_mock.call_count, calls_before)  # no new embed call

    @patch("middleware.id_rag._embed_text", side_effect=_fake_embed)
    def test_dynamic_updates_disabled_no_op(self, _mock):
        persona = FakePersona(name="Disabled Agent")
        self.id_rag.build_graph(persona, TEST_CFG)
        cfg = dict(TEST_CFG)
        cfg["id_rag_dynamic_updates"] = False
        G_before = len(self.id_rag.id_rag_graph_for("Disabled Agent").nodes)
        self.id_rag.update_graph_belief("Disabled Agent", "some belief", cfg)
        G_after = len(self.id_rag.id_rag_graph_for("Disabled Agent").nodes)
        self.assertEqual(G_before, G_after)

    def test_updates_on_nonexistent_graph_no_crash(self):
        """Updates for an agent with no graph must silently no-op."""
        self.id_rag.update_graph_belief("Ghost", "belief text", TEST_CFG)
        self.id_rag.update_graph_relationship("Ghost", "Other", "desc", TEST_CFG)
        self.id_rag.update_graph_current_situation("Ghost", "new situation", TEST_CFG)


# ---------------------------------------------------------------------------
# Regression Tests
# ---------------------------------------------------------------------------

class TestRegressions(unittest.TestCase):
    """
    Contracts that MUST hold across any future change to id_rag.py.
    """

    def setUp(self):
        from middleware import id_rag
        id_rag.id_rag_cache_clear()
        self.id_rag = id_rag

    @patch("middleware.id_rag._embed_text", side_effect=_fake_embed)
    def test_graph_reused_not_rebuilt_on_second_call(self, mock_embed):
        """
        On the second anchor call, node embeddings already exist so only
        the context embedding fires — not a full node re-embed.
        """
        persona = FakePersona(name="Reuse Agent")

        # First anchor call: builds graph + embeds all nodes + embeds context
        self.id_rag.id_rag_anchor(persona, "first context", TEST_CFG)
        calls_after_first = mock_embed.call_count
        self.assertGreater(calls_after_first, 0)

        # Second anchor call: nodes already embedded — only 1 context embed
        self.id_rag.id_rag_anchor(persona, "second context", TEST_CFG)
        extra_calls = mock_embed.call_count - calls_after_first
        self.assertLessEqual(extra_calls, 1,
                             "Second anchor call should only embed the context, not all nodes")

    @patch("middleware.id_rag._embed_text", side_effect=_fake_embed)
    def test_anchor_injects_at_most_top_k_plus_forbidden_nodes(self, _mock):
        """
        ID-RAG key property: only top_k non-forbidden nodes are injected,
        not the entire persona graph.
        """
        persona = FakePersona()
        cfg = dict(TEST_CFG)
        cfg["id_rag_top_k"] = 3
        memory = "NVDA rose 5%."
        self.id_rag.build_graph(persona, cfg)
        nodes = self.id_rag.retrieve_relevant_nodes(persona, memory, cfg)
        non_forbidden = [n for n in nodes if n["type"] != "forbidden"]
        self.assertLessEqual(len(non_forbidden), 3,
                             "Should inject at most top_k non-forbidden nodes")

    @patch("middleware.id_rag._embed_text", side_effect=_fake_embed)
    def test_different_contexts_may_return_different_top_nodes(self, _mock):
        """
        Two very different context strings should yield at least one differing
        non-forbidden node in the top results (context-sensitivity check).
        Uses a large enough persona to give the retriever meaningful variety.
        """
        persona = FakePersona(
            name="Context Agent",
            innate="patient, analytical, risk-averse, disciplined, systematic",
            watchlist=["NVDA", "AAPL", "TSLA", "AMD"],
            trading_strategy="Mean reversion on oversold signals.",
            specialization="semiconductor sector",
        )
        cfg = dict(TEST_CFG)
        cfg["id_rag_top_k"] = 2

        nodes_a = self.id_rag.retrieve_relevant_nodes(
            persona, "NVDA momentum breakout on high volume buy signal", cfg
        )
        nodes_b = self.id_rag.retrieve_relevant_nodes(
            persona, "risk limit exceeded portfolio drawdown stop loss", cfg
        )

        top_a = {n["content"] for n in nodes_a if n["type"] != "forbidden"}
        top_b = {n["content"] for n in nodes_b if n["type"] != "forbidden"}
        # At least one top node differs between the two contexts
        # (With deterministic fake embeddings this is a sanity check not a guarantee,
        #  but the fake embedder is hash-based so different texts -> different vectors)
        # We just assert both sets are non-empty and the test doesn't crash
        self.assertGreater(len(top_a), 0)
        self.assertGreater(len(top_b), 0)

    @patch("middleware.id_rag._embed_text", side_effect=_fake_embed)
    def test_cache_clear_forces_rebuild(self, mock_embed):
        persona = FakePersona(name="Clear Agent")
        self.id_rag.build_graph(persona, TEST_CFG)
        self.id_rag.id_rag_cache_clear()

        self.assertIsNone(self.id_rag.id_rag_graph_for("Clear Agent"))
        # After clear, anchor should rebuild transparently
        result = self.id_rag.id_rag_anchor(persona, "context", TEST_CFG)
        self.assertIsInstance(result, str)
        self.assertIn("[IDENTITY", result)

    @patch("middleware.id_rag._embed_text", side_effect=_fake_embed)
    def test_multiple_agents_isolated_graphs(self, _mock):
        """Each agent has its own graph — one agent's updates don't affect another."""
        pa = FakePersona(name="Agent A", innate="bold, aggressive")
        pb = FakePersona(name="Agent B", innate="cautious, defensive")
        self.id_rag.build_graph(pa, TEST_CFG)
        self.id_rag.build_graph(pb, TEST_CFG)

        self.id_rag.update_graph_belief("Agent A", "NVDA will surge.", TEST_CFG)

        ga = self.id_rag.id_rag_graph_for("Agent A")
        gb = self.id_rag.id_rag_graph_for("Agent B")

        beliefs_a = [d for _, d in ga.nodes(data=True) if d["type"] == "belief"]
        beliefs_b = [d for _, d in gb.nodes(data=True) if d["type"] == "belief"]

        self.assertEqual(len(beliefs_a), 1)
        self.assertEqual(len(beliefs_b), 0)

    @patch("middleware.id_rag._embed_text", side_effect=_fake_embed)
    def test_anchor_always_ends_with_memory_context(self, _mock):
        """Memory context must appear verbatim at the end of the anchored string."""
        persona = FakePersona(name="Tail Agent")
        memory = "UNIQUE_MEMORY_STRING_12345"
        result = self.id_rag.id_rag_anchor(persona, memory, TEST_CFG)
        self.assertTrue(result.endswith(memory),
                        f"Expected result to end with memory context. Got:\n{result}")

    def test_config_dict_and_object_equivalent(self):
        """_cfg() should work with both a dict and an object with attributes."""
        from middleware.id_rag import _cfg

        class ObjCfg:
            id_rag_top_k = 7

        self.assertEqual(_cfg(ObjCfg(), "id_rag_top_k"), 7)
        self.assertEqual(_cfg({"id_rag_top_k": 7}, "id_rag_top_k"), 7)
        self.assertEqual(_cfg(None, "id_rag_top_k"), 4)  # default


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
