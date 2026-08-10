"""
Tests for middleware/memory_compression.py

Covers:
  Unit        — each stage and helper in isolation
  Regression  — contracts that must not break, including the methodological
                guard that the compressor never reads the contradiction oracle

Run from backend_server:
    python tests/test_memory_compression.py
"""

import datetime
import os
import sys
import unittest

BACKEND = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from middleware import memory_compression as mc


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

BASE_TIME    = datetime.datetime(2026, 6, 24, 9, 30)
SEC_PER_STEP = 60
SYMBOLS      = ["NVDA", "AAPL", "TSLA", "AMD", "GOOGL"]


def step_time(step: int) -> datetime.datetime:
    """Wall-clock time of a given simulation step."""
    return BASE_TIME + datetime.timedelta(seconds=step * SEC_PER_STEP)


class FakeNode:
    """Mimics the fields of ConceptNode that the compressor reads."""

    def __init__(self, node_id, description, created_step,
                 poignancy=5, keywords=None, embedding_key=None):
        self.node_id       = node_id
        self.description   = description
        self.created       = step_time(created_step)
        self.last_accessed = self.created
        self.poignancy     = poignancy
        self.keywords      = keywords if keywords is not None else set()
        self.embedding_key = embedding_key or description
        self.type          = "event"


class FakeAMem:
    def __init__(self, embeddings=None):
        self.embeddings = embeddings or {}


class FakePersona:
    def __init__(self, embeddings=None):
        self.a_mem = FakeAMem(embeddings)


class FakeMarket:
    """
    Market double. SCRIPTED_NEWS deliberately raises on access — the
    compressor must never touch the contradiction oracle (see the module
    docstring in memory_compression.py).
    """

    SYMBOLS = SYMBOLS

    def __init__(self, current_step=100):
        self.current_time = step_time(current_step)
        self.sec_per_step = SEC_PER_STEP
        self.current_prices = {s: 100.0 for s in SYMBOLS}

    @property
    def SCRIPTED_NEWS(self):
        raise AssertionError(
            "memory_compression read SCRIPTED_NEWS — that is the evaluator's "
            "contradiction oracle. Using it here invalidates the stale-context "
            "measurement."
        )


def retrieved_from(nodes, focal_points=1):
    """Wrap nodes in the dict shape new_retrieve() returns."""
    if focal_points == 1:
        return {"focal_a": list(nodes)}
    return {f"focal_{i}": list(nodes) for i in range(focal_points)}


# ---------------------------------------------------------------------------
# Unit — helpers
# ---------------------------------------------------------------------------

class TestHelpers(unittest.TestCase):

    def test_symbol_from_lowercase_keywords(self):
        """market_perceive stores tickers lowercased."""
        node = FakeNode(1, "some text", 10, keywords={"nvda", "news"})
        self.assertEqual(mc._node_symbol(node, SYMBOLS), "NVDA")

    def test_symbol_falls_back_to_description(self):
        """Reflection thoughts never pass through _build_keywords()."""
        node = FakeNode(1, "TSLA looks overextended here", 10, keywords=set())
        self.assertEqual(mc._node_symbol(node, SYMBOLS), "TSLA")

    def test_symbol_none_for_general_node(self):
        node = FakeNode(1, "the market feels uncertain", 10, keywords={"mood"})
        self.assertIsNone(mc._node_symbol(node, SYMBOLS))

    def test_age_in_steps(self):
        market = FakeMarket(current_step=100)
        node = FakeNode(1, "old news", 40)
        self.assertAlmostEqual(mc._age_in_steps(node, market), 60.0)

    def test_age_zero_when_market_incomplete(self):
        class Bare:
            current_time = None
            sec_per_step = 0
        self.assertEqual(mc._age_in_steps(FakeNode(1, "x", 5), Bare()), 0.0)

    def test_created_key_handles_missing_timestamp(self):
        """A node without `created` must not break sorting."""
        class NoTime:
            pass
        self.assertEqual(mc._created_key(NoTime()), mc._EPOCH)

    def test_cosine_identical_and_orthogonal(self):
        self.assertAlmostEqual(mc._cosine([1.0, 0.0], [1.0, 0.0]), 1.0)
        self.assertAlmostEqual(mc._cosine([1.0, 0.0], [0.0, 1.0]), 0.0)

    def test_cosine_empty_is_zero(self):
        self.assertEqual(mc._cosine([], [1.0]), 0.0)

    def test_estimate_tokens(self):
        self.assertEqual(mc._estimate_tokens("a" * 40, {"mc_chars_per_token": 4}), 10)

    def test_cfg_reads_dict_object_and_default(self):
        class ObjCfg:
            mc_keep_newest_per_symbol = 9
        self.assertEqual(mc._cfg(ObjCfg(), "mc_keep_newest_per_symbol"), 9)
        self.assertEqual(mc._cfg({"mc_keep_newest_per_symbol": 9},
                                 "mc_keep_newest_per_symbol"), 9)
        self.assertEqual(mc._cfg(None, "mc_keep_newest_per_symbol"), 2)


# ---------------------------------------------------------------------------
# Unit — stage 1: flatten + dedup
# ---------------------------------------------------------------------------

class TestStage1Dedup(unittest.TestCase):

    def test_exact_duplicates_removed_across_focal_points(self):
        """
        new_retrieve scores one node pool against every focal point, so the
        same node comes back once per focal point.
        """
        node = FakeNode(1, "NVDA rises 4%", 70, keywords={"nvda"})
        persona = FakePersona()
        kept, dropped = mc._flatten_and_dedup(retrieved_from([node], 3), persona)
        self.assertEqual(len(kept), 1)
        self.assertEqual(dropped, 2)

    def test_near_duplicates_removed_by_cosine(self):
        a = FakeNode(1, "NVDA climbs 4 percent", 70, embedding_key="a")
        b = FakeNode(2, "NVDA rises 4%",         69, embedding_key="b")
        persona = FakePersona({"a": [1.0, 0.0, 0.0], "b": [0.999, 0.01, 0.0]})
        kept, dropped = mc._flatten_and_dedup(
            retrieved_from([a, b]), persona, {"mc_near_duplicate_threshold": 0.95}
        )
        self.assertEqual(len(kept), 1)
        self.assertEqual(dropped, 1)

    def test_near_duplicate_keeps_newer_node(self):
        older = FakeNode(1, "NVDA up",   10, embedding_key="a")
        newer = FakeNode(2, "NVDA up530", 80, embedding_key="b")
        persona = FakePersona({"a": [1.0, 0.0], "b": [1.0, 0.0]})
        kept, _ = mc._flatten_and_dedup(
            retrieved_from([older, newer]), persona,
            {"mc_near_duplicate_threshold": 0.95}
        )
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0].node_id, 2, "newer node should survive")

    def test_distinct_nodes_all_kept(self):
        a = FakeNode(1, "NVDA news", 10, embedding_key="a")
        b = FakeNode(2, "AAPL news", 20, embedding_key="b")
        persona = FakePersona({"a": [1.0, 0.0], "b": [0.0, 1.0]})
        kept, dropped = mc._flatten_and_dedup(retrieved_from([a, b]), persona)
        self.assertEqual(len(kept), 2)
        self.assertEqual(dropped, 0)

    def test_missing_embeddings_does_not_drop(self):
        """No embedding available -> cannot judge similarity -> keep both."""
        a = FakeNode(1, "NVDA news", 10)
        b = FakeNode(2, "NVDA news copy", 20)
        kept, _ = mc._flatten_and_dedup(retrieved_from([a, b]), FakePersona())
        self.assertEqual(len(kept), 2)


# ---------------------------------------------------------------------------
# Unit — stage 2: symbol-scoped recency pruning
# ---------------------------------------------------------------------------

class TestStage2SymbolPruning(unittest.TestCase):

    def test_keeps_only_newest_k_per_symbol(self):
        nodes = [
            FakeNode(1, "NVDA a", 5,  keywords={"nvda"}),
            FakeNode(2, "NVDA b", 30, keywords={"nvda"}),
            FakeNode(3, "NVDA c", 70, keywords={"nvda"}),
        ]
        kept, dropped, _ = mc._prune_by_symbol_recency(
            nodes, SYMBOLS, {"mc_keep_newest_per_symbol": 2}
        )
        self.assertEqual(len(kept), 2)
        self.assertEqual(dropped, 1)
        self.assertEqual({n.node_id for n in kept}, {2, 3})

    def test_each_symbol_pruned_independently(self):
        nodes = [
            FakeNode(1, "NVDA a", 5,  keywords={"nvda"}),
            FakeNode(2, "NVDA b", 70, keywords={"nvda"}),
            FakeNode(3, "AAPL a", 10, keywords={"aapl"}),
            FakeNode(4, "AAPL b", 60, keywords={"aapl"}),
        ]
        kept, dropped, _ = mc._prune_by_symbol_recency(
            nodes, SYMBOLS, {"mc_keep_newest_per_symbol": 1}
        )
        self.assertEqual(len(kept), 2)
        self.assertEqual(dropped, 2)
        self.assertEqual({n.node_id for n in kept}, {2, 4})

    def test_newer_counts_rank_within_symbol(self):
        nodes = [
            FakeNode(1, "NVDA oldest", 5,  keywords={"nvda"}),
            FakeNode(2, "NVDA mid",    30, keywords={"nvda"}),
            FakeNode(3, "NVDA newest", 70, keywords={"nvda"}),
        ]
        _kept, _dropped, newer = mc._prune_by_symbol_recency(nodes, SYMBOLS)
        self.assertEqual(newer[3], 0)   # newest has none newer
        self.assertEqual(newer[2], 1)
        self.assertEqual(newer[1], 2)   # oldest superseded twice

    def test_general_nodes_form_their_own_group(self):
        nodes = [
            FakeNode(1, "market mood a", 5,  keywords={"mood"}),
            FakeNode(2, "market mood b", 30, keywords={"mood"}),
            FakeNode(3, "NVDA news",     70, keywords={"nvda"}),
        ]
        kept, _dropped, _ = mc._prune_by_symbol_recency(
            nodes, SYMBOLS, {"mc_keep_newest_per_symbol": 1}
        )
        # one survivor from the general group, one from NVDA
        self.assertEqual(len(kept), 2)
        self.assertIn(3, {n.node_id for n in kept})


# ---------------------------------------------------------------------------
# Unit — stage 3: supersession annotation
# ---------------------------------------------------------------------------

class TestStage3Annotation(unittest.TestCase):

    def setUp(self):
        self.market = FakeMarket(current_step=100)

    def test_old_and_superseded_gets_full_marker(self):
        node = FakeNode(1, "NVDA earnings beat", 5, keywords={"nvda"})
        line, marked = mc._annotate(node, self.market, SYMBOLS, {1: 3})
        self.assertTrue(marked)
        self.assertIn("95 steps ago", line)
        self.assertIn("3 newer NVDA events since", line)

    def test_old_but_not_superseded_gets_age_only(self):
        node = FakeNode(1, "NVDA earnings beat", 5, keywords={"nvda"})
        line, marked = mc._annotate(node, self.market, SYMBOLS, {1: 0})
        self.assertFalse(marked)
        self.assertIn("95 steps ago", line)
        self.assertNotIn("newer", line)

    def test_recent_node_gets_plain_line(self):
        node = FakeNode(1, "NVDA rises 4%", 95, keywords={"nvda"})
        line, marked = mc._annotate(node, self.market, SYMBOLS, {1: 0})
        self.assertFalse(marked)
        self.assertEqual(line, "- NVDA rises 4%")

    def test_singular_event_wording(self):
        node = FakeNode(1, "NVDA old", 5, keywords={"nvda"})
        line, _ = mc._annotate(node, self.market, SYMBOLS, {1: 1})
        self.assertIn("1 newer NVDA event since", line)
        self.assertNotIn("events since", line)

    def test_annotation_can_be_disabled(self):
        node = FakeNode(1, "NVDA old", 5, keywords={"nvda"})
        line, marked = mc._annotate(
            node, self.market, SYMBOLS, {1: 3}, {"mc_annotate_superseded": False}
        )
        self.assertFalse(marked)
        self.assertEqual(line, "- NVDA old")


# ---------------------------------------------------------------------------
# Unit — stage 4: token budget
# ---------------------------------------------------------------------------

class TestStage4Budget(unittest.TestCase):

    def setUp(self):
        self.market = FakeMarket(current_step=100)

    def test_drops_until_under_budget(self):
        items = [(f"- line {i} padded out", FakeNode(i, "x", 90), False)
                 for i in range(10)]
        cfg = {"mc_max_memory_tokens": 10, "mc_chars_per_token": 4}
        kept, dropped = mc._enforce_budget(items, self.market, cfg)
        text = "\n".join(l for l, _n, _a in kept)
        self.assertLessEqual(mc._estimate_tokens(text, cfg), 10)
        self.assertEqual(len(kept) + dropped, 10)

    def test_evicts_lowest_value_first(self):
        """High poignancy + recent should outlive low poignancy + old."""
        keep = FakeNode(1, "important", 99, poignancy=9)
        drop = FakeNode(2, "trivial",   5,  poignancy=1)
        items = [("- important note here", keep, False),
                 ("- trivial note here",   drop, False)]
        cfg = {"mc_max_memory_tokens": 6, "mc_chars_per_token": 4}
        kept, dropped = mc._enforce_budget(items, self.market, cfg)
        self.assertEqual(dropped, 1)
        self.assertEqual(kept[0][1].node_id, 1)

    def test_zero_budget_disables_enforcement(self):
        items = [("- a", FakeNode(1, "a", 90), False)]
        kept, dropped = mc._enforce_budget(items, self.market,
                                           {"mc_max_memory_tokens": 0})
        self.assertEqual(dropped, 0)
        self.assertEqual(len(kept), 1)


# ---------------------------------------------------------------------------
# Unit — public entry point
# ---------------------------------------------------------------------------

class TestCompressMemories(unittest.TestCase):

    def setUp(self):
        self.market  = FakeMarket(current_step=100)
        self.persona = FakePersona()

    def test_returns_string_and_stats(self):
        nodes = [FakeNode(1, "NVDA rises 4%", 95, keywords={"nvda"})]
        text, stats = mc.compress_memories(
            retrieved_from(nodes), self.persona, self.market
        )
        self.assertIsInstance(text, str)
        self.assertIsInstance(stats, dict)

    def test_empty_retrieval_message(self):
        text, stats = mc.compress_memories({}, self.persona, self.market)
        self.assertEqual(text, "No relevant memories.")
        self.assertEqual(stats["nodes_in"], 0)
        self.assertEqual(stats["nodes_out"], 0)

    def test_disabled_passthrough_keeps_every_node(self):
        nodes = [FakeNode(i, f"NVDA event {i}", i * 10, keywords={"nvda"})
                 for i in range(6)]
        text, stats = mc.compress_memories(
            retrieved_from(nodes), self.persona, self.market,
            {"memory_compression_enabled": False},
        )
        self.assertFalse(stats["enabled"])
        self.assertEqual(stats["nodes_out"], stats["nodes_in"])
        self.assertEqual(len(text.strip().split("\n")), 6)

    def test_stats_keys_present(self):
        """_generate_report reads these keys by name."""
        nodes = [FakeNode(1, "NVDA rises", 95, keywords={"nvda"})]
        _text, stats = mc.compress_memories(
            retrieved_from(nodes), self.persona, self.market
        )
        for key in ("enabled", "nodes_in", "nodes_out", "dropped_duplicate",
                    "dropped_stale", "dropped_budget", "annotated_stale",
                    "compression_ratio", "chars_in", "chars_out"):
            self.assertIn(key, stats)

    def test_malformed_nodes_fall_back_without_crashing(self):
        class Broken:
            description = "broken node"
        text, stats = mc.compress_memories(
            retrieved_from([Broken()]), self.persona, self.market
        )
        self.assertIsInstance(text, str)
        self.assertGreaterEqual(stats["nodes_in"], 1)


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------

class TestRegressions(unittest.TestCase):
    """Contracts that must hold across any future change."""

    def setUp(self):
        self.market  = FakeMarket(current_step=100)
        self.persona = FakePersona()

    def test_never_reads_scripted_news_oracle(self):
        """
        THE methodological guard. FakeMarket.SCRIPTED_NEWS raises on access.
        If the compressor ever reads it, the compressor and the evaluator
        share an oracle and the stale-context reduction becomes meaningless.
        """
        nodes = [
            FakeNode(1, "NVDA earnings beat, rises 8%",   5,  keywords={"nvda"}),
            FakeNode(2, "NVDA supply bottleneck, -6%",    30, keywords={"nvda"}),
            FakeNode(3, "NVDA new contract, rises 4%",    70, keywords={"nvda"}),
        ]
        # Raises AssertionError from the property if the oracle is touched.
        text, stats = mc.compress_memories(
            retrieved_from(nodes, 3), self.persona, self.market
        )
        self.assertIsInstance(text, str)
        self.assertTrue(stats["enabled"])

    def test_superseded_headline_is_dropped(self):
        """
        The core mechanism: the step-5 headline disappears because newer NVDA
        events exist — established on recency alone, with no oracle.
        """
        nodes = [
            FakeNode(1, "NVDA earnings beat, rises 8%",   5,  keywords={"nvda"}),
            FakeNode(2, "NVDA supply bottleneck, -6%",    30, keywords={"nvda"}),
            FakeNode(3, "NVDA new contract, rises 4%",    70, keywords={"nvda"}),
        ]
        text, _stats = mc.compress_memories(
            retrieved_from(nodes), self.persona, self.market,
            {"mc_keep_newest_per_symbol": 2},
        )
        self.assertNotIn("earnings beat", text)
        self.assertIn("new contract", text)

    def test_cross_focal_point_duplication_collapsed(self):
        """Three focal points returning the same node must yield one line."""
        node = FakeNode(1, "NVDA rises 4%", 95, keywords={"nvda"})
        text, stats = mc.compress_memories(
            retrieved_from([node], 3), self.persona, self.market
        )
        self.assertEqual(stats["nodes_in"], 3)
        self.assertEqual(stats["nodes_out"], 1)
        self.assertEqual(text.count("NVDA rises 4%"), 1)

    def test_output_never_exceeds_token_budget(self):
        nodes = [FakeNode(i, f"NVDA event number {i} with padding text",
                          50 + i, keywords={"nvda"})
                 for i in range(40)]
        cfg = {"mc_keep_newest_per_symbol": 40,
               "mc_max_memory_tokens": 20,
               "mc_chars_per_token": 4}
        text, _stats = mc.compress_memories(
            retrieved_from(nodes), self.persona, self.market, cfg
        )
        self.assertLessEqual(mc._estimate_tokens(text, cfg), 20)

    def test_compression_ratio_never_exceeds_one(self):
        nodes = [FakeNode(i, f"NVDA event {i}", i, keywords={"nvda"})
                 for i in range(20)]
        _text, stats = mc.compress_memories(
            retrieved_from(nodes, 2), self.persona, self.market
        )
        self.assertLessEqual(stats["compression_ratio"], 1.0)

    def test_ablation_arm_strictly_larger_than_compressed(self):
        """
        Disabled must produce a genuine uncompressed baseline, otherwise the
        with/without comparison measures nothing.
        """
        nodes = [FakeNode(i, f"NVDA event number {i}", i * 5, keywords={"nvda"})
                 for i in range(10)]
        retrieved = retrieved_from(nodes, 2)

        off, off_stats = mc.compress_memories(
            retrieved, self.persona, self.market,
            {"memory_compression_enabled": False},
        )
        on, on_stats = mc.compress_memories(
            retrieved, self.persona, self.market,
            {"memory_compression_enabled": True, "mc_keep_newest_per_symbol": 2},
        )
        self.assertGreater(off_stats["nodes_out"], on_stats["nodes_out"])
        self.assertGreater(len(off), len(on))

    def test_multi_symbol_portfolio_keeps_each_ticker_represented(self):
        """Pruning is per-symbol, so no ticker is silently starved."""
        nodes = []
        for i, sym in enumerate(["NVDA", "AAPL", "TSLA"]):
            for j in range(4):
                nodes.append(FakeNode(
                    f"{sym}{j}", f"{sym} moves on day {j}",
                    10 + i * 4 + j, keywords={sym.lower()},
                ))
        text, _stats = mc.compress_memories(
            retrieved_from(nodes), self.persona, self.market,
            {"mc_keep_newest_per_symbol": 1, "mc_max_memory_tokens": 0},
        )
        for sym in ["NVDA", "AAPL", "TSLA"]:
            self.assertIn(sym, text, f"{sym} was dropped entirely")

    def test_result_is_deterministic(self):
        """Same input must give byte-identical output — no LLM, no randomness."""
        nodes = [FakeNode(i, f"NVDA event {i}", i * 7, keywords={"nvda"})
                 for i in range(12)]
        retrieved = retrieved_from(nodes, 2)
        first,  _  = mc.compress_memories(retrieved, self.persona, self.market)
        second, _  = mc.compress_memories(retrieved, self.persona, self.market)
        self.assertEqual(first, second)

    def test_does_not_mutate_retrieved_input(self):
        """trading_reverie reuses `retrieved`; compression must not corrupt it."""
        nodes = [FakeNode(i, f"NVDA event {i}", i * 5, keywords={"nvda"})
                 for i in range(6)]
        retrieved = retrieved_from(nodes, 2)
        before = {k: list(v) for k, v in retrieved.items()}
        mc.compress_memories(retrieved, self.persona, self.market)
        self.assertEqual(list(retrieved.keys()), list(before.keys()))
        for key in before:
            self.assertEqual(retrieved[key], before[key])


if __name__ == "__main__":
    unittest.main(verbosity=2)
