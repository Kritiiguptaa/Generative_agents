"""
Tests for MiddlewareWrapper: the ablation arms, and the composition guard.

TWO JOBS

1. The ablation arms work and are distinguishable. The paper's "coupled
   interventions" limitation is that the middleware changes context length,
   identity information and the action interface together, so no effect can be
   attributed to one of them. The wrapper's `components` switch is what will
   separate them, and these tests pin that each single-component arm actually
   runs only its own stage.

2. Nothing goes around the wrapper. This is the structural half of the fix:
   build_memory_context and run_action_filtering_step used to be called by both
   harnesses independently, and in run 06 the paired harness assembled a
   DIFFERENT middleware without anything failing. A grep-based guard is exactly
   right for that, because the failure mode is a call reappearing in a harness,
   which is a property of the source and not of one run's behaviour. It
   complements test_harness_parity.py, which checks behaviour.

Run from backend_server:
    python -m pytest tests/test_wrapper_ablations.py
"""

import os
import shutil
import sys
import unittest
from pathlib import Path
from unittest import mock

BACKEND = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from test_golden_pipeline import (  # noqa: E402
    _FakeOllama,
    _storage_root,
)

from middleware.middleware_wrapper import (  # noqa: E402
    ALL_COMPONENTS,
    COMPONENT_COMPRESSION,
    COMPONENT_FILTER,
    COMPONENT_IDRAG,
    MiddlewareWrapper,
    PreparedContext,
    WrapperConfig,
    raw_memory_context,
)

# The marker id_rag.format_retrieved_nodes() opens the identity block with.
# Used to assert presence or absence of the block rather than guessing at its
# length, which varies with which nodes are retrieved.
IDENTITY_MARKER = "[IDENTITY"

ABLATION_SIM = "_ablation_probe"


class WrapperConfigTest(unittest.TestCase):
    """The configuration object itself, which needs no simulation."""

    def test_default_is_the_full_middleware(self):
        """
        An unconfigured wrapper must be the shipped arm.

        This is what lets every existing call site pass nothing and keep its
        behaviour, and it is the property the golden files depend on.
        """
        config = WrapperConfig()
        self.assertEqual(frozenset(config.components), ALL_COMPONENTS)
        self.assertTrue(config.is_full_middleware)

    def test_a_mixed_arm_is_not_the_full_middleware(self):
        """
        is_full_middleware must be False for any research arm.

        Reporting an ablation arm's numbers as the middleware's would be the
        worst available mistake here, so the distinction is explicit rather
        than left to the caller to remember.
        """
        config = WrapperConfig(components=frozenset({COMPONENT_FILTER}))
        self.assertFalse(config.is_full_middleware)

    def test_unknown_component_is_rejected_loudly(self):
        """
        A misspelled component name must raise, not be ignored.

        Silently ignoring it would run an arm with a stage disabled while the
        config still claimed to name it, which would put a wrong row in the
        ablation table with nothing in the logs to reveal it.
        """
        with self.assertRaises(ValueError):
            WrapperConfig(components=frozenset({"compresion"}))
        with self.assertRaises(ValueError):
            WrapperConfig(components=frozenset({COMPONENT_FILTER, "magic"}))


class RawMemoryContextTest(unittest.TestCase):
    """The shared pass-through used by the baseline and by the ablation arm."""

    class _Node:
        """Minimal stand-in for a ConceptNode: only .description is read."""

        def __init__(self, description):
            self.description = description

    def test_it_concatenates_and_reports_no_shrinkage(self):
        """Every node survives, and the stats say compression did not run."""
        retrieved = {
            "focal one": [self._Node("alpha"), self._Node("beta")],
            "focal two": [self._Node("gamma")],
        }
        context, stats = raw_memory_context(retrieved)

        self.assertEqual(context, "alpha\nbeta\ngamma")
        self.assertFalse(stats["enabled"])
        self.assertEqual(stats["nodes_in"], 3)
        self.assertEqual(stats["nodes_out"], 3)
        self.assertEqual(stats["compression_ratio"], 1.0)
        # Size is recorded even though nothing was compressed: an arm was once
        # observed choosing "analyze" on every decision and the size was the
        # thing needed to test why.
        self.assertEqual(stats["chars_out"], len(context))

    def test_empty_retrieval_is_not_an_error(self):
        """No memories yet, as on the first step, must produce empty output."""
        context, stats = raw_memory_context({})
        self.assertEqual(context, "")
        self.assertEqual(stats["nodes_in"], 0)

    def test_none_entries_are_tolerated(self):
        """A focal point with no nodes must not raise."""
        context, stats = raw_memory_context({"focal": None})
        self.assertEqual(context, "")
        self.assertEqual(stats["nodes_in"], 0)


class AblationArmTest(unittest.TestCase):
    """
    Each arm runs only its own stages, checked against a real persona.

    A real persona and market are used rather than hand-built fakes, because
    the stages read a wide slice of persona state (associative memory for
    dedup, scratch for the legal menu) and a fake that diverged from the real
    shape would make these tests pass while the arms were broken.

    Personas are loaded in TradingReverie.__init__, so no step needs to run.
    """

    @classmethod
    def setUpClass(cls):
        """Boot one simulation under the stub and reuse it for every arm."""
        cls._fake = _FakeOllama()
        import requests as _requests
        from middleware import id_rag as _id_rag

        _id_rag.id_rag_cache_clear()
        cls._patcher = mock.patch.object(_requests, "post",
                                         side_effect=cls._fake.post)
        cls._patcher.start()

        target = Path(_storage_root()) / ABLATION_SIM
        if target.exists():
            shutil.rmtree(target)

        from trading_reverie import TradingReverie
        cls.sim = TradingReverie(
            sim_code=ABLATION_SIM,
            fork_sim_code="base_trading_3agent",
            use_middleware=True,
            fresh=True,
            seed=42,
            temperature=0.7,
        )
        cls.persona = next(iter(cls.sim.personas.values()))
        cls.market = cls.sim.market

    @classmethod
    def tearDownClass(cls):
        """Stop the patch and remove the throwaway simulation folder."""
        cls._patcher.stop()
        target = Path(_storage_root()) / ABLATION_SIM
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)

    def _retrieved(self):
        """
        A retrieval-shaped dict built from the persona's real memory nodes.

        Uses whatever the bootstrap memory holds rather than synthesising
        nodes, so the compression stage sees the node shape it sees in a run.
        """
        nodes = list(getattr(self.persona.a_mem, "seq_event", []))[:6]
        if not nodes:
            nodes = list(getattr(self.persona.a_mem, "seq_thought", []))[:6]
        return {"What should this agent trade right now?": nodes}

    def _prepare(self, components):
        """Run prepare() for one component set and return the result."""
        wrapper = MiddlewareWrapper(
            config=WrapperConfig(components=frozenset(components)))
        return wrapper.prepare(self.persona, self.market, self._retrieved())

    def test_filter_disabled_enumerates_no_menu(self):
        """
        An arm without the filter must not build a legal-action menu.

        This is the acceptance criterion for the filter ablation. The menu is
        not merely unused when the filter is off: enumerating it would mean the
        maximum legal quantity still reached the prompt, so the arm would carry
        the filter's main informational effect while claiming not to.
        """
        for components in ({COMPONENT_COMPRESSION},
                           {COMPONENT_IDRAG},
                           {COMPONENT_COMPRESSION, COMPONENT_IDRAG}):
            with self.subTest(components=sorted(components)):
                prepared = self._prepare(components)
                self.assertIsNone(prepared.legal_actions)

    def test_filter_enabled_enumerates_a_menu(self):
        """
        An arm with the filter must build one, containing at least HOLD.

        HOLD is legal by construction, so an empty menu would mean the
        enumeration failed rather than that no action was available.
        """
        prepared = self._prepare({COMPONENT_FILTER})
        self.assertIsNotNone(prepared.legal_actions)
        self.assertTrue(len(prepared.legal_actions) >= 1)
        self.assertIn("HOLD", {a.type for a in prepared.legal_actions})

    def test_compression_disabled_passes_memories_through(self):
        """Without compression the stats must report that it did not run."""
        prepared = self._prepare({COMPONENT_FILTER})
        self.assertFalse(prepared.compression_stats["enabled"])
        self.assertEqual(prepared.compression_stats["nodes_in"],
                         prepared.compression_stats["nodes_out"])

    def test_compression_enabled_runs_the_stage(self):
        """With compression the stats must come from the real stage."""
        prepared = self._prepare({COMPONENT_COMPRESSION})
        self.assertTrue(prepared.compression_stats["enabled"])

    def test_idrag_disabled_omits_the_identity_block(self):
        """No identity block when ID-RAG is off."""
        prepared = self._prepare({COMPONENT_COMPRESSION})
        self.assertNotIn(IDENTITY_MARKER, prepared.memory_context)

    def test_idrag_enabled_prepends_the_identity_block(self):
        """
        The identity block is present, and it comes first.

        Order matters: ID-RAG runs after compression so the identity facts are
        prepended to the already-budgeted memory block. If it ran first, the
        identity block would be inside the budget compression enforces and
        could be trimmed away.
        """
        prepared = self._prepare({COMPONENT_COMPRESSION, COMPONENT_IDRAG})
        self.assertIn(IDENTITY_MARKER, prepared.memory_context)
        self.assertTrue(prepared.memory_context.lstrip().startswith(
            IDENTITY_MARKER))

    def test_every_single_component_arm_prepares(self):
        """
        Each of the three single-component arms runs without error.

        These are the three arms the ablation sweep will add, so a failure here
        means the sweep cannot be run at all.
        """
        for component in sorted(ALL_COMPONENTS):
            with self.subTest(component=component):
                prepared = self._prepare({component})
                self.assertIsInstance(prepared, PreparedContext)
                self.assertIsInstance(prepared.memory_context, str)

    def test_the_full_arm_runs_all_three_stages(self):
        """The default arm compresses, anchors, and enumerates a menu."""
        prepared = self._prepare(ALL_COMPONENTS)
        self.assertTrue(prepared.compression_stats["enabled"])
        self.assertIn(IDENTITY_MARKER, prepared.memory_context)
        self.assertIsNotNone(prepared.legal_actions)


class CompositionGuardTest(unittest.TestCase):
    """
    Neither harness may compose the pipeline itself.

    Source-level on purpose. The failure being prevented is a pipeline call
    reappearing inside a harness, which is a property of the code rather than
    of any one run, and which a behavioural test only catches once the two
    harnesses have already diverged. test_harness_parity.py covers the
    behavioural side.
    """

    # The functions that make up the pipeline. A harness calling any of these
    # directly is assembling its own middleware, which is what let run 06's
    # paired harness run a different configuration from the runner.
    COMPOSITION_CALLS = (
        "compress_memories",
        "id_rag_anchor",
        "run_action_filtering_step",
    )

    @staticmethod
    def _source(filename):
        """Read one backend source file."""
        with open(os.path.join(BACKEND, filename), encoding="utf-8") as handle:
            return handle.read()

    @staticmethod
    def _referenced_names(source):
        """
        Names this source actually calls or imports, from its parse tree.

        Parsed rather than grepped. These function names appear in several
        explanatory comments and in trading_reverie's module docstring, and
        those mentions are wanted: they are what tell the next reader why the
        call is not there. Text matching cannot tell prose from code, so it
        reported the docstring as a violation. The AST sees only real code,
        which makes the guard exact instead of approximate.

        Imports are collected as well as calls, because importing one of these
        into a harness is the step before calling it, and leaving the import in
        place invites the call back.
        """
        import ast

        names = set()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name):
                    names.add(func.id)
                elif isinstance(func, ast.Attribute):
                    names.add(func.attr)
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    names.add(alias.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    names.add(alias.name.split(".")[-1])
        return names

    def _assert_harness_does_not_compose(self, filename):
        """Fail if a harness calls or imports any pipeline function directly."""
        names = self._referenced_names(self._source(filename))
        for call in self.COMPOSITION_CALLS:
            with self.subTest(call=call):
                self.assertNotIn(
                    call, names,
                    f"{filename} calls or imports {call}() directly. The "
                    f"pipeline is composed by MiddlewareWrapper; a harness "
                    f"that assembles it itself can run a different middleware "
                    f"from the other harness, which is run 06's defect 2. Use "
                    f"wrapper.prepare() and wrapper.decide() instead.",
                )

    def test_runner_does_not_compose_the_pipeline(self):
        """trading_reverie.py must go through the wrapper."""
        self._assert_harness_does_not_compose("trading_reverie.py")

    def test_paired_harness_does_not_compose_the_pipeline(self):
        """paired_eval.py must go through the wrapper."""
        self._assert_harness_does_not_compose("paired_eval.py")

    def test_the_wrapper_itself_does_compose_the_pipeline(self):
        """
        The guard above must be testing something.

        If the pipeline functions were renamed or moved, the two tests above
        would pass trivially while nothing composed anything. This asserts the
        names still exist where the wrapper calls them.
        """
        names = self._referenced_names(
            self._source(os.path.join("middleware", "middleware_wrapper.py")))
        for call in self.COMPOSITION_CALLS:
            with self.subTest(call=call):
                self.assertIn(call, names)

    def test_middleware_package_exports_a_real_wrapper(self):
        """
        `from middleware import MiddlewareWrapper` must not yield None.

        It used to: the import sat in a try/except ImportError with a
        None fallback while middleware_wrapper.py was an empty file, so a
        wiring mistake surfaced far away as "NoneType is not callable", or
        never surfaced at all.
        """
        import middleware
        self.assertIsNotNone(middleware.MiddlewareWrapper)
        self.assertIs(middleware.MiddlewareWrapper, MiddlewareWrapper)


if __name__ == "__main__":
    unittest.main()
