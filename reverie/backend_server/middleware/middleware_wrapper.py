"""
middleware/middleware_wrapper.py

One entry point for the middleware pipeline, and the home of the ablation
switch.

WHY THIS EXISTS
The three live steps, memory compression then ID-RAG identity retrieval then
the state-derived legal-action filter, used to be called separately by two
harnesses. Each harness had to remember to call all of them, in the right
order, with the right arguments. That is how run 06's defect 2 happened:
paired_eval.py omitted the ID-RAG graph updates entirely, so its middleware arm
anchored on a frozen bootstrap graph and produced 175 trade attempts at ~2,036
context chars where the runner produced 50 at ~800, diverging at the first
decision. The same class of fault recurred in the order-rejection memory, which
the two harnesses wrote differently until build_order_feedback() was shared.

Composing the steps in one place removes the opportunity. Both harnesses call
prepare() then decide(), and tests/test_wrapper_ablations.py fails if either
harness goes back to calling the steps directly.

WHAT THIS IS NOT
This file does not implement any pipeline logic of its own. It composes the
existing functions and nothing else: no new validation, no new compression, no
reordering. A refactor that changes behaviour while claiming not to is worse
than no refactor, so the default path here is a straight transcription of what
build_memory_context() and make_trading_decision() did before, and
tests/golden/ holds it to that byte for byte.

THE BASELINE ARM IS NOT ROUTED THROUGH HERE
The baseline is the control. It keeps its own untouched code path in
trading_reverie.make_trading_decision(). An arm built here out of zero
components would resemble the baseline but is not it, and conflating the two
would make the comparison circular.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from middleware.memory_compression import compress_memories
from middleware.id_rag import id_rag_anchor
from middleware.action_filtering import (
    run_action_filtering_step,
    get_legal_actions,
)

# The three components the wrapper can compose. Named here so a typo in a
# config is caught at construction rather than silently disabling a stage: an
# arm that quietly ran without compression would be reported as the full
# middleware and would corrupt the ablation table.
COMPONENT_COMPRESSION = "compression"
COMPONENT_IDRAG = "idrag"
COMPONENT_FILTER = "filter"

ALL_COMPONENTS = frozenset({
    COMPONENT_COMPRESSION,
    COMPONENT_IDRAG,
    COMPONENT_FILTER,
})


@dataclass(frozen=True)
class WrapperConfig:
    """
    Which components run.

    The default is all three, which reproduces the shipped middleware arm
    exactly. Any other value is a RESEARCH ARM, not the middleware: a mixed arm
    tells you what one stage contributes, and must never be reported as though
    it were the system under test.

    Per-component settings are deliberately absent. Each stage already reads
    MiddlewareConfig for its own knobs, and duplicating them here would create
    a second place for them to disagree, which is the exact failure this file
    exists to remove.
    """

    components: frozenset = ALL_COMPONENTS

    def __post_init__(self):
        """Reject unknown component names instead of ignoring them."""
        unknown = set(self.components) - set(ALL_COMPONENTS)
        if unknown:
            raise ValueError(
                f"unknown middleware component(s): {sorted(unknown)}. "
                f"Known components: {sorted(ALL_COMPONENTS)}")

    @property
    def is_full_middleware(self) -> bool:
        """True only for the shipped configuration, all three components."""
        return frozenset(self.components) == ALL_COMPONENTS


@dataclass
class PreparedContext:
    """
    Everything the decision call needs, after the input-side steps have run.

    Carries persona and market so that decide() can keep the short signature
    decide(prepared, llm_call): the alternative is making every caller pass the
    same four objects twice, which invites them to pass different ones.

    legal_actions is None when the filter component is disabled, which is how
    decide() knows there is no menu to validate against. It is enumerated here
    rather than inside decide() so an ablation arm can be inspected without
    running a model call.
    """

    persona: object
    market: object
    memory_context: str
    compression_stats: Dict[str, object]
    legal_actions: Optional[List[object]] = None


@dataclass
class Decision:
    """
    The outcome of one decision call.

    `decision` is the accepted action, after any correction.
    `stats` is the untouched stats dict from the underlying step, which carries
    the model's request as it arrived, before correction. Those two must travel
    together: every hallucination in the middleware arm is already gone from
    `decision` by construction, so a metric computed from `decision` alone can
    only ever report zero. That is the defect CHANGES.md section 1.1 is about,
    and keeping the raw request attached is what prevents rediscovering it.
    """

    decision: Dict[str, object]
    stats: Dict[str, object]
    compression_stats: Dict[str, object] = field(default_factory=dict)

    @property
    def disposition(self) -> str:
        """"success", "retry" or "fallback", as the underlying step reported."""
        return str(self.stats.get("status", "success"))

    @property
    def requested(self) -> Dict[str, object]:
        """What the model asked for before any correction was applied."""
        return dict(self.stats.get("requested") or {})


def raw_memory_context(retrieved: dict) -> tuple:
    """
    Concatenate retrieved memories verbatim, with stats saying nothing shrank.

    Used by the compression-disabled ablation arm, and by the baseline arm in
    trading_reverie.build_memory_context(), which is why it lives at module
    level rather than inside the class: one definition, so the two cannot
    drift.

    The stats dict is fully populated rather than just {"enabled": False}. An
    arm was once observed choosing "analyze" on 100% of decisions, and the
    leading explanation was that an uncompressed block crowds out the
    instruction, which was untestable while the size was not recorded.
    nodes_out equals nodes_in because nothing is dropped.
    """
    raw_nodes = []
    for nodes in (retrieved or {}).values():
        raw_nodes.extend(nodes or [])
    memory_context = "\n".join(n.description for n in raw_nodes)
    stats = {
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
    return memory_context, stats


class MiddlewareWrapper:
    """
    Composes the middleware's input-side steps and its decision step.

    Usage, identical in both harnesses:

        wrapper = MiddlewareWrapper(action_log_path=path)
        prepared = wrapper.prepare(persona, market, retrieved)
        outcome  = wrapper.decide(prepared, llm_call)

    `recorder` is accepted and stored but not yet used. It is the seam for
    WP-B's MetricsRecorder, kept here so that adding the recorder does not
    require touching both harnesses again.
    """

    def __init__(self,
                 config: Optional[WrapperConfig] = None,
                 recorder: object = None,
                 action_log_path: Optional[str] = None):
        """
        Store the configuration. Defaults reproduce the shipped middleware.

        A None config becomes the full three-component arm rather than an error,
        so that existing call sites which pass nothing keep working unchanged.
        """
        self.config = config or WrapperConfig()
        self.recorder = recorder
        self.action_log_path = action_log_path

    def prepare(self, persona, market, retrieved: dict) -> PreparedContext:
        """
        Run the input-side steps and return the context the prompt will carry.

        Order is compression then ID-RAG, which is the order the shipped
        middleware uses and the order the golden files pin. ID-RAG runs second
        because it prepends the identity block to the already-compressed memory
        block; swapping them would put the identity facts inside the budget
        that compression enforces and silently trim them.
        """
        components = self.config.components

        if COMPONENT_COMPRESSION in components:
            memory_context, compression_stats = compress_memories(
                retrieved, persona, market)
        else:
            # Ablation arm: hand over the retrieval output verbatim, so the
            # effect of compression alone can be read off the difference.
            memory_context, compression_stats = raw_memory_context(retrieved)

        if COMPONENT_IDRAG in components:
            memory_context = id_rag_anchor(persona, memory_context)

        legal_actions = None
        if COMPONENT_FILTER in components:
            legal_actions = get_legal_actions(persona, market)

        return PreparedContext(
            persona=persona,
            market=market,
            memory_context=memory_context,
            compression_stats=compression_stats,
            legal_actions=legal_actions,
        )

    def decide(self,
               prepared: PreparedContext,
               llm_call: Callable[[str], str]) -> Decision:
        """
        Make one decision from a prepared context.

        With the filter enabled this delegates to run_action_filtering_step(),
        which builds the prompt from the legal menu, validates the response,
        retries once with the validation error appended, then substitutes a
        legal HOLD. It is delegated to rather than reimplemented so that the
        retry and fallback semantics, and the stats dict the entire
        hallucination metric depends on, are impossible to get subtly wrong
        here.

        With the filter disabled the decision goes through the format-check path
        instead, which parses and normalises the response but offers no legal
        menu, leaving illegality to be caught by the executor. That is the
        "filter only" ablation's complement.
        """
        if COMPONENT_FILTER in self.config.components:
            decision, stats = run_action_filtering_step(
                prepared.persona,
                prepared.market,
                prepared.memory_context,
                llm_call,
                self.action_log_path,
            )
        else:
            # Deferred import: free_form_decision lives in trading_reverie,
            # which imports this package, so importing it at module scope would
            # be circular. paired_eval.py defers the same way.
            from trading_reverie import free_form_decision
            decision, stats = free_form_decision(
                prepared.persona,
                prepared.market,
                prepared.memory_context,
                llm_call,
                self.action_log_path,
            )

        return Decision(
            decision=decision,
            stats=stats,
            compression_stats=prepared.compression_stats,
        )
