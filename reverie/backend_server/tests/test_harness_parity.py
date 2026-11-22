"""
Behavioural parity between the two harnesses that run the middleware arm.

WHY THIS EXISTS
trading_reverie.TradingReverie and paired_eval.PairedEvaluation are both
supposed to run the same middleware. In run 06 they did not: PairedEvaluation
overrode _step_agent and omitted the ID-RAG graph updates, so its middleware
arm anchored on a frozen bootstrap identity graph. Same seed, same advancing
arm, same step count produced 175 trade attempts at ~2,036 context chars in the
paired harness against 50 at ~800 in the runner, diverging at the very first
decision. The paired numbers in that sweep describe a middleware configuration
that the independent arm never ran.

That defect is fixed. What guards it today is
tests/test_run06_defects.py::test_both_harnesses_update_the_same_graph_functions,
which greps both source files for the names of the graph-update functions.

That guard has a real hole. It is a substring check, so it passes when:
  - the calls are present but in the wrong ORDER relative to retrieval, which
    is precisely how the original defect behaved, since
    update_graph_current_situation() runs before retrieval and that is why the
    two harnesses diverged at step 1 rather than drifting apart gradually;
  - the calls are present but passed different arguments;
  - the calls are present but sit in a branch that never executes.

This file closes that hole by comparing behaviour instead of source text: both
harnesses are run over identical inputs with an identical seed, and their
middleware-arm decisions and context sizes must match exactly.

DETERMINISM
Unlike the golden harness, scripted responses here are keyed by a hash of the
prompt, not by call order. PairedEvaluation decides twice per agent per step,
once for each arm, so an order-keyed script would hand the two harnesses
different responses for a reason that has nothing to do with the defect being
tested. Keying on the prompt makes the stub order-independent: identical
prompts get identical responses, and any genuine divergence in what the two
harnesses put in front of the model shows up as a different decision.

Run from backend_server:
    python -m pytest tests/test_harness_parity.py
"""

import hashlib
import os
import shutil
import sys
import unittest
from pathlib import Path
from unittest import mock

BACKEND = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

# tests/ is not a package (no __init__.py), so the sibling module is imported
# by plain name with this directory on the path, the same way the other test
# modules reach backend_server.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from test_golden_pipeline import (  # noqa: E402
    _DECISION_SCRIPT,
    _FakeHTTPResponse,
    _classify_prompt,
    _deterministic_embedding,
    _interaction_response,
    _plan_response,
    _storage_root,
)

# Shorter than the golden run. The original defect was visible at step 1, so a
# few steps is enough to catch it, and the paired harness costs two decisions
# per agent per step.
PARITY_STEPS = 6
PARITY_SEED = 42
PARITY_TEMPERATURE = 0.7


def _script_for_prompt(prompt: str) -> str:
    """
    Pick a scripted model response from a hash of the prompt.

    Order-independent on purpose, see the module docstring: the paired harness
    makes twice as many decision calls as the runner, so anything keyed on call
    count would diverge for a reason unrelated to the defect under test.

    Hashing the prompt also keeps the retry path meaningful, because
    run_action_filtering_step appends "Validation error: ... Try again." to the
    prompt before retrying. The retry therefore hashes differently and can
    succeed, rather than being handed the same rejected response forever.
    """
    digest = hashlib.sha256(prompt.encode("utf-8", "replace")).digest()
    return _DECISION_SCRIPT[digest[0] % len(_DECISION_SCRIPT)]


class _PromptKeyedOllama:
    """
    Deterministic Ollama stand-in whose answers depend only on the prompt text.

    Holds no counter that could make the answer depend on how many calls came
    before it, which is what lets the same instance serve two harnesses that
    call it a different number of times.
    """

    def __init__(self):
        self.decision_prompts = []

    def post(self, url, json=None, timeout=None, **kwargs):
        """Route one request to its canned response, keyed by content alone."""
        payload = json or {}
        if "/api/embed" in url:
            text = payload.get("prompt") or payload.get("input") or ""
            return _FakeHTTPResponse(
                {"embedding": _deterministic_embedding(str(text))})

        prompt = str(payload.get("prompt", ""))
        kind = _classify_prompt(prompt)

        if kind == "daily_plan":
            body = _plan_response()
        elif kind == "interaction":
            body = _interaction_response()
        elif kind in ("decision_middleware", "decision_baseline"):
            if kind == "decision_middleware":
                self.decision_prompts.append(prompt)
            body = _script_for_prompt(prompt)
        else:
            body = "{}"

        return _FakeHTTPResponse({"response": body, "done_reason": "stop"})


def _middleware_rows(log: list) -> list:
    """
    Extract the middleware arm's decisions from either harness's log.

    The two logs have different shapes. The runner writes one flat entry per
    decision, with the action nested under "decision" and the validator outcome
    under "filter_status". The paired harness writes one entry per matched
    pair, with each arm nested under "middleware" and "baseline" and the
    outcome under that arm's "stats". Normalising both to one shape is what
    makes them comparable at all.

    Context size is deliberately NOT compared here, because neither log holds a
    field the other does. The runner records compression["chars_out"], the
    compressed memory block before the ID-RAG anchor is prepended; the paired
    harness records "context_chars", the length of the whole context after it.
    Comparing one against the other would report a difference on every row even
    when the harnesses agree perfectly. What the two harnesses actually put in
    front of the model is compared instead, from the prompts the stub saw, in
    test_decision_prompts_match_across_harnesses.
    """
    rows = []
    for entry in log:
        if not isinstance(entry, dict):
            continue
        if isinstance(entry.get("middleware"), dict):
            # Paired harness: each arm is a flat dict under its own key, with
            # the action inline rather than nested under "decision".
            arm = entry["middleware"]
            rows.append({
                "agent": entry.get("agent"),
                "action": arm.get("action"),
                "symbol": arm.get("symbol"),
                "quantity": arm.get("quantity"),
                "filter_status": arm.get("filter_status"),
                "requested": arm.get("requested") or {},
                "kind": arm.get("kind"),
                "disposition": arm.get("disposition"),
            })
        else:
            # Runner: action nested under "decision", and the hallucination
            # fields carry a "halluc_" prefix the paired harness does not use.
            decision = entry.get("decision") or {}
            rows.append({
                "agent": entry.get("agent"),
                "action": decision.get("action"),
                "symbol": decision.get("symbol"),
                "quantity": decision.get("quantity"),
                "filter_status": entry.get("filter_status"),
                "requested": entry.get("requested") or {},
                "kind": entry.get("halluc_kind"),
                "disposition": entry.get("halluc_disposition"),
            })
    return rows


def _run_harness(kind: str) -> tuple:
    """
    Run one harness over the stub and return (decision rows, decision prompts).

    The prompts come from the stub rather than from either log, because this is
    the only measurement of "what was put in front of the model" that both
    harnesses can be compared on: their logs record different context fields.
    It is also the most direct possible test of defect 2, which was a context
    difference of ~800 against ~2,036 characters.

    `kind` is "runner" for TradingReverie or "paired" for PairedEvaluation
    advancing with the middleware arm, which is the configuration that must
    match the runner: both then execute the middleware decision and carry the
    same trajectory forward.

    The ID-RAG embedding cache is process-global, so it is cleared before each
    harness. Without that, the second harness would read vectors the first one
    cached and the comparison would be contaminated by run order.
    """
    sim_code = f"_parity_{kind}_s{PARITY_SEED}"
    fake = _PromptKeyedOllama()

    import requests as _requests
    from middleware import id_rag as _id_rag

    _id_rag.id_rag_cache_clear()

    with mock.patch.object(_requests, "post", side_effect=fake.post):
        target = Path(_storage_root()) / sim_code
        if target.exists():
            shutil.rmtree(target)

        if kind == "runner":
            from trading_reverie import TradingReverie
            sim = TradingReverie(
                sim_code=sim_code,
                fork_sim_code="base_trading",
                use_middleware=True,
                fresh=True,
                seed=PARITY_SEED,
                temperature=PARITY_TEMPERATURE,
            )
        else:
            from paired_eval import PairedEvaluation
            sim = PairedEvaluation(
                sim_code=sim_code,
                fork_sim_code="base_trading",
                use_middleware=True,
                fresh=True,
                seed=PARITY_SEED,
                temperature=PARITY_TEMPERATURE,
                advance_with="middleware",
            )
        log = sim.run(PARITY_STEPS)

    return _middleware_rows(log), list(fake.decision_prompts)


def _cleanup(kind: str) -> None:
    """Remove the throwaway simulation folder one harness run created."""
    target = Path(_storage_root()) / f"_parity_{kind}_s{PARITY_SEED}"
    if target.exists():
        shutil.rmtree(target, ignore_errors=True)


class HarnessParityTest(unittest.TestCase):
    """
    Both harnesses must put the same middleware in front of the model.

    BOTH TESTS BELOW ARE CURRENTLY EXPECTED FAILURES. They are not aspirational
    and they are not broken: they found a live divergence on the first run, of
    the same class as run 06's defect 2 and in a place that defect's fix did not
    reach. It is recorded here rather than silently repaired because fixing it
    changes the paired harness's behaviour, which is a decision for the project
    owner and not something to slip into a refactor.

    THE DIVERGENCE
    The two harnesses write different order-feedback memories for the same
    rejection.

    The runner, trading_reverie.py:1198-1213, picks between three branches and
    includes the specific reason:
        "...order to buy 9999 NVDA was REJECTED: quantity outside legal limit.
         Cash $402,600. Do not repeat this order unless..."
    with separate wording for a filtered order and for a clamped one ("the
    order was reduced"), and it fires on `illegal_request and
    is_trade_request(req)`.

    The paired harness, paired_eval.py:241-248, has a single branch and a
    generic string:
        "...order to buy 9999 NVDA was REJECTED/ADJUSTED. Cash $402,600.
         Do not repeat this order unless..."
    and it fires on `driver["score"]["infeasible"]`, a different condition.

    WHY IT MATTERS
    That feedback is a memory node, so it enters retrieval and shapes every
    later decision. CHANGES.md 23.7 records that feeding rejections back is
    precisely what broke the run-05 livelock, 48 rejections collapsing into 1
    episode, so this text is load-bearing rather than cosmetic.

    Observed with 6 steps and seed 42: decision prompts 0 to 4 are byte
    identical, and the first divergence appears at prompt 5, immediately after
    the first rejection is written to memory. The anchor block then differs too,
    the runner retrieving TRAIT analytical and RISK_PROFILE where the paired
    harness retrieves BACKGROUND and TRAIT disciplined, which is downstream of
    the feedback difference rather than a second independent fault: the graphs
    agree until the first feedback write.

    WHEN IT IS FIXED
    Remove the expectedFailure decorators. unittest reports a fixed
    expectedFailure as an unexpected success, so the suite will tell you.
    """

    @unittest.expectedFailure
    def test_middleware_decisions_match_across_harnesses(self):
        """
        The runner and the paired harness must agree decision for decision.

        A mismatch means the paired harness is once again measuring a different
        middleware configuration from the one the independent arm runs, which
        is run 06's defect 2 and which invalidates every paired number in the
        sweep. The first differing row is reported, because the original defect
        showed up at step 1 and the step at which divergence starts says
        whether the cause is setup or drift.
        """
        self.addCleanup(_cleanup, "runner")
        self.addCleanup(_cleanup, "paired")

        runner_rows, _ = _run_harness("runner")
        paired_rows, _ = _run_harness("paired")

        self.assertEqual(
            len(runner_rows), len(paired_rows),
            f"different number of middleware decisions: runner "
            f"{len(runner_rows)}, paired {len(paired_rows)}",
        )

        for index, (expected, actual) in enumerate(zip(runner_rows, paired_rows)):
            self.assertEqual(
                expected, actual,
                f"middleware decision {index} differs between harnesses.\n"
                f"  runner: {expected}\n"
                f"  paired: {actual}\n"
                f"Divergence at index 0 points at setup, for example an "
                f"identity graph refreshed in one harness and not the other; "
                f"divergence later points at state drift.",
            )

    @unittest.expectedFailure
    def test_decision_prompts_match_across_harnesses(self):
        """
        Both harnesses must send the model the same middleware decision prompts.

        This is the direct test of defect 2's mechanism. That defect was not a
        different choice made from the same information, it was a different
        prompt: a frozen identity graph produced a different anchor block, so
        the paired harness fed ~2,036 characters where the runner fed ~800.
        Comparing the prompts themselves catches that at the first decision,
        including the case where both harnesses happen to pick the same action
        from different context.

        Lengths are compared before full text so a failure names the size
        difference, which is readable, rather than dumping two multi-kilobyte
        prompts.
        """
        self.addCleanup(_cleanup, "runner")
        self.addCleanup(_cleanup, "paired")

        _, runner_prompts = _run_harness("runner")
        _, paired_prompts = _run_harness("paired")

        self.assertEqual(
            len(runner_prompts), len(paired_prompts),
            f"different number of middleware decision calls: runner "
            f"{len(runner_prompts)}, paired {len(paired_prompts)}",
        )
        self.assertEqual(
            [len(p) for p in runner_prompts], [len(p) for p in paired_prompts],
            "middleware decision prompt SIZES differ between harnesses, which "
            "is the signature of defect 2: a stale identity graph producing a "
            "different anchor block",
        )
        for index, (expected, actual) in enumerate(zip(runner_prompts,
                                                       paired_prompts)):
            self.assertEqual(
                expected, actual,
                f"middleware decision prompt {index} differs between "
                f"harnesses while being the same length",
            )


if __name__ == "__main__":
    unittest.main()
