"""
Golden-file regression harness for the decision pipeline.

WHY THIS EXISTS
The middleware is about to be refactored behind a single entry point
(MiddlewareWrapper). That refactor must not change behaviour, because Run 06
and Run 07 are only comparable if the pipeline they measure stays identical.
Reasoning about an inheritance refactor is not enough: run 06's defect 2 was a
harness that looked correct and diverged at the first decision.

So this file pins what the pipeline does *today*. It runs the real
TradingReverie over the real base_trading fork for a fixed number of steps,
with every network call replaced by a deterministic stub, and snapshots the
result. After the refactor the snapshot must match byte for byte.

WHAT IS PINNED
  1. Every decision: step, agent, action, symbol, quantity, status, error.
  2. The full report dict, which is what the paper's tables are built from.
  3. Per-step context sizes and compression statistics, so a change in what
     reaches the prompt is caught even when the decision happens to match.
  4. The ordered sequence of LLM call categories. This is the part a decision
     snapshot alone would miss: run 06's defect was a *reordering*, an identity
     graph updated before retrieval in one harness and never in the other.

DETERMINISM
Nothing here touches Ollama. requests.post is patched at the `requests` module
level, which covers both gpt_structure and id_rag since both do `import
requests` and resolve the attribute at call time. Embeddings are a hash of the
input text, so cosine similarity, near-duplicate removal and ID-RAG top-k are
all reproducible. Chat responses are scripted per call category.

Run from backend_server:
    python -m pytest tests/test_golden_pipeline.py

To (re)capture the golden files, which must be done on a commit whose
behaviour you intend to freeze:
    python tests/test_golden_pipeline.py --capture
"""

import hashlib
import json
import math
import os
import shutil
import sys
import unittest
from pathlib import Path
from unittest import mock

BACKEND = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

GOLDEN_DIR = Path(__file__).parent / "golden"

# Steps and seed are part of the contract: changing either invalidates the
# stored files. 20 steps is enough to exercise a day boundary, several
# interactions and the retry path, and still runs in seconds under the stub.
GOLDEN_STEPS = 20
GOLDEN_SEED = 42
GOLDEN_TEMPERATURE = 0.7

# Embedding width. Any fixed width works; 64 keeps the snapshot small while
# leaving cosine similarity meaningful enough that dedup and top-k actually
# discriminate between texts.
EMBED_DIM = 64

SYMBOLS = ["NVDA", "AAPL", "TSLA", "AMD", "GOOGL"]


# ---------------------------------------------------------------------------
# Deterministic stand-ins for the two kinds of Ollama call
# ---------------------------------------------------------------------------

def _deterministic_embedding(text: str) -> list:
    """
    Map a string to a fixed unit-length vector, the same way every time.

    Hashes the text, then expands the digest into EMBED_DIM floats in [-1, 1]
    and normalises. Being a pure function of the text is the whole point:
    memory_compression dedups by cosine similarity and id_rag ranks by it, so a
    random vector would make both non-reproducible and this harness useless.
    Equal texts get identical vectors, which is what lets the near-duplicate
    stage fire at all.
    """
    digest = hashlib.sha256(text.encode("utf-8", "replace")).digest()
    # Stretch the 32-byte digest to the width we need by re-hashing with a
    # counter, so EMBED_DIM is not capped by the digest size.
    raw = bytearray()
    counter = 0
    while len(raw) < EMBED_DIM:
        raw.extend(hashlib.sha256(digest + bytes([counter])).digest())
        counter += 1
    vec = [(raw[i] / 127.5) - 1.0 for i in range(EMBED_DIM)]
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def _classify_prompt(prompt: str) -> str:
    """
    Decide which kind of model call a prompt represents.

    Dispatching on marker text rather than on call order means the stub stays
    correct if the number of non-decision calls changes, while still letting us
    record the real order separately. The markers are taken verbatim from the
    prompt builders: action_filtering.build_prompt, free_form_decision,
    trading_daily_plan.generate_daily_plan and trading_interactions.
    """
    if "You are a trading agent planner" in prompt:
        return "daily_plan"
    if "Two trading agents briefly exchange insights" in prompt:
        return "interaction"
    # Both arms' decision prompts end with this instruction; the middleware arm
    # additionally carries an explicit legal-action menu.
    if "Respond ONLY in this JSON format" in prompt:
        if "You MUST choose from ONLY these actions" in prompt:
            return "decision_middleware"
        return "decision_baseline"
    return "other"


def _plan_response() -> str:
    """
    A valid daily plan whose schedule sums to the 390 minutes the planner wants.

    Returned verbatim for every planning call. A plan that parses matters
    because the fallback path assigns every agent the same canned plan, which
    flattens exactly the persona differentiation the pipeline is supposed to
    produce; pinning the parsed path keeps the snapshot representative.
    """
    return json.dumps({
        "daily_req": [
            "Review overnight news and pre-market moves",
            "Scan the watchlist for setups at the open",
            "Execute high-conviction trades only",
            "Check risk limits at mid-day",
            "Square up before the close",
        ],
        "schedule_blocks": [
            {"task": "Pre-market review", "minutes": 60},
            {"task": "Open trade window", "minutes": 120},
            {"task": "Mid-day review", "minutes": 90},
            {"task": "Risk check", "minutes": 60},
            {"task": "Close out", "minutes": 60},
        ],
    })


def _interaction_response() -> str:
    """
    A valid peer-exchange summary, returned verbatim for every interaction call.

    Same reasoning as the plan: the parsed path is the one a real run takes
    when the model behaves, so that is what the golden file should describe.
    """
    return json.dumps({
        "topic": "momentum in semis",
        "summary": "Both traders see continued strength in semiconductor names.",
        "follow_up": "Re-check NVDA at the next open.",
        "symbol": "NVDA",
    })


# Raw model outputs for decision calls, cycled in order. This deliberately
# mixes clean responses with the failure modes the validator exists to catch,
# so the snapshot covers accept, retry, fallback and the executor guard rather
# than only the happy path:
#   0 small legal buy            -> accepted in both arms
#   1 plain hold                 -> accepted
#   2 sell with no position      -> legal-menu rejection (mw) / executor (base)
#   3 truncated JSON             -> parse failure, burns the retry
#   4 unknown verb "short"       -> illegal request in both arms
#   5 absurd quantity            -> over-limit rejection (mw) / clamp (base)
#   6 ANALYZE                    -> normalised to HOLD, must not count as illegal
#   7 legal buy, different name  -> accepted
_DECISION_SCRIPT = [
    '{"action": "BUY", "symbol": "AAPL", "quantity": 1, "reasoning": "small starter position"}',
    '{"action": "HOLD", "symbol": null, "quantity": 0, "reasoning": "waiting for confirmation"}',
    '{"action": "SELL", "symbol": "GOOGL", "quantity": 5, "reasoning": "trimming exposure"}',
    '{"action": "BUY", "symbol": "TSLA", "quantity":',
    '{"action": "SHORT", "symbol": "NVDA", "quantity": 10, "reasoning": "expecting a pullback"}',
    '{"action": "BUY", "symbol": "NVDA", "quantity": 9999, "reasoning": "high conviction"}',
    '{"action": "ANALYZE", "symbol": null, "quantity": 0, "reasoning": "studying the tape"}',
    '{"action": "BUY", "symbol": "AMD", "quantity": 2, "reasoning": "adding on strength"}',
]


def _other_response() -> str:
    """
    Response for every call that is not a plan, an interaction or a decision.

    Reflection and the inherited cognitive modules make many calls with many
    different expected shapes. Satisfying all of them is not this harness's
    job: what matters is that the answer is the same every time, so those paths
    take the same branch on every run. A bare JSON object does that.
    """
    return "{}"


class _FakeHTTPResponse:
    """Minimal stand-in for a requests Response, enough for both call sites."""

    def __init__(self, payload: dict):
        self.status_code = 200
        self._payload = payload

    def json(self):
        """Return the canned decoded body."""
        return self._payload

    def raise_for_status(self):
        """Never raises: the stub only ever produces successful responses."""
        return None

    @property
    def text(self):
        """Serialised body, for the plain-text fallback in ollama_request."""
        return json.dumps(self._payload)


class _FakeOllama:
    """
    Stands in for the whole Ollama server, and records what was asked of it.

    One instance per run. It dispatches on the request URL to tell embeddings
    from completions, then on prompt markers to tell the kinds of completion
    apart. It keeps two separate counters on purpose:

      `calls` records every completion in order, which is what catches a
      reordering of the pipeline.

      `decision_index` advances only on decision calls, so that a refactor
      which legitimately changes how many planning or reflection calls happen
      does not shift which scripted decision each agent receives. Without that
      split, an unrelated change would rewrite every decision in the snapshot
      and the diff would be unreadable.
    """

    def __init__(self):
        self.calls = []
        self.decision_index = 0
        self.embedding_calls = 0

    def post(self, url, json=None, timeout=None, **kwargs):
        """Route one request to the right canned response and record it."""
        payload = json or {}
        if "/api/embed" in url:
            self.embedding_calls += 1
            text = payload.get("prompt") or payload.get("input") or ""
            return _FakeHTTPResponse({"embedding": _deterministic_embedding(str(text))})

        prompt = str(payload.get("prompt", ""))
        kind = _classify_prompt(prompt)
        self.calls.append(kind)

        if kind == "daily_plan":
            body = _plan_response()
        elif kind == "interaction":
            body = _interaction_response()
        elif kind in ("decision_middleware", "decision_baseline"):
            body = _DECISION_SCRIPT[self.decision_index % len(_DECISION_SCRIPT)]
            self.decision_index += 1
        else:
            body = _other_response()

        return _FakeHTTPResponse({"response": body, "done_reason": "stop"})


# ---------------------------------------------------------------------------
# Running the pipeline and reducing it to a comparable snapshot
# ---------------------------------------------------------------------------

def _scrub(value):
    """
    Strip anything that changes between runs for reasons we do not care about.

    Timestamps and absolute paths differ on every run and in every checkout, so
    comparing them would make the golden file fail for no reason. Everything
    else is kept, including floats, because a changed portfolio value is
    exactly the kind of drift this harness is for.
    """
    if isinstance(value, dict):
        return {k: _scrub(v) for k, v in value.items()
                if k not in ("timestamp", "time", "curr_time", "path",
                             "sim_folder", "log_path")}
    if isinstance(value, list):
        return [_scrub(v) for v in value]
    if isinstance(value, float):
        # Round to cents. Float noise at the 1e-12 level is not behaviour
        # change, and pinning it would make the file fail across Python builds.
        return round(value, 2)
    if isinstance(value, str) and BACKEND.replace("\\", "/") in value.replace("\\", "/"):
        return "<backend>"
    return value


def _decision_rows(log: list) -> list:
    """
    Reduce the run log to the decision fields the paper's claims depend on.

    The chosen action lives in the nested "decision" dict and the validator
    outcome in "filter_status", not at the top level of the entry. Reading the
    top level instead returns None for every field, which silently produces a
    golden file that pins nothing: the first version of this function did
    exactly that, and all 60 captured actions were null.

    Reasoning text is reduced to a boolean. It is model prose, it would bloat
    the file, and the only thing downstream depends on is whether it is empty,
    since an empty reasoning scores clean on both the stale-context scan and
    the persona score for free.

    Cash and portfolio value are included because they are the cheapest
    available proof that the whole trajectory, not merely each isolated
    decision, is unchanged.
    """
    rows = []
    for entry in log:
        if not isinstance(entry, dict):
            continue
        decision = entry.get("decision") or {}
        compression = entry.get("compression") or {}
        rows.append(_scrub({
            "step": entry.get("step"),
            "agent": entry.get("agent"),
            "action": decision.get("action"),
            "symbol": decision.get("symbol"),
            "quantity": decision.get("quantity"),
            "has_reasoning": bool((decision.get("reasoning") or "").strip()),
            "filter_status": entry.get("filter_status"),
            "error_reason": entry.get("error_reason"),
            "illegal_request": entry.get("illegal_request"),
            "outcome": entry.get("outcome"),
            "hallucination": entry.get("hallucination"),
            "halluc_kind": entry.get("halluc_kind"),
            "halluc_disposition": entry.get("halluc_disposition"),
            "requested": entry.get("requested") or {},
            "requested_quantity": entry.get("requested_quantity"),
            "filled_quantity": entry.get("filled_quantity"),
            # Compressed memory size, which is the quantity the compression
            # stage controls. Note this excludes the ID-RAG identity block,
            # because chars_out is measured before the anchor is prepended.
            "context_chars": compression.get("chars_out"),
            "cash": entry.get("cash"),
            "portfolio_value": entry.get("portfolio_value"),
        }))
    return rows


def run_pipeline_snapshot(use_middleware: bool,
                          steps: int = GOLDEN_STEPS,
                          seed: int = GOLDEN_SEED) -> dict:
    """
    Run one arm end to end against the stub and return a comparable snapshot.

    Uses the real TradingReverie and the real base_trading fork, because a
    hand-built fake pipeline would not catch a refactor that breaks the wiring
    between the real pieces, which is the only failure mode worth testing for.

    The simulation writes into the shared storage directory like a real run, so
    the caller is responsible for removing the folder afterwards; the sim code
    is namespaced to make that safe.
    """
    arm = "middleware" if use_middleware else "baseline"
    sim_code = f"_golden_{arm}_s{seed}"

    fake = _FakeOllama()

    # Patch at the `requests` module so both gpt_structure and id_rag are
    # covered: each does `import requests` and looks up .post at call time.
    # id_rag's embedding cache is process-global, so it is cleared first or a
    # previous arm's vectors would leak into this one.
    import requests as _requests
    from middleware import id_rag as _id_rag

    _id_rag.id_rag_cache_clear()

    with mock.patch.object(_requests, "post", side_effect=fake.post):
        from trading_reverie import TradingReverie

        storage = Path(_storage_root())
        target = storage / sim_code
        if target.exists():
            shutil.rmtree(target)

        sim = TradingReverie(
            sim_code=sim_code,
            fork_sim_code="base_trading",
            use_middleware=use_middleware,
            fresh=True,
            seed=seed,
            temperature=GOLDEN_TEMPERATURE,
        )
        log = sim.run(steps)
        report = sim._generate_report(log, steps)

    return {
        "arm": arm,
        "steps": steps,
        "seed": seed,
        "temperature": GOLDEN_TEMPERATURE,
        "decisions": _decision_rows(log),
        "report": _scrub(report),
        # The ordered call sequence. A reordering of the pipeline shows up here
        # even when every decision is unchanged, which is the gap a
        # decision-only snapshot leaves and the one defect 2 fell through.
        "llm_call_sequence": fake.calls,
        "llm_call_counts": {k: fake.calls.count(k) for k in sorted(set(fake.calls))},
        "embedding_calls": fake.embedding_calls,
    }


def run_log_and_report(use_middleware: bool,
                       steps: int = GOLDEN_STEPS,
                       seed: int = GOLDEN_SEED) -> tuple:
    """
    Run one arm and return its raw (log, report), without reducing either.

    Exists for test_metrics_recorder's equivalence check, which needs the full
    log entries and the real report dict rather than the trimmed snapshot the
    golden comparison uses. Kept here so there is one definition of how a
    deterministic run is set up: a second copy would be free to drift from the
    stub configuration the golden files were captured under, and then the two
    test files would disagree about what "a 20-step run" means.

    The caller is responsible for removing the simulation folder; the sim code
    is namespaced for that.
    """
    arm = "middleware" if use_middleware else "baseline"
    sim_code = f"_equiv_{arm}_s{seed}"
    fake = _FakeOllama()

    import requests as _requests
    from middleware import id_rag as _id_rag

    _id_rag.id_rag_cache_clear()

    with mock.patch.object(_requests, "post", side_effect=fake.post):
        from trading_reverie import TradingReverie

        target = Path(_storage_root()) / sim_code
        if target.exists():
            shutil.rmtree(target)

        sim = TradingReverie(
            sim_code=sim_code,
            fork_sim_code="base_trading",
            use_middleware=use_middleware,
            fresh=True,
            seed=seed,
            temperature=GOLDEN_TEMPERATURE,
        )
        log = sim.run(steps)
        report = sim._generate_report(log, steps)

    shutil.rmtree(Path(_storage_root()) / sim_code, ignore_errors=True)
    return log, report


def _storage_root() -> str:
    """Absolute path of the simulation storage directory the runner writes to."""
    from utils import fs_storage
    return fs_storage


def golden_path(arm: str) -> Path:
    """Location of the stored snapshot for one arm."""
    return GOLDEN_DIR / f"pipeline_{GOLDEN_STEPS}step_{arm}.json"


def capture() -> None:
    """
    Record both arms' snapshots as the golden files.

    Run this only on a commit whose behaviour you intend to freeze. Capturing
    after a refactor defeats the entire purpose, since the file would then
    describe the refactor's own output rather than the behaviour it was
    supposed to preserve.
    """
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    for use_middleware in (True, False):
        snapshot = run_pipeline_snapshot(use_middleware)
        path = golden_path(snapshot["arm"])
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(snapshot, handle, indent=2, sort_keys=True)
        _cleanup_sim(snapshot["arm"])
        print(f"[golden] wrote {path.name}: "
              f"{len(snapshot['decisions'])} decisions, "
              f"{len(snapshot['llm_call_sequence'])} llm calls")


def _cleanup_sim(arm: str) -> None:
    """Remove the throwaway simulation folder a snapshot run created."""
    target = Path(_storage_root()) / f"_golden_{arm}_s{GOLDEN_SEED}"
    if target.exists():
        shutil.rmtree(target, ignore_errors=True)


# ---------------------------------------------------------------------------
# The test
# ---------------------------------------------------------------------------

class GoldenPipelineTest(unittest.TestCase):
    """
    Asserts the pipeline still behaves exactly as the golden files describe.

    Absence of a golden file is a failure, not a pass. A harness that silently
    regenerates its own baseline proves nothing, and would quietly bless
    whatever the code happens to do on the day it runs.
    """

    def _assert_arm_matches(self, arm: str, use_middleware: bool):
        """Compare one arm's live snapshot against its stored golden file."""
        path = golden_path(arm)
        self.assertTrue(
            path.exists(),
            f"missing golden file {path}. Capture it on a known-good commit "
            f"with: python tests/test_golden_pipeline.py --capture",
        )
        with open(path, encoding="utf-8") as handle:
            expected = json.load(handle)

        self.addCleanup(_cleanup_sim, arm)
        actual = run_pipeline_snapshot(use_middleware)

        # Compare the cheap structural things first so a failure message points
        # at the smallest difference rather than dumping the whole snapshot.
        self.assertEqual(expected["llm_call_counts"], actual["llm_call_counts"],
                         "the mix of LLM calls changed")
        self.assertEqual(expected["llm_call_sequence"], actual["llm_call_sequence"],
                         "the ORDER of LLM calls changed, which is how run 06's "
                         "paired harness silently ran a different middleware")
        self.assertEqual(expected["embedding_calls"], actual["embedding_calls"],
                         "the number of embedding calls changed")
        self.assertEqual(expected["decisions"], actual["decisions"],
                         "at least one decision changed")
        self.assertEqual(expected["report"], actual["report"],
                         "the report the paper's tables are built from changed")

    def test_middleware_arm_is_unchanged(self):
        """The middleware arm must survive the wrapper refactor untouched."""
        self._assert_arm_matches("middleware", True)

    def test_baseline_arm_is_unchanged(self):
        """
        The baseline arm must survive it too.

        The wrapper is only supposed to touch the middleware path, so any
        movement here means the refactor leaked into the control arm and every
        cross-arm comparison in the paper would be invalid.
        """
        self._assert_arm_matches("baseline", False)


if __name__ == "__main__":
    if "--capture" in sys.argv:
        capture()
    else:
        unittest.main()
