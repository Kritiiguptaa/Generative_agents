"""
middleware/metrics_recorder.py

One place every metric is recorded, and one place every rate is computed.

WHY THIS EXISTS

Two reasons, and the first is the important one.

1. THE DETECTOR MUST NOT SIT DOWNSTREAM OF THE FILTER.
   Two 200-step runs once both reported "Total hallucinations: 0". That was not
   a result, it was a wiring fault: the metric was computed from the decision
   that reached execution, and by then run_action_filtering_step had already
   rejected every illegal request and substituted a legal HOLD. The middleware
   arm was structurally incapable of returning a nonzero number (CHANGES.md
   section 1.1).

   This recorder makes that specific mistake impossible to write. A disposition
   event is refused unless a request event with the same id already exists, so
   there is no way to record what happened to a request without having first
   recorded what the model actually asked for.

2. THE COUNTERS WERE DUPLICATED AND HAD DRIFTED.
   _generate_report and _print_report exist twice, in trading_reverie.py and
   again in paired_eval.py. The copies disagree: the independent report gained
   abstention and a non-participant warning, the paired one never did, so two
   reports of the same run were not comparable on the measure that decides
   whether a clean rate was earned or merely abstained into.

WHAT A RATE MEANS HERE
Every rate's denominator is the count of raw `request` events, not of executed
actions and not of decisions. A rejected attempt counts in the numerator AND
the denominator. Getting this wrong produced a reported rate of 145.5% in run
04, where Alex Chen's 48 hallucinations were divided by 33 attempts because the
rejected requests had never been counted as attempts at all.

Rates are therefore always returned next to their denominator, and abstention
and participation travel with them, because an arm that declines to trade can
buy a clean rate: run 06's middleware arm reported 0.0% of 21 attempts against
the baseline's 66.9% of 184, with one agent requesting no trades at all in
three of four seeds.

DURABILITY
Events stream to JSON Lines and the summary is written to a temporary file and
renamed over the target. Run 07 died when the NFS export returned ENOSPC while
the client still showed 99 GB free, and a half-written summary that looks
complete is worse than no summary: a truncated report with a plausible-looking
total is the kind of artefact that ends up in a paper.
"""

import json
import os
import subprocess
import tempfile
from typing import Callable, Dict, List, Optional

# Event type names. Constants rather than bare strings so a typo is an
# AttributeError here instead of an event that silently never matches any
# metric and quietly lowers a count.
EVENT_META = "meta"
EVENT_REQUEST = "request"
EVENT_DISPOSITION = "disposition"
EVENT_EXECUTED = "executed"
EVENT_CONTEXT = "context"
EVENT_ERROR = "error"

# Dispositions a request can end in.
DISPOSITION_ACCEPTED = "accepted"
DISPOSITION_RETRY = "retry"
DISPOSITION_FALLBACK = "fallback"
DISPOSITION_REJECTED = "rejected"
DISPOSITION_RESIZED = "resized"

# Error kinds, kept apart on purpose. The "OLLAMA ERROR" string sentinel makes
# a dead server look like a model that cannot produce JSON, so every fallback
# rate computed while the server was flaky is suspect. These must never share a
# counter: one is infrastructure, the other is the thing being measured.
ERROR_SERVER = "server_error"
ERROR_PARSE = "parse_failure"

# Returned wherever a value could not be determined. Never guessed, never
# defaulted to something plausible: a wrong recorded model or temperature would
# silently misattribute a whole sweep.
UNKNOWN = "NOT FOUND"


# Caches for the two expensive lookups. Both describe the process, not the
# run: the checkout cannot change under a running simulation, and the server
# version will not either. Without this, every report generation paid two git
# subprocesses and an HTTP timeout, which took the test suite from 10 seconds
# to 72 because an unreachable server costs the full timeout each time.
_git_cache: Dict[str, object] = {}
_ollama_version_cache: Dict[str, str] = {}


def git_commit_info(repo_dir: Optional[str] = None) -> Dict[str, object]:
    """
    Short commit hash of the working tree, plus whether it has local edits.

    Recorded because a report currently says nothing about what produced it.
    Run 06 and Run 07 differ by sampling settings that appear nowhere in the
    artefacts, which is exactly the confusion this prevents.

    The dirty flag matters as much as the hash: a result produced from an
    edited tree is not reproducible from that commit, and saying so is the
    difference between an honest artefact and a misleading one.

    Returns UNKNOWN for both fields when git is unavailable or this is not a
    repository, rather than inventing a value.
    """
    cwd = repo_dir or os.path.dirname(os.path.abspath(__file__))
    if cwd in _git_cache:
        return dict(_git_cache[cwd])

    def _git(*args):
        """Run one git command, returning None on any failure."""
        try:
            out = subprocess.run(
                ("git",) + args,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except Exception:
            return None
        if out.returncode != 0:
            return None
        return out.stdout.strip()

    commit = _git("rev-parse", "--short", "HEAD")
    if commit is None:
        info = {"git_commit": UNKNOWN, "git_dirty": UNKNOWN}
    else:
        status = _git("status", "--porcelain")
        info = {
            "git_commit": commit,
            # None means the status call failed, which is not the same as clean.
            "git_dirty": UNKNOWN if status is None else bool(status.strip()),
        }
    _git_cache[cwd] = dict(info)
    return info


def ollama_version(base_url: Optional[str] = None) -> str:
    """
    The Ollama server's reported version, or UNKNOWN if it cannot be asked.

    Best effort by design. The server is often not running when a report is
    generated, and in the stubbed test harness it never is, so an unreachable
    server must produce UNKNOWN rather than an exception or a wrong value.
    """
    key = base_url or ""
    if key in _ollama_version_cache:
        return _ollama_version_cache[key]
    try:
        import requests
        from middleware.id_rag import _cfg
        url = (base_url or _cfg(None, "ollama_base_url")) + "/api/version"
        data = requests.get(url, timeout=5).json()
        version = str(data.get("version") or UNKNOWN)
    except Exception:
        version = UNKNOWN
    _ollama_version_cache[key] = version
    return version


def reset_metadata_caches() -> None:
    """
    Forget the cached commit and server version.

    For tests that need a fresh lookup. Not called in normal operation: within
    one run neither value can change, and re-reading them per report is what
    made the suite eight times slower.
    """
    _git_cache.clear()
    _ollama_version_cache.clear()


class MetricsRecorder:
    """
    Collects events during a run and computes the summary from them.

    Deliberately dumb about trading. It knows nothing about symbols, prices or
    personas; it knows that requests have dispositions and that rates have
    denominators. Keeping the domain out of here is what lets both harnesses
    and, later, the chaos runs share it.

    Metrics are registered functions over the event list rather than methods,
    so DCS, BDI and CVR can be added once their definitions are frozen without
    touching this class.
    """

    def __init__(self, config_name: str = UNKNOWN):
        """Start an empty recorder. `config_name` names the arm or sweep cell."""
        self.events: List[Dict[str, object]] = []
        self.meta: Dict[str, object] = {"config_name": config_name}
        self._next_request_id = 0
        self._known_requests = set()
        self._metrics: Dict[str, Callable] = {}
        self._register_default_metrics()

    # -- recording ---------------------------------------------------------

    def record_meta(self, **fields) -> None:
        """
        Record what produced this run: model, temperature, seed, steps.

        Merged rather than replaced so a caller can add fields as they become
        known, for example the step count only at the end of the run.
        """
        self.meta.update(fields)

    def capture_environment(self, **fields) -> None:
        """
        Record the run's metadata plus the commit and Ollama version.

        Separate from record_meta so that the slow calls, two subprocesses and
        an HTTP request, happen once at a point the caller chooses rather than
        on every metadata update.
        """
        self.meta.update(fields)
        self.meta.update(git_commit_info())
        self.meta.setdefault("ollama_version", ollama_version())

    def record_request(self, step: int, agent: str, arm: str,
                       action: Optional[str], symbol: Optional[str],
                       quantity: Optional[object],
                       is_trade_request: bool = True) -> int:
        """
        Record what the model asked for, BEFORE any correction, and return its id.

        This is logging point 1 and the only place the model's actual request is
        observable. Everything downstream has already been corrected, which is
        why no disposition can be recorded without one of these.

        `is_trade_request` marks whether this counts toward a rate's
        denominator. A HOLD is a decision but not a trade attempt, and counting
        it would dilute every rate; an invented verb IS a trade attempt, because
        the model was trying to trade.
        """
        request_id = self._next_request_id
        self._next_request_id += 1
        self._known_requests.add(request_id)
        self.events.append({
            "type": EVENT_REQUEST,
            "request_id": request_id,
            "step": step,
            "agent": agent,
            "arm": arm,
            "action": action,
            "symbol": symbol,
            "quantity": quantity,
            "is_trade_request": bool(is_trade_request),
        })
        return request_id

    def record_disposition(self, request_id: int, disposition: str,
                           kind: str = "", reason: str = "",
                           infeasible: bool = False,
                           reached_execution: bool = True) -> None:
        """
        Record how a request ended, refusing to do so without the request.

        THE RAISE IS THE POINT. Permitting a disposition with no matching
        request would allow a metric to be computed from corrected decisions
        alone, which is the fault that made two runs both report zero
        hallucinations. An unknown id means the caller is measuring downstream
        of the filter, so this fails loudly instead of producing a number.

        `infeasible` is carried separately from `kind` because an arm's rate is
        about whether the request could not have been honoured, while kind says
        why, and the kinds are not comparable across arms: the middleware arm
        cannot produce `unowned` at all, since its menu is built from holdings.

        `reached_execution` is separate from `disposition` because they are
        different facts and a request can carry both. A decision whose first
        attempt was rejected and retried into a HOLD has disposition "fallback"
        AND never reached execution; a baseline order that was resized has
        disposition "accepted" and DID reach execution. Inferring one from the
        other would collapse the caught-versus-executed split, which is the
        only cross-arm comparison the paper's headline claim rests on: the
        filtered arm catches requests before execution, the baseline can only
        catch them at execution, and its resized orders execute anyway.
        """
        if request_id not in self._known_requests:
            raise ValueError(
                f"disposition for unknown request_id {request_id!r}. "
                f"record_request() must be called first, with the model's "
                f"request as it arrived. Recording a disposition without one "
                f"means the metric is reading the corrected decision, which is "
                f"how two runs both reported 0 hallucinations.")
        self.events.append({
            "type": EVENT_DISPOSITION,
            "request_id": request_id,
            "disposition": disposition,
            "kind": kind,
            "reason": reason,
            "infeasible": bool(infeasible),
            "reached_execution": bool(reached_execution),
        })

    def record_executed(self, request_id: Optional[int], action: str,
                        symbol: Optional[str], quantity: object,
                        price: Optional[float] = None,
                        cash_after: Optional[float] = None) -> None:
        """Record the transition that actually happened. Logging point 3."""
        self.events.append({
            "type": EVENT_EXECUTED,
            "request_id": request_id,
            "action": action,
            "symbol": symbol,
            "quantity": quantity,
            "price": price,
            "cash_after": cash_after,
        })

    def record_context(self, step: int, agent: str, arm: str,
                       context_chars: int = 0,
                       has_reasoning: bool = False,
                       stale_context: bool = False,
                       state_contradiction: bool = False,
                       persona_score: Optional[float] = None) -> None:
        """
        Record prompt size and the reasoning-derived checks for one decision.

        has_reasoning is recorded because an empty reasoning scores clean on
        both the stale-context scan and the persona score for free, so a run
        that often fails to parse looks like a run that reasons well. It is the
        denominator those two rates need.
        """
        self.events.append({
            "type": EVENT_CONTEXT,
            "step": step,
            "agent": agent,
            "arm": arm,
            "context_chars": context_chars,
            "has_reasoning": bool(has_reasoning),
            "stale_context": bool(stale_context),
            "state_contradiction": bool(state_contradiction),
            "persona_score": persona_score,
        })

    def record_error(self, kind: str, detail: str = "") -> None:
        """
        Record a server failure or a parse failure, never conflated.

        A server error is not a hallucination and not a parser fallback. The
        "OLLAMA ERROR" sentinel made the two indistinguishable, so a flaky
        server inflated the model's apparent failure rate.
        """
        if kind not in (ERROR_SERVER, ERROR_PARSE):
            raise ValueError(
                f"unknown error kind {kind!r}; expected {ERROR_SERVER!r} or "
                f"{ERROR_PARSE!r}. These must stay distinct: one is "
                f"infrastructure, the other is the measurement.")
        self.events.append({
            "type": EVENT_ERROR,
            "kind": kind,
            "detail": detail,
        })

    # -- metrics -----------------------------------------------------------

    def register_metric(self, name: str, fn: Callable) -> None:
        """
        Add a named metric computed over the event list.

        The seam for DCS, BDI and CVR, whose definitions are not yet frozen.
        Registering rather than hard-coding means adding one does not touch
        this class, and a metric cannot be added without being named.
        """
        self._metrics[name] = fn

    def _register_default_metrics(self) -> None:
        """Register the metrics the current report needs."""
        self.register_metric("trade_attempts", _metric_trade_attempts)
        self.register_metric("infeasible_requests", _metric_infeasible)
        self.register_metric("caught_pre_execution", _metric_caught)
        self.register_metric("executed_malformed", _metric_executed_malformed)
        self.register_metric("parser_fallbacks", _metric_parser_fallbacks)
        self.register_metric("server_errors", _metric_server_errors)
        self.register_metric("parse_failures", _metric_parse_failures)
        self.register_metric("abstention", _metric_abstention)
        self.register_metric("participation", _metric_participation)
        self.register_metric("stale_context_hits", _metric_stale)
        self.register_metric("state_contradictions", _metric_contradictions)

    def summary(self) -> Dict[str, object]:
        """
        Compute every registered metric, with metadata attached.

        Rates are not computed here. Each metric returns its own count and,
        where a rate is meaningful, its denominator, so that the report can
        print the denominator beside the number. A bare percentage with an
        unstated denominator is how a 145.5% rate reached a report.
        """
        result = {"meta": dict(self.meta)}
        for name, fn in self._metrics.items():
            result[name] = fn(self.events)
        return result

    # -- output ------------------------------------------------------------

    def write_events(self, path: str) -> None:
        """
        Stream every event to JSON Lines.

        One object per line, so a run killed mid-write loses at most the last
        line rather than the whole file, which is what a single JSON array
        would risk.
        """
        _atomic_write(path, "".join(
            json.dumps(event, sort_keys=True) + "\n" for event in self.events))

    def write_summary(self, path: str) -> None:
        """
        Write the summary, atomically, so a disk-full cannot truncate it.

        Written to a temporary file in the same directory and renamed over the
        target: rename within a filesystem is atomic, so the target is either
        the old complete summary or the new complete one, never half of either.
        Run 07 died on ENOSPC, and a truncated summary that still parses is the
        kind of artefact that reaches a paper.
        """
        _atomic_write(path, json.dumps(self.summary(), indent=2, sort_keys=True))


def collect_run_metadata(model: object = None,
                         temperature: object = None,
                         seed: object = None,
                         steps: object = None,
                         config_name: object = None,
                         include_ollama_version: bool = True) -> Dict[str, object]:
    """
    Everything needed to say what produced a report.

    A report currently records none of this. Run 06 and Run 07 differ by their
    sampling settings, temperature 0 against 0.7, and those settings appear
    nowhere in the artefacts, so two reports can be indistinguishable while
    describing runs whose absolute rates are not comparable.

    The model is read from the live configuration rather than assumed, because
    OLLAMA_CHAT_MODEL is an environment variable and a sweep that changed it
    would otherwise produce reports claiming the default.

    Anything that cannot be determined is UNKNOWN. Never a plausible default: a
    wrongly recorded temperature would misattribute a whole sweep, and that is
    worse than an obviously missing one.
    """
    if model is None:
        try:
            from persona.prompt_template.gpt_structure import OLLAMA_CHAT_MODEL
            model = OLLAMA_CHAT_MODEL
        except Exception:
            model = UNKNOWN

    meta = {
        "model": model if model is not None else UNKNOWN,
        "temperature": temperature if temperature is not None else UNKNOWN,
        "seed": seed if seed is not None else UNKNOWN,
        "steps": steps if steps is not None else UNKNOWN,
        "config_name": config_name if config_name is not None else UNKNOWN,
    }
    meta.update(git_commit_info())
    # Skippable because it is an HTTP call: in the stubbed test harness there is
    # no server, and paying a timeout per report for a guaranteed UNKNOWN is
    # waste.
    meta["ollama_version"] = (ollama_version() if include_ollama_version
                              else UNKNOWN)
    return meta


# Fixed field width, so the metadata block lines up with the existing report
# lines ("Middleware      : ...", "Steps simulated : ..."). The existing labels
# are not renamed, reordered or reformatted; these are appended beneath them,
# because the Run 06 log parser matches on those labels.
_META_LABEL_WIDTH = 16

_META_LINE_ORDER = (
    ("model", "Model"),
    ("temperature", "Temperature"),
    ("seed", "Seed"),
    ("config_name", "Config"),
    ("git_commit", "Commit"),
    ("git_dirty", "Tree dirty"),
    ("ollama_version", "Ollama"),
)


def format_meta_lines(meta: Dict[str, object],
                      skip: Optional[frozenset] = None) -> List[str]:
    """
    Render the metadata as report lines, in a fixed order.

    Shared by both harnesses so their reports cannot describe the same run
    differently, which is how the independent report came to carry abstention
    while the paired one never did.

    `skip` drops fields the surrounding report already prints. The paired
    report prints its own "Seed" line, and emitting a second line with the
    identical label would leave a line-oriented parser with two candidates for
    one field. "Steps" is always omitted for the same reason: both reports
    already print a step count under their own label.
    """
    skip = skip or frozenset()
    lines = []
    for key, label in _META_LINE_ORDER:
        if key not in meta or key in skip:
            continue
        value = meta[key]
        if isinstance(value, bool):
            value = "yes" if value else "no"
        lines.append(f"{label:<{_META_LABEL_WIDTH}}: {value}")
    return lines


def recorder_from_trading_log(log: list, arm: str = "middleware",
                              config_name: str = UNKNOWN) -> MetricsRecorder:
    """
    Build a recorder from an existing trading_log entry list.

    WHY AN ADAPTER RATHER THAN A REWRITE
    The recorder has to produce the same numbers the current report produces
    before it can be trusted to replace it. Translating a real run's log into
    events and comparing the two summaries proves that without changing how any
    run behaves, so equivalence is established first and the swap carries no
    risk of silently moving a published number.

    It is also the shape WP6's log parser needs, so the analysis scripts can
    read old runs through the same metric definitions as new ones.

    MAPPING, which mirrors _generate_report exactly:
      one request per log entry, since every entry is one decision;
      is_trade_request from the recorded pre-correction request, NOT from the
        executed action, because the denominator must be what the model asked
        to do (an executed-action denominator divided by ~0 in an arm where
        every illegal trade was blocked, and reported 0.0%);
      infeasible from the entry's `hallucination` flag, which is already the
        union of the two detection points the two arms need;
      reached_execution from halluc_disposition, where "caught_pre_execution"
        means it did not.
    """
    from middleware.action_filtering import is_trade_request

    recorder = MetricsRecorder(config_name=config_name)
    for entry in log:
        if not isinstance(entry, dict):
            continue
        decision = entry.get("decision") or {}
        requested = entry.get("requested") or {}
        action = decision.get("action", "hold")

        request_id = recorder.record_request(
            step=entry.get("step"),
            agent=entry.get("agent"),
            arm=arm,
            action=action,
            symbol=decision.get("symbol"),
            quantity=decision.get("quantity"),
            is_trade_request=bool(is_trade_request(requested)),
        )

        disposition = str(entry.get("filter_status") or "success")
        if disposition == "success":
            disposition = DISPOSITION_ACCEPTED
        recorder.record_disposition(
            request_id,
            disposition=disposition,
            kind=str(entry.get("halluc_kind") or ""),
            reason=str(entry.get("error_reason") or ""),
            infeasible=bool(entry.get("hallucination")),
            reached_execution=(
                entry.get("halluc_disposition") != "caught_pre_execution"),
        )

        compression = entry.get("compression") or {}
        recorder.record_context(
            step=entry.get("step"),
            agent=entry.get("agent"),
            arm=arm,
            context_chars=int(compression.get("chars_out") or 0),
            has_reasoning=bool(entry.get("has_reasoning")),
            stale_context=bool(entry.get("stale_context")),
            state_contradiction=bool(entry.get("state_contradiction")),
            persona_score=entry.get("persona_drift_score"),
        )
    return recorder


def _atomic_write(path: str, text: str) -> None:
    """
    Write text to path via a temporary file and a rename.

    fsync before the rename, so the rename cannot land before the data on a
    filesystem that reorders them, which would leave an empty file where a
    complete one is claimed.
    """
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=directory, delete=False,
        prefix=".metrics-", suffix=".tmp")
    try:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        os.replace(handle.name, path)
    except Exception:
        handle.close()
        if os.path.exists(handle.name):
            os.unlink(handle.name)
        raise


# ---------------------------------------------------------------------------
# Default metrics
#
# Each takes the event list and returns a dict. Where a rate is meaningful the
# denominator is returned alongside the count, never a bare percentage: the
# report prints both so that "0 of 21" cannot be read as "0 of 184".
# ---------------------------------------------------------------------------

def _requests(events, trade_only=True):
    """Request events, optionally only those that count as trade attempts."""
    return [e for e in events
            if e["type"] == EVENT_REQUEST
            and (e.get("is_trade_request") or not trade_only)]


def _dispositions_by_request(events):
    """Map request id to its disposition event, for joining the two streams."""
    return {e["request_id"]: e
            for e in events if e["type"] == EVENT_DISPOSITION}


def _metric_trade_attempts(events):
    """Count of requests that count toward a rate's denominator."""
    return {"count": len(_requests(events))}


def _metric_infeasible(events):
    """
    Requests that could not have been honoured as asked, and the denominator.

    The denominator is trade attempts, including the rejected ones. Excluding
    them is what produced a 145.5% rate.
    """
    attempts = _requests(events)
    dispositions = _dispositions_by_request(events)
    infeasible = sum(1 for r in attempts
                     if dispositions.get(r["request_id"], {}).get("infeasible"))
    return {"count": infeasible, "denominator": len(attempts)}


def _metric_caught(events):
    """
    Infeasible requests stopped before they could execute.

    Reported next to executed_malformed because the raw infeasible count is not
    comparable across arms on its own: the filtered arm catches requests before
    execution, the baseline can only catch them at execution, and the
    baseline's resized orders execute anyway.
    """
    dispositions = _dispositions_by_request(events)
    return {"count": sum(1 for d in dispositions.values()
                         if d.get("infeasible")
                         and not d.get("reached_execution"))}


def _metric_executed_malformed(events):
    """
    Infeasible requests that reached execution anyway. The headline number.

    This is the claim that survived every caveat in run 06: 0 in the middleware
    arm in every seed, 3 to 7 in the baseline in every seed.
    """
    dispositions = _dispositions_by_request(events)
    return {"count": sum(1 for d in dispositions.values()
                         if d.get("infeasible")
                         and d.get("reached_execution"))}


def _metric_parser_fallbacks(events):
    """Decisions that ended in a substituted HOLD, over all requests."""
    dispositions = _dispositions_by_request(events)
    total = len(_requests(events, trade_only=False))
    return {
        "count": sum(1 for d in dispositions.values()
                     if d.get("disposition") == DISPOSITION_FALLBACK),
        "denominator": total,
    }


def _metric_server_errors(events):
    """Infrastructure failures. Never a hallucination, never a fallback."""
    return {"count": sum(1 for e in events
                         if e["type"] == EVENT_ERROR
                         and e.get("kind") == ERROR_SERVER)}


def _metric_parse_failures(events):
    """Model output that could not be parsed. The measurement, not the server."""
    return {"count": sum(1 for e in events
                         if e["type"] == EVENT_ERROR
                         and e.get("kind") == ERROR_PARSE)}


def _metric_abstention(events):
    """
    Share of decisions that were a HOLD, with its denominator.

    Always reported beside any rate. An arm can improve its rate by trading
    less, and nothing in a bare "0 hallucinations" distinguishes an arm that
    traded heavily and stayed clean from one that mostly declined to act.
    """
    all_requests = _requests(events, trade_only=False)
    holds = sum(1 for r in all_requests
                if str(r.get("action") or "").lower() == "hold")
    return {"count": holds, "denominator": len(all_requests)}


def _metric_participation(events):
    """
    Agents that requested at least one trade, out of all agents seen.

    A non-participant is the strongest version of the abstention problem: run
    06 had one agent requesting no trades at all in three of four seeds, while
    the report still printed a rate for the arm.
    """
    agents = {r["agent"] for r in _requests(events, trade_only=False)}
    traders = {r["agent"] for r in _requests(events)}
    return {"count": len(traders), "denominator": len(agents),
            "non_participants": sorted(agents - traders)}


def _metric_stale(events):
    """
    Stale-context hits over decisions that had reasoning to score.

    The denominator is reasoned decisions, not all decisions: an empty
    reasoning cannot be stale, so counting it as clean inflates the pass rate.
    """
    contexts = [e for e in events if e["type"] == EVENT_CONTEXT]
    reasoned = [e for e in contexts if e.get("has_reasoning")]
    return {"count": sum(1 for e in reasoned if e.get("stale_context")),
            "denominator": len(reasoned)}


def _metric_contradictions(events):
    """State contradictions over reasoned decisions. Same denominator logic."""
    contexts = [e for e in events if e["type"] == EVENT_CONTEXT]
    reasoned = [e for e in contexts if e.get("has_reasoning")]
    return {"count": sum(1 for e in reasoned if e.get("state_contradiction")),
            "denominator": len(reasoned)}
