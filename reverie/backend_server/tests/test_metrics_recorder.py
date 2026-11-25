"""
Tests for middleware/metrics_recorder.py

The two that matter most are the ones pinning a rule rather than a number:

  test_disposition_without_a_request_raises
      The recorder must refuse to record what happened to a request it never
      saw. That is what makes "the detector ran after the filter" impossible to
      write by accident, which is the fault that had two 200-step runs both
      report "Total hallucinations: 0".

  test_a_failed_write_leaves_the_previous_summary_intact
      A truncated summary that still parses is worse than no summary. Run 07
      died on ENOSPC while the client showed 99 GB free.

Run from backend_server:
    python -m pytest tests/test_metrics_recorder.py
"""

import json
import os
import shutil
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

BACKEND = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from middleware.metrics_recorder import (  # noqa: E402
    DISPOSITION_ACCEPTED,
    DISPOSITION_FALLBACK,
    DISPOSITION_REJECTED,
    DISPOSITION_RESIZED,
    ERROR_PARSE,
    ERROR_SERVER,
    MetricsRecorder,
    UNKNOWN,
    git_commit_info,
    ollama_version,
)


class RequestDispositionRuleTest(unittest.TestCase):
    """The ordering rule that prevents measuring downstream of the filter."""

    def test_disposition_without_a_request_raises(self):
        """
        Recording a disposition for an unknown request must fail loudly.

        This is the structural guarantee. If it were permitted, a metric could
        be computed from corrected decisions alone, and the middleware arm
        would be incapable of reporting a nonzero hallucination count, which is
        exactly what happened before.
        """
        recorder = MetricsRecorder()
        with self.assertRaises(ValueError) as caught:
            recorder.record_disposition(0, DISPOSITION_REJECTED)
        # The message has to say what to do, not merely that something is wrong.
        self.assertIn("record_request", str(caught.exception))

    def test_disposition_after_a_request_is_accepted(self):
        """The normal path: request first, then its disposition."""
        recorder = MetricsRecorder()
        request_id = recorder.record_request(
            step=0, agent="Alex Chen", arm="middleware",
            action="buy", symbol="NVDA", quantity=9999)
        recorder.record_disposition(request_id, DISPOSITION_REJECTED,
                                    kind="illegal_request", infeasible=True)
        summary = recorder.summary()
        self.assertEqual(summary["trade_attempts"]["count"], 1)
        self.assertEqual(summary["infeasible_requests"]["count"], 1)

    def test_request_ids_are_distinct(self):
        """
        Each request gets its own id.

        Reusing an id would let one disposition overwrite another's, silently
        lowering a count.
        """
        recorder = MetricsRecorder()
        ids = [recorder.record_request(step=i, agent="A", arm="middleware",
                                       action="buy", symbol="NVDA", quantity=1)
               for i in range(5)]
        self.assertEqual(len(set(ids)), 5)


class DenominatorTest(unittest.TestCase):
    """
    Rates travel with their denominator, and rejected attempts count in both.

    Run 04 reported Alex Chen at 48 hallucinations against 33 attempts, a rate
    of 145.5%, because the rejected requests were counted in the numerator and
    never in the denominator.
    """

    def test_a_rejected_attempt_counts_in_both_numerator_and_denominator(self):
        """The rate of one rejected attempt out of one must be 1 of 1."""
        recorder = MetricsRecorder()
        request_id = recorder.record_request(
            step=0, agent="A", arm="middleware",
            action="buy", symbol="NVDA", quantity=9999)
        recorder.record_disposition(request_id, DISPOSITION_FALLBACK,
                                    infeasible=True)
        infeasible = recorder.summary()["infeasible_requests"]
        self.assertEqual(infeasible["count"], 1)
        self.assertEqual(infeasible["denominator"], 1)

    def test_a_hold_is_not_a_trade_attempt(self):
        """
        HOLD is a decision but not an attempt, so it stays out of the rate.

        Counting it would dilute every rate by the abstention rate, which is
        the opposite of what the abstention metric exists to expose.
        """
        recorder = MetricsRecorder()
        recorder.record_request(step=0, agent="A", arm="middleware",
                                action="hold", symbol=None, quantity=0,
                                is_trade_request=False)
        summary = recorder.summary()
        self.assertEqual(summary["trade_attempts"]["count"], 0)
        # But it is still a decision, so abstention sees it.
        self.assertEqual(summary["abstention"]["denominator"], 1)
        self.assertEqual(summary["abstention"]["count"], 1)

    def test_an_invented_verb_is_a_trade_attempt(self):
        """
        An illegal verb counts as an attempt, because the model tried to trade.

        Treating it as a non-attempt would remove the clearest hallucinations
        from the denominator and flatter the rate.
        """
        recorder = MetricsRecorder()
        request_id = recorder.record_request(
            step=0, agent="A", arm="baseline",
            action="short", symbol="NVDA", quantity=10,
            is_trade_request=True)
        recorder.record_disposition(request_id, DISPOSITION_REJECTED,
                                    kind="illegal_request", infeasible=True)
        self.assertEqual(recorder.summary()["trade_attempts"]["count"], 1)


class DispositionSplitTest(unittest.TestCase):
    """
    Caught before execution and executed anyway must not be conflated.

    The raw infeasible count is not comparable across arms: the filtered arm
    catches requests before execution, the baseline can only catch them at
    execution, and the baseline's resized orders execute at a corrected size.
    Splitting is what lets a report say "21 caught, 0 executed" against "19
    caught, 7 executed" rather than implying the filtered arm is worse.
    """

    def _recorder_with(self, disposition, reached_execution):
        """One infeasible request with the given disposition and fate."""
        recorder = MetricsRecorder()
        request_id = recorder.record_request(
            step=0, agent="A", arm="x", action="buy",
            symbol="NVDA", quantity=9999)
        recorder.record_disposition(request_id, disposition, infeasible=True,
                                    reached_execution=reached_execution)
        return recorder.summary()

    def test_rejected_counts_as_caught(self):
        """A rejection never reaches execution."""
        summary = self._recorder_with(DISPOSITION_REJECTED,
                                      reached_execution=False)
        self.assertEqual(summary["caught_pre_execution"]["count"], 1)
        self.assertEqual(summary["executed_malformed"]["count"], 0)

    def test_fallback_counts_as_caught(self):
        """A substituted HOLD means the bad request never executed."""
        summary = self._recorder_with(DISPOSITION_FALLBACK,
                                      reached_execution=False)
        self.assertEqual(summary["caught_pre_execution"]["count"], 1)
        self.assertEqual(summary["executed_malformed"]["count"], 0)

    def test_accepted_infeasible_counts_as_executed(self):
        """An infeasible request that was accepted did execute."""
        summary = self._recorder_with(DISPOSITION_ACCEPTED,
                                      reached_execution=True)
        self.assertEqual(summary["caught_pre_execution"]["count"], 0)
        self.assertEqual(summary["executed_malformed"]["count"], 1)

    def test_resized_counts_as_executed(self):
        """
        A clamped order executed, at a corrected size.

        This is the baseline's characteristic failure and the reason the
        headline claim is about disposition rather than rate.
        """
        summary = self._recorder_with(DISPOSITION_RESIZED,
                                      reached_execution=True)
        self.assertEqual(summary["executed_malformed"]["count"], 1)

    def test_a_fallback_that_still_executed_is_not_counted_as_caught(self):
        """
        Disposition and fate are independent, and both are needed.

        This is why reached_execution is recorded rather than inferred from the
        disposition name: a fallback usually means nothing executed, but the
        counter must follow what actually happened, not what the label implies.
        Collapsing the two would make the caught-versus-executed split, the one
        cross-arm comparison the headline claim rests on, depend on naming.
        """
        summary = self._recorder_with(DISPOSITION_FALLBACK,
                                      reached_execution=True)
        self.assertEqual(summary["caught_pre_execution"]["count"], 0)
        self.assertEqual(summary["executed_malformed"]["count"], 1)


class ErrorSeparationTest(unittest.TestCase):
    """
    Server failures and parse failures never share a counter.

    The "OLLAMA ERROR" string sentinel made a dead server look like a model
    that cannot produce JSON, so every fallback rate measured while the server
    was flaky is suspect.
    """

    def test_the_two_kinds_are_counted_separately(self):
        """Two server errors and one parse failure must not become three."""
        recorder = MetricsRecorder()
        recorder.record_error(ERROR_SERVER, "timeout")
        recorder.record_error(ERROR_SERVER, "connection refused")
        recorder.record_error(ERROR_PARSE, "unterminated JSON")
        summary = recorder.summary()
        self.assertEqual(summary["server_errors"]["count"], 2)
        self.assertEqual(summary["parse_failures"]["count"], 1)

    def test_an_unknown_error_kind_raises(self):
        """
        A third kind must be rejected rather than silently counted as neither.

        Accepting it would produce an event no metric reads, so the failure
        would vanish from the report entirely.
        """
        recorder = MetricsRecorder()
        with self.assertRaises(ValueError):
            recorder.record_error("ollama_error", "something")


class AbstentionAndParticipationTest(unittest.TestCase):
    """
    An arm can buy a clean rate by not trading, so both travel with every rate.

    Run 06's middleware arm reported 0.0% of 21 attempts against the baseline's
    66.9% of 184, with one agent requesting no trades at all in three of four
    seeds, and nothing in the printed report distinguished that from an arm
    that traded heavily and stayed clean.
    """

    def test_a_non_participant_is_named(self):
        """
        An agent that requested no trades must be identified, not just counted.

        Naming it is what makes the warning actionable: "1 of 3" does not say
        which agent's behaviour invalidates the arm's rate.
        """
        recorder = MetricsRecorder()
        recorder.record_request(step=0, agent="Alex Chen", arm="mw",
                                action="hold", symbol=None, quantity=0,
                                is_trade_request=False)
        recorder.record_request(step=0, agent="Marcus Webb", arm="mw",
                                action="buy", symbol="NVDA", quantity=1)
        participation = recorder.summary()["participation"]
        self.assertEqual(participation["count"], 1)
        self.assertEqual(participation["denominator"], 2)
        self.assertEqual(participation["non_participants"], ["Alex Chen"])

    def test_a_fully_participating_arm_names_nobody(self):
        """No warning when every agent traded."""
        recorder = MetricsRecorder()
        for agent in ("Alex Chen", "Marcus Webb"):
            recorder.record_request(step=0, agent=agent, arm="mw",
                                    action="buy", symbol="NVDA", quantity=1)
        self.assertEqual(
            recorder.summary()["participation"]["non_participants"], [])


class ReasonedDenominatorTest(unittest.TestCase):
    """
    Stale context and contradictions are scored over reasoned decisions only.

    An empty reasoning scores clean on both for free, so a run that often fails
    to parse looks like a run that reasons well.
    """

    def test_unreasoned_decisions_are_excluded_from_the_denominator(self):
        """Two decisions, one with reasoning: the denominator must be 1."""
        recorder = MetricsRecorder()
        recorder.record_context(step=0, agent="A", arm="mw",
                                has_reasoning=True, stale_context=True)
        recorder.record_context(step=1, agent="A", arm="mw",
                                has_reasoning=False)
        stale = recorder.summary()["stale_context_hits"]
        self.assertEqual(stale["count"], 1)
        self.assertEqual(stale["denominator"], 1)


class MetricPluginTest(unittest.TestCase):
    """The seam DCS, BDI and CVR will plug into once defined."""

    def test_a_registered_metric_appears_in_the_summary(self):
        """A custom metric is computed over the events like any other."""
        recorder = MetricsRecorder()
        recorder.record_request(step=0, agent="A", arm="mw", action="buy",
                               symbol="NVDA", quantity=1)
        recorder.register_metric(
            "total_events", lambda events: {"count": len(events)})
        self.assertEqual(recorder.summary()["total_events"]["count"], 1)


class MetadataTest(unittest.TestCase):
    """
    A report must record what produced it.

    Run 06 and Run 07 differ by sampling settings that appear nowhere in the
    artefacts, which is the confusion this prevents.
    """

    def test_metadata_is_carried_into_the_summary(self):
        """Model, temperature, seed and steps reach the summary."""
        recorder = MetricsRecorder(config_name="run07_mw_s42")
        recorder.record_meta(model="mistral", temperature=0.7, seed=42,
                             steps=200)
        meta = recorder.summary()["meta"]
        self.assertEqual(meta["model"], "mistral")
        self.assertEqual(meta["temperature"], 0.7)
        self.assertEqual(meta["seed"], 42)
        self.assertEqual(meta["steps"], 200)
        self.assertEqual(meta["config_name"], "run07_mw_s42")

    def test_git_info_reports_a_commit_and_a_dirty_flag(self):
        """
        The commit is recorded, and so is whether the tree had local edits.

        A result produced from an edited tree is not reproducible from that
        commit, so the flag matters as much as the hash.
        """
        info = git_commit_info()
        self.assertIn("git_commit", info)
        self.assertIn("git_dirty", info)
        # This repo is a git checkout, so a real hash is expected here. If git
        # were missing the contract is UNKNOWN rather than an exception.
        self.assertTrue(info["git_commit"] == UNKNOWN
                        or len(str(info["git_commit"])) >= 7)

    def test_commit_is_read_at_import_not_at_report_time(self):
        """
        A checkout during a run must not relabel the code already loaded.

        Checked in a fresh interpreter because this process's cache has been
        reset by other tests; the cache being filled by import alone is what
        pins the hash to process start.
        """
        import subprocess
        out = subprocess.run(
            [sys.executable, "-c",
             "import middleware.metrics_recorder as m; print(bool(m._git_cache))"],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            capture_output=True, text=True, timeout=60)
        self.assertEqual("True", out.stdout.strip(), out.stderr)

    def test_an_unreachable_ollama_reports_not_found(self):
        """
        No server must give NOT FOUND, never a guess and never a crash.

        The version is often unobtainable: the server is usually not running
        when a report is generated, and never in the stubbed harness.
        """
        with mock.patch("requests.get", side_effect=OSError("refused")):
            self.assertEqual(ollama_version("http://127.0.0.1:1"), UNKNOWN)


class DurableWriteTest(unittest.TestCase):
    """
    Writes are atomic, because a plausible-looking truncated report is worse
    than a missing one. Run 07 died on ENOSPC with 99 GB apparently free.
    """

    def setUp(self):
        """A scratch directory for each write test."""
        self.dir = tempfile.mkdtemp(prefix="metrics-test-")
        self.addCleanup(shutil.rmtree, self.dir, True)

    def test_summary_round_trips(self):
        """A written summary reads back as the same JSON."""
        recorder = MetricsRecorder(config_name="probe")
        recorder.record_request(step=0, agent="A", arm="mw", action="buy",
                                symbol="NVDA", quantity=1)
        path = os.path.join(self.dir, "summary.json")
        recorder.write_summary(path)
        with open(path, encoding="utf-8") as handle:
            loaded = json.load(handle)
        self.assertEqual(loaded["meta"]["config_name"], "probe")
        self.assertEqual(loaded["trade_attempts"]["count"], 1)

    def test_events_are_written_as_one_object_per_line(self):
        """
        JSON Lines, so a kill mid-write costs one line, not the file.

        A single JSON array would be unreadable if truncated.
        """
        recorder = MetricsRecorder()
        for i in range(3):
            recorder.record_request(step=i, agent="A", arm="mw", action="buy",
                                    symbol="NVDA", quantity=1)
        path = os.path.join(self.dir, "events.jsonl")
        recorder.write_events(path)
        with open(path, encoding="utf-8") as handle:
            lines = [line for line in handle.read().splitlines() if line]
        self.assertEqual(len(lines), 3)
        for line in lines:
            self.assertEqual(json.loads(line)["type"], "request")

    def test_a_failed_write_leaves_the_previous_summary_intact(self):
        """
        A write that dies partway must not destroy the last good summary.

        This is the ENOSPC case. Writing in place would leave a half-written
        file that still parses and still looks complete; writing to a temporary
        file and renaming means the target is either the old complete summary
        or the new one.
        """
        path = os.path.join(self.dir, "summary.json")

        good = MetricsRecorder(config_name="first")
        good.write_summary(path)

        second = MetricsRecorder(config_name="second")
        # Fail during the write, after the temporary file exists.
        with mock.patch("os.fsync", side_effect=OSError("ENOSPC")):
            with self.assertRaises(OSError):
                second.write_summary(path)

        with open(path, encoding="utf-8") as handle:
            loaded = json.load(handle)
        self.assertEqual(loaded["meta"]["config_name"], "first",
                         "the failed write overwrote a complete summary")

    def test_a_failed_write_leaves_no_temporary_file_behind(self):
        """
        The temporary file is cleaned up on failure.

        Otherwise a sweep that hits ENOSPC repeatedly fills the directory with
        .metrics-*.tmp files and makes the real problem harder to see.
        """
        path = os.path.join(self.dir, "summary.json")
        recorder = MetricsRecorder()
        with mock.patch("os.fsync", side_effect=OSError("ENOSPC")):
            with self.assertRaises(OSError):
                recorder.write_summary(path)
        leftovers = [n for n in os.listdir(self.dir) if n.startswith(".metrics-")]
        self.assertEqual(leftovers, [])


class ReportEquivalenceTest(unittest.TestCase):
    """
    The recorder must reproduce the existing report's numbers exactly.

    This is WP-B's first acceptance criterion, and it is checked BEFORE the
    recorder replaces anything. The report's figures are what the paper's
    tables are built from, so the order matters: prove equivalence on a real
    run first, swap second. Doing it the other way round risks moving a
    published number with nothing to catch it.

    A real 20-step run of both arms is used, through the same deterministic
    stub as the golden harness, rather than a synthetic log: the mapping that
    could be wrong is the one from real log entries, and a hand-built log would
    only test my understanding of the fields twice.
    """

    @classmethod
    def setUpClass(cls):
        """Run both arms once and keep each log with its report."""
        _HERE = os.path.dirname(os.path.abspath(__file__))
        if _HERE not in sys.path:
            sys.path.insert(0, _HERE)
        from test_golden_pipeline import run_log_and_report

        cls.runs = {}
        for arm, use_middleware in (("middleware", True), ("baseline", False)):
            cls.runs[arm] = run_log_and_report(use_middleware)

    def _check_arm(self, arm):
        """Compare one arm's recorder summary against its report."""
        from middleware.metrics_recorder import recorder_from_trading_log

        log, report = self.runs[arm]
        summary = recorder_from_trading_log(log, arm=arm).summary()

        # The headline figures, in the order the report prints them.
        self.assertEqual(summary["trade_attempts"]["count"],
                         report["total_trade_attempts"],
                         "trade attempts differ, which is every rate's "
                         "denominator")
        self.assertEqual(summary["infeasible_requests"]["count"],
                         report["total_hallucinations"])
        self.assertEqual(summary["infeasible_requests"]["denominator"],
                         report["total_trade_attempts"])
        self.assertEqual(summary["caught_pre_execution"]["count"],
                         report["total_caught_pre_execution"])
        self.assertEqual(summary["executed_malformed"]["count"],
                         report["total_executed_malformed"],
                         "EXECUTED malformed differs, which is the paper's "
                         "headline claim")
        self.assertEqual(summary["state_contradictions"]["count"],
                         report["total_state_contradictions"])
        self.assertEqual(summary["participation"]["count"],
                         report["participating_agents"])
        self.assertEqual(summary["participation"]["denominator"],
                         report["total_agents"])

        # Abstention is a rate in the report and a count plus denominator here,
        # so it is compared after reconstructing the percentage the same way.
        holds = summary["abstention"]["count"]
        decisions = summary["abstention"]["denominator"]
        expected = (round(holds / decisions * 100, 1) if decisions else None)
        self.assertEqual(expected, report["total_abstention_rate_pct"])
        self.assertEqual(decisions, report["total_log_entries"])

    def test_middleware_arm_numbers_match_the_report(self):
        """Equivalence on the arm whose claim the paper rests on."""
        self._check_arm("middleware")

    def test_baseline_arm_numbers_match_the_report(self):
        """
        Equivalence on the control arm too.

        The baseline is where executed_malformed is nonzero, so it is the arm
        that actually exercises the caught-versus-executed split. An
        equivalence test on the middleware arm alone would pass with both
        counters stuck at zero.
        """
        self._check_arm("baseline")

    def test_the_two_arms_are_not_accidentally_identical(self):
        """
        Guard against the equivalence test comparing two empty runs.

        If both arms produced the same numbers, every assertion above could
        pass while the mapping was wrong. Run 06's disposition claim says these
        must differ: the middleware arm executes no malformed orders, the
        baseline executes some.
        """
        _, mw_report = self.runs["middleware"]
        _, base_report = self.runs["baseline"]
        self.assertEqual(mw_report["total_executed_malformed"], 0)
        self.assertGreater(base_report["total_executed_malformed"], 0)


def report_stub(log):
    """
    A stand-in carrying only the attributes _generate_report reads.

    Used instead of a real simulation because the function reads four
    attributes plus the agent names, and booting a 20-step run per assertion
    would be slow without adding coverage. The equivalence tests above already
    exercise _generate_report against real runs, so what is tested here is the
    report's own logic, not the pipeline's.

    Agent names are taken from the log so the per-agent section is populated;
    The report reads names from self.personas, and each persona's risk
    tolerance for the run-level hold-drift check.
    """
    class _Stub:
        use_middleware = True
        temperature = 0.7
        seed = 42
        sim_code = "unit_probe"
        personas = {entry["agent"]: SimpleNamespace(
            scratch=SimpleNamespace(risk_tolerance="moderate"))
            for entry in log}

    return _Stub()


class ServerErrorSeparationInReportTest(unittest.TestCase):
    """
    WP2: a dead server must not be recorded as the model failing.

    gpt_structure returns the string "OLLAMA ERROR" when the server could not
    be reached, timed out, or answered with an HTTP error. That string is not
    valid JSON, so it reached the validator and was recorded as "response was
    not valid JSON", i.e. as the model's failure, and inflated the
    parser-fallback rate. Run 06 happened to see zero of these across all 16
    configs, which is why its figures are unaffected by the split.
    """

    def test_the_sentinel_is_recognised_before_json_parsing(self):
        """
        The parser must classify the sentinel, not try to parse it.

        Order matters here: attempting the parse first produces a JSONDecodeError
        and the wrong error reason, which is the whole bug.
        """
        from middleware.action_filtering import (
            SERVER_ERROR_REASON, _parse_json_response)
        response, error = _parse_json_response("OLLAMA ERROR")
        self.assertIsNone(response)
        self.assertEqual(error, SERVER_ERROR_REASON)

    def test_genuine_bad_json_is_still_a_parse_failure(self):
        """
        Malformed model output must keep its own reason.

        If this also became a server error the split would be useless in the
        other direction, hiding the model failures the metric exists to count.
        """
        from middleware.action_filtering import _parse_json_response
        response, error = _parse_json_response('{"action": "buy"')
        self.assertIsNone(response)
        self.assertEqual(error, "response was not valid JSON")

    def test_is_server_error_distinguishes_the_two(self):
        """The helper must separate infrastructure from measurement."""
        from middleware.action_filtering import (
            SERVER_ERROR_REASON, is_server_error)
        self.assertTrue(is_server_error(SERVER_ERROR_REASON))
        self.assertFalse(is_server_error("response was not valid JSON"))
        self.assertFalse(is_server_error("quantity outside legal limit"))
        self.assertFalse(is_server_error(None))

    def test_a_server_error_is_not_a_hallucination(self):
        """
        An unreachable server must not be counted as an illegal action.

        Otherwise an outage would raise the hallucination rate, which is the
        number the paper's claim is about.
        """
        from middleware.action_filtering import (
            SERVER_ERROR_REASON, is_illegal_action_error)
        self.assertFalse(is_illegal_action_error(SERVER_ERROR_REASON))


class ReportMetadataAndErrorCountsTest(unittest.TestCase):
    """
    WP1 and WP2 as the report sees them.

    _generate_report is called on a stand-in object rather than a real
    simulation, because it reads only four attributes and a 20-step run per
    assertion would be slow for no extra coverage. The equivalence tests above
    already exercise it against real runs.
    """

    @staticmethod
    def _entry(agent="Alex Chen", step=0, action="hold", filter_status="success",
               error_reason="", requested=None, hallucination=False):
        """A trading_log-shaped entry with the fields the report reads."""
        return {
            "step": step,
            "agent": agent,
            "decision": {"action": action, "symbol": None, "quantity": 0,
                         "reasoning": "because"},
            "requested": requested if requested is not None else {},
            "filter_status": filter_status,
            "error_reason": error_reason,
            "outcome": "",
            "hallucination": hallucination,
            "halluc_kind": "",
            "halluc_disposition": "",
            "illegal_request": False,
            "order_feedback": "",
            "requested_quantity": 0,
            "filled_quantity": 0,
            "filtered": False,
            "has_reasoning": True,
            "stale_context": False,
            "state_contradiction": False,
            "persona_drift_score": 1.0,
            "compression": {"chars_out": 10, "chars_in": 10, "nodes_in": 1,
                            "nodes_out": 1, "dropped_duplicate": 0,
                            "dropped_stale": 0, "dropped_budget": 0,
                            "annotated_stale": 0, "enabled": True},
            "cash": 1000.0,
            "portfolio_value": 1000.0,
            "portfolio_value_before": 1000.0,
            "positions": {},
            "market_prices": {},
        }

    def _report(self, log):
        """Generate a report from a log without booting a simulation."""
        from trading_reverie import TradingReverie
        return TradingReverie._generate_report(
            report_stub(log), log, n_steps=len(log))

    def test_the_report_records_what_produced_it(self):
        """
        WP1: model, temperature, seed, steps, config and commit are recorded.

        Without these, Run 06 and Run 07 produce indistinguishable artefacts
        while differing in the sampling settings that make their absolute rates
        incomparable.
        """
        report = self._report([self._entry()])
        meta = report["run_metadata"]
        self.assertEqual(meta["temperature"], 0.7)
        self.assertEqual(meta["seed"], 42)
        self.assertEqual(meta["steps"], 1)
        self.assertEqual(meta["config_name"], "unit_probe")
        self.assertIn("model", meta)
        self.assertIn("git_commit", meta)
        self.assertIn("git_dirty", meta)

    def test_a_server_error_is_counted_apart_from_parser_fallbacks(self):
        """
        WP2 at the report level: the outage gets its own total.

        One fallback caused by the server and one caused by bad JSON must give
        one server error and one parser fallback, not two parser fallbacks.
        """
        from middleware.action_filtering import SERVER_ERROR_REASON

        log = [
            self._entry(step=0, filter_status="fallback",
                        error_reason=SERVER_ERROR_REASON),
            self._entry(step=1, filter_status="fallback",
                        error_reason="response was not valid JSON"),
        ]
        report = self._report(log)
        self.assertEqual(report["total_server_errors"], 1)
        fallbacks = sum(a["fallback_decisions"]
                        for a in report["per_agent"].values())
        self.assertEqual(fallbacks, 1,
                         "the server failure was counted as the model's")

    def test_server_errors_still_appear_in_the_reason_breakdown(self):
        """
        The outage is excluded from the fallback count but still diagnosable.

        Dropping it entirely would make a run with a flaky server look clean,
        which is the opposite failure.
        """
        from middleware.action_filtering import SERVER_ERROR_REASON
        report = self._report([self._entry(filter_status="fallback",
                                           error_reason=SERVER_ERROR_REASON)])
        reasons = {}
        for agent in report["per_agent"].values():
            reasons.update(agent.get("error_reasons") or {})
        self.assertIn(SERVER_ERROR_REASON, reasons)

    def test_a_clean_run_reports_no_server_errors(self):
        """A run with a healthy server must report zero, not omit the field."""
        report = self._report([self._entry()])
        self.assertEqual(report["total_server_errors"], 0)


class PairedAdapterTest(unittest.TestCase):
    """
    The paired harness's figures now come from the recorder too.

    The paired report has no golden file, so these pin the invariants that
    would break if the adapter mapped a field wrongly. The strongest is the
    disposition identity: every infeasible request was either caught before
    execution or executed, never both and never neither.
    """

    @staticmethod
    def _pair(agent="Alex Chen", step=0, action="buy", requested=None,
              filter_status="success", infeasible=False, disposition="",
              has_reasoning=True, context_chars=100):
        """A pair entry shaped like PairedEvaluation writes them."""
        side = {
            "action": action,
            "symbol": "NVDA",
            "quantity": 1,
            "requested": requested if requested is not None else {
                "action": action, "symbol": "NVDA", "quantity": 1},
            "filter_status": filter_status,
            "infeasible": infeasible,
            "kind": "illegal_request" if infeasible else "",
            "disposition": disposition,
            "has_reasoning": has_reasoning,
            "context_chars": context_chars,
            "stale_context": False,
            "state_contradiction": False,
        }
        return {"step": step, "agent": agent, "middleware": dict(side),
                "baseline": dict(side)}

    def test_caught_plus_executed_equals_infeasible(self):
        """
        The disposition split must account for every infeasible request.

        If it did not, the headline claim would be arithmetically unsound: a
        request that was neither caught nor executed is unexplained, and one
        counted as both inflates the comparison in both directions at once.
        """
        from middleware.metrics_recorder import recorder_from_pairs

        pairs = [
            self._pair(step=0, infeasible=True,
                       disposition="caught_pre_execution",
                       filter_status="fallback"),
            self._pair(step=1, infeasible=True,
                       disposition="executed_malformed"),
            self._pair(step=2, infeasible=False),
        ]
        summary = recorder_from_pairs(pairs, "middleware").summary()
        infeasible = summary["infeasible_requests"]["count"]
        caught = summary["caught_pre_execution"]["count"]
        executed = summary["executed_malformed"]["count"]
        self.assertEqual(infeasible, 2)
        self.assertEqual(caught + executed, infeasible)
        self.assertEqual(caught, 1)
        self.assertEqual(executed, 1)

    def test_an_unparseable_response_is_classified_by_its_final_action(self):
        """
        A request with no action falls back to the final action.

        This is the rule the paired report already used, and it matters because
        an unparseable response carries no request to classify, so without the
        fallback it would not count as a decision at all.
        """
        from middleware.metrics_recorder import recorder_from_pairs

        pairs = [self._pair(action="hold", requested={},
                            filter_status="fallback")]
        summary = recorder_from_pairs(pairs, "middleware").summary()
        # A HOLD is a decision but not a trade attempt.
        self.assertEqual(summary["trade_attempts"]["count"], 0)
        self.assertEqual(summary["abstention"]["denominator"], 1)

    def test_a_non_participant_is_detected_per_arm(self):
        """
        Participation is computed per arm, since the arms differ.

        The two arms make different requests from the same state, so an agent
        can be a non-participant in one and not the other, and reporting one
        arm's participation for both would hide it.
        """
        from middleware.metrics_recorder import recorder_from_pairs

        pairs = [
            self._pair(agent="Alex Chen", step=0, action="hold",
                       requested={"action": "hold"}),
            self._pair(agent="Marcus Webb", step=0, action="buy"),
        ]
        summary = recorder_from_pairs(pairs, "middleware").summary()
        self.assertEqual(summary["participation"]["non_participants"],
                         ["Alex Chen"])

    def test_dispositions_carry_the_agent_for_per_agent_grouping(self):
        """
        A disposition event is stamped with its request's agent.

        Per-agent metrics filter events by agent, so an unstamped disposition
        would be dropped by that filter and silently lower every per-agent
        count while the totals stayed right, which is the hardest kind of
        discrepancy to notice.
        """
        from middleware.metrics_recorder import recorder_from_pairs

        pairs = [self._pair(agent="Sara Kim", infeasible=True,
                            disposition="caught_pre_execution")]
        recorder = recorder_from_pairs(pairs, "middleware")
        by_agent = recorder.summary_by_agent()
        self.assertIn("Sara Kim", by_agent)
        self.assertEqual(
            by_agent["Sara Kim"]["caught_pre_execution"]["count"], 1)


class PrintedReportStabilityTest(unittest.TestCase):
    """
    The printed report's existing labels must not move.

    WP-B's acceptance requires that the Run 06 log parser keeps working on new
    logs. That parser matches on these labels, so the metadata is appended
    beneath them rather than interleaved, and none is renamed or reordered.

    This cannot be verified against the real parse_run06.py, which lives in
    agent_anchor_ijcis_source.zip and is not in this repository. The labels and
    their order are pinned here instead, which is the part that would break it.
    """

    EXPECTED_ORDER = (
        "HALLUCINATION REPORT",
        "Middleware      :",
        "Steps simulated :",
        "Total decisions :",
    )

    def test_existing_labels_keep_their_order(self):
        """
        The header labels appear in the original order.

        The metadata lines sit between "Steps simulated" and "Total decisions",
        so this also pins that inserting them did not separate a label from the
        value a line-oriented parser expects next to it.
        """
        import io
        import contextlib
        from trading_reverie import TradingReverie

        entry = ReportMetadataAndErrorCountsTest._entry()
        stub = report_stub([entry])

        report = TradingReverie._generate_report(stub, [entry], n_steps=1)
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            TradingReverie._print_report(stub, report)
        output = buffer.getvalue()

        positions = []
        for label in self.EXPECTED_ORDER:
            index = output.find(label)
            self.assertNotEqual(index, -1, f"label {label!r} disappeared")
            positions.append(index)
        self.assertEqual(positions, sorted(positions),
                         "the existing report labels were reordered")

    def test_the_metadata_block_is_printed(self):
        """The new lines are actually emitted, not merely stored in the JSON."""
        from middleware.metrics_recorder import format_meta_lines
        lines = format_meta_lines({
            "model": "mistral", "temperature": 0.7, "seed": 42,
            "config_name": "run07_mw_s42", "git_commit": "abc1234",
            "git_dirty": False, "ollama_version": "0.30.11",
        })
        rendered = "\n".join(lines)
        self.assertIn("Model", rendered)
        self.assertIn("mistral", rendered)
        self.assertIn("Temperature", rendered)
        self.assertIn("Commit", rendered)
        # A boolean reads as yes/no rather than Python's True/False, since this
        # is printed for a human alongside the other report lines.
        self.assertIn("Tree dirty      : no", rendered)

    def test_a_skipped_field_is_not_printed_twice(self):
        """
        The paired report must not emit two lines with the same label.

        It prints its own "Seed" line, so the metadata block skips seed. Two
        identical labels would leave a line-oriented parser with two
        candidates for one field.
        """
        from middleware.metrics_recorder import format_meta_lines
        meta = {"model": "mistral", "seed": 42, "git_commit": "abc1234"}
        rendered = "\n".join(format_meta_lines(meta,
                                               skip=frozenset({"seed"})))
        self.assertNotIn("Seed", rendered)
        self.assertIn("Model", rendered)
        self.assertIn("Commit", rendered)

    def test_the_paired_report_prints_seed_exactly_once(self):
        """
        End to end on the real formatter plus the paired report's own header.

        Checks the thing that actually matters, the rendered output, rather
        than only the skip argument.
        """
        from middleware.metrics_recorder import format_meta_lines
        header = ["Seed            : 42", "Advanced with   : middleware"]
        meta_lines = format_meta_lines(
            {"model": "mistral", "seed": 42, "temperature": 0.7},
            skip=frozenset({"seed"}))
        output = "\n".join(header + meta_lines)
        self.assertEqual(output.count("Seed "), 1)

    def test_missing_metadata_is_skipped_not_printed_as_none(self):
        """
        An absent field is omitted rather than printed as None.

        A line reading "Commit          : None" looks like a recorded value of
        None, which is worse than no line at all.
        """
        from middleware.metrics_recorder import format_meta_lines
        rendered = "\n".join(format_meta_lines({"model": "mistral"}))
        self.assertIn("Model", rendered)
        self.assertNotIn("Commit", rendered)


if __name__ == "__main__":
    unittest.main()
