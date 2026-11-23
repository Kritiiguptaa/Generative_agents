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


if __name__ == "__main__":
    unittest.main()
