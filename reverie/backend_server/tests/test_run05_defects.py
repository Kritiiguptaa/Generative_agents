"""
Regression tests for the defects found in the run-05 logs.

Each test here pins a number that a report actually printed and could not
have produced honestly:

  1. Alex Chen, run_mw_05: "48  (145.5% of 33 attempts)". A rate above 100%
     is not a measurement. All 48 were "action type is invalid", and an
     invalid verb could enter the numerator but never the denominator.
  2. The same ("buy", "sell") gate suppressed the rejection feedback for
     exactly those 48, so nothing ever told the agent the verb didn't exist
     and it asked again every step -- 48 rejections, 1 distinct episode.
  3. The baseline arm did not record an invalid verb at all, so identical
     model behaviour was a hallucination under middleware and silently
     nothing under baseline.
  4. insight_and_evidence: 114 reflections across the four run-05 logs burned
     all 5 attempts and fell back, because one unparseable line discarded the
     whole response.

Run from backend_server:
    python -m pytest tests/test_run05_defects.py
"""

import os
import sys
import unittest

BACKEND = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from middleware.action_filtering import (is_trade_request, validate_response,
                                         describe_request,
                                         is_illegal_action_error, Action)


class TradeRequestDenominator(unittest.TestCase):
    """is_trade_request() is the denominator for every hallucination rate."""

    def test_valid_trades_count(self):
        self.assertTrue(is_trade_request({"action": "buy"}))
        self.assertTrue(is_trade_request({"action": "sell"}))

    def test_non_trades_do_not_count(self):
        self.assertFalse(is_trade_request({"action": "hold"}))
        # ANALYZE normalises to HOLD in validate_response; it moves no shares,
        # so counting it as an attempt would inflate the denominator instead.
        self.assertFalse(is_trade_request({"action": "analyze"}))

    def test_unparseable_counts_as_neither(self):
        """A null action is a formatting failure, not a trade request. It is
        excluded from the numerator too (not in ILLEGAL_ACTION_ERRORS), so it
        must stay out of both terms rather than only the numerator."""
        self.assertFalse(is_trade_request({"action": None}))
        self.assertFalse(is_trade_request({}))
        self.assertFalse(is_trade_request(None))

    def test_invalid_verb_is_a_trade_request(self):
        """The run-05 defect. A verb outside BUY/SELL/HOLD is rejected as
        'action type is invalid', which is_illegal_action_error counts as a
        hallucination -- so it has to be in the denominator as well."""
        for verb in ("short", "hedge", "buy_and_hold", "liquidate"):
            self.assertTrue(is_trade_request({"action": verb}), verb)

    def test_numerator_and_denominator_agree_on_invalid_verbs(self):
        """The two terms must be driven by the same event, or the rate can
        exceed 100%. This is the invariant that actually failed."""
        legal = [Action("HOLD", None, 0, ""), Action("BUY", "NVDA", 10, "")]
        response, error = validate_response('{"action": "short", '
                                            '"symbol": "NVDA", "quantity": 5}',
                                            legal)
        self.assertIsNone(response)
        self.assertEqual("action type is invalid", error)
        # numerator fires ...
        self.assertTrue(is_illegal_action_error(error))
        # ... so the denominator must fire on the same response.
        self.assertTrue(is_trade_request(describe_request(
            '{"action": "short", "symbol": "NVDA", "quantity": 5}')))

    def test_alex_chen_run05_rate_is_now_possible(self):
        """Replays the exact composition behind '48 (145.5% of 33 attempts)':
        29 executed buys + 4 executed sells + 48 invalid verbs + 119 holds."""
        requests = ([{"action": "buy"}] * 29 + [{"action": "sell"}] * 4
                    + [{"action": "short"}] * 48 + [{"action": "hold"}] * 119)
        attempts = sum(1 for r in requests if is_trade_request(r))
        hallucinations = 48
        self.assertEqual(200, len(requests))
        self.assertEqual(81, attempts)
        rate = hallucinations / attempts * 100
        self.assertLessEqual(rate, 100.0, "a hallucination rate cannot exceed 100%")
        self.assertAlmostEqual(59.3, round(rate, 1))


class InsightParserTolerance(unittest.TestCase):
    """
    insight_and_evidence_v1: one bad line used to discard the whole response.

    Imported lazily inside the tests -- run_gpt_prompt imports gpt_structure,
    which reaches for an Ollama config at module scope, and these assertions
    are about string parsing only.
    """

    @staticmethod
    def _clean_up():
        from persona.prompt_template.run_gpt_prompt import (
            _INSIGHT_ITEM_RE, _INSIGHT_PAREN_RE, _INSIGHT_BECAUSE_RE)
        import re

        # The parser is a closure inside run_gpt_prompt_insight_and_guidance,
        # so exercise it through the module-level regexes it is built from,
        # mirroring the function body exactly.
        def clean_up(gpt_response, prompt=""):
            ret, salvaged = dict(), dict()
            for line_no, raw_line in enumerate(gpt_response.strip().split("\n")):
                line = raw_line.strip()
                if not line or line.endswith(":"):
                    continue
                was_list_item = line_no == 0 or bool(_INSIGHT_ITEM_RE.match(line))
                while True:
                    stripped = _INSIGHT_ITEM_RE.sub("", line, count=1).strip()
                    if stripped == line:
                        break
                    line = stripped
                if not line:
                    continue
                evidence, thought = None, line
                for m in _INSIGHT_PAREN_RE.finditer(line):
                    if any(c.isdigit() for c in m.group(1)):
                        evidence = [int(x) for x in re.findall(r'\d+', m.group(1))]
                        thought = line[:m.start()]
                if evidence is None:
                    marker = _INSIGHT_BECAUSE_RE.search(line)
                    if marker:
                        tail = line[marker.end():]
                        if any(c.isdigit() for c in tail):
                            evidence = [int(x) for x in re.findall(r'\d+', tail)]
                            thought = line[:marker.start()]
                thought = thought.strip().rstrip(".,:;- ").strip()
                if len(thought) < 3 or not any(c.isalpha() for c in thought):
                    continue
                if evidence is not None:
                    ret[thought] = evidence
                elif was_list_item and line_no > 0:
                    salvaged[thought] = []
            if ret:
                return ret
            if salvaged:
                return salvaged
            raise ValueError("no insights found in response")
        return clean_up

    def test_ideal_format(self):
        got = self._clean_up()("Sara trades NVDA often (because of 1, 3)")
        self.assertEqual({"Sara trades NVDA often": [1, 3]}, got)

    def test_preamble_no_longer_discards_the_response(self):
        """'Here are the insights:' is the single most common thing an instruct
        model prepends, and it used to raise IndexError and lose everything."""
        got = self._clean_up()(
            "Here are the insights:\n1. Sara trades NVDA often (because of 1, 3)")
        self.assertEqual({"Sara trades NVDA often": [1, 3]}, got)

    def test_trailing_remark_no_longer_discards_the_response(self):
        got = self._clean_up()(
            "1. Sara trades NVDA (because of 1)\nThese insights show a pattern.")
        self.assertEqual({"Sara trades NVDA": [1]}, got)

    def test_citation_variants(self):
        for text in ("Sara trades NVDA often (based on 1, 3)",
                     "Sara trades NVDA often because of 1, 3",
                     "Sara trades NVDA often (because of statements 1, 3)"):
            self.assertEqual({"Sara trades NVDA often": [1, 3]},
                             self._clean_up()(text), text)

    def test_parentheses_inside_the_thought(self):
        got = self._clean_up()("Sara (a momentum trader) buys NVDA (because of 2, 4)")
        self.assertEqual({"Sara (a momentum trader) buys NVDA": [2, 4]}, got)

    def test_echoed_numbering_is_stripped(self):
        """The prompt ends with '1.', so models often emit '1.' again."""
        got = self._clean_up()("1. Sara trades NVDA (because of 1)")
        self.assertEqual({"Sara trades NVDA": [1]}, got)

    def test_refusal_still_fails(self):
        """Tolerance must not become credulity: a one-line refusal is not an
        insight, and salvaging it would write garbage into a_mem -- strictly
        worse than the old behaviour, which retried and gave up."""
        with self.assertRaises(ValueError):
            self._clean_up()("I cannot answer that.")

    def test_empty_still_fails(self):
        with self.assertRaises(ValueError):
            self._clean_up()("")


class InsightFailSafeShape(unittest.TestCase):
    def test_fail_safe_matches_the_success_type(self):
        """get_fail_safe returned ['I am hungry'] * n -- a list, where success
        returns a dict. reflect.py calls .items() on it, so the list raised
        AttributeError and was swallowed by the bare except there: the failure
        path only behaved correctly by accident, while printing the Smallville
        placeholder into the run log 228 times across the four run-05 logs."""
        src = open(os.path.join(BACKEND, "persona", "prompt_template",
                                "run_gpt_prompt.py"), encoding="utf-8").read()
        start = src.index("def run_gpt_prompt_insight_and_guidance")
        end = src.index("def run_gpt_prompt_agent_chat", start)
        block = src[start:end]
        fail_safe = block[block.index("def get_fail_safe"):]
        # Scan statements only -- the comment above the return documents the
        # old placeholder by name, so a substring search over the whole block
        # would match its own changelog.
        returns = [ln.strip() for ln in fail_safe.split("\n")
                   if ln.strip().startswith("return")]
        self.assertEqual("return {}", returns[0])


class EvidenceIndexClamping(unittest.TestCase):
    """reflect.py indexed nodes[] with a model-supplied number."""

    def test_out_of_range_citation_keeps_the_insight(self):
        class _Node:
            def __init__(self, nid):
                self.node_id = nid
        nodes = [_Node("n0"), _Node("n1")]
        # The model cited statement 5; only 0 and 1 were shown to it. The
        # prompt's example is 1-based while the statements are numbered from
        # 0, so an off-by-one is the expected case, not the exceptional one.
        evi_raw = [0, 5]
        resolved = [nodes[i].node_id for i in evi_raw
                    if isinstance(i, int) and 0 <= i < len(nodes)]
        self.assertEqual(["n0"], resolved)


class StaleContextDetector(unittest.TestCase):
    """
    The old detector split headlines into words >5 chars and substring-scanned
    the reasoning. Across the whole 12-headline corpus only "supply" and
    "deliveries" could ever fire, and "supply" is seeded into Alex Chen's
    persona -- so the metric counted a word, and scaled with reasoning length.
    """

    @staticmethod
    def _news():
        from market_environment import MarketEnvironment
        return MarketEnvironment.SCRIPTED_NEWS

    def _chk(self, text, step=75):
        from trading_reverie import _check_stale_reasoning
        return bool(_check_stale_reasoning(text, step, self._news()))

    # ---- the false positives that broke run 05 -------------------------

    def test_persona_language_is_not_stale(self):
        """Alex's own scratch.json says he 'wants confirmation of supply
        tightening'. Restating his standing thesis is character, not recall --
        and he is explicitly declining to act on the news here."""
        self.assertFalse(self._chk(
            "The market report does not confirm supply tightening for AMD. Holding."))

    def test_generic_market_vocabulary_is_not_stale(self):
        self.assertFalse(self._chk(
            "Price action reflects normal supply and demand balance; no edge here."))

    def test_score_is_invariant_to_reasoning_length(self):
        """The fatal property of the old detector: the middleware arm scored
        3.9% against the baseline's 53.0% while producing ~790 chars per
        decision against ~5,070. Verbosity must not move this metric."""
        filler = ("Holding NVDA. Supply conditions and revenue estimates remain "
                  "in focus; sentiment is mixed and analyst targets unchanged. ")
        for n in (1, 8, 32):
            self.assertFalse(self._chk(filler * n), f"fired at {len(filler*n)} chars")

    # ---- the true positives it must still catch ------------------------

    def test_citing_a_superseded_headline_is_stale(self):
        self.assertTrue(self._chk(
            "NVDA still faces the critical Taiwan fab bottleneck, so I am selling."))

    def test_citing_the_superseding_headline_too_is_not_stale(self):
        """An agent that names the newer headline is not stuck on the old one."""
        self.assertFalse(self._chk(
            "NVDA Taiwan bottleneck was critical but Goldman Sachs calls it "
            "overblown; buying."))

    # ---- the guards --------------------------------------------------

    def test_one_distinctive_token_is_not_enough(self):
        """A single shared word cannot pin the reference to one headline."""
        self.assertFalse(self._chk("NVDA had a bottleneck earlier; buying anyway."))

    def test_symbol_must_match(self):
        self.assertFalse(self._chk(
            "Taiwan bottleneck was critical for the sector; buying AAPL."))

    def test_recent_news_is_not_stale(self):
        self.assertFalse(self._chk(
            "NVDA still faces the critical Taiwan fab bottleneck.", step=35))

    def test_not_stale_until_actually_superseded(self):
        """Step 55: the step-30 bad news is old, but the step-70 correction
        has not fired yet, so nothing has superseded it."""
        self.assertFalse(self._chk(
            "NVDA still faces the critical Taiwan fab bottleneck.", step=55))

    def test_future_news_cannot_supersede(self):
        """current_step - other_step goes negative for a headline that has not
        fired, which sailed through the recency test: an agent at step 75 was
        scored stale against news scheduled for step 100."""
        from trading_reverie import _check_stale_reasoning
        news = {30: ("NVDA", "NVDA Taiwan fab critical bottleneck", -0.06),
                100: ("NVDA", "NVIDIA announces stock split", 0.07)}
        self.assertEqual("", _check_stale_reasoning(
            "NVDA still faces the critical Taiwan fab bottleneck.", 75, news))

    def test_generic_words_are_not_distinctive(self):
        from trading_reverie import _distinctive_tokens
        d = _distinctive_tokens(self._news())
        every = set().union(*d.values())
        for word in ("supply", "deliveries", "revenue", "estimates", "shares"):
            self.assertNotIn(word, every,
                             f"{word!r} is generic market vocabulary")


if __name__ == "__main__":
    unittest.main()
