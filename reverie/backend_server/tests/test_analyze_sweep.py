"""analyze_sweep.py: parsing, bucketing, and the seed-level arithmetic.

The numbers asserted here are worked out by hand from the synthetic logs, so a
change in how rates are pooled or paired shows up as a failure, not as a table
that merely looks plausible.
"""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import analyze_sweep as A


def block(name, decisions=100, attempts=10, halluc=2, hold=90.0, persona="0.95",
          drift=3, scored=100, buy=10, contra=4, reasoned=100, pnl="+100.00",
          start="100,000", end="100,100", flag=""):
    hall = (f"{halluc}  ({100.0 * halluc / attempts:.1f}% of {attempts} attempts)"
            if attempts else "0  (N/A (no trades requested))")
    return f"""  {name}
    Decisions      : {decisions}  (active={buy}, requested={attempts})
    Action hallucinations : {hall}
      caught=0  executed=0  episodes=0 (0.0%)
    Parser fallbacks      : 0  (0.0% of decisions)
    Decisions w/ reasoning: {reasoned}  (the only ones stale can score)
    Stale context hits    : 0  (0.0% of {reasoned} reasoned)
    State contradictions  : {contra}  ({100.0 * contra / reasoned:.1f}% of {reasoned} reasoned)
    Persona consistency   : avg={persona}  drift events={drift}  (scored {scored})
    Actions        : buy={buy}  sell=0  hold={decisions - buy}  analyze=0
    Abstention     : {hold}% HOLD{flag}
    Portfolio PnL  : ${pnl}  (${start} -> ${end})

"""


def log(*blocks, report=True):
    head = "HALLUCINATION REPORT\n" if report else "still running\n"
    return head + "PER-AGENT\n\n" + "".join(blocks)


class _Sweep:
    """A temp sweep folder plus a temp fork with three bucketed agents."""

    def __init__(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = os.path.join(self.tmp.name, "out")
        self.storage = os.path.join(self.tmp.name, "storage")
        os.makedirs(self.out)
        fork = os.path.join(self.storage, "f")
        names = {"Cara Cons": "conservative", "Mo Mod": "moderate-high", "Ann Aggr": "aggressive"}
        os.makedirs(os.path.join(fork, "reverie"))
        with open(os.path.join(fork, "reverie", "meta.json"), "w") as f:
            json.dump({"persona_names": list(names)}, f)
        for n, tol in names.items():
            d = os.path.join(fork, "personas", n, "bootstrap_memory")
            os.makedirs(d)
            with open(os.path.join(d, "scratch.json"), "w") as f:
                json.dump({"risk_tolerance": tol, "watchlist": ["AAPL", "GOOGL"]}, f)

    def write(self, name, text):
        with open(os.path.join(self.out, name), "w") as f:
            f.write(text)

    def write_sim(self, run, arm, seed, entries):
        d = os.path.join(self.storage, f"{run}_{arm}_s{seed}", "reverie")
        os.makedirs(d)
        with open(os.path.join(d, "trading_log.json"), "w") as f:
            json.dump(entries, f)

    def close(self):
        self.tmp.cleanup()

    def load(self):
        buckets = A.load_buckets(self.storage, "f")
        return A.load_sweep(self.out, buckets)


class ParseLog(unittest.TestCase):
    def test_reads_every_field(self):
        r = A.parse_log(log(block("Ann Aggr", flag="   <- PERSONA DRIFT: aggressive profile, held most of the run")))["Ann Aggr"]
        self.assertEqual((r["decisions"], r["attempts"], r["halluc"]), (100, 10, 2))
        self.assertEqual((r["contra"], r["reasoned"], r["drift"], r["scored"]), (4, 100, 3, 100))
        self.assertEqual((r["buy"], r["hold"]), (10, 90))
        self.assertEqual(r["hold_pct"], 90.0)
        self.assertTrue(r["hold_drift"])
        self.assertEqual((r["pnl"], r["start"], r["end"]), (100.0, 100000.0, 100100.0))

    def test_an_agent_that_never_traded_has_zero_attempts_not_a_rate(self):
        r = A.parse_log(log(block("Cara Cons", attempts=0, halluc=0, buy=0)))["Cara Cons"]
        self.assertEqual((r["halluc"], r["attempts"]), (0, 0))

    def test_unscored_persona_is_none_not_zero(self):
        r = A.parse_log(log(block("Cara Cons", persona="N/A", scored=0, drift=0)))["Cara Cons"]
        self.assertIsNone(r["persona"])


class Buckets(unittest.TestCase):
    def test_bucket_comes_from_the_persona_files_and_moderate_high_is_moderate(self):
        s = _Sweep()
        self.addCleanup(s.close)
        self.assertEqual(A.load_buckets(s.storage, "f"),
                         {"Cara Cons": "conservative", "Mo Mod": "moderate", "Ann Aggr": "aggressive"})


class Aggregation(unittest.TestCase):
    def setUp(self):
        self.s = _Sweep()
        self.addCleanup(self.s.close)

    def test_a_truncated_log_is_skipped_and_said_so(self):
        self.s.write("mw_s42.log", log(block("Ann Aggr"), report=False))
        rows, notes = self.s.load()
        self.assertEqual(rows, [])
        self.assertTrue(any("truncated" in n for n in notes))

    def test_log_without_agent_blocks_is_reported_unparsed(self):
        self.s.write("pmw_s42.log", "PAIRED REPORT\nnothing per agent here\n")
        rows, notes = self.s.load()
        self.assertEqual(rows, [])
        self.assertTrue(any("unparsed" in n for n in notes))

    def test_hallucination_rate_is_pooled_within_a_seed_and_bucket(self):
        # two aggressive agents in one seed: 2/10 and 3/30 -> 5/40 = 12.5%, not the
        # mean of 20% and 10%
        s = self.s
        s.write("mw_s42.log", log(block("Ann Aggr", attempts=10, halluc=2),
                                  block("Cara Cons", attempts=30, halluc=3)))
        rows, _ = s.load()
        for r in rows:
            if r["agent"] == "Cara Cons":
                r["bucket"] = "aggressive"            # put both in one bucket
        m = A.seed_bucket_metrics(rows)[("mw", "aggressive", 42)]
        self.assertAlmostEqual(m["halluc"], 12.5)
        self.assertEqual(m["attempts"], 40)

    def test_zero_attempts_is_left_out_not_averaged_in_as_zero(self):
        s = self.s
        s.write("mw_s42.log", log(block("Ann Aggr", attempts=0, halluc=0, buy=0)))
        rows, _ = s.load()
        m = A.seed_bucket_metrics(rows)[("mw", "aggressive", 42)]
        self.assertIsNone(m["halluc"])

    def test_paired_difference_is_per_seed_and_counts_direction(self):
        s = self.s
        # hold % mw vs base for the aggressive bucket: seed 42: 96 vs 84 (+12),
        # seed 43: 90 vs 90 (0)
        s.write("mw_s42.log", log(block("Ann Aggr", hold=96.0)))
        s.write("base_s42.log", log(block("Ann Aggr", hold=84.0)))
        s.write("mw_s43.log", log(block("Ann Aggr", hold=90.0)))
        s.write("base_s43.log", log(block("Ann Aggr", hold=90.0)))
        rows, notes = s.load()
        text = A.report(rows, notes)
        self.assertIn("+12.00", text)
        self.assertIn("mean +6.00", text)
        self.assertIn("lower in 0/2 seeds", text)

    def test_seed_is_the_unit_agents_are_averaged_before_seeds(self):
        # seed 42: aggressive agents hold 80 and 100 -> bucket 90; seed 43: 70 -> 70.
        # mean over seeds = 80, not the mean of the three agent-runs (83.3)
        s = self.s
        s.write("mw_s42.log", log(block("Ann Aggr", hold=80.0), block("Cara Cons", hold=100.0)))
        s.write("mw_s43.log", log(block("Ann Aggr", hold=70.0)))
        rows, _ = s.load()
        for r in rows:
            if r["agent"] == "Cara Cons":
                r["bucket"] = "aggressive"
        sbm = A.seed_bucket_metrics(rows)
        mean, _, n = A.mean_sd([sbm[("mw", "aggressive", sd)]["hold"] for sd in (42, 43)])
        self.assertAlmostEqual(mean, 80.0)
        self.assertEqual(n, 2)

    def test_persona_coverage_is_shown_and_low_coverage_is_called_out(self):
        # a moderate agent that mostly holds: 17 of 100 decisions scored
        s = self.s
        s.write("mw_s42.log", log(block("Mo Mod", scored=17, persona="1.00", buy=17, hold=83.0),
                                  block("Ann Aggr", scored=100)))
        s.write("base_s42.log", log(block("Mo Mod", scored=9, persona="1.00", buy=9, hold=91.0),
                                    block("Ann Aggr", scored=100)))
        rows, notes = s.load()
        text = A.report(rows, notes)
        self.assertIn("persona scored", text)
        self.assertIn("17% (17/100)", text)               # mw / moderate
        self.assertIn("LOW PERSONA COVERAGE", text)
        self.assertIn("mw / moderate: 17 of 100", text)
        self.assertIn("base / moderate: 9 of 100", text)
        self.assertNotIn("mw / aggressive", text)         # 100% covered, not called out
        self.assertIn("[persona scored: mw 17% (17/100), base 9% (9/100)]", text)

    def test_coverage_pools_scored_and_decisions_across_agents_and_seeds(self):
        s = self.s
        s.write("mw_s42.log", log(block("Mo Mod", scored=20), block("Cara Cons", scored=100)))
        s.write("mw_s43.log", log(block("Mo Mod", scored=0, persona="N/A", drift=0)))
        rows, _ = s.load()
        sbm = A.seed_bucket_metrics(rows)
        self.assertEqual(A.persona_coverage(sbm, "mw", "moderate", [42, 43]), (20, 200))
        self.assertEqual(A.persona_coverage(sbm, "mw", "conservative", [42, 43]), (100, 100))

    def test_csv_has_one_row_per_arm_seed_agent(self):
        s = self.s
        s.write("mw_s42.log", log(block("Ann Aggr"), block("Cara Cons")))
        rows, _ = s.load()
        path = os.path.join(s.tmp.name, "o.csv")
        A.write_csv(rows, path)
        with open(path) as f:
            self.assertEqual(len(f.read().strip().splitlines()), 3)   # header + 2


def paired_log(seed=42, advanced="middleware", mw_only=3, base_only=9,
               mw_rate="12.5", base_rate="30.0", report=True):
    sep = "=" * 68
    rows = [("trade attempts", "40", "40"), ("infeasible requests", "5", "12"),
            ("infeasible rate %", mw_rate, base_rate), ("distinct episodes", "3", "6"),
            ("caught pre-execution", "5", "0"), ("EXECUTED malformed", "0", "12"),
            ("abstention %", "92.0", "92.0"), ("participating agents", "7", "7"),
            ("stale context hits", "1", "2"), ("state contradictions", "4", "3"),
            ("avg context chars", "201.5", "960.2")]
    body = "\n".join(f"{a:<28}{b:>14}{c:>14}" for a, b, c in rows)
    head = f"\n{sep}\nPAIRED EVALUATION REPORT\n{sep}\n" if report else "\nstill running\n"
    return (head + f"Seed            : {seed}\nAdvanced with   : {advanced}\nMatched pairs   : 900\n"
            f"Model           : mistral\n\n{'':<28}{'middleware':>14}{'baseline':>14}\n{'-' * 56}\n{body}\n"
            f"\nDiscordant pairs (the informative ones):\n"
            f"  middleware infeasible, baseline fine : {mw_only}\n"
            f"  baseline infeasible, middleware fine : {base_only}\n"
            f"  both infeasible                      : 2\n"
            f"  neither                              : 886\n\nnote text\n")


class WilsonInterval(unittest.TestCase):
    def test_known_values(self):
        lo, hi = A.wilson(0, 10)
        self.assertAlmostEqual(lo, 0.0, places=6)
        self.assertAlmostEqual(hi, 27.75, places=1)          # z^2 / (n + z^2)
        lo, hi = A.wilson(5, 10)
        self.assertAlmostEqual(lo, 23.66, places=1)
        self.assertAlmostEqual(hi, 76.34, places=1)

    def test_no_trials_is_none_not_zero(self):
        self.assertIsNone(A.wilson(0, 0))

    def test_pooled_section_pools_attempts_and_shows_the_interval(self):
        s = _Sweep()
        self.addCleanup(s.close)
        s.write("mw_s42.log", log(block("Ann Aggr", attempts=10, halluc=2)))
        s.write("mw_s43.log", log(block("Ann Aggr", attempts=30, halluc=3)))
        rows, notes = s.load()
        text = A.pooled_halluc_section(rows)
        self.assertIn("5/40 = 12.5%", text)                  # pooled, not a mean of 20% and 10%
        self.assertIn("[5.5-26.1]", text)                     # Wilson for 5/40
        self.assertIn("floor on the real uncertainty", text)


class PairedReports(unittest.TestCase):
    def setUp(self):
        self.s = _Sweep()
        self.addCleanup(self.s.close)

    def test_parses_the_table_and_the_discordant_cells(self):
        r = A.parse_paired_log(paired_log(mw_only=3, base_only=9))
        self.assertEqual(r["advanced_with"], "middleware")
        self.assertEqual(r["pairs"], 900)
        self.assertEqual((r["mw"]["infeasible_pct"], r["base"]["infeasible_pct"]), (12.5, 30.0))
        self.assertEqual(r["cont"], {"mw_only": 3, "base_only": 9, "both": 2, "neither": 886})

    def test_a_non_paired_or_garbled_text_is_none(self):
        self.assertIsNone(A.parse_paired_log("HALLUCINATION REPORT\nnothing"))
        self.assertIsNone(A.parse_paired_log("PAIRED EVALUATION REPORT\nSeed : 1\n"))

    def test_load_paired_skips_truncated_and_the_per_agent_loader_ignores_paired_logs(self):
        s = self.s
        s.write("pmw_s42.log", paired_log(seed=42))
        s.write("pmw_s43.log", paired_log(seed=43, report=False))
        paired, _ = A.load_paired(s.out)
        self.assertEqual([(r["arm"], r["seed"]) for r in paired], [("pmw", 42)])
        # a truncated paired run never prints its header, so the per-agent loader
        # is the one that reports it; the finished one is not called unparsed
        rows, rnotes = s.load()
        self.assertEqual(rows, [])
        self.assertTrue(any("skipped pmw_s43.log" in n and "truncated" in n for n in rnotes))
        self.assertFalse(any("pmw_s42" in n for n in rnotes))

    def test_direction_is_read_from_the_discordant_cells_per_seed(self):
        s = self.s
        s.write("pmw_s42.log", paired_log(seed=42, mw_only=3, base_only=9))     # middleware better
        s.write("pmw_s43.log", paired_log(seed=43, mw_only=8, base_only=2))     # baseline better
        s.write("pmw_s44.log", paired_log(seed=44, mw_only=4, base_only=4))     # tie
        text = A.paired_section(A.load_paired(s.out)[0])
        self.assertIn("middleware better", text)
        self.assertIn("baseline better", text)
        self.assertIn("pooled discordant pairs: mw-only 15, base-only 15", text)
        self.assertIn("middleware better 1, baseline better 1, tie 1", text)

    def test_both_advance_modes_must_agree_for_the_result_to_be_called_robust(self):
        s = self.s
        s.write("pmw_s42.log", paired_log(seed=42, advanced="middleware", mw_only=3, base_only=9))
        s.write("pbase_s42.log", paired_log(seed=42, advanced="baseline", mw_only=9, base_only=3))
        text = A.paired_section(A.load_paired(s.out)[0])
        self.assertIn("DISAGREE", text)
        s.write("pbase_s42.log", paired_log(seed=42, advanced="baseline", mw_only=2, base_only=8))
        text = A.paired_section(A.load_paired(s.out)[0])
        self.assertIn("point the same way", text)


class Rescore(unittest.TestCase):
    """Persona numbers recomputed from trading_log.json with the current scorer."""

    def setUp(self):
        self.s = _Sweep()
        self.addCleanup(self.s.close)

    @staticmethod
    def _entry(agent, action, symbol, reasoning):
        return {"agent": agent, "step": 0,
                "decision": {"action": action, "symbol": symbol, "quantity": 1 if symbol else 0,
                             "reasoning": reasoning},
                "requested": {"action": action, "symbol": symbol, "quantity": 1 if symbol else 0},
                "persona_drift_score": 0.5}                       # what an old scorer stored

    def test_recomputes_a_hold_that_the_old_scorer_flagged_and_keeps_a_real_drift(self):
        s = self.s
        # printed (stale) numbers: 5 drift events, avg 0.5
        s.write("mw_s42.log", log(block("Cara Cons", persona="0.50", drift=5, scored=2)))
        s.write_sim("t1", "mw", 42, [
            self._entry("Cara Cons", "hold", None, "Tesla recalls worry me, so I wait."),   # in character
            self._entry("Cara Cons", "buy", "TSLA", "Buying TSLA on the dip."),             # off watchlist
        ])
        rows, _ = s.load()
        personas = A.load_personas(s.storage, "f")
        changed, notes = A.rescore_rows(rows, s.storage, "t1", personas)
        r = rows[0]
        # hold: 1 check (language) passes -> 1.0.  buy: buy check fails, language passes -> 0.5
        self.assertEqual((r["persona"], r["drift"], r["scored"]), (0.75, 1, 2))
        self.assertEqual(changed, 1)
        self.assertEqual(notes, [])

    def test_a_missing_sim_keeps_the_printed_numbers_and_says_so(self):
        s = self.s
        s.write("mw_s42.log", log(block("Cara Cons", persona="0.50", drift=5, scored=2)))
        rows, _ = s.load()
        changed, notes = A.rescore_rows(rows, s.storage, "t1", A.load_personas(s.storage, "f"))
        self.assertEqual((rows[0]["persona"], rows[0]["drift"]), (0.5, 5))
        self.assertEqual(changed, 0)
        self.assertTrue(any("no trading_log.json" in n for n in notes))

    def test_an_agent_with_nothing_scoreable_becomes_none_not_zero(self):
        s = self.s
        s.write("mw_s42.log", log(block("Mo Mod", persona="1.00", drift=0, scored=5)))
        s.write_sim("t1", "mw", 42, [self._entry("Mo Mod", "hold", None, "waiting")])   # moderate hold: nothing to score
        rows, _ = s.load()
        A.rescore_rows(rows, s.storage, "t1", A.load_personas(s.storage, "f"))
        self.assertEqual((rows[0]["persona"], rows[0]["scored"]), (None, 0))


if __name__ == "__main__":
    unittest.main()
