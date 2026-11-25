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
                json.dump({"risk_tolerance": tol}, f)

    def write(self, name, text):
        with open(os.path.join(self.out, name), "w") as f:
            f.write(text)

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

    def test_csv_has_one_row_per_arm_seed_agent(self):
        s = self.s
        s.write("mw_s42.log", log(block("Ann Aggr"), block("Cara Cons")))
        rows, _ = s.load()
        path = os.path.join(s.tmp.name, "o.csv")
        A.write_csv(rows, path)
        with open(path) as f:
            self.assertEqual(len(f.read().strip().splitlines()), 3)   # header + 2


if __name__ == "__main__":
    unittest.main()
