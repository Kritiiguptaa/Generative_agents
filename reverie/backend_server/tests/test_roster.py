"""The base_trading roster: 9 agents, 3 per risk bucket, limits ordered by bucket.

These guard the data, not the simulation: a persona whose label and risk limit
disagree would make the profiles overlap while the scorer, which reads only the
label, reports nothing wrong.
"""
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from middleware.persona_anchoring import _normalise_risk_tolerance

SIM = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..",
                   "environment", "frontend_server", "storage", "base_trading")
KNOWN = {"NVDA", "AAPL", "AMD", "GOOGL", "TSLA"}


def _roster():
    meta = json.load(open(os.path.join(SIM, "reverie", "meta.json"), encoding="utf-8"))
    out = {}
    for name in meta["persona_names"]:
        p = os.path.join(SIM, "personas", name, "bootstrap_memory", "scratch.json")
        out[name] = json.load(open(p, encoding="utf-8"))
    return out


class Roster(unittest.TestCase):
    def setUp(self):
        self.r = _roster()

    def by_bucket(self):
        b = {"conservative": [], "moderate": [], "aggressive": []}
        for d in self.r.values():
            b[_normalise_risk_tolerance(d["risk_tolerance"])].append(d)
        return b

    def test_nine_agents_three_per_bucket(self):
        self.assertEqual(9, len(self.r))
        self.assertEqual([3, 3, 3], [len(v) for v in self.by_bucket().values()])

    def test_risk_limits_do_not_overlap_across_buckets(self):
        b = self.by_bucket()
        hi = lambda k: max(d["risk_limit_per_trade"] for d in b[k])
        lo = lambda k: min(d["risk_limit_per_trade"] for d in b[k])
        self.assertLess(hi("conservative"), lo("moderate"))
        self.assertLess(hi("moderate"), lo("aggressive"))

    def test_name_matches_folder_and_fields_are_usable(self):
        for name, d in self.r.items():
            self.assertEqual(name, d["name"])
            self.assertTrue(d["watchlist"])
            self.assertTrue(set(d["watchlist"]) <= KNOWN, name)
            self.assertGreater(d["cash_balance"], 0)

    def test_watchlists_are_not_all_identical(self):
        self.assertGreater(len({tuple(sorted(d["watchlist"])) for d in self.r.values()}), 5)


if __name__ == "__main__":
    unittest.main()
