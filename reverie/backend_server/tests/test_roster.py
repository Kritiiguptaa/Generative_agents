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
    with open(os.path.join(SIM, "reverie", "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    out = {}
    for name in meta["persona_names"]:
        p = os.path.join(SIM, "personas", name, "bootstrap_memory", "scratch.json")
        with open(p, encoding="utf-8") as f:
            out[name] = json.load(f)
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


ACTIVE = os.path.join(os.path.dirname(SIM), "base_trading_active")


class ActiveFork(unittest.TestCase):
    """
    base_trading_active exists to test one thing: does stronger persona wording
    make the aggressive agents trade more? That only means something if the copy
    differs from base_trading in nothing else, so this pins the difference.
    """

    AGGRESSIVE = {"Marcus Webb", "Jade Whitfield", "Kofi Mensah"}

    @staticmethod
    def _persona(root, name):
        p = os.path.join(root, "personas", name, "bootstrap_memory", "scratch.json")
        with open(p, encoding="utf-8") as f:
            return json.load(f)

    def test_same_roster_in_the_same_order(self):
        with open(os.path.join(SIM, "reverie", "meta.json"), encoding="utf-8") as f:
            a = json.load(f)
        with open(os.path.join(ACTIVE, "reverie", "meta.json"), encoding="utf-8") as f:
            b = json.load(f)
        self.assertEqual(a["persona_names"], b["persona_names"])
        self.assertEqual(b["fork_sim_code"], "base_trading_active")

    def test_only_the_aggressive_agents_background_differs_and_only_by_addition(self):
        with open(os.path.join(SIM, "reverie", "meta.json"), encoding="utf-8") as f:
            names = json.load(f)["persona_names"]
        for n in names:
            base, act = self._persona(SIM, n), self._persona(ACTIVE, n)
            changed = {k for k in set(base) | set(act) if base.get(k) != act.get(k)}
            if n in self.AGGRESSIVE:
                self.assertEqual(changed, {"learned"}, n)
                self.assertTrue(act["learned"].startswith(base["learned"].rstrip()), n)
                self.assertGreater(len(act["learned"]), len(base["learned"]), n)
            else:
                self.assertEqual(changed, set(), f"{n} must be identical to base_trading")

    def test_the_added_wording_states_a_habit_not_an_instruction_to_trade(self):
        for n in self.AGGRESSIVE:
            added = self._persona(ACTIVE, n)["learned"][len(self._persona(SIM, n)["learned"].rstrip()):]
            self.assertNotRegex(added.lower(), r"\b(must|you should|always trade|every step)\b", n)


if __name__ == "__main__":
    unittest.main()
