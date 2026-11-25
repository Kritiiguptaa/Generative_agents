#!/usr/bin/env python3
"""
Aggregate a sweep's logs into per-bucket, per-arm tables, paired by seed.

    python3 analyze_sweep.py --out /workspace/run08
    python3 analyze_sweep.py --out /workspace/run07 --fork base_trading_3agent --csv run07.csv

WHY THE UNIT IS THE SEED. Agents inside one run share a market and a seed, so
they are not independent samples; seeds are. Every table therefore averages
over agents FIRST (per seed, per bucket) and only then looks across seeds, and
the paired section subtracts baseline from middleware seed by seed. With four
seeds there is no honest p-value, so none is printed: the paired section shows
each seed's difference and how many seeds point the same way, which is all that
n=4 supports.

WHAT IT READS. The per-agent block that trading_reverie.py prints at the end of
a run, from <arm>_s<seed>.log in --out. A log with no "REPORT" header is a
truncated run and is skipped with a warning, the same rule run_sweep.sh uses
for .done. The paired_eval arms (pmw, pbase) print a different report; they
are listed as unparsed rather than guessed at.

Rates that have a denominator of zero (hallucination rate for an agent that
never traded) are left out of the mean, never counted as 0%: an agent that did
nothing has not demonstrated zero hallucinations.
"""
import argparse
import csv
import json
import os
import re
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from middleware.persona_anchoring import _normalise_risk_tolerance
except Exception:                                    # keeps the script usable alone
    def _normalise_risk_tolerance(raw):
        tol = (raw or "moderate").strip().lower().replace("_", "-")
        if "conserv" in tol or tol in ("low", "very-low"):
            return "conservative"
        if "aggress" in tol or tol in ("high", "very-high"):
            return "aggressive"
        return "moderate"

BUCKETS = ("conservative", "moderate", "aggressive")
LOG_NAME = re.compile(r"^(?P<arm>.+)_s(?P<seed>\d+)\.log$")
HEADER = re.compile(r"^  (\S+(?: \S+)*)$")
NUM = r"([\d,]+(?:\.\d+)?)"

PATTERNS = {
    "decisions": re.compile(r"Decisions\s+:\s+(\d+)\s+\(active=(\d+), requested=(\d+)\)"),
    "halluc": re.compile(r"Action hallucinations\s+:\s+(\d+)\s+\(" + NUM + r"% of (\d+) attempts\)"),
    "fallbacks": re.compile(r"Parser fallbacks\s+:\s+(\d+)"),
    "stale": re.compile(r"Stale context hits\s+:\s+(\d+)\s+\(.*?of (\d+) reasoned"),
    "contra": re.compile(r"State contradictions\s+:\s+(\d+)\s+\(.*?of (\d+) reasoned"),
    "persona": re.compile(r"Persona consistency\s+:\s+avg=(\S+)\s+drift events=(\d+)\s+\(scored (\d+)\)"),
    "actions": re.compile(r"Actions\s+:\s+buy=(\d+)\s+sell=(\d+)\s+hold=(\d+)\s+analyze=(\d+)"),
    "hold": re.compile(r"Abstention\s+:\s+" + NUM + r"% HOLD(.*)$"),
    "pnl": re.compile(r"Portfolio PnL\s+:\s+\$([+-]?[\d,]+(?:\.\d+)?)\s+\(\$([\d,]+) -> \$([\d,]+)\)"),
}


def _f(s):
    return float(str(s).replace(",", ""))


def parse_log(text):
    """-> {agent: metrics dict} for every per-agent block in one log."""
    lines = text.splitlines()
    out = {}
    i = 0
    while i < len(lines):
        m = HEADER.match(lines[i])
        if m and i + 1 < len(lines) and PATTERNS["decisions"].search(lines[i + 1]):
            agent, row = m.group(1), {}
            j = i + 1
            while j < len(lines) and lines[j].strip():
                ln = lines[j]
                for key, pat in PATTERNS.items():
                    g = pat.search(ln)
                    if not g:
                        continue
                    if key == "decisions":
                        row.update(decisions=int(g[1]), active=int(g[2]), attempts_req=int(g[3]))
                    elif key == "halluc":
                        row.update(halluc=int(g[1]), attempts=int(g[3]))
                    elif key == "fallbacks":
                        row["fallbacks"] = int(g[1])
                    elif key == "stale":
                        row.update(stale=int(g[1]), reasoned=int(g[2]))
                    elif key == "contra":
                        row.update(contra=int(g[1]), reasoned=int(g[2]))
                    elif key == "persona":
                        row.update(persona=None if g[1] == "N/A" else float(g[1]),
                                   drift=int(g[2]), scored=int(g[3]))
                    elif key == "actions":
                        row.update(buy=int(g[1]), sell=int(g[2]), hold=int(g[3]), analyze=int(g[4]))
                    elif key == "hold":
                        row.update(hold_pct=_f(g[1]), hold_drift="PERSONA DRIFT" in g[2])
                    elif key == "pnl":
                        row.update(pnl=_f(g[1]), start=_f(g[2]), end=_f(g[3]))
                j += 1
            # "N/A (no trades requested)" has no attempts count: zero attempts.
            row.setdefault("halluc", 0)
            row.setdefault("attempts", 0)
            out.setdefault(agent, row)
            i = j
        else:
            i += 1
    return out


def load_buckets(storage, fork):
    """agent -> bucket, from the fork's own persona files."""
    with open(os.path.join(storage, fork, "reverie", "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    res = {}
    for name in meta["persona_names"]:
        p = os.path.join(storage, fork, "personas", name, "bootstrap_memory", "scratch.json")
        with open(p, encoding="utf-8") as f:
            tol = json.load(f).get("risk_tolerance")
        res[name] = _normalise_risk_tolerance(tol)
    return res


def load_sweep(out_dir, buckets):
    """-> (rows, notes). One row per (arm, seed, agent)."""
    rows, notes = [], []
    for fn in sorted(os.listdir(out_dir)):
        m = LOG_NAME.match(fn)
        if not m:
            continue
        arm, seed = m["arm"], int(m["seed"])
        with open(os.path.join(out_dir, fn), encoding="utf-8", errors="replace") as f:
            text = f.read()
        if not re.search(r"REPORT$", text, re.M):
            notes.append(f"skipped {fn}: no report in the log (truncated run)")
            continue
        parsed = parse_log(text)
        if not parsed:
            notes.append(f"unparsed {fn}: no per-agent blocks (a paired_eval report?)")
            continue
        for agent, r in parsed.items():
            rows.append(dict(arm=arm, seed=seed, agent=agent,
                             bucket=buckets.get(agent, "unknown"), **r))
    return rows, notes


# ---------------------------------------------------------------- statistics
def mean_sd(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None, 0
    return statistics.mean(xs), (statistics.stdev(xs) if len(xs) > 1 else 0.0), len(xs)


def _rate(num, den):
    return None if not den else 100.0 * num / den


def seed_bucket_metrics(rows):
    """{(arm, bucket, seed): metrics}. Agents are pooled WITHIN a seed and bucket."""
    grp = defaultdict(list)
    for r in rows:
        grp[(r["arm"], r["bucket"], r["seed"])].append(r)
    res = {}
    for key, rs in grp.items():
        dec = sum(r.get("decisions", 0) for r in rs)
        res[key] = dict(
            agents=len(rs),
            hold=statistics.mean(r["hold_pct"] for r in rs if "hold_pct" in r)
            if any("hold_pct" in r for r in rs) else None,
            halluc=_rate(sum(r["halluc"] for r in rs), sum(r["attempts"] for r in rs)),
            attempts=sum(r["attempts"] for r in rs),
            contra=_rate(sum(r.get("contra", 0) for r in rs), sum(r.get("reasoned", 0) for r in rs)),
            persona=statistics.mean([r["persona"] for r in rs if r.get("persona") is not None] or [float("nan")])
            if any(r.get("persona") is not None for r in rs) else None,
            drift_per100=_rate(sum(r.get("drift", 0) for r in rs), dec),
            pnl_pct=statistics.mean(100.0 * r["pnl"] / r["start"] for r in rs if r.get("start")) if any(r.get("start") for r in rs) else None,
        )
    return res


METRICS = (("hold", "hold %"), ("halluc", "halluc % of attempts"),
           ("contra", "state-contra %"), ("persona", "persona avg"),
           ("drift_per100", "drift ev/100 dec"), ("pnl_pct", "PnL %"))


def _fmt(m, sd, n, digits=1):
    if m is None:
        return "   n/a"
    return f"{m:.{digits}f}±{sd:.{digits}f} (n={n})" if n > 1 else f"{m:.{digits}f} (n=1)"


def report(rows, notes):
    out = []
    arms = sorted({r["arm"] for r in rows})
    seeds = sorted({r["seed"] for r in rows})
    out.append("# Sweep analysis\n")
    out.append(f"arms parsed: {', '.join(arms) or 'none'}   seeds: {', '.join(map(str, seeds)) or 'none'}")
    for arm in arms:
        have = sorted({r["seed"] for r in rows if r["arm"] == arm})
        out.append(f"  {arm:<10} seeds present: {have}")
    for n in notes:
        out.append(f"  NOTE {n}")
    if not rows:
        return "\n".join(out)

    sbm = seed_bucket_metrics(rows)
    out.append("\n## Per arm and bucket (mean ± sd ACROSS SEEDS of the per-seed bucket value)")
    for arm in arms:
        out.append(f"\n{arm}")
        out.append("  " + f"{'bucket':<13}" + "".join(f"{lab:<24}" for _, lab in METRICS))
        for b in BUCKETS + ("unknown",):
            seeds_b = sorted(s for (a, bb, s) in sbm if a == arm and bb == b)
            if not seeds_b:
                continue
            cells = []
            for key, _ in METRICS:
                m, sd, n = mean_sd([sbm[(arm, b, s)][key] for s in seeds_b])
                cells.append(_fmt(m, sd, n, 2 if key == "persona" else 1))
            out.append("  " + f"{b:<13}" + "".join(f"{c:<24}" for c in cells))

    if "mw" in arms and "base" in arms:
        out.append("\n## Paired: middleware minus baseline, per seed (negative = middleware lower)")
        out.append("Rates with zero attempts in either arm are left out of that seed.")
        for b in BUCKETS:
            common = sorted({s for (a, bb, s) in sbm if a == "mw" and bb == b}
                            & {s for (a, bb, s) in sbm if a == "base" and bb == b})
            if not common:
                continue
            out.append(f"\n{b}  (seeds {common})")
            for key, lab in METRICS:
                diffs = []
                for s in common:
                    x, y = sbm[("mw", b, s)][key], sbm[("base", b, s)][key]
                    if x is not None and y is not None:
                        diffs.append(x - y)
                if not diffs:
                    out.append(f"  {lab:<22} n/a")
                    continue
                lower = sum(d < 0 for d in diffs)
                out.append(f"  {lab:<22} per seed: "
                           + " ".join(f"{d:+.2f}" for d in diffs)
                           + f"   mean {statistics.mean(diffs):+.2f}   lower in {lower}/{len(diffs)} seeds")

    out.append("\n## Per agent (mean over seeds)")
    out.append(f"  {'arm':<8}{'agent':<18}{'bucket':<14}{'hold %':>8}{'persona':>9}{'drift ev':>10}{'buys':>7}{'seeds':>7}")
    key = lambda r: (r["arm"], BUCKETS.index(r["bucket"]) if r["bucket"] in BUCKETS else 9, r["agent"])
    done = set()
    for r in sorted(rows, key=key):
        k = (r["arm"], r["agent"])
        if k in done:
            continue
        done.add(k)
        rs = [x for x in rows if (x["arm"], x["agent"]) == k]
        hold = statistics.mean(x["hold_pct"] for x in rs if "hold_pct" in x)
        per = [x["persona"] for x in rs if x.get("persona") is not None]
        out.append(f"  {r['arm']:<8}{r['agent']:<18}{r['bucket']:<14}{hold:>8.1f}"
                   f"{(statistics.mean(per) if per else float('nan')):>9.2f}"
                   f"{statistics.mean(x.get('drift', 0) for x in rs):>10.1f}"
                   f"{statistics.mean(x.get('buy', 0) for x in rs):>7.1f}{len(rs):>7}")

    out.append("\nCaveats: seeds are the independent unit and there are few of them, so read the")
    out.append("per-seed numbers and the 'lower in k/n' counts, not the means alone. Hold rate is")
    out.append("informational: it says how much an agent acted, not whether it acted well.")
    return "\n".join(out)


def write_csv(rows, path):
    cols = ["arm", "seed", "agent", "bucket", "decisions", "attempts", "halluc", "fallbacks",
            "contra", "reasoned", "stale", "persona", "drift", "scored", "buy", "sell", "hold",
            "analyze", "hold_pct", "hold_drift", "pnl", "start", "end"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default="/workspace/run08", help="sweep output folder")
    ap.add_argument("--fork", default="base_trading", help="roster the sweep used")
    ap.add_argument("--storage", default="../../environment/frontend_server/storage")
    ap.add_argument("--csv", help="also write one row per (arm, seed, agent)")
    args = ap.parse_args(argv)
    buckets = load_buckets(args.storage, args.fork)
    rows, notes = load_sweep(args.out, buckets)
    print(report(rows, notes))
    if args.csv:
        write_csv(rows, args.csv)
        print(f"\nwrote {len(rows)} rows to {args.csv}")


if __name__ == "__main__":
    main()
