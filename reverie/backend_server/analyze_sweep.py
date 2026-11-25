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

PAIRED ARMS. pmw and pbase (paired_eval) are parsed separately: both arms are
evaluated on identical world states, so the informative numbers are the
discordant pairs (one arm's request infeasible, the other's fine). They are
shown per seed with the direction, never turned into a p-value: request pairs
inside a run are not independent, so a McNemar p would be optimistic.

RESCORE. --rescore recomputes the persona numbers from each sim's
trading_log.json with the CURRENT scorer, so a scorer change does not require
rerunning a sweep. It needs the sim folders (storage/<run>_<arm>_s<seed>/).

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
import math
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from middleware.persona_anchoring import (_normalise_risk_tolerance,
                                              score_persona_consistency,
                                              PERSONA_DRIFT_THRESHOLD)
except Exception:                                    # keeps the script usable alone
    def _normalise_risk_tolerance(raw):
        tol = (raw or "moderate").strip().lower().replace("_", "-")
        if "conserv" in tol or tol in ("low", "very-low"):
            return "conservative"
        if "aggress" in tol or tol in ("high", "very-high"):
            return "aggressive"
        return "moderate"
    score_persona_consistency = None
    PERSONA_DRIFT_THRESHOLD = 0.5

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


def load_personas(storage, fork):
    """agent -> {bucket, risk_tolerance, watchlist}, for rescoring."""
    with open(os.path.join(storage, fork, "reverie", "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    res = {}
    for name in meta["persona_names"]:
        p = os.path.join(storage, fork, "personas", name, "bootstrap_memory", "scratch.json")
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        res[name] = dict(bucket=_normalise_risk_tolerance(d.get("risk_tolerance")),
                         risk_tolerance=d.get("risk_tolerance"),
                         watchlist=d.get("watchlist", []))
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
        if "PAIRED EVALUATION REPORT" in text:
            continue                                  # see load_paired()
        parsed = parse_log(text)
        if not parsed:
            notes.append(f"unparsed {fn}: no per-agent blocks (a paired_eval report?)")
            continue
        for agent, r in parsed.items():
            rows.append(dict(arm=arm, seed=seed, agent=agent,
                             bucket=buckets.get(agent, "unknown"), **r))
    return rows, notes


# ------------------------------------------------------------ paired reports
PAIRED_ROWS = {
    "trade attempts": "attempts", "infeasible requests": "infeasible",
    "infeasible rate %": "infeasible_pct", "distinct episodes": "episodes",
    "caught pre-execution": "caught", "EXECUTED malformed": "executed_malformed",
    "abstention %": "abstention_pct", "participating agents": "participants",
    "stale context hits": "stale", "state contradictions": "contradictions",
    "avg context chars": "ctx_chars",
}
PAIRED_CONTINGENCY = {
    "middleware infeasible, baseline fine": "mw_only",
    "baseline infeasible, middleware fine": "base_only",
    "both infeasible": "both",
    "neither": "neither",
}


def _num(s):
    try:
        return float(str(s).replace(",", ""))
    except ValueError:
        return None


def parse_paired_log(text):
    """-> dict for one paired_eval report, or None if the text is not one."""
    if "PAIRED EVALUATION REPORT" not in text:
        return None
    out = dict(mw={}, base={}, cont={})
    for line in text.splitlines():
        m = re.match(r"^(Advanced with|Matched pairs)\s*:\s*(.+?)\s*$", line)
        if m:
            out["advanced_with" if m[1] == "Advanced with" else "pairs"] = (
                m[2] if m[1] == "Advanced with" else int(_num(m[2]) or 0))
            continue
        label, rest = line[:28].strip(), line[28:].split()
        if label in PAIRED_ROWS and len(rest) == 2:
            out["mw"][PAIRED_ROWS[label]] = _num(rest[0])
            out["base"][PAIRED_ROWS[label]] = _num(rest[1])
            continue
        m = re.match(r"^\s+(.+?)\s*:\s*(\d+)\s*$", line)
        if m and m[1] in PAIRED_CONTINGENCY:
            out["cont"][PAIRED_CONTINGENCY[m[1]]] = int(m[2])
    if len(out["cont"]) != 4 or not out["mw"]:
        return None
    return out


def load_paired(out_dir):
    """-> (list of dicts with arm and seed added, notes)."""
    res, notes = [], []
    for fn in sorted(os.listdir(out_dir)):
        m = LOG_NAME.match(fn)
        if not m:
            continue
        with open(os.path.join(out_dir, fn), encoding="utf-8", errors="replace") as f:
            text = f.read()
        if "PAIRED EVALUATION REPORT" not in text:
            continue
        if not re.search(r"REPORT$", text, re.M):
            notes.append(f"skipped {fn}: no report in the log (truncated run)")
            continue
        parsed = parse_paired_log(text)
        if parsed is None:
            notes.append(f"unparsed {fn}: a paired report this script could not read")
            continue
        res.append(dict(parsed, arm=m["arm"], seed=int(m["seed"])))
    return res, notes


def paired_section(paired):
    out = ["\n## Paired evaluation (both arms on identical world states)",
           "Discordant pairs are the informative ones. mw-only = middleware infeasible,",
           "baseline fine; base-only = baseline infeasible, middleware fine. base-only >",
           "mw-only means middleware prevented infeasible requests on that seed."]
    for arm in sorted({r["arm"] for r in paired}):
        rs = sorted((r for r in paired if r["arm"] == arm), key=lambda r: r["seed"])
        adv = rs[0].get("advanced_with", "?")
        out.append(f"\n{arm}  (world advanced with {adv})")
        out.append(f"  {'seed':<6}{'pairs':>7}{'attempts':>10}{'infeasible % mw / base':>26}"
                   f"{'mw-only':>9}{'base-only':>11}   reading")
        better = worse = tie = 0
        for r in rs:
            c = r["cont"]
            if c["base_only"] > c["mw_only"]:
                reading, better = "middleware better", better + 1
            elif c["base_only"] < c["mw_only"]:
                reading, worse = "baseline better", worse + 1
            else:
                reading, tie = "tie", tie + 1
            f = lambda v: "n/a" if v is None else f"{v:.1f}"
            out.append(f"  {r['seed']:<6}{r.get('pairs', 0):>7}{f(r['mw'].get('attempts')):>10}"
                       f"{f(r['mw'].get('infeasible_pct')) + ' / ' + f(r['base'].get('infeasible_pct')):>26}"
                       f"{c['mw_only']:>9}{c['base_only']:>11}   {reading}")
        mo = sum(r["cont"]["mw_only"] for r in rs)
        bo = sum(r["cont"]["base_only"] for r in rs)
        out.append(f"  pooled discordant pairs: mw-only {mo}, base-only {bo}   "
                   f"seeds: middleware better {better}, baseline better {worse}, tie {tie}")
    arms = {r["arm"] for r in paired}
    if len(arms) == 2:
        signs = {a: (sum(r["cont"]["base_only"] - r["cont"]["mw_only"] for r in paired if r["arm"] == a))
                 for a in arms}
        agree = all(v > 0 for v in signs.values()) or all(v < 0 for v in signs.values())
        out.append("\n  " + ("Both advance modes point the same way: the direction does not depend on "
                              "which arm moves the world." if agree else
                              "The two advance modes DISAGREE or tie: the result depends on which arm "
                              "moves the world, so it is not robust."))
    return "\n".join(out)


# ----------------------------------------------------------------- intervals
def wilson(k, n, z=1.96):
    """95% Wilson score interval for k successes in n trials, in percent."""
    if not n:
        return None
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * max(0.0, centre - half), 100 * min(1.0, centre + half)


def pooled_halluc_section(rows):
    out = ["\n## Pooled hallucination rate with a 95% Wilson interval",
           "All seeds and agents pooled. Attempts inside a run are not independent (same",
           "agents, same seed), so this interval is a floor on the real uncertainty."]
    for arm in sorted({r["arm"] for r in rows}):
        for b in BUCKETS:
            rs = [r for r in rows if r["arm"] == arm and r["bucket"] == b]
            k, n = sum(r["halluc"] for r in rs), sum(r["attempts"] for r in rs)
            if not rs:
                continue
            iv = wilson(k, n)
            out.append(f"  {arm:<8}{b:<14}" + (f"{k}/{n} = {100.0 * k / n:.1f}%  [{iv[0]:.1f}-{iv[1]:.1f}]"
                                                 if iv else "no attempts"))
    return "\n".join(out)


# ------------------------------------------------------------------- rescore
class _Scratch:
    def __init__(self, tol, watchlist):
        self.risk_tolerance, self.watchlist = tol, watchlist


class _Persona:
    def __init__(self, tol, watchlist):
        self.scratch = _Scratch(tol, watchlist)


def rescore_entries(entries, personas):
    """
    Recompute persona numbers per agent from trading_log entries with the current
    scorer, the way the harness aggregates them: the score of each decision is
    score_persona_consistency(reasoning, persona, requested); None is not averaged
    in; a drift event is a score at or under the threshold.
    """
    if score_persona_consistency is None:
        raise RuntimeError("the persona scorer could not be imported")
    scores = defaultdict(list)
    for e in entries:
        info = personas.get(e["agent"])
        if info is None:
            continue
        s = score_persona_consistency((e.get("decision") or {}).get("reasoning", ""),
                                      _Persona(info["risk_tolerance"], info["watchlist"]),
                                      e.get("requested") or {})
        if s is not None:
            scores[e["agent"]].append(s)
    return {a: dict(persona=round(sum(v) / len(v), 2) if v else None,
                    drift=sum(1 for x in v if x <= PERSONA_DRIFT_THRESHOLD), scored=len(v))
            for a, v in scores.items()}


def rescore_rows(rows, storage, run, personas):
    """Replace persona/drift/scored in rows from the sim logs. -> (changed, missing notes)."""
    changed, notes, cache = 0, [], {}
    for r in rows:
        key = (r["arm"], r["seed"])
        if key not in cache:
            path = os.path.join(storage, f"{run}_{r['arm']}_s{r['seed']}", "reverie", "trading_log.json")
            if not os.path.exists(path):
                cache[key] = None
                notes.append(f"rescore: no trading_log.json for {r['arm']}_s{r['seed']} ({path}); "
                             "kept the persona numbers the run printed")
            else:
                with open(path, encoding="utf-8") as f:
                    cache[key] = rescore_entries(json.load(f), personas)
        new = (cache[key] or {}).get(r["agent"])
        if cache[key] is not None:
            new = new or dict(persona=None, drift=0, scored=0)
            if (new["persona"], new["drift"], new["scored"]) != (r.get("persona"), r.get("drift"), r.get("scored")):
                changed += 1
            r.update(new)
    return changed, notes


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
            scored=sum(r.get("scored", 0) for r in rs),
            decisions=dec,
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


# Persona coverage: the share of decisions the persona score actually covers. A
# moderate agent has no language check, and a HOLD that names no off-profile
# thing carries no evidence, so a moderate agent that mostly holds can have a
# persona average built on a handful of buys (in the first nine-agent runs:
# Alex 0 of 100, Tomas 1 of 100, Sara 17 of 100). An average like that reads as
# 1.00 and means "no evidence". Coverage is printed beside every persona
# number, and anything under this share is called out.
LOW_PERSONA_COVERAGE = 50.0


def persona_coverage(sbm, arm, bucket, seeds):
    """-> (scored, decisions) pooled over the given seeds, for one arm and bucket."""
    s = sum(sbm[(arm, bucket, sd)]["scored"] for sd in seeds if (arm, bucket, sd) in sbm)
    d = sum(sbm[(arm, bucket, sd)]["decisions"] for sd in seeds if (arm, bucket, sd) in sbm)
    return s, d


def _cov_text(scored, decisions):
    return "n/a" if not decisions else f"{100.0 * scored / decisions:.0f}% ({scored}/{decisions})"


def report(rows, notes, paired=None):
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
        if paired:
            out.append(paired_section(paired))
        return "\n".join(out)

    sbm = seed_bucket_metrics(rows)
    low_cov = []
    out.append("\n## Per arm and bucket (mean ± sd ACROSS SEEDS of the per-seed bucket value)")
    for arm in arms:
        out.append(f"\n{arm}")
        out.append("  " + f"{'bucket':<13}" + "".join(f"{lab:<24}" for _, lab in METRICS)
                   + "persona scored")
        for b in BUCKETS + ("unknown",):
            seeds_b = sorted(s for (a, bb, s) in sbm if a == arm and bb == b)
            if not seeds_b:
                continue
            cells = []
            for key, _ in METRICS:
                m, sd, n = mean_sd([sbm[(arm, b, s)][key] for s in seeds_b])
                cells.append(_fmt(m, sd, n, 2 if key == "persona" else 1))
            sc, dc = persona_coverage(sbm, arm, b, seeds_b)
            cells.append(_cov_text(sc, dc))
            if dc and 100.0 * sc / dc < LOW_PERSONA_COVERAGE:
                low_cov.append((arm, b, sc, dc))
            out.append("  " + f"{b:<13}" + "".join(f"{c:<24}" for c in cells))

    if low_cov:
        out.append("\nLOW PERSONA COVERAGE (persona score covers under "
                   f"{LOW_PERSONA_COVERAGE:.0f}% of decisions):")
        for arm, b, sc, dc in low_cov:
            out.append(f"  {arm} / {b}: {sc} of {dc} decisions scored. The persona average there "
                       "rests on those decisions only (mostly buys); a 1.00 is absence of "
                       "evidence, not consistency.")

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
                line = (f"  {lab:<22} per seed: "
                        + " ".join(f"{d:+.2f}" for d in diffs)
                        + f"   mean {statistics.mean(diffs):+.2f}   lower in {lower}/{len(diffs)} seeds")
                if key in ("persona", "drift_per100"):
                    line += (f"   [persona scored: mw {_cov_text(*persona_coverage(sbm, 'mw', b, common))}"
                             f", base {_cov_text(*persona_coverage(sbm, 'base', b, common))}]")
                out.append(line)

    out.append(pooled_halluc_section(rows))
    if paired:
        out.append(paired_section(paired))

    out.append("\n## Per agent (mean over seeds)")
    out.append(f"  {'arm':<8}{'agent':<18}{'bucket':<14}{'hold %':>8}{'persona':>9}{'scored':>9}{'drift ev':>10}{'buys':>7}{'seeds':>7}")
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
        sc = sum(x.get("scored", 0) for x in rs)
        dc = sum(x.get("decisions", 0) for x in rs)
        cov = f"{100.0 * sc / dc:.0f}%" if dc else "n/a"
        out.append(f"  {r['arm']:<8}{r['agent']:<18}{r['bucket']:<14}{hold:>8.1f}"
                   f"{(statistics.mean(per) if per else float('nan')):>9.2f}{cov:>9}"
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
    ap.add_argument("--rescore", action="store_true",
                    help="recompute persona numbers from each sim's trading_log.json "
                         "with the current scorer (needs the sim folders in --storage)")
    ap.add_argument("--run", help="run id used in the sim names (default: the --out folder name)")
    args = ap.parse_args(argv)
    buckets = load_buckets(args.storage, args.fork)
    rows, notes = load_sweep(args.out, buckets)
    paired, pnotes = load_paired(args.out)
    notes += pnotes
    if args.rescore:
        run = args.run or os.path.basename(os.path.normpath(args.out))
        changed, rnotes = rescore_rows(rows, args.storage, run, load_personas(args.storage, args.fork))
        notes += rnotes
        notes.append(f"persona numbers rescored from trading_log.json with the current scorer; "
                     f"{changed} of {len(rows)} agent rows differ from what the run printed")
    print(report(rows, notes, paired))
    if args.csv:
        write_csv(rows, args.csv)
        print(f"\nwrote {len(rows)} rows to {args.csv}")


if __name__ == "__main__":
    main()
