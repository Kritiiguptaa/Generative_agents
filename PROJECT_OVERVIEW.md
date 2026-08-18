# Agent Anchor — project overview

Everything a newcomer needs to understand what this repo is, what it measures,
and how the pieces fit. Read this first; then read
[CHANGES.md](CHANGES.md) §23 for the run-06 results and §25 for what to do next.

Working branch: **`armaans-frontend`**.

---

## 1. What this project is

A fork of Stanford's **Generative Agents** (Park et al., UIST '23 — the
"Smallville" simulation) repurposed into a **trading floor**.

Three LLM-driven trader personas run in a simulated market for N steps. Each
step every agent perceives prices and news, retrieves memories, and returns a
trading decision as JSON. The point is not to make money — PnL is
instrumentation, not a goal.

**The research question:**

> Does a middleware layer sitting between the agent's memory and the LLM reduce
> *hallucinated actions* — decisions an agent is not entitled to make, or that
> contradict its own state — without simply making the agent do nothing?

Every run is therefore **two arms**:

| Arm | Flag | What it is |
|---|---|---|
| **middleware** | *(default)* | Memory compression + ID-RAG + persona anchoring + action filtering |
| **baseline** | `--no-middleware` | The raw Stanford retrieval pipeline, same prompt vocabulary |

The two arms must be **comparable**. Most of the engineering effort in this repo
has gone into removing confounds that made them incomparable — see CHANGES.md.

---

## 2. Why this was hard: the metric kept measuring the wrong thing

This is the single most important thing to understand before touching anything.

The project's history is a sequence of discovering that a reported number was an
artefact:

| Run | Reported | What it actually was |
|---|---|---|
| 01 | `0 hallucinations (0.0%)` in **both** arms | The detector sat *downstream* of the filter whose job is to erase hallucinations. It was structurally incapable of returning nonzero. |
| 02 | `100% analyze, 0 trades` | Personas' `currently` field was frozen at "about to enter", so agents stayed permanently pre-entry. And `ANALYZE` was a free escape hatch. |
| 03 | middleware 50.6% vs baseline 14.3% | Different detection stages, different populations, and one livelocked agent contributing 27 of 44 events. |
| 03 | "compression degrades persona consistency" | The scorer penalised *short* reasoning. It was measuring brevity. |
| 04–05 | `145.5%` hallucination rate | The denominator excluded rejected actions. |
| 02–05 | "5× stale-context reduction" | The detector scaled with reasoning length. Compression shortens reasoning. It was measuring verbosity. |

**The lesson, which applies to anything you add:** a metric that moves is not
evidence. Before quoting a number, construct the input that would make it fire
spuriously and check that it doesn't.

---

## 3. The three agents

Defined in
`environment/frontend_server/storage/base_trading/personas/<Name>/bootstrap_memory/scratch.json`.

They are deliberately constructed to have **different failure modes**:

| | Alex Chen | Marcus Webb | Sara Kim |
|---|---|---|---|
| role | quant HF manager, 15 yrs | retail day trader, 3 yrs | news-driven momentum, 8 yrs |
| innate | analytical, disciplined, patient | impulsive, FOMO-driven, risk-seeking | reactive, news-focused, overconfident |
| risk tolerance | `moderate` | `aggressive` | `moderate-high` |
| watchlist | NVDA AAPL AMD GOOGL | TSLA NVDA AMD | NVDA TSLA AAPL GOOGL |
| starting book | 200 NVDA @ $192.20, large portfolio | $25k cash, no positions | 100 NVDA @ $192.20 |
| designed failure | over-caution / waiting for confirmation forever | **over-sizing beyond available cash** | **acting on stale headlines** |

Marcus is the source of most `clamped_cash` / `illegal_request` events. Sara is
the intended target of the stale-context detector.

Fields that matter to the code:

- `innate`, `learned`, `trading_strategy` — **stable traits**. Persona
  differentiation lives here.
- `currently` — **rebuilt from live state every step** by `refresh_currently()`
  (CHANGES §11.2). It is facts, not intent: cash, holdings, mark-to-market,
  max single-trade size. Historically it was static and this froze the agents.
- `risk_tolerance` — free text, normalised into conservative/moderate/aggressive
  by `_normalise_risk_tolerance()`. `moderate-high` used to match no branch at
  all.

---

## 4. The market

`market_environment.py` (synthetic) and `historical_market_environment.py`
(replays cached real 1-minute bars). Symbols: **NVDA, AAPL, TSLA, AMD, GOOGL**.

The experimental substance is **`SCRIPTED_NEWS`** — a fixed calendar of 12
headlines designed so that each symbol has a *narrative arc that reverses*:

```
NVDA:  step 5  +8%  "smashes Q4 estimates"
       step 30 -6%  "critical supply bottleneck; deliveries at risk"
       step 70 +4%  "supply fears overblown, maintains Buy"
       step 100 +7% "10-for-1 stock split"

TSLA:  step 15 -7%  "recalls 200k vehicles; NHTSA probe"
       step 40 +9%  "Q1 deliveries beat street by 18%"
       step 80 -5%  "Musk sells $2.1B of shares"

AAPL:  step 60 -4%  "DOJ antitrust suit"
AMD:   step 50 +6%  "gains share as NVDA supply tightens"
```

An agent still citing the step-30 NVDA supply bottleneck at step 90 — after
step 70 explicitly refuted it — is reasoning from stale context. **That** is what
the stale-context metric is supposed to detect. (It didn't, until `4c6897e`.)

News price impacts **persist and compound** via a per-symbol `_news_multiplier`
(CHANGES §11.3). Before that fix they were erased on the next tick, so the whole
contradictory-news premise lasted exactly one step.

Order execution models spread, slippage and commission. Cost basis books at the
**fill** price, not the mid.

---

## 5. The middleware

`reverie/backend_server/middleware/`. Four layers are in the decision path:

### `memory_compression.py`
Compresses the **retrieval output** — what reaches the LLM this step — not the
memory store. Four deterministic stages, **no LLM call**: flatten+dedup by
`node_id` then by cosine similarity (newer node wins), symbol-scoped recency
pruning (keeps newest K per ticker, which is what drops superseded news),
then budget trimming.

Measured effect: ~8,691 → ~1,562 nodes, ~5,159 → ~782 prompt chars.

Distinct from `reflect()`, which compresses the *store* by synthesising events
into thought nodes.

### `id_rag.py`
Identity Retrieval-Augmented Generation. Instead of prepending the whole persona
block every step, identity is stored as a knowledge graph and only the top-k
most contextually relevant nodes are retrieved by cosine similarity
(`nomic-embed-text`). The *forbidden* node is always appended regardless of
score. Updated dynamically after reflection and after agent interactions.

This is the real call site used by `make_trading_decision()` —
`anchor_memory_context` is only reachable from `input_stabilizer.py`.

### `persona_anchoring.py`
Builds the identity block prepended to every prompt, and scores how in-character
the returned reasoning is (`score_persona_consistency` → `float | None`).

**Caveat:** this scorer currently returns ~1.00 on ~99.8% of decisions. It has
essentially no discriminative power. Do not retune it into significance —
see CHANGES §19.4.

### `action_filtering.py`
The gate. Validates the model's JSON, rejects illegal actions, retries once,
then substitutes a legal `HOLD`. Builds the legal-action menu from
`min(cash, risk_cap)`.

Key exports: `run_action_filtering_step`, `is_illegal_action_error`,
`describe_request`, `coerce_quantity`, `is_trade_request`, `Action`.

Others in the folder (`chaos_injector`, `output_validator`,
`semantic_density_gating`, `metrics_recorder`, `input_stabilizer`,
`middleware_wrapper`) exist but are not all on the measured path.

---

## 6. What gets measured

Written to
`environment/frontend_server/storage/<sim>/reverie/hallucination_report.json`,
plus a per-step `trading_log.json`.

### Hallucination
"The model requested something it wasn't entitled to." Note this includes
requests that **executed at a clamped size** — the request is what's counted, not
the outcome.

Kinds:

```
illegal_request          verb / symbol / structure the agent may not use
clamped_cash             asked to buy more than cash allows
clamped_risk             exceeded the per-trade risk cap
clamped_holdings         asked to sell more than owned
unaffordable  over_risk_limit  unowned
untradeable_symbol  unknown_symbol  market_closed
```

**Disposition** is the honest comparison:

| | `caught_pre_execution` | `executed_malformed` |
|---|---|---|
| middleware | many | should be **0** |
| baseline | 0 (no filter exists) | all of them |

> Middleware caught 44 infeasible requests and executed **0**.
> The baseline caught 3 and executed **all 3**.

That statement is valid. "50.6% vs 14.3%" was not.

### Denominator: `trade_attempts`
What the model **requested**, via `is_trade_request()` — which is an
*inverted* test: anything not in `{"hold", "analyze"}` counts. An allowlist of
`("buy","sell")` could only undercount, and produced the 145.5% figure.

### Episodes
`_count_episodes()` reports **raw and distinct**, never distinct alone. A large
gap means an agent is **livelocked** — reissuing the same rejected request every
step because blocking an order leaves its state unchanged. Fixed by feeding
rejections back into memory (`record_order_feedback`, both arms).

### Stale context
Rewritten in `4c6897e`. Requires **all four**: headline ≥20 steps old, superseded
by a later opposite-sign same-symbol headline, agent names that symbol, ≥2
distinctive tokens match (news stopwords excluded). Verified length-invariant.

**Run 06+ figures are not comparable to runs 02–05.**

### State grounding
`check_state_grounding()` — compares the reasoning against the actual book,
arithmetically. **The only detector that scores a HOLD**, and the only one that
survived audit unchanged.

### Reporting rules
- `0/0` prints `N/A`, never `0.0%`. A vacuous result must not look measured.
- Headline and per-agent rates use the **same** denominator.
- Report across seeds as **median and range**. n=1 supports no rate claim.

---

## 7. `paired_eval.py` — the controlled comparison

Independent arms drift into different states, so by step 50 you are comparing two
different agents rather than one agent under two conditions.

`paired_eval.py` runs **one** canonical simulation. Each step both decision paths
run against the *same* persona state, prices and retrieved memories; the two
requests are recorded as a matched pair. Only `--advance-with` actually executes.
Reports a **McNemar contingency table** — the discordant cells are the only ones
carrying information.

**Stated limitation:** the non-advancing arm's request is a counterfactual
one-step-ahead request from the *other* arm's trajectory. **Run both directions.**
If the conclusion flips, the effect is not robust and must not be reported as one.

**Never compare a paired result to an independent-arm result.** Different
populations, different denominators.

---

## 8. Repo layout

```
reverie/backend_server/
  trading_reverie.py            main runner: the step loop, execution, reporting
  paired_eval.py                matched-pair harness (McNemar)
  run_sweep.sh                  multi-seed sweep with a wall-clock budget
  market_environment.py         synthetic market + SCRIPTED_NEWS
  historical_market_environment.py   replays cached real 1-min bars
  market_perceive.py            market events -> memory nodes; order feedback
  trading_persona.py            persona construction
  trading_daily_plan.py         per-day plan (feeds a retrieval focal point)
  middleware/                   the four measured layers + extras
  persona/                      Stanford cognitive modules (retrieve, reflect, plan)
  tests/                        127 tests
    test_hallucination_metrics.py
    test_run05_defects.py
    test_memory_compression.py
    test_id_rag.py

environment/frontend_server/    Django 2.2 replay UI
  storage/<sim>/                per-run output: reverie/, personas/, movement/
  storage/base_trading/         the fork every run starts from
  templates/trading_map/        the trading floor UI
  static_dirs/assets/office/    LimeZu office tileset + build.py

CHANGES.md                      chronological working log — what/why/still open
PROJECT_OVERVIEW.md             this file
```

---

## 9. The demo UI

A run is not just a JSON report — there is a **live, watchable trading floor**.
Agents walk to their desks, walk to the market board when they place an order,
and the decision that moved them scrolls past in a log beside the map. It is the
fastest way to tell whether a run is behaving, long before the report exists.

Rebuilt in `010960f`; the defects that made it "basic and broken" are in
CHANGES.md §18.

### 9.1 Starting it

```bash
cd environment/frontend_server
python manage.py runserver
```

The simulation runner (`trading_reverie.py`) is a **separate process**. The UI
reads what the runner writes to `storage/<sim>/`, so the two are independent —
you can start the UI mid-run, or open a finished run days later.

### 9.2 Routes

| URL | What it is |
|---|---|
| `/` | **Sim index.** Every directory in `storage/` with a `reverie/meta.json`. Trading runs sort first, then by decision count. Shows steps, decisions and a LIVE badge. |
| `/map/<sim_code>/<step>/` | **The trading floor.** The main view. |
| `/trading_poll/<sim_code>/` | JSON the map polls every **4 s** for new steps. |
| `/trading_persona_state/<sim_code>/<step>/<Persona_Name>/` | The per-agent detail panel, fetched on click. |
| `/replay/<sim_code>/<step>/` | Inherited Smallville replay (village sims). |

The index exists so a `sim_code` never has to be hand-typed into a URL. It
filters on `meta.json` because half-created folders and the Stanford village
storage dirs sit in the same directory and would otherwise render as dead links.

### 9.3 What is on the screen

Rendered with **Phaser 3.55.2** over a 44×32 tile LimeZu office map, 16 px tiles
at `ZOOM = 2` (1408×1024 backing store) with `image-rendering: pixelated`, so the
pixel art stays sharp instead of blurring.

- **Topbar** — sim name, clock, current step / max step, and a **LIVE** dot when
  `meta.json` says the runner is still writing.
- **Ticker** — the five symbols with their price at the current step.
- **Map panel** — the office. Each agent has a fixed desk (`sim_data.DESK_TILES`).
  On a `buy` or `sell` the agent walks to the **market board**
  (`MARKET_BOARD_TILE = [22, 4]`), which is how a trade is visible at a glance;
  multiple traders at the board fan out sideways rather than stacking. Agent
  interactions place both parties on the meeting row (`MEETING_ROW = 9`).
  Nameplates are staggered by `2 + (i % 3) * 7` px so co-located agents stay
  readable.
- **Transport** — prev / pause / next, scrubbing through steps. In live mode
  playback pauses automatically when the run ends.
- **Decision log** — one row per decision: step, agent, action, symbol, quantity.
  Clicking an agent card **filters the log to that agent**; the filter label has
  a clear button. The pane auto-scrolls to the current row — and scrolls *only
  that pane*, which is the fix for the whole page yanking.
- **Agent cards** — one per persona: name, role, emoji, and live figures.
- **Detail panel** — click an agent to fetch the full record for the current
  step: reasoning text, the decision, and retry / fallback status. If that agent
  has no record at the current step, it falls back to the nearest earlier step
  that does.

### 9.4 How live mode works

`trading_map()` renders once with whatever exists at page load. Then the page
polls `/trading_poll/<sim_code>/` every 4 s and appends new steps, so a run in
progress becomes visible **step by step as it happens** rather than only after it
finishes. Polling stops when `is_running` goes false.

Both the initial render and the poll go through the same
`sim_data.build_replay()` ingestion layer — there is no second copy of the
parsing logic to drift out of sync.

### 9.5 Two things that will confuse you

- **Restart the server after editing `views.py`.** Templates reload per request
  but Python does not, so a stale view looks exactly like a template bug.
- **A dead run can show LIVE forever.** `is_running` comes from `meta.json`, and
  a run killed with Ctrl-C never clears the flag. `map_office_001` is in this
  state right now. Harmless — edit the flag if it bothers you.

---

## 10. Running it

### Prerequisites
Ollama on `:11434` with **`mistral`** (decisions) and **`nomic-embed-text`**
(embeddings).

Two environment gotchas that cost real time:

- **DGX driver pin.** `dgxhnode3` has driver 535 — pin `OLLAMA_VERSION=0.9.6` or
  Ollama silently runs **CPU-only**. A run that is 10× slow is this, not the model.
- **Small models need `format="json"` and `max_tokens ≥ 300`.** Without both,
  decisions silently fall back and you read it as bad model behaviour. The daily
  planner had exactly this bug (`stop=["\n\n"]` truncating pretty-printed JSON).

### A single run

```bash
cd reverie/backend_server

python -u trading_reverie.py --fork base_trading --sim myrun \
  --steps 200 --seed 42 --fresh
python -u trading_reverie.py --fork base_trading --sim myrun_base \
  --steps 200 --seed 42 --fresh --no-middleware
```

`--fresh` deletes and re-forks. **Reusing a `--sim` name without it exits with an
error** — personas persist but the market does not, so a "resumed" run has agents
holding positions bought at step-N prices against a market rebuilt at step 0.
That is not a bug to work around; do not recycle names.

`python -u` matters: without it Python block-buffers into the log and a live
`tail` shows nothing for minutes, which reads as a hang.

### The sweep (what run 06 is)

```bash
cd /workspace/Generative_agents/reverie/backend_server
chmod +x run_sweep.sh
nohup ./run_sweep.sh > /workspace/sweep.log 2>&1 &
tail -f /workspace/sweep.log
```

4 seeds × 4 configs × 200 steps, 10 h cap, `.done` resume markers. See
CHANGES §21.

### The demo UI

```bash
cd environment/frontend_server
python manage.py runserver
```

Then `http://localhost:8000/`. Full description in **§9**.

---

## 11. Current state (2026-08-18)

- Runs 02–05 analysed. Every headline number from them has been traced to either
  a real effect or an artefact; see the table in §2.
- Run-05 fixes committed: `264db38` (denominator, feedback gate, arm symmetry,
  reflection parser), `4c6897e` (stale detector replaced outright).
- Frontend rebuilt: `010960f`.
- **Run 06 has landed and been analysed (CHANGES §23).** All 16 configs
  completed cleanly — 0 Ollama errors, fallbacks ≤2%. One claim survives:
  *middleware executed 0 infeasible requests in every seed; the baseline
  executed 3–7 in every seed.*
- **Run 06 also failed at its own purpose.** `--seed` reached only the market
  generator while the decision call decoded greedily at `temperature=0`, so the
  middleware arm replayed one trajectory across all four seeds — four seeds
  bought n≈1. Two further defects: `paired_eval.py` never updated the ID-RAG
  graph (so it ran different middleware from the runner, diverging at step 1),
  and the report let an abstaining arm look clean. All three are fixed in the
  working tree (CHANGES §24).
- Test suite: **141 passing** (`tests/test_run06_defects.py` is new).
- **Run 07 is the re-sweep.** `run_sweep.sh` is already pointed at it.

**Next actions are in [CHANGES.md §25](CHANGES.md).** Item 1 is a 5-minute
smoke test that must pass before the sweep is worth launching.

---

## 12. Things that will bite you

1. **Django parses `{% %}` inside HTML comments.** A commented-out `{% extends %}`
   is still an extends tag.
2. **`isinstance(True, int)` is `True`.** `{"quantity": true}` becomes a 1-share
   order unless `bool` is rejected explicitly. `coerce_quantity` does.
3. **The prompt used to advertise a budget the agent didn't have** — raw risk cap
   printed while the menu was built from `min(cash, risk_cap)`. The model was
   told $5,017, shown $76, and given no BUY option.
4. **Blocking an order without feedback causes livelock.** State doesn't change,
   so the next prompt is identical and the model reissues the same request.
5. **Retrieval recency was inverted** in the inherited Stanford code — oldest
   memory scored most-recent. Fixed, but it means run 04+ is not comparable to
   run 03.
6. **`"OLLAMA ERROR"` is returned as a string** by every failure path in
   `ollama_request` — connection refused, timeout, model missing. Downstream it
   fails to parse and is logged as a *model* parse failure. Still unfixed; a
   flaky server reads as "the model produced bad JSON 40% of the time".
7. **`hold` is legal by construction.** An arm can improve its hallucination rate
   simply by trading less. Always read trade attempts alongside the rate.
8. **Don't change the model and a metric definition in the same run.** You lose
   the ability to attribute the difference.

---

## 13. Credit

Built on **Generative Agents: Interactive Simulacra of Human Behavior** —
Park, O'Brien, Cai, Morris, Liang, Bernstein (UIST '23). See
[README.md](README.md) for the original setup instructions and citation.

Office tileset by [LimeZu](https://twitter.com/lime_px).
