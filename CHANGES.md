# Trading simulation — hallucination metric fixes

Working log of what was changed, what was deliberately left alone, and what is
still open. Written for the trading-sim work on branch `armaans-frontend`.

**Status as of the latest entry (2026-08-18):** runs 02–05 are done and analysed.
The run-05 fixes are committed (`264db38`, `4c6897e`). **Run 06 — the first
multi-seed sweep — has landed and been analysed (§23).** It produced one claim
that survives and three defects that invalidate most of the rest; the fixes for
all three are committed (`f2ea081`) and **run 07 is the re-run, now executing on
the DGX**. The seed fix is confirmed working on real hardware (§24.4). The
frontend was rebuilt in `010960f`.

**If you are catching up, read section 23 first** — it carries the run-06
results and the current next actions. It supersedes §22. Sections 19–21 explain
how we got here.

**Section index**

| § | Contents |
|---|---|
| 0–9 | The original defect: the metric could not observe anything. Tier 1/2/3. |
| 10–12 | Run 02 — agents refused to trade at all. |
| 13–16 | Run 03 — first usable numbers; livelock and prompt-budget bugs found. |
| 17 | Run 04 and the `261ceec` partial fix. |
| 18 | **The demo UI** — what it is, plus the rebuild (`010960f`). |
| 19 | **Run 05 analysis — the >100% rate and the stale-context confound.** |
| 20 | Fixes from run 05 (`264db38`, `4c6897e`). |
| 21 | Run 06 — the multi-seed sweep, currently in flight. |
| 22 | What to do next *(superseded by §23.5)*. |
| 23 | **Run 06 analysis — the one surviving claim and three defects.** |
| 24 | Fixes from run 06 (seed/temperature, ID-RAG parity, abstention). |
| 25 | **What to do next.** |

Companion document: [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) — what the project
is, what the code does, and how the pieces fit. Read that first if you have never
seen this repo.

---

## 0. Why any of this happened

Two 200-step runs — one with middleware, one without — both reported
`Total hallucinations : 0 (0.0% of all decisions)`.

That was not a coincidence and not a lucky result. The metric was wired
downstream of the component whose entire job is to prevent the thing being
counted, so it was **structurally incapable of returning a nonzero number** in
the middleware arm, and had nothing to count at all in the baseline arm.

Everything in sections 1–2 follows from that.

---

## 1. Tier 1 — the metric could not observe anything

All committed in `8ad0fe2`.

### 1.1 Count the requested action, not the executed one

**Files:** `middleware/action_filtering.py`, `trading_reverie.py`

`run_action_filtering_step()` rejects illegal actions, retries once, then
substitutes a legal `hold`. By the time `execute_trading_action()` saw a
decision, every hallucination had already been erased. The detector sat
downstream of the fix.

- `run_action_filtering_step()` now returns `(decision, stats)` where `stats`
  carries `illegal_request`, `status`, `error_reason`, `requested`.
- New `ILLEGAL_ACTION_ERRORS` / `is_illegal_action_error()` separates *illegal
  trade requests* from *malformed JSON*. Without this split the metric would
  have just been measuring how badly the model formats JSON.
- New `describe_request()` captures what the model asked for before correction.
- The baseline arm has no filter, so illegality still surfaces at execution.
  The report takes the **union** of both detection points.

### 1.2 Clamping is a hallucination, not a silent repair

**File:** `trading_reverie.py` — `execute_trading_action()`

An agent asking to buy 5,000 shares it could not afford was silently reduced to
the affordable quantity, printed as an informational `[ACTION FILTER]` line, and
logged `filtered=False`. The exact trigger the module docstring advertises
(Marcus Webb over-buying) was being repaired and reported clean. Same for
selling more shares than owned.

`execute_trading_action()` now returns `hallucination`, `halluc_kind`,
`requested_quantity`, `filled_quantity`. Kinds:

```
clamped_cash  clamped_risk  clamped_holdings  unaffordable  over_risk_limit
unowned  untradeable_symbol  unknown_symbol  market_closed
```

The order still executes at the clamped size — that is the middleware doing its
job. The *request* is what gets counted.

### 1.3 Refuse to silently resume a simulation

**File:** `trading_reverie.py` — `__init__`, `main()`

`if not sim_path.exists()` meant reusing a `--sim` name **resumed** it. Personas
are persisted by `_save()`; the market is not. So a resumed run started agents
holding positions bought at step-N prices against a market rebuilt at step 0.

This is why the two arms reported different starting portfolios for the same
agent from the same fork ($442,708 vs $540,108 for Alex Chen). They were not the
same experiment, so no cross-arm PnL or rate comparison was valid.

Now: reusing a sim name raises `SystemExit` with an explanation. `--fresh`
deletes and re-forks. Resuming is refused rather than repaired because there is
no market checkpoint to resume against.

### 1.4 Make the two arms comparable

Three separate divergences, all confounding:

| | baseline (before) | middleware (before) |
|---|---|---|
| `ANALYZE` offered | yes | **no** |
| symbol universe | any priced symbol | **watchlist only** |
| market-hours check | **none** | HOLD-only when closed |

So `analyze=200` vs `hold=200` was partly a prompt difference, and the
middleware arm could never produce an out-of-watchlist hallucination because it
was constrained out of the action space — another guaranteed zero.

- `ANALYZE` added to `get_legal_actions`, the prompt menu, and
  `validate_response`. Legal even when the market is closed.
- Market-hours and tradeable-universe checks moved **into**
  `execute_trading_action`, so both arms are bound by them.
- The baseline prompt now states the same market status, tradeable symbols and
  action vocabulary the filtered arm's menu states.

### 1.5 Report arithmetic

- New `trade_attempts` denominator — what the model **requested**, not what it
  was permitted to do. The old `active_decisions` denominator meant a run where
  every illegal trade was blocked divided by ~0.
- `active or 1` removed. A 0/0 now prints `N/A`, not `0.0%`. This is what made a
  vacuous result look like a measured one.
- Headline and per-agent rates use the **same** denominator (they were `/total`
  and `/active` respectively, so they were never comparable).
- Per-kind breakdown added.

---

## 2. Tier 2 — metrics that scored empty output as good

All committed in `8ad0fe2`.

### 2.1 Persona drift could never fire

**File:** `middleware/persona_anchoring.py`

A fallback decision has `reasoning=""`. That violated only the watchlist check
and scored exactly `round(1 - 1/3, 2) == 0.67`. The drift test was
`drift_score < 0.67` — and `0.67 < 0.67` is `False`.

**Every parser fallback was recorded as perfectly in character.** That is why
drift events read 0 in both arms.

- `score_persona_consistency()` now returns `None` for empty/whitespace
  reasoning. Not scoreable is not the same as scored well.
- Threshold extracted to `PERSONA_DRIFT_THRESHOLD = 0.5` ("strictly more than
  half the checks violated"). For the 3-check case this is behaviourally
  identical to the old `< 0.67`; the real change is that unscoreable decisions
  no longer participate.

### 2.2 The consistency denominator was wrong for 2 of 3 agents

Same file. `checks = 3` was hard-coded, but the language check only runs for
conservative or aggressive profiles. Alex (`moderate`) and Sara
(`moderate-high`) were divided by 3 while only 2 checks could fire —
structurally scoring higher than Marcus (`aggressive`).

`moderate-high` matched **no** branch at all, in either the scorer or
`_risk_rules`, so Sara was also being handed guidelines for a profile she
doesn't have.

- Denominator now counts only checks that ran.
- New `_normalise_risk_tolerance()` buckets free text into
  conservative/moderate/aggressive. Used by both the scorer and `_risk_rules`.

### 2.3 Stale-context rate was diluted by empty strings

**File:** `trading_reverie.py`

Stale context is only observable in reasoning text. An empty reasoning scores
clean for free, so dividing by *all* decisions let a high fallback rate
masquerade as a low stale-context rate — the single most likely innocent
explanation for a large baseline-vs-middleware gap on that metric.

- Logs `has_reasoning` per step.
- Rate is now `hits / reasoned_decisions`.
- Report prints fallback count and reasoned count directly above it, so the two
  can't be confused.

Demonstration — 8 of 10 decisions are empty-reasoning fallbacks, 2 stale hits
among the 2 real ones:

```
Parser fallbacks      : 8  (80.0% of decisions)
Decisions w/ reasoning: 2  (the only ones stale/drift can score)
Stale context hits    : 2  (100.0% of 2 reasoned)
```

Old code reported that as 20%.

### 2.4 The two arms parsed quantities differently

**File:** `middleware/action_filtering.py`

The filtered arm demanded a real `int` and rejected `10.0` / `"10"`, burning a
retry and usually landing on a fallback HOLD. The baseline coerced all three
with `int()`. So part of the filtered arm's do-nothing rate was **parser
strictness**, not decision quality — the exact thing `_split_combined_action`'s
own docstring argues against.

New shared `coerce_quantity()`, used by both arms. `bool` is rejected
explicitly: `isinstance(True, int)` is `True` in Python, so `{"quantity": true}`
would otherwise have become a 1-share order.

### 2.5 Baseline context size is now visible

The baseline arm hard-coded `mc_stats = {"enabled": False}`, which made the
"why does it always analyze?" question untestable. It now reports real
`nodes_in` / `chars_in`, and the report prints average context chars per
decision.

---

## 3. Changes that rode along

Not part of either tier, but touched because they were inside code being
rewritten. Flagged here so they aren't a surprise in review.

- **Cost basis now uses fill price, not mid.** Cash was already debited
  `fill["total_value"]`, so booking `avg_price` at the mid understated the basis
  by spread + slippage + commission. `trading_reverie.py`, buy path.
- **`anchor_memory_context` docstring corrected.** It claimed to be "the single
  call site used by `make_trading_decision()`". It isn't — `id_rag_anchor` is.
  The function is only reachable from `input_stabilizer.py`.
- **`validate_response` no longer emits `"HOLD quantity must be 0"`.** A no-op
  action carrying a stray quantity is normalised to 0 rather than rejected,
  which previously burned a retry and inflated the fallback rate.

---

## 4. Interface changes — read this before pulling

### Signatures

| Function | Before | After |
|---|---|---|
| `run_action_filtering_step` | `-> decision` | `-> (decision, stats)` |
| `free_form_decision` | `-> decision` | `-> (decision, stats)` |
| `make_trading_decision` | `-> (decision, mc_stats)` | `-> (decision, mc_stats, filter_stats)` |
| `score_persona_consistency` | `-> float` | `-> float \| None` |
| `execute_trading_action` | 3 result keys | 7 result keys |

New public names: `coerce_quantity`, `is_illegal_action_error`,
`describe_request`, `ILLEGAL_ACTION_ERRORS`, `PERSONA_DRIFT_THRESHOLD`,
`_normalise_risk_tolerance`.

### CLI

```bash
--fresh    # delete an existing --sim folder and re-fork it
```

Reusing a `--sim` name without `--fresh` now **exits with an error**. This is
intentional; do not work around it by recycling names.

### New keys in `trading_log.json`

`requested`, `filter_status`, `halluc_kind`, `illegal_request`,
`requested_quantity`, `filled_quantity`, `filtered`, `has_reasoning`.

`hallucination` changed meaning: it used to be `result["filtered"]` (order
rejected), it is now "the model requested something it wasn't entitled to",
which includes clamped orders that still executed.

### New keys in `hallucination_report.json`

Top level: `total_decisions`, `total_trade_attempts`.
Per agent: `trade_attempts`, `hallucination_kinds`, `fallback_decisions`,
`fallback_rate_pct`, `reasoned_decisions`, `scored_decisions`.
Under `memory_compression`: `chars_in`, `chars_out`, `avg_context_chars`.

**`overall_hallucination_rate_pct`, `action_hallucination_rate_pct` and
`stale_context_rate_pct` can now be `null`** (meaning "nothing to measure"),
where before they were always a float.

Checked: `sim_data.py` passes the report through wholesale and no current
template renders those fields, so nothing breaks today. Any future consumer must
handle `null`.

---

## 5. Tier 3 — known, NOT fixed

Deliberately left. None of these change a number in the report; they were
skipped so the GPU run could happen sooner.

1. **`start_portfolio` captured post-first-trade.** `_generate_report` reads the
   first *log* entry, which is written after execution, so spread and commission
   from step 0 are already baked in. Biases PnL slightly. Snapshot before the
   loop instead. `trading_reverie.py`, `_generate_report`.
2. **Unguarded `KeyError` on a rejected order.** `execute_order` can return
   `{"status": "rejected"}` with no `total_value` key; the buy path indexes it
   directly. Currently unreachable, but unguarded.
3. **`compress_memories` swallows every exception** and still returns
   `enabled: True` (`memory_compression.py:408`). A silently failing compressor
   is indistinguishable from a working one in the report. *This is the one worth
   pulling forward* — if compression fails on the DGX you would read the run as
   "compression ran and didn't help".
4. **`datetime.utcnow()`** deprecated in 3.12+. `action_filtering.py:291`.
5. **`_bar_index` never resets across days** and `is_open` ignores weekends.
   Latent — only matters on multi-day runs. At `sec_per_step=60` a 200-step run
   ends at 12:50 on the same day, so it does not bite yet. It would on any run
   past ~390 steps.

---

## 6. Deeper findings — diagnosed, NOT fixed

Found while the GPU run was in progress. These do not affect whether the metrics
*can* observe anything (Tier 1/2 fixed that). They affect whether what they
observe **means what it appears to mean**.

### 6.1 Temperature 0 — ~~likely explains `analyze=200`~~ **REFUTED by run 02**

> **This hypothesis was wrong.** Run 02 produced 71–122 *distinct* reasoning
> strings per agent out of 200 decisions. The model reasons freshly each step
> and independently concludes "wait" every time. Temperature is not the cause.
> The real cause is in section 10.2. The description below is kept for the
> record; the setting is still `temperature=0` and arguably still worth
> raising, but it is not what produced the all-`analyze` run.

`ollama_request` defaults to `temperature=0` (`gpt_structure.py:55`) and the
trading call does not override it (`trading_reverie.py:149`). That is greedy
decoding: same prompt in, same tokens out, every time. The prompt barely changes
step to step, so the model lands on the same answer repeatedly.

The agents are not making 200 decisions. They are making one and echoing it.

**Fix:** `temperature≈0.7` on the decision call. Leave embeddings at 0.

*Confidence: the setting is a fact; that it causes the 200/200 is a strong
hypothesis, not proven.*

### 6.2 `"OLLAMA ERROR"` makes infrastructure failure look like model failure

> **Did not occur in run 02** — 0 `OLLAMA ERROR` lines and 0 parser fallbacks in
> both arms, so the server was healthy and this did not distort those results.
> The design flaw is real and still unfixed; it just wasn't a factor this time.

Every failure path in `ollama_request` — connection refused, timeout, model not
pulled, HTTP error — returns the **string** `"OLLAMA ERROR"`. Downstream that
fails to parse and is logged as `"response was not valid JSON"` → parser
fallback.

So a flaky server reads as "the model produced bad JSON 40% of the time". This
directly undermines the fallback-rate metric added in 2.3. Given the known
driver-pin issue that makes Ollama silently fall back to CPU, this should be
fixed before trusting any fallback number.

**Fix:** distinguishable sentinel or a raised exception; count `llm_errors`
separately from parse failures.

### 6.3 The stale-context detector is effectively a word search for "supply"

Verified against `SCRIPTED_NEWS`:

- **Only two keywords in the entire news table can ever trigger it** — `supply`
  and `deliveries`. Every other word appears once, or always with the same
  sentiment, so it can never look contradicted.
- **It cannot tell right from wrong.** At step 75, reasoning that cites the
  *freshest* NVDA news (step 70) and reasoning that cites the *stalest* (step 30)
  produce byte-identical flags.
- `deliveries` collides across NVDA and TSLA, so mentioning Tesla deliveries
  flags you for stale Nvidia news.

A 75% stale rate therefore means "75% of the time the agent said the word
supply", which is unremarkable when supply headlines dominate memory.

**Fix:** check which symbol the agent is discussing and which side it took, not
whether a keyword appears.

### 6.4 The daily planner probably fails every run, silently

```python
# decision call  — trading_reverie.py:149
ollama_request(prompt, max_tokens=300, timeout=300, format="json")
# planner call   — trading_daily_plan.py:53
ollama_request(prompt, max_tokens=400, stop=["\n\n"], timeout=180)
```

The decision call's own comment explains why both differences are wrong: no
`format="json"` means invalid JSON isn't structurally prevented, and
`stop=["\n\n"]` cuts pretty-printed JSON off at the first blank line. The failure
is swallowed by `except Exception: pass`, so every agent quietly gets the same
hardcoded fallback plan.

Knock-on: `current_focus_str` feeds one of three retrieval focal points. Shared
generic plan → shared focal point → personas less differentiated than intended.

This is the same class of bug as the earlier `max_tokens 220→400` commit; the
lesson was never copied to this call site.

*Confidence: inferred from the code, not observed — needs a real mistral call to
confirm.*

### 6.5 Retrieval's recency score is inverted

`new_retrieve` sorts memories **oldest first** (`retrieve.py:488-492`), then
`extract_recency` assigns `0.99¹` (highest) to index 0 and `0.99ⁿ` (lowest) to
the last. Oldest memory gets the highest recency score.

Second twist: retrieved nodes get `last_accessed = now`, moving them to the end
of the sort next time — so a memory you just used is demoted.

Inherited from the original Stanford code, so not introduced here. It matters
more in this project because "does the agent cling to stale memories?" is the
thing being measured. Recency carries the smallest weight (0.5 vs relevance 3,
importance 2), so it is a thumb on the scale rather than the whole story.

### 6.6 Documented noise filter does not exist

`market_perceive.py`'s docstring claims "*Idle / flat-market events (magnitude
< 1%) are suppressed entirely*". There is no such check. The only gate is the
market's own 0.3% emission threshold, so roughly 3× more price noise enters
memory than documented — feeding the 9,000-node context.

### 6.7 Poignancy scale contradicts its own comment

`score_market_event` claims "1% move → 2, 5% move → 7". The arithmetic
(`int(mag*0.8)+1`) gives 1 and 5. Importance runs low, the reflection budget
drains slower than intended, agents reflect less often.

Also minor: `get_summarized_latest_events(50)` is called inside the per-event
loop rather than once.

---

## 7. Verification status

**Tested and passing:**
- 77 existing tests (`tests/test_memory_compression.py`, `tests/test_id_rag.py`)
- Filter path: ANALYZE validation, illegal-vs-formatting classification,
  illegal request surviving filtering, legal request not flagged
- Execution path: all 9 hallucination kinds, legal actions staying clean,
  cost-basis change
- Report path: `trade_attempts` denominator, `N/A` for 0/0, stale rate over
  reasoned decisions, drift threshold
- `--fresh` guard refusing a resume

**Not tested:**
- **The entire LLM path.** Ollama is not reachable from the dev machine, so no
  end-to-end run has exercised `make_trading_decision` against a real model.
  Everything above was verified with stubs.
- Section 6 findings are diagnosis only — 6.1 and 6.4 in particular need a real
  run to confirm.

---

## 8. Suggested order from here

*(Revised after run 02. The original list led with 6.1, which run 02 refuted.)*

1. **Re-run both arms** with the section 11 fixes. Those three changes are
   behavioural and their effect is unmeasurable without a real run. Until agents
   actually trade, the hallucination metric still has nothing to observe.
2. **6.4 (daily planner)** — still unconfirmed and still silently swallowing its
   own failure. Cheap to fix, and it feeds the retrieval focal points.
3. **Tier 3 item 3** (compression failing silently) — cheap, protects the
   interpretation of the next run.
4. **6.3 (stale-context detector)** — before writing anything up. Run 02 showed
   this metric moving (70% → 37.5%, 69% → 24%), which makes it tempting to
   quote. It is a word search for "supply"/"deliveries" and cannot tell a
   correct citation from a stale one. Do not publish it as-is.
5. **6.2 (error sentinel)** — did not bite run 02, but the next run may not be
   as lucky.
6. **10.7** (reflection truncation), then remaining Tier 3, then 6.5–6.7.

---

## 10. Run 02 results — what the data actually showed

Two 200-step runs on the DGX (`run_mw_02`, `run_base_02`), analysed from the
reports, filter logs and stdout logs.

### 10.1 Headline

Both arms: **600 decisions, 100% `analyze`, 0 trade attempts, 0 parser
fallbacks, 0 Ollama errors, 200/200 decisions with real reasoning.**

The pipeline was healthy. Nothing failed. The model simply chose to do nothing,
600 times out of 600.

Note this is a *different* zero from the original bug. The metric now correctly
reports `N/A` — "no agent ever requested a trade" — instead of printing `0.0%`
for a 0/0. That part worked as designed.

### 10.2 Root cause: `currently` is frozen at "about to enter"

Every persona's bootstrap `currently` describes the agent as pre-entry, and the
model restated it back nearly verbatim:

| persona `currently` | model's most frequent reasoning |
|---|---|
| Alex: "wants confirmation of supply tightening **before entering**" | "Wait for confirmation of supply tightening in NVDA before considering AMD" (×17) |
| Marcus: "watching for a **dip entry**… buy $5,000 on any pullback" | "Wait for a dip in NVDA price to enter with a minimum investment of $5,000" (×17) |
| Sara: "considering adding 50 more **if price holds above $191**" | "Wait for the NVDA price to hold above $191 before considering adding 50 more" (×10) |

`scratch.currently` is **never written** by the trading loop. The only writer in
the codebase is `plan.py:464` (`revise_identity`), which belongs to the village
simulation. The trading sim uses `trading_daily_plan.ensure_daily_plan()`
instead and never touches the field.

**The stated conditions were objectively met and ignored:**

```
Sara needs NVDA > $191     ->  272/272 observations qualified (100% of the run)
Marcus needs a dip         ->  122 down-ticks occurred
Alex needs supply news     ->  fired at step 30 and again at step 50
```

A static intention keeps an agent permanently pre-entry, because no prompt ever
tells it the condition has been satisfied.

### 10.3 The `ANALYZE` addition was a regression

Adding `ANALYZE` to the filtered arm for symmetry (section 1.4) removed the only
trades that were happening:

| | before (`8ad0fe2^`) | after |
|---|---|---|
| middleware arm | hold 200/195/189, **16 trades** | analyze 200/200/200, **0 trades** |

The risk was flagged when the change was made; run 02 confirmed it. `ANALYZE` is
a free escape hatch — a model can always justify gathering more information, so
an agent offered it never has to commit. Fixed in section 11.1 by removing it
from **both** arms rather than adding it to both.

### 10.4 New bug: scripted news price shocks were erased after one tick

`HistoricalMarketEnvironment.tick()` overwrote `current_prices[symbol]` with
`series[idx]` at the top of every tick, discarding the previous step's news
impact. The large moves in the logs were the *snap-backs*, not the news:

```
TSLA rises 7.6% to $383.11    <- reversion of the step-15 -7% recall
TSLA falls 8.40% to $381.88   <- reversion of the step-40 +9% deliveries beat
```

The entire contradictory-news design — the premise of the experiment — had price
impacts lasting exactly one step. Fixed in section 11.3.

### 10.5 The market was nearly flat

```
NVDA   $199.38 – $201.10   (0.86% range over the whole run)
TSLA   $380.59 – $384.05   (0.91%)
AMD    $511.72 – $524.76   (2.5%)
```

200 steps × 60s = 200 minutes of 1-minute bars. Combined with 10.4, there was
very little for a willing agent to react to. The 11.3 fix substantially changes
this, since news shocks now persist and compound.

### 10.6 What worked

- **Memory compression:** 8,691 → 1,562 nodes (Alex); average prompt context
  5,159 → 782 chars. Stale-context hits fell for Marcus (70% → 37.5%) and Sara
  (69% → 24%).
- **The fresh-state guard (1.3):** PnL was byte-identical across arms
  (−212 / 0 / −106), confirming both runs started from the same state against
  the same deterministic market. That comparison was impossible before.
- **Alex's stale rate was 75% in *both* arms** — his reasoning fixates on the
  word "supply" regardless of how much context is removed. That is finding 6.3
  visible in the data.

### 10.7 Also observed, not yet fixed

82 reflection insight calls truncated at `num_predict=150` and 30 more at
`num_predict=30` (`generate_insights_and_evidence`). Incomplete thoughts are
being written into associative memory. Not in the decision path
(`max_tokens=300`), so it did not affect the decisions directly.

---

## 11. Fixes from the run 02 analysis

### 11.1 `ANALYZE` removed from both arms

**Files:** `middleware/action_filtering.py`, `trading_reverie.py`

Removed from `get_legal_actions`, both prompt templates, and the JSON format
hints. `HOLD` already expresses "no trade this step" without implying a deferred
decision.

A model that emits `ANALYZE` anyway is **normalised to `HOLD`** rather than
rejected, in both arms — it is expressing "no trade", not an illegal trade, and
rejecting it would burn a retry and inflate the fallback rate with a decision
that was semantically fine.

### 11.2 `currently` is rebuilt from live state every step

**File:** `trading_reverie.py` — new `refresh_currently()`, called from
`_step_agent` (after perceive, before anything reads the field) and from
`__init__` (before the first daily plan is generated).

`currently` now carries facts, not intent:

```
before: "Marcus has $25,000 in cash and no open positions. He is watching
         TSLA and NVDA for a dip entry."
after:  "Marcus Webb has $25,000 in cash and holds 10 NVDA at avg $180.00
         (now $200.00, +11.1%). Total portfolio value $27,000. Maximum value
         for a single trade is $5,400."
```

Persona differentiation is unaffected — it lives in `trading_strategy`,
`innate` and `learned`, which are stable traits rather than a status that goes
stale.

### 11.3 Scripted news impacts now persist

**File:** `historical_market_environment.py`

New `_news_multiplier` per symbol. The replayed bar is multiplied by the
accumulated news factor, so the real price path is preserved *and* the shock
survives, compounding as later headlines for the same symbol arrive.

Verified — TSLA now traces the intended narrative arc, each level holding:

```
step 15 (-7% recall)            383.45 -> 355.92, holds at ~356
step 40 (+9% deliveries beat)   354.66 -> 387.71, holds at ~387
step 80 (-5% Musk sells)        395.79 -> 375.34, holds at ~375
```

Reversion on the tick after the +9% headline: **8.4% before → 0.15% after**.

### 11.4 Verification

77 existing tests still pass. Fix-specific checks cover: `ANALYZE` absent from
the legal set and both prompts, `ANALYZE`→`HOLD` normalisation, `currently`
containing no intent language and correct mark-to-market, and news shocks
persisting across all three TSLA headlines.

Still untested: the LLM path (Ollama unreachable locally). **11.1 and 11.2 are
behavioural changes whose effect can only be measured by a real run.**

---

## 12. Run commands

```bash
# DGX
cd /workspace/Generative_agents
git fetch origin
git checkout armaans-frontend
git pull origin armaans-frontend

cd reverie/backend_server
python trading_reverie.py --fork base_trading --sim run_mw_02   --steps 200 \
  2>&1 | tee /workspace/run_mw_02.log
python trading_reverie.py --fork base_trading --sim run_base_02 --steps 200 --no-middleware \
  2>&1 | tee /workspace/run_base_02.log
```

Reports land at
`environment/frontend_server/storage/<sim>/reverie/hallucination_report.json`.

**Read the output in this order:**

1. **Parser fallbacks %** — first, always. A high rate means most decisions have
   empty reasoning and every downstream metric is measuring almost nothing.
   (Caveat 6.2: this number currently conflates server failure with model
   failure.)
2. **Trade attempts** — if 0 or near-0 in either arm, the rate prints `N/A` and
   you have a policy problem, not a middleware result.
3. **Hallucinations by kind** — `illegal_request` in the middleware arm vs
   `clamped_cash` / `unowned` / etc. in the baseline. This is the comparison the
   whole exercise is for.
4. **avg prompt context chars** in the baseline — input to the "always analyze"
   question.

`trade_attempts=0` printing `N/A` is correct behaviour, not a bug.

---

# ============================================================
# Run 03 analysis and the fixes that followed
# ============================================================

## 13. Run 03 results

First run where the metric produced usable numbers.

| | baseline | middleware |
|---|---|---|
| decisions | 600 | 600 |
| trade attempts | 21 | 87 |
| hallucinations | 3 (14.3%) | 44 (50.6%) |
| kinds | all `clamped_*` | all `illegal_request` |
| fallbacks | 2 | 3 |
| Ollama errors | 0 | 0 |

The three run-02 fixes all landed:

- **ANALYZE removal** — `analyze: 0` for every agent in both arms; trading resumed.
- **`refresh_currently`** — Alex's reasoning became varied and genuine (88 distinct
  strings) instead of restating his bootstrap intent text.
- **News persistence** — the stale detector fired on a real contradiction:
  `'deliveries' (from step 30, 127 steps ago; contradicted at step 40)`.

### 13.1 All 44 middleware hallucinations were genuine — and all one bug

Audited every one against `trading_log.json`. **Zero false positives**: every
rejected request exceeded available cash. But the pattern was damning:

```
step 36-59   Marcus Webb   buy NVDA x10   cash=$76.41   cost=$2,030
step 175-199 Sara Kim      buy NVDA x10   cash=$199     cost=$2,266
```

27 of the 44 were the identical request `buy NVDA x10`. Cash bound all 44; the
risk cap additionally bound 20.

### 13.2 Root cause: the prompt advertised a budget the agent did not have

`get_legal_actions()` has always applied `min(cash, risk_cap)` when building the
BUY menu. `build_prompt()` printed the **raw risk cap**:

```
Max single-trade size: $5,017.00     <- risk cap
- Cash: $76.41                       <- reality
You MUST choose from ONLY these actions:
- HOLD ...                           <- no BUY on the menu
```

The model was told it could spend $5,017, shown $76, and given a menu with no
BUY. The requests are only partly the model's fault. Same defect in
`refresh_currently()` for the baseline arm.

### 13.3 Second cause: filtering without feedback causes livelock

Blocking an order leaves cash and holdings untouched, so the next step rebuilds
a near-identical prompt and the model reissues the same request.

```
MW-Marcus   steps 15-62: cash = $76.41, ONE distinct value across 48 steps
BASE-Marcus steps 15-62: cash moved through 5 values, $21,077-$22,117
```

The baseline never livelocks because doing *something* always changes its state.

### 13.4 Why 50.6% vs 14.3% was never a valid comparison

Three independent reasons, none of which is "middleware hallucinates more":

1. **Different detection stages.** The filtered arm catches requests before
   execution; the baseline can only catch them at execution, where they
   *execute anyway* at a clamped size.
2. **Different populations.** MW-Marcus was broke at $76 while BASE-Marcus had
   $21K. Not one agent under two conditions — two agents in different
   situations.
3. **Repeat inflation.** One livelocked agent contributed 27 of 44.

The comparison that *is* valid and was true all along:

> Middleware caught 44 infeasible requests and executed **0**.
> The baseline caught 3 and executed **all 3**.

### 13.5 Refuted: "compression degrades persona consistency"

Marcus's score fell 0.98 (baseline) to 0.82 (middleware), which read as the
anchoring layer damaging identity. It was a defect in the scorer.

The watchlist check penalised reasoning that named **no** watchlist ticker.
Compressed context produces shorter reasoning, so:

```
Marcus BASE: names a watchlist symbol in 93% of reasonings -> 0.978
Marcus MW:   46%                                            -> 0.820
```

It was measuring brevity, not character. It also missed company names entirely
— "NVIDIA smashes Q4 estimates" scored as watchlist-blind because `NVIDIA` does
not contain `NVDA`. After correcting the check to match its own docstring
("acting on a symbol not in the watchlist"), rescoring run 03's reasoning gives
**1.000 in both arms** for Marcus and Sara. The gap was entirely artifact.

---

## 14. Fixes from the run 03 analysis

All verified; 100 tests pass (`tests/test_hallucination_metrics.py` adds 23).

| # | Fix | Files |
|---|---|---|
| A1 | Budget shown = `min(cash, risk_cap)`; wording derived from the actual menu, never from `budget > 0` | `action_filtering.py`, `trading_reverie.py` |
| A1b | `currently` reports cash % and position weight % (bootstrap had this; my first version dropped it) | `trading_reverie.py` |
| A3 | `--seed` on both runners | `trading_reverie.py` |
| B2 | `check_state_grounding()` — reasoning vs actual book; **the only detector that scores a HOLD** | `trading_reverie.py` |
| B3 | Drift threshold comparison `<` to `<=` (a 2-check agent scored exactly 0.50 and passed) | `persona_anchoring.py`, `trading_reverie.py` |
| B4 | Removed the stale-signal check — it double-counted `stale_context` | `persona_anchoring.py` |
| B5 | `halluc_disposition`: `caught_pre_execution` vs `executed_malformed` | `trading_reverie.py` |
| B6 | `_count_episodes()` — raw AND distinct, never distinct alone | `trading_reverie.py` |
| C1 | `record_order_feedback()` — rejections/partial fills enter memory, **both arms** | `market_perceive.py`, `trading_reverie.py` |
| C2 | Explicit no-borrow / no-naked-sell rules in both prompts | both prompts |
| D1 | CSV logs `requested_action/symbol/quantity` (all 44 rejections logged as `hold,,0`) | `action_filtering.py` |
| D2 | Watchlist check matches its docstring + company-name aliases | `persona_anchoring.py` |
| A2 | **`paired_eval.py`** — new harness, both arms decide from identical state | new file |
| E | venue-rejection KeyError; `start_portfolio` uses pre-trade value; compression failures surfaced; daily planner `format="json"` and no blank-line stop; recency ordering inverted; reflection caps 150 to 300/400; `utcnow()`; frozen-price warning | various |

### 14.1 The recency bug (E)

`new_retrieve()` sorted nodes **ascending** by `last_accessed`, and
`extract_recency()` assigns `recency_decay ** i` by list position — so position
0 got the *largest* weight. The oldest memory was scored most-recent and the
newest was decayed hardest. Now sorted `reverse=True`.

This changes retrieval in **both** arms, so run 04 is not comparable to run 03.

### 14.2 paired_eval.py — what it does and does not claim

One canonical simulation advances the world. Each step, both decision paths run
against the **same** persona state, prices and retrieved memories; the requests
are recorded as a matched pair. Only `--advance-with` executes.

Reports a McNemar contingency table — the discordant cells are the only ones
carrying information about a within-subject difference.

**Limitation, stated plainly:** the non-advancing arm's requests are
counterfactual one-step-ahead requests from the *other* arm's trajectory. Run
`--advance-with` both ways. If the conclusion flips, the effect is not robust
and must not be reported as one.

---

## 15. Still open after run 03

- **n=1.** Three baseline events cannot support a rate claim. Needs several
  seeds per arm before anything is reportable.
- **6.2** OLLAMA error sentinel still conflates server failure with model failure.
- **6.3** stale-context word search is still a bare substring scan.
- **6.6 / 6.7** noise filter and poignancy scale — untouched.
- **Persona layer has almost no signal.** After the D2 correction the scorer
  returns ~1.0 nearly everywhere on run 03 data (1 drift event total). That is
  honest — agents genuinely did not name off-watchlist stocks — but it means
  this layer currently demonstrates nothing either way.
- **`hold` is still legal by construction.** `check_state_grounding` closes part
  of the gap, but an agent that holds with vague reasoning remains unfalsifiable.
  Middleware-Marcus held *more* than baseline-Marcus (193 vs 181), so an arm can
  still improve its rate by trading less.

---

## 16. Run commands (run 04)

```bash
cd /workspace/Generative_agents
git pull origin armaans-frontend
cd reverie/backend_server

# Independent arms, as before -- now with an explicit shared seed
python trading_reverie.py --fork base_trading --sim run_mw_04   --steps 200 --seed 42 \
  2>&1 | tee /workspace/run_mw_04.log
python trading_reverie.py --fork base_trading --sim run_base_04 --steps 200 --seed 42 --no-middleware \
  2>&1 | tee /workspace/run_base_04.log

# The controlled comparison -- run BOTH directions
python paired_eval.py --sim paired_04_mw   --steps 200 --seed 42 --advance-with middleware \
  2>&1 | tee /workspace/paired_04_mw.log
python paired_eval.py --sim paired_04_base --steps 200 --seed 42 --advance-with baseline \
  2>&1 | tee /workspace/paired_04_base.log

# Multiple seeds (the thing that actually makes a claim possible)
for s in 42 43 44 45 46; do
  python paired_eval.py --sim paired_s$s --steps 200 --seed $s --advance-with middleware \
    2>&1 | tee /workspace/paired_s$s.log
done
```

**Read in this order:**

1. **`EXECUTED malformed`** — the headline. Middleware should be 0.
2. **`caught before execution`** — proves the 0 is not vacuous.
3. **`distinct episodes`** vs raw — if they diverge a lot, an agent is livelocked.
4. **`State contradictions`** — the only number that scores HOLDs.
5. **Discordant pairs** in the paired report — `baseline_only > middleware_only`
   is the actual middleware effect.

---

# ============================================================
# Runs 04–05, the frontend rebuild, and the run 06 sweep
# ============================================================

## 17. Run 04 and the `261ceec` / `da7b555` fixes

Run 04 was the first run with `--seed`, and it surfaced a hallucination rate
**above 100%** — more hallucinations than trade attempts, which is arithmetically
impossible if the denominator is right.

Two commits followed:

- **`261ceec`** — a retry that carried the illegal request forward was counted
  twice. Real bug, real fix.
- **`da7b555`** — `validate_response` now logs *why* it rejected something, and
  `paired_eval`'s non-advancing probe no longer writes into the order log (it was
  polluting the trade record with counterfactual orders).

**Correction on the record:** after run 04 these were reported as having resolved
the >100% problem. **They did not.** `261ceec` fixed a genuine double-count on a
different code path. The rate was still >100% in run 05, because the denominator
itself was wrong — see 19.1.

---

## 18. The demo UI, and its rebuild (`010960f`)

Unrelated to the metrics, done while the GPU runs were executing. The replay UI
was described as "basic and broken"; it was two separate problems.

### 18.0 What the demo UI is

Worth stating plainly, because it is easy to miss that this exists at all: a run
is not only a JSON report. There is a **live, watchable trading floor**, and it
is the fastest way to tell whether a run is behaving long before the report is
written.

Django 2.2 + Phaser 3.55.2, over a 44×32 tile LimeZu office map.

| URL | What it is |
|---|---|
| `/` | **Sim index** — every `storage/` dir with a `reverie/meta.json`, trading runs first, showing steps / decisions / LIVE badge |
| `/map/<sim_code>/<step>/` | **The trading floor** — the main view |
| `/trading_poll/<sim_code>/` | JSON the map polls every 4 s for new steps |
| `/trading_persona_state/<sim_code>/<step>/<Persona_Name>/` | per-agent detail panel, fetched on click |

```bash
cd environment/frontend_server && python manage.py runserver
# http://localhost:8000/
```

On screen: topbar with clock / step / **LIVE** dot, a five-symbol price ticker,
the office map, transport controls (prev / pause / next), a decision log, agent
cards, and a click-through detail panel carrying that step's reasoning and
retry/fallback status.

**The map is not decoration.** Agents sit at fixed desks (`sim_data.DESK_TILES`)
and **walk to the market board** (`MARKET_BOARD_TILE = [22, 4]`) on a `buy` or
`sell`, fanning out sideways when several trade at once; interactions put both
parties on `MEETING_ROW = 9`. A livelocked agent, or an arm that has stopped
trading entirely, is visible as *nobody ever leaving their desk* — which is how
the run-02 "100% analyze" and the run-03 livelock both looked on screen.

**Live mode:** the page renders once with what exists at load, then polls
`trading_poll` every 4 s and appends new steps, so a run in progress plays out
step by step as it happens. Polling stops when `is_running` goes false. Both the
initial render and the poll go through the same `sim_data.build_replay()`
ingestion layer, so there is no second copy of the parsing to drift.

The simulation runner is a **separate process** — the UI only reads what
`trading_reverie.py` writes to `storage/<sim>/`. Start it mid-run, or open a
finished run days later; it makes no difference.

Full walkthrough in [PROJECT_OVERVIEW.md §9](PROJECT_OVERVIEW.md).

### 18.1 The office map shipped with a stale Walls layer

`static_dirs/assets/office/visuals/office.json` had Walls gids **181/182** — a
transparent row of the tileset — where the current `build.py` emits **145/146**
(tan brick). Result: the office rendered with holes where walls should be.

252 cells differed. Only the Walls layer; the **Collisions layer was
byte-identical**, so `sim_data.py`'s desk / market-board / meeting-room tile
coordinates were unaffected and no pathfinding changed.

Regenerated by re-running `build.py` and remapping the block gids
(`32125 → 1073`) into the shipped JSON.

*Method note:* the first diff of this was wrong. It compared against the shipped
`office.tmx`, which was **itself stale**, and concluded "all layers SAME". The
reference has to be a fresh `build.py` run, not another checked-in artefact.

### 18.2 The trading UI was Bootstrap 3 with no zoom

- New `static_dirs/css/trading_ui.css` (~500 lines) — self-contained dark
  terminal theme. Replaces Bootstrap 3 and a leftover clustrmaps tracker.
- `trading_map.html` is now standalone rather than extending `base.html`.
- 2× render (`1408×1024`) plus `camera.setZoom(2)` and
  `image-rendering: pixelated`, so the pixel art is legible.
- Removed a permanently-visible 130×58 px `speech_bubble` image that covered
  each agent.
- Nameplates staggered by `2 + (agent_index % 3) * 7` px so co-located agents
  don't overlap.
- Decision-log auto-scroll now scrolls the `.log-scroll` pane only.
  `scrollIntoView({block:"nearest"})` was yanking the whole page.
- Landing page (`landing.html` + `translator/views.py::landing`) now indexes
  every sim in `storage/` that has a `reverie/meta.json`, trading runs first.

**Gotcha worth remembering:** Django parses `{% %}` tags **inside HTML
comments**. A commented-out `{% extends %}` produced
`<ExtendsNode: extends "base.html"> must be the first tag`. Don't write template
tags in HTML comments.

**Gotcha 2:** the dev server was running with `--noreload`, so `views.py` edits
were ignored while template edits took effect immediately — which made the
landing page look like a template bug for a while. Restart the server after
touching Python.

Verified with a live Django server, static-asset HTTP checks, poll-payload shape
validation and three headless-Chrome screenshots.

---

## 19. Run 05 analysis — four DGX logs

Four configs on the DGX (middleware, baseline, paired-advance-middleware,
paired-advance-baseline). Three defects with real consequences, and one metric
that turned out to be measuring the wrong thing entirely.

### 19.1 The 145.5% hallucination rate — the denominator excluded rejected actions

Alex Chen's decision composition:

```
29 buy  +  4 sell  +  48 invalid  +  119 hold   = 200
```

- `validate_response()` rejects any verb outside BUY/SELL/HOLD as
  `"action type is invalid"`, which **counts as a hallucination** → 48.
- The denominator tested `action in ("buy", "sell")` — and a rejected action's
  verb is neither. → 33.

`48 / 33 = 145.5%`. Replayed that exact composition to reproduce it. After the
fix: `48 / 81 = 59.3%`.

The same `("buy","sell")` test appeared in **four** places, and two of them had
behavioural — not just cosmetic — consequences:

**(a) The feedback gate — this is what caused the livelock.** `record_order_feedback()`
was gated on the same test, so a rejected *illegal-verb* request never entered
memory. 48 rejections produced **1 hallucination episode**: the agent was never
told it had been rejected, so it reissued the identical request forever.

**(b) The baseline arm was asymmetric.** In the middleware arm an unknown verb
was an `illegal_request` hallucination; in the baseline arm it fell through and
was silently nothing. **Identical model behaviour scored differently by arm** —
which is the exact confound the whole exercise exists to avoid.

### 19.2 The stale-context metric was measuring reasoning length

This is the largest finding, and it invalidates a headline number that had
already been quoted.

The detector was a substring scan for news keywords. Against the 12-headline
`SCRIPTED_NEWS` corpus, **only two keywords can ever fire**: `supply` and
`deliveries`. And `supply` is **seeded into Alex Chen's persona `currently`
field**, so he trips it by construction.

Measured false positives:

```
"Price action reflects normal supply and demand balance"          -> STALE
"The market report does not confirm supply tightening"            -> STALE   (declining to act!)
same decision, phrased concisely                                  -> clean
```

The last line is the problem. The metric scales with **how much the agent wrote**:

```
baseline    ~5,069 chars avg reasoning  ->  53.0% stale
middleware    ~786 chars avg reasoning  ->   3.9% stale
```

Memory compression shortens reasoning. So the reported **"5× stale-context
reduction" was a verbosity measurement, not a grounding measurement.**

Do not quote any stale-context number from runs 02–05.

### 19.3 The reflection parser failed all-or-nothing, and its fail-safe was "I am hungry"

Three separate faults in the same path:

1. `__func_clean_up` for `run_gpt_prompt_insight_and_guidance` parsed the whole
   response or nothing. One malformed line discarded every insight in the batch.
2. Its `get_fail_safe(n)` returned `["I am hungry"] * n` — inherited Stanford
   placeholder text. **That is the answer to "why is it always I am hungry":**
   every failed reflection wrote the literal string *"I am hungry"* into the
   agent's associative memory as a genuine thought, where it then surfaced in
   retrieval and in the UI.
3. `generate_insights_and_evidence` indexed `nodes[i]` blind on model-cited
   evidence line numbers. Statements are numbered from 0 by `enumerate` while the
   prompt's own example reads 1-based, so off-by-one is the *expected* case; a
   single bad index raised `IndexError` and the bare `except` discarded the whole
   reflection.

### 19.4 Findings reported but deliberately NOT fixed

- **Persona consistency has no discriminative power.** It returns `1.00` on
  ~99.8% of decisions (2 drift events in ~1,100). Not retuned — lowering the
  threshold until it fires would be manufacturing a finding.
- **`check_state_grounding` is sound.** It is arithmetic against the actual book,
  not a word search. It is the one detector that survived audit unchanged.
- **`paired_eval --advance-with` flips the conclusion** depending on direction.
  This is structural (see 14.2), not a code defect. It means the paired result is
  not robust and must not be reported as a single number.

### 19.5 Corrected estimate on the record

An earlier note estimated ~1,140 wasted Ollama calls from 228 log lines. Each
failure emits **two** log lines, so it is 114 failures × 5 calls ≈ **570**.

---

## 20. Fixes from the run 05 analysis

Two commits. Test suite is now **127 tests**; `tests/test_run05_defects.py` is
new and holds the regression coverage.

### 20.1 `264db38` — denominator, feedback gate, arm symmetry, reflection parser

**New predicate**, in `middleware/action_filtering.py`:

```python
NON_TRADE_ACTIONS = frozenset({"hold", "analyze"})

def is_trade_request(requested):
    action = (requested or {}).get("action")
    if not isinstance(action, str):
        return False
    return action.strip().lower() not in NON_TRADE_ACTIONS
```

The inversion is the whole point: **anything that is not explicitly a no-op is a
trade attempt**, including verbs nobody anticipated. The old allowlist could only
ever undercount.

Applied at all four sites:

| Site | File | Effect |
|---|---|---|
| Denominator | `trading_reverie.py` ~L1228 | 145.5% → 59.3% |
| Feedback gate | `trading_reverie.py` ~L1081 | rejections now enter memory → breaks the livelock |
| Baseline arm | `trading_reverie.py` ~L301 | unknown verb is now an `illegal_request` in *both* arms |
| Paired harness | `paired_eval.py` | attempt counts now match the main runner |

Reflection parser (`persona/prompt_template/run_gpt_prompt.py`):

- `__func_clean_up` rewritten to per-line tolerant parsing with three
  module-level regexes (`_INSIGHT_ITEM_RE`, `_INSIGHT_PAREN_RE`,
  `_INSIGHT_BECAUSE_RE`). One bad line no longer discards the batch.
- `get_fail_safe(n)` → `return {}`. **No more "I am hungry".**

`persona/cognitive_modules/reflect.py` — evidence indices clamped rather than
indexed blind:

```python
evidence_node_id = [nodes[i].node_id for i in evi_raw
                    if isinstance(i, int) and 0 <= i < len(nodes)]
```

Two bugs of my own were caught by the new tests before commit: a doubled list
prefix (`"1. " + "1. Sara"`), and a model refusal (`"I cannot answer that."`)
being salvaged as an insight — fixed with a `while` strip loop and an
`elif was_list_item and line_no > 0` guard respectively.

### 20.2 `4c6897e` — stale-context detector replaced outright

Not patched. Replaced. The old one could not be repaired because keyword presence
is not evidence of staleness.

New `_check_stale_reasoning()` requires **all four** conditions:

1. the cited headline is at least `STALE_MIN_AGE_STEPS = 20` steps old;
2. it has been **superseded** by a later same-symbol headline of opposite sign;
3. the agent names that symbol;
4. at least `STALE_MIN_TOKEN_HITS = 2` *distinctive* tokens match.

Supporting pieces: `_NEWS_STOPWORDS` (drops `supply`, `deliveries`, `revenue`,
`estimates`, `shares` — words that appear in ordinary market talk),
`_headline_tokens()`, `_distinctive_tokens()`.

**Verified length-invariant** from 138 to 4,416 characters of reasoning — the
same decision phrased long and short now scores the same.

Two more of my own bugs, both caught in test:

```python
# substring, not text.split() -- punctuation stays glued ("overblown;")
if any(t in text for t in distinctive.get(other_step, set())):
    continue
```

```python
# a headline that has not fired yet cannot supersede anything.
# current_step - other_step went negative and passed the recency test,
# so a step-75 agent scored stale against step-100 news.
if other_step > current_step:
    continue
```

**The future-news hole existed in the OLD detector too.** Any stale-context
number from runs 02–05 is affected by it independently of 19.2.

### 20.3 Comparability warning

**Run 06 stale-context figures are not comparable to runs 02–05.** Different
metric, deliberately. Do not put them in the same table or chart.

---

## 21. Run 06 — the multi-seed sweep (in flight)

`reverie/backend_server/run_sweep.sh` (`1619af1`).

```
STEPS=200   SEEDS=(42 43 44 45)   MAX_HOURS=10   OUT=/workspace/run06
```

**Why seeds and not longer runs:** every report so far has been n=1, and a single
seed cannot support a rate claim no matter how many steps it covers. On the DGX a
200-step run costs ~15–25 min (measured off run 05: median 1.0 s/LLM call, 779
calls per 200 steps), so the budget buys seeds.

**Design decisions in the script:**

- Uses `set -u`, **not** `set -e` — one crashed config must not kill the
  remaining seeds.
- A seed is finished completely (all 4 configs) before the next starts. Kill it
  at any point and every finished seed is a complete, analysable matched set.
  Interleaving would leave a half-populated grid that answers nothing.
- Preflight on Ollama `:11434` and both models (`mistral`, `nomic-embed-text`).
- `.done` markers, so re-running resumes instead of restarting.
- The budget check never kills a running config, so the real ceiling is
  `MAX_HOURS` + one config (~50 min).
- `python -u` — without it Python block-buffers into the log file and a live
  `tail` shows nothing for minutes, which reads as a hang.

**The four configs per seed:**

```bash
python -u trading_reverie.py --fork base_trading --sim run06_mw_s42 \
  --steps 200 --seed 42 --fresh
python -u trading_reverie.py --fork base_trading --sim run06_base_s42 \
  --steps 200 --seed 42 --fresh --no-middleware
python -u paired_eval.py --fork base_trading --sim run06_pmw_s42 \
  --steps 200 --seed 42 --fresh --advance-with middleware
python -u paired_eval.py --fork base_trading --sim run06_pbase_s42 \
  --steps 200 --seed 42 --fresh --advance-with baseline
```

**Running it:**

```bash
cd /workspace/Generative_agents/reverie/backend_server
chmod +x run_sweep.sh
nohup ./run_sweep.sh > /workspace/sweep.log 2>&1 &

tail -f /workspace/sweep.log            # START / OK / FAIL only
tail -f /workspace/run06/mw_s42.log     # a specific config's output
grep -c "^STEP " /workspace/run06/mw_s42.log
ls /workspace/run06/*.done | wc -l      # progress, out of 16
```

`sweep.log` is intentionally quiet — per-config stdout goes to
`/workspace/run06/<name>.log`. An apparently idle terminal is the normal state.

To watch output live instead, run the four commands under `tmux` with
`2>&1 | tee`. You lose the budget cap and `.done` resume; you gain visibility.
Do not run both at once — they contend for the same GPU and Ollama server.

**Model choice — staying on mistral.** The middleware is explicitly built for
small-model failure modes (`_split_combined_action`, `coerce_quantity`,
string-`"null"` handling). A larger model may not exhibit them, giving a floor
effect where the middleware has nothing to fix. There is also a seed/throughput
tradeoff: at 1.0 s/call four seeds fit in 10 h; 10× slower fits zero. And do not
change the model and the metric definitions in the same run — run 06 already
changes the stale metric.

`[Insufficient Permissions]` from `nvidia-smi` on the H100 is cosmetic — the
memory fields are blocked, the GPU is being used.

---

## 22. What to do next

In order. Items 1–3 are what the next session should actually pick up.

### 1. Analyse run 06 when the sweep lands

16 configs = 4 seeds × 4 arms in `/workspace/run06/`. Reports at
`environment/frontend_server/storage/<sim>/reverie/hallucination_report.json`.

Read each config in this order:

1. **Parser fallbacks %** — if high, every downstream metric is measuring almost
   nothing. (Caveat 6.2: this still conflates server failure with model failure.)
2. **Trade attempts** — `N/A` here means a policy problem, not a middleware result.
3. **`EXECUTED malformed`** — the headline. Middleware should be 0.
4. **`caught before execution`** — proves the 0 above is not vacuous.
5. **Distinct episodes vs raw** — divergence means an agent livelocked. The
   `264db38` feedback-gate fix should have reduced this substantially versus run
   05; that is a specific, checkable prediction.
6. **State contradictions** — the only detector that scores a HOLD.
7. **Discordant pairs** in the paired reports — and check whether the conclusion
   survives both `--advance-with` directions. If it flips, say so; do not report
   a single number.

Report across seeds as **median and range**, never a single seed's figure.

### 2. Make runs self-identifying before any two-model comparison

`OLLAMA_CHAT_MODEL` should be an env var, and the model name should be written
into `hallucination_report.json`. Right now a report does not record which model
produced it, so a mistral run and a phi3 run are indistinguishable after the
fact. **This must land before any second model is swept**, not after.

### 3. Write the comparability rules into this file

Two rules that keep being rediscovered:

- **Never compare a paired result to an independent-arm result.** Different
  populations, different denominators.
- **Run 06 stale-context is not comparable to runs 02–05.** Different metric
  (§20.2).

### Still open, lower priority

- **6.2** — `"OLLAMA ERROR"` sentinel still makes server failure look like model
  failure. Has not bitten a run yet.
- **6.6 / 6.7** — the documented-but-absent noise filter, and the poignancy scale
  that contradicts its own comment.
- **Persona consistency demonstrates nothing** (§19.4). Either find a scenario
  where it can fire honestly, or drop it from the write-up. Do not retune it into
  significance.
- **`hold` is still legal by construction.** An arm can improve its rate by
  trading less. `check_state_grounding` closes part of the gap only.
- **Tier 3 leftovers** (§5) — `_bar_index` never resets across days, and
  `is_open` ignores weekends. Latent; bites on any run past ~390 steps.

### Known non-issue

`map_office_001`'s `meta.json` still carries `sim_running: true` from a run that
died. That page will poll forever showing LIVE. Harmless; edit the flag if it
bothers you.

---

# ============================================================
# Run 06 — results, the three defects it exposed, and the fixes
# ============================================================

## 23. Run 06 analysis

16 configs = 4 seeds (42–45) × 4 arms × 200 steps. All 16 completed: 200 steps
each, a full report in every log, **0 `OLLAMA ERROR` and 0 tracebacks**, parser
fallbacks ≤2% everywhere.

That last point matters before anything else is read. §6.2 says the
`"OLLAMA ERROR"` string sentinel makes server failure look like model failure,
so fallback rates are normally suspect. This run the server was clean, so every
downstream number is measuring real model behaviour rather than infrastructure.

**The reports are in the stdout logs.** `_print_report` writes the whole thing
to stdout, so `/workspace/run06/<name>.log` is sufficient and
`hallucination_report.json` is not needed for analysis. Worth knowing before
copying files off the DGX again.

### 23.1 The one claim that survives

Median (range) across the four seeds, independent arms:

| | middleware | baseline |
|---|---|---|
| trade attempts | 21.5 (21–50) | 184.5 (66–203) |
| hallucination rate | 0.0% (0–46.0) | 66.9% (51.5–80.0) |
| caught pre-execution | 5.8 (0–23) | 126.5 (31–141) |
| **EXECUTED malformed** | **0** (0–0) | **4** (3–7) |

> **Middleware executed 0 infeasible requests in every seed. The baseline
> executed 3–7 in every seed.**

This is invariant across all four independent seeds *and* all eight paired
runs, and it is the same disposition claim §13.4 identified as the only valid
one. It survives every caveat below.

The baseline's failure mode is overwhelmingly **naked short selling**:
`unowned` runs 7–93 per agent per seed and dwarfs every `clamped_*` kind.
`base_s42` Marcus starts with $25k cash and **no positions**, then issues
`sell=100` — 60 of them infeasible. §14's C2 fix put explicit no-naked-sell
rules in both prompts; the baseline ignores them.

The middleware arm produces **zero** `unowned` and zero `clamped_*` in all four
seeds — only `illegal_request`. That is not better model behaviour, it is
§13.4 point 1: `get_legal_actions` builds the menu from `min(cash, risk_cap)`
and holdings, so the model cannot ask. **The kind-breakdown is not comparable
across arms. The disposition is.**

### 23.2 Defect 1 — the sweep did not deliver n=4

§21's entire stated purpose was "every report so far has been n=1, and a single
seed cannot support a rate claim". It did not work.

The middleware arm has **near-zero seed variance**:

```
Marcus Webb   buy=7  hold=193  PnL +$608.62   on ALL FOUR seeds
mw_s44 and mw_s45                             identical for all three agents
Alex Chen     0 trade attempts                in seeds 43, 44 and 45
```

Meanwhile the baseline arm varies 66–203 attempts across the same seeds.

**Cause.** `--seed` reached `HistoricalMarketEnvironment(seed=seed)` and nothing
else. The decision call ran at `temperature=0` — greedy decoding, which is
deterministic *regardless of seed*. Compression collapses the middleware
prompt to ~800 chars of largely seed-invariant content, so greedy decoding
returns the same answer every time; the baseline's 3,400–7,600-char context
still carries enough market-path variation to diverge.

So the middleware arm is effectively **n≈1.5, not n=4**, and its rate cannot be
reported as a median-and-range as though four seeds were independent.

Note the trap: seeding alone would have fixed nothing, and raising temperature
alone would have destroyed reproducibility. **Both are required, and neither
works without the other.**

### 23.3 Defect 2 — the paired harness ran different middleware

Same nominal configuration — middleware arm, seed 42, `--advance-with
middleware`, 200 steps, same fork:

| | `trading_reverie.py` (`mw_s42`) | `paired_eval.py` (`pmw_s42`) |
|---|---|---|
| mw trade attempts | 50 | 175 |
| mw avg context chars | 719–882 | 2,036 |

Retrieval was ruled out first: both call `new_retrieve(persona, focal_points,
n_count=15)` with the same three focal points, and both build context through
the same shared `build_memory_context()`. Those are identical.

**Cause.** `trading_reverie._step_agent` updates the ID-RAG identity graph every
step — `update_graph_current_situation()` plus `update_graph_belief()` for each
new thought node. `paired_eval._step_agent` **never called either**. So
`id_rag_anchor()` in the paired harness was reading a frozen bootstrap identity
graph while the independent runner read one refreshed every step.

The decisive evidence is that the two diverge at **step 1**, not gradually:

```
mw_s42   step 1   Marcus Webb: buy TSLA x13
pmw_s42  step 1   Marcus Webb: buy AMD  x9
```

Divergence at the first decision rules out trajectory drift. `update_graph_
current_situation()` runs *before* retrieval on step 1 in the runner, so the
graphs already differed by the first prompt.

**Consequence: the run-06 paired numbers do not describe the same middleware
the independent arm runs, and they predate the fix.** This is a stronger
statement than the existing "never compare a paired result to an
independent-arm result" rule (§7) — that rule was about populations and
denominators; this was the paired harness running a different middleware
configuration outright.

### 23.4 Defect 3 — an arm could abstain its way to a clean rate

The middleware arm's 0.0% is not a clean-behaviour result. It is `0/21` from an
arm that mostly declined to act, against the baseline's `66.9%` of 184. §15 and
§22 both warned "`hold` is legal by construction — an arm can improve its rate
by trading less"; run 06 is that warning firing at full volume, with Alex Chen a
**non-participant** in three of four seeds.

Nothing in the printed report distinguished that from an arm that traded
heavily and stayed clean. Both print `Total hallucinations : 0`.

Partly corroborating: `check_state_grounding` — the only detector that scores a
HOLD, and per §19.4 the one that survived audit unchanged — reports **1.5–2.5%**
contradictions in the middleware arm against **0.3–1.2%** in the baseline. The
abstaining arm's *holds* are slightly worse grounded. That is the honest
counterweight to the 0%.

### 23.5 Paired results — the conclusion flips, confirming §19.4

Discordant cells, `baseline_only` vs `middleware_only`:

| seed | advance=middleware | advance=baseline | |
|---|---|---|---|
| 42 | 63 vs 74 → **mw worse** | 145 vs 1 → mw better | **flips** |
| 43 | 64 vs 2 → mw better | 24 vs 6 → mw better | consistent |
| 44 | 58 vs 53 → mw better | 15 vs 30 → **mw worse** | **flips** |
| 45 | 56 vs 42 → mw better | 61 vs 8 → mw better | consistent |

**2 of 4 seeds flip on direction.** §19.4 called this structural and predicted
it; n=4 confirms it. Per §14.2 the paired result is not robust and **must be
reported as the table, never as a single number**. And per §23.3 these
particular numbers are superseded anyway.

### 23.6 Stale context — reportable, held as secondary

New detector (`4c6897e`), so **not comparable to runs 02–05** (§20.3).
Middleware 0–13 hits, baseline 0–100. The detector is verified length-invariant,
so this is not §19.2 repeating. But baseline stale hits still correlate with
both context size and trade volume, and Alex swings 0% (s43) to 50% (s42)
between seeds. Secondary evidence, not a headline.

### 23.7 Livelock — the `264db38` feedback-gate fix worked

§22 item 5's specific checkable prediction, and it came out right. Raw
hallucinations per distinct episode:

```
run 05 (baseline)   48 rejections -> 1 episode     ratio 48
run 06 (baseline)   117/28  144/52  34/10  144/52  ratio 2.8-4.2
```

Feeding rejections back into memory broke the livelock. One residual:
`mw_s42` Alex is 23/3 (ratio 7.7), the worst remaining, and it is the only
middleware seed with any hallucinations at all.

---

## 24. Fixes from the run 06 analysis

All three defects above are fixed in the working tree. Test suite is now
**141 tests**; `tests/test_run06_defects.py` is new and holds the regression
coverage (14 tests).

### 24.1 Defect 1 — the seed now reaches the sampler

| File | Change |
|---|---|
| `persona/prompt_template/gpt_structure.py` | `ollama_request(..., seed=None)`; sets `options["seed"]` only when given |
| `trading_reverie.py` | `make_trading_decision(..., seed, temperature)`; threaded from `TradingReverie.__init__`; new `--temperature` CLI arg |
| `paired_eval.py` | `_call_llm` forwards `self.temperature` / `self.seed`; new `--temperature` |
| `run_sweep.sh` | explicit `TEMPERATURE=0.7`, passed to all four configs |

`seed` is **omitted** rather than defaulted to 0 when not supplied — a default
of 0 would silently pin every unseeded call site to one sampler state.

**Default temperature is now 0.7** (the value §6.1 recommended). This is a
deliberate behavioural change: at 0 the fix is inert. Use `--temperature 0` only
to reproduce a pre-run-07 result.

**Comparability warning: run 07 absolute rates are not comparable to run 06.**
Sampling changed. What *is* comparable is the disposition claim (§23.1) and the
seed-variance question, which is the point of the re-run.

Per §11 item 8 — do not change the model in the same run. Run 07 stays on
mistral.

### 24.2 Defect 2 — the paired harness now updates the identity graph

`paired_eval._step_agent` gained the block that mirrors
`trading_reverie._step_agent`: capture `len(persona.a_mem.seq_thought)` before
`reflect()`, then `update_graph_current_situation()` and `update_graph_belief()`
for each new thought.

Tests assert both harnesses reference both functions, so the two cannot drift
apart again silently — which is how this defect survived from `A2` (§14) until
now.

### 24.3 Defect 3 — abstention is reported next to the rate

New in `hallucination_report.json`:

```
top level : total_abstention_rate_pct, participating_agents, total_agents
per agent : abstention_rate_pct, participated
```

Printed directly under the headline rate it qualifies:

```
Abstention (HOLD)    : 96.5% of decisions   <- the rate above falls as this rises
  !! 1 of 3 agents requested NO trades at all. The rate above rests entirely on the other 2.
```

and per agent:

```
    Abstention     : 100.0% HOLD   <- NON-PARTICIPANT: requested no trades all run
```

The `State contradictions` line now says "includes HOLDs, so it scores an
abstaining arm too", since that is the one detector that can score the arm the
abstention warning is about.

These are **companions to the rate, never replacements**. Same principle as
§6's raw-and-distinct episodes rule: collapsing is a way to read the data, not
a correction to it.

### 24.4 Verification

- 141 tests pass (127 existing + 14 new).
- `bash -n run_sweep.sh` clean; all three patched Python files compile.
- **The LLM path is verified on the DGX** (2026-08-18). Two 20-step runs at
  `--temperature 0.7`, seeds 42 and 43, `diff`ed on their decision lines. The
  diff is non-empty and carries real behavioural divergence rather than
  reordering:

  ```
  Marcus Webb   buy NVDA x22   vs   hold None x0
  Marcus Webb   buy NVDA x22   vs   buy NVDA x16
  Sara Kim      buy AAPL x46   vs   buy AAPL x19
  Alex Chen     sell NVDA x50  vs   (absent)
  ```

  Different actions, symbols and sizes. Ollama honours the `seed` option and
  the seed now reaches agent behaviour — which the mocked payload test could
  not establish, only that the value was sent.

  Also visible at 20 steps: **Alex Chen trades again.** He was a
  non-participant in three of four run-06 seeds (200 holds, 0 requests). Read
  this as encouraging, not as a result — 20 steps is not 200, and the new
  abstention line (§24.3) is what should judge it.

- **Still open at the time of writing: whether the divergence survives 200
  steps.** The smoke test proves the seed reaches the sampler; it does not
  prove the arms stay apart once compression settles and the prompts converge.
  That is the first thing §25 item 2 checks in the run-07 reports.

---

## 25. What to do next

Supersedes §22.

### ~~1. Smoke-test the seed fix~~ — DONE, PASSED (2026-08-18)

Kept for the record, and because it is the right gate before any future
sampling change. Two short runs, different seeds, same everything else:

```bash
cd /workspace/Generative_agents/reverie/backend_server
python -u trading_reverie.py --fork base_trading --sim seedcheck_a \
  --steps 20 --seed 42 --temperature 0.7 --fresh 2>&1 | tee /workspace/seedcheck_a.log
python -u trading_reverie.py --fork base_trading --sim seedcheck_b \
  --steps 20 --seed 43 --temperature 0.7 --fresh 2>&1 | tee /workspace/seedcheck_b.log

diff <(grep "decision:" /workspace/seedcheck_a.log) \
     <(grep "decision:" /workspace/seedcheck_b.log)
```

**A non-empty diff is the pass condition.** An empty diff means the seed is not
reaching agent behaviour and everything below is a waste of GPU time. ~5
minutes, and it protects the whole sweep.

**Result: non-empty, with genuine behavioural divergence** — evidence in §24.4.
Run this gate again after any change to temperature, model, or the decision
prompt.

### 2. Run 07 — the re-sweep  ← CURRENT

`run_sweep.sh` is already updated (`OUT=/workspace/run07`, `TEMPERATURE=0.7`,
`run07_*` sim names). Same 4 seeds × 4 configs × 200 steps, 10 h cap, `.done`
resume.

```bash
cd /workspace/Generative_agents/reverie/backend_server
chmod +x run_sweep.sh
nohup ./run_sweep.sh > /workspace/sweep07.log 2>&1 &
```

Read the results in this order — the first item is new and is the one that
decides whether the sweep worked at all:

1. **Seed variance in the middleware arm.** Compare action distributions and
   PnL across `run07_mw_s42..s45`. If Marcus is `buy=7 hold=193 PnL +$608.62`
   on all four again, defect 1 is not fixed. Everything else is moot.
2. **Abstention %** and the non-participant warning — is the middleware arm
   still buying its rate by not trading?
3. **Parser fallbacks %** — caveat 6.2 still applies.
4. **`EXECUTED malformed`** — the headline. Middleware should be 0.
5. **`caught before execution`** — proves the 0 is not vacuous.
6. **Distinct episodes vs raw** — should stay at the run-06 ratio of ~3.
7. **State contradictions** — the honest counterweight to a low rate.
8. **Discordant pairs**, both `--advance-with` directions. Expect the flip to
   persist; report the table.

Median and range across seeds, never a single seed's figure.

### 3. Make runs self-identifying (carried forward from §22 item 2)

`OLLAMA_CHAT_MODEL` should be an env var, and the model name **and now the
temperature and seed** should be written into `hallucination_report.json`.
Right now a report does not record what produced it. Run 06 vs run 07 will
differ by sampling settings that appear nowhere in the artefacts, which is
exactly the confusion this prevents. **Must land before any second model is
swept.**

### Still open, lower priority

- **6.2** — `"OLLAMA ERROR"` sentinel still conflates server failure with model
  failure. Has not bitten a run yet; run 06 was clean.
- **6.6 / 6.7** — the documented-but-absent noise filter, and the poignancy
  scale that contradicts its own comment.
- **Persona consistency demonstrates nothing** (§19.4). Run 06 confirms it:
  `avg=1.00`, drift events 0–8 out of ~1,200 decisions. Either find a scenario
  where it can fire honestly, or drop it from the write-up. Do not retune it
  into significance.
- **Tier 3 leftovers** (§5) — `_bar_index` never resets across days, `is_open`
  ignores weekends. Latent; bites past ~390 steps.

### Known non-issue

`map_office_001`'s `meta.json` still carries `sim_running: true` from a run that
died. That page polls forever showing LIVE. Harmless.

