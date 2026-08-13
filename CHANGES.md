# Trading simulation — hallucination metric fixes

Working log of what was changed, what was deliberately left alone, and what is
still open. Written for the trading-sim work on branch `armaans-frontend`.

**Status as of the latest entry:** Tier 1 and Tier 2 are committed and pushed
(`8ad0fe2`). Tier 3 is untouched. A further round of findings (section 6) is
diagnosed but not fixed.

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

### 6.1 Temperature 0 — likely explains `analyze=200`

`ollama_request` defaults to `temperature=0` (`gpt_structure.py:55`) and the
trading call does not override it (`trading_reverie.py:149`). That is greedy
decoding: same prompt in, same tokens out, every time. The prompt barely changes
step to step, so the model lands on the same answer repeatedly.

The agents are not making 200 decisions. They are making one and echoing it.

**Fix:** `temperature≈0.7` on the decision call. Leave embeddings at 0.

*Confidence: the setting is a fact; that it causes the 200/200 is a strong
hypothesis, not proven.*

### 6.2 `"OLLAMA ERROR"` makes infrastructure failure look like model failure

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

1. **6.1 (temperature) and 6.2 (error sentinel)** — few lines each, and both
   change what the numbers mean. Without 6.1 you are measuring one decision per
   agent; without 6.2 the fallback rate is untrustworthy.
2. **Tier 3 item 3** (compression failing silently) — cheap, and it protects the
   interpretation of the run.
3. **6.3 and 6.4** — before writing anything up. A stale-context metric that
   fires on the word "supply" is not defensible, and a silently failing planner
   makes the agents less distinct than they look.
4. Remaining Tier 3, then 6.5–6.7.

---

## 9. Run commands

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
