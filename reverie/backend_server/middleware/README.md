# middleware/

What each file in this package is, and whether it runs.

This file exists because the package used to contain five files that did
nothing: four were empty and one held a commented-out class skeleton. Reading
the directory listing suggested a system with eight or nine components, the
paper's capstone report described four layers, and the code ran three. Anyone
reasoning about the architecture from the filenames was reasoning about a
design that was never finished.

## The pipeline that actually runs

Composed by `MiddlewareWrapper`, in this order. Both harnesses,
`trading_reverie.py` and `paired_eval.py`, call `prepare()` then `decide()` and
compose nothing themselves. A test fails if either calls a stage directly, see
`tests/test_wrapper_ablations.py`.

| File | Status | What it does |
|---|---|---|
| `middleware_wrapper.py` | **live** | The single entry point. Composes the three stages below and carries the `components` ablation switch. |
| `memory_compression.py` | **live** | Stage 1. Flatten and dedup by node id then cosine 0.95, keep newest 2 per symbol, annotate past 20 steps, trim to a 350-token budget. No LLM call. |
| `id_rag.py` | **live** | Stage 2. Retrieves the top 4 identity nodes by cosine over `nomic-embed-text`, always appends the forbidden node, truncates to 300 chars, and refreshes the graph each step. |
| `action_filtering.py` | **live** | Stage 3. Builds the legal menu from `min(cash, risk_cap)` and holdings, validates the response, retries once with the error appended, then substitutes a legal HOLD. Also holds the format check. |
| `metrics_recorder.py` | **live** | Records every metric and computes every rate. Refuses a disposition without its request, which is what prevents measuring downstream of the filter. |
| `config/middleware_config.py` | **live** | Per-stage settings for compression and ID-RAG. |

### Order is load bearing

ID-RAG runs **after** compression. The identity block is prepended to the
already-budgeted memory block; if it ran first it would sit inside the budget
compression enforces and could be trimmed away.

## Legacy, kept deliberately

| File | Status | Why it is still here |
|---|---|---|
| `persona_anchoring.py` | **part live, part legacy** | `score_persona_consistency()`, `hold_rate_drift()` and `PERSONA_DRIFT_THRESHOLD` are live, used by both harnesses as a post-hoc measure. `create_persona_anchor()` and `anchor_memory_context()` are legacy with no caller: ID-RAG replaced them. |

**This is the "layer 3" discrepancy.** The capstone report describes layer 3 as
persona anchoring, an intervention that builds an identity block. In the code
that function has no caller, and what runs is a measurement applied after the
decision. So layer 3 is a metric, not a layer; the anchoring intervention is
ID-RAG (stage 2). The metric used to be saturated, 0.98 to 1.00 in both arms,
because it read only the reasoning's wording. It now also scores the
pre-filter request (buying off the watchlist, scoreable with no reasoning) and
flags an aggressive agent that HOLDs most of a run (`hold_rate_drift`). Numbers
from before that change are not comparable; see WP-G and WP9 before citing it.

## Removed

Deleted rather than documented, because an empty file named after a component
is a claim that the component exists.

| File | Was | Why removed |
|---|---|---|
| `input_stabilizer.py` | 2,813 bytes of real but dead code | The **original four-layer orchestrator**: density gating, then compression, then persona anchoring, then action filtering. It imported `SemanticDensityGating`, `MemoryCompression`, `PersonaAnchoring` and `ActionFiltering`, none of which exist, so it could not be imported at all. It is where the capstone's four-layer description comes from, and it never ran. `MiddlewareWrapper` is what it was meant to be, over the three stages that exist. |
| `output_validator.py` | 0 bytes, never imported | Validation already lives in `action_filtering.validate_response`. A second validator could only drift from the first, and the detector-position bug is exactly what this package has already been burned by. |
| `semantic_density_gating.py` | 5 commented lines | Would filter memories by information content, which overlaps compression's dedup and recency pruning and would make the compression ablation ambiguous. Not built by decision, not by omission. |

Their config fields went too: `enable_stabilizer`,
`entropy_discard_patterns` and `persona_anchor_template`. A config field that
nothing reads is worse than an absent one, because it reads as a supported
switch.

## Still a stub, on purpose

| File | Status | Note |
|---|---|---|
| `chaos_injector.py` | **0 bytes, undecided** | Planned as an experiment layer, outside the middleware, perturbing what both arms receive identically. It needs its own sweep, so it is pending a decision rather than abandoned. Claim nothing for it in the paper while it is empty. |

## The ablation switch

`WrapperConfig.components` selects which stages run. The default is all three,
which is the shipped arm and the only configuration whose numbers may be
reported as "the middleware".

```python
WrapperConfig()                                      # shipped middleware
WrapperConfig(components=frozenset({"compression"}))  # research arm
WrapperConfig(components=frozenset({"idrag"}))        # research arm
WrapperConfig(components=frozenset({"filter"}))       # research arm
```

An unknown component name raises. An arm silently missing a stage would be
reported as the full middleware and would put a wrong row in the ablation
table.

`is_full_middleware` distinguishes the shipped arm from a research arm. A mixed
arm answers what one stage contributes; it is not the system under test.

## The baseline is not in here

The baseline arm keeps its own path in `trading_reverie.make_trading_decision`
and is deliberately **not** routed through the wrapper. It is the control, and
an arm assembled from zero components would resemble it without being it.
Conflating the two would make the comparison circular.

## Future work

### Persona evidence for agents that mostly hold

**The gap.** `score_persona_consistency()` scores a decision on two things: the
language check, which exists only for conservative and aggressive profiles, and
the order itself (a BUY of an off-watchlist symbol). A moderate profile has no
characteristic vocabulary, and a HOLD is not an act, so a moderate agent that
mostly holds has almost nothing scored. In the first 100-step nine-agent runs
Alex Chen had 0 of 100 decisions scored, Tomas Herrera 1 of 100 and Sara Kim 17
of 100, against 100 of 100 for every conservative and aggressive agent. Their
persona average is therefore built on a handful of buys, or is `n/a`. This is a
consequence of scoring what the agent did (4c79fa4) and of two false-positive
fixes (the "maximum position" phrase and HOLD commentary), not a defect in any
of them: the removed checks were measuring the wrong thing.

**What is in place.** `analyze_sweep.py` prints the share of decisions the
persona score covers next to every persona average, per arm and bucket and per
agent, and calls out any bucket under 50%. Persona scores are therefore never
read without their coverage. `analyze_sweep.py --rescore` recomputes the persona
numbers from each sim's `trading_log.json` with the current scorer, so a scorer
change can be applied to a finished sweep without rerunning it (keep the sim
folders under `storage/`). Until the check below exists, report persona
consistency for moderate agents as "buys only, n = X" and do not interpret it.

**The best fix: score a HOLD against the persona's own rules.** Every persona
file carries numeric rules in `active_constraints` (Alex Chen: `stop_loss_threshold` 0.08,
`profit_take_threshold` 0.15, "cut losers at 8% without exception, trim winners
at 15%"), and the run already tracks every position's `avg_price` and the
current prices. A HOLD that sits through a breach of the agent's own stop-loss
or profit-take rule is out of character on an objective, state-grounded
criterion; a HOLD with no rule triggered carries no evidence either way, which
is the existing "not scoreable" rule. This beats keyword lists for three reasons:

- It uses no vocabulary. The language check's recent bugs ("all in" inside
  "fall in", "panic" describing the market) are the failure mode of matching
  words; a price against a cost basis cannot make that mistake.
- It gives moderate agents a real test. Alex's "never chases, cuts losers at 8%,
  trims winners at 15%" is the substance of a moderate profile.
- It is deterministic and cheap, and needs no judgement call about which phrases
  are in character.

**What it needs before it can be built.**

1. Nothing reads `active_constraints` today. Check that the thresholds are
   present and sensible for all nine personas: Marcus Webb and Sara Kim have no
   `profit_take_threshold`, so for them only the stop-loss rule applies.
2. It can be computed offline. Each `trading_log.json` entry stores the
   decision's reasoning, the requested order, the agent's positions with cost
   basis (`qty`, `avg_price`) and the market prices at that step, so a stop-loss
   or profit-take breach can be evaluated from the log alone, without a rerun.
   The same is true of the whole persona score: reasoning, order, and the
   persona's watchlist and risk tolerance are all recoverable, so a change to the
   scorer can be applied to a finished sweep as long as its `storage/<sim>/reverie/`
   folders are kept. (`score_persona_consistency()` would take the prices as an
   argument, as the harness already passes the order.)
3. Coverage will be low and should be measured, not assumed. Most agents hold
   cash for most of a run (about 7% of decisions were trades), so rules trigger
   rarely. The honest outcome may be a small but objective set of scored holds,
   which is still better than none.

**Second: replace the absolute hold-rate limit with a relative one.**
`hold_rate_drift()` flags an aggressive agent holding more than 50% of the run.
In run07 the aggressive agent held 84-96.5% in all eight runs, baseline arm
included, so the flag fires every time and carries no information. A roster-level
check is more useful: flag when the profile ordering is inverted (an aggressive
agent holding more than the conservative bucket's mean in the same run), and
compare each agent's hold rate with the baseline arm's for the same seed.

**Last resort: per-persona rationale lists.** A hold-reason lexicon per persona
(in-character and out-of-character reasons) would score the reasoning text. It
is keyword matching, so it inherits the weaknesses above, and it adds a
subjective choice about what "in character" means for each of nine personas. Do
it only if the rule check cannot be built, validate it against real logs first,
and match whole words.

**Not recommended.** An LLM judge for character. It is expensive, not
deterministic, and substitutes one model's opinion of a persona for a measurement.
