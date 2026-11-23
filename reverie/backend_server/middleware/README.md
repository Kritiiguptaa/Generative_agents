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
