
# Agent Anchor — Frontend Integration Task (for Claude Code)

**Read `AGENT_ANCHOR_CONTEXT.md` first if it's in this repo/chat — it has full project state.**
Style: direct, terse. No unnecessary abstraction. Don't fabricate file paths, schemas, or numbers — inspect the actual repo before writing code that depends on them.

---

## 1. Goal

Two frontend deliverables for the 3-agent trading sim:

1. **Trading dashboard (new build)** — portfolio value over time, current positions, price chart, per-timestep decision log with reasoning.
2. **Map/character UI (adapted, not built from scratch)** — the original Park et al. Django + Phaser.js Smallville UI, reduced from 25 agents to 3 trading agents (Alex Chen, Sara Kim, Marcus Webb).

Both live in the same Django project and share one data-ingestion layer — don't write two separate parsers for the same logs.

## 2. Repos

- **Reference only (do not blindly clone/vendor wholesale):** `joonspk-research/generative_agents` — the original Park et al. repo. Frontend lives in `environment/frontend_server/` (Django app + Phaser.js map). Pull specific files/patterns from here as reference architecture, not a full copy-paste.
- **Target (single active repo):** `Kritiiguptaa/Generative_agents` — this already has an old, untouched copy of the Park et al. frontend sitting inside it. **First step is inventorying what's actually there** — don't assume it matches upstream or is even runnable.

## 3. Data flow — default decision

No live sim loop / websocket exists yet. Default to **replay mode**: the frontend reads from log files produced after a sim run finishes, not live during a run.

- Build a single data-loading module/service as the seam between logs and both UIs, so live mode can be added later without a rewrite.
- **Do not assume log schemas or paths.** The repo has CSV logging from `action_filtering.py` (retry vs. fallback distinction matters — don't drop it), output from `historical_market_environment.py`, and hallucination score CSVs/JSON. Locate and inspect the actual files first; derive the schema from what's really there.

## 4. Task breakdown

### Step 1 — Inventory
- What's in the copied frontend folder: does it run at all, which Django version, what static assets exist, how stale is it vs. upstream.
- Locate actual sim output files and their real columns/shape (CSV/JSON) — this defines the data contract for step 4.

### Step 2 — Map/character UI adaptation
- Reduce to 3 agents: Alex Chen (hedge fund manager), Sara Kim (momentum trader), Marcus Webb (retail day trader).
- Map art: reskin the scene to a trading environment (e.g. trading floor / desk setup / office with screens) instead of Smallville's residential map — this should read as *your* trading sim, not a repurposed social sim. Doesn't need custom pixel art from scratch: reuse/recolor existing tileable assets from the original repo's tileset where possible, swap sprites/backgrounds where it matters most (agent desks, a shared "market" tile showing current prices, etc.). Cosmetic fidelity to Smallville is explicitly not the goal — thematic fit to trading is.
- Per-agent state panel: current action, reasoning, persona snippet — sourced from replay data.
- Timestep scrubber/playback control (the original repo isn't built for this at small scale — implement it).

### Step 3 — Trading dashboard (new)
- Portfolio value over time, per agent.
- Current positions/holdings table.
- Price chart (from `historical_market_environment.py` data).
- Decision log: timestep, agent, action, symbol, qty, reasoning, and whether the record was a retry or a fallback HOLD (preserve this distinction — it's load-bearing for CVR reporting later).
- Surface CVR/fallback rate if the data supports it; don't compute metrics that require a full experimental run that hasn't happened yet.

### Step 4 — Wiring
- One Django project, two views (dashboard, map), one shared ingestion layer.
- No duplicated parsing logic between the two UIs.

## 5. Explicit don'ts

- Don't fabricate log schemas, file paths, or metric values — inspect real files first.
- Don't drop the retry-vs-fallback distinction from output validator logs.
- Don't swap frameworks — Django + Phaser.js stays as-is.
- Don't try to wire live/websocket mode now — replay only, leave the seam for later.
- Don't treat the hallucination baseline (0.214) as valid data to display as-is — it's flagged as a suspected structural defect (see context file §6); if surfacing it anywhere, label it clearly as unvalidated.

## 6. If something's ambiguous

Inspect the repo before asking — you have direct access I don't. Only ask Kriti if the repo itself doesn't resolve it (e.g., a genuine design preference, not something discoverable by reading code).

---

## Opening prompt to paste into Claude Code

> Working on Agent Anchor's frontend. Read `FRONTEND_INTEGRATION_TASK.md` first, then inventory the current frontend folder and actual sim log outputs in this repo before writing any code. Don't assume schemas — check the real files. Start with Step 1.
