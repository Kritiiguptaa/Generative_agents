"""
File: sim_data.py
Description: Shared ingestion layer for trading-sim replay data. Reads the
real files produced by reverie/backend_server/trading_reverie.py and
middleware/action_filtering.py for a given sim_code and normalizes them into
one per-step, per-agent structure that both the map view and (later) the
trading dashboard consume. This is the single seam -- no duplicated parsing
between the two UIs (see FRONTEND_INTEGRATION_TASK.md, section 4).

Replay-only: reads finished/in-progress log files off disk. No live/websocket
mode -- that seam is intentionally left for later.
"""
import csv
import json
import os

STORAGE_ROOT = "storage"

ACTION_ICONS = {
    "buy": "\U0001F4C8",   # up chart
    "sell": "\U0001F4C9",  # down chart
    "hold": "✋",      # raised hand
}

# ---------------------------------------------------------------------------
# Office map geometry (static_dirs/assets/office/visuals/munder_office.json,
# vendored from Munder Difflin -- see ATTRIBUTION.md in that directory -- 34x22
# tiles @16px). Every tile below was read out of the map's own spawn-points /
# zones object layers and its collision layer (see office.tmj), not eyeballed.
#
# This replaces the project's original 44x32 office.json geometry (the old
# comments/constants are preserved in git history). Swapping the map is a
# presentation-layer change -- it does not touch how trading_reverie.py or
# action_filtering.py decide what an agent does, only where that decision
# renders on screen.
# ---------------------------------------------------------------------------

# The map's own spawn-points layer names six desks "pc-1".."pc-6" in a row at
# y=13 (tile), three tiles apart. One per trading agent, in persona order.
DESK_TILES = {
    "Alex Chen": [2, 13],
    "Marcus Webb": [6, 13],
    "Sara Kim": [10, 13],
}

# The map's "boardroom" zone (tiles x:9-17, y:3-7) is the meeting-room-with-a-
# screen equivalent of the old map's presentation screens -- this tile sits
# inside it on open floor (collision-verified), the "check the market board"
# spot agents walk to on a real BUY/SELL.
MARKET_BOARD_TILE = [13, 7]

# Walkable extent of the boardroom's open floor row (collision layer: y=7 is
# clear from x=9 to x=17). Several agents commonly BUY on the same step, so
# they're spread along this row instead of all stacking on one tile.
MARKET_BOARD_SPAN = (9, 17)

# The "pc" desk row (y=13) runs walkable from x=1 to x=24 (collision-verified).
# Fallback meeting spot, used only when an agent has no known desk position.
MEETING_ROW = 13

# The map's "warroom-seat" spawn point -- a walkable tile just past the
# boardroom zone, reachable from every desk via the open floor. Agents meet
# here rather than at their own desks, which read as "barely moved".
MEETING_ROOM_TILE = [23, 7]

# trading_interactions.maybe_interaction() records a meeting on a single step,
# but a conversation that appears and vanishes inside one 1600 ms replay tick is
# unreadable -- and the walk down is ~33 tiles, longer than one tick. So the
# meeting is held on screen for this many steps.
#
# The later steps are *visual* only: the interaction itself really did happen on
# one step. To keep the map from contradicting the decision log, an agent is
# only held in the lounge on a follow-on step if their actual logged decision
# there was a no-op. Anyone who traded leaves for the market board as normal.
MEETING_HOLD_STEPS = 3

# Break area: the map's own cafeteria zone (tiles x:24-31, y:12-19), with a
# "cafe-stand-coffee" spawn point at exactly this tile -- walkable and open in
# the collision layer, right in front of the coffee machine.
#
# BREAK_STRIDE spaces the agents two tiles apart along the same open row
# (x=25..30 is clear) instead of crowding a single tile.
BREAK_ROOM_TILE = [26, 20]
BREAK_STRIDE = 2

# An agent that holds this many steps running is sent to the break area instead
# of sitting motionless at their desk.
#
# This is a *rendering* of a real property -- the length of an unbroken HOLD
# streak, read straight off the decision log -- and not a simulated event: the
# sim has no notion of a coffee break. It earns its place because over-caution
# is Alex Chen's designed failure mode and is otherwise completely invisible on
# the map, where "holding for 30 steps" and "holding once" look identical. The
# moment the agent trades, the streak resets and they go back to work.
HOLD_STREAK_FOR_BREAK = 6


def _sim_path(sim_code, *parts):
    return os.path.join(STORAGE_ROOT, sim_code, *parts)


def _load_json(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_meta(sim_code):
    """reverie/meta.json -- persona list, sec_per_step, start date, current step."""
    return _load_json(_sim_path(sim_code, "reverie", "meta.json"), {})


def load_desk_positions(sim_code, persona_names):
    """
    Static desk tile per agent. Trading agents never move -- trading_reverie.py
    never rewrites environment/{step}.json after step 0 -- so the
    highest-numbered environment/*.json is authoritative for every step.
    """
    env_dir = _sim_path(sim_code, "environment")
    positions = {}
    if not os.path.isdir(env_dir):
        return positions
    steps = sorted(
        int(f.split(".")[0]) for f in os.listdir(env_dir) if f.endswith(".json")
    )
    if not steps:
        return positions
    latest = _load_json(os.path.join(env_dir, f"{steps[-1]}.json"), {})
    for name in persona_names:
        if name in DESK_TILES:
            # DESK_TILES wins over environment/*.json for agents we know.
            #
            # That file is vestigial: it is written once at fork time and the
            # trading loop never updates it, and base_trading ships the aisle
            # tiles beside each cluster (17/22/27 on y=8) from before the office
            # map existed. Honouring it renders every agent standing a tile
            # clear of their own desk. DESK_TILES holds the chairs, which is
            # this map's actual answer to "where does this agent sit".
            positions[name] = list(DESK_TILES[name])
        elif name in latest:
            # Anyone we have no desk for -- a renamed persona, or an inherited
            # village sim -- still gets their recorded position rather than
            # being dumped at [0,0] inside a wall.
            positions[name] = [latest[name]["x"], latest[name]["y"]]
    return positions


def load_persona_snippet(sim_code, persona_name):
    """scratch.json fields for the per-agent panel's static persona snippet."""
    scratch = _load_json(
        _sim_path(sim_code, "personas", persona_name, "bootstrap_memory", "scratch.json"),
        {},
    )
    return {
        "name": scratch.get("name", persona_name),
        "innate": scratch.get("innate", ""),
        "learned": scratch.get("learned", ""),
        "currently": scratch.get("currently", ""),
        "trading_strategy": scratch.get("trading_strategy", ""),
        "trader_type": scratch.get("trader_type", ""),
        "risk_tolerance": scratch.get("risk_tolerance", ""),
        "watchlist": scratch.get("watchlist", []),
        "cash_balance": scratch.get("cash_balance"),
        "positions": scratch.get("positions", {}),
    }


def load_trading_log(sim_code):
    """
    reverie/trading_log.json -- written once at the end of trading_reverie.py's
    run(). List of per-step, per-agent decision/outcome/portfolio records. May
    not exist yet for an in-progress or crashed run -- callers must handle [].
    """
    return _load_json(_sim_path(sim_code, "reverie", "trading_log.json"), []) or []


def load_action_filter_log(sim_code):
    """
    reverie/action_filter_log.csv -- written incrementally, one row per agent
    decision, by middleware/action_filtering.py's log_action(). Columns:
    timestamp,agent,action,symbol,quantity,status,error_reason. status is
    "success" | "retry" | "fallback" -- this distinction must survive into
    the UI; it's load-bearing for CVR reporting later (don't drop it).
    Has no step column, so rows come back in file order (== chronological ==
    decision order) for the caller to align against trading_log.json by
    per-agent ordinal position.
    """
    path = _sim_path(sim_code, "reverie", "action_filter_log.csv")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_hallucination_report(sim_code):
    """reverie/hallucination_report.json, written once at the end of a run."""
    return _load_json(_sim_path(sim_code, "reverie", "hallucination_report.json"))


def load_interactions(sim_code):
    """
    reverie/trading_interactions.json -- real peer exchanges produced by
    trading_interactions.py's maybe_interaction(), which fires every 12 steps
    and has an LLM generate an actual conversation between two of the agents.
    Each entry: {step, topic, summary, follow_up, symbol, agents:[a, b]}.
    Written incrementally by trading_reverie.py's run() (same per-step flush as
    trading_log.json), so this is live-safe. Absent/[] for runs shorter than
    the 12-step cadence -- callers must handle that.
    """
    return _load_json(_sim_path(sim_code, "reverie", "trading_interactions.json"), []) or []


def _meeting_tiles(agent_a, agent_b, desk_positions, map_width=44):
    """
    Where two interacting agents stand to talk: the lounge on the lower floor,
    offset by one tile so they stand side by side rather than overlapping.

    Falls back to the bullpen aisle only if we have no desk position for either
    agent, which means the sim predates the office map and its tiles cannot be
    trusted to be walkable.
    """
    if agent_a in desk_positions or agent_b in desk_positions:
        mx, my = MEETING_ROOM_TILE
        mx = max(0, min(mx, map_width - 2))
        return [mx, my], [mx + 1, my]

    ax = desk_positions.get(agent_a, [0, 0])[0]
    bx = desk_positions.get(agent_b, [0, 0])[0]
    mid_x = (ax + bx) // 2
    # Keep the pair inside the map even if the midpoint lands on the last column.
    mid_x = max(0, min(mid_x, map_width - 2))
    return [mid_x, MEETING_ROW], [mid_x + 1, MEETING_ROW]


def _apply_hold_streaks(movement, persona_names, map_width=44):
    """
    Walk each agent's decisions in step order and, once they have held
    HOLD_STREAK_FOR_BREAK steps without trading, move them to the break area
    for the remainder of that streak.

    Only the movement tile and the emoji change -- the decision, its reasoning
    and every metric attached to it are left exactly as logged, so the agent
    cards and decision log still report the HOLD that actually happened.
    """
    bx, by = BREAK_ROOM_TILE
    for offset, agent in enumerate(sorted(persona_names)):
        streak = 0
        for step in sorted(movement.keys()):
            rec = movement[step].get(agent)
            if rec is None:
                continue
            if rec.get("action") in ("buy", "sell"):
                streak = 0
                continue
            streak += 1
            if streak >= HOLD_STREAK_FOR_BREAK:
                # One machine each, rather than three people on one tile.
                rec["movement"] = [min(bx + offset * BREAK_STRIDE, map_width - 1), by]
                rec["pronunciatio"] = "☕"        # hot beverage
                rec["on_break"] = True
                rec["hold_streak"] = streak


def load_run_status(sim_code):
    """
    Whether trading_reverie.py is still actively writing new steps for this
    sim. meta.json's sim_running flag is set True at run start and after
    every step, then False once run() finishes (see trading_reverie.py's
    _write_run_status()). Older sims / a meta.json predating this flag
    default to "not running" -- correct, since those are already-complete
    replay data with nothing left to poll for.
    """
    meta = load_meta(sim_code)
    return {
        "current_step": meta.get("step", 0),
        "is_running": bool(meta.get("sim_running", False)),
    }


# A single-step price move at or above this is a scripted news shock rather
# than ordinary drift (ticks in these runs are around a quarter of a percent).
# The headline TEXT is not recoverable -- trading_log.json stores market_prices
# and never the event -- so the timeline reports the move, not the story.
SHOCK_PCT = 2.0


def build_events(sim_code):
    """
    Flatten a run into one step-ordered timeline of things worth looking at.

    Every entry carries the step it happened on so the page can deep-link into
    /map/<sim>/<step>/. Nothing here is inferred beyond what the log records:
    a "shock" is a measured price move, a "hallucination" is the log's own
    flag, a "break" is a counted HOLD streak.
    """
    log = load_trading_log(sim_code)
    events = []

    # --- market shocks: derived from the prices each entry carries ----------
    prices_by_step = {}
    for entry in log:
        prices_by_step.setdefault(entry["step"], entry.get("market_prices") or {})
    steps_sorted = sorted(prices_by_step)
    for i, step in enumerate(steps_sorted):
        if i == 0:
            continue
        prev = prices_by_step[steps_sorted[i - 1]]
        for sym, px in prices_by_step[step].items():
            was = prev.get(sym)
            if not was:
                continue
            pct = (px - was) / was * 100
            if abs(pct) >= SHOCK_PCT:
                events.append({
                    "step": step, "kind": "shock", "agent": "",
                    "title": "%s %s%.1f%%" % (sym, "+" if pct > 0 else "", pct),
                    "detail": "Scripted news shock. Previous %.2f, now %.2f."
                              % (was, px),
                    "severity": "up" if pct > 0 else "down",
                })

    # --- per-decision events ------------------------------------------------
    streaks = {}
    for entry in log:
        step, agent = entry["step"], entry["agent"]
        decision = entry.get("decision") or {}
        action = (decision.get("action") or "hold").lower()
        requested = entry.get("requested") or {}

        if entry.get("hallucination"):
            # The most important row on the page: what the model asked for,
            # next to what it was allowed to do.
            req_txt = "%s %s x%s" % (requested.get("action", "?"),
                                     requested.get("symbol", "-"),
                                     requested.get("quantity", "?"))
            events.append({
                "step": step, "kind": "hallucination", "agent": agent,
                "title": "%s -- %s" % (agent, entry.get("halluc_kind") or "illegal_request"),
                "detail": "Requested %s. %s. Filter: %s." % (
                    req_txt,
                    entry.get("error_reason") or "not permitted",
                    entry.get("halluc_disposition") or entry.get("filter_status") or "-"),
                "severity": "bad",
            })
        elif action in ("buy", "sell"):
            events.append({
                "step": step, "kind": "trade", "agent": agent,
                "title": "%s %s %s x%s" % (agent, action.upper(),
                                           decision.get("symbol") or "-",
                                           decision.get("quantity")),
                "detail": entry.get("outcome") or "",
                "severity": "up" if action == "buy" else "down",
            })

        if not entry.get("has_reasoning", True):
            events.append({
                "step": step, "kind": "fallback", "agent": agent,
                "title": "%s -- parser fallback" % agent,
                "detail": "Empty reasoning; no detector can score this decision.",
                "severity": "warn",
            })
        if entry.get("state_contradiction"):
            events.append({
                "step": step, "kind": "contradiction", "agent": agent,
                "title": "%s -- state contradiction" % agent,
                "detail": str(entry.get("state_contradiction")),
                "severity": "warn",
            })
        if entry.get("stale_context"):
            events.append({
                "step": step, "kind": "stale", "agent": agent,
                "title": "%s -- stale context" % agent,
                "detail": str(entry.get("stale_context")),
                "severity": "warn",
            })

        # Hold streaks, counted the same way the map counts them.
        if action in ("buy", "sell"):
            streaks[agent] = 0
        else:
            streaks[agent] = streaks.get(agent, 0) + 1
            if streaks[agent] == HOLD_STREAK_FOR_BREAK:
                events.append({
                    "step": step, "kind": "idle", "agent": agent,
                    "title": "%s -- %d steps without trading" % (agent, streaks[agent]),
                    "detail": "Sent to the break area on the map.",
                    "severity": "dim",
                })

    for entry in load_interactions(sim_code):
        agents = entry.get("agents") or []
        events.append({
            "step": entry.get("step"), "kind": "interaction",
            "agent": ", ".join(agents),
            "title": "Peer interaction: %s" % " and ".join(agents),
            "detail": entry.get("summary") or "",
            "severity": "info",
        })

    events.sort(key=lambda e: (e["step"] if e["step"] is not None else 0,
                               e["kind"], e["agent"]))
    return events


def build_decisions(sim_code):
    """Every decision, trimmed to what the inspector panel shows."""
    out = []
    for entry in load_trading_log(sim_code):
        decision = entry.get("decision") or {}
        out.append({
            "step": entry["step"],
            "agent": entry["agent"],
            "action": (decision.get("action") or "hold").lower(),
            "symbol": decision.get("symbol"),
            "quantity": decision.get("quantity"),
            "reasoning": decision.get("reasoning") or "",
            "requested": entry.get("requested") or {},
            "hallucination": bool(entry.get("hallucination")),
            "halluc_kind": entry.get("halluc_kind") or "",
            "disposition": entry.get("halluc_disposition") or "",
            "filter_status": entry.get("filter_status") or "",
            "error_reason": entry.get("error_reason") or "",
            "cash": entry.get("cash"),
            "portfolio_value": entry.get("portfolio_value"),
            "positions": entry.get("positions") or {},
            "stale_context": entry.get("stale_context"),
            "state_contradiction": entry.get("state_contradiction"),
            "drift": entry.get("persona_drift_score"),
            "compression": entry.get("compression") or {},
            "outcome": entry.get("outcome") or "",
        })
    return out


def _action_description(decision):
    action = (decision.get("action") or "hold").lower()
    symbol = decision.get("symbol")
    qty = decision.get("quantity")
    if action == "hold" or not symbol:
        return "HOLD"
    return f"{action.upper()} {qty} {symbol}"


def build_replay(sim_code):
    """
    The single normalized structure both the map view and (later) the
    dashboard read. Shaped like reverie/compress_sim_storage.py's
    master_movement.json ({step: {persona: {...}}}) since that's the
    contract the existing Phaser replay code already understands -- but
    dense (an entry per persona per step we have data for) rather than
    sparse, since trading decisions can change every step, unlike
    Smallville's idle-most-of-the-time chat/description fields that
    sparsity was optimizing for.
    """
    meta = load_meta(sim_code)
    persona_names = meta.get("persona_names", [])
    desk_positions = load_desk_positions(sim_code, persona_names)
    log = load_trading_log(sim_code)
    filter_rows = load_action_filter_log(sim_code)

    # Align action_filter_log.csv rows to trading_log.json entries by
    # per-agent ordinal position: the Nth CSV row for an agent is that
    # agent's Nth decision, i.e. it corresponds to that agent's Nth entry in
    # the log, in step order. The CSV has no step column, so this ordinal
    # join is the reliable link -- wall-clock timestamps drift under retries.
    filter_rows_by_agent = {}
    for row in filter_rows:
        filter_rows_by_agent.setdefault(row["agent"], []).append(row)
    agent_cursor = {}

    movement = {}
    market_prices_by_step = {}

    for entry in log:
        step = entry["step"]
        agent = entry["agent"]
        decision = entry.get("decision", {})

        idx = agent_cursor.get(agent, 0)
        rows = filter_rows_by_agent.get(agent, [])
        filter_row = rows[idx] if idx < len(rows) else None
        agent_cursor[agent] = idx + 1

        if filter_row:
            status = filter_row["status"]
            error_reason = filter_row["error_reason"]
        else:
            status = "hallucination" if entry.get("hallucination") else "unknown"
            error_reason = ""

        action = (decision.get("action") or "hold").lower()
        at_desk = desk_positions.get(agent, [0, 0])
        tile = MARKET_BOARD_TILE if action in ("buy", "sell") else at_desk

        step_bucket = movement.setdefault(step, {})
        step_bucket[agent] = {
            "movement": tile,
            # Normalised verb, kept so later passes (the meeting hold) can ask
            # "did this agent actually trade?" without re-parsing the log entry.
            "action": action,
            # Seated at their own desk, so the frontend can turn them to face
            # the monitor instead of leaving them facing whichever way they
            # happened to walk in from.
            "at_desk": tile == at_desk,
            "pronunciatio": ACTION_ICONS.get(
                (decision.get("action") or "hold").lower(), "✋"
            ),
            "description": _action_description(decision),
            "reasoning": decision.get("reasoning", ""),
            "outcome": entry.get("outcome", ""),
            "status": status,
            "error_reason": error_reason,
            "hallucination": bool(entry.get("hallucination")),
            "stale_context": entry.get("stale_context"),
            "persona_drift_score": entry.get("persona_drift_score"),
            "cash": entry.get("cash"),
            "portfolio_value": entry.get("portfolio_value"),
            "positions": entry.get("positions", {}),
            "chat": None,
        }
        market_prices_by_step[step] = entry.get("market_prices", {})

    # Spread agents who are all at the market board on the same step along the
    # board row, so two people checking prices at once don't render stacked on
    # one tile. Sorted by name so a given step always lays out identically.
    for step_bucket in movement.values():
        at_board = sorted(
            name for name, rec in step_bucket.items()
            if rec["movement"] == MARKET_BOARD_TILE
        )
        if len(at_board) < 2:
            continue
        start_x = MARKET_BOARD_TILE[0] - (len(at_board) - 1) // 2
        lo, hi = MARKET_BOARD_SPAN
        start_x = max(lo, min(start_x, hi - len(at_board) + 1))
        for offset, name in enumerate(at_board):
            step_bucket[name]["movement"] = [start_x + offset, MARKET_BOARD_TILE[1]]

    # Send agents on a long unbroken HOLD streak to the break area. Done before
    # the interaction overlay so that a real meeting still wins: an agent who is
    # both idle and in a conversation should be shown in the conversation.
    _apply_hold_streaks(movement, persona_names)

    # Overlay real peer interactions on top of the decision-driven movement.
    # maybe_interaction() runs before the agents' decisions within a step, so
    # for that step the meeting is what the two of them are actually doing --
    # it takes precedence over the desk/market-board tile their decision would
    # otherwise put them on. Everyone not in the interaction is unaffected.
    interactions = load_interactions(sim_code)
    interactions_by_step = {}
    for entry in interactions:
        step = entry.get("step")
        agents = entry.get("agents") or []
        if step is None or len(agents) != 2:
            continue
        interactions_by_step[step] = entry

        tiles = _meeting_tiles(agents[0], agents[1], desk_positions)

        # Hold the meeting on screen for MEETING_HOLD_STEPS. The first of those
        # is the step the interaction actually happened on, where it takes
        # precedence over the decision outright (maybe_interaction() runs before
        # the agents decide). The rest are presentation only, so a real trade
        # on those steps wins and the agent walks off to the board instead.
        for offset in range(MEETING_HOLD_STEPS):
            held_step = step + offset
            step_bucket = movement.get(held_step)
            if step_bucket is None:
                break                       # run ended mid-meeting

            for agent, tile in zip(agents, tiles):
                rec = step_bucket.get(agent)
                if rec is None:
                    continue
                if offset > 0 and rec.get("action") in ("buy", "sell"):
                    continue                # they traded; don't fake a chat
                rec["movement"] = tile
                rec["chat"] = entry.get("summary", "")
                rec["chat_topic"] = entry.get("topic", "")
                rec["chat_with"] = agents[1] if agent == agents[0] else agents[0]
                rec["pronunciatio"] = "\U0001F4AC"  # speech balloon
                rec["in_meeting"] = True

    steps = sorted(movement.keys())
    run_status = load_run_status(sim_code)

    return {
        "sim_code": sim_code,
        "meta": meta,
        "persona_names": persona_names,
        "desk_positions": desk_positions,
        "steps": steps,
        "movement": movement,
        "market_prices_by_step": market_prices_by_step,
        "persona_snippets": {
            name: load_persona_snippet(sim_code, name) for name in persona_names
        },
        "hallucination_report": load_hallucination_report(sim_code),
        "interactions_by_step": interactions_by_step,
        "is_running": run_status["is_running"],
        "backend_current_step": run_status["current_step"],
    }
