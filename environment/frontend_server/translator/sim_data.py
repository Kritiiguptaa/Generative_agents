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
# Office map geometry (static_dirs/assets/office/visuals/office.json, 44x32
# tiles @16px). Every tile below was read out of the map's own authoring
# matrices, not eyeballed -- see office-main/matrix/: game_object_maze.csv
# labels desks as 32140 and the presentation screen as 32147,
# arena_maze.csv marks the "open office bullpen" (32131) at x:16-42 y:2-15,
# and collision_maze.csv is the walkability ground truth.
#
# The bullpen has three identical desk clusters (x:17-19, x:22-24, x:27-29,
# desks on rows y=6/7 with computers above on y=5) -- one per trading agent.
# ---------------------------------------------------------------------------

# Where agents sit. Row y=8 is the aisle immediately below each desk cluster;
# x=18/23/28 are blocked there (chairs), so each agent stands on the walkable
# tile at the left edge of their own cluster.
DESK_TILES = {
    "Alex Chen": [17, 8],
    "Marcus Webb": [22, 8],
    "Sara Kim": [27, 8],
}

# The two presentation screens (game object 32147) span x:20-21 and x:24-25 on
# rows y=2-3. Row y=4 beneath them is open floor, and x=22 sits centred
# between the two screens -- that's the "check the market board" spot agents
# walk to on a real BUY/SELL.
MARKET_BOARD_TILE = [22, 4]

# Walkable extent of the market-board row inside the bullpen (collision_maze
# shows y=4 clear from x=16 to x=42). Several agents commonly BUY on the same
# step, so they're spread along this row instead of all stacking on one tile.
MARKET_BOARD_SPAN = (16, 42)

# Fully-walkable aisle running the width of the bullpen (collision_maze.csv
# shows y=9 clear for 38 tiles from x=5). When trading_interactions.py fires a
# real peer interaction the two agents involved walk out here to meet.
MEETING_ROW = 9


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
        if name in latest:
            positions[name] = [latest[name]["x"], latest[name]["y"]]
        elif name in DESK_TILES:
            # Sims forked before the office map landed have no entry (or a
            # stale one) for this agent -- fall back to the documented office
            # desk rather than dumping them at [0,0] inside a wall.
            positions[name] = list(DESK_TILES[name])
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
    Where two interacting agents stand to talk. Derived from their real desk
    positions -- the midpoint between the two desks, pushed out to the open
    MEETING_ROW aisle -- then offset by one tile so they stand side by side
    facing each other instead of overlapping on one tile.
    """
    ax = desk_positions.get(agent_a, [0, 0])[0]
    bx = desk_positions.get(agent_b, [0, 0])[0]
    mid_x = (ax + bx) // 2
    # Keep the pair inside the map even if the midpoint lands on the last column.
    mid_x = max(0, min(mid_x, map_width - 2))
    return [mid_x, MEETING_ROW], [mid_x + 1, MEETING_ROW]


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
        step_bucket = movement.setdefault(step, {})
        for agent, tile in zip(agents, tiles):
            rec = step_bucket.get(agent)
            if rec is None:
                continue
            rec["movement"] = tile
            rec["chat"] = entry.get("summary", "")
            rec["chat_topic"] = entry.get("topic", "")
            rec["chat_with"] = agents[1] if agent == agents[0] else agents[0]
            rec["pronunciatio"] = "\U0001F4AC"  # speech balloon

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
