"""
File: trading_daily_plan.py
Description: Daily plan generation and updates for trading agents.
  Uses an LLM when available, with a deterministic fallback.
"""

import json

from persona.prompt_template.gpt_structure import ollama_request


PLAN_MINUTES = 390  # 9:30am-4:00pm trading day


def ensure_daily_plan(persona, market, force: bool = False) -> bool:
    """Ensure the agent has a plan for the current market day."""
    today = market.current_time.strftime("%Y-%m-%d")
    last = getattr(persona.scratch, "last_plan_date", None)
    has_plan = bool(getattr(persona.scratch, "daily_req", []))

    if force or (last != today) or not has_plan:
        plan = generate_daily_plan(persona, market)
        _apply_plan(persona, plan, today)
        return True
    return False


def generate_daily_plan(persona, market) -> dict:
    """Create a daily trading plan using the agent profile and context."""
    interaction_notes = getattr(persona.scratch, "interaction_notes", [])
    recent_notes = "\n".join(interaction_notes[-3:]) or "None"

    prompt = f"""You are a trading agent planner. Create a daily plan.
Agent: {persona.scratch.name}
Role: {persona.scratch.trader_type}
Traits: {persona.scratch.innate}
Background: {persona.scratch.learned}
Current situation: {persona.scratch.currently}
Risk tolerance: {persona.scratch.risk_tolerance}
Market time: {market.current_time.strftime('%Y-%m-%d %H:%M')}
Recent interactions:\n{recent_notes}

Return JSON only with keys:
- daily_req: list of 5-7 concise plan items
- schedule_blocks: list of {{"task": str, "minutes": int}} totaling {PLAN_MINUTES}

Rules:
- Focus on pre-market review, trade window, mid-day review, risk checks, and close.
- Keep tasks trading-focused and time-bound.
- Use plain ASCII.
"""

    raw = ollama_request(prompt, max_tokens=220, stop=["\n\n"], timeout=180)
    try:
        cleaned = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        data = json.loads(cleaned)
        daily_req = [str(x).strip() for x in data.get("daily_req", []) if str(x).strip()]
        blocks = data.get("schedule_blocks", [])
        schedule = []
        for block in blocks:
            task = str(block.get("task", "")).strip()
            minutes = int(block.get("minutes", 0))
            if task and minutes > 0:
                schedule.append({"task": task, "minutes": minutes})
        schedule = _normalize_schedule(schedule)
        if daily_req and schedule:
            return {"daily_req": daily_req, "schedule": schedule}
    except Exception:
        pass

    # Fallback plan
    daily_req = [
        "Review overnight news and pre-market moves",
        "Scan watchlist for setups at the open",
        "Execute 1-2 high-conviction trades",
        "Re-assess positions and risk mid-day",
        "Review portfolio performance and update notes",
        "Plan adjustments for the next session",
    ]
    schedule = [
        {"task": "pre-market review and watchlist scan", "minutes": 30},
        {"task": "open-session trade window", "minutes": 90},
        {"task": "monitor positions and news flow", "minutes": 90},
        {"task": "mid-day risk review and trims", "minutes": 60},
        {"task": "afternoon trade window", "minutes": 60},
        {"task": "close review and journaling", "minutes": 60},
    ]
    return {"daily_req": daily_req, "schedule": _normalize_schedule(schedule)}


def apply_interaction_to_plan(persona, summary: str) -> None:
    """Adjust the daily plan based on a new interaction summary."""
    if not summary:
        return

    note = f"Follow up on: {summary}"
    daily_req = list(getattr(persona.scratch, "daily_req", []))
    if note not in daily_req:
        daily_req.append(note)

    schedule = _from_scratch_schedule(persona)
    follow_block = {"task": "follow up on teammate insights", "minutes": 30}
    schedule = schedule[:] if schedule else []
    if schedule:
        schedule.insert(1, follow_block)
    else:
        schedule = [follow_block]
    schedule = _normalize_schedule(schedule)

    persona.scratch.daily_req = daily_req
    persona.scratch.f_daily_schedule = _schedule_to_list(schedule)
    persona.scratch.f_daily_schedule_hourly_org = list(persona.scratch.f_daily_schedule)

    notes = list(getattr(persona.scratch, "interaction_notes", []))
    notes.append(summary)
    persona.scratch.interaction_notes = notes[-10:]


def current_focus_str(persona, market) -> str:
    """Return the current plan block based on market time."""
    schedule = _from_scratch_schedule(persona)
    if not schedule:
        return "No plan set."

    day_start = market.current_time.replace(hour=9, minute=30, second=0, microsecond=0)
    if market.current_time < day_start:
        return schedule[0]["task"]

    minutes = int((market.current_time - day_start).total_seconds() // 60)
    elapsed = 0
    for block in schedule:
        elapsed += block["minutes"]
        if minutes < elapsed:
            return block["task"]
    return schedule[-1]["task"]


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _apply_plan(persona, plan: dict, date_str: str) -> None:
    persona.scratch.daily_req = plan.get("daily_req", [])
    schedule = plan.get("schedule", [])
    persona.scratch.f_daily_schedule = _schedule_to_list(schedule)
    persona.scratch.f_daily_schedule_hourly_org = list(persona.scratch.f_daily_schedule)
    persona.scratch.last_plan_date = date_str


def _normalize_schedule(schedule: list) -> list:
    """Ensure schedule totals PLAN_MINUTES by trimming or padding."""
    total = sum(int(b.get("minutes", 0)) for b in schedule)
    if total <= 0:
        return []

    if total > PLAN_MINUTES:
        trimmed = []
        running = 0
        for block in schedule:
            if running + block["minutes"] >= PLAN_MINUTES:
                remaining = PLAN_MINUTES - running
                if remaining > 0:
                    trimmed.append({"task": block["task"], "minutes": remaining})
                break
            trimmed.append(block)
            running += block["minutes"]
        return trimmed

    if total < PLAN_MINUTES:
        schedule = list(schedule)
        schedule.append({
            "task": "monitor market and manage risk",
            "minutes": PLAN_MINUTES - total,
        })
    return schedule


def _from_scratch_schedule(persona) -> list:
    raw = getattr(persona.scratch, "f_daily_schedule", [])
    schedule = []
    for item in raw:
        if isinstance(item, list) and len(item) == 2:
            task, minutes = item
            schedule.append({"task": str(task), "minutes": int(minutes)})
    return schedule


def _schedule_to_list(schedule: list) -> list:
    return [[b["task"], int(b["minutes"])] for b in schedule]
