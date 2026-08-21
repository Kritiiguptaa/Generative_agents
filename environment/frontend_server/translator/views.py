"""
Author: Joon Sung Park (joonspk@stanford.edu)
File: views.py
"""
import os
import string
import random
import json
from os import listdir
import os

import datetime
from django.shortcuts import render, redirect, HttpResponseRedirect
from django.http import HttpResponse, JsonResponse
from global_methods import *

from django.contrib.staticfiles.templatetags.staticfiles import static
from .models import *
from . import sim_data


def _get_latest_live_state_file(sim_code, persona_name, step):
  """
  Resolve the most relevant live state snapshot file for a persona.
  """
  live_dir = f"storage/{sim_code}/personas/{persona_name}/live_state"
  if not os.path.exists(live_dir):
    return None, step

  live_steps = []
  for filename in os.listdir(live_dir):
    if filename.endswith(".json"):
      try:
        live_steps += [int(filename.split(".")[0])]
      except:
        pass

  if not live_steps:
    return None, step

  live_steps = sorted(live_steps)
  target_step = step if step in live_steps else live_steps[-1]

  curr_sim_code_file = "temp_storage/curr_sim_code.json"
  if check_if_file_exists(curr_sim_code_file):
    with open(curr_sim_code_file) as json_file:
      curr_sim_code = json.load(json_file).get("sim_code")
    if curr_sim_code == sim_code:
      target_step = live_steps[-1]

  return f"{live_dir}/{target_step}.json", target_step


def _split_associative_memory(associative):
  """
  Split associative memory nodes into event/chat/thought lists.
  """
  a_mem_event = []
  a_mem_chat = []
  a_mem_thought = []

  ordered_nodes = sorted(
    associative.values(),
    key=lambda node: node.get("node_count", 0),
    reverse=True,
  )

  for node_details in ordered_nodes:
    if node_details.get("type") == "event":
      a_mem_event += [node_details]
    elif node_details.get("type") == "chat":
      a_mem_chat += [node_details]
    elif node_details.get("type") == "thought":
      a_mem_thought += [node_details]

  return a_mem_event, a_mem_chat, a_mem_thought

# ---------------------------------------------------------------------------
# landing-page console
# ---------------------------------------------------------------------------

def _console_payload(sim_code, meta, log):
  """
  Per-sim data for the handheld console on the landing page: where each agent
  sits on the office floor, and the one-line summary its ID card shows.

  Kept deliberately thin. The console is a preview, not a second run explorer,
  so it carries counts and the last action -- never a claim about what the
  middleware stopped, which only /run/ is allowed to make.
  """
  names = meta.get("persona_names", [])
  desks = sim_data.load_desk_positions(sim_code, names)

  # One pass over the log instead of one per agent; these runs reach a few
  # thousand records and the index renders every sim.
  last, trades, holds, flagged = {}, {}, {}, {}
  for entry in log:
    agent = entry.get("agent")
    if agent is None:
      continue
    decision = entry.get("decision") or {}
    action = (decision.get("action") or "hold").lower()
    if action in ("buy", "sell"):
      trades[agent] = trades.get(agent, 0) + 1
    else:
      holds[agent] = holds.get(agent, 0) + 1
    if entry.get("hallucination"):
      flagged[agent] = flagged.get(agent, 0) + 1
    last[agent] = sim_data._action_description(decision)

  agents = []
  for name in names:
    snippet = sim_data.load_persona_snippet(sim_code, name)
    tile = desks.get(name) or [0, 0]
    agents += [{
      "name": name,
      # trader_type is the closest thing the persona files carry to a job
      # title; runs forked before it existed fall back to a neutral label.
      # trader_type is stored as a slug ("hedge_fund"); the card shows it as a
      # title, so it is un-slugged here rather than in three template filters.
      "role": (snippet.get("trader_type") or "trading agent").replace("_", " "),
      "risk": snippet.get("risk_tolerance") or "",
      "watchlist": (snippet.get("watchlist") or [])[:4],
      "x": tile[0], "y": tile[1],
      "trades": trades.get(name, 0),
      "holds": holds.get(name, 0),
      "flagged": flagged.get(name, 0),
      "last": last.get(name, ""),
    }]

  return {"agents": agents, "flagged": sum(flagged.values())}


def landing(request):
  """
  Index of every sim in storage/ that actually has trading data, so the map
  view is reachable without hand-typing a sim_code into the URL. A directory
  only lists if reverie/meta.json exists -- the Smallville storage dirs and
  half-created folders sitting alongside the trading runs would otherwise
  render as dead links.
  """
  sims = []
  if os.path.isdir(sim_data.STORAGE_ROOT):
    for name in sorted(os.listdir(sim_data.STORAGE_ROOT)):
      meta = sim_data.load_meta(name)
      if not meta:
        continue
      status = sim_data.load_run_status(name)
      log = sim_data.load_trading_log(name)
      # Trading sims are the ones with trading personas; a Smallville sim has
      # meta.json too, so the presence of a decision log is what separates them.
      sims += [{
        "sim_code": name,
        "personas": meta.get("persona_names", []),
        "steps": status["current_step"],
        "decisions": len(log),
        "is_running": status["is_running"],
        "is_trading": bool(log) or bool(
          set(meta.get("persona_names", [])) & set(sim_data.DESK_TILES)),
        # Everything the console on the landing page draws and reads out. It is
        # built here rather than fetched per-click because the whole index is a
        # handful of small JSON files -- one pass at render time is cheaper than
        # a poll endpoint, and it keeps the console honest: it can only show
        # what the run actually recorded.
        "console": _console_payload(name, meta, log),
      }]
  sims.sort(key=lambda s: (not s["is_trading"], -s["decisions"], s["sim_code"]))
  # The index also picks up Smallville village sims and half-created folders.
  # They are real storage, so they stay listed -- but as a quiet name-only
  # strip, not as full cards competing with the runs worth opening.
  runs  = [s for s in sims if s["is_trading"] and s["decisions"]]
  other = [s for s in sims if s not in runs]

  context = {
    "sims": sims,
    "runs": runs,
    "other": other,
    # The console is driven client-side (canvas floor + ID card), so the same
    # index the table renders is handed to JS as one blob rather than being
    # re-fetched.
    "sims_json": json.dumps(runs),
    "total_decisions": sum(s["decisions"] for s in runs),
    "total_flagged": sum(s["console"]["flagged"] for s in runs),
  }
  template = "landing/landing.html"
  return render(request, template, context)


def demo(request, sim_code, step, play_speed="2"): 
  move_file = f"compressed_storage/{sim_code}/master_movement.json"
  meta_file = f"compressed_storage/{sim_code}/meta.json"
  step = int(step)
  play_speed_opt = {"1": 1, "2": 2, "3": 4,
                    "4": 8, "5": 16, "6": 32}
  if play_speed not in play_speed_opt: play_speed = 2
  else: play_speed = play_speed_opt[play_speed]

  # Loading the basic meta information about the simulation.
  meta = dict() 
  with open (meta_file) as json_file: 
    meta = json.load(json_file)

  sec_per_step = meta["sec_per_step"]
  start_datetime = datetime.datetime.strptime(meta["start_date"] + " 00:00:00", 
                                              '%B %d, %Y %H:%M:%S')
  for i in range(step): 
    start_datetime += datetime.timedelta(seconds=sec_per_step)
  start_datetime = start_datetime.strftime("%Y-%m-%dT%H:%M:%S")

  # Loading the movement file
  raw_all_movement = dict()
  with open(move_file) as json_file: 
    raw_all_movement = json.load(json_file)
 
  # Loading all names of the personas
  persona_names = dict()
  persona_names = []
  persona_names_set = set()
  for p in list(raw_all_movement["0"].keys()): 
    persona_names += [{"original": p, 
                       "underscore": p.replace(" ", "_"), 
                       "initial": p[0] + p.split(" ")[-1][0]}]
    persona_names_set.add(p)

  # <all_movement> is the main movement variable that we are passing to the 
  # frontend. Whereas we use ajax scheme to communicate steps to the frontend
  # during the simulation stage, for this demo, we send all movement 
  # information in one step. 
  all_movement = dict()

  # Preparing the initial step. 
  # <init_prep> sets the locations and descriptions of all agents at the
  # beginning of the demo determined by <step>. 
  init_prep = dict() 
  for int_key in range(step+1): 
    key = str(int_key)
    val = raw_all_movement[key]
    for p in persona_names_set: 
      if p in val: 
        init_prep[p] = val[p]
  persona_init_pos = dict()
  for p in persona_names_set: 
    persona_init_pos[p.replace(" ","_")] = init_prep[p]["movement"]
  all_movement[step] = init_prep

  # Finish loading <all_movement>
  for int_key in range(step+1, len(raw_all_movement.keys())): 
    all_movement[int_key] = raw_all_movement[str(int_key)]

  context = {"sim_code": sim_code,
             "step": step,
             "persona_names": persona_names,
             "persona_init_pos": json.dumps(persona_init_pos), 
             "all_movement": json.dumps(all_movement), 
             "start_datetime": start_datetime,
             "sec_per_step": sec_per_step,
             "play_speed": play_speed,
             "mode": "demo"}
  template = "demo/demo.html"

  return render(request, template, context)


def UIST_Demo(request): 
  return demo(request, "March20_the_ville_n25_UIST_RUN-step-1-141", 2160, play_speed="3")


def home(request):
  f_curr_sim_code = "temp_storage/curr_sim_code.json"
  f_curr_step = "temp_storage/curr_step.json"

  if not check_if_file_exists(f_curr_step): 
    context = {}
    template = "home/error_start_backend.html"
    return render(request, template, context)

  with open(f_curr_sim_code) as json_file:  
    sim_code = json.load(json_file)["sim_code"]
  
  with open(f_curr_step) as json_file:  
    step = json.load(json_file)["step"]

  os.remove(f_curr_step)

  persona_names = []
  persona_names_set = set()
  for i in find_filenames(f"storage/{sim_code}/personas", ""): 
    x = i.split("/")[-1].strip()
    if x[0] != ".": 
      persona_names += [[x, x.replace(" ", "_")]]
      persona_names_set.add(x)

  persona_init_pos = []
  file_count = []
  for i in find_filenames(f"storage/{sim_code}/environment", ".json"):
    x = i.split("/")[-1].strip()
    if x[0] != ".": 
      file_count += [int(x.split(".")[0])]
  curr_json = f'storage/{sim_code}/environment/{str(max(file_count))}.json'
  with open(curr_json) as json_file:  
    persona_init_pos_dict = json.load(json_file)
    for key, val in persona_init_pos_dict.items(): 
      if key in persona_names_set: 
        persona_init_pos += [[key, val["x"], val["y"]]]

  context = {"sim_code": sim_code,
             "step": step, 
             "persona_names": persona_names,
             "persona_init_pos": persona_init_pos,
             "mode": "simulate"}
  template = "home/home.html"
  return render(request, template, context)


def replay(request, sim_code, step): 
  sim_code = sim_code
  step = int(step)

  persona_names = []
  persona_names_set = set()
  for i in find_filenames(f"storage/{sim_code}/personas", ""): 
    x = i.split("/")[-1].strip()
    if x[0] != ".": 
      persona_names += [[x, x.replace(" ", "_")]]
      persona_names_set.add(x)

  persona_init_pos = []
  file_count = []
  for i in find_filenames(f"storage/{sim_code}/environment", ".json"):
    x = i.split("/")[-1].strip()
    if x[0] != ".": 
      file_count += [int(x.split(".")[0])]
  curr_json = f'storage/{sim_code}/environment/{str(max(file_count))}.json'
  with open(curr_json) as json_file:  
    persona_init_pos_dict = json.load(json_file)
    for key, val in persona_init_pos_dict.items(): 
      if key in persona_names_set: 
        persona_init_pos += [[key, val["x"], val["y"]]]

  context = {"sim_code": sim_code,
             "step": step,
             "persona_names": persona_names,
             "persona_init_pos": persona_init_pos, 
             "mode": "replay"}
  template = "home/home.html"
  return render(request, template, context)


def replay_persona_state(request, sim_code, step, persona_name): 
  sim_code = sim_code
  step = int(step)

  persona_name_underscore = persona_name
  persona_name = " ".join(persona_name.split("_"))
  is_live = False
  live_state_file, live_step = _get_latest_live_state_file(sim_code,
                                                           persona_name,
                                                           step)
  if live_state_file and os.path.exists(live_state_file):
    with open(live_state_file) as json_file:
      live_state = json.load(json_file)
    scratch = live_state.get("scratch", dict())
    spatial = live_state.get("spatial", dict())
    associative = live_state.get("associative", dict())
    step = live_step
    is_live = True
  else:
    memory = f"storage/{sim_code}/personas/{persona_name}/bootstrap_memory"
    if not os.path.exists(memory): 
      memory = f"compressed_storage/{sim_code}/personas/{persona_name}/bootstrap_memory"

    with open(memory + "/scratch.json") as json_file:  
      scratch = json.load(json_file)

    with open(memory + "/spatial_memory.json") as json_file:  
      spatial = json.load(json_file)

    with open(memory + "/associative_memory/nodes.json") as json_file:  
      associative = json.load(json_file)

  a_mem_event, a_mem_chat, a_mem_thought = _split_associative_memory(associative)
  
  context = {"sim_code": sim_code,
             "step": step,
             "persona_name": persona_name, 
             "persona_name_underscore": persona_name_underscore, 
             "scratch": scratch,
             "spatial": spatial,
             "a_mem_event": a_mem_event,
             "a_mem_chat": a_mem_chat,
             "a_mem_thought": a_mem_thought,
             "is_live": is_live}
  template = "persona_state/persona_state.html"
  return render(request, template, context)


def trading_map(request, sim_code, step=0):
  """
  Trading-floor map view. No bulk preload any more -- the template fetches
  step data live from trading_poll() on an interval instead of this view
  baking the whole run into the page. Works the same whether trading_reverie.py
  is still actively appending steps to this sim_code or already finished --
  a finished run's first poll just returns everything at once.
  """
  step = int(step)
  replay = sim_data.build_replay(sim_code)

  persona_names = []
  for p in replay["persona_names"]:
    underscore = p.replace(" ", "_")
    initial = p[0] + p.split(" ")[-1][0]
    persona_names += [{"original": p, "underscore": underscore, "initial": initial}]

  persona_init_pos = {
    p.replace(" ", "_"): replay["desk_positions"].get(p, [0, 0])
    for p in replay["persona_names"]
  }

  meta = replay["meta"]
  sec_per_step = meta.get("sec_per_step", 60)
  start_date = meta.get("start_date")
  if start_date:
    start_datetime = datetime.datetime.strptime(
      start_date + " 09:30:00", '%B %d, %Y %H:%M:%S')
  else:
    start_datetime = datetime.datetime.now()
  start_datetime = start_datetime.strftime("%Y-%m-%dT%H:%M:%S")

  persona_snippets = {
    p.replace(" ", "_"): snippet
    for p, snippet in replay["persona_snippets"].items()
  }

  context = {
    "sim_code": sim_code,
    "step": step,
    "persona_names": persona_names,
    "persona_init_pos": json.dumps(persona_init_pos),
    "persona_snippets": json.dumps(persona_snippets),
    "start_datetime": start_datetime,
    "sec_per_step": sec_per_step,
    "is_running": replay["is_running"],
    "has_data": bool(replay["steps"]),
    # Map geometry comes from sim_data so the tile constants live in exactly
    # one place -- the frontend never hardcodes its own copy.
    "market_board_tile": json.dumps(sim_data.MARKET_BOARD_TILE),
    "break_room_tile": json.dumps(sim_data.BREAK_ROOM_TILE),
  }
  template = "trading_map/trading_map.html"
  return render(request, template, context)


NON_TRADE_ACTIONS = frozenset({"hold", "analyze"})


def _is_trade_attempt(d):
  """
  Did the model ask to trade on this decision?

  Mirrors action_filtering.is_trade_request: an INVERTED test, because an
  allowlist of buy/sell can only ever undercount. Anything that is not
  explicitly a no-op counts, including verbs nobody anticipated.

  The third clause is the one that matters. A rejected request is logged with
  a substituted HOLD as its final action, and older logs carry no `requested`
  field at all -- so testing only the final verb drops every rejection from
  the denominator while keeping it in the numerator. That is exactly the
  defect behind the 145.5% rate in CHANGES section 19.1, and leaving it here
  would reintroduce it on this page.
  """
  requested = (d.get("requested") or {}).get("action")
  if isinstance(requested, str) and requested.strip().lower() not in NON_TRADE_ACTIONS:
    return True
  if d.get("action") in ("buy", "sell"):
    return True
  return bool(d.get("hallucination"))


def has_report_counts(report):
  """True when the report actually carries the disposition counts."""
  return report.get("total_caught_pre_execution") is not None \
      and report.get("total_executed_malformed") is not None


def run_explorer(request, sim_code):
  """
  <BACKEND to FRONTEND>
  The run's evidence page: what condition this was, what the middleware caught
  versus executed, every notable event in step order (each deep-linking into
  the map), and every decision's requested-vs-final record.

  Distinct from trading_map(): that shows a run happening, this shows what a
  run *found*. Both read the same storage through sim_data.
  """
  meta = sim_data.load_meta(sim_code)
  report = sim_data.load_hallucination_report(sim_code) or {}
  events = sim_data.build_events(sim_code)
  decisions = sim_data.build_decisions(sim_code)
  status = sim_data.load_run_status(sim_code)
  persona_names = meta.get("persona_names", [])

  steps = sorted({d["step"] for d in decisions})

  # Per-agent scorecards, pairing the designed failure mode against what the
  # run actually did. The design intent is fixed (it comes from the persona
  # construction, see PROJECT_OVERVIEW section 3); everything else is counted.
  designed = {
    "Alex Chen":   "over-caution -- waits for confirmation indefinitely",
    "Marcus Webb": "over-sizing beyond available cash",
    "Sara Kim":    "acting on stale headlines",
  }
  agents = []
  for name in persona_names:
    stats = (report.get("per_agent") or {}).get(name, {})
    own = [d for d in decisions if d["agent"] == name]
    holds = sum(1 for d in own if d["action"] not in ("buy", "sell"))
    longest, run_len = 0, 0
    for d in own:
      run_len = 0 if d["action"] in ("buy", "sell") else run_len + 1
      longest = max(longest, run_len)
    agents.append({
      "name": name,
      "snippet": sim_data.load_persona_snippet(sim_code, name),
      "designed": designed.get(name, ""),
      "stats": stats,
      "decisions": len(own),
      "trades": sum(1 for d in own if d["action"] in ("buy", "sell")),
      "holds": holds,
      "hallucinations": sum(1 for d in own if d["hallucination"]),
      "longest_hold_streak": longest,
      "kinds": stats.get("hallucination_kinds") or {},
      "error_reasons": stats.get("error_reasons") or {},
    })

  # Headline counts. hallucination_report.json is only written when a run
  # finishes, so for a run still in flight (or one killed early) it is absent
  # and every report.* lookup renders 0. Showing "0 caught" next to agent cards
  # counting 28 hallucinations is worse than showing nothing -- it is a
  # confident wrong number, which is the exact failure this project keeps
  # rediscovering. So derive the counts from the log when the report is absent,
  # and mark them as derived.
  headline = {
    "caught": report.get("total_caught_pre_execution"),
    "executed": report.get("total_executed_malformed"),
    "attempts": report.get("total_trade_attempts"),
    "rate": report.get("overall_hallucination_rate_pct"),
    "derived": False,
    "disposition_unknown": False,
  }
  if not has_report_counts(report):
    caught = sum(1 for d in decisions
                 if d["hallucination"] and d["disposition"] != "executed_malformed")
    executed = sum(1 for d in decisions if d["disposition"] == "executed_malformed")
    attempts = sum(1 for d in decisions if _is_trade_attempt(d))
    # Logs written before the disposition field existed carry "" on every
    # record. Deriving from that yields "all caught, none executed", which is
    # not a cautious reading -- it is a fabricated one, and it is the single
    # claim this page exists to make. Say unknown instead.
    unknown = any(d["hallucination"] for d in decisions) and \
        not any(d["disposition"] for d in decisions)
    headline = {
      "caught": caught,
      "executed": executed,
      "attempts": attempts,
      # 0/0 prints N/A, never 0.0% -- a vacuous result must not look measured.
      "rate": round((caught + executed) / attempts * 100, 1) if attempts else None,
      "derived": True,
      "disposition_unknown": unknown,
    }

  # Whatever compression recorded, averaged over the decisions that carry it.
  comp_in = [d["compression"].get("nodes_in") for d in decisions
             if isinstance(d.get("compression"), dict) and d["compression"].get("nodes_in")]
  comp_out = [d["compression"].get("nodes_out") for d in decisions
              if isinstance(d.get("compression"), dict) and d["compression"].get("nodes_out")]
  compression = None
  if comp_in and comp_out:
    compression = {
      "nodes_in": round(sum(comp_in) / len(comp_in), 1),
      "nodes_out": round(sum(comp_out) / len(comp_out), 1),
      "samples": len(comp_in),
    }

  context = {
    "sim_code": sim_code,
    "meta": meta,
    "report": report,
    "has_report": bool(report),
    "headline": headline,
    "events_json": json.dumps(events),
    "decisions_json": json.dumps(decisions),
    "event_count": len(events),
    "agents": agents,
    "persona_names": persona_names,
    "compression": compression,
    "is_running": status["is_running"],
    "first_step": steps[0] if steps else 0,
    "last_step": steps[-1] if steps else 0,
    "middleware_enabled": report.get("middleware_enabled"),
    # Recorded nowhere in the report today -- see CHANGES section 22 item 2.
    # Shown as unknown rather than guessed, because a wrong model label is
    # worse than an absent one.
    "model_name": report.get("model") or meta.get("model") or "",
    "seed": report.get("seed") if report.get("seed") is not None else meta.get("seed"),
  }
  return render(request, "trading_map/run_explorer.html", context)


def trading_poll(request, sim_code):
  """
  <BACKEND to FRONTEND, live mode>
  Polling counterpart to trading_map()'s one-time render. Reads through the
  same sim_data.build_replay() ingestion layer -- no duplicated parsing --
  and hands back everything accumulated so far plus whether trading_reverie.py
  is still actively writing new steps (meta.json's sim_running flag). The
  frontend calls this on an interval so a run in progress becomes visible
  step-by-step as it actually happens, instead of only after it finishes.
  """
  replay = sim_data.build_replay(sim_code)
  return JsonResponse({
    "current_step": max(replay["steps"]) if replay["steps"] else -1,
    "is_running": replay["is_running"],
    "movement": replay["movement"],
    "market_prices_by_step": replay["market_prices_by_step"],
    "interactions_by_step": replay["interactions_by_step"],
  })


def trading_persona_state(request, sim_code, step, persona_name):
  """
  Per-agent detail panel for the trading map. Like replay_persona_state, but
  sourced from sim_data.py so it carries the requested step's decision,
  reasoning, and retry/fallback status alongside the static persona snippet
  from scratch.json -- not just the static snapshot.
  """
  step = int(step)
  persona_name_underscore = persona_name
  persona_name = " ".join(persona_name.split("_"))

  replay = sim_data.build_replay(sim_code)
  snippet = replay["persona_snippets"].get(persona_name, {})
  step_record = replay["movement"].get(step, {}).get(persona_name)

  # Fall back to the nearest earlier step that has a record for this persona,
  # since a given step may not have one for every agent.
  if step_record is None:
    for s in sorted(replay["movement"].keys(), reverse=True):
      if s <= step and persona_name in replay["movement"][s]:
        step_record = replay["movement"][s][persona_name]
        break

  context = {
    "sim_code": sim_code,
    "step": step,
    "persona_name": persona_name,
    "persona_name_underscore": persona_name_underscore,
    "snippet": snippet,
    "record": step_record,
    "has_record": step_record is not None,
  }
  template = "persona_state/trading_persona_state.html"
  return render(request, template, context)


def path_tester(request):
  context = {}
  template = "path_tester/path_tester.html"
  return render(request, template, context)


def process_environment(request): 
  """
  <FRONTEND to BACKEND> 
  This sends the frontend visual world information to the backend server. 
  It does this by writing the current environment representation to 
  "storage/environment.json" file. 

  ARGS:
    request: Django request
  RETURNS: 
    HttpResponse: string confirmation message. 
  """
  # f_curr_sim_code = "temp_storage/curr_sim_code.json"
  # with open(f_curr_sim_code) as json_file:  
  #   sim_code = json.load(json_file)["sim_code"]

  data = json.loads(request.body)
  step = data["step"]
  sim_code = data["sim_code"]
  environment = data["environment"]

  with open(f"storage/{sim_code}/environment/{step}.json", "w") as outfile:
    outfile.write(json.dumps(environment, indent=2))

  return HttpResponse("received")


def update_environment(request): 
  """
  <BACKEND to FRONTEND> 
  This sends the backend computation of the persona behavior to the frontend
  visual server. 
  It does this by reading the new movement information from 
  "storage/movement.json" file.

  ARGS:
    request: Django request
  RETURNS: 
    HttpResponse
  """
  # f_curr_sim_code = "temp_storage/curr_sim_code.json"
  # with open(f_curr_sim_code) as json_file:  
  #   sim_code = json.load(json_file)["sim_code"]

  data = json.loads(request.body)
  step = data["step"]
  sim_code = data["sim_code"]

  response_data = {"<step>": -1}
  if (check_if_file_exists(f"storage/{sim_code}/movement/{step}.json")):
    with open(f"storage/{sim_code}/movement/{step}.json") as json_file: 
      response_data = json.load(json_file)
      response_data["<step>"] = step

  return JsonResponse(response_data)


def path_tester_update(request): 
  """
  Processing the path and saving it to path_tester_env.json temp storage for 
  conducting the path tester. 

  ARGS:
    request: Django request
  RETURNS: 
    HttpResponse: string confirmation message. 
  """
  data = json.loads(request.body)
  camera = data["camera"]

  with open(f"temp_storage/path_tester_env.json", "w") as outfile:
    outfile.write(json.dumps(camera, indent=2))

  return HttpResponse("received")









