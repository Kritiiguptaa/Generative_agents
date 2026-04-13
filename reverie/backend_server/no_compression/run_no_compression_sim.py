#!/usr/bin/env python3
"""
Run a Generative Agents simulation without any memory-compression stage.

This runner is designed for small simulations (default: 2 agents) and writes
hallucination outputs directly from the live simulation state.
"""

import argparse
import json
import os
import sys
import time
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

THIS_FILE = Path(__file__).resolve()
BACKEND_SERVER_DIR = THIS_FILE.parents[1]

if str(BACKEND_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_SERVER_DIR))

# Reverie paths in utils.py are relative to backend_server/, so lock cwd.
os.chdir(BACKEND_SERVER_DIR)

from reverie import ReverieServer  # noqa: E402
from utils import fs_storage  # noqa: E402


def _read_json(file_path: Path) -> Dict[str, Any]:
    with file_path.open("r", encoding="utf-8") as infile:
        return json.load(infile)


def _write_json(file_path: Path, payload: Dict[str, Any]) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as outfile:
        json.dump(payload, outfile, indent=2)


def _sim_storage_path(sim_code: str) -> Path:
    # fs_storage is relative to backend_server/ (e.g. ../../environment/...)
    return (BACKEND_SERVER_DIR / fs_storage / sim_code).resolve()


def _format_duration(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)

    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _ollama_is_reachable(timeout_seconds: float = 2.0) -> bool:
    try:
        import requests

        response = requests.get("http://localhost:11434/api/tags", timeout=timeout_seconds)
        return response.ok
    except Exception:
        return False


def _canonical_pair(a: str, b: str) -> Tuple[str, str]:
    return (a, b) if a < b else (b, a)


def _expected_pairs(agent_names: Iterable[str]) -> Set[Tuple[str, str]]:
    sorted_names = sorted(agent_names)
    return {_canonical_pair(a, b) for a, b in combinations(sorted_names, 2)}


def _extract_interaction_pairs(
    movement: Dict[str, Any],
    agent_names: Iterable[str],
) -> Set[Tuple[str, str]]:
    valid_agents = set(agent_names)
    discovered: Set[Tuple[str, str]] = set()
    persona_payload = movement.get("persona", {})

    if not isinstance(persona_payload, dict):
        return discovered

    for owner_name, owner_payload in persona_payload.items():
        if not isinstance(owner_payload, dict):
            continue

        chat = owner_payload.get("chat")
        if not isinstance(chat, list):
            continue

        speakers: Set[str] = set()
        for turn in chat:
            if isinstance(turn, (list, tuple)) and len(turn) >= 1 and isinstance(turn[0], str):
                speaker = turn[0].strip()
                if speaker in valid_agents:
                    speakers.add(speaker)

        if len(speakers) >= 2:
            for a, b in combinations(sorted(speakers), 2):
                discovered.add(_canonical_pair(a, b))
        elif len(speakers) == 1 and owner_name in valid_agents:
            only_speaker = next(iter(speakers))
            if only_speaker != owner_name:
                discovered.add(_canonical_pair(owner_name, only_speaker))

    return discovered


def _force_group_meeting(
    env_file: Path,
    agent_names: List[str],
    meeting_point: Optional[Tuple[int, int]] = None,
) -> Tuple[int, int]:
    env_payload = _read_json(env_file)

    if meeting_point is not None:
        meeting_x, meeting_y = meeting_point
    else:
        anchor_name = agent_names[0]
        if anchor_name not in env_payload:
            raise KeyError(f"Anchor agent '{anchor_name}' missing from {env_file}")
        meeting_x = int(env_payload[anchor_name]["x"])
        meeting_y = int(env_payload[anchor_name]["y"])

    for agent_name in agent_names:
        if agent_name not in env_payload:
            continue
        env_payload[agent_name]["x"] = int(meeting_x)
        env_payload[agent_name]["y"] = int(meeting_y)

    _write_json(env_file, env_payload)
    return int(meeting_x), int(meeting_y)


def _build_next_environment(
    current_env: Dict[str, Any],
    movement: Dict[str, Any],
) -> Dict[str, Any]:
    next_env: Dict[str, Any] = {}
    persona_moves = movement.get("persona", {})

    for persona_name, state in current_env.items():
        persona_move = persona_moves.get(persona_name, {})
        movement_xy = persona_move.get("movement", [])

        if isinstance(movement_xy, list) and len(movement_xy) >= 2:
            next_x, next_y = int(movement_xy[0]), int(movement_xy[1])
        else:
            # Fall back to previous tile if movement data is absent.
            next_x, next_y = int(state["x"]), int(state["y"])

        next_env[persona_name] = {
            "maze": state.get("maze", "the_ville"),
            "x": next_x,
            "y": next_y,
        }

    return next_env


def _prepare_next_environment(sim_storage: Path, processed_step: int) -> Path:
    current_env_file = sim_storage / "environment" / f"{processed_step}.json"
    movement_file = sim_storage / "movement" / f"{processed_step}.json"

    if not current_env_file.exists():
        raise FileNotFoundError(f"Missing environment file: {current_env_file}")
    if not movement_file.exists():
        raise FileNotFoundError(f"Missing movement file: {movement_file}")

    current_env = _read_json(current_env_file)
    movement = _read_json(movement_file)
    next_env = _build_next_environment(current_env, movement)

    next_env_file = sim_storage / "environment" / f"{processed_step + 1}.json"
    _write_json(next_env_file, next_env)
    return next_env_file


def _build_summary(
    origin: str,
    target: str,
    steps: int,
    hallucination_history: Dict[str, Any],
    agent_names: Iterable[str],
    seen_interaction_pairs: Optional[Set[Tuple[str, str]]] = None,
    required_interaction_pairs: Optional[Set[Tuple[str, str]]] = None,
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "origin": origin,
        "target": target,
        "steps_requested": steps,
        "agents": {},
    }

    for persona_name in agent_names:
        entries = hallucination_history.get(persona_name, [])
        if not entries:
            summary["agents"][persona_name] = {
                "records": 0,
                "avg_overall": 0.0,
                "latest_overall": 0.0,
                "latest_summary": "No hallucination records.",
            }
            continue

        overall_values = [float(item["scores"]["overall"]) for item in entries]
        latest = entries[-1]

        summary["agents"][persona_name] = {
            "records": len(entries),
            "avg_overall": round(sum(overall_values) / len(overall_values), 4),
            "latest_overall": round(float(latest["scores"]["overall"]), 4),
            "latest_summary": latest.get("summary", ""),
        }

    if seen_interaction_pairs is not None:
        seen_pairs_sorted = sorted([list(pair) for pair in seen_interaction_pairs])
        interaction_block: Dict[str, Any] = {
            "pairs_seen": seen_pairs_sorted,
            "pairs_seen_count": len(seen_pairs_sorted),
        }

        if required_interaction_pairs is not None:
            missing_pairs = sorted([
                list(pair)
                for pair in (required_interaction_pairs - seen_interaction_pairs)
            ])
            interaction_block["pairs_required"] = sorted([
                list(pair) for pair in required_interaction_pairs
            ])
            interaction_block["pairs_required_count"] = len(required_interaction_pairs)
            interaction_block["pairs_missing"] = missing_pairs
            interaction_block["coverage"] = (
                round(
                    len(seen_interaction_pairs) / max(1, len(required_interaction_pairs)),
                    4,
                )
            )

        summary["interaction"] = interaction_block

    return summary


def run_no_compression_simulation(
    origin: str,
    target: str,
    steps: int,
    expected_agents: int,
    hallucination_interval: int,
    live_state_interval: int,
    server_sleep: float,
    progress_every: int,
    disable_hallucination: bool,
    require_pairwise_interactions: bool,
    interaction_report_every: int,
    force_interaction: bool,
    force_meeting_every: int,
    meeting_x: Optional[int],
    meeting_y: Optional[int],
    auto_extend_for_interactions: bool,
    max_extra_steps: int,
    best_effort_interactions: bool,
) -> None:
    print(f"[start] origin={origin} -> target={target}, steps={steps}")
    rs = ReverieServer(origin, target)

    loaded_agents = sorted(rs.personas.keys())
    print(f"[agents] loaded ({len(loaded_agents)}): {', '.join(loaded_agents)}")

    if expected_agents > 0 and len(loaded_agents) != expected_agents:
        raise ValueError(
            f"Expected {expected_agents} agents, but loaded {len(loaded_agents)} "
            f"from '{origin}'."
        )

    rs.server_sleep = max(0.0, float(server_sleep))
    rs.live_state_save_interval = max(1, int(live_state_interval))
    rs.hallucination_enabled = not disable_hallucination
    rs.hallucination_interval = max(1, int(hallucination_interval))
    rs.hallucination_save_interval = max(10, rs.hallucination_interval)

    if force_interaction:
        os.environ["FORCE_AGENT_INTERACTION"] = "1"
    else:
        os.environ.pop("FORCE_AGENT_INTERACTION", None)

    if rs.hallucination_enabled:
        hallu_mode = f"every {rs.hallucination_interval} step(s)"
    else:
        hallu_mode = "disabled"
    print(
        f"[config] server_sleep={rs.server_sleep:.3f}s, "
        f"live_state_every={rs.live_state_save_interval}, "
        f"hallucination={hallu_mode}"
    )

    expected_pairwise = _expected_pairs(loaded_agents)
    seen_pairwise: Set[Tuple[str, str]] = set()
    if require_pairwise_interactions:
        print(f"[interaction] requiring pairwise chats across {len(expected_pairwise)} pairs")
        if best_effort_interactions:
            print("[interaction] best-effort mode enabled (missing pairs will not fail run)")

    if meeting_x is not None and meeting_y is not None:
        fixed_meeting_point: Optional[Tuple[int, int]] = (int(meeting_x), int(meeting_y))
    else:
        fixed_meeting_point = None

    sim_storage = _sim_storage_path(target)
    progress_every = max(1, int(progress_every))
    interaction_report_every = max(1, int(interaction_report_every))
    started_at = time.perf_counter()
    max_total_steps = steps + (max_extra_steps if auto_extend_for_interactions else 0)
    steps_done = 0

    while steps_done < max_total_steps:
        current_step = rs.step
        current_env_file = sim_storage / "environment" / f"{current_step}.json"
        if not current_env_file.exists():
            raise FileNotFoundError(
                f"Cannot run step {current_step}: missing environment file "
                f"{current_env_file}"
            )

        if force_meeting_every > 0 and steps_done % force_meeting_every == 0:
            meeting_point = _force_group_meeting(
                current_env_file,
                loaded_agents,
                fixed_meeting_point,
            )
            print(
                f"[interaction] forced meeting at ({meeting_point[0]}, {meeting_point[1]}) "
                f"on sim step {current_step}"
            )

        step_started = time.perf_counter()
        rs.start_server(1)

        movement_file = sim_storage / "movement" / f"{current_step}.json"
        if movement_file.exists():
            movement_payload = _read_json(movement_file)
            newly_seen_pairs = _extract_interaction_pairs(movement_payload, loaded_agents)
            if newly_seen_pairs:
                prior_count = len(seen_pairwise)
                seen_pairwise.update(newly_seen_pairs & expected_pairwise)
                if len(seen_pairwise) > prior_count:
                    new_pair_count = len(seen_pairwise) - prior_count
                    print(
                        f"[interaction] discovered {new_pair_count} new pair(s), "
                        f"coverage={len(seen_pairwise)}/{len(expected_pairwise)}"
                    )

        next_env_file = _prepare_next_environment(sim_storage, current_step)

        steps_done += 1
        step_elapsed = time.perf_counter() - step_started
        if steps_done <= steps:
            planned_total = steps
        else:
            planned_total = max_total_steps

        if steps_done == 1 or steps_done % progress_every == 0 or steps_done == planned_total:
            elapsed = time.perf_counter() - started_at
            avg_per_step = elapsed / steps_done
            eta_seconds = avg_per_step * max(0, planned_total - steps_done)
            print(
                f"[progress] {steps_done}/{planned_total} "
                f"(step={current_step}, step_time={step_elapsed:.2f}s, "
                f"avg={avg_per_step:.2f}s, eta={_format_duration(eta_seconds)}) "
                f"-> wrote {next_env_file.name}"
            )

        if require_pairwise_interactions and (
            steps_done % interaction_report_every == 0 or steps_done == steps
        ):
            print(
                f"[interaction] pair coverage: {len(seen_pairwise)}/{len(expected_pairwise)}"
            )

        if steps_done < steps:
            continue

        if not require_pairwise_interactions:
            break

        missing_pairs = expected_pairwise - seen_pairwise
        if not missing_pairs:
            break

        if not auto_extend_for_interactions:
            break

        if steps_done == steps:
            print(
                f"[interaction] base run ended with {len(missing_pairs)} missing pair(s); "
                f"extending up to {max_extra_steps} extra steps"
            )

    # Persist all persona memory + final hallucination report.
    rs.save()
    rs._save_hallucination_report()

    hallu_file = sim_storage / "reverie" / "hallucination_analysis.json"
    hallucination_history = _read_json(hallu_file) if hallu_file.exists() else rs.hallucination_history
    for persona_name in loaded_agents:
        hallucination_history.setdefault(persona_name, [])

    summary = _build_summary(
        origin,
        target,
        steps_done,
        hallucination_history,
        loaded_agents,
        seen_interaction_pairs=seen_pairwise,
        required_interaction_pairs=expected_pairwise if require_pairwise_interactions else None,
    )
    summary_file = sim_storage / "reverie" / "hallucination_summary_no_compression.json"
    _write_json(summary_file, summary)

    total_elapsed = time.perf_counter() - started_at
    missing_pairs = expected_pairwise - seen_pairwise

    print(f"[done] simulation saved to: {sim_storage}")
    print(f"[done] wall time: {_format_duration(total_elapsed)}")
    print(f"[done] hallucination log: {hallu_file}")
    print(f"[done] hallucination summary: {summary_file}")
    if require_pairwise_interactions:
        print(f"[done] pairwise coverage: {len(seen_pairwise)}/{len(expected_pairwise)}")
        if missing_pairs:
            missing_str = ", ".join([f"({a}, {b})" for a, b in sorted(missing_pairs)])
            if best_effort_interactions:
                print(
                    f"[warning] Pairwise interaction requirement not fully met. "
                    f"Missing pairs: {missing_str}"
                )
            else:
                raise RuntimeError(
                    f"Pairwise interaction requirement not met. Missing pairs: {missing_str}"
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a 2-agent style simulation without memory compression and "
            "export hallucination metrics."
        )
    )
    parser.add_argument(
        "--origin",
        default="base_test_hallucination",
        help="Existing simulation folder to fork from.",
    )
    parser.add_argument(
        "--target",
        default=f"no_compression_2agents_{int(time.time())}",
        help="New simulation folder name under environment/frontend_server/storage.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=8000,
        help="Number of simulation steps to run.",
    )
    parser.add_argument(
        "--expected-agents",
        type=int,
        default=2,
        help="Fail fast unless this many agents are loaded from origin.",
    )
    parser.add_argument(
        "--server-sleep",
        type=float,
        default=0.0,
        help="Seconds to sleep inside backend loop each cycle (0 is fastest).",
    )
    parser.add_argument(
        "--live-state-interval",
        type=int,
        default=25,
        help="Write live-state snapshots every N steps (1 keeps original behavior).",
    )
    parser.add_argument(
        "--hallucination-interval",
        type=int,
        default=25,
        help="Compute hallucination metrics every N steps (1 keeps original behavior).",
    )
    parser.add_argument(
        "--disable-hallucination",
        action="store_true",
        help="Disable hallucination scoring during simulation for maximum speed.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=50,
        help="Print progress every N simulation steps.",
    )
    parser.add_argument(
        "--skip-ollama-check",
        action="store_true",
        help="Skip startup check that verifies Ollama server reachability.",
    )
    parser.add_argument(
        "--require-pairwise-interactions",
        action="store_true",
        help="Fail run unless every pair of loaded agents chats at least once.",
    )
    parser.add_argument(
        "--interaction-report-every",
        type=int,
        default=100,
        help="Print interaction coverage every N steps.",
    )
    parser.add_argument(
        "--force-interaction",
        action="store_true",
        help="Force decide_to_talk to return true via environment override.",
    )
    parser.add_argument(
        "--force-meeting-every",
        type=int,
        default=0,
        help="Teleport all agents to one tile every N steps (0 disables).",
    )
    parser.add_argument(
        "--meeting-x",
        type=int,
        default=None,
        help="X coordinate for forced meetings (requires --meeting-y).",
    )
    parser.add_argument(
        "--meeting-y",
        type=int,
        default=None,
        help="Y coordinate for forced meetings (requires --meeting-x).",
    )
    parser.add_argument(
        "--auto-extend-for-interactions",
        action="store_true",
        help="If pairwise requirement is unmet at --steps, continue up to --max-extra-steps.",
    )
    parser.add_argument(
        "--max-extra-steps",
        type=int,
        default=2000,
        help="Upper bound for extension when --auto-extend-for-interactions is enabled.",
    )
    parser.add_argument(
        "--best-effort-interactions",
        action="store_true",
        help="When pairwise requirement is enabled, report missing pairs but do not fail run.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.steps <= 0:
        print("[error] --steps must be > 0")
        return 1
    if args.live_state_interval <= 0:
        print("[error] --live-state-interval must be > 0")
        return 1
    if args.hallucination_interval <= 0:
        print("[error] --hallucination-interval must be > 0")
        return 1
    if args.progress_every <= 0:
        print("[error] --progress-every must be > 0")
        return 1
    if args.server_sleep < 0:
        print("[error] --server-sleep must be >= 0")
        return 1
    if args.interaction_report_every <= 0:
        print("[error] --interaction-report-every must be > 0")
        return 1
    if args.force_meeting_every < 0:
        print("[error] --force-meeting-every must be >= 0")
        return 1
    if args.max_extra_steps < 0:
        print("[error] --max-extra-steps must be >= 0")
        return 1
    if (args.meeting_x is None) != (args.meeting_y is None):
        print("[error] --meeting-x and --meeting-y must be set together")
        return 1
    if args.auto_extend_for_interactions and not args.require_pairwise_interactions:
        print("[error] --auto-extend-for-interactions requires --require-pairwise-interactions")
        return 1

    if not args.skip_ollama_check and not _ollama_is_reachable():
        print("[error] Ollama is not reachable at http://localhost:11434")
        print("[error] Start it with: ollama serve")
        print("[error] If Ollama is intentionally unavailable, rerun with --skip-ollama-check")
        return 1

    try:
        run_no_compression_simulation(
            origin=args.origin,
            target=args.target,
            steps=args.steps,
            expected_agents=args.expected_agents,
            hallucination_interval=args.hallucination_interval,
            live_state_interval=args.live_state_interval,
            server_sleep=args.server_sleep,
            progress_every=args.progress_every,
            disable_hallucination=args.disable_hallucination,
            require_pairwise_interactions=args.require_pairwise_interactions,
            interaction_report_every=args.interaction_report_every,
            force_interaction=args.force_interaction,
            force_meeting_every=args.force_meeting_every,
            meeting_x=args.meeting_x,
            meeting_y=args.meeting_y,
            auto_extend_for_interactions=args.auto_extend_for_interactions,
            max_extra_steps=args.max_extra_steps,
            best_effort_interactions=args.best_effort_interactions,
        )
        return 0
    except Exception as exc:
        print(f"[error] {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
