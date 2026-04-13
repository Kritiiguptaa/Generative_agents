"""
File: hallucination_extensions.py
Description: Drop-in extensions for hallucination_calculator.py that add two
new detection methods:

  1. ConversationFactChecker  — uses an LLM-as-judge to find factually wrong
     statements made by a persona in conversation, verified against their
     memory stream.

  2. MemoryProvenanceChecker  — uses embedding cosine similarity (already
     present in the codebase) to detect memory nodes that do not correspond
     to any real simulation event written in the movement/*.json files.

HOW TO INTEGRATE:
  In reverie.py, replace _calculate_persona_hallucination() with the extended
  version shown at the bottom of this file, or import and call these classes
  directly wherever you build your hallucination report.

DEPENDENCIES (all already present in the generative-agents repo):
  - openai  (or whatever LLM client utils.py configures)
  - numpy
  - global_methods.get_embedding()
  - utils.openai_config / ChatGPT() / similar call wrapper

COST NOTES:
  - ConversationFactChecker: 1 extra LLM call per conversation turn (~5-10
    per 8 000 steps). Negligible.
  - MemoryProvenanceChecker: pure embedding arithmetic, no extra LLM calls.
    Runs in <1 ms per node.
"""

from __future__ import annotations

import json
import math
import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy

# ---------------------------------------------------------------------------
# Re-use whatever LLM wrapper the codebase already exposes.
# The generative-agents repo ships several flavours; pick the right one for
# your install.  The pattern below matches the helper used in
# persona/prompt_template/*.py files.
# ---------------------------------------------------------------------------
try:
    from utils import *          # noqa: F401,F403  (brings in openai key setup)
    from global_methods import * # noqa: F401,F403  (brings in get_embedding)
except ImportError:
    pass  # running standalone / tests — callers must supply stubs


# ===========================================================================
# HELPER: cosine similarity
# ===========================================================================

def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """
    Returns cosine similarity in [-1, 1] between two embedding vectors.
    Returns 0.0 if either vector is zero-length.
    """
    a = numpy.array(vec_a, dtype=float)
    b = numpy.array(vec_b, dtype=float)
    norm_a = numpy.linalg.norm(a)
    norm_b = numpy.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(numpy.dot(a, b) / (norm_a * norm_b))


# ===========================================================================
# 1. CONVERSATION FACT CHECKER
# ===========================================================================

class ConversationFactChecker:
    """
    Detects factually wrong statements a persona made during a conversation
    by comparing them against the persona's own memory stream using an
    LLM-as-judge.

    Usage (inside reverie.py start_server loop):

        if persona.scratch.chat:          # a conversation just finished
            checker = ConversationFactChecker(persona)
            result  = checker.check(persona.scratch.chat)
            # result["violations"] is a list of dicts {statement, reason, severity}

    The checker extracts every claim made by *this* persona in the
    conversation, then asks a small LLM call whether each claim is supported
    by the persona's memory stream.  Claims with no memory support are
    flagged as hallucinations.
    """

    # Thresholds
    MAX_MEMORY_EVENTS = 60    # how many recent events to feed the judge
    MAX_MEMORY_THOUGHTS = 20  # how many recent thoughts to feed the judge

    def __init__(self, persona):
        """
        INPUT:
            persona: Persona instance (must have .scratch and .a_mem)
        """
        self.persona = persona

    # ------------------------------------------------------------------ #
    #  PUBLIC API                                                         #
    # ------------------------------------------------------------------ #

    def check(self, chat: List[List[str]]) -> Dict[str, Any]:
        """
        Run the fact-check against a single conversation.

        INPUT:
            chat: list of [speaker_name, utterance] pairs, e.g.
                  [["Isabella Rodriguez", "I finished my painting yesterday"],
                   ["Maria Lopez", "That sounds great!"]]
        OUTPUT:
            {
              "persona_name"  : str,
              "conversation"  : list,
              "violations"    : list of {statement, reason, severity},
              "violation_count": int,
              "score"         : float  # 0.0 (clean) – 1.0 (all claims wrong)
            }
        """
        persona_name = self.persona.scratch.name
        my_utterances = [utt for speaker, utt in chat
                         if speaker == persona_name]

        if not my_utterances:
            return self._empty_result(chat)

        memory_context = self._build_memory_context()
        persona_profile = self._build_persona_profile()
        violations = []

        for utterance in my_utterances:
            v = self._judge_utterance(utterance, memory_context, persona_profile)
            violations.extend(v)

        total_claims = self._estimate_claim_count(my_utterances)
        score = min(1.0, len(violations) / max(total_claims, 1))

        return {
            "persona_name": persona_name,
            "conversation": chat,
            "violations": violations,
            "violation_count": len(violations),
            "score": round(score, 4),
        }

    # ------------------------------------------------------------------ #
    #  PRIVATE HELPERS                                                    #
    # ------------------------------------------------------------------ #

    def _build_memory_context(self) -> str:
        """
        Builds a plain-text summary of the persona's recent memories to
        feed to the LLM judge.
        """
        a_mem = getattr(self.persona, 'a_mem', None)
        lines: List[str] = []

        if a_mem:
            for node in list(a_mem.seq_event)[:self.MAX_MEMORY_EVENTS]:
                ts = node.created.strftime("%Y-%m-%d %H:%M") if node.created else "?"
                lines.append(f"[event {ts}] {node.description}")
            for node in list(a_mem.seq_thought)[:self.MAX_MEMORY_THOUGHTS]:
                ts = node.created.strftime("%Y-%m-%d %H:%M") if node.created else "?"
                lines.append(f"[thought {ts}] {node.description}")

        return "\n".join(lines) if lines else "(no memories yet)"

    def _build_persona_profile(self) -> str:
        """Compact profile string from scratch fields."""
        s = self.persona.scratch
        parts = []
        if getattr(s, 'name', None):
            parts.append(f"Name: {s.name}")
        if getattr(s, 'innate', None):
            parts.append(f"Traits: {s.innate}")
        if getattr(s, 'learned', None):
            parts.append(f"Background: {s.learned}")
        if getattr(s, 'currently', None):
            parts.append(f"Currently: {s.currently}")
        if getattr(s, 'lifestyle', None):
            parts.append(f"Lifestyle: {s.lifestyle}")
        return "\n".join(parts)

    def _judge_utterance(
        self,
        utterance: str,
        memory_context: str,
        persona_profile: str,
    ) -> List[Dict[str, Any]]:
        """
        Sends one LLM call to judge whether the utterance contains any claims
        that are unsupported by or contradict the persona's memory.

        Returns a (possibly empty) list of violation dicts.
        """
        prompt = self._build_judge_prompt(utterance, memory_context, persona_profile)

        try:
            raw = self._call_llm(prompt)
            return self._parse_judge_response(raw, utterance)
        except Exception as exc:
            print(f"[ConversationFactChecker] LLM call failed: {exc}")
            return []

    def _build_judge_prompt(
        self,
        utterance: str,
        memory_context: str,
        persona_profile: str,
    ) -> str:
        return f"""You are a strict consistency auditor for an AI agent simulation.

AGENT PROFILE:
{persona_profile}

AGENT MEMORY STREAM (most recent first):
{memory_context}

STATEMENT MADE BY THIS AGENT IN CONVERSATION:
"{utterance}"

TASK:
Identify any factual claims in the statement that either:
  (a) contradict the agent's memory stream, OR
  (b) reference events, relationships, or facts with NO supporting evidence
      in the memory stream.

Do NOT flag:
  - Opinions, feelings, or intentions
  - Claims about general world knowledge (not about the agent's own life)
  - Statements clearly supported by the memory stream

Return ONLY valid JSON with this exact schema (no markdown, no explanation):
{{
  "violations": [
    {{
      "statement": "<exact quote of the problematic claim>",
      "reason": "<one sentence explaining the contradiction or lack of support>",
      "severity": "low" | "medium" | "high"
    }}
  ]
}}

If there are no violations return: {{"violations": []}}"""

    def _call_llm(self, prompt: str) -> str:
        """
        Calls the LLM using the same wrapper the rest of the codebase uses.

        The generative-agents repo exposes several helpers.  We try them in
        order so this works across different repo versions.
        """
        # --- Option 1: ChatGPT helper (most common in this repo) ----------
        try:
            from persona.prompt_template.gpt_structure import ChatGPT_request
            return ChatGPT_request(prompt)
        except (ImportError, Exception):
            pass

        # --- Option 2: direct openai call (fallback) ----------------------
        try:
            import openai
            resp = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=512,
            )
            return resp.choices[0].message.content
        except Exception:
            pass

        # --- Option 3: newer openai SDK (>=1.0) ---------------------------
        try:
            from openai import OpenAI
            client = OpenAI()
            resp = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=512,
            )
            return resp.choices[0].message.content
        except Exception as exc:
            raise RuntimeError(f"All LLM call attempts failed: {exc}") from exc

    def _parse_judge_response(
        self,
        raw: str,
        original_utterance: str,
    ) -> List[Dict[str, Any]]:
        """Parse the JSON the LLM judge returns."""
        try:
            clean = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            data = json.loads(clean)
            violations = data.get("violations", [])
            # Attach the full utterance for traceability
            for v in violations:
                v["utterance"] = original_utterance
            return violations
        except Exception as exc:
            print(f"[ConversationFactChecker] Failed to parse judge response: {exc}")
            print(f"  Raw response: {raw[:200]}")
            return []

    def _estimate_claim_count(self, utterances: List[str]) -> int:
        """
        Rough heuristic: count sentences across all utterances as a proxy for
        the number of distinct claims made.
        """
        total = 0
        for utt in utterances:
            total += max(1, utt.count('.') + utt.count('!') + utt.count('?'))
        return total

    def _empty_result(self, chat) -> Dict[str, Any]:
        return {
            "persona_name": self.persona.scratch.name,
            "conversation": chat,
            "violations": [],
            "violation_count": 0,
            "score": 0.0,
        }


# ===========================================================================
# 2. MEMORY PROVENANCE CHECKER
# ===========================================================================

class MemoryProvenanceChecker:
    """
    Detects memory nodes that do not correspond to any real simulation event
    by comparing each node's embedding to the embedding of the actual
    movement description recorded for the same time step.

    This uses the embeddings already stored in a_mem — no extra LLM calls.

    Usage (inside reverie.py, e.g. every 50 steps):

        checker = MemoryProvenanceChecker(
            persona      = persona,
            sim_storage  = Path(f"{fs_storage}/{self.sim_code}"),
            sec_per_step = self.sec_per_step,
            start_time   = self.start_time,
        )
        result = checker.check()
        # result["invented"] is a list of suspect memory nodes

    A node is flagged as "invented" if its cosine similarity to the actual
    movement description for its creation step falls below SIMILARITY_THRESHOLD.
    Only "event" type nodes are checked (not thoughts/reflections, which are
    legitimately generated by the agent and have no direct movement record).
    """

    SIMILARITY_THRESHOLD = 0.30   # below this -> likely invented / hallucinated
    UNCERTAIN_THRESHOLD = 0.50    # below this -> uncertain / worth logging

    def __init__(
        self,
        persona,
        sim_storage: Path,
        sec_per_step: int,
        start_time: datetime.datetime,
    ):
        """
        INPUT:
            persona      : Persona instance (must have .a_mem with .embeddings)
            sim_storage  : Path to the current simulation folder
                           e.g. Path("environment/frontend_server/storage/demo")
            sec_per_step : seconds of sim-time per step (from reverie_meta)
            start_time   : simulation start datetime (from reverie_meta)
        """
        self.persona = persona
        self.sim_storage = Path(sim_storage)
        self.sec_per_step = sec_per_step
        self.start_time = start_time

        # Cache movement descriptions to avoid re-reading files
        self._movement_cache: Dict[int, Optional[str]] = {}

    # ------------------------------------------------------------------ #
    #  PUBLIC API                                                         #
    # ------------------------------------------------------------------ #

    def check(self) -> Dict[str, Any]:
        """
        Run provenance check across all event-type memory nodes.

        OUTPUT:
            {
              "persona_name"     : str,
              "nodes_checked"    : int,
              "invented"         : list of InventedMemory dicts,
              "uncertain"        : list of InventedMemory dicts,
              "invented_count"   : int,
              "uncertain_count"  : int,
              "score"            : float,  # invented / nodes_checked
            }
        """
        a_mem = getattr(self.persona, 'a_mem', None)
        if not a_mem:
            return self._empty_result("No associative memory found.")

        event_nodes = [n for n in a_mem.seq_event]  # event-type only
        if not event_nodes:
            return self._empty_result("No event nodes in memory.")

        invented: List[Dict] = []
        uncertain: List[Dict] = []
        checked = 0

        for node in event_nodes:
            step = self._node_to_step(node.created)
            if step is None:
                continue  # can't map to a step — skip

            actual_desc = self._get_actual_description(step)
            if actual_desc is None:
                continue  # movement file missing — skip (e.g. step 0)

            node_embedding = self._get_node_embedding(node, a_mem)
            actual_embedding = self._get_actual_embedding(actual_desc)

            if node_embedding is None or actual_embedding is None:
                continue

            sim = _cosine_similarity(node_embedding, actual_embedding)
            checked += 1

            record = {
                "node_id": node.node_count,
                "created": node.created.strftime("%Y-%m-%d %H:%M:%S"),
                "step": step,
                "memory": node.description,
                "actual_event": actual_desc,
                "similarity": round(sim, 4),
            }

            if sim < self.SIMILARITY_THRESHOLD:
                record["verdict"] = "INVENTED"
                invented.append(record)
            elif sim < self.UNCERTAIN_THRESHOLD:
                record["verdict"] = "UNCERTAIN"
                uncertain.append(record)

        score = round(len(invented) / max(checked, 1), 4)

        return {
            "persona_name": self.persona.scratch.name,
            "nodes_checked": checked,
            "invented": invented,
            "uncertain": uncertain,
            "invented_count": len(invented),
            "uncertain_count": len(uncertain),
            "score": score,
        }

    # ------------------------------------------------------------------ #
    #  PRIVATE HELPERS                                                    #
    # ------------------------------------------------------------------ #

    def _node_to_step(self, created: Optional[datetime.datetime]) -> Optional[int]:
        """
        Converts a memory node's creation datetime to the simulation step
        number by computing elapsed seconds since start_time and dividing
        by sec_per_step.
        """
        if created is None or self.sec_per_step <= 0:
            return None
        try:
            elapsed_secs = (created - self.start_time).total_seconds()
            step = int(elapsed_secs / self.sec_per_step)
            return max(0, step)
        except Exception:
            return None

    def _get_actual_description(self, step: int) -> Optional[str]:
        """
        Loads the movement description for this persona at the given step
        from the movement/{step}.json file.

        Returns None if the file doesn't exist or the persona isn't in it.
        """
        if step in self._movement_cache:
            return self._movement_cache[step]

        movement_file = self.sim_storage / "movement" / f"{step}.json"
        if not movement_file.exists():
            self._movement_cache[step] = None
            return None

        try:
            with movement_file.open() as f:
                data = json.load(f)
            persona_name = self.persona.scratch.name
            desc = (data
                    .get("persona", {})
                    .get(persona_name, {})
                    .get("description", None))
            self._movement_cache[step] = desc
            return desc
        except Exception:
            self._movement_cache[step] = None
            return None

    def _get_node_embedding(self, node, a_mem) -> Optional[List[float]]:
        """
        Retrieves the pre-computed embedding for this memory node.
        The generative-agents repo stores embeddings in a_mem.embeddings
        keyed by node.embedding_key.
        """
        try:
            embeddings = getattr(a_mem, 'embeddings', {})
            vec = embeddings.get(node.embedding_key)
            if vec is not None:
                return vec
        except Exception:
            pass
        return None

    def _get_actual_embedding(self, description: str) -> Optional[List[float]]:
        """
        Gets or computes the embedding for the actual movement description.
        Uses the same get_embedding() function the rest of the codebase uses,
        so no extra API setup needed.
        """
        try:
            # get_embedding() is imported from global_methods via utils
            vec = get_embedding(description)
            return vec
        except Exception as exc:
            print(f"[MemoryProvenanceChecker] get_embedding failed: {exc}")
            return None

    def _empty_result(self, reason: str) -> Dict[str, Any]:
        return {
            "persona_name": getattr(self.persona.scratch, 'name', 'UNKNOWN'),
            "nodes_checked": 0,
            "invented": [],
            "uncertain": [],
            "invented_count": 0,
            "uncertain_count": 0,
            "score": 0.0,
            "note": reason,
        }
