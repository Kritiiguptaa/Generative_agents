"""
Author: Generated for Hallucination Analysis
File: hallucination_calculator.py
Description: Calculates hallucination metrics for agents' persona/context/action/plan
consistency. Uses actual scratch and associative memory structures from the
generative agents codebase.

Hallucination Categories:
1. Persona-Context: Inconsistency between persona identity fields and memory content
2. Context-Action: Mismatch between recent memories/context and current action
3. Action-Plan: Deviation between current action and what the schedule says should happen now
4. Plan-Persona: Incompatibility between the daily plan and persona's identity/lifestyle

Fixes applied:
- Action-Plan Check B/C: was comparing parent task keywords INTO child act_description
  (always high overlap since decomposed actions contain parent text). Now checks
  action keywords against the hourly_org schedule (the non-decomposed ground truth).
- Action-Plan Check D: duration check was trivially passing because decomposed
  act_duration is always much smaller than the hourly slot duration. Replaced with
  a check that the current action type (sleep vs active) matches the plan.
- Plan-Persona Check D: hourly_org always sums to exactly 1440 so the duration
  check never fired. Replaced with innate-traits-in-plan alignment check.
- _keywords_from_text: added domain-specific stopwords that are too generic for
  schedule matching (wake, sleep, work, etc.) so keyword overlap is more signal-rich.
"""

import sys
import json
import datetime
from typing import Dict, List, Tuple, Any

sys.path.append('../../')
from global_methods import *


class HallucinationCalculator:
    """
    Calculates hallucination scores for agent consistency across four dimensions:
    persona-context, context-action, action-plan, and plan-persona.

    Design notes:
    - Each dimension's score is independently computed and stored once.
    - `inconsistencies` is only populated during `calculate_overall_hallucination()`
      or when calling sub-methods individually for the first time (guarded by a
      `_calculated` flag to prevent duplication on repeated calls).
    - Scores are float in [0.0, 1.0]; higher = more hallucination.
    - checks_failed is always a float; each sub-check contributes a weight in (0, 1]
      so the final score is a meaningful proportion.
    """

    def __init__(self, persona):
        """
        INPUT:
            persona: The Persona class instance (must have .scratch, .a_mem, .s_mem)
        """
        self.persona = persona
        self.scores: Dict[str, float] = {}
        self.inconsistencies: List[str] = []
        self._calculated = set()   # tracks which dimensions have been run

    # ------------------------------------------------------------------ #
    #  HELPERS                                                             #
    # ------------------------------------------------------------------ #

    def _keywords_from_text(self, text: str, n: int = 5) -> List[str]:
        """
        Returns up to n meaningful (len > 3) lowercase words from text.
        Filters common stop-words AND domain-generic schedule words so that
        keyword matching is actually discriminative between different personas.

        The domain stopwords (wake, sleep, work, etc.) appear in virtually
        every agent's schedule regardless of persona, so including them caused
        overlap ratios to be artificially high and checks to never fire.
        """
        STOPWORDS = {
            # General English stopwords
            "that", "this", "with", "from", "they", "have", "been",
            "will", "their", "there", "were", "what", "when", "which",
            "about", "would", "could", "should", "some", "also",
            # Domain-generic schedule words — too common to be discriminative
            "wake", "wakes", "waking", "sleep", "sleeps", "sleeping",
            "goes", "going", "work", "works", "working", "time", "home",
            "make", "makes", "making", "take", "takes", "taking",
            "around", "every", "usually", "likes", "enjoy", "enjoys",
            "start", "starts", "starting", "finish", "done", "plan",
            "plans", "today", "daily", "routine", "morning", "evening",
            "night", "afternoon", "lunch", "dinner", "breakfast",
        }
        words = [w.strip(".,;:!?\"'()") for w in str(text).lower().split()]
        filtered = [w for w in words if len(w) > 3 and w not in STOPWORDS]
        return filtered[:n]

    def _overlap_ratio(self, keywords: List[str], text: str) -> float:
        """
        Returns fraction of keywords found in text (0.0 – 1.0).
        If keywords list is empty, returns 1.0 (nothing to check = no failure).
        """
        if not keywords:
            return 1.0
        text_lower = text.lower()
        hits = sum(1 for kw in keywords if kw in text_lower)
        return hits / len(keywords)

    def _get_planned_task_at_current_time(self) -> Tuple[str, int]:
        """
        Uses scratch's own index helper to find what task the persona SHOULD be
        doing right now according to f_daily_schedule (the decomposed schedule).

        Returns (task_description, duration_minutes) or ("", 0) if unavailable.
        """
        scratch = self.persona.scratch
        if not scratch.f_daily_schedule or not scratch.curr_time:
            return ("", 0)
        try:
            idx = scratch.get_f_daily_schedule_index()
            if 0 <= idx < len(scratch.f_daily_schedule):
                return tuple(scratch.f_daily_schedule[idx])
        except Exception:
            pass
        return ("", 0)

    def _get_hourly_planned_task_at_current_time(self) -> Tuple[str, int]:
        """
        Same as above but for f_daily_schedule_hourly_org (the non-decomposed
        hourly schedule — the ground truth broad strokes plan).
        """
        scratch = self.persona.scratch
        if not scratch.f_daily_schedule_hourly_org or not scratch.curr_time:
            return ("", 0)
        try:
            idx = scratch.get_f_daily_schedule_hourly_org_index()
            if 0 <= idx < len(scratch.f_daily_schedule_hourly_org):
                return tuple(scratch.f_daily_schedule_hourly_org[idx])
        except Exception:
            pass
        return ("", 0)

    def _is_sleep_action(self, text: str) -> bool:
        """Returns True if the text describes a sleep/rest activity."""
        sleep_words = {"sleep", "sleeping", "asleep", "nap", "napping", "in bed", "resting"}
        text_lower = text.lower()
        return any(w in text_lower for w in sleep_words)

    def _add_issue(self, msg: str):
        """Append an inconsistency message (avoids exact duplicates)."""
        if msg not in self.inconsistencies:
            self.inconsistencies.append(msg)

    # ------------------------------------------------------------------ #
    #  DIMENSION 1 — PERSONA ↔ CONTEXT                                    #
    # ------------------------------------------------------------------ #

    def calculate_persona_context_hallucination(self) -> float:
        """
        Checks that the persona's stated identity (name, innate traits,
        learned traits, currently, lifestyle) is reflected in their associative
        memory (events + thoughts).

        Sub-checks:
        A) Persona appears as subject in at least one stored memory event.
        B) Innate traits have some footprint in memory descriptions.
        C) 'currently' field keywords appear in recent events or thoughts.
        D) 'learned' field keywords appear in memory.

        OUTPUT:
            score in [0.0, 1.0]  (higher = more hallucination)
        """
        if 'persona_context' in self._calculated:
            return self.scores.get('persona_context', 0.0)

        scratch = self.persona.scratch
        a_mem = getattr(self.persona, 'a_mem', None)

        checks_failed = 0.0
        total_checks = 0

        # Build searchable memory corpus
        all_event_desc = " ".join(
            n.description for n in (a_mem.seq_event if a_mem else [])
        ).lower()
        all_thought_desc = " ".join(
            n.description for n in (a_mem.seq_thought if a_mem else [])
        ).lower()
        all_mem_text = all_event_desc + " " + all_thought_desc

        # --- Check A: persona name appears as subject in memory events ---
        total_checks += 1
        if a_mem and a_mem.seq_event:
            persona_name = scratch.name or ""
            subject_hits = [
                n for n in a_mem.seq_event
                if str(n.subject).lower() == persona_name.lower()
            ]
            if not subject_hits:
                checks_failed += 1.0
                self._add_issue(
                    f"Persona '{persona_name}' never appears as subject "
                    f"in {len(a_mem.seq_event)} memory events."
                )

        # --- Check B: innate traits in memory ----------------------------
        total_checks += 1
        if scratch.innate:
            innate_kws = self._keywords_from_text(scratch.innate, n=6)
            ratio = self._overlap_ratio(innate_kws, all_mem_text)
            if ratio < 0.25:
                checks_failed += 1.0
                self._add_issue(
                    f"Innate traits '{scratch.innate}' have very low overlap "
                    f"({ratio:.0%}) with memory content."
                )

        # --- Check C: 'currently' field reflected in recent events -------
        total_checks += 1
        if scratch.currently and a_mem and a_mem.seq_event:
            curr_kws = self._keywords_from_text(scratch.currently, n=5)
            recent_text = " ".join(
                n.description for n in a_mem.seq_event[:20]
            ).lower()
            ratio = self._overlap_ratio(curr_kws, recent_text)
            if ratio < 0.2:
                checks_failed += 1.0
                self._add_issue(
                    f"'currently' field ('{scratch.currently}') has low overlap "
                    f"({ratio:.0%}) with the 20 most recent events."
                )

        # --- Check D: 'learned' traits in memory -------------------------
        total_checks += 1
        if scratch.learned:
            learned_kws = self._keywords_from_text(scratch.learned, n=5)
            ratio = self._overlap_ratio(learned_kws, all_mem_text)
            if ratio < 0.2:
                checks_failed += 1.0
                self._add_issue(
                    f"Learned traits '{scratch.learned}' have very low overlap "
                    f"({ratio:.0%}) with memory content."
                )

        score = (checks_failed / total_checks) if total_checks > 0 else 0.0
        self.scores['persona_context'] = score
        self._calculated.add('persona_context')
        return score

    # ------------------------------------------------------------------ #
    #  DIMENSION 2 — CONTEXT ↔ ACTION                                     #
    # ------------------------------------------------------------------ #

    def calculate_context_action_hallucination(self) -> float:
        """
        Checks that the current action is consistent with recent memory context.

        Sub-checks:
        A) act_address sector/arena matches the persona's known living_area or
           recent event addresses.
        B) act_description keywords appear somewhere in the last N memory events.
        C) act_event triple subject matches persona name (identity coherence).
        D) If persona is marked as chatting, a recent chat memory exists.

        OUTPUT:
            score in [0.0, 1.0]
        """
        if 'context_action' in self._calculated:
            return self.scores.get('context_action', 0.0)

        scratch = self.persona.scratch
        a_mem = getattr(self.persona, 'a_mem', None)

        checks_failed = 0.0
        total_checks = 0

        # --- Check A: act_address plausibility --------------------------
        # act_address format: "world:sector:arena:object"
        # Check that the sector appears in living_area OR recent event descriptions.
        total_checks += 1
        if scratch.act_address:
            addr_parts = scratch.act_address.split(":")
            sector = addr_parts[1].lower() if len(addr_parts) > 1 else ""
            arena  = addr_parts[2].lower() if len(addr_parts) > 2 else ""

            living_area_str = str(scratch.living_area or "").lower()
            recent_event_desc = " ".join(
                n.description for n in (a_mem.seq_event[:30] if a_mem else [])
            ).lower()
            combined_context = living_area_str + " " + recent_event_desc

            if sector and sector not in combined_context:
                checks_failed += 1.0
                self._add_issue(
                    f"Current action location sector '{sector}' (from "
                    f"act_address '{scratch.act_address}') not found in "
                    f"living_area or recent memory."
                )
            elif arena and arena not in combined_context:
                checks_failed += 0.5
                self._add_issue(
                    f"Current action arena '{arena}' not found in recent context."
                )

        # --- Check B: act_description coherence with recent memories -----
        total_checks += 1
        if scratch.act_description and a_mem and a_mem.seq_event:
            action_kws = self._keywords_from_text(scratch.act_description, n=4)
            recent_text = " ".join(
                n.description for n in a_mem.seq_event[:15]
            ).lower()
            ratio = self._overlap_ratio(action_kws, recent_text)
            if ratio < 0.25:
                checks_failed += 1.0
                self._add_issue(
                    f"Current action '{scratch.act_description}' has low "
                    f"coherence ({ratio:.0%}) with the 15 most recent memories."
                )

        # --- Check C: act_event subject must be the persona itself -------
        total_checks += 1
        if scratch.act_event and scratch.act_event[0]:
            event_subject = str(scratch.act_event[0]).lower()
            persona_name  = str(scratch.name or "").lower()
            if event_subject != persona_name:
                checks_failed += 1.0
                self._add_issue(
                    f"act_event subject '{scratch.act_event[0]}' does not match "
                    f"persona name '{scratch.name}'."
                )

        # --- Check D: chatting state consistency -------------------------
        total_checks += 1
        if scratch.chatting_with and a_mem:
            chat_partner = scratch.chatting_with.lower()
            last_chat = a_mem.get_last_chat(chat_partner)
            if not last_chat:
                checks_failed += 1.0
                self._add_issue(
                    f"Persona is marked as chatting with '{scratch.chatting_with}' "
                    f"but no chat memory exists for them."
                )

        score = (checks_failed / total_checks) if total_checks > 0 else 0.0
        self.scores['context_action'] = score
        self._calculated.add('context_action')
        return score

    # ------------------------------------------------------------------ #
    #  DIMENSION 3 — ACTION ↔ PLAN                                        #
    # ------------------------------------------------------------------ #

    def calculate_action_plan_hallucination(self) -> float:
        """
        Checks that the current action matches what the schedule planned for
        this exact time slot.

        Sub-checks:
        A) A daily plan (f_daily_schedule) exists and is non-empty.
        B) act_description keywords appear in the hourly_org schedule text
           (the non-decomposed ground truth). This fixes the original bug where
           parent-task keywords trivially appeared in decomposed child actions,
           making the check always pass.
        C) The hourly_org plan exists and the CURRENT HOURLY SLOT task has
           keyword overlap with act_description — checked bidirectionally.
        D) Sleep/active state consistency: if the hourly plan says sleeping,
           the action should be sleeping, and vice versa. This catches the
           most egregious action-plan mismatches without being fooled by
           decomposition text formatting.

        OUTPUT:
            score in [0.0, 1.0]
        """
        if 'action_plan' in self._calculated:
            return self.scores.get('action_plan', 0.0)

        scratch = self.persona.scratch

        checks_failed = 0.0
        total_checks = 0

        # --- Check A: plan existence ------------------------------------
        total_checks += 1
        if not scratch.f_daily_schedule:
            checks_failed += 1.0
            self._add_issue("f_daily_schedule is empty — no daily plan exists.")
            self.scores['action_plan'] = 1.0
            self._calculated.add('action_plan')
            return 1.0

        # Build the full hourly_org text for broad searching
        hourly_org_text = " ".join(
            task for task, _ in (scratch.f_daily_schedule_hourly_org or [])
        ).lower()

        # --- Check B: act_description keywords appear in hourly_org ------
        # FIX: Previously compared planned_task keywords INTO act_description.
        # Decomposed act_description always contains the parent task text, so
        # overlap was trivially 100% and the check never fired.
        # Now we check act_description keywords against the FULL hourly_org
        # text — this tells us if the action TYPE exists anywhere in today's
        # plan at all.
        total_checks += 1
        if scratch.act_description and hourly_org_text:
            action_kws = self._keywords_from_text(scratch.act_description, n=5)
            ratio = self._overlap_ratio(action_kws, hourly_org_text)
            if ratio < 0.2:
                checks_failed += 1.0
                self._add_issue(
                    f"Current action '{scratch.act_description}' keywords "
                    f"({action_kws}) have low overlap ({ratio:.0%}) with today's "
                    f"hourly plan — action may not be planned for today."
                )
        elif not hourly_org_text:
            checks_failed += 0.5
            self._add_issue("f_daily_schedule_hourly_org is empty.")

        # --- Check C: current hourly slot bidirectional alignment --------
        # FIX: Now checks BOTH directions:
        #   - hourly slot keywords in act_description (original direction, broken)
        #   - act_description keywords in hourly slot (new direction)
        # Takes the MINIMUM ratio — both must align for no hallucination.
        total_checks += 1
        if scratch.f_daily_schedule_hourly_org:
            hourly_task, _ = self._get_hourly_planned_task_at_current_time()
            if hourly_task and scratch.act_description:
                hourly_kws = self._keywords_from_text(hourly_task, n=5)
                action_kws = self._keywords_from_text(scratch.act_description, n=5)

                # Direction 1: does hourly task appear in action description?
                ratio_h_in_a = self._overlap_ratio(hourly_kws, scratch.act_description)
                # Direction 2: does action appear in hourly slot text?
                ratio_a_in_h = self._overlap_ratio(action_kws, hourly_task)

                # Use minimum — both sides must agree
                ratio = min(ratio_h_in_a, ratio_a_in_h)

                if ratio < 0.2:
                    checks_failed += 1.0
                    self._add_issue(
                        f"At {scratch.curr_time.strftime('%H:%M') if scratch.curr_time else '??:??'}, "
                        f"hourly plan says '{hourly_task}' but current action is "
                        f"'{scratch.act_description}' "
                        f"(bidirectional overlap: {ratio:.0%})."
                    )
            elif not hourly_task:
                checks_failed += 0.5
                self._add_issue(
                    "Could not determine hourly planned task for the current time slot."
                )
        else:
            checks_failed += 1.0
            self._add_issue("f_daily_schedule_hourly_org is empty.")

        # --- Check D: sleep/active state consistency ---------------------
        # FIX: Original check compared act_duration to planned_dur, but
        # decomposed tasks always have much shorter durations than the hourly
        # slot (e.g., 5 min vs 60 min), so ratio was always < 0.33 and the
        # check would fire constantly — or was skipped entirely.
        # Now we check the binary sleep/active state, which is the most
        # meaningful coarse alignment between action and plan.
        total_checks += 1
        hourly_task, _ = self._get_hourly_planned_task_at_current_time()
        if hourly_task and scratch.act_description:
            plan_is_sleep   = self._is_sleep_action(hourly_task)
            action_is_sleep = self._is_sleep_action(scratch.act_description)
            if plan_is_sleep != action_is_sleep:
                checks_failed += 1.0
                state_plan   = "sleep" if plan_is_sleep   else "active"
                state_action = "sleep" if action_is_sleep else "active"
                self._add_issue(
                    f"Plan expects '{state_plan}' state ('{hourly_task}') but "
                    f"action is '{state_action}' ('{scratch.act_description}')."
                )

        score = (checks_failed / total_checks) if total_checks > 0 else 0.0
        self.scores['action_plan'] = score
        self._calculated.add('action_plan')
        return score

    # ------------------------------------------------------------------ #
    #  DIMENSION 4 — PLAN ↔ PERSONA                                       #
    # ------------------------------------------------------------------ #

    def calculate_plan_persona_hallucination(self) -> float:
        """
        Checks that the planned activities are compatible with the persona's
        identity (lifestyle, innate traits, occupation, daily_plan_req).

        Sub-checks:
        A) Lifestyle keywords appear in the daily plan.
        B) daily_plan_req keywords appear in the plan.
        C) 'currently' keywords (occupation / ongoing projects) appear in plan.
        D) Innate trait keywords appear in the plan.
           FIX: Original Check D verified hourly_org total == 1440 min, which
           is always true by construction in generate_hourly_schedule(), so the
           check never fired. Replaced with innate-traits-in-plan alignment,
           which actually varies between personas and catches cases where the
           plan is generic/inconsistent with the persona's character.

        OUTPUT:
            score in [0.0, 1.0]
        """
        if 'plan_persona' in self._calculated:
            return self.scores.get('plan_persona', 0.0)

        scratch = self.persona.scratch

        checks_failed = 0.0
        total_checks = 0

        # Build searchable plan text from both schedule variants
        plan_text = " ".join(
            task for task, _ in (scratch.f_daily_schedule or [])
        ).lower()
        hourly_text = " ".join(
            task for task, _ in (scratch.f_daily_schedule_hourly_org or [])
        ).lower()
        combined_plan_text = plan_text + " " + hourly_text

        if not combined_plan_text.strip():
            self.scores['plan_persona'] = 1.0
            self._calculated.add('plan_persona')
            self._add_issue("Both schedule lists are empty — plan-persona check skipped.")
            return 1.0

        # --- Check A: lifestyle → plan alignment -------------------------
        total_checks += 1
        if scratch.lifestyle:
            lifestyle_kws = self._keywords_from_text(scratch.lifestyle, n=6)
            ratio = self._overlap_ratio(lifestyle_kws, combined_plan_text)
            if ratio < 0.3:
                checks_failed += 1.0
                self._add_issue(
                    f"Lifestyle '{scratch.lifestyle}' has low alignment "
                    f"({ratio:.0%}) with the daily plan."
                )

        # --- Check B: daily_plan_req → plan alignment --------------------
        total_checks += 1
        if scratch.daily_plan_req:
            req_kws = self._keywords_from_text(scratch.daily_plan_req, n=6)
            ratio = self._overlap_ratio(req_kws, combined_plan_text)
            if ratio < 0.25:
                checks_failed += 1.0
                self._add_issue(
                    f"daily_plan_req '{scratch.daily_plan_req}' has low alignment "
                    f"({ratio:.0%}) with the daily plan."
                )

        # --- Check C: 'currently' → plan alignment -----------------------
        total_checks += 1
        if scratch.currently:
            curr_kws = self._keywords_from_text(scratch.currently, n=5)
            ratio = self._overlap_ratio(curr_kws, combined_plan_text)
            if ratio < 0.2:
                checks_failed += 1.0
                self._add_issue(
                    f"'currently' field '{scratch.currently}' has low alignment "
                    f"({ratio:.0%}) with the daily plan."
                )

        # --- Check D: innate traits → plan alignment ---------------------
        # FIX: Replaced the always-passing duration==1440 check.
        # Innate traits describe the persona's core character and should leave
        # a footprint in what activities they plan. A painter should have
        # painting in their plan; a researcher should have research/reading, etc.
        total_checks += 1
        if scratch.innate:
            innate_kws = self._keywords_from_text(scratch.innate, n=6)
            ratio = self._overlap_ratio(innate_kws, combined_plan_text)
            if ratio < 0.15:
                checks_failed += 1.0
                self._add_issue(
                    f"Innate traits '{scratch.innate}' have low alignment "
                    f"({ratio:.0%}) with the daily plan — persona may be acting "
                    f"out of character."
                )

        score = (checks_failed / total_checks) if total_checks > 0 else 0.0
        self.scores['plan_persona'] = score
        self._calculated.add('plan_persona')
        return score

    # ------------------------------------------------------------------ #
    #  OVERALL                                                             #
    # ------------------------------------------------------------------ #

    def calculate_overall_hallucination(self) -> float:
        """
        Calculates weighted average of all four dimensions.

        Weights:
            Persona-Context : 25%
            Context-Action  : 25%
            Action-Plan     : 25%
            Plan-Persona    : 25%

        OUTPUT:
            overall_hallucination_score in [0.0, 1.0]
        """
        try:
            d1 = self.calculate_persona_context_hallucination()
            d2 = self.calculate_context_action_hallucination()
            d3 = self.calculate_action_plan_hallucination()
            d4 = self.calculate_plan_persona_hallucination()

            overall = (d1 + d2 + d3 + d4) / 4.0
            self.scores['overall'] = overall
            return overall

        except Exception as e:
            name = getattr(self.persona, 'name', 'UNKNOWN')
            print(f"[HallucinationCalculator] Error for '{name}': "
                  f"{type(e).__name__}: {e}")
            print("  TIP: Ensure persona has .scratch, .a_mem, .s_mem populated.")
            self.scores['overall'] = 0.0
            return 0.0

    # ------------------------------------------------------------------ #
    #  REPORTING                                                           #
    # ------------------------------------------------------------------ #

    def get_report(self) -> Dict[str, Any]:
        """
        Returns a comprehensive hallucination report dict.
        Triggers full calculation if not yet done.
        """
        if 'overall' not in self.scores:
            self.calculate_overall_hallucination()

        return {
            'timestamp': datetime.datetime.now().isoformat(),
            'persona_name': self.persona.name,
            'scores': {
                'persona_context': round(self.scores.get('persona_context', 0.0), 4),
                'context_action':  round(self.scores.get('context_action',  0.0), 4),
                'action_plan':     round(self.scores.get('action_plan',     0.0), 4),
                'plan_persona':    round(self.scores.get('plan_persona',    0.0), 4),
                'overall':         round(self.scores.get('overall',         0.0), 4),
            },
            'inconsistencies': self.inconsistencies,
            'summary': self._generate_summary()
        }

    def _generate_summary(self) -> str:
        overall = self.scores.get('overall', 0.0)
        if overall < 0.2:
            status = "CONSISTENT — agent maintains strong coherence"
        elif overall < 0.4:
            status = "MOSTLY CONSISTENT — minor inconsistencies detected"
        elif overall < 0.6:
            status = "MODERATE HALLUCINATION — notable inconsistencies present"
        elif overall < 0.8:
            status = "SIGNIFICANT HALLUCINATION — major inconsistencies detected"
        else:
            status = "SEVERE HALLUCINATION — agent highly incoherent"
        return (f"{status} "
                f"(Score: {overall:.2f}/1.0, Issues: {len(self.inconsistencies)})")


# ------------------------------------------------------------------ #
#  CONVENIENCE FUNCTION                                               #
# ------------------------------------------------------------------ #

def analyze_persona_hallucination(persona, verbose: bool = False) -> Dict[str, Any]:
    """
    Convenience wrapper to analyze a single persona's hallucination.

    INPUT:
        persona: Persona instance
        verbose: Whether to print a detailed report to stdout
    OUTPUT:
        Hallucination report dict
    """
    calculator = HallucinationCalculator(persona)
    report = calculator.get_report()

    if verbose:
        W = 60
        print(f"\n{'=' * W}")
        print(f"HALLUCINATION ANALYSIS: {persona.name}")
        print(f"{'=' * W}")
        print(f"  Overall Score   : {report['scores']['overall']:.4f}")
        print(f"  Persona-Context : {report['scores']['persona_context']:.4f}")
        print(f"  Context-Action  : {report['scores']['context_action']:.4f}")
        print(f"  Action-Plan     : {report['scores']['action_plan']:.4f}")
        print(f"  Plan-Persona    : {report['scores']['plan_persona']:.4f}")
        print(f"\n  Summary: {report['summary']}")
        if report['inconsistencies']:
            print(f"\n  Inconsistencies found ({len(report['inconsistencies'])}):")
            for inc in report['inconsistencies']:
                print(f"    • {inc}")
        print(f"{'=' * W}\n")

    return report

##----------------------------------------------------------------------
#original code(neeche vala)
##-----------------------------------------------------------------------------

# """
# Author: Generated for Hallucination Analysis
# File: hallucination_calculator.py
# Description: Calculates hallucination metrics for agents' persona/context/action/plan
# consistency. Minimal implementation to detect inconsistencies.

# Hallucination Categories:
# 1. Persona-Context: Inconsistency between persona identity and stored memories
# 2. Context-Action: Mismatch between remembered context and current actions
# 3. Action-Plan: Deviation between executed actions and planned activities
# 4. Plan-Persona: Incompatibility between plans and persona attributes (lifestyle, values)
# """

# import sys
# import json
# from datetime import datetime
# from typing import Dict, List, Tuple, Any

# sys.path.append('../../')

# from global_methods import *


# class HallucinationCalculator:
#     """
#     Calculates hallucination scores for agent consistency across four dimensions:
#     persona, context, actions, and plans.
#     """
    
#     def __init__(self, persona):
#         """
#         Initialize hallucination calculator with a persona instance.
        
#         INPUT:
#             persona: The Persona class instance
#         OUTPUT:
#             None
#         """
#         self.persona = persona
#         self.scores = {}
#         self.inconsistencies = []
#         self._validate_memory_structures()
    
#     def _validate_memory_structures(self):
#         """
#         Validates that required memory structures exist and are accessible.
#         Ensures graceful handling of minimal/empty memory setups.
#         """
#         # Ensure associative memory has required attributes
#         if hasattr(self.persona, 'a_mem') and self.persona.a_mem:
#             if not hasattr(self.persona.a_mem, 'seq_event'):
#                 self.persona.a_mem.seq_event = []
#             if not hasattr(self.persona.a_mem, 'seq_thought'):
#                 self.persona.a_mem.seq_thought = []
#             if not hasattr(self.persona.a_mem, 'seq_chat'):
#                 self.persona.a_mem.seq_chat = []
    
#     def calculate_persona_context_hallucination(self) -> float:
#         """
#         Measures consistency between persona's stated identity and associative memory.
        
#         Checks:
#         - Persona name appears in memories
#         - Lifestyle claims match memory patterns
#         - Identity traits consistent with recorded events
        
#         OUTPUT:
#             hallucination_score (0.0-1.0): Higher = more hallucination detected
#         """
#         score = 0.0
#         checks_failed = 0
#         total_checks = 0
        
#         # Check 1: Persona identity in memories
#         total_checks += 1
#         persona_name = self.persona.name
#         if hasattr(self.persona, 'a_mem') and self.persona.a_mem:
#             # Count events where persona is subject
#             seq_event = getattr(self.persona.a_mem, 'seq_event', [])
#             events_as_subject = [e for e in seq_event 
#                                 if hasattr(e, 'subject') and persona_name in str(e.subject)]
#             if len(events_as_subject) == 0 and len(seq_event) > 0:
#                 checks_failed += 1
#                 self.inconsistencies.append(
#                     f"Persona '{persona_name}' not found as subject in associative memory"
#                 )
        
#         # Check 2: Lifestyle consistency
#         total_checks += 1
#         if hasattr(self.persona.scratch, 'lifestyle'):
#             lifestyle = self.persona.scratch.lifestyle
#             # Check if lifestyle activities appear in daily schedule
#             if hasattr(self.persona.scratch, 'f_daily_schedule'):
#                 schedule = self.persona.scratch.f_daily_schedule
#                 lifestyle_keywords = lifestyle.lower().split()[:3]  # First 3 words
#                 schedule_str = str(schedule).lower()
#                 matches = sum(1 for kw in lifestyle_keywords if kw in schedule_str)
#                 if matches < len(lifestyle_keywords):
#                     checks_failed += 0.5
#                     self.inconsistencies.append(
#                         f"Lifestyle '{lifestyle}' not well reflected in schedule"
#                     )
        
#         if total_checks > 0:
#             score = checks_failed / total_checks
        
#         self.scores['persona_context'] = score
#         return score
    
#     def calculate_context_action_hallucination(self) -> float:
#         """
#         Measures consistency between stored memories (context) and current actions.
        
#         Checks:
#         - Current action aligns with recent memories
#         - Action doesn't contradict known locations/locations
#         - Action history is coherent with memory
        
#         OUTPUT:
#             hallucination_score (0.0-1.0): Higher = more hallucination detected
#         """
#         score = 0.0
#         checks_failed = 0
#         total_checks = 0
        
#         # Check 1: Current location consistency
#         total_checks += 1
#         if (hasattr(self.persona.scratch, 'curr_tile') and 
#             hasattr(self.persona.s_mem, 'tree')):
#             curr_loc = str(self.persona.scratch.curr_tile)
#             # Verify location exists in spatial memory
#             try:
#                 loc_exists = self.persona.s_mem.search(curr_loc)
#                 if not loc_exists:
#                     checks_failed += 1
#                     self.inconsistencies.append(
#                         f"Current location '{curr_loc}' not found in spatial memory"
#                     )
#             except:
#                 pass
        
#         # Check 2: Action-Memory coherence
#         total_checks += 1
#         if (hasattr(self.persona.scratch, 'act_description') and 
#             hasattr(self.persona, 'a_mem')):
#             curr_action = self.persona.scratch.act_description
#             seq_event = getattr(self.persona.a_mem, 'seq_event', [])
#             recent_events = seq_event[-10:] if seq_event else []
#             action_keywords = str(curr_action).lower().split()[:2]
#             recent_str = str(recent_events).lower()
#             action_coherence = sum(1 for kw in action_keywords if kw in recent_str)
#             if action_coherence == 0 and len(recent_events) > 0:
#                 checks_failed += 0.5
#                 self.inconsistencies.append(
#                     f"Current action '{curr_action}' has low coherence with recent memories"
#                 )
        
#         if total_checks > 0:
#             score = checks_failed / total_checks
        
#         self.scores['context_action'] = score
#         return score
    
#     def calculate_action_plan_hallucination(self) -> float:
#         """
#         Measures consistency between planned activities and actual/current actions.
        
#         Checks:
#         - Planned activities align with current action
#         - Time-based plans are being followed
#         - Action sequence follows plan order
        
#         OUTPUT:
#             hallucination_score (0.0-1.0): Higher = more hallucination detected
#         """
#         score = 0.0
#         checks_failed = 0
#         total_checks = 0
        
#         # Check 1: Daily schedule existence
#         total_checks += 1
#         if (hasattr(self.persona.scratch, 'f_daily_schedule') and 
#             hasattr(self.persona.scratch, 'act_description')):
#             daily_plan = self.persona.scratch.f_daily_schedule
#             curr_action = self.persona.scratch.act_description
            
#             if daily_plan is None or len(daily_plan) == 0:
#                 checks_failed += 1
#                 self.inconsistencies.append("No daily plan exists for persona")
#             else:
#                 # Check if current action is in the plan
#                 plan_str = str(daily_plan).lower()
#                 action_str = str(curr_action).lower()
#                 if len(action_str) > 3:
#                     action_keywords = action_str.split()[:3]
#                     plan_alignment = sum(1 for kw in action_keywords if kw in plan_str)
#                     if plan_alignment == 0:
#                         checks_failed += 0.5
#                         self.inconsistencies.append(
#                             f"Current action not aligned with daily plan"
#                         )
        
#         # Check 2: Hourly schedule coherence
#         total_checks += 1
#         if hasattr(self.persona.scratch, 'f_daily_schedule_hourly_org'):
#             hourly_plan = self.persona.scratch.f_daily_schedule_hourly_org
#             if hourly_plan is None or len(hourly_plan) == 0:
#                 checks_failed += 0.5
#                 self.inconsistencies.append("No hourly schedule exists")
        
#         if total_checks > 0:
#             score = checks_failed / total_checks
        
#         self.scores['action_plan'] = score
#         return score
    
#     def calculate_plan_persona_hallucination(self) -> float:
#         """
#         Measures consistency between plans and persona attributes (lifestyle, values, traits).
        
#         Checks:
#         - Plans align with lifestyle description
#         - Plan activities match persona's known traits
#         - Schedule respects persona's constraints
        
#         OUTPUT:
#             hallucination_score (0.0-1.0): Higher = more hallucination detected
#         """
#         score = 0.0
#         checks_failed = 0
#         total_checks = 0
        
#         # Check 1: Plan-Lifestyle alignment
#         total_checks += 1
#         if (hasattr(self.persona.scratch, 'lifestyle') and 
#             hasattr(self.persona.scratch, 'f_daily_schedule')):
#             lifestyle = str(self.persona.scratch.lifestyle).lower()
#             plan = str(self.persona.scratch.f_daily_schedule).lower()
            
#             # Extract key lifestyle words
#             lifestyle_words = [w for w in lifestyle.split() if len(w) > 3][:5]
#             if lifestyle_words:
#                 lifestyle_in_plan = sum(1 for w in lifestyle_words if w in plan)
#                 alignment_ratio = lifestyle_in_plan / len(lifestyle_words)
#                 if alignment_ratio < 0.3:  # Less than 30% alignment
#                     checks_failed += 1
#                     self.inconsistencies.append(
#                         f"Plan poorly aligned with lifestyle: {alignment_ratio:.1%} match"
#                     )
        
#         # Check 2: Persona traits consistency
#         total_checks += 1
#         if (hasattr(self.persona.scratch, 'first_name') and
#             hasattr(self.persona.scratch, 'f_daily_schedule')):
#             # Check that persona's identity is consistent across attributes
#             first_name = self.persona.scratch.first_name
#             if first_name and len(first_name) > 0:
#                 # This is a basic check - consistency is maintained if name exists
#                 checks_failed += 0  # No failure for this check
#             else:
#                 checks_failed += 0.5
#                 self.inconsistencies.append("Persona missing first_name attribute")
        
#         if total_checks > 0:
#             score = checks_failed / total_checks
        
#         self.scores['plan_persona'] = score
#         return score
    
#     def calculate_overall_hallucination(self) -> float:
#         """
#         Calculates weighted average of all hallucination dimensions.
        
#         Weights (equal for now, can be adjusted):
#         - Persona-Context: 25%
#         - Context-Action: 25%
#         - Action-Plan: 25%
#         - Plan-Persona: 25%
        
#         OUTPUT:
#             overall_hallucination_score (0.0-1.0): Higher = more hallucination
#         """
#         try:
#             dimensions = [
#                 self.calculate_persona_context_hallucination(),
#                 self.calculate_context_action_hallucination(),
#                 self.calculate_action_plan_hallucination(),
#                 self.calculate_plan_persona_hallucination()
#             ]
            
#             overall_score = sum(dimensions) / len(dimensions) if dimensions else 0.0
#             self.scores['overall'] = overall_score
#             return overall_score
#         except Exception as e:
#             print(f"[Hallucination Calculator] Error calculating hallucination for {self.persona.name}")
#             print(f"   Error Type: {type(e).__name__}")
#             print(f"   Details: {str(e)}")
#             print(f"   TIP: Check that persona has necessary memory structures (a_mem, s_mem, scratch)")
#             self.scores['overall'] = 0.0
#             return 0.0
    
#     def get_report(self) -> Dict[str, Any]:
#         """
#         Returns a comprehensive hallucination report.
        
#         OUTPUT:
#             Dict containing scores, inconsistencies, and summary
#         """
#         if not self.scores:
#             self.calculate_overall_hallucination()
        
#         return {
#             'timestamp': datetime.now().isoformat(),
#             'persona_name': self.persona.name,
#             'scores': {
#                 'persona_context': self.scores.get('persona_context', 0.0),
#                 'context_action': self.scores.get('context_action', 0.0),
#                 'action_plan': self.scores.get('action_plan', 0.0),
#                 'plan_persona': self.scores.get('plan_persona', 0.0),
#                 'overall': self.scores.get('overall', 0.0)
#             },
#             'inconsistencies': self.inconsistencies,
#             'summary': self._generate_summary()
#         }
    
#     def _generate_summary(self) -> str:
#         """
#         Generates a human-readable summary of hallucination findings.
        
#         OUTPUT:
#             Summary string
#         """
#         overall = self.scores.get('overall', 0.0)
        
#         if overall < 0.2:
#             status = "CONSISTENT - Agent maintains strong coherence"
#         elif overall < 0.4:
#             status = "MOSTLY CONSISTENT - Minor inconsistencies detected"
#         elif overall < 0.6:
#             status = "MODERATE HALLUCINATION - Notable inconsistencies present"
#         elif overall < 0.8:
#             status = "SIGNIFICANT HALLUCINATION - Major inconsistencies detected"
#         else:
#             status = "SEVERE HALLUCINATION - Agent highly incoherent"
        
#         return f"{status} (Score: {overall:.2f}/1.0, Issues: {len(self.inconsistencies)})"


# def analyze_persona_hallucination(persona, verbose=False) -> Dict[str, Any]:
#     """
#     Convenience function to analyze a single persona's hallucination.
    
#     INPUT:
#         persona: Persona instance
#         verbose: Whether to print detailed report
#     OUTPUT:
#         Hallucination report dict
#     """
#     calculator = HallucinationCalculator(persona)
#     report = calculator.get_report()
    
#     if verbose:
#         print(f"\n{'='*60}")
#         print(f"HALLUCINATION ANALYSIS: {persona.name}")
#         print(f"{'='*60}")
#         print(f"Overall Score: {report['scores']['overall']:.2f}/1.0")
#         print(f"Persona-Context: {report['scores']['persona_context']:.2f}")
#         print(f"Context-Action: {report['scores']['context_action']:.2f}")
#         print(f"Action-Plan: {report['scores']['action_plan']:.2f}")
#         print(f"Plan-Persona: {report['scores']['plan_persona']:.2f}")
#         print(f"\nSummary: {report['summary']}")
#         if report['inconsistencies']:
#             print(f"\nInconsistencies Found:")
#             for inc in report['inconsistencies']:
#                 print(f"  - {inc}")
#         print(f"{'='*60}\n")
    
#     return report

##------------------------------------------------------------------
#####Down code-better
##-------------------------------------------------------------------
"""
Author: Generated for Hallucination Analysis
File: hallucination_calculator.py
Description: Calculates hallucination metrics for agents' persona/context/action/plan
consistency. Uses actual scratch and associative memory structures from the
generative agents codebase.

Hallucination Categories:
1. Persona-Context: Inconsistency between persona identity fields and memory content
2. Context-Action: Mismatch between recent memories/context and current action
3. Action-Plan: Deviation between current action and what the schedule says should happen now
4. Plan-Persona: Incompatibility between the daily plan and persona's identity/lifestyle
"""

# import sys
# import json
# import datetime
# from typing import Dict, List, Tuple, Any

# sys.path.append('../../')
# from global_methods import *


# class HallucinationCalculator:
#     """
#     Calculates hallucination scores for agent consistency across four dimensions:
#     persona-context, context-action, action-plan, and plan-persona.

#     Design notes:
#     - Each dimension's score is independently computed and stored once.
#     - `inconsistencies` is only populated during `calculate_overall_hallucination()`
#       or when calling sub-methods individually for the first time (guarded by a
#       `_calculated` flag to prevent duplication on repeated calls).
#     - Scores are float in [0.0, 1.0]; higher = more hallucination.
#     - checks_failed is always a float; each sub-check contributes a weight in (0, 1]
#       so the final score is a meaningful proportion.
#     """

#     def __init__(self, persona):
#         """
#         INPUT:
#             persona: The Persona class instance (must have .scratch, .a_mem, .s_mem)
#         """
#         self.persona = persona
#         self.scores: Dict[str, float] = {}
#         self.inconsistencies: List[str] = []
#         self._calculated = set()   # tracks which dimensions have been run

#     # ------------------------------------------------------------------ #
#     #  HELPERS                                                             #
#     # ------------------------------------------------------------------ #

#     def _keywords_from_text(self, text: str, n: int = 5) -> List[str]:
#         """
#         Returns up to n meaningful (len > 3) lowercase words from text.
#         Filters common stop-words so keyword matching is more signal-rich.
#         """
#         STOPWORDS = {
#             "that", "this", "with", "from", "they", "have", "been",
#             "will", "their", "there", "were", "what", "when", "which",
#             "about", "would", "could", "should", "some", "also"
#         }
#         words = [w.strip(".,;:!?\"'()") for w in str(text).lower().split()]
#         filtered = [w for w in words if len(w) > 3 and w not in STOPWORDS]
#         return filtered[:n]

#     def _overlap_ratio(self, keywords: List[str], text: str) -> float:
#         """
#         Returns fraction of keywords found in text (0.0 – 1.0).
#         """
#         if not keywords:
#             return 1.0   # nothing to check → no failure
#         text_lower = text.lower()
#         hits = sum(1 for kw in keywords if kw in text_lower)
#         return hits / len(keywords)

#     def _get_planned_task_at_current_time(self) -> Tuple[str, int]:
#         """
#         Uses scratch's own index helper to find what task the persona SHOULD be
#         doing right now according to f_daily_schedule.

#         Returns (task_description, duration_minutes) or ("", 0) if unavailable.
#         """
#         scratch = self.persona.scratch
#         if not scratch.f_daily_schedule or not scratch.curr_time:
#             return ("", 0)
#         try:
#             idx = scratch.get_f_daily_schedule_index()
#             if 0 <= idx < len(scratch.f_daily_schedule):
#                 return tuple(scratch.f_daily_schedule[idx])
#         except Exception:
#             pass
#         return ("", 0)

#     def _get_hourly_planned_task_at_current_time(self) -> Tuple[str, int]:
#         """
#         Same as above but for the hourly_org variant (the non-decomposed schedule).
#         """
#         scratch = self.persona.scratch
#         if not scratch.f_daily_schedule_hourly_org or not scratch.curr_time:
#             return ("", 0)
#         try:
#             idx = scratch.get_f_daily_schedule_hourly_org_index()
#             if 0 <= idx < len(scratch.f_daily_schedule_hourly_org):
#                 return tuple(scratch.f_daily_schedule_hourly_org[idx])
#         except Exception:
#             pass
#         return ("", 0)

#     def _add_issue(self, msg: str):
#         """Append an inconsistency message (avoids duplicates)."""
#         if msg not in self.inconsistencies:
#             self.inconsistencies.append(msg)

#     # ------------------------------------------------------------------ #
#     #  DIMENSION 1 — PERSONA ↔ CONTEXT                                    #
#     # ------------------------------------------------------------------ #

#     def calculate_persona_context_hallucination(self) -> float:
#         """
#         Checks that the persona's stated identity (name, age, innate traits,
#         learned traits, currently, lifestyle) is reflected in their associative
#         memory (events + thoughts).

#         Sub-checks (each weighted equally at 1.0 / total_checks):
#         A) Persona appears as subject in at least one stored memory event.
#         B) Innate traits have some footprint in memory descriptions/keywords.
#         C) 'currently' field keywords appear in recent events or thoughts.
#         D) 'learned' field keywords appear in memory.

#         OUTPUT:
#             score in [0.0, 1.0]
#         """
#         if 'persona_context' in self._calculated:
#             return self.scores.get('persona_context', 0.0)

#         scratch = self.persona.scratch
#         a_mem = getattr(self.persona, 'a_mem', None)

#         checks_failed = 0.0
#         total_checks = 0

#         # All memory descriptions concatenated for broad searching
#         all_event_desc = " ".join(
#             n.description for n in (a_mem.seq_event if a_mem else [])
#         ).lower()
#         all_thought_desc = " ".join(
#             n.description for n in (a_mem.seq_thought if a_mem else [])
#         ).lower()
#         all_mem_text = all_event_desc + " " + all_thought_desc

#         # --- Check A: name appears as subject in memory events ----------
#         total_checks += 1
#         if a_mem and a_mem.seq_event:
#             persona_name = scratch.name or ""
#             subject_hits = [
#                 n for n in a_mem.seq_event
#                 if str(n.subject).lower() == persona_name.lower()
#             ]
#             if not subject_hits:
#                 checks_failed += 1.0
#                 self._add_issue(
#                     f"Persona '{persona_name}' never appears as subject "
#                     f"in {len(a_mem.seq_event)} memory events."
#                 )

#         # --- Check B: innate traits in memory ----------------------------
#         total_checks += 1
#         if scratch.innate:
#             innate_kws = self._keywords_from_text(scratch.innate, n=6)
#             ratio = self._overlap_ratio(innate_kws, all_mem_text)
#             if ratio < 0.25:
#                 checks_failed += 1.0
#                 self._add_issue(
#                     f"Innate traits '{scratch.innate}' have very low overlap "
#                     f"({ratio:.0%}) with memory content."
#                 )

#         # --- Check C: 'currently' field reflected in recent events -------
#         total_checks += 1
#         if scratch.currently and a_mem and a_mem.seq_event:
#             curr_kws = self._keywords_from_text(scratch.currently, n=5)
#             recent_text = " ".join(
#                 n.description for n in a_mem.seq_event[:20]
#             ).lower()
#             ratio = self._overlap_ratio(curr_kws, recent_text)
#             if ratio < 0.2:
#                 checks_failed += 1.0
#                 self._add_issue(
#                     f"'currently' field ('{scratch.currently}') has low overlap "
#                     f"({ratio:.0%}) with the 20 most recent events."
#                 )

#         # --- Check D: 'learned' traits in memory -------------------------
#         total_checks += 1
#         if scratch.learned:
#             learned_kws = self._keywords_from_text(scratch.learned, n=5)
#             ratio = self._overlap_ratio(learned_kws, all_mem_text)
#             if ratio < 0.2:
#                 checks_failed += 1.0
#                 self._add_issue(
#                     f"Learned traits '{scratch.learned}' have very low overlap "
#                     f"({ratio:.0%}) with memory content."
#                 )

#         score = (checks_failed / total_checks) if total_checks > 0 else 0.0
#         self.scores['persona_context'] = score
#         self._calculated.add('persona_context')
#         return score

#     # ------------------------------------------------------------------ #
#     #  DIMENSION 2 — CONTEXT ↔ ACTION                                     #
#     # ------------------------------------------------------------------ #

#     def calculate_context_action_hallucination(self) -> float:
#         """
#         Checks that the current action is consistent with recent memory context.

#         Sub-checks:
#         A) act_address sector/arena matches the persona's known living_area or
#            recent event addresses.
#         B) act_description keywords appear somewhere in the last N memory events.
#         C) act_event triple subject matches persona name (identity coherence).
#         D) If persona is marked as chatting, a recent chat memory exists.

#         OUTPUT:
#             score in [0.0, 1.0]
#         """
#         if 'context_action' in self._calculated:
#             return self.scores.get('context_action', 0.0)

#         scratch = self.persona.scratch
#         a_mem = getattr(self.persona, 'a_mem', None)

#         checks_failed = 0.0
#         total_checks = 0

#         # --- Check A: act_address plausibility --------------------------
#         # act_address format: "world:sector:arena:object"
#         # We check that the sector/arena substring appears in either
#         # living_area OR the last 30 event descriptions (the persona was
#         # recently there).
#         total_checks += 1
#         if scratch.act_address:
#             addr_parts = scratch.act_address.split(":")
#             # Use sector (index 1) as the meaningful location unit
#             sector = addr_parts[1].lower() if len(addr_parts) > 1 else ""
#             arena  = addr_parts[2].lower() if len(addr_parts) > 2 else ""

#             living_area_str = str(scratch.living_area or "").lower()
#             recent_event_desc = " ".join(
#                 n.description for n in (a_mem.seq_event[:30] if a_mem else [])
#             ).lower()
#             combined_context = living_area_str + " " + recent_event_desc

#             if sector and sector not in combined_context:
#                 checks_failed += 1.0
#                 self._add_issue(
#                     f"Current action location sector '{sector}' (from "
#                     f"act_address '{scratch.act_address}') not found in "
#                     f"living_area or recent memory."
#                 )
#             elif arena and arena not in combined_context:
#                 checks_failed += 0.5
#                 self._add_issue(
#                     f"Current action arena '{arena}' not found in recent context."
#                 )

#         # --- Check B: act_description coherence with recent memories -----
#         total_checks += 1
#         if scratch.act_description and a_mem and a_mem.seq_event:
#             action_kws = self._keywords_from_text(scratch.act_description, n=4)
#             recent_text = " ".join(
#                 n.description for n in a_mem.seq_event[:15]
#             ).lower()
#             ratio = self._overlap_ratio(action_kws, recent_text)
#             if ratio < 0.25:
#                 checks_failed += 1.0
#                 self._add_issue(
#                     f"Current action '{scratch.act_description}' has low "
#                     f"coherence ({ratio:.0%}) with the 15 most recent memories."
#                 )

#         # --- Check C: act_event subject must be the persona itself -------
#         total_checks += 1
#         if scratch.act_event and scratch.act_event[0]:
#             event_subject = str(scratch.act_event[0]).lower()
#             persona_name  = str(scratch.name or "").lower()
#             if event_subject != persona_name:
#                 checks_failed += 1.0
#                 self._add_issue(
#                     f"act_event subject '{scratch.act_event[0]}' does not match "
#                     f"persona name '{scratch.name}'."
#                 )

#         # --- Check D: chatting state consistency -------------------------
#         total_checks += 1
#         if scratch.chatting_with and a_mem:
#             # There should be at least one chat memory with this person
#             chat_partner = scratch.chatting_with.lower()
#             last_chat = a_mem.get_last_chat(chat_partner)
#             if not last_chat:
#                 checks_failed += 1.0
#                 self._add_issue(
#                     f"Persona is marked as chatting with '{scratch.chatting_with}' "
#                     f"but no chat memory exists for them."
#                 )

#         score = (checks_failed / total_checks) if total_checks > 0 else 0.0
#         self.scores['context_action'] = score
#         self._calculated.add('context_action')
#         return score

#     # ------------------------------------------------------------------ #
#     #  DIMENSION 3 — ACTION ↔ PLAN                                        #
#     # ------------------------------------------------------------------ #

#     def calculate_action_plan_hallucination(self) -> float:
#         """
#         Checks that the current action matches what the schedule planned for
#         this exact time slot.

#         Sub-checks:
#         A) A daily plan exists and is non-empty.
#         B) The current act_description aligns with what f_daily_schedule says
#            should be happening at curr_time (using the scratch index helper).
#         C) The hourly plan also exists and the act_description aligns with the
#            current hourly slot.
#         D) act_duration is within a reasonable range of the scheduled duration
#            for this slot (not wildly over/under).

#         OUTPUT:
#             score in [0.0, 1.0]
#         """
#         if 'action_plan' in self._calculated:
#             return self.scores.get('action_plan', 0.0)

#         scratch = self.persona.scratch

#         checks_failed = 0.0
#         total_checks = 0

#         # --- Check A: plan existence ------------------------------------
#         total_checks += 1
#         if not scratch.f_daily_schedule:
#             checks_failed += 1.0
#             self._add_issue("f_daily_schedule is empty — no daily plan exists.")
#             # Can't do further plan checks; short-circuit.
#             self.scores['action_plan'] = 1.0
#             self._calculated.add('action_plan')
#             return 1.0

#         # --- Check B: decomposed schedule vs current action --------------
#         total_checks += 1
#         planned_task, planned_dur = self._get_planned_task_at_current_time()
#         if planned_task and scratch.act_description:
#             planned_kws = self._keywords_from_text(planned_task, n=5)
#             ratio = self._overlap_ratio(planned_kws, scratch.act_description)
#             if ratio < 0.2:
#                 checks_failed += 1.0
#                 self._add_issue(
#                     f"At {scratch.curr_time.strftime('%H:%M') if scratch.curr_time else '??:??'}, "
#                     f"plan says '{planned_task}' but current action is "
#                     f"'{scratch.act_description}' (overlap {ratio:.0%})."
#                 )
#         elif not planned_task:
#             checks_failed += 0.5
#             self._add_issue(
#                 "Could not determine planned task for the current time slot."
#             )

#         # --- Check C: hourly_org schedule alignment ----------------------
#         total_checks += 1
#         if scratch.f_daily_schedule_hourly_org:
#             hourly_task, hourly_dur = self._get_hourly_planned_task_at_current_time()
#             if hourly_task and scratch.act_description:
#                 hourly_kws = self._keywords_from_text(hourly_task, n=5)
#                 ratio = self._overlap_ratio(hourly_kws, scratch.act_description)
#                 if ratio < 0.2:
#                     checks_failed += 1.0
#                     self._add_issue(
#                         f"Hourly plan says '{hourly_task}' but current action is "
#                         f"'{scratch.act_description}' (overlap {ratio:.0%})."
#                     )
#         else:
#             checks_failed += 1.0
#             self._add_issue("f_daily_schedule_hourly_org is empty.")

#         # --- Check D: duration sanity check ------------------------------
#         total_checks += 1
#         if (scratch.act_duration is not None and
#                 planned_dur > 0 and
#                 scratch.act_description):
#             # Flag if actual duration is more than 3× or less than 1/3 of plan
#             ratio = scratch.act_duration / planned_dur
#             if ratio > 3.0 or ratio < 0.33:
#                 checks_failed += 1.0
#                 self._add_issue(
#                     f"act_duration ({scratch.act_duration} min) is far from "
#                     f"planned duration ({planned_dur} min) for '{planned_task}'."
#                 )

#         score = (checks_failed / total_checks) if total_checks > 0 else 0.0
#         self.scores['action_plan'] = score
#         self._calculated.add('action_plan')
#         return score

#     # ------------------------------------------------------------------ #
#     #  DIMENSION 4 — PLAN ↔ PERSONA                                       #
#     # ------------------------------------------------------------------ #

#     def calculate_plan_persona_hallucination(self) -> float:
#         """
#         Checks that the planned activities are compatible with the persona's
#         identity (lifestyle, innate traits, occupation, daily_plan_req).

#         Sub-checks:
#         A) Lifestyle keywords appear in the daily plan.
#         B) daily_plan_req keywords appear in the plan (the persona should be
#            doing what they set out to do today).
#         C) 'currently' keywords (occupation / ongoing projects) appear in plan.
#         D) The total plan duration sums to ~1440 minutes (a full day).

#         OUTPUT:
#             score in [0.0, 1.0]
#         """
#         if 'plan_persona' in self._calculated:
#             return self.scores.get('plan_persona', 0.0)

#         scratch = self.persona.scratch

#         checks_failed = 0.0
#         total_checks = 0

#         # Flatten the plan into a single searchable string
#         plan_text = " ".join(
#             task for task, _ in (scratch.f_daily_schedule or [])
#         ).lower()
#         hourly_text = " ".join(
#             task for task, _ in (scratch.f_daily_schedule_hourly_org or [])
#         ).lower()
#         combined_plan_text = plan_text + " " + hourly_text

#         if not combined_plan_text.strip():
#             self.scores['plan_persona'] = 1.0
#             self._calculated.add('plan_persona')
#             self._add_issue("Both schedule lists are empty — plan-persona check skipped.")
#             return 1.0

#         # --- Check A: lifestyle → plan alignment -------------------------
#         total_checks += 1
#         if scratch.lifestyle:
#             lifestyle_kws = self._keywords_from_text(scratch.lifestyle, n=6)
#             ratio = self._overlap_ratio(lifestyle_kws, combined_plan_text)
#             if ratio < 0.3:
#                 checks_failed += 1.0
#                 self._add_issue(
#                     f"Lifestyle '{scratch.lifestyle}' has low alignment "
#                     f"({ratio:.0%}) with the daily plan."
#                 )

#         # --- Check B: daily_plan_req → plan alignment --------------------
#         total_checks += 1
#         if scratch.daily_plan_req:
#             req_kws = self._keywords_from_text(scratch.daily_plan_req, n=6)
#             ratio = self._overlap_ratio(req_kws, combined_plan_text)
#             if ratio < 0.25:
#                 checks_failed += 1.0
#                 self._add_issue(
#                     f"daily_plan_req '{scratch.daily_plan_req}' has low alignment "
#                     f"({ratio:.0%}) with the daily plan."
#                 )

#         # --- Check C: 'currently' → plan alignment -----------------------
#         total_checks += 1
#         if scratch.currently:
#             curr_kws = self._keywords_from_text(scratch.currently, n=5)
#             ratio = self._overlap_ratio(curr_kws, combined_plan_text)
#             if ratio < 0.2:
#                 checks_failed += 1.0
#                 self._add_issue(
#                     f"'currently' field '{scratch.currently}' has low alignment "
#                     f"({ratio:.0%}) with the daily plan."
#                 )

#         # --- Check D: plan covers a full day (total ≈ 1440 min) ----------
#         total_checks += 1
#         if scratch.f_daily_schedule_hourly_org:
#             total_min = sum(dur for _, dur in scratch.f_daily_schedule_hourly_org)
#             # Allow a 15-minute tolerance either side of 1440
#             if not (1425 <= total_min <= 1455):
#                 checks_failed += 1.0
#                 self._add_issue(
#                     f"Hourly plan total duration is {total_min} min "
#                     f"(expected ~1440 for a full day)."
#                 )

#         score = (checks_failed / total_checks) if total_checks > 0 else 0.0
#         self.scores['plan_persona'] = score
#         self._calculated.add('plan_persona')
#         return score

#     # ------------------------------------------------------------------ #
#     #  OVERALL                                                             #
#     # ------------------------------------------------------------------ #

#     def calculate_overall_hallucination(self) -> float:
#         """
#         Calculates weighted average of all four dimensions.

#         Weights:
#             Persona-Context : 25%
#             Context-Action  : 25%
#             Action-Plan     : 25%
#             Plan-Persona    : 25%

#         OUTPUT:
#             overall_hallucination_score in [0.0, 1.0]
#         """
#         try:
#             d1 = self.calculate_persona_context_hallucination()
#             d2 = self.calculate_context_action_hallucination()
#             d3 = self.calculate_action_plan_hallucination()
#             d4 = self.calculate_plan_persona_hallucination()

#             overall = (d1 + d2 + d3 + d4) / 4.0
#             self.scores['overall'] = overall
#             return overall

#         except Exception as e:
#             name = getattr(self.persona, 'name', 'UNKNOWN')
#             print(f"[HallucinationCalculator] Error for '{name}': "
#                   f"{type(e).__name__}: {e}")
#             print("  TIP: Ensure persona has .scratch, .a_mem, .s_mem populated.")
#             self.scores['overall'] = 0.0
#             return 0.0

#     # ------------------------------------------------------------------ #
#     #  REPORTING                                                           #
#     # ------------------------------------------------------------------ #

#     def get_report(self) -> Dict[str, Any]:
#         """
#         Returns a comprehensive hallucination report dict.
#         Triggers calculation if not yet done.
#         """
#         if 'overall' not in self.scores:
#             self.calculate_overall_hallucination()

#         return {
#             'timestamp': datetime.datetime.now().isoformat(),
#             'persona_name': self.persona.name,
#             'scores': {
#                 'persona_context': round(self.scores.get('persona_context', 0.0), 4),
#                 'context_action':  round(self.scores.get('context_action',  0.0), 4),
#                 'action_plan':     round(self.scores.get('action_plan',     0.0), 4),
#                 'plan_persona':    round(self.scores.get('plan_persona',    0.0), 4),
#                 'overall':         round(self.scores.get('overall',         0.0), 4),
#             },
#             'inconsistencies': self.inconsistencies,
#             'summary': self._generate_summary()
#         }

#     def _generate_summary(self) -> str:
#         overall = self.scores.get('overall', 0.0)
#         if overall < 0.2:
#             status = "CONSISTENT — agent maintains strong coherence"
#         elif overall < 0.4:
#             status = "MOSTLY CONSISTENT — minor inconsistencies detected"
#         elif overall < 0.6:
#             status = "MODERATE HALLUCINATION — notable inconsistencies present"
#         elif overall < 0.8:
#             status = "SIGNIFICANT HALLUCINATION — major inconsistencies detected"
#         else:
#             status = "SEVERE HALLUCINATION — agent highly incoherent"
#         return (f"{status} "
#                 f"(Score: {overall:.2f}/1.0, Issues: {len(self.inconsistencies)})")


# # ------------------------------------------------------------------ #
# #  CONVENIENCE FUNCTION                                               #
# # ------------------------------------------------------------------ #

# def analyze_persona_hallucination(persona, verbose: bool = False) -> Dict[str, Any]:
#     """
#     Convenience wrapper to analyze a single persona's hallucination.

#     INPUT:
#         persona: Persona instance
#         verbose: Whether to print a detailed report to stdout
#     OUTPUT:
#         Hallucination report dict
#     """
#     calculator = HallucinationCalculator(persona)
#     report = calculator.get_report()

#     if verbose:
#         W = 60
#         print(f"\n{'=' * W}")
#         print(f"HALLUCINATION ANALYSIS: {persona.name}")
#         print(f"{'=' * W}")
#         print(f"  Overall Score   : {report['scores']['overall']:.4f}")
#         print(f"  Persona-Context : {report['scores']['persona_context']:.4f}")
#         print(f"  Context-Action  : {report['scores']['context_action']:.4f}")
#         print(f"  Action-Plan     : {report['scores']['action_plan']:.4f}")
#         print(f"  Plan-Persona    : {report['scores']['plan_persona']:.4f}")
#         print(f"\n  Summary: {report['summary']}")
#         if report['inconsistencies']:
#             print(f"\n  Inconsistencies found ({len(report['inconsistencies'])}):")
#             for inc in report['inconsistencies']:
#                 print(f"    • {inc}")
#         print(f"{'=' * W}\n")

#     return report