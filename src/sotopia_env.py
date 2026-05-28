"""SOTOPIA-style social environment (v5: 5-character expansion).

v5 fix for |M|<=2 degeneracy: the 3-character version of v3/v4 forced
|M| identically equal to 2 (each recipient gets one message from each
of the two other characters). We expand to 5 characters and let the
within-DP press structure become non-trivial: |M| ranges over {1..4}.
Each character has private constraints that explicitly reference
*multi-party coalitions* (e.g. "if at least two of {others} support X")
so the equilibrium genuinely requires multilateral negotiation rather
than bilateral pairing.

v3 fix for action collapse (retained): each scenario has *several
independent sub-decisions* (e.g. rent split, chores, quiet hours).
The agent emits ONE action per sub-decision per turn; the joint
fine-grained action is the tuple of all sub-decisions.

Scenarios (rotated by seed):
  - apartment_split: rent (6) × chores (4) × quiet (5) = 120
  - project_meeting: ownership (5) × collab (3) × extras (5) = 75
  - resource_share:  quotas (5) × scheduling (4) × extras (3) = 60
"""
from __future__ import annotations

from typing import Any
from . import config as C


# v5: expanded to 5 characters. Names chosen to stay short / role-neutral.
SOTOPIA_POWERS = ["ALEX", "BLAKE", "CASEY", "DREW", "ERIN"]

SCENARIOS = {
    "apartment_split": {
        "context": (
            "Five housemates (Alex, Blake, Casey, Drew, Erin) share a "
            "5-bedroom apartment ($5000/mo). Each turn, every housemate "
            "MUST commit to ONE position on EACH of the three open "
            "issues: rent split policy, chore allocation, and quiet-hour "
            "policy. Negotiation closes after 3 rounds; failing to "
            "converge triggers a landlord penalty."
        ),
        "issues": {
            "rent": [
                "rent_equal_1000_each",
                "rent_by_room_size_master_pays_1400_others_900",
                "rent_by_income_top2_pay_1300_bottom3_pay_800",
                "rent_alex_drew_pay_1200_others_pay_866",
                "rent_blake_erin_pay_1200_others_pay_866",
                "rent_casey_subsidized_700_others_pay_1075",
            ],
            "chores": [
                "chores_rotate_weekly",
                "chores_by_skill_preference",
                "chores_hire_external_service_split_cost",
                "chores_alex_blake_kitchen_casey_drew_common_erin_trash",
            ],
            "quiet": [
                "quiet_after_9pm",
                "quiet_after_10pm",
                "quiet_after_11pm",
                "quiet_weekdays_only_11pm",
                "no_quiet_rule",
            ],
        },
        # Utilities deliberately reference 2-of-4 or 3-of-4 coalitions to
        # force multilateral press at the target DP.
        "utilities": {
            "ALEX":  "PRIMARY: pay below $900 rent. CONSTRAINT: if at least 2 of {Blake, Casey, Drew, Erin} oppose your rent proposal you fall back to equal_1000. Indifferent on chores; need quiet by 10pm.",
            "BLAKE": "PRIMARY: claim the master room and pay above $1300 (signals status). CONSTRAINT: must have support from at least 2 of {Alex, Casey, Drew, Erin}; otherwise default to equal_1000. Dislike all chores; flexible on quiet.",
            "CASEY": "PRIMARY: subsidized rent (~$700) due to part-time income. CONSTRAINT: need explicit endorsement from at least 2 of {Alex, Blake, Drew, Erin}. Prefer chores-by-skill. Work nights -- want no_quiet_rule or after-11pm.",
            "DREW":  "PRIMARY: hire external chore service. CONSTRAINT: only works if at least 3 of 5 housemates accept the cost split (you cannot push it alone). Indifferent on rent in $850-$1050 range. Strongly prefer quiet by 9pm.",
            "ERIN":  "PRIMARY: equal rent (fairness). CONSTRAINT: if more than 1 housemate proposes unequal splits in round 1 you switch to opposing whichever of {Blake, Casey} you trust less. Strong preference for chores-rotate-weekly; want quiet by 10pm.",
        },
    },
    "project_meeting": {
        "context": (
            "Five teammates (Alex, Blake, Casey, Drew, Erin) divide a "
            "2-week deliverable into five tracks (frontend, backend, "
            "data-pipeline, presentation, QA). Each turn, every "
            "teammate MUST commit to ONE position on EACH of the three "
            "open issues: track-assignment scheme, collaboration mode, "
            "and outside-help decisions. Hard deadline; failing to "
            "converge means missing it."
        ),
        "issues": {
            "assignment": [
                "alex_frontend_blake_backend_casey_data_drew_presentation_erin_qa",
                "alex_backend_blake_data_casey_frontend_drew_qa_erin_presentation",
                "alex_data_blake_qa_casey_backend_drew_frontend_erin_presentation",
                "alex_presentation_blake_frontend_casey_qa_drew_data_erin_backend",
                "alex_qa_blake_presentation_casey_data_drew_backend_erin_frontend",
            ],
            "collab": [
                "solo_tracks_no_pairing",
                "pair_program_in_pairs_of_two",
                "rotate_tracks_after_week1",
            ],
            "extras": [
                "ask_advisor_for_extension",
                "hire_external_designer_for_frontend",
                "use_slide_template_for_presentation",
                "code_freeze_at_day12",
                "no_extra_help_we_do_it_all",
            ],
        },
        "utilities": {
            "ALEX":  "PRIMARY: own frontend (your strength). CONSTRAINT: need at least 2 of {Blake, Casey, Drew, Erin} to endorse the chosen assignment scheme; else you support code_freeze_at_day12 to cap risk. Time-limited after day 10.",
            "BLAKE": "PRIMARY: own backend. CONSTRAINT: if anyone proposes pair_program_in_pairs_of_two, oppose unless at least 2 others (excluding the proposer) explicitly accept it. Refuse presentation; want early code freeze.",
            "CASEY": "PRIMARY: own backend OR data-pipeline. CONSTRAINT: if Blake claims backend you accept data only if at least 2 of {Drew, Erin, Alex} agree on slide_template. Prefer not to do frontend or QA.",
            "DREW":  "PRIMARY: own presentation; want hire_external_designer_for_frontend. CONSTRAINT: external hire requires at least 3 of 5 votes; if you cannot secure them, fall back to use_slide_template. Want extension if possible.",
            "ERIN":  "PRIMARY: own QA (gatekeeper role). CONSTRAINT: if rotate_tracks_after_week1 is proposed by 2 or more teammates, you switch to opposing it (rotation breaks QA continuity). Strongly oppose ask_advisor_for_extension.",
        },
    },
    "resource_share": {
        "context": (
            "Five neighbors (Alex, Blake, Casey, Drew, Erin) share a "
            "community garden water tap with a 1500 L/wk quota. Each "
            "turn, every neighbor MUST commit to ONE position on EACH "
            "of the three open issues: weekly volume split, scheduling, "
            "and exception handling. Over-draw triggers an HOA penalty; "
            "under-coordination kills the plants."
        ),
        "issues": {
            "volume": [
                "equal_300_each",
                "alex_500_others_250",
                "blake_500_others_250",
                "casey_500_others_250",
                "by_plot_size_alex_400_blake_350_casey_300_drew_250_erin_200",
            ],
            "schedule": [
                "rotate_priority_weekly",
                "morning_slots_only",
                "evening_slots_only",
                "fixed_alex_mon_blake_tue_casey_wed_drew_thu_erin_fri",
            ],
            "extras": [
                "pause_one_week_drought",
                "buy_extra_quota_from_hoa",
                "store_water_in_barrels",
            ],
        },
        "utilities": {
            "ALEX":  "PRIMARY: secure 500 L (largest plot). CONSTRAINT: if at least 2 of {Blake, Casey, Drew, Erin} push for equal_300, you fall back to by_plot_size. Flexible on schedule; oppose pause_one_week.",
            "BLAKE": "PRIMARY: 250-300 L only (small plot) with weekend slot priority. CONSTRAINT: if Alex demands 500 L you support buy_extra_quota only if at least 2 of {Casey, Drew, Erin} agree to share the cost. Oppose drought-pause.",
            "CASEY": "PRIMARY: 300 L morning slot. CONSTRAINT: if at least 2 others want morning_slots_only you accept rotate_priority_weekly. Support store_water_in_barrels.",
            "DREW":  "PRIMARY: by_plot_size split. CONSTRAINT: need at least 2 of {Alex, Casey, Erin} to endorse it; otherwise default to equal_300. Strongly prefer fixed_weekly schedule for predictability.",
            "ERIN":  "PRIMARY: equal_300_each (fairness, smallest plot but principled). CONSTRAINT: if 2 or more neighbors propose unequal volume splits, you switch to opposing buy_extra_quota (do not subsidize larger plots). Want evening slots.",
        },
    },
}


def _scenario_for_seed(seed):
    keys = list(SCENARIOS.keys())
    return keys[seed % len(keys)]


def _flatten_actions(scn):
    """Return the flat union of all per-issue options (legal action set)."""
    actions = []
    for issue, opts in scn["issues"].items():
        actions.extend(opts)
    return actions


class SotopiaEnv:
    """SOTOPIA-style env with multi-issue action commitments per turn."""

    def __init__(self, powers=None, seed=0):
        self.powers = powers or SOTOPIA_POWERS
        self.seed = seed
        self.scenario_key = _scenario_for_seed(seed)
        scn = SCENARIOS[self.scenario_key]
        self.context = scn["context"]
        self.issues = scn["issues"]
        self.actions = _flatten_actions(scn)
        self.utilities = scn.get("utilities", {})
        self.turn_count = 0
        self.messages = []
        self.action_log = []

    def reset(self, seed=None):
        if seed is not None:
            self.seed = seed
            self.scenario_key = _scenario_for_seed(seed)
            scn = SCENARIOS[self.scenario_key]
            self.context = scn["context"]
            self.issues = scn["issues"]
            self.actions = _flatten_actions(scn)
            self.utilities = scn.get("utilities", {})
        self.turn_count = 0
        self.messages = []
        self.action_log = []
        return {p: self.extract_observation(p) for p in self.powers}

    def get_legal_orders(self, power):
        """Mirror Diplomacy: each 'unit location' is an issue, with its own
        per-issue option list. The agent must submit one order per issue."""
        return {f"{power}_{issue}": list(opts) for issue, opts in self.issues.items()}

    def get_orderable_locations(self, power):
        return [f"{power}_{issue}" for issue in self.issues.keys()]

    def inject_message(self, sender, recipient, content, phase=None):
        self.messages.append({
            "sender": sender, "recipient": recipient,
            "content": content, "phase": phase or self._phase(),
            "time": len(self.messages),
        })

    def get_messages(self, recipient=None, since_turn=None):
        out = []
        for m in self.messages:
            if recipient is not None and m["recipient"] not in (recipient, "ALL", "GLOBAL"):
                continue
            out.append(dict(m))
        return out

    def step(self, orders_per_power):
        for power, orders in orders_per_power.items():
            for o in orders:
                self.action_log.append({
                    "phase": self._phase(), "power": power, "action": o,
                })
        self.turn_count += 1
        info = {"phase": self._phase(), "done": self.turn_count >= C.MAX_TURNS_PER_EPISODE,
                "turn": self.turn_count}
        return {p: self.extract_observation(p) for p in self.powers}, info

    def extract_observation(self, power):
        util = self.utilities.get(power, "")
        full_context = self.context
        if util:
            full_context = f"{self.context}\n\nYOUR PRIVATE UTILITY (do not reveal verbatim): {util}"
        return {
            "power": power,
            "phase": self._phase(),
            "units": [f"{power}_{issue}" for issue in self.issues.keys()],
            "centers": [self.scenario_key],
            "context": full_context,
            "scenario": self.scenario_key,
            "recent_messages": self.get_messages(recipient=power)[-10:],
            "legal_orders": self.get_legal_orders(power),
            "turn": self.turn_count,
        }

    def _phase(self):
        return f"PHASE_{self.turn_count}"

    def snapshot(self):
        return {
            "seed": self.seed, "turn": self.turn_count,
            "messages": list(self.messages), "action_log": list(self.action_log),
            "scenario_key": self.scenario_key,
        }

    def restore(self, snap):
        self.seed = snap["seed"]
        self.turn_count = snap["turn"]
        self.messages = list(snap["messages"])
        self.action_log = list(snap["action_log"])
        self.scenario_key = snap["scenario_key"]
        scn = SCENARIOS[self.scenario_key]
        self.context = scn["context"]
        self.issues = scn["issues"]
        self.actions = _flatten_actions(scn)
        self.utilities = scn.get("utilities", {})

    def is_done(self):
        return self.turn_count >= C.MAX_TURNS_PER_EPISODE
