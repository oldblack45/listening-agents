"""Diplomacy environment adapter for Pilot B0.

Simplified to 3 powers (ENGLAND, FRANCE, GERMANY) + short horizon (6 phases ~ 3 years
of Spring/Fall movement, no retreats/builds processed in our pilot).

Exposes:
- reset(seed) -> initial obs dict per power
- get_legal_orders(power) -> dict[unit_loc -> list[str]] (one option per unit list)
- inject_message(sender, recipient, content, phase=None)
- step(orders_per_power) -> next obs dict per power, plus phase delta
- extract_observation(power) -> dict (units, centers, recent_messages, phase)
- snapshot() / restore(state_dict): for do-intervention rollback
- abstract action representation: orders -> canonical fine + coarse string
"""
from __future__ import annotations

import random
from typing import Any

import diplomacy

from . import config as C


# Coarse action mapping: every order maps into one of these high-level intents
# Used for FS_binary @ coarse granularity (H6).
COARSE_INTENTS = {
    "HOLD": "hold",
    "MOVE": "move",
    "SUPPORT": "support",
    "CONVOY": "convoy",
}


def order_to_coarse(order: str) -> str:
    """Map a Diplomacy order string to a coarse intent label.

    Examples:
      'F LON H'          -> 'hold'
      'F LON - NTH'      -> 'move'
      'F LON S A YOR'    -> 'support'
      'F LON C A LON-NTH'-> 'convoy'
    """
    parts = order.strip().split()
    if len(parts) < 3:
        return "hold"
    op = parts[2]
    if op == "H":
        return "hold"
    if op == "-":
        return "move"
    if op == "S":
        return "support"
    if op == "C":
        return "convoy"
    return "move"


class DiplomacyEnv:
    """Lightweight wrapper around diplomacy.Game."""

    def __init__(self, powers: list[str] | None = None, seed: int = 0):
        self.powers = powers or C.DIPLOMACY_POWERS
        self.seed = seed
        self.game: diplomacy.Game | None = None
        self.turn_count = 0

    def reset(self, seed: int | None = None) -> dict[str, dict]:
        """Reset to initial state. Returns per-power observations."""
        if seed is not None:
            self.seed = seed
        random.seed(self.seed)
        # POWER_CHOICE allows assigning specific powers; default rule set permits press
        self.game = diplomacy.Game(rules=["POWER_CHOICE"])
        self.turn_count = 0
        return {p: self.extract_observation(p) for p in self.powers}

    def get_legal_orders(self, power: str) -> dict[str, list[str]]:
        """Return per-unit-location list of legal orders."""
        assert self.game is not None
        all_possible = self.game.get_all_possible_orders()
        out: dict[str, list[str]] = {}
        for loc in self.game.get_orderable_locations(power):
            out[loc] = list(all_possible.get(loc, []))
        return out

    def inject_message(self, sender: str, recipient: str, content: str,
                       phase: str | None = None) -> None:
        assert self.game is not None
        msg = diplomacy.Message(
            sender=sender, recipient=recipient,
            message=content,
            phase=phase or self.game.get_current_phase(),
        )
        self.game.add_message(msg)

    def get_messages(self, recipient: str | None = None,
                     since_turn: int | None = None) -> list[dict]:
        assert self.game is not None
        msgs = self.game.messages
        # messages is a SortedDict of time_sent -> Message
        out = []
        for t, m in msgs.items():
            if recipient is not None and m.recipient not in (recipient, "GLOBAL"):
                continue
            out.append({
                "sender": m.sender, "recipient": m.recipient,
                "content": m.message, "phase": m.phase, "time": t,
            })
        return out

    def step(self, orders_per_power: dict[str, list[str]]) -> tuple[dict[str, dict], dict]:
        """Submit orders for all powers and advance one phase."""
        assert self.game is not None
        for power, orders in orders_per_power.items():
            self.game.set_orders(power, orders)
        # default-hold for non-active powers (others) to keep game running
        for p in self.game.powers:
            if p not in orders_per_power and not self.game.powers[p].is_eliminated():
                # default: hold everything
                pass  # leaving empty -> game treats as hold
        self.game.process()
        self.turn_count += 1
        info = {
            "phase": self.game.get_current_phase(),
            "done": self.game.is_game_done,
            "turn": self.turn_count,
        }
        return {p: self.extract_observation(p) for p in self.powers}, info

    def extract_observation(self, power: str) -> dict:
        assert self.game is not None
        p = self.game.powers[power]
        return {
            "power": power,
            "phase": self.game.get_current_phase(),
            "units": list(p.units),
            "centers": list(p.centers),
            "recent_messages": self.get_messages(recipient=power)[-10:],
            "legal_orders": self.get_legal_orders(power),
            "turn": self.turn_count,
        }

    def snapshot(self) -> dict:
        """Return a JSON-serializable snapshot for do-intervention rollback."""
        assert self.game is not None
        return {
            "state": self.game.to_dict(),
            "turn_count": self.turn_count,
        }

    def restore(self, snap: dict) -> None:
        self.game = diplomacy.Game.from_dict(snap["state"])
        self.turn_count = snap["turn_count"]

    def is_done(self) -> bool:
        return self.game is None or self.game.is_game_done or self.turn_count >= C.MAX_TURNS_PER_EPISODE


def orders_to_string(orders: list[str]) -> str:
    """Canonical fine-grained representation for KL estimation."""
    return "; ".join(sorted(orders))


def orders_to_coarse(orders: list[str]) -> str:
    """Coarse representation: sorted tuple of (unit_type, intent)."""
    items = []
    for o in orders:
        parts = o.strip().split()
        if len(parts) >= 2:
            items.append(f"{parts[0]}:{order_to_coarse(o)}")
    return ";".join(sorted(items))
