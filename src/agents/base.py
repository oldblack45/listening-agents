"""BaseAgent abstract class. All agents return {message, orders}."""
from __future__ import annotations
import json
import re
from abc import ABC, abstractmethod
from typing import Any

from .. import config as C
from .. import llm_client


def _format_observation(obs):
    legal = obs.get("legal_orders", {})
    legal_lines = []
    for loc, opts in legal.items():
        # v2: show up to 20 options (was 6) so the rich Sotopia vocab is visible
        sample = "; ".join(opts[:20])
        legal_lines.append(f"  {loc}: {sample}{'...' if len(opts) > 20 else ''}")
    recent = obs.get("recent_messages", [])[-5:]
    msg_lines = [f"  [{m['sender']}->{m['recipient']}] {m['content']}" for m in recent]
    context = obs.get("context", "")
    context_line = f"Scenario: {context}\n" if context else ""
    return (
        context_line +
        f"Phase: {obs['phase']} (turn {obs['turn']})\n"
        f"You play: {obs['power']}\n"
        f"Your units: {', '.join(obs['units']) or '(none)'}\n"
        f"Your supply centers: {', '.join(obs['centers']) or '(none)'}\n"
        f"Recent messages addressed to you:\n" + ("\n".join(msg_lines) or "  (none)") + "\n"
        f"Legal orders per unit location:\n" + ("\n".join(legal_lines) or "  (no units)") + "\n"
    )


def _parse_response(text, legal_per_loc):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    parsed = {}
    if match:
        try:
            parsed = json.loads(match.group(0))
        except Exception:
            parsed = {}
    raw_orders = parsed.get("orders", []) or []
    if isinstance(raw_orders, str):
        raw_orders = [raw_orders]
    valid_orders = []
    used_locs = set()
    # Build a reverse lookup: option string -> location, for envs whose
    # order labels are NOT in the Diplomacy "UNIT LOC VERB ..." form (e.g.
    # SotopiaEnv where each label is a single concrete commitment string).
    opt_to_loc = {}
    for loc, opts in legal_per_loc.items():
        for opt in opts:
            opt_to_loc.setdefault(opt, loc)
    for o in raw_orders:
        if not isinstance(o, str):
            continue
        o = o.strip()
        # Diplomacy path: 'F LON H' -> loc = parts[1]
        parts = o.split()
        if len(parts) >= 2:
            loc = parts[1]
            if loc in legal_per_loc and o in legal_per_loc[loc]:
                valid_orders.append(o)
                used_locs.add(loc)
                continue
        # Sotopia path: single-word commitment matches a legal option directly
        if o in opt_to_loc:
            loc = opt_to_loc[o]
            if loc not in used_locs:
                valid_orders.append(o)
                used_locs.add(loc)
    for loc, opts in legal_per_loc.items():
        if loc not in used_locs:
            hold = next((o for o in opts if o.endswith(" H")), opts[0] if opts else None)
            if hold:
                valid_orders.append(hold)
    raw_msgs = parsed.get("messages", []) or []
    if not isinstance(raw_msgs, list):
        raw_msgs = []
    messages = {}
    for m in raw_msgs:
        if not isinstance(m, dict):
            continue
        to = m.get("to") or m.get("recipient") or ""
        content = m.get("content") or m.get("message") or m.get("text") or ""
        if isinstance(to, str) and isinstance(content, str) and to and content:
            messages[to.upper()] = content.strip()[:500]
    return {"messages": messages, "orders": valid_orders, "_raw": text}


class BaseAgent(ABC):
    name = "base"

    def __init__(self, power, model=C.AGENT_MODEL, temperature=C.TEMPERATURE, incentive="C"):
        self.power = power
        self.model = model
        self.temperature = temperature
        self.incentive = incentive  # "C"=cooperative, "A"=adversarial

    @abstractmethod
    def build_prompt(self, observation):
        ...

    def act(self, observation, seed=None, temperature=None, use_cache=True):
        prompt = self.build_prompt(observation)
        t = temperature if temperature is not None else self.temperature
        r = llm_client.chat(
            model=self.model, messages=prompt,
            temperature=t, max_tokens=C.MAX_OUTPUT_TOKENS_AGENT,
            seed=seed, use_cache=use_cache, tag=f"{self.name}:{self.power}:t{t}",
        )
        return _parse_response(r.text, observation.get("legal_orders", {}))
