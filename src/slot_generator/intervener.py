"""6-type intervention generator.

identity            : m -> m (sanity)
synonym             : preserve (s,c,a,t); paraphrase propositional content via LLM at low temp
fact_replace        : change a key fact (locations, units) but keep speech_act + addressee
counterfactual      : negate the commitment (e.g. "will support" -> "will not support")
random_string       : equal-length random ASCII
cross_episode_swap  : replace with a random message drawn from a different episode bank
"""
from __future__ import annotations
import random
import string
import re
from typing import Iterable

from .. import config as C
from .. import llm_client


INTERVENTION_TYPES = (
    "identity", "synonym", "fact_replace",
    "counterfactual", "random_string", "cross_episode_swap",
)


def _llm_rewrite(prompt_user: str, system: str, seed: int) -> str:
    r = llm_client.chat(
        model=C.AGENT_MODEL,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": prompt_user}],
        temperature=0.3, max_tokens=C.MAX_OUTPUT_TOKENS_INTERVENE,
        seed=seed, tag="intervene-rewrite",
    )
    return r.text.strip().strip("`").strip()


def synonym(message: str, fourtuple: dict, seed: int) -> str:
    sys_p = (
        "Paraphrase a Diplomacy press message. Keep the same speech act, commitment strength, "
        "addressee, and time reference. Output the paraphrased sentence on a single line, no quotes."
    )
    user = (
        f"Speech act: {fourtuple['speech_act']}\n"
        f"Commitment: {fourtuple['commitment_strength']}\n"
        f"Addressee: {fourtuple['addressee']}\n"
        f"Time: {fourtuple['temporal_marker']}\n"
        f"Original message: {message}\n"
        f"Paraphrase:"
    )
    return _strip_first_line(_llm_rewrite(user, sys_p, seed))


def fact_replace(message: str, fourtuple: dict, seed: int) -> str:
    sys_p = (
        "Rewrite a Diplomacy press message by replacing the key fact (target location, unit, "
        "or province name) with a plausible different one. Keep the same speech act and addressee. "
        "Output a single line, no quotes."
    )
    user = f"Original: {message}\nRewrite with one fact swapped:"
    return _strip_first_line(_llm_rewrite(user, sys_p, seed))


def counterfactual(message: str, fourtuple: dict, seed: int) -> str:
    sys_p = (
        "Rewrite a Diplomacy press message by negating the commitment "
        "('will' -> 'will not', 'agree' -> 'refuse', etc.). Keep speech act and addressee. "
        "Output a single line, no quotes."
    )
    user = f"Original: {message}\nRewrite with the commitment negated:"
    return _strip_first_line(_llm_rewrite(user, sys_p, seed))


def random_string(message: str, seed: int) -> str:
    rng = random.Random(seed)
    L = max(20, len(message))
    chars = string.ascii_letters + string.digits + "   ,.;:!?"
    return "".join(rng.choice(chars) for _ in range(L))


def cross_episode_swap(pool: list[str], seed: int) -> str:
    if not pool:
        return "(swap unavailable)"
    return random.Random(seed).choice(pool)


def _strip_first_line(text: str) -> str:
    text = text.strip()
    # Take first non-empty line (LLMs sometimes add explanations)
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line[:500]
    return text[:500]


def generate_interventions(
    message: str,
    fourtuple: dict,
    cross_pool: list[str],
    seed_base: int = 0,
) -> dict[str, str]:
    """Return {intervention_type: produced_message} for all 6 types."""
    return {
        "identity": message,
        "synonym": synonym(message, fourtuple, seed_base + 1),
        "fact_replace": fact_replace(message, fourtuple, seed_base + 2),
        "counterfactual": counterfactual(message, fourtuple, seed_base + 3),
        "random_string": random_string(message, seed_base + 4),
        "cross_episode_swap": cross_episode_swap(cross_pool, seed_base + 5),
    }
