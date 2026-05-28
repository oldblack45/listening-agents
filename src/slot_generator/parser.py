"""4-tuple parser: (speech_act, commitment_strength, addressee, temporal_marker).

Pure-LLM extraction with strict JSON output. No spaCy dep (keeps pilot env minimal).
The Searle-like speech-act set is fixed to 12 categories:
  COMMIT REQUEST THREAT INFORM QUESTION PROMISE REJECT APOLOGIZE PROPOSE
  ACKNOWLEDGE WARN SUGGEST
commitment_strength: COMMITTED or NON_COMMITTED (binary)
addressee: power name string (e.g. FRANCE) or "ALL"
temporal_marker: PAST PRESENT FUTURE NONE
"""
from __future__ import annotations
import json
import re
from .. import config as C
from .. import llm_client

FOURTUPLE_KEYS = ("speech_act", "commitment_strength", "addressee", "temporal_marker")

SPEECH_ACTS = (
    "COMMIT", "REQUEST", "THREAT", "INFORM", "QUESTION", "PROMISE",
    "REJECT", "APOLOGIZE", "PROPOSE", "ACKNOWLEDGE", "WARN", "SUGGEST",
)

_SYSTEM = (
    "You are a discourse analyst. Extract a structured 4-tuple from a Diplomacy press message. "
    "Output ONLY JSON, no commentary.\n"
    "Keys:\n"
    "  speech_act: one of {acts}\n"
    "  commitment_strength: COMMITTED or NON_COMMITTED\n"
    "  addressee: the recipient power name in uppercase (or ALL)\n"
    "  temporal_marker: PAST, PRESENT, FUTURE, or NONE\n"
).format(acts=", ".join(SPEECH_ACTS))


def parse_tuple(sender: str, recipient: str, content: str,
                judge_model: str = C.JUDGE_MODEL) -> dict:
    """Return a dict with the 4 keys above. Always returns valid values (fallback to defaults)."""
    user = (
        f"Sender: {sender}\nRecipient: {recipient}\nMessage: {content}\n\n"
        "Return JSON with keys: speech_act, commitment_strength, addressee, temporal_marker."
    )
    r = llm_client.chat(
        model=judge_model,
        messages=[{"role": "system", "content": _SYSTEM},
                  {"role": "user", "content": user}],
        temperature=0.0, max_tokens=C.MAX_OUTPUT_TOKENS_JUDGE,
        seed=7, tag="parse_tuple",
    )
    return _coerce(r.text, recipient)


def _coerce(text: str, default_addressee: str) -> dict:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    parsed = {}
    if m:
        try:
            parsed = json.loads(m.group(0))
        except Exception:
            parsed = {}
    sa = str(parsed.get("speech_act", "INFORM")).upper().strip()
    if sa not in SPEECH_ACTS:
        sa = "INFORM"
    cs = str(parsed.get("commitment_strength", "NON_COMMITTED")).upper().strip()
    if cs not in ("COMMITTED", "NON_COMMITTED"):
        cs = "NON_COMMITTED"
    ad = str(parsed.get("addressee", default_addressee)).upper().strip() or default_addressee.upper()
    tm = str(parsed.get("temporal_marker", "NONE")).upper().strip()
    if tm not in ("PAST", "PRESENT", "FUTURE", "NONE"):
        tm = "NONE"
    return {"speech_act": sa, "commitment_strength": cs, "addressee": ad, "temporal_marker": tm}


def tuples_equal(a: dict, b: dict) -> bool:
    return all(a.get(k) == b.get(k) for k in FOURTUPLE_KEYS)
