"""Validator: BLEU >= 0.3 ∧ SBERT-cosine >= 0.6 ∧ 4-tuple consistency.

Implements the acceptance gate of Eq. 3 in the paper:
- BLEU: sacrebleu corpus-BLEU-4 with exponential smoothing
- cosine: cosine similarity between Sentence-BERT embeddings
  (paraphrase-MiniLM-L6-v2; cached singleton model)
- 4-tuple consistency: re-parse via parser.parse_tuple and compare

For 'identity' the validator always passes.
For 'random_string'/'cross_episode_swap' we DO NOT require BLEU/cosine bounds
(those interventions are *meant* to break semantic similarity); they still pass
into the experiment as adversarial controls.
"""
from __future__ import annotations
import math
import re
from collections import Counter

from .parser import parse_tuple, tuples_equal
from .. import config as C


# ---- SBERT cosine (singleton model, lazy-loaded) -----------------------

_SBERT_MODEL = None
_SBERT_NAME = "sentence-transformers/paraphrase-MiniLM-L6-v2"


def _get_sbert():
    global _SBERT_MODEL
    if _SBERT_MODEL is None:
        from sentence_transformers import SentenceTransformer  # type: ignore
        _SBERT_MODEL = SentenceTransformer(_SBERT_NAME)
    return _SBERT_MODEL


def cosine_sbert(a: str, b: str) -> float:
    """Cosine similarity between Sentence-BERT embeddings of two strings.

    Returns a value in [-1, 1]; in practice non-degenerate sentence pairs
    fall in [0, 1]. Empty strings short-circuit to 0.0.
    """
    if not a.strip() or not b.strip():
        return 0.0
    model = _get_sbert()
    embs = model.encode([a, b], normalize_embeddings=True, convert_to_numpy=True)
    return float((embs[0] * embs[1]).sum())


# ---- BLEU (sacrebleu) --------------------------------------------------

def bleu_score(reference: str, hypothesis: str) -> float:
    """Corpus-BLEU-4 with exponential smoothing, normalised to [0, 1]."""
    from sacrebleu import corpus_bleu  # type: ignore
    b = corpus_bleu([hypothesis], [[reference]], smooth_method="exp")
    return float(b.score) / 100.0


# ---- Validation --------------------------------------------------------

def validate_intervention(
    intervention_type: str,
    sender: str,
    recipient: str,
    original: str,
    rewritten: str,
    original_tuple: dict | None = None,
    judge_model: str = C.JUDGE_MODEL,
    require_similarity: bool | None = None,
) -> dict:
    """Return dict {pass: bool, bleu, cosine, tuple_match, reason}."""
    if require_similarity is None:
        require_similarity = intervention_type in ("identity", "synonym", "fact_replace", "counterfactual")

    bleu = bleu_score(original, rewritten)
    cos = cosine_sbert(original, rewritten)

    # 4-tuple consistency
    tuple_match = True
    if intervention_type in ("synonym", "identity"):
        ot = original_tuple or parse_tuple(sender, recipient, original, judge_model)
        nt = parse_tuple(sender, recipient, rewritten, judge_model)
        tuple_match = tuples_equal(ot, nt)
    elif intervention_type in ("fact_replace",):
        # speech_act + addressee must match (commitment_strength and temporal_marker may drift slightly)
        ot = original_tuple or parse_tuple(sender, recipient, original, judge_model)
        nt = parse_tuple(sender, recipient, rewritten, judge_model)
        tuple_match = (ot["speech_act"] == nt["speech_act"]) and (ot["addressee"] == nt["addressee"])
    elif intervention_type == "counterfactual":
        # commitment_strength should flip OR remain (rewrite negates content, not strength label)
        tuple_match = True
    else:
        tuple_match = True  # random / swap: not gated

    ok = True
    reasons = []
    if require_similarity:
        if bleu < C.BLEU_MIN:
            ok = False; reasons.append(f"bleu={bleu:.2f}<{C.BLEU_MIN}")
        if cos < C.COSINE_MIN:
            ok = False; reasons.append(f"cos={cos:.2f}<{C.COSINE_MIN}")
        if not tuple_match:
            ok = False; reasons.append("4-tuple drift")

    return {
        "pass": ok, "bleu": bleu, "cosine": cos, "tuple_match": tuple_match,
        "reason": ";".join(reasons) if reasons else "ok",
    }


def slot_pass_rate(records: list[dict]) -> float:
    if not records:
        return 0.0
    return sum(1 for r in records if r.get("pass")) / len(records)


    # 4-tuple consistency
    tuple_match = True
    if intervention_type in ("synonym", "identity"):
        ot = original_tuple or parse_tuple(sender, recipient, original, judge_model)
        nt = parse_tuple(sender, recipient, rewritten, judge_model)
        tuple_match = tuples_equal(ot, nt)
    elif intervention_type in ("fact_replace",):
        # speech_act + addressee must match (commitment_strength and temporal_marker may drift slightly)
        ot = original_tuple or parse_tuple(sender, recipient, original, judge_model)
        nt = parse_tuple(sender, recipient, rewritten, judge_model)
        tuple_match = (ot["speech_act"] == nt["speech_act"]) and (ot["addressee"] == nt["addressee"])
    elif intervention_type == "counterfactual":
        # commitment_strength should flip OR remain (rewrite negates content, not strength label)
        tuple_match = True
    else:
        tuple_match = True  # random / swap: not gated

    ok = True
    reasons = []
    if require_similarity:
        if bleu < C.BLEU_MIN:
            ok = False; reasons.append(f"bleu={bleu:.2f}<{C.BLEU_MIN}")
        if cos < C.COSINE_MIN:
            ok = False; reasons.append(f"cos={cos:.2f}<{C.COSINE_MIN}")
        if not tuple_match:
            ok = False; reasons.append("4-tuple drift")

    return {
        "pass": ok, "bleu": bleu, "cosine": cos, "tuple_match": tuple_match,
        "reason": ";".join(reasons) if reasons else "ok",
    }


def slot_pass_rate(records: list[dict]) -> float:
    if not records:
        return 0.0
    return sum(1 for r in records if r.get("pass")) / len(records)
