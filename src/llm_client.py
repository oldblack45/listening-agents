"""LLM client wrapper for Pilot B0."""
from __future__ import annotations
import hashlib, json, sqlite3, threading, time
from dataclasses import dataclass
from typing import Any
import openai
from . import config as C

_SEM = threading.Semaphore(C.MAX_CONCURRENT_REQUESTS)
_LOG_LOCK = threading.Lock()
_CACHE_LOCK = threading.Lock()


def _cache_conn():
    conn = sqlite3.connect(str(C.CACHE_DB), timeout=30, isolation_level=None)
    conn.execute("CREATE TABLE IF NOT EXISTS llm_cache (key TEXT PRIMARY KEY, response_json TEXT NOT NULL, created_at REAL NOT NULL)")
    return conn


def _cache_key(model, messages, temperature, max_tokens, seed):
    payload = json.dumps({"m": model, "msg": messages, "t": temperature, "mt": max_tokens, "s": seed}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cache_get(key):
    with _CACHE_LOCK:
        conn = _cache_conn()
        try:
            row = conn.execute("SELECT response_json FROM llm_cache WHERE key=?", (key,)).fetchone()
        finally:
            conn.close()
    return None if row is None else json.loads(row[0])


def _cache_put(key, value):
    with _CACHE_LOCK:
        conn = _cache_conn()
        try:
            conn.execute("INSERT OR REPLACE INTO llm_cache (key, response_json, created_at) VALUES (?,?,?)", (key, json.dumps(value, ensure_ascii=False), time.time()))
        finally:
            conn.close()


def _log_call(record):
    with _LOG_LOCK:
        with open(C.LLM_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


_CLIENT = openai.OpenAI(base_url=C.LLM_API_BASE, api_key=C.LLM_API_KEY, timeout=C.REQUEST_TIMEOUT_S)


@dataclass
class ChatResult:
    text: str
    usage: dict
    cached: bool
    elapsed: float


def _try_one_call(model, messages, temperature, max_tokens, seed):
    """Single API call, returns (text, usage, elapsed). Raises on empty choices or errors."""
    with _SEM:
        t0 = time.time()
        kwargs = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        ml = model.lower()
        is_reasoning = (("opus" in ml) or ("sonnet" in ml) or ml.startswith("gpt-5") or ("gemini-3" in ml))
        if is_reasoning:
            if ("opus" in ml) or ("sonnet" in ml):
                effort = "medium"
            elif "gemini-3" in ml:
                effort = "low"  # leave more of the token budget for the visible reply
            else:
                effort = "low"
            kwargs["extra_body"] = {"reasoning_effort": effort}
        if seed is not None:
            kwargs["seed"] = seed
        resp = _CLIENT.chat.completions.create(**kwargs)
        elapsed = time.time() - t0
    # Defensive: some upstream (Claude Opus 4.7 via proxy) occasionally returns no choices.
    if not getattr(resp, "choices", None):
        raise RuntimeError(f"empty choices from {model} (finish_reason unavailable)")
    text = (resp.choices[0].message.content or "").strip()
    if not text:
        raise RuntimeError(f"empty content from {model} (choices[0].message.content is empty)")
    u = getattr(resp, "usage", None)
    usage = {
        "prompt": getattr(u, "prompt_tokens", 0) if u else 0,
        "completion": getattr(u, "completion_tokens", 0) if u else 0,
        "total": getattr(u, "total_tokens", 0) if u else 0,
    }
    return text, usage, elapsed


def chat(model, messages, temperature=0.7, max_tokens=600, seed=None, use_cache=True, tag=""):
    key = _cache_key(model, messages, temperature, max_tokens, seed)
    if use_cache:
        hit = _cache_get(key)
        if hit is not None:
            return ChatResult(text=hit["text"], usage=hit.get("usage", {}), cached=True, elapsed=0.0)
    last_err = None
    # Track how many consecutive empty-response errors we hit on the primary model;
    # if too many, fall back to JUDGE_FALLBACK (which is typically GPT-4o).
    empty_streak = 0
    fallback_model = getattr(C, "JUDGE_FALLBACK", None)
    EMPTY_BEFORE_FALLBACK = 3
    using_fallback = False
    active_model = model
    active_seed = seed
    for attempt in range(C.MAX_RETRIES):
        try:
            text, usage, elapsed = _try_one_call(active_model, messages, temperature, max_tokens, active_seed)
            _log_call({"ts": time.time(), "model": active_model, "tag": tag, "temperature": temperature, "seed": active_seed, "attempt": attempt, "elapsed": elapsed, "usage": usage, "len_out": len(text), "fallback": using_fallback})
            result = {"text": text, "usage": usage}
            if use_cache:
                _cache_put(key, result)
            return ChatResult(text=text, usage=usage, cached=False, elapsed=elapsed)
        except Exception as e:
            last_err = e
            msg_str = str(e)
            is_empty = ("empty choices" in msg_str) or ("empty content" in msg_str) or ("list index out of range" in msg_str)
            if is_empty:
                empty_streak += 1
                # Rotate seed deterministically to dodge a stuck empty response.
                if active_seed is not None:
                    active_seed = active_seed + 10000 * (attempt + 1)
                # After EMPTY_BEFORE_FALLBACK consecutive empties, fall back.
                if (not using_fallback) and fallback_model and empty_streak >= EMPTY_BEFORE_FALLBACK:
                    using_fallback = True
                    active_model = fallback_model
                    _log_call({"ts": time.time(), "tag": tag, "event": "fallback_engaged", "from": model, "to": fallback_model, "attempt": attempt})
            else:
                empty_streak = 0
            wait = min(60.0, C.RETRY_BASE_SECONDS * (2 ** min(attempt, 4)))
            _log_call({"ts": time.time(), "model": active_model, "tag": tag, "attempt": attempt, "error": f"{type(e).__name__}: {e}", "wait": wait, "empty_streak": empty_streak, "fallback": using_fallback})
            time.sleep(wait)
    raise RuntimeError(f"chat({model}) failed after {C.MAX_RETRIES} retries: {last_err}")


def chat_n(model, messages, n, temperature=0.7, max_tokens=600, base_seed=1000, tag=""):
    out = []
    for i in range(n):
        r = chat(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens, seed=base_seed + i, use_cache=True, tag=f"{tag}#n{i}")
        out.append(r.text)
    return out
