"""Global configuration for the per-message counterfactual attribution
framework used in *Are Agents Listening to Each Other?*.

Endpoints and API keys are read from environment variables so the same
script works against OpenAI / Anthropic official endpoints or any
OpenAI-compatible proxy (e.g., a local LiteLLM gateway).

Required environment variables (set whichever applies):
    OPENAI_API_BASE   (default: https://api.openai.com/v1)
    OPENAI_API_KEY
    ANTHROPIC_API_KEY (only if you call Claude models directly without a proxy)

Optional:
    PILOT_FAST=1      run the smoke-size sweep (1 episode per cell)
    MODEL_TAG=haiku   suffix appended to metric files for cross-model runs
"""
from __future__ import annotations

import os
from pathlib import Path

_FAST = os.environ.get("PILOT_FAST", "0") == "1"

# ---- LLM endpoints (env-driven) ----------------------------------------
LLM_API_BASE = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
LLM_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# Default models used in the paper. Override with command-line args or
# environment variables where appropriate.
AGENT_MODEL = os.environ.get("AGENT_MODEL", "gpt-4o-2024-08-06")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gpt-4o-2024-08-06")
JUDGE_FALLBACK = os.environ.get("JUDGE_FALLBACK", "gpt-4o-2024-08-06")

# Tag appended to metric filenames to separate multi-model runs.
MODEL_TAG = os.environ.get("MODEL_TAG", "")

# ---- Diplomacy ---------------------------------------------------------
DIPLOMACY_POWERS = ["ENGLAND", "FRANCE", "GERMANY", "ITALY", "AUSTRIA", "RUSSIA", "TURKEY"]
MAX_TURNS_PER_EPISODE = 6        # Spring/Fall x 3 years; orders only
EPISODES_PER_CELL = 20

# ---- Cell matrix (4 archs x 4 interventions) ---------------------------
INTERVENTIONS = [
    "identity",
    "fact_replace",
    "counterfactual",
    "random_string",
]
ARCHS = ["react", "autogen", "genagents", "camel"]
INCENTIVE = "C"  # cooperative

# ---- Sampling ----------------------------------------------------------
ACTION_SAMPLES_PER_DO = 8
TEMPERATURE = 0.7
NOISE_BASELINE_TEMPS = [0.7]     # same-tau replicate baseline
NOISE_SAMPLES_PER_TEMP = 8

# ---- Validation thresholds (paper Eq. 3) -------------------------------
SLOT_PASS_RATE_GATE = 0.70
FS_EXCESS_SIGMA = 1.5
BLEU_MIN = 0.3
COSINE_MIN = 0.6

# ---- Concurrency / robustness ------------------------------------------
MAX_CONCURRENT_REQUESTS = 16
MAX_RETRIES = 8
RETRY_BASE_SECONDS = 10.0
REQUEST_TIMEOUT_S = 60.0

# ---- Token budget guards -----------------------------------------------
MAX_OUTPUT_TOKENS_AGENT = 600
MAX_OUTPUT_TOKENS_JUDGE = 200
MAX_OUTPUT_TOKENS_INTERVENE = 300

# ---- Paths -------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "pilot_b0"
RAW_DIR = DATA_DIR / "raw_episodes"
MSG_DIR = DATA_DIR / "messages"
INT_DIR = DATA_DIR / "interventions"
ACT_DIR = DATA_DIR / "action_logs"
METRIC_DIR = DATA_DIR / "metrics"
CACHE_DIR = DATA_DIR / "cache"
LLM_LOG = DATA_DIR / "llm_calls.jsonl"

for _p in [RAW_DIR, MSG_DIR, INT_DIR, ACT_DIR, METRIC_DIR, CACHE_DIR]:
    _p.mkdir(parents=True, exist_ok=True)

CACHE_DB = CACHE_DIR / "llm_cache.sqlite"

if _FAST:
    MAX_TURNS_PER_EPISODE = 3
    EPISODES_PER_CELL = 1
    ACTION_SAMPLES_PER_DO = 4
    NOISE_SAMPLES_PER_TEMP = 4
    NOISE_BASELINE_TEMPS = [0.7]
