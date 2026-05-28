# data/pilot_b0/

Created at runtime by `scripts/run_v4.py` and friends. Holds:

```
raw_episodes/    base episodes per (architecture, model, scenario)
messages/        per-DP incoming-message bundles
interventions/   pragmatic-controlled perturbations passing the Eq. 3 gate
action_logs/     Monte-Carlo action samples under each (do, baseline) condition
metrics/         per-DP FSKL-excess and driver-regime labels
cache/           SQLite cache of LLM responses (resume-safe)
llm_calls.jsonl  per-call accounting log
```

To regenerate from scratch, see the **Reproducing the paper** section
in the top-level [README](../../README.md).

The full sweep used in the paper is on the order of several gigabytes
of raw episodes and a few hundred megabytes of intervention/metric
files. We do not redistribute the raw artefacts because (i) Diplomacy
press transcripts can be reproduced deterministically from the cache,
and (ii) SOTOPIA scenarios contain character profiles whose
distribution is governed by the upstream SOTOPIA license. The cache
file alone is sufficient to recompute every published number without
re-issuing API calls; we are happy to share it on request.
