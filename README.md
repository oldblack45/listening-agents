# Are Agents Listening to Each Other?

**Measuring What Drives Agent Actions in LLM-Based Multi-Agent Systems**

Reference implementation for the ICONIP 2026 paper. The framework
performs per-message **counterfactual attribution** at multi-agent
decision points (DPs) and labels each DP as *single-driver*,
*mixed-driver*, or *no-driver*.

It combines three components, one per challenge:

1. **Pragmatic-controlled perturbation generator** that fixes speech
   act, commitment strength, addressee, and temporal marker and
   admits only propositional substitution (intervention impurity).
2. **Noise-corrected attribution score $\mathrm{FS}_{\text{KL-excess}}$**
   that subtracts a same-temperature replicate baseline with a
   $1.5\sigma$ margin (sampling noise).
3. **Within-DP attribution vector and driver-regime label** that scores
   every incoming message under fixed co-incoming context and labels
   the DP as single-/mixed-/no-driver (cross-message interference).

---

## Repository layout

```
src/
  agents/             ReAct, AutoGen, GenAgents, CAMEL adapters
  slot_generator/     Pragmatic 4-tuple extractor + intervener + Eq. 3 gate
  env_adapter.py      Diplomacy press environment wrapper
  sotopia_env.py      SOTOPIA social environment wrapper
  runner_v4.py        Per-DP runner for Diplomacy
  runner_v4_sotopia.py  Per-DP runner for SOTOPIA
  llm_client.py       Cached OpenAI-compatible chat client
  metrics.py          FSKL-excess and noise-baseline KL
  analysis.py         Cell-level metric aggregation
  config.py           Sweep matrix + thresholds (Eq. 3, sigma margin)
scripts/
  run_v4.py                 Diplomacy main sweep
  run_v4_sotopia.py         SOTOPIA main sweep
  run_e2_allmask_v2.py      No-driver refinement (autonomous vs diffuse)
  run_e3_pairwise.py        Mixed-driver refinement (additive / competing / ...)
  run_mask_loo.py           MASK-LOO ablation (pragmatic-control necessity)
  classify_driver_structure.py     Apply Eq. 11 label rule
  analyze_v5.py / analyze_sotopia_v5.py    Per-cell summaries
  analyze_ablation.py       Reproduces Table 1
  analyze_intervention_agreement.py        Cross-operator Cohen's kappa
  recompute_anatomy.py      Sub-regime statistics for Fig. 5
  pick_case_studies.py / make_case_studies.py    Section 5.4 boxes
  make_fig_calibration.py   Figure 3
  make_figs_modeshare.py    Figure 4
  make_fig_anatomy.py       Figure 5
  make_fig_cases.py         Section 5.4 case-study panels
  compute_review_stats.py / fill_results.py    Misc numbers cited in text
data/
  pilot_b0/                 Created at runtime (see data/README.md)
```

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Tested with Python 3.10 and 3.11.

---

## Configuration

The runner talks to any OpenAI-compatible endpoint. Set:

```bash
export OPENAI_API_BASE="https://api.openai.com/v1"     # or your proxy
export OPENAI_API_KEY="sk-..."

# Optional, defaults shown
export AGENT_MODEL="gpt-4o-2024-08-06"
export JUDGE_MODEL="gpt-4o-2024-08-06"
export JUDGE_FALLBACK="gpt-4o-2024-08-06"

# Smoke-size run (1 episode per cell, smaller MC budget)
export PILOT_FAST=1
```

To reproduce the Claude Haiku 4.5 row, point the same variables at an
Anthropic-compatible gateway and set `AGENT_MODEL=claude-haiku-4-5`.

---

## Reproducing the paper

The complete 849-DP sweep is on the order of a day of wall-clock at
`MAX_CONCURRENT_REQUESTS=16`. We recommend running the smoke version
first:

```bash
PILOT_FAST=1 python -m scripts.run_v4
PILOT_FAST=1 python -m scripts.run_v4_sotopia
```

Full sweep:

```bash
python -m scripts.run_v4               # Diplomacy, all 4 archs x 4 ops
python -m scripts.run_v4_sotopia       # SOTOPIA equivalent

python -m scripts.classify_driver_structure   # apply Eq. 11
python -m scripts.analyze_v5                  # per-cell summary
python -m scripts.analyze_sotopia_v5

# Sub-regime refinements (Section 5.2 paragraphs 2-3)
python -m scripts.run_e2_allmask_v2
python -m scripts.run_e3_pairwise

# MASK-LOO ablation (Table 1, bottom row)
python -m scripts.run_mask_loo
python -m scripts.analyze_ablation

# Figures
python -m scripts.make_fig_calibration         # Fig. 3
python -m scripts.make_figs_modeshare          # Fig. 4
python -m scripts.make_fig_anatomy             # Fig. 5
python -m scripts.make_fig_cases               # Section 5.4
```

All intermediate artefacts land in `data/pilot_b0/`. The SQLite LLM
cache makes every step resumable.

---

## License

MIT. See [LICENSE](LICENSE).
