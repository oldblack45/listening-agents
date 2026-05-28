"""Make mode-share figures from driver_structure_classification.csv.

Outputs:
  paper/figures/fig_modeshare.pdf  - stacked bars per (env, arch, model)
  paper/figures/fig_modeshare_arch.pdf - per (env, arch) averaged over models
"""
from __future__ import annotations
import csv
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams.update({
    "font.size": 13,
    "font.weight": "bold",
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
})

ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "data" / "pilot_b0" / "analysis" / "driver_structure_classification.csv"
OUT_DIR = ROOT / "paper" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ARCH_ORDER = ["react", "autogen", "genagents", "camel"]
ARCH_LABEL = {"react": "ReAct", "autogen": "AutoGen", "genagents": "GenAgents", "camel": "CAMEL"}
COL_SINGLE = "#2C5F8D"   # steel blue
COL_MIXED  = "#C97B2A"   # amber / ochre
COL_NO     = "#6C757D"   # slate gray


def load_rows():
    with open(CSV_PATH, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fig_per_model(rows):
    """4 subplots: (env, model) -> stacked bar across archs (avg over interventions)."""
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 6.4), sharey=True)
    axes = axes.flatten()
    panels = [("diplomacy", "gpt4o", "Diplomacy / GPT-4o"),
              ("diplomacy", "haiku", "Diplomacy / Haiku 4.5"),
              ("sotopia",   "gpt4o", "SOTOPIA / GPT-4o"),
              ("sotopia",   "haiku", "SOTOPIA / Haiku 4.5")]

    for ax, (env, mdl, title) in zip(axes, panels):
        # for each arch, avg over interventions
        agg = {a: {"s": [], "m": [], "n": []} for a in ARCH_ORDER}
        for r in rows:
            if r["env"] != env or r["model_tag"] != mdl: continue
            a = r["arch"]
            agg[a]["s"].append(float(r["pct_single"]))
            agg[a]["m"].append(float(r["pct_mixed"]))
            agg[a]["n"].append(float(r["pct_no"]))
        xs = list(range(len(ARCH_ORDER)))
        s_vals = [sum(agg[a]["s"])/max(len(agg[a]["s"]),1) for a in ARCH_ORDER]
        m_vals = [sum(agg[a]["m"])/max(len(agg[a]["m"]),1) for a in ARCH_ORDER]
        n_vals = [sum(agg[a]["n"])/max(len(agg[a]["n"]),1) for a in ARCH_ORDER]
        ax.bar(xs, s_vals, color=COL_SINGLE, label="single-driver")
        ax.bar(xs, m_vals, bottom=s_vals, color=COL_MIXED, label="mixed-driver")
        bot2 = [s+m for s,m in zip(s_vals, m_vals)]
        ax.bar(xs, n_vals, bottom=bot2, color=COL_NO, label="no-driver")
        ax.set_xticks(xs)
        ax.set_xticklabels([ARCH_LABEL[a] for a in ARCH_ORDER], rotation=0, fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylim(0, 100)
        if ax is axes[0] or ax is axes[2]:
            ax.set_ylabel("% of decision points", fontsize=13, fontweight='bold')
        for i, (s, m, n) in enumerate(zip(s_vals, m_vals, n_vals)):
            if s >= 8:
                ax.text(i, s/2, f"{s:.0f}", ha='center', va='center', fontsize=12, color='white', weight='bold')
            if m >= 5:
                ax.text(i, s + m/2, f"{m:.0f}", ha='center', va='center', fontsize=12, color='white', weight='bold')
            if n >= 8:
                ax.text(i, s+m+n/2, f"{n:.0f}", ha='center', va='center', fontsize=12, color='white', weight='bold')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # shared legend
    handles = [mpatches.Patch(color=COL_SINGLE, label='single-driver'),
               mpatches.Patch(color=COL_MIXED,  label='mixed-driver'),
               mpatches.Patch(color=COL_NO,     label='no-driver')]
    fig.legend(handles=handles, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.00), frameon=False, fontsize=11, prop={'weight':'bold','size':11})
    plt.tight_layout(rect=(0,0,1,0.96))
    out = OUT_DIR / "fig_modeshare.pdf"
    plt.savefig(out, bbox_inches="tight")
    print("wrote", out)
    plt.close(fig)


def fig_overall(rows):
    """Single panel: per (env, arch) averaged over models AND interventions."""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True)
    for ax, env in zip(axes, ["diplomacy", "sotopia"]):
        agg = {a: {"s": [], "m": [], "n": []} for a in ARCH_ORDER}
        for r in rows:
            if r["env"] != env: continue
            a = r["arch"]
            agg[a]["s"].append(float(r["pct_single"]))
            agg[a]["m"].append(float(r["pct_mixed"]))
            agg[a]["n"].append(float(r["pct_no"]))
        xs = list(range(len(ARCH_ORDER)))
        s_vals = [sum(agg[a]["s"])/max(len(agg[a]["s"]),1) for a in ARCH_ORDER]
        m_vals = [sum(agg[a]["m"])/max(len(agg[a]["m"]),1) for a in ARCH_ORDER]
        n_vals = [sum(agg[a]["n"])/max(len(agg[a]["n"]),1) for a in ARCH_ORDER]
        ax.bar(xs, s_vals, color=COL_SINGLE)
        ax.bar(xs, m_vals, bottom=s_vals, color=COL_MIXED)
        bot2 = [s+m for s,m in zip(s_vals, m_vals)]
        ax.bar(xs, n_vals, bottom=bot2, color=COL_NO)
        ax.set_xticks(xs)
        ax.set_xticklabels([ARCH_LABEL[a] for a in ARCH_ORDER], rotation=15)
        ax.set_title("Diplomacy" if env=="diplomacy" else "SOTOPIA")
        ax.set_ylim(0, 100)
        if ax is axes[0]:
            ax.set_ylabel("% of decision points")
        for i, (s, m, n) in enumerate(zip(s_vals, m_vals, n_vals)):
            ax.text(i, s/2, f"{s:.0f}", ha='center', va='center', fontsize=8, color='white', weight='bold')
            if m >= 4:
                ax.text(i, s + m/2, f"{m:.0f}", ha='center', va='center', fontsize=8, color='white', weight='bold')
            ax.text(i, s+m+n/2, f"{n:.0f}", ha='center', va='center', fontsize=8, color='white', weight='bold')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    handles = [mpatches.Patch(color=COL_SINGLE, label='single-driver'),
               mpatches.Patch(color=COL_MIXED,  label='mixed-driver'),
               mpatches.Patch(color=COL_NO,     label='no-driver')]
    fig.legend(handles=handles, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.03), frameon=False, fontsize=9)
    plt.tight_layout(rect=(0,0,1,0.93))
    out = OUT_DIR / "fig_modeshare_arch.pdf"
    plt.savefig(out, bbox_inches="tight")
    print("wrote", out)
    plt.close(fig)


def main():
    rows = load_rows()
    fig_per_model(rows)
    fig_overall(rows)


if __name__ == "__main__":
    main()
