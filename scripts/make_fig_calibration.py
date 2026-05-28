"""Make fig_calibration.pdf: §5.1 calibration figure.

Two panels, sharing the scientific palette of fig_modeshare / fig_cases:
  (a) Mean FS_KL per (architecture, operator), averaged over Diplomacy and
      SOTOPIA. Bars are clustered by operator (identity / fact-replace /
      counterfactual / random-string) so the controlled severity gradient
      is visible at a glance.
  (b) Within-DP rank gap (top-ranked FS_KL minus bottom-ranked FS_KL),
      per architecture under the three informative operators.

Reads pre-computed analysis JSONs under data/pilot_b0/analysis/.
"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams.update({
    "font.size": 13,
    "font.weight": "bold",
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
})

ROOT = Path(__file__).resolve().parents[2]
ANA = ROOT / "data" / "pilot_b0" / "analysis"
OUT = ROOT / "paper" / "figures" / "fig_calibration.pdf"

ARCHS = ["react", "autogen", "genagents", "camel"]
ARCH_LABEL = ["ReAct", "AutoGen", "GenAgents", "CAMEL"]
OPS = ["identity", "fact_replace", "counterfactual", "random_string"]
OP_LABEL = ["identity", "fact-replace", "counterfactual", "random-string"]
INF_OPS = ["fact_replace", "counterfactual", "random_string"]
INF_OP_LABEL = ["fact-replace", "counterfactual", "random-string"]

# Scientific palette (cool-dominant, ColorBrewer/Okabe-Ito inspired)
# Operator-axis: a 4-stop cool->warm gradient that mirrors the severity ladder.
OP_COLORS = {
    "identity":       "#C5CBD3",   # cool neutral gray (sanity lower bound)
    "fact_replace":   "#5B8FB9",   # mid blue
    "counterfactual": "#2C5F8D",   # steel blue (strongest informative)
    "random_string":  "#C97B2A",   # amber (upper bound)
}
ARCH_COLORS = {
    "react":     "#2C5F8D",   # steel blue
    "autogen":   "#C97B2A",   # amber
    "genagents": "#6C757D",   # slate gray
    "camel":     "#5B8FB9",   # mid blue
}


def load_main():
    dipl = json.loads((ANA / "v5_main_table.json").read_text(encoding="utf-8"))
    soto = json.loads((ANA / "v5_sotopia_main_table.json").read_text(encoding="utf-8"))
    return dipl, soto


def load_ranking():
    dipl = json.loads((ANA / "v5_ranking.json").read_text(encoding="utf-8"))
    soto = json.loads((ANA / "v5_sotopia_ranking.json").read_text(encoding="utf-8"))
    return dipl, soto


def panel_a(ax):
    """Mean FS_KL per (arch, op), averaged across envs (and models, if present)."""
    dipl, soto = load_main()
    rows = dipl + soto

    # bucket: (arch, op) -> list of fs_fine_mean
    bucket = {(a, o): [] for a in ARCHS for o in OPS}
    err = {(a, o): [] for a in ARCHS for o in OPS}
    for r in rows:
        key = (r["arch"], r["intervention"])
        if key not in bucket:
            continue
        bucket[key].append(r["fs_fine_mean"])
        # half-CI as proxy for error bar
        err[key].append(0.5 * (r["fs_fine_hi"] - r["fs_fine_lo"]))

    width = 0.20
    xs = np.arange(len(ARCHS))
    for k, op in enumerate(OPS):
        means = []
        errs = []
        for a in ARCHS:
            vs = bucket[(a, op)]
            es = err[(a, op)]
            means.append(np.mean(vs) if vs else 0.0)
            errs.append(np.mean(es) if es else 0.0)
        offset = (k - 1.5) * width
        ax.bar(xs + offset, means, width=width,
               color=OP_COLORS[op], yerr=errs,
               error_kw=dict(ecolor='#444', lw=0.7, capsize=2),
               label=OP_LABEL[k], edgecolor='none')

    ax.axhline(0, color='#444', lw=0.6)
    # Indicate noise margin band (~1.5 sigma typical magnitude ~ 0.1)
    ax.axhspan(-0.10, 0.10, color='#888', alpha=0.10, zorder=0)
    ax.set_xticks(xs)
    ax.set_xticklabels(ARCH_LABEL, fontsize=12, fontweight='bold', rotation=15, ha='right')
    ax.set_ylabel(r"mean $\mathbf{FS}_{\mathbf{KL\!-\!excess}}$", fontsize=13, fontweight='bold')
    ax.set_title("(a) Operator severity gradient", fontsize=14, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Compact legend, two columns
    ax.legend(fontsize=11, loc='upper left', frameon=False, ncol=2,
              handlelength=1.2, columnspacing=0.8, prop={'weight':'bold','size':11})


def panel_b(ax):
    """Within-DP ranking gap per arch under informative operators."""
    dipl, soto = load_ranking()
    rows = dipl + soto

    bucket = {(a, o): [] for a in ARCHS for o in INF_OPS}
    for r in rows:
        key = (r["arch"], r["intervention"])
        if key not in bucket:
            continue
        bucket[key].append(r.get("mean_rank_gap", 0.0))

    width = 0.20
    xs = np.arange(len(INF_OPS))
    for k, a in enumerate(ARCHS):
        vals = [np.mean(bucket[(a, o)]) if bucket[(a, o)] else 0.0
                for o in INF_OPS]
        offset = (k - 1.5) * width
        ax.bar(xs + offset, vals, width=width,
               color=ARCH_COLORS[a], label=ARCH_LABEL[k], edgecolor='none')

    ax.axhline(0, color='#444', lw=0.6)
    ax.set_xticks(xs)
    ax.set_xticklabels(INF_OP_LABEL, fontsize=12, fontweight='bold', rotation=15, ha='right')
    ax.set_ylabel(r"mean within-DP rank gap", fontsize=13, fontweight='bold')
    ax.set_title("(b) Within-DP ranking gap", fontsize=14, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=11, loc='upper left', frameon=False, ncol=2,
              handlelength=1.2, columnspacing=0.8, prop={'weight':'bold','size':11})


def main():
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.0))
    panel_a(axes[0])
    panel_b(axes[1])
    plt.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT, bbox_inches="tight")
    print("wrote", OUT)
    plt.close(fig)


if __name__ == "__main__":
    main()
