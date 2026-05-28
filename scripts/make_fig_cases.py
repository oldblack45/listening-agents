"""Make fig_cases.pdf: per-message FSKL bars for the three case-study DPs.

Uses the unified muted academic palette shared with fig_modeshare and fig_anatomy.

Cases (matching the tcolorbox boxes in §5.3):
  Case 1 SINGLE  : sotopia/genagents/haiku ep3 -> ALEX, driver = DREW (+1.96)
  Case 2 MIXED   : diplomacy/autogen/gpt4o ep3 -> RUSSIA, drivers = ENGLAND (+0.98), GERMANY (+0.65)
  Case 3 NO      : diplomacy/react/gpt4o ep0 -> GERMANY, all negative
"""
from __future__ import annotations
from pathlib import Path
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
OUT = ROOT / "paper" / "figures" / "fig_cases.pdf"

# Scientific palette (ColorBrewer / Okabe-Ito inspired)
COL_SINGLE = "#2C5F8D"   # steel blue     - driver in single mode
COL_MIXED  = "#C97B2A"   # amber          - driver in mixed mode
COL_NO     = "#6C757D"   # slate gray     - all-negative no-driver bars
COL_BG     = "#C5CBD3"   # cool neutral   - non-driver bars

CASES = [
    {
        "title": "Case 1 single-driver\n(Sotopia / GenAgents / Haiku)",
        "senders": ["BLAKE", "CASEY", "ERIN", "DREW"],
        "vals":    [-0.08,    0.06,   0.30,   1.96],
        "driver_idx": [3],
        "mode_color": COL_SINGLE,
    },
    {
        "title": "Case 2 mixed-driver\n(Diplomacy / AutoGen / GPT-4o)",
        "senders": ["AUSTRIA", "TURKEY", "GERMANY", "ENGLAND"],
        "vals":    [-0.13,     -0.05,   0.65,      0.98],
        "driver_idx": [2, 3],
        "mode_color": COL_MIXED,
    },
    {
        "title": "Case 3 no-driver\n(Diplomacy / ReAct / GPT-4o)",
        "senders": ["ENGLAND", "FRANCE", "RUSSIA"],
        "vals":    [-1.00,     -0.68,   -0.60],
        "driver_idx": [],
        "mode_color": COL_NO,
    },
]


def main():
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.0))
    for ax, case in zip(axes, CASES):
        senders = case["senders"]
        vals = case["vals"]
        n = len(senders)
        colors = [COL_BG] * n
        for idx in case["driver_idx"]:
            colors[idx] = case["mode_color"]
        # if no-driver: use mode color (maroon) for all bars at lower opacity feel
        if not case["driver_idx"]:
            colors = [case["mode_color"]] * n

        ys = list(range(n))
        ax.barh(ys, vals, color=colors, edgecolor='none')
        ax.set_yticks(ys)
        ax.set_yticklabels(senders, fontsize=12, fontweight='bold')
        ax.invert_yaxis()
        ax.axvline(0, color='#444', linewidth=0.6)
        # noise floor band: 1.5 sigma = approx 0.10 for these
        ax.axvspan(-0.10, 0.10, color='#888', alpha=0.10, zorder=0)
        ax.set_xlabel(r"$\mathbf{FS}_{\mathbf{KL\!-\!excess}}$", fontsize=13, fontweight='bold')
        ax.set_title(case["title"], fontsize=12, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # uniform x range across panels for comparability
        ax.set_xlim(-1.2, 2.2)
        for i, v in enumerate(vals):
            ha = 'left' if v >= 0 else 'right'
            offset = 0.05 if v >= 0 else -0.05
            ax.text(v + offset, i, f"{v:+.2f}", va='center', ha=ha, fontsize=11, color='#111', weight='bold')

    plt.tight_layout()
    plt.savefig(OUT, bbox_inches="tight")
    print("wrote", OUT)
    plt.close(fig)


if __name__ == "__main__":
    main()
