"""Make the combined anatomy figure: no-driver (autonomous/diffuse) + mixed-driver
(additive / competing / redundant / partial-or-neither), two panels side by side.

Replaces tab:nodriver-anatomy and tab:mixed-anatomy in the paper.

Hardcoded counts come from the analysis pipelines already reported in the paper.
"""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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
OUT_DIR = ROOT / "paper" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Scientific palette (ColorBrewer / Okabe-Ito inspired)
COL_AUT  = "#2C5F8D"   # autonomous = steel blue
COL_DIF  = "#C5CBD3"   # diffuse    = cool neutral gray
COL_ADD  = "#2C5F8D"   # additive   = steel blue
COL_COMP = "#C97B2A"   # competing  = amber
COL_RED  = "#6C757D"   # redundant  = slate gray
COL_PRT  = "#C5CBD3"   # partial    = cool neutral gray

# Rows: (env, model, n_no, diffuse, autonomous, n_mixed, add, comp, red, prt)
ROWS = [
    ("Dipl.",   "GPT-4o",   33, 45.5, 54.5, 4, 0, 3, 1, 0),
    ("Dipl.",   "Haiku",    15, 46.7, 53.3, 4, 1, 3, 0, 0),
    ("SOTOPIA", "GPT-4o",   36, 30.6, 69.4, 4, 3, 1, 0, 0),
    ("SOTOPIA", "Haiku",    10, 20.0, 80.0, 3, 2, 1, 0, 0),
]


def main():
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8))

    # ---- LEFT: no-driver refinement (% stacked) ----
    ax = axes[0]
    ys = list(range(len(ROWS)))
    labels = [f"{e}  {m}" for (e, m, *_) in ROWS]
    aut_pct = [r[4] for r in ROWS]
    dif_pct = [r[3] for r in ROWS]
    ns      = [r[2] for r in ROWS]

    ax.barh(ys, aut_pct, color=COL_AUT, label="autonomous")
    ax.barh(ys, dif_pct, left=aut_pct, color=COL_DIF, label="diffuse")
    for i, (a, d, n) in enumerate(zip(aut_pct, dif_pct, ns)):
        if i == 0:
            ax.text(a / 2, i, f"{a:.0f}\nautonomous", va='center', ha='center', fontsize=10, color='white', weight='bold')
            ax.text(a + d / 2, i, f"{d:.0f}\ndiffuse", va='center', ha='center', fontsize=10, color='#222', weight='bold')
        else:
            ax.text(a / 2, i, f"{a:.0f}", va='center', ha='center', fontsize=11, color='white', weight='bold')
            ax.text(a + d / 2, i, f"{d:.0f}", va='center', ha='center', fontsize=11, color='#222', weight='bold')
        ax.text(102, i, f"n={n}", va='center', ha='left', fontsize=11, color='#222', weight='bold')
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=12, fontweight='bold')
    ax.invert_yaxis()
    ax.set_xlim(0, 115)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("% of no-driver DPs", fontsize=13, fontweight='bold')
    ax.set_title("(a) No-driver refinement", fontsize=14, fontweight='bold')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # ---- RIGHT: mixed-driver refinement (counts stacked, normalized to % within row) ----
    ax = axes[1]
    add_pct, comp_pct, red_pct, prt_pct, ns2 = [], [], [], [], []
    for r in ROWS:
        nm = r[5]
        ns2.append(nm)
        if nm == 0:
            add_pct.append(0); comp_pct.append(0); red_pct.append(0); prt_pct.append(0); continue
        add_pct.append(100 * r[6] / nm)
        comp_pct.append(100 * r[7] / nm)
        red_pct.append(100 * r[8] / nm)
        prt_pct.append(100 * r[9] / nm)

    ax.barh(ys, add_pct, color=COL_ADD, label="additive")
    left = add_pct[:]
    ax.barh(ys, comp_pct, left=left, color=COL_COMP, label="competing")
    left = [l + c for l, c in zip(left, comp_pct)]
    ax.barh(ys, red_pct, left=left, color=COL_RED, label="redundant")
    left = [l + r2 for l, r2 in zip(left, red_pct)]
    ax.barh(ys, prt_pct, left=left, color=COL_PRT, label="partial/neither")

    LABELS_B = ["additive", "competing", "redundant", "partial"]
    # Pick the row where each label is most prominent (largest pct), so all
    # four legend texts appear at least once inside the bars.
    all_pcts = [add_pct, comp_pct, red_pct, prt_pct]
    label_row = [max(range(len(ROWS)), key=lambda i: all_pcts[k][i]) for k in range(4)]
    for i, (a, c, rd, p, n) in enumerate(zip(add_pct, comp_pct, red_pct, prt_pct, ns2)):
        # show counts from original row, not %
        row = ROWS[i]
        rawcounts = [row[6], row[7], row[8], row[9]]
        rawpcts = [a, c, rd, p]
        cursor = 0
        for k, (cnt, pct) in enumerate(zip(rawcounts, rawpcts)):
            if pct >= 12:
                color = 'white' if k != 3 else '#222'
                if i == label_row[k]:
                    ax.text(cursor + pct / 2, i, f"{cnt}\n{LABELS_B[k]}", va='center', ha='center', fontsize=9, color=color, weight='bold')
                else:
                    ax.text(cursor + pct / 2, i, f"{cnt}", va='center', ha='center', fontsize=11, color=color, weight='bold')
            cursor += pct
        ax.text(102, i, f"n={n}", va='center', ha='left', fontsize=11, color='#222', weight='bold')
    ax.set_yticks(ys)
    ax.set_yticklabels([])  # share with left panel
    ax.invert_yaxis()
    ax.set_xlim(0, 115)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("% of mixed-driver DPs", fontsize=13, fontweight='bold')
    ax.set_title("(b) Mixed-driver refinement", fontsize=14, fontweight='bold')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    plt.tight_layout()
    out = OUT_DIR / "fig_anatomy.pdf"
    plt.savefig(out, bbox_inches="tight")
    print("wrote", out)
    plt.close(fig)


if __name__ == "__main__":
    main()
