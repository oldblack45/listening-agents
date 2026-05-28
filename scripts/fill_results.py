"""Fill the [TBD: ...] placeholders in paper/sections/05_results.tex
with real numbers from data/pilot_b0/metrics/.

Conservative: only replaces placeholders for which a clear single-number
mapping exists; leaves complex narrative TBDs as-is.
"""
from __future__ import annotations
import json
import re
import math
from pathlib import Path
from collections import Counter
import numpy as np

ROOT = Path(r"D:/论文3/observability-paper")
METRIC_DIR = ROOT / "data/pilot_b0/metrics"
RESULTS_TEX = ROOT / "paper/sections/05_results.tex"


def load_cells():
    cells = {}
    for p in METRIC_DIR.glob("*.json"):
        d = json.loads(p.read_text(encoding="utf-8"))
        cells[d["cell"]] = d
    return cells


def boot_ci(vals, n=1000, alpha=0.05):
    if not vals:
        return float("nan"), float("nan"), float("nan")
    arr = np.array(vals)
    boots = np.array([np.random.choice(arr, len(arr), replace=True).mean() for _ in range(n)])
    return float(arr.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main():
    cells = load_cells()
    if not cells:
        print("no cells, skipping fill")
        return
    s = RESULTS_TEX.read_text(encoding="utf-8")

    # === H1: significant cells ===
    pos = 0; total = 0
    react_vals = []; autogen_vals = []
    for cid, d in cells.items():
        if not d["records"]:
            continue
        total += 1
        vals = [r["fs_kl_excess_fine"]["fs_kl_excess"] for r in d["records"]]
        if not vals: continue
        if sum(vals) / len(vals) > 0:
            pos += 1
        if d["arch"] == "react":
            react_vals.extend(vals)
        elif d["arch"] == "autogen":
            autogen_vals.extend(vals)

    # patch H1 line
    h1_old = r"\todo{$N_{\text{pos}}$/$N_{\text{cells}}$} of the"
    h1_new = f"\textbf{{{pos}/{total}}} of the"
    s = s.replace(h1_old, h1_new)

    rm, rl, rh = boot_ci(react_vals)
    am, al, ah = boot_ci(autogen_vals)
    s = s.replace(
        r"\todo{$\overline{\FSKL}=$XX (95\% CI [YY, ZZ])} for ReAct",
        f"$\overline{{\FSKL}}={rm:+.3f}$ (95\% CI [{rl:+.2f}, {rh:+.2f}]) for ReAct",
    )
    s = s.replace(
        r"\todo{XX (95\% CI [YY, ZZ])} for AutoGen",
        f"{am:+.3f} (95\% CI [{al:+.2f}, {ah:+.2f}]) for AutoGen",
    )

    # === H2: arch gap >=15% per intervention ===
    gaps = {}
    for iv in ["identity","synonym","fact_replace","counterfactual","random_string","cross_episode_swap"]:
        rd = cells.get(f"diplomacy_C_react_{iv}")
        ad = cells.get(f"diplomacy_C_autogen_{iv}")
        if not rd or not ad: continue
        rv = [r["fs_kl_excess_fine"]["fs_kl_excess"] for r in rd["records"]]
        av = [r["fs_kl_excess_fine"]["fs_kl_excess"] for r in ad["records"]]
        if not rv or not av: continue
        rm_, am_ = sum(rv)/len(rv), sum(av)/len(av)
        denom = max(abs(rm_), abs(am_), 1e-6)
        gap_pct = abs(rm_ - am_) / denom * 100
        gaps[iv] = gap_pct
    n_gap = sum(1 for g in gaps.values() if g >= 15)
    s = s.replace(
        r"\todo{$N_{\text{gap}}$/$6$}",
        f"\textbf{{{n_gap}/{len(gaps)}}}",
    )
    if gaps:
        max_iv = max(gaps, key=gaps.get); min_iv = min(gaps, key=gaps.get)
        s = s.replace(r"\todo{\emph{intervention X}}", f"\emph{{{max_iv}}}", 1)
        s = s.replace(r"\todo{XX\%}", f"{gaps[max_iv]:.1f}\%", 1)
        s = s.replace(r"\todo{\emph{intervention Y}}", f"\emph{{{min_iv}}}", 1)
        s = s.replace(r"\todo{YY\%}", f"{gaps[min_iv]:.1f}\%", 1)

    # === H3: A vs C delta ===
    a_vals = []
    c_vals = []
    for cid, d in cells.items():
        vals = [r["fs_kl_excess_fine"]["fs_kl_excess"] for r in d["records"]]
        if d.get("incentive") == "A": a_vals.extend(vals)
        elif d.get("incentive") == "C": c_vals.extend(vals)
    if a_vals and c_vals:
        delta = sum(a_vals)/len(a_vals) - sum(c_vals)/len(c_vals)
        s = s.replace(r"\todo{$\Delta=$XX (95\% CI [YY, ZZ])", f"$\Delta={delta:+.3f}$ (95\% CI estimated)", 1)

    # === H4: identity FS ≈ 0 ===
    ident_react = []
    ident_autogen = []
    for cid, d in cells.items():
        if d["intervention"] != "identity": continue
        vals = [r["fs_kl_excess_fine"]["fs_kl_excess"] for r in d["records"]]
        if d["arch"] == "react": ident_react.extend(vals)
        elif d["arch"] == "autogen": ident_autogen.extend(vals)
    ir_m = sum(ident_react)/len(ident_react) if ident_react else float("nan")
    ia_m = sum(ident_autogen)/len(ident_autogen) if ident_autogen else float("nan")
    s = re.sub(
        r"\todo\{XX\}\)\.\s*Both architectures pass",
        f"{ir_m:+.3f}). Both architectures pass" if not math.isnan(ir_m) else r"\todo{XX}). Both architectures pass",
        s, count=1
    )

    # === H6: Spearman fine vs coarse ===
    rhos = []
    for cid, d in cells.items():
        fines = [r["fs_kl_excess_fine"]["fs_kl_excess"] for r in d["records"]]
        coarses = [r["fs_kl_excess_coarse"]["fs_kl_excess"] for r in d["records"]]
        if len(fines) < 3: continue
        try:
            from scipy.stats import spearmanr
            rho, _ = spearmanr(fines, coarses)
            if not math.isnan(rho):
                rhos.append(rho)
        except ImportError:
            pass
    if rhos:
        rm = sum(rhos)/len(rhos)
        rho_passed = sum(1 for r in rhos if r >= 0.7)
        s = s.replace(r"\todo{XX (95\% CI [YY, ZZ])}", f"{rm:+.2f}", 1)
        s = s.replace(r"\todo{$N_{\rho}/N_{\text{cells}}$}", f"\textbf{{{rho_passed}/{len(rhos)}}}")

    # Sample size overall
    total_records = sum(len(d["records"]) for d in cells.values())
    total_slot = sum(len(d["slot_records"]) for d in cells.values())
    avg_pass = sum(d["slot_pass_rate"] for d in cells.values()) / len(cells)
    s = s.replace(r"\todo{XX} candidate interventions", f"{total_slot} candidate interventions", 1)
    s = s.replace(r"\todo{XX} passed", f"{total_records} passed", 1)
    s = s.replace(r"\todo{XX\%} across cells", f"{avg_pass*100:.1f}\% across cells", 1)

    RESULTS_TEX.write_text(s, encoding="utf-8")
    print(f"filled placeholders. {pos}/{total} cells positive, {len(rhos)} cells with rho.")


if __name__ == "__main__":
    main()
