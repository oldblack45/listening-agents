"""Cross-intervention agreement for §5.5.

For each DP we have 4 intervention realizations (identity, fact_replace,
counterfactual, random_string). Identity is the calibration lower bound and
does not produce drivers; we use the 3 informative interventions:
fact_replace, counterfactual, random_string.

For each DP, compute the mode label under each of the 3 interventions and
report:
  - pairwise agreement rate (% of DPs where two interventions agree)
  - 3-way agreement rate (% where all 3 agree)
  - majority-vote vs single-intervention disagreement
"""
from __future__ import annotations
import json
from pathlib import Path
from collections import defaultdict, Counter

ROOT = Path(__file__).resolve().parents[2]
METRIC_DIR = ROOT / "data" / "pilot_b0" / "metrics"
OUT_CSV = ROOT / "data" / "pilot_b0" / "analysis" / "intervention_agreement.csv"


def parse_cell(fname):
    stem = Path(fname).stem
    parts = stem.split("_")
    env = parts[0]; arch = parts[2]; rest = parts[3:]
    KNOWN_TAGS = {"haiku","gemini","sonnet","opus","gpt5"}
    tag = ""
    if rest and rest[-1] in KNOWN_TAGS:
        tag = rest[-1]; rest = rest[:-1]
    iv = "_".join(rest)
    return env, arch, iv, tag


def classify(scores, rho=2.0):
    """scores: list of (sender, fs_kl_excess)."""
    items = sorted([v for _,v in scores], reverse=True)
    pos = [v for v in items if v > 0]
    if len(pos)==0: return "no"
    if len(pos)==1: return "single"
    return "single" if pos[0] >= rho*max(pos[1],1e-9) else "mixed"


def main():
    INFO_IVS = ["fact_replace", "counterfactual", "random_string"]
    # DP_key -> {iv: mode}
    dp_modes = defaultdict(dict)
    files = sorted(METRIC_DIR.glob("*_C_*.json"))
    for f in files:
        env, arch, iv, tag = parse_cell(f.name)
        if iv not in INFO_IVS: continue
        d = json.loads(f.read_text(encoding="utf-8"))
        # group by DP
        by_dp = defaultdict(list)
        for r in d.get("records", []):
            dp_key = (env, arch, tag, r["ep"], r["recipient"], r.get("phase",""))
            by_dp[dp_key].append((r["sender"], r["fs_kl_excess_fine"]["fs_kl_excess"]))
        for dp_key, sender_scores in by_dp.items():
            dp_modes[dp_key][iv] = classify(sender_scores)

    # Filter DPs with all 3 interventions
    full = {k: v for k, v in dp_modes.items() if len(v) == 3}
    print(f"# Total DPs with all 3 interventions: {len(full)}")

    # 3-way agreement
    n3 = 0
    for k, modes in full.items():
        if len(set(modes.values())) == 1:
            n3 += 1
    print(f"# 3-way agreement: {n3}/{len(full)} = {100*n3/len(full):.1f}%")

    # Pairwise agreement
    pairs = [("fact_replace", "counterfactual"),
             ("fact_replace", "random_string"),
             ("counterfactual", "random_string")]
    for a, b in pairs:
        n = sum(1 for k, m in full.items() if m[a] == m[b])
        print(f"# {a} vs {b} agreement: {n}/{len(full)} = {100*n/len(full):.1f}%")

    # Majority vs each single
    n_maj_eq = {iv: 0 for iv in INFO_IVS}
    for k, modes in full.items():
        c = Counter(modes.values())
        majority, cnt = c.most_common(1)[0]
        for iv in INFO_IVS:
            if modes[iv] == majority:
                n_maj_eq[iv] += 1
    print()
    for iv in INFO_IVS:
        print(f"# {iv} matches majority: {n_maj_eq[iv]}/{len(full)} = {100*n_maj_eq[iv]/len(full):.1f}%")

    # Per-cell agreement breakdown
    cell_stats = defaultdict(lambda: {"n": 0, "agree3": 0})
    for k, modes in full.items():
        env, arch, tag, ep, recip, phase = k
        cell = (env, arch, tag)
        cell_stats[cell]["n"] += 1
        if len(set(modes.values())) == 1:
            cell_stats[cell]["agree3"] += 1
    print("\n# Per-cell 3-way agreement:")
    for cell, st in sorted(cell_stats.items()):
        print(f"  {cell}: {st['agree3']}/{st['n']} = {100*st['agree3']/max(st['n'],1):.1f}%")

    # CSV
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", encoding="utf-8") as f:
        f.write("env,arch,model_tag,n_dps,n_3way_agree,pct_3way_agree\n")
        for cell, st in sorted(cell_stats.items()):
            env, arch, tag = cell
            f.write(f"{env},{arch},{tag or 'gpt4o'},{st['n']},{st['agree3']},"
                    f"{100*st['agree3']/max(st['n'],1):.2f}\n")
    print(f"\nWrote {OUT_CSV}")


if __name__ == "__main__":
    main()
