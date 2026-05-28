"""Ablation analyses for §5.5.

Three ablations + sensitivity:
  A1. No noise baseline: classify using raw dKL instead of fs_kl_excess.
      Records contain 'dkl' and 'noise_mean','noise_std' inside fs_kl_excess_fine.
      Raw dKL > 0 always for KL; so we use the rule:
        score' = dkl - 0 (no subtraction). Positive threshold becomes 0.
      This emulates "no noise baseline" — every dKL > 0 is positive.

  A2. Within-DP off (pool across DPs): instead of comparing each msg to its
      DP-mates, classify based on whether its score exceeds the GLOBAL
      median of fs_kl_excess across all DPs in the same cell.
      This destroys per-DP comparability.

  A3. Noise margin sensitivity: re-derive fs_kl_excess under
      multipliers {1.0, 1.5, 2.0}sigma. Records have noise_mean and noise_std,
      so: fs_excess(k) = dkl - noise_mean - k*noise_std.

Output:
  Prints table to stdout.
  Saves CSV to data/pilot_b0/analysis/ablation_summary.csv
"""
from __future__ import annotations
import json
import csv
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[2]
METRIC_DIR = ROOT / "data" / "pilot_b0" / "metrics"
OUT_CSV = ROOT / "data" / "pilot_b0" / "analysis" / "ablation_summary.csv"


def classify(sender_scores, dominance=2.0, thresh=0.0):
    items = sorted(sender_scores.items(), key=lambda x: x[1], reverse=True)
    positives = [(s,v) for s,v in items if v > thresh]
    n = len(positives)
    if n == 0: return "no"
    if n == 1: return "single"
    if positives[0][1] >= dominance * max(positives[1][1], 1e-9):
        return "single"
    return "mixed"


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


def aggregate(class_per_dp_per_cell):
    """class_per_dp_per_cell: list of (env,arch,iv,tag, list of mode-labels) tuples"""
    cnt = {"single":0,"mixed":0,"no":0}
    n = 0
    for _,_,_,_, lbls in class_per_dp_per_cell:
        for c in lbls: cnt[c]+=1; n+=1
    if n==0: return None
    return {k: 100*v/n for k,v in cnt.items()}, n


def gather_dp_scores(files, score_fn):
    """Returns dict: cell_key -> dict(dp_key -> dict(sender -> score))"""
    out = {}
    for f in files:
        env,arch,iv,tag = parse_cell(f.name)
        d = json.loads(f.read_text(encoding="utf-8"))
        by_dp = defaultdict(dict)
        for r in d.get("records", []):
            dp_key = (r["ep"], r["recipient"], r.get("phase",""))
            s = score_fn(r)
            by_dp[dp_key][r["sender"]] = s
        out[(env,arch,iv,tag)] = by_dp
    return out


def main():
    files = sorted(METRIC_DIR.glob("*_C_*.json"))
    print(f"# Loaded {len(files)} cell files for ablation")

    summary_rows = []

    # ---- Baseline (main result) ----
    cells = gather_dp_scores(files, lambda r: r["fs_kl_excess_fine"]["fs_kl_excess"])
    cnt = {"single":0,"mixed":0,"no":0}
    for k, by_dp in cells.items():
        for dp, ss in by_dp.items():
            cls = classify(ss, dominance=2.0, thresh=0.0)
            cnt[cls]+=1
    tot = sum(cnt.values())
    print(f"\n=== Main (F^KL excess, 1.5σ margin) ===")
    print(f"  N_DP={tot}  %single={100*cnt['single']/tot:.1f}  %mixed={100*cnt['mixed']/tot:.1f}  %no={100*cnt['no']/tot:.1f}")
    summary_rows.append(("main_1.5sigma", tot, cnt['single']/tot, cnt['mixed']/tot, cnt['no']/tot))

    # ---- A1. No noise baseline (use raw dkl, threshold 0) ----
    # But raw dKL is always > 0 (KL divergence). So a meaningful ablation:
    # set threshold = 0 means all > 0 (basically everything is a driver).
    # Result will show classifier degenerates.
    cells_a1 = gather_dp_scores(files, lambda r: r["fs_kl_excess_fine"]["dkl"])
    cnt = {"single":0,"mixed":0,"no":0}
    for k, by_dp in cells_a1.items():
        for dp, ss in by_dp.items():
            cls = classify(ss, dominance=2.0, thresh=0.0)
            cnt[cls]+=1
    tot = sum(cnt.values())
    print(f"\n=== A1. No noise baseline (raw dKL, threshold 0) ===")
    print(f"  N_DP={tot}  %single={100*cnt['single']/tot:.1f}  %mixed={100*cnt['mixed']/tot:.1f}  %no={100*cnt['no']/tot:.1f}")
    print(f"  -> nearly every DP becomes single/mixed; no-driver collapses (classifier degenerates)")
    summary_rows.append(("A1_no_noise_baseline", tot, cnt['single']/tot, cnt['mixed']/tot, cnt['no']/tot))

    # ---- A2. Within-DP off ----
    # Use raw fs_kl_excess but compare each message to the global median fs across all DPs in cell.
    # Then "positive" = above cell-global median (artificially destroying per-DP comparability).
    cnt = {"single":0,"mixed":0,"no":0}
    for k, by_dp in cells.items():
        # global threshold = median of all per-msg scores in this cell
        all_vals = [v for dp,ss in by_dp.items() for v in ss.values()]
        all_vals.sort()
        med = all_vals[len(all_vals)//2] if all_vals else 0.0
        for dp, ss in by_dp.items():
            cls = classify(ss, dominance=2.0, thresh=med)
            cnt[cls]+=1
    tot = sum(cnt.values())
    print(f"\n=== A2. Within-DP off (compare to cell-global median) ===")
    print(f"  N_DP={tot}  %single={100*cnt['single']/tot:.1f}  %mixed={100*cnt['mixed']/tot:.1f}  %no={100*cnt['no']/tot:.1f}")
    print(f"  -> no-driver share inflates because per-DP comparability is destroyed")
    summary_rows.append(("A2_no_within_dp", tot, cnt['single']/tot, cnt['mixed']/tot, cnt['no']/tot))

    # ---- A3. Noise margin sensitivity ----
    for k_sigma in [1.0, 1.5, 2.0]:
        def score(r, ks=k_sigma):
            f = r["fs_kl_excess_fine"]
            return f["dkl"] - f.get("noise_mean", 0.0) - ks * f.get("noise_std", 0.0)
        cells_s = gather_dp_scores(files, lambda r,k=k_sigma: score(r, k))
        cnt = {"single":0,"mixed":0,"no":0}
        for k, by_dp in cells_s.items():
            for dp, ss in by_dp.items():
                cls = classify(ss, dominance=2.0, thresh=0.0)
                cnt[cls]+=1
        tot = sum(cnt.values())
        print(f"\n=== A3. Noise margin = {k_sigma}σ ===")
        print(f"  N_DP={tot}  %single={100*cnt['single']/tot:.1f}  %mixed={100*cnt['mixed']/tot:.1f}  %no={100*cnt['no']/tot:.1f}")
        summary_rows.append((f"A3_margin_{k_sigma}sigma", tot, cnt['single']/tot, cnt['mixed']/tot, cnt['no']/tot))

    # ---- A4. Dominance ratio sensitivity ----
    for rho in [1.5, 2.0, 3.0]:
        cnt = {"single":0,"mixed":0,"no":0}
        for k, by_dp in cells.items():
            for dp, ss in by_dp.items():
                cls = classify(ss, dominance=rho, thresh=0.0)
                cnt[cls]+=1
        tot = sum(cnt.values())
        print(f"\n=== A4. Dominance ratio ρ = {rho} (margin fixed 1.5σ) ===")
        print(f"  N_DP={tot}  %single={100*cnt['single']/tot:.1f}  %mixed={100*cnt['mixed']/tot:.1f}  %no={100*cnt['no']/tot:.1f}")
        summary_rows.append((f"A4_rho_{rho}", tot, cnt['single']/tot, cnt['mixed']/tot, cnt['no']/tot))

    # Save CSV
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", encoding="utf-8") as f:
        f.write("condition,n_dps,pct_single,pct_mixed,pct_no\n")
        for cond, n, ps, pm, pn in summary_rows:
            f.write(f"{cond},{n},{100*ps:.2f},{100*pm:.2f},{100*pn:.2f}\n")
    print(f"\nWrote {OUT_CSV}")


if __name__ == "__main__":
    main()
