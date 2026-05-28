"""Pick representative case studies for §5.4.

For each of {single, mixed, no} pick one decision point and dump:
  - sender messages (truncated)
  - per-message F^KL excess (mean only; we don't have CI per msg in records)
  - recipient's reasoning trace base vs intervened (truncated)

Also makes a small bar-chart figure paper/figures/fig_cases.pdf with 3 panels.
"""
from __future__ import annotations
import json
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
METRIC_DIR = ROOT / "data" / "pilot_b0" / "metrics"
OUT_DIR = ROOT / "paper" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CASES_TXT = ROOT / "paper" / "case_studies.txt"


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
    items = sorted(scores, reverse=True)
    pos = [v for v in items if v > 0]
    if len(pos)==0: return "no"
    if len(pos)==1: return "single"
    return "single" if pos[0] >= rho*max(pos[1],1e-9) else "mixed"


def main():
    # Use counterfactual intervention because it gives the strongest pragmatic-preserving signal
    target_iv = "counterfactual"
    candidates = {"single": [], "mixed": [], "no": []}

    for f in sorted(METRIC_DIR.glob(f"*_C_*_{target_iv}*.json")):
        env, arch, iv, tag = parse_cell(f.name)
        if iv != target_iv: continue
        if tag != "":  # use GPT-4o for case studies (default model)
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        # group by DP
        by_dp = defaultdict(list)
        for r in d.get("records", []):
            dp_key = (r["ep"], r["recipient"], r.get("phase",""))
            by_dp[dp_key].append(r)
        for dp_key, recs in by_dp.items():
            if len(recs) < 3: continue  # need |M|>=3 for interesting case
            scores = [(r["sender"], r["fs_kl_excess_fine"]["fs_kl_excess"], r) for r in recs]
            scores.sort(key=lambda x: x[1], reverse=True)
            cls = classify([s for _,s,_ in scores])
            # quality criteria:
            # single: top1 >> top2 (large gap)
            # mixed: top1, top2 both positive, gap small
            # no: all scores negative or near zero
            top1 = scores[0][1]; top2 = scores[1][1] if len(scores)>1 else -1e9
            if cls == "single" and top1 > 0.3 and (top1 - top2) > 0.2:
                candidates["single"].append((env, arch, dp_key, scores, top1 - top2))
            elif cls == "mixed" and top1 > 0.15 and top2 > 0.15:
                candidates["mixed"].append((env, arch, dp_key, scores, top1 - top2))
            elif cls == "no" and all(s < -0.02 for _,s,_ in scores):
                candidates["no"].append((env, arch, dp_key, scores, abs(scores[-1][1])))

    print("# Candidates per mode:")
    for k, v in candidates.items():
        print(f"  {k}: {len(v)} candidates")

    # Pick the most extreme one per mode
    picks = {}
    for mode in ["single", "mixed", "no"]:
        if not candidates[mode]:
            print(f"  WARNING: no candidate for {mode}")
            continue
        # pick the one with largest gap (or for 'no', most negative)
        picks[mode] = max(candidates[mode], key=lambda x: x[-1])

    # Write text dump
    out_lines = []
    for mode in ["single", "mixed", "no"]:
        if mode not in picks: continue
        env, arch, dp_key, scores, gap = picks[mode]
        out_lines.append(f"\n=== {mode.upper()}-DRIVER CASE ===")
        out_lines.append(f"env={env} arch={arch} intervention={target_iv}")
        out_lines.append(f"DP=(ep={dp_key[0]}, recipient={dp_key[1]}, phase={dp_key[2]})")
        out_lines.append(f"  messages (sorted by F^KL):")
        for sender, score, rec in scores:
            content = rec["content"].replace("\n", " ")[:120]
            out_lines.append(f"    [{score:+.3f}] {sender}: {content}")
        # show trace_base of the top-1 record (truncated)
        top_rec = scores[0][2]
        tb = top_rec.get("trace_base","")[:300].replace("\n"," ")
        ti = top_rec.get("trace_intv","")[:300].replace("\n"," ")
        out_lines.append(f"  trace_base (top sender intervened): {tb}")
        out_lines.append(f"  trace_intv (top sender intervened): {ti}")

    txt = "\n".join(out_lines)
    print(txt)
    CASES_TXT.write_text(txt, encoding="utf-8")
    print(f"\nwrote {CASES_TXT}")

    # Plot: 3 panels bar chart
    fig, axes = plt.subplots(1, 3, figsize=(9, 2.7))
    titles = {"single": "Single-driver", "mixed": "Mixed-driver", "no": "No-driver"}
    for ax, mode in zip(axes, ["single","mixed","no"]):
        if mode not in picks: continue
        env, arch, dp_key, scores, _ = picks[mode]
        senders = [s[0] for s in scores]
        vals = [s[1] for s in scores]
        colors = ['#2b7a3d' if v>0.05 else ('#e08a00' if v>0 else '#b03030') for v in vals]
        ax.barh(range(len(senders)), vals, color=colors)
        ax.set_yticks(range(len(senders)))
        ax.set_yticklabels(senders, fontsize=8)
        ax.axvline(0, color='k', linewidth=0.6)
        ax.set_xlabel(r"$F^{\mathrm{KL}}$ excess", fontsize=9)
        ax.set_title(f"{titles[mode]} ({env}/{arch})\nep={dp_key[0]} recip={dp_key[1]}", fontsize=8)
        ax.invert_yaxis()
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    out_path = OUT_DIR / "fig_cases.pdf"
    plt.savefig(out_path, bbox_inches="tight")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
