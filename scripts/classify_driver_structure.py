"""Classify each decision point into {single-driver, mixed-driver, no-driver}
based on per-message F^KL excess scores.

Logic per DP = (cell, ep, recipient, phase):
  - Each incoming sender contributes one record per intervention (4 total).
  - We take F^KL excess (fine) for the chosen 'attribution intervention'.
  - F^KL > 0 means already exceeds noise floor at 1.5 sigma.

Classification rule (per intervention type, reported separately):
  Let signals = {sender : fs_kl_excess > 0}
  - no-driver: |signals| == 0
  - single-driver: |signals| == 1 OR
                   (|signals| >= 2 AND top1 >= 2 * top2 AND top1 - top2 > median_noise_std)
  - mixed-driver: otherwise (>=2 signals, top1 not dominant)

Outputs CSV summary by (model_tag, env, arch, intervention).
"""
from __future__ import annotations
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
METRIC_DIR = ROOT / "data" / "pilot_b0" / "metrics"


def parse_cell_name(fname: str):
    """diplomacy_C_react_random_string[_haiku].json -> (env, incentive, arch, iv, tag)"""
    stem = Path(fname).stem
    parts = stem.split("_")
    # env C arch iv (iv may be multi-token like 'random_string', 'fact_replace')
    # known archs: react autogen genagents camel
    # known ivs: identity fact_replace counterfactual random_string
    env = parts[0]
    incentive = parts[1]
    arch = parts[2]
    rest = parts[3:]
    # tag detection: last token in {haiku, gemini, sonnet, ...}
    KNOWN_TAGS = {"haiku", "gemini", "sonnet", "opus", "gpt5"}
    tag = ""
    if rest and rest[-1] in KNOWN_TAGS:
        tag = rest[-1]
        rest = rest[:-1]
    iv = "_".join(rest)
    return env, incentive, arch, iv, tag


def classify_dp(sender_scores: dict[str, float], dominance_ratio: float = 2.0):
    """sender_scores: {sender: fs_kl_excess}"""
    items = sorted(sender_scores.items(), key=lambda x: x[1], reverse=True)
    positives = [(s, v) for s, v in items if v > 0]
    n_pos = len(positives)
    if n_pos == 0:
        return "no", n_pos, items
    if n_pos == 1:
        return "single", n_pos, items
    top1_v = positives[0][1]
    top2_v = positives[1][1]
    # dominance test: top1 must be at least 2x top2 to be 'single'
    if top1_v >= dominance_ratio * max(top2_v, 1e-9):
        return "single", n_pos, items
    return "mixed", n_pos, items


def main():
    files = sorted(METRIC_DIR.glob("*_C_*.json"))
    print(f"Found {len(files)} C-condition metric files")

    # per-cell DP classification
    # key: (env, arch, iv, tag) -> list of (dp_key, class, n_pos, scores)
    cell_dps: dict[tuple, list] = defaultdict(list)

    for f in files:
        env, inc, arch, iv, tag = parse_cell_name(f.name)
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"  skip {f.name}: {e}")
            continue
        # group records by DP = (ep, recipient, phase)
        by_dp: dict[tuple, dict[str, float]] = defaultdict(dict)
        for r in d.get("records", []):
            dp_key = (r["ep"], r["recipient"], r.get("phase", ""))
            score = r["fs_kl_excess_fine"]["fs_kl_excess"]
            by_dp[dp_key][r["sender"]] = score
        for dp_key, sender_scores in by_dp.items():
            cls, n_pos, items = classify_dp(sender_scores)
            cell_dps[(env, arch, iv, tag)].append((dp_key, cls, n_pos, items))

    # aggregate to (env, arch, iv, tag) ratios
    print("\n=== Driver-structure ratios per (env, arch, intervention, model_tag) ===")
    header = f"{'env':10s} {'arch':10s} {'iv':16s} {'tag':7s} {'nDP':>4s} {'%single':>8s} {'%mixed':>8s} {'%no':>6s} {'avg|M|':>7s}"
    print(header)
    print("-" * len(header))
    rows = []
    for key in sorted(cell_dps.keys()):
        env, arch, iv, tag = key
        dps = cell_dps[key]
        n = len(dps)
        if n == 0:
            continue
        cnt = {"single": 0, "mixed": 0, "no": 0}
        m_sizes = []
        for _, cls, n_pos, items in dps:
            cnt[cls] += 1
            m_sizes.append(len(items))
        avg_m = sum(m_sizes) / len(m_sizes)
        row = (env, arch, iv, tag or "gpt4o", n,
               100 * cnt["single"] / n, 100 * cnt["mixed"] / n, 100 * cnt["no"] / n, avg_m)
        rows.append(row)
        print(f"{env:10s} {arch:10s} {iv:16s} {tag or 'gpt4o':7s} {n:4d} "
              f"{100*cnt['single']/n:8.1f} {100*cnt['mixed']/n:8.1f} {100*cnt['no']/n:6.1f} {avg_m:7.2f}")

    # aggregate over intervention (mean ratios) per (env, arch, tag)
    print("\n=== Aggregated across interventions: (env, arch, model_tag) ===")
    by_arch: dict[tuple, list] = defaultdict(list)
    for row in rows:
        env, arch, iv, tag, n, ps, pm, pn, am = row
        by_arch[(env, arch, tag)].append((n, ps, pm, pn, am))
    h2 = f"{'env':10s} {'arch':10s} {'tag':7s} {'%single':>8s} {'%mixed':>8s} {'%no':>6s} {'avg|M|':>7s}"
    print(h2)
    print("-" * len(h2))
    for key in sorted(by_arch.keys()):
        env, arch, tag = key
        vals = by_arch[key]
        # weight by n DPs (but each iv samples same DPs, so just mean is fine)
        ms = sum(v[1] for v in vals) / len(vals)
        mm = sum(v[2] for v in vals) / len(vals)
        mn = sum(v[3] for v in vals) / len(vals)
        am = sum(v[4] for v in vals) / len(vals)
        print(f"{env:10s} {arch:10s} {tag:7s} {ms:8.1f} {mm:8.1f} {mn:6.1f} {am:7.2f}")

    # also do agg over (env, arch) ignoring model tag
    print("\n=== Aggregated over interventions AND models: (env, arch) — average across models ===")
    by_arch_anymodel: dict[tuple, list] = defaultdict(list)
    for row in rows:
        env, arch, iv, tag, n, ps, pm, pn, am = row
        by_arch_anymodel[(env, arch)].append((ps, pm, pn, am))
    h3 = f"{'env':10s} {'arch':10s} {'%single':>8s} {'%mixed':>8s} {'%no':>6s} {'avg|M|':>7s}"
    print(h3)
    print("-" * len(h3))
    for key in sorted(by_arch_anymodel.keys()):
        env, arch = key
        vals = by_arch_anymodel[key]
        ms = sum(v[0] for v in vals) / len(vals)
        mm = sum(v[1] for v in vals) / len(vals)
        mn = sum(v[2] for v in vals) / len(vals)
        am = sum(v[3] for v in vals) / len(vals)
        print(f"{env:10s} {arch:10s} {ms:8.1f} {mm:8.1f} {mn:6.1f} {am:7.2f}")

    # save CSV
    out = ROOT / "data" / "pilot_b0" / "analysis" / "driver_structure_classification.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("env,arch,intervention,model_tag,n_dps,pct_single,pct_mixed,pct_no,avg_M\n")
        for row in rows:
            env, arch, iv, tag, n, ps, pm, pn, am = row
            f.write(f"{env},{arch},{iv},{tag},{n},{ps:.2f},{pm:.2f},{pn:.2f},{am:.2f}\n")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
