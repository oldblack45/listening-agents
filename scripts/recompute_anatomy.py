"""Restrict e2 (no-driver all-mask) and e3 (mixed pairwise) anatomy to the
DP set classified by the attribution operators (fact_replace + counterfactual,
GPT-4o + Haiku, no other backbones).

A DP is taken in the no-driver subset iff at least one attribution operator
labels it no-driver on its full sender vector under (margin=1.5sigma, rho=2).
Similarly for mixed.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "code" / "scripts"))

from run_informative_only import (
    parse_name, classify, INFORMATIVE, ALLOWED_TAGS,
)

METRIC = ROOT / "data" / "pilot_b0" / "metrics"
ANALYSIS = ROOT / "data" / "pilot_b0" / "analysis"


def build_label_index():
    """Return {(env, arch, tag, ep, recipient, phase): {iv: label}}"""
    labels = defaultdict(dict)
    files = list(METRIC.glob("*_C_*counterfactual*.json")) + \
            list(METRIC.glob("*_C_*fact_replace*.json"))
    for f in files:
        env, inc, arch, iv, tag = parse_name(f.name)
        if iv not in INFORMATIVE or tag not in ALLOWED_TAGS:
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        by_dp = defaultdict(dict)
        for r in d.get("records", []):
            base = (env, arch, tag, r["ep"], r["recipient"], r.get("phase",""))
            by_dp[base][r["sender"]] = r["fs_kl_excess_fine"]["fs_kl_excess"]
        for base, scores in by_dp.items():
            labels[base][iv] = classify(scores, rho=2.0)
    return labels


def is_no_under_attribution(L):
    """DP counts as no-driver if any attribution operator labels it no."""
    return any(v == "no" for v in L.values())


def is_mixed_under_attribution(L):
    return any(v == "mixed" for v in L.values())


def normalize_tag(model):
    """e2/e3 'model' field is 'gpt4o' or 'haiku'; metric tag is '' or 'haiku'."""
    return "" if model == "gpt4o" else model


def load_results(prefix):
    out = []
    for f in sorted(ANALYSIS.glob(f"{prefix}_*.json")):
        if f.stem.endswith("_results"):
            continue
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        if "results" not in d:
            continue
        for r in d["results"]:
            out.append(r)
    return out


def main():
    labels = build_label_index()
    no_keys    = {k for k, L in labels.items() if is_no_under_attribution(L)}
    mixed_keys = {k for k, L in labels.items() if is_mixed_under_attribution(L)}
    print(f"Informative-only no-driver DPs: {len(no_keys)}")
    print(f"Informative-only mixed-driver DPs: {len(mixed_keys)}")

    # ---- e2 (all-mask) ----
    e2 = load_results("e2_allmask")
    n_e2_in, n_diffuse = 0, 0
    by_env_e2 = defaultdict(lambda: [0, 0])  # [in, diffuse]
    for r in e2:
        key = (r["env"], r["arch"], normalize_tag(r["model"]),
               r["ep"], r["recipient"], r.get("phase", ""))
        if key not in no_keys:
            continue
        n_e2_in += 1
        by_env_e2[r["env"]][0] += 1
        if r["is_diffuse"]:
            n_diffuse += 1
            by_env_e2[r["env"]][1] += 1
    auto = n_e2_in - n_diffuse
    print(f"\n=== E2 (no-driver all-mask) on informative-only subset ===")
    print(f"  total analyzed: {n_e2_in}")
    if n_e2_in:
        print(f"  autonomous: {auto} ({100*auto/n_e2_in:.1f}%)")
        print(f"  diffuse   : {n_diffuse} ({100*n_diffuse/n_e2_in:.1f}%)")
    for env, (tot, diff) in by_env_e2.items():
        a = tot - diff
        if tot:
            print(f"  {env}: n={tot}  autonomous={a} ({100*a/tot:.1f}%)  diffuse={diff} ({100*diff/tot:.1f}%)")

    # ---- e3 (mixed pairwise) ----
    e3 = load_results("e3_pairwise")
    cats = defaultdict(int)
    n_e3_in = 0
    by_env_e3 = defaultdict(lambda: defaultdict(int))
    for r in e3:
        key = (r["env"], r["arch"], normalize_tag(r["model"]),
               r["ep"], r["recipient"], r.get("phase", ""))
        if key not in mixed_keys:
            continue
        n_e3_in += 1
        cats[r["interaction"]] += 1
        by_env_e3[r["env"]][r["interaction"]] += 1
    print(f"\n=== E3 (mixed pairwise) on informative-only subset ===")
    print(f"  total analyzed: {n_e3_in}")
    for c, v in sorted(cats.items(), key=lambda x: -x[1]):
        if n_e3_in:
            print(f"  {c:18s} {v:3d} ({100*v/n_e3_in:.1f}%)")
    for env, cd in by_env_e3.items():
        tot = sum(cd.values())
        print(f"  {env} (n={tot}):")
        for c, v in sorted(cd.items(), key=lambda x: -x[1]):
            print(f"    {c:18s} {v:3d} ({100*v/tot:.1f}%)")

    # save
    OUT = ANALYSIS / "anatomy_informative_only.json"
    OUT.write_text(json.dumps({
        "n_no_keys": len(no_keys),
        "n_mixed_keys": len(mixed_keys),
        "e2_n_analyzed": n_e2_in,
        "e2_autonomous": auto if n_e2_in else 0,
        "e2_diffuse": n_diffuse,
        "e2_by_env": {k: list(v) for k, v in by_env_e2.items()},
        "e3_n_analyzed": n_e3_in,
        "e3_categories": dict(cats),
        "e3_by_env": {k: dict(v) for k, v in by_env_e3.items()},
    }, indent=2), encoding="utf-8")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
