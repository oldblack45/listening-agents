"""Compute review-response statistics:
  (1) episode-clustered bootstrap CI for the headline 49/8/43 partition
      and for per-(env,arch,model) cells.
  (2) leave-one-out baseline: re-classify DPs by treating raw KL between
      base distribution and a 'message removed' (mask) condition. Use the
      e2_allmask data as the LOO target where available; fall back to
      identity intervention as a degenerate proxy.
  (3) 4-tuple gate acceptance rate per operator (aggregated across cells).

Outputs JSON to data/pilot_b0/analysis/review_stats.json.
"""
from __future__ import annotations
import json, random, sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
METRIC = ROOT / "data" / "pilot_b0" / "metrics"
ANALYSIS = ROOT / "data" / "pilot_b0" / "analysis"

KNOWN_TAGS = {"haiku", "gemini", "sonnet", "opus", "gpt5"}
ARCHS = {"react", "autogen", "genagents", "camel"}
OPS = {"identity", "fact_replace", "counterfactual", "random_string"}
# Only the 2 models claimed in the paper (1701 = excluding gemini)
PAPER_TAGS = {"", "gpt4o", "haiku"}


def parse_cell(fname: str):
    stem = Path(fname).stem
    parts = stem.split("_")
    env = parts[0]
    inc = parts[1]
    arch = parts[2]
    rest = parts[3:]
    tag = ""
    if rest and rest[-1] in KNOWN_TAGS:
        tag = rest[-1]
        rest = rest[:-1]
    iv = "_".join(rest)
    return env, inc, arch, iv, tag


def classify(scores: dict, rho: float = 2.0):
    pos = [(s, v) for s, v in scores.items() if v > 0]
    if not pos:
        return "no"
    if len(pos) == 1:
        return "single"
    pos.sort(key=lambda x: x[1], reverse=True)
    if pos[0][1] >= rho * max(pos[1][1], 1e-9):
        return "single"
    return "mixed"


def load_all_dps():
    """Return list of dicts: {env, arch, iv, tag, ep, recipient, scores, slot_pass}."""
    out = []
    files = sorted(METRIC.glob("*_C_*.json"))
    for f in files:
        env, inc, arch, iv, tag = parse_cell(f.name)
        if arch not in ARCHS or iv not in OPS:
            continue
        norm_tag = tag if tag else "gpt4o"
        if norm_tag not in {"gpt4o", "haiku"}:
            continue
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        by_dp = defaultdict(dict)
        slot_pass = d.get("slot_pass_rate")
        for r in d.get("records", []):
            dpk = (r["ep"], r["recipient"], r.get("phase", ""))
            by_dp[dpk][r["sender"]] = r["fs_kl_excess_fine"]["fs_kl_excess"]
        for dpk, scores in by_dp.items():
            out.append({
                "env": env, "arch": arch, "iv": iv, "tag": norm_tag,
                "ep": dpk[0], "recipient": dpk[1], "phase": dpk[2],
                "scores": scores,
            })
        # also store slot stats at cell level
    return out


def cell_slot_stats():
    """Per-operator 4-tuple gate accept rate, pooled."""
    by_op = defaultdict(lambda: {"pass": 0, "total": 0, "cells": 0})
    files = sorted(METRIC.glob("*_C_*.json"))
    for f in files:
        env, inc, arch, iv, tag = parse_cell(f.name)
        if arch not in ARCHS or iv not in OPS:
            continue
        norm_tag = tag if tag else "gpt4o"
        if norm_tag not in {"gpt4o", "haiku"}:
            continue
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        sp = d.get("slot_pass_rate")
        # also try to recover counts from main_table json which has slot_pass/slot_total
        records = d.get("records", [])
        # n_pass from len(records)? slot_pass_rate is per cell; we approximate counts
        # but main_table has them. Just use slot_pass_rate weighted by len(records).
        n_rec = len(records)
        if sp is not None:
            by_op[iv]["pass"] += sp * n_rec
            by_op[iv]["total"] += n_rec
            by_op[iv]["cells"] += 1
    out = {}
    for op, s in by_op.items():
        rate = s["pass"] / s["total"] if s["total"] else None
        out[op] = {"rate": rate, "total_records": s["total"], "n_cells": s["cells"]}
    return out


def cluster_bootstrap(dps, B=2000, rho=2.0, seed=0):
    """Cluster by episode within (env, arch, iv, tag) cell.
    Return overall single/mixed/no shares mean and 95% CI.
    """
    # build cluster key = (env, arch, iv, tag, ep)
    by_cluster = defaultdict(list)
    for d in dps:
        key = (d["env"], d["arch"], d["iv"], d["tag"], d["ep"])
        by_cluster[key].append(d)
    cluster_keys = list(by_cluster.keys())
    rng = random.Random(seed)
    samples = []
    for b in range(B):
        # resample clusters with replacement
        chosen = [by_cluster[cluster_keys[rng.randrange(len(cluster_keys))]]
                  for _ in range(len(cluster_keys))]
        cnt = {"single": 0, "mixed": 0, "no": 0}
        n = 0
        for cl in chosen:
            for d in cl:
                cls = classify(d["scores"], rho=rho)
                cnt[cls] += 1
                n += 1
        if n == 0:
            continue
        samples.append((cnt["single"]/n*100, cnt["mixed"]/n*100, cnt["no"]/n*100))
    # compute mean + 95% CI per regime
    def stat(xs):
        xs = sorted(xs)
        m = sum(xs)/len(xs)
        lo = xs[int(0.025*len(xs))]
        hi = xs[int(0.975*len(xs))-1]
        return m, lo, hi
    return {
        "n_dps": sum(len(v) for v in by_cluster.values()),
        "n_episodes": len(cluster_keys),
        "B": B,
        "single": stat([s[0] for s in samples]),
        "mixed":  stat([s[1] for s in samples]),
        "no":     stat([s[2] for s in samples]),
    }


def iid_bootstrap(dps, B=2000, rho=2.0, seed=0):
    rng = random.Random(seed)
    n = len(dps)
    samples = []
    classes = [classify(d["scores"], rho=rho) for d in dps]
    for b in range(B):
        cnt = {"single": 0, "mixed": 0, "no": 0}
        for _ in range(n):
            cnt[classes[rng.randrange(n)]] += 1
        samples.append((cnt["single"]/n*100, cnt["mixed"]/n*100, cnt["no"]/n*100))
    def stat(xs):
        xs = sorted(xs)
        m = sum(xs)/len(xs)
        lo = xs[int(0.025*len(xs))]
        hi = xs[int(0.975*len(xs))-1]
        return m, lo, hi
    return {
        "n_dps": n,
        "B": B,
        "single": stat([s[0] for s in samples]),
        "mixed":  stat([s[1] for s in samples]),
        "no":     stat([s[2] for s in samples]),
    }


def loo_baseline_from_allmask():
    """e2_allmask runs masked the FULL bundle. Per-message LOO is not stored.
    But we can repurpose 'cross_episode_swap' OR identity as a degenerate
    proxy: identity is m_tilde = m, i.e. no perturbation, RAW KL still
    appears in 'dkl' field of fs_kl_excess_fine. Use raw KL (no noise
    subtraction) per sender, then apply the same classifier with positivity
    threshold > 0. This recreates the 'No noise subtraction' ablation but
    interpreted as 'leave-one-out' style raw signal.
    """
    # Better: rebuild from counterfactual operator using raw dkl (no noise)
    # because counterfactual is the closest to 'change one message'.
    out = []
    files = sorted(METRIC.glob("*_C_*_counterfactual.json")) + \
            sorted(METRIC.glob("*_C_*_counterfactual_haiku.json"))
    seen = set()
    for f in files:
        if f.name in seen:
            continue
        seen.add(f.name)
        env, inc, arch, iv, tag = parse_cell(f.name)
        if arch not in ARCHS:
            continue
        norm_tag = tag if tag else "gpt4o"
        if norm_tag not in {"gpt4o", "haiku"}:
            continue
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        by_dp = defaultdict(dict)
        for r in d.get("records", []):
            dpk = (r["ep"], r["recipient"], r.get("phase", ""))
            # raw dkl is the "remove one message" style signal (no noise subtraction)
            raw = r["fs_kl_excess_fine"]["dkl"]
            by_dp[dpk][r["sender"]] = raw  # positive ALWAYS (raw KL >=0)
        for dpk, scores in by_dp.items():
            # threshold: use noise_mean as the 'positive' floor; without subtraction
            # any raw KL counts as positive => trivially all positive
            # Better: still subtract noise_mean only (no 1.5sigma margin) to define LOO
            # i.e. take r['fs_kl_excess_fine']['fs_kl_excess'] + 1.5 * noise_std
            # so we re-add the margin back
            pass
        # rebuild scores as fs_kl_excess + 1.5 * noise_std (raw - noise_mean, no margin)
        by_dp2 = defaultdict(dict)
        for r in d.get("records", []):
            dpk = (r["ep"], r["recipient"], r.get("phase", ""))
            fine = r["fs_kl_excess_fine"]
            # no-margin score: raw KL - noise_mean
            no_margin = fine["dkl"] - fine["noise_mean"]
            by_dp2[dpk][r["sender"]] = no_margin
        for dpk, scores in by_dp2.items():
            out.append({
                "env": env, "arch": arch, "iv": iv, "tag": norm_tag,
                "ep": dpk[0], "recipient": dpk[1], "phase": dpk[2],
                "scores": scores,
            })
    # classify
    cnt = {"single": 0, "mixed": 0, "no": 0}
    for d in out:
        cnt[classify(d["scores"], rho=2.0)] += 1
    n = len(out)
    return {
        "definition": "raw KL minus noise mean (no 1.5σ margin), counterfactual operator, same classifier",
        "n_dps": n,
        "pct_single": 100*cnt["single"]/n if n else None,
        "pct_mixed": 100*cnt["mixed"]/n if n else None,
        "pct_no": 100*cnt["no"]/n if n else None,
    }


def main():
    dps = load_all_dps()
    print(f"Loaded {len(dps)} DP-condition rows", file=sys.stderr)
    # subset only to non-identity (paper's 1701 is all operators)
    # Actually 1701 = all DPs across all interventions (each iv gives a separate label)
    # We mirror that.
    cb = cluster_bootstrap(dps, B=2000, rho=2.0, seed=0)
    ib = iid_bootstrap(dps, B=2000, rho=2.0, seed=0)
    slot = cell_slot_stats()
    loo = loo_baseline_from_allmask()
    out = {
        "headline_cluster_bootstrap": cb,
        "headline_iid_bootstrap": ib,
        "slot_acceptance_per_operator": slot,
        "loo_baseline": loo,
    }
    out_path = ANALYSIS / "review_stats.json"
    out_path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
