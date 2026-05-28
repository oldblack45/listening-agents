"""E3: pairwise ablation for mixed-driver DPs (Diplomacy + Sotopia, GPT-4o + Haiku).

For each mixed-driver DP, identify the top-2 senders by FS_KL_excess (from
random_string intervention records) and run:

  base         : all msgs intact
  drop_top1    : top-1 msg replaced with random_string
  drop_top2    : top-2 msg replaced with random_string
  drop_both    : both top-1 and top-2 msgs replaced with random_string

Then for each condition compute fs_excess against the noise floor:

  fs1 = FS_KL_excess(base, drop_top1, noise)
  fs2 = FS_KL_excess(base, drop_top2, noise)
  fs12 = FS_KL_excess(base, drop_both, noise)

Decision rule:
  - additive   if  fs12 >= 0.7 * (fs1 + fs2) AND fs12 > max(fs1, fs2)
  - competing  if  fs12 < min(fs1, fs2)  OR  fs12 < 0.7 * max(fs1, fs2)
  - redundant  if  fs12 ~ max(fs1, fs2)  (within 30%)  AND  min(fs1,fs2) > 0
  - other      otherwise

Outputs: data/pilot_b0/analysis/e3_pairwise_<env>_<model>.json

Usage:
  python code/scripts/run_e3_pairwise.py --env diplomacy --model gpt4o
"""
from __future__ import annotations
import argparse
import json
import random
import string
import sys
import time
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src import config as C
from src.metrics import noise_kl_samples, fs_kl_excess

METRIC_DIR = ROOT / "data" / "pilot_b0" / "metrics"
ANALYSIS_DIR = ROOT / "data" / "pilot_b0" / "analysis"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)


def classify(scores, rho=2.0):
    items = sorted(scores, reverse=True)
    pos = [v for v in items if v > 0]
    if len(pos) == 0:
        return "no"
    if len(pos) == 1:
        return "single"
    return "single" if pos[0] >= rho * max(pos[1], 1e-9) else "mixed"


def random_string_like(s, rng):
    chars = string.ascii_letters + string.digits + " ,.;:!?"
    return "".join(rng.choice(chars) for _ in range(len(s)))


def find_mixed_dps(env, model_tag):
    """Use random_string-derived classification to find mixed-driver DPs.
    Return each with top-2 sender info."""
    suffix = f"_{model_tag}" if model_tag else ""
    out = []
    for arch in C.ARCHS:
        path = METRIC_DIR / f"{env}_C_{arch}_random_string{suffix}.json"
        if not path.exists():
            continue
        d = json.loads(path.read_text(encoding="utf-8"))
        by_dp = defaultdict(list)
        for r in d["records"]:
            dp_key = (r["ep"], r["recipient"], r.get("phase", ""))
            by_dp[dp_key].append((r["sender"], r["fs_kl_excess_fine"]["fs_kl_excess"]))
        for dp_key, sender_scores in by_dp.items():
            cls = classify([s for _, s in sender_scores])
            if cls != "mixed":
                continue
            ranked = sorted(sender_scores, key=lambda x: x[1], reverse=True)
            if len(ranked) < 2 or ranked[1][1] <= 0:
                continue
            out.append({
                "arch": arch,
                "ep_i": dp_key[0],
                "recipient": dp_key[1],
                "phase": dp_key[2],
                "top1_sender": ranked[0][0],
                "top1_fs": ranked[0][1],
                "top2_sender": ranked[1][0],
                "top2_fs": ranked[1][1],
            })
    return out


def classify_interaction(fs1, fs2, fs12, eps=1e-6):
    """Classify the interaction type."""
    # Normalize negatives to 0 for ratio logic (still keep raw values).
    p1 = max(fs1, 0.0)
    p2 = max(fs2, 0.0)
    p12 = max(fs12, 0.0)
    s = p1 + p2
    mx = max(p1, p2)
    if p12 <= 0 and (p1 > 0 or p2 > 0):
        # dropping both shifts dist NO MORE than noise floor -> internal cancellation
        return "competing"
    if s <= eps:
        return "neither"
    if p12 < 0.7 * mx:
        return "competing"
    if abs(p12 - mx) / max(mx, eps) < 0.3 and min(p1, p2) > 0:
        return "redundant"
    if p12 >= 0.7 * s and p12 > mx:
        return "additive"
    return "partial"


def run_dp(env, dp, target_phase, scenario_mod, model_tag):
    arch = dp["arch"]
    ep_i = dp["ep_i"]
    recipient = dp["recipient"]
    top1_sender = dp["top1_sender"]
    top2_sender = dp["top2_sender"]

    play_until = scenario_mod.play_until_target_phase
    restore_inject = scenario_mod._restore_and_inject
    sample_traces = scenario_mod._sample_with_traces

    group_seed = 100 + C.ARCHS.index(arch) + 1
    seed = group_seed * 1000 + ep_i
    ep = play_until(arch, seed, "C", env, target_phase)
    if ep is None:
        return None
    all_msgs = ep["msgs_at_target"]

    # locate top1, top2 indices (first match by sender->recipient)
    def find_idx(sender):
        for i, m in enumerate(all_msgs):
            if m["recipient"] == recipient and m["sender"] == sender:
                return i
        return None
    idx1 = find_idx(top1_sender)
    idx2 = find_idx(top2_sender)
    if idx1 is None or idx2 is None or idx1 == idx2:
        return None

    rng = random.Random(seed * 41 + hash(recipient))

    # Base
    env_b = restore_inject(env, ep["snap_pre_press"], all_msgs)
    obs_base = env_b.extract_observation(recipient)
    base_fine, _, _ = sample_traces(
        arch, recipient, obs_base, n=C.ACTION_SAMPLES_PER_DO,
        seed_base=seed * 13 + 111, temperature=C.TEMPERATURE, incentive="C",
    )
    # Noise
    noise_fines = []
    for rep in range(3):
        nf, _, _ = sample_traces(
            arch, recipient, obs_base, n=C.ACTION_SAMPLES_PER_DO,
            seed_base=seed * 13 + 6000 + 100 * rep, temperature=C.TEMPERATURE, incentive="C",
        )
        noise_fines.append(nf)
    nk = noise_kl_samples(noise_fines, base_fine)

    def make_perturbed(drop_indices):
        out = []
        for i, m in enumerate(all_msgs):
            if i in drop_indices:
                out.append({
                    "sender": m["sender"], "recipient": m["recipient"],
                    "content": random_string_like(m["content"], rng),
                })
            else:
                out.append(m)
        return out

    def sample_cond(perturbed, salt):
        env_p = restore_inject(env, ep["snap_pre_press"], perturbed)
        obs_p = env_p.extract_observation(recipient)
        f, _, _ = sample_traces(
            arch, recipient, obs_p, n=C.ACTION_SAMPLES_PER_DO,
            seed_base=seed * 13 + salt, temperature=C.TEMPERATURE, incentive="C",
        )
        return f

    drop1_fine = sample_cond(make_perturbed({idx1}), 2001)
    drop2_fine = sample_cond(make_perturbed({idx2}), 3001)
    drop12_fine = sample_cond(make_perturbed({idx1, idx2}), 4001)

    fs1 = fs_kl_excess(base_fine, drop1_fine, nk)["fs_kl_excess"]
    fs2 = fs_kl_excess(base_fine, drop2_fine, nk)["fs_kl_excess"]
    fs12 = fs_kl_excess(base_fine, drop12_fine, nk)["fs_kl_excess"]
    interaction = classify_interaction(fs1, fs2, fs12)

    return {
        "env": env, "model": model_tag or "gpt4o",
        "arch": arch, "ep": ep_i, "recipient": recipient, "phase": target_phase,
        "top1_sender": top1_sender, "top2_sender": top2_sender,
        "fs_drop_top1": fs1,
        "fs_drop_top2": fs2,
        "fs_drop_both": fs12,
        "interaction": interaction,
        "noise_mean": float(sum(nk) / max(1, len(nk))),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", choices=["diplomacy", "sotopia"], required=True)
    ap.add_argument("--model", choices=["gpt4o", "haiku"], required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if args.model == "haiku":
        C.MODEL_TAG = "haiku"
        C.AGENT_MODEL = "claude-haiku-4.5"
    else:
        C.MODEL_TAG = ""
        C.AGENT_MODEL = "gpt-4o"
    print(f"[E3] env={args.env} model={args.model} AGENT_MODEL={C.AGENT_MODEL}", flush=True)

    if args.env == "diplomacy":
        from src import runner_v4 as scenario_mod
        target_phase = "S1901M"
    else:
        from src import runner_v4_sotopia as scenario_mod
        target_phase = "PHASE_1"

    model_tag = "" if args.model == "gpt4o" else "haiku"
    dps = find_mixed_dps(args.env, model_tag)
    print(f"[E3] Found {len(dps)} mixed-driver DPs in {args.env}/{args.model}", flush=True)
    if args.limit:
        dps = dps[:args.limit]
        print(f"[E3] LIMIT applied -> {len(dps)} DPs", flush=True)

    results = []
    t0 = time.time()
    for i, dp in enumerate(dps, 1):
        et = time.time() - t0
        eta = et / i * (len(dps) - i) if i > 0 else 0
        print(f"[{i}/{len(dps)}] arch={dp['arch']} ep={dp['ep_i']} recip={dp['recipient']} "
              f"top1={dp['top1_sender']}({dp['top1_fs']:.2f}) "
              f"top2={dp['top2_sender']}({dp['top2_fs']:.2f}) "
              f"elapsed={et:.0f}s ETA={eta:.0f}s", flush=True)
        try:
            r = run_dp(args.env, dp, target_phase, scenario_mod, model_tag)
            if r is not None:
                results.append(r)
                print(f"  -> fs1={r['fs_drop_top1']:+.3f} fs2={r['fs_drop_top2']:+.3f} "
                      f"fs12={r['fs_drop_both']:+.3f} -> {r['interaction']}", flush=True)
            else:
                print(f"  -> SKIP (could not reproduce DP)", flush=True)
        except Exception as e:
            print(f"  -> ERROR: {type(e).__name__}: {e}", flush=True)

    out_path = ANALYSIS_DIR / f"e3_pairwise_{args.env}_{args.model}.json"
    from collections import Counter
    cnt = Counter(r["interaction"] for r in results)
    summary = {
        "env": args.env, "model": args.model,
        "n_total_mixed": len(dps), "n_analyzed": len(results),
        "interaction_counts": dict(cnt),
        "elapsed_s": time.time() - t0,
        "results": results,
    }
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")
    n = len(results)
    print(f"\n=== E3 SUMMARY ({args.env}/{args.model}) ===")
    print(f"Analyzed: {n}/{len(dps)} mixed-driver DPs")
    for k, v in cnt.most_common():
        print(f"  {k:12s}: {v}/{n} = {100*v/max(n,1):.1f}%")


if __name__ == "__main__":
    main()
