"""E2 v2: all-mask ablation for no-driver DPs (Diplomacy + Sotopia, GPT-4o + Haiku).

For each no-driver DP (classified from random_string intervention records), we:
  1. Re-derive the snapshot at target_phase using the same seed scheme.
  2. Sample K base actions for the recipient (all incoming msgs intact).
  3. Sample 3 noise replicates at same T for a noise floor.
  4. Replace EVERY incoming message to recipient with a random ASCII string of
     the same length, then sample K masked actions.
  5. Compute FS_KL_excess = D_KL(base || masked) - noise_mean - 1.5*noise_std.
  6. is_diffuse = fs_excess > 0.

Outputs: data/pilot_b0/analysis/e2_allmask_<env>_<model>.json

Usage:
  python code/scripts/run_e2_allmask_v2.py --env diplomacy --model gpt4o
  python code/scripts/run_e2_allmask_v2.py --env diplomacy --model haiku
  python code/scripts/run_e2_allmask_v2.py --env sotopia   --model gpt4o
  python code/scripts/run_e2_allmask_v2.py --env sotopia   --model haiku
"""
from __future__ import annotations
import argparse
import json
import random
import string
import sys
import time
import math
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


def find_no_driver_dps(env, model_tag):
    """Use random_string-derived classification (most aggressive) to find
    no-driver DPs across all 4 archs."""
    suffix = f"_{model_tag}" if model_tag else ""
    out = []
    for arch in C.ARCHS:
        path = METRIC_DIR / f"{env}_C_{arch}_random_string{suffix}.json"
        if not path.exists():
            print(f"  skip (missing): {path.name}")
            continue
        d = json.loads(path.read_text(encoding="utf-8"))
        by_dp = defaultdict(list)
        for r in d["records"]:
            dp_key = (r["ep"], r["recipient"], r.get("phase", ""))
            by_dp[dp_key].append((r["sender"], r["fs_kl_excess_fine"]["fs_kl_excess"]))
        for dp_key, scores in by_dp.items():
            cls = classify([s for _, s in scores])
            if cls == "no" and len(scores) >= 1:
                out.append({
                    "arch": arch,
                    "ep_i": dp_key[0],
                    "recipient": dp_key[1],
                    "phase": dp_key[2],
                    "n_msgs": len(scores),
                })
    return out


def run_dp(env, arch, ep_i, recipient, target_phase, scenario_module, model_tag):
    """Reproduce DP and run base + all-mask comparison."""
    play_until = scenario_module.play_until_target_phase
    restore_inject = scenario_module._restore_and_inject
    sample_traces = scenario_module._sample_with_traces

    # Match group_seed scheme used by run_v4.py / run_v4_sotopia.py:
    # group_seed = 100 + (arch_index_1based)
    group_seed = 100 + C.ARCHS.index(arch) + 1
    seed = group_seed * 1000 + ep_i
    ep = play_until(arch, seed, "C", env, target_phase)
    if ep is None:
        return None
    all_msgs = ep["msgs_at_target"]
    if not any(m["recipient"] == recipient for m in all_msgs):
        return None

    # Base
    env_b = restore_inject(env, ep["snap_pre_press"], all_msgs)
    obs_base = env_b.extract_observation(recipient)
    base_fine, _, _ = sample_traces(
        arch, recipient, obs_base, n=C.ACTION_SAMPLES_PER_DO,
        seed_base=seed * 11 + 999, temperature=C.TEMPERATURE, incentive="C",
    )
    # Noise
    noise_fines = []
    for rep in range(3):
        nf, _, _ = sample_traces(
            arch, recipient, obs_base, n=C.ACTION_SAMPLES_PER_DO,
            seed_base=seed * 11 + 5000 + 100 * rep, temperature=C.TEMPERATURE, incentive="C",
        )
        noise_fines.append(nf)
    nk = noise_kl_samples(noise_fines, base_fine)
    # All-mask
    rng = random.Random(seed * 31 + hash(recipient))
    masked = []
    for m in all_msgs:
        if m["recipient"] == recipient:
            masked.append({
                "sender": m["sender"], "recipient": m["recipient"],
                "content": random_string_like(m["content"], rng),
            })
        else:
            masked.append(m)
    env_m = restore_inject(env, ep["snap_pre_press"], masked)
    obs_mask = env_m.extract_observation(recipient)
    mask_fine, _, _ = sample_traces(
        arch, recipient, obs_mask, n=C.ACTION_SAMPLES_PER_DO,
        seed_base=seed * 11 + 8888, temperature=C.TEMPERATURE, incentive="C",
    )
    fs = fs_kl_excess(base_fine, mask_fine, nk)
    return {
        "env": env, "model": model_tag or "gpt4o",
        "arch": arch, "ep": ep_i, "recipient": recipient, "phase": target_phase,
        "n_msgs_to_recip": sum(1 for m in all_msgs if m["recipient"] == recipient),
        "dkl_allmask": fs["dkl"],
        "noise_mean": fs["noise_mean"], "noise_std": fs["noise_std"],
        "fs_excess_allmask": fs["fs_kl_excess"],
        "is_diffuse": fs["fs_kl_excess"] > 0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", choices=["diplomacy", "sotopia"], required=True)
    ap.add_argument("--model", choices=["gpt4o", "haiku"], required=True)
    ap.add_argument("--limit", type=int, default=None, help="Limit DPs for smoke test")
    args = ap.parse_args()

    # Toggle MODEL_TAG and AGENT_MODEL dynamically
    if args.model == "haiku":
        C.MODEL_TAG = "haiku"
        C.AGENT_MODEL = "claude-haiku-4.5"
    else:
        C.MODEL_TAG = ""
        C.AGENT_MODEL = "gpt-4o"
    print(f"[E2v2] env={args.env} model={args.model} AGENT_MODEL={C.AGENT_MODEL}", flush=True)

    # Pick scenario module (must be imported AFTER config tweak so agents pick up model)
    if args.env == "diplomacy":
        from src import runner_v4 as scenario_mod
        target_phase = "S1901M"
    else:
        from src import runner_v4_sotopia as scenario_mod
        target_phase = "PHASE_1"

    model_tag = "" if args.model == "gpt4o" else "haiku"
    dps = find_no_driver_dps(args.env, model_tag)
    print(f"[E2v2] Found {len(dps)} no-driver DPs in {args.env}/{args.model}", flush=True)
    if args.limit:
        dps = dps[:args.limit]
        print(f"[E2v2] LIMIT applied -> {len(dps)} DPs", flush=True)

    results = []
    t0 = time.time()
    for i, dp in enumerate(dps, 1):
        et = time.time() - t0
        eta = et / i * (len(dps) - i) if i > 0 else 0
        print(f"[{i}/{len(dps)}] arch={dp['arch']} ep={dp['ep_i']} recip={dp['recipient']} "
              f"elapsed={et:.0f}s ETA={eta:.0f}s", flush=True)
        try:
            r = run_dp(args.env, dp["arch"], dp["ep_i"], dp["recipient"],
                      target_phase, scenario_mod, model_tag)
            if r is not None:
                results.append(r)
                print(f"  -> fs_excess_allmask={r['fs_excess_allmask']:+.3f} "
                      f"diffuse={r['is_diffuse']}", flush=True)
            else:
                print(f"  -> SKIP (snapshot/recipient not reproducible)", flush=True)
        except Exception as e:
            print(f"  -> ERROR: {type(e).__name__}: {e}", flush=True)

    out_path = ANALYSIS_DIR / f"e2_allmask_{args.env}_{args.model}.json"
    summary = {
        "env": args.env, "model": args.model, "n_total_no_driver": len(dps),
        "n_analyzed": len(results),
        "n_diffuse": sum(1 for r in results if r["is_diffuse"]),
        "elapsed_s": time.time() - t0,
        "results": results,
    }
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}", flush=True)
    n = len(results)
    nd = summary["n_diffuse"]
    print(f"\n=== E2 SUMMARY ({args.env}/{args.model}) ===")
    print(f"Analyzed: {n}/{len(dps)} no-driver DPs")
    print(f"Diffuse-driven: {nd}/{n} = {100*nd/max(n,1):.1f}%")
    print(f"True autonomous: {n-nd}/{n} = {100*(n-nd)/max(n,1):.1f}%")


if __name__ == "__main__":
    main()
