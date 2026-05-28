"""MASK-LOO baseline: per-(env, arch, model) sampled subset of DPs.

For each chosen DP and each sender at that DP, we replace ONLY that sender's
message with an empty placeholder ("<MASK>"), keep all other co-incoming
messages and the public history intact, and re-sample the recipient's action.
We then compute the full noise-corrected FS_KL_excess and classify the DP
under the same labelling rule as the main pipeline.

This is the *true* leave-one-out comparator: same noise floor, same 1.5σ
margin, same classifier, but the perturbation is information removal rather
than pragmatic-controlled substitution.

Outputs: data/pilot_b0/analysis/mask_loo_<env>_<model>.json (and a pooled
summary file mask_loo_summary.json).

Usage:
  python code/scripts/run_mask_loo.py --env diplomacy --model gpt4o
  python code/scripts/run_mask_loo.py --env diplomacy --model haiku
  python code/scripts/run_mask_loo.py --env sotopia   --model gpt4o
  python code/scripts/run_mask_loo.py --env sotopia   --model haiku

For speed we sample at most --per_cell DPs per (arch) within an (env, model)
slice (default 10), drawn from the existing counterfactual records so that
we score the same DPs that already appear in the main pipeline.
"""
from __future__ import annotations
import argparse, json, sys, time, random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src import config as C
from src.metrics import fs_kl_excess, noise_kl_samples

METRIC_DIR = ROOT / "data" / "pilot_b0" / "metrics"
ANALYSIS_DIR = ROOT / "data" / "pilot_b0" / "analysis"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

MASK_TOKEN = "<MASK>"


def classify(scores, rho=2.0):
    items = sorted(scores, reverse=True)
    pos = [v for v in items if v > 0]
    if len(pos) == 0:
        return "no"
    if len(pos) == 1:
        return "single"
    return "single" if pos[0] >= rho * max(pos[1], 1e-9) else "mixed"


def load_dp_index(env, model_tag):
    """Read counterfactual metric files for the (env, model) slice; index by
    (arch, ep, recipient) -> list of (sender, original_msg_idx_proxy)."""
    suffix = f"_{model_tag}" if model_tag else ""
    out = {}
    for arch in C.ARCHS:
        path = METRIC_DIR / f"{env}_C_{arch}_counterfactual{suffix}.json"
        if not path.exists():
            print(f"  skip (missing): {path.name}", file=sys.stderr)
            continue
        d = json.loads(path.read_text(encoding="utf-8"))
        by_dp = defaultdict(list)
        for r in d.get("records", []):
            dp_key = (arch, r["ep"], r["recipient"], r.get("phase", ""))
            by_dp[dp_key].append({
                "sender": r["sender"],
                "content": r["content"],
            })
        out[arch] = by_dp
    return out


def run_dp(env, arch, ep_i, recipient, target_phase, scenario_mod):
    """For one DP (env, arch, ep, recipient), MASK each incoming sender in
    turn and compute fs_kl_excess for each. Return list of (sender, fs_excess).
    """
    play_until = scenario_mod.play_until_target_phase
    restore_inject = scenario_mod._restore_and_inject
    sample_traces = scenario_mod._sample_with_traces

    group_seed = 100 + C.ARCHS.index(arch) + 1
    seed = group_seed * 1000 + ep_i
    ep = play_until(arch, seed, "C", env, target_phase)
    if ep is None:
        return None
    all_msgs = ep["msgs_at_target"]
    incoming = [(i, m) for i, m in enumerate(all_msgs) if m["recipient"] == recipient]
    if not incoming:
        return None

    # Base sampling
    env_b = restore_inject(env, ep["snap_pre_press"], all_msgs)
    obs_base = env_b.extract_observation(recipient)
    base_fine, _, _ = sample_traces(
        arch, recipient, obs_base, n=C.ACTION_SAMPLES_PER_DO,
        seed_base=seed * 13 + 777, temperature=C.TEMPERATURE, incentive="C",
    )
    # Noise floor: 3 replicates on the SAME base obs
    noise_fines = []
    for rep in range(3):
        nf, _, _ = sample_traces(
            arch, recipient, obs_base, n=C.ACTION_SAMPLES_PER_DO,
            seed_base=seed * 13 + 4000 + 100 * rep, temperature=C.TEMPERATURE, incentive="C",
        )
        noise_fines.append(nf)
    nk = noise_kl_samples(noise_fines, base_fine)

    sender_scores = []
    for (msg_idx, m) in incoming:
        # Mask only this sender's message; keep co-incoming intact
        env_m = restore_inject(
            env, ep["snap_pre_press"], all_msgs,
            substitute={"index": msg_idx, "new_content": MASK_TOKEN},
        )
        obs_mask = env_m.extract_observation(recipient)
        mask_fine, _, _ = sample_traces(
            arch, recipient, obs_mask, n=C.ACTION_SAMPLES_PER_DO,
            seed_base=seed * 13 + 8000 + (hash(m["sender"]) % 1000),
            temperature=C.TEMPERATURE, incentive="C",
        )
        fs = fs_kl_excess(base_fine, mask_fine, nk)
        sender_scores.append({
            "sender": m["sender"],
            "fs_kl_excess": fs["fs_kl_excess"],
            "dkl": fs["dkl"],
            "noise_mean": fs["noise_mean"],
            "noise_std": fs["noise_std"],
        })
    return {
        "env": env, "arch": arch, "ep": ep_i, "recipient": recipient,
        "phase": target_phase, "n_incoming": len(incoming),
        "sender_scores": sender_scores,
        "label": classify([s["fs_kl_excess"] for s in sender_scores]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", choices=["diplomacy", "sotopia"], required=True)
    ap.add_argument("--model", choices=["gpt4o", "haiku"], required=True)
    ap.add_argument("--per_cell", type=int, default=10,
                    help="DPs per arch to sample (default 10 -> ~40 DPs per (env,model))")
    ap.add_argument("--rng_seed", type=int, default=42)
    args = ap.parse_args()

    if args.model == "haiku":
        C.MODEL_TAG = "haiku"
        C.AGENT_MODEL = "claude-haiku-4.5"
    else:
        C.MODEL_TAG = ""
        C.AGENT_MODEL = "gpt-4o-2024-08-06"
    print(f"[MASK-LOO] env={args.env} model={args.model} AGENT_MODEL={C.AGENT_MODEL}",
          flush=True)

    if args.env == "diplomacy":
        from src import runner_v4 as scenario_mod
        target_phase = "S1901M"
    else:
        from src import runner_v4_sotopia as scenario_mod
        target_phase = "PHASE_1"

    model_tag = "" if args.model == "gpt4o" else "haiku"
    dp_index = load_dp_index(args.env, model_tag)

    rng = random.Random(args.rng_seed)
    chosen = []
    for arch, by_dp in dp_index.items():
        keys = sorted(by_dp.keys())  # deterministic order
        rng.shuffle(keys)
        chosen.extend(keys[:args.per_cell])
    print(f"[MASK-LOO] {len(chosen)} DPs queued ({args.per_cell} per arch)",
          flush=True)

    results = []
    t0 = time.time()
    for i, key in enumerate(chosen, 1):
        arch, ep_i, recipient, phase = key
        elapsed = time.time() - t0
        eta = elapsed / max(i - 1, 1) * (len(chosen) - i + 1) if i > 1 else 0
        print(f"[{i}/{len(chosen)}] arch={arch} ep={ep_i} recip={recipient} "
              f"elapsed={elapsed:.0f}s ETA={eta:.0f}s", flush=True)
        try:
            r = run_dp(args.env, arch, ep_i, recipient, target_phase, scenario_mod)
            if r is None:
                print("  -> SKIP (snapshot/recipient not reproducible)", flush=True)
                continue
            results.append(r)
            print(f"  -> label={r['label']} senders={len(r['sender_scores'])} "
                  f"top_fs={max((s['fs_kl_excess'] for s in r['sender_scores']), default=0):+.3f}",
                  flush=True)
        except Exception as e:
            print(f"  -> ERROR: {type(e).__name__}: {e}", flush=True)

    out_path = ANALYSIS_DIR / f"mask_loo_{args.env}_{args.model}.json"
    summary = {
        "env": args.env, "model": args.model,
        "per_cell": args.per_cell, "n_chosen": len(chosen),
        "n_analyzed": len(results),
        "elapsed_s": time.time() - t0,
        "results": results,
    }
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}", flush=True)

    # quick summary print
    cnt = {"single": 0, "mixed": 0, "no": 0}
    for r in results:
        cnt[r["label"]] += 1
    n = sum(cnt.values())
    if n:
        print(f"\n=== MASK-LOO SUMMARY ({args.env}/{args.model}) ===")
        print(f"Analyzed: {n} DPs")
        print(f"single: {cnt['single']/n*100:.1f}%  "
              f"mixed: {cnt['mixed']/n*100:.1f}%  "
              f"no: {cnt['no']/n*100:.1f}%")


if __name__ == "__main__":
    main()
