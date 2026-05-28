"""V4 driver: 24 cells (4 archs x 6 ivs, diplomacy, incentive=C) with within-DP ranking.

 Resumes by skipping (arch) whose 6 v4 metrics files all already exist.

 Usage:
   python code/scripts/run_v4.py                              # full 24 cells
   python code/scripts/run_v4.py --archs react                # one arch only
   python code/scripts/run_v4.py --archs react --n-episodes 2 # smoke test
 """
from __future__ import annotations
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

# Ensure repo root on sys.path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def cell_files_exist_v4(scenario, incentive, arch, METRIC_DIR, INTERVENTIONS, tag=""):
    for iv in INTERVENTIONS:
        fname = f"{scenario}_{incentive}_{arch}_{iv}_{tag}.json" if tag else f"{scenario}_{incentive}_{arch}_{iv}.json"
        p = Path(METRIC_DIR) / fname
        if not p.exists():
            return False
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            if "v4" not in d.get("runner_version", ""):
                return False
        except Exception:
            return False
    return True


def now():
    return datetime.now().strftime("%H:%M:%S")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", nargs="+", default=None)
    ap.add_argument("--n-episodes", type=int, default=5)
    ap.add_argument("--target-phase", default="S1901M")
    ap.add_argument("--incentive", default="C")
    ap.add_argument("--scenario", default="diplomacy")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    from src import config as C
    from src.runner_v4 import run_group_v4

    archs = args.archs or C.ARCHS
    groups = [(args.scenario, args.incentive, a) for a in archs]

    print(f"[v4] starting at {now()}", flush=True)
    print(f"[v4] target_phase={args.target_phase} n_episodes={args.n_episodes} "
          f"incentive={args.incentive} archs={archs}", flush=True)
    print(f"[v4] ACTION_SAMPLES_PER_DO={C.ACTION_SAMPLES_PER_DO} "
          f"MAX_CONCURRENT_REQUESTS={C.MAX_CONCURRENT_REQUESTS}", flush=True)
    print(f"[v4] TOTAL GROUPS: {len(groups)} ({len(groups) * 6} cells)", flush=True)

    t0 = time.time()
    done = 0
    for i, (s, inc, a) in enumerate(groups, 1):
        if not args.force and cell_files_exist_v4(s, inc, a, C.METRIC_DIR, C.INTERVENTIONS, tag=getattr(C, "MODEL_TAG", "")):
            print(f"=== V4 GROUP {i}/{len(groups)}: {s}/{a}/{inc} SKIP (already v4) ===", flush=True)
            done += 1
            continue
        eta_str = ""
        if done > 0:
            avg = (time.time() - t0) / done
            remaining = (len(groups) - i + 1) * avg
            eta_str = f" ETA={remaining / 3600:.1f}h"
        print(f"=== V4 GROUP {i}/{len(groups)}: {s}/{a}/{inc} START at {now()}{eta_str} ===", flush=True)
        gt0 = time.time()
        try:
            run_group_v4(
                arch=a, n_episodes=args.n_episodes,
                group_seed=100 + i, scenario=s, incentive=inc,
                target_phase=args.target_phase,
            )
            done += 1
            elapsed = time.time() - gt0
            total_elapsed = time.time() - t0
            print(f"  -> GROUP {i}/{len(groups)} DONE in {elapsed:.0f}s "
                  f"(total {total_elapsed / 3600:.1f}h, done={done})", flush=True)
        except Exception as e:
            print(f"  -> GROUP {i}/{len(groups)} ERROR {type(e).__name__}: {e}", flush=True)
            import traceback
            traceback.print_exc()

    print(f"[v4] complete at {now()} ({(time.time() - t0) / 3600:.1f}h total)", flush=True)


if __name__ == "__main__":
    main()