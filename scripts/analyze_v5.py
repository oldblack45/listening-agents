"""V5 全量分析：对 16 cells (4 archs x 4 ivs) 计算

- |M| 分布（按 (arch, ep, recipient) 唯一聚合）
- F^KL excess 主表（mean / median / n / 95% bootstrap CI）
- within-DP ranking：每个 DP 内对 |M| 条 incoming msg 按 F^KL fine 排序，
  统计 top-rank share / driver share (F^KL > 0)
- slot pass rate

输出：
  data/pilot_b0/analysis/v5_main_table.csv
  data/pilot_b0/analysis/v5_m_distribution.json
  data/pilot_b0/analysis/v5_ranking.json
  data/pilot_b0/analysis/v5_summary.json
"""
from __future__ import annotations
import json
from pathlib import Path
from collections import defaultdict

import numpy as np

ROOT = Path(r"D:/论文3/observability-paper")
METRIC_DIR = ROOT / "data/pilot_b0/metrics"
OUT_DIR = ROOT / "data/pilot_b0/analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ARCHS = ["react", "autogen", "genagents", "camel"]
INTVS = ["identity", "fact_replace", "counterfactual", "random_string"]


def load_cell(arch, iv):
    p = METRIC_DIR / f"diplomacy_C_{arch}_{iv}.json"
    return json.loads(p.read_text(encoding="utf-8"))


def boot_ci(arr, B=2000, ci=95):
    if len(arr) == 0:
        return (float("nan"),) * 3
    arr = np.asarray(arr, dtype=float)
    m = float(np.mean(arr))
    if len(arr) == 1:
        return m, m, m
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(arr), size=(B, len(arr)))
    boots = arr[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return m, float(lo), float(hi)


def m_distribution_for_arch(arch):
    """对该 arch 用 identity cell 推断 |M| 分布（identity 涵盖所有 DP，
    因为 identity slot 永远 pass）。每条 record 是 (ep, recipient, sender) — 同一
    (ep, recipient) 的 record 数即 |M|。"""
    cell = load_cell(arch, "identity")
    by_dp = defaultdict(set)
    for r in cell["records"]:
        by_dp[(r["ep"], r["recipient"])].add(r["sender"])
    m_values = [len(s) for s in by_dp.values()]
    if not m_values:
        return {"arch": arch, "n_dps": 0}
    arr = np.array(m_values)
    return {
        "arch": arch,
        "n_dps": int(len(arr)),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "max": int(arr.max()),
        "min": int(arr.min()),
        "ge2": int((arr >= 2).sum()),
        "ge3": int((arr >= 3).sum()),
        "ge4": int((arr >= 4).sum()),
        "ge5": int((arr >= 5).sum()),
        "ge6": int((arr >= 6).sum()),
        "hist": {int(k): int((arr == k).sum()) for k in range(1, int(arr.max()) + 1)},
    }


def main_table():
    """每 (arch, iv) 报告：
      n_records, slot_pass_rate, fs_fine mean/CI/median, fs_coarse mean/CI/median
    """
    rows = []
    for arch in ARCHS:
        for iv in INTVS:
            cell = load_cell(arch, iv)
            recs = cell["records"]
            slot_n = len(cell.get("slot_records", []))
            slot_pass = sum(1 for r in cell.get("slot_records", []) if r["pass"])
            fine = [r["fs_kl_excess_fine"]["fs_kl_excess"] for r in recs]
            coarse = [r["fs_kl_excess_coarse"]["fs_kl_excess"] for r in recs]
            mf, lof, hif = boot_ci(fine)
            mc, loc, hic = boot_ci(coarse)
            rows.append({
                "arch": arch,
                "intervention": iv,
                "n_records": len(recs),
                "slot_pass": slot_pass,
                "slot_total": slot_n,
                "slot_pass_rate": slot_pass / max(1, slot_n),
                "fs_fine_mean": mf, "fs_fine_lo": lof, "fs_fine_hi": hif,
                "fs_fine_median": float(np.median(fine)) if fine else float("nan"),
                "fs_coarse_mean": mc, "fs_coarse_lo": loc, "fs_coarse_hi": hic,
                "fs_coarse_median": float(np.median(coarse)) if coarse else float("nan"),
                "fs_fine_pos_share": float(np.mean([1 if x > 0 else 0 for x in fine])) if fine else float("nan"),
            })
    return rows


def ranking_analysis():
    """合并所有非-identity 干预（fact_replace/counterfactual/random_string）的记录，
    对每个 (arch, intervention, ep, recipient) 内的 senders 按 fs_fine 排序，
    统计：
      - DPs with |M|>=2（可比 top-rank）
      - DPs with |M|>=3（更可靠的 ranking 验证）
      - mean rank gap = |fs_top - fs_bottom|
      - driver share = fraction of senders with fs > 0
    """
    out = []
    for arch in ARCHS:
        for iv in [x for x in INTVS if x != "identity"]:
            cell = load_cell(arch, iv)
            by_dp = defaultdict(list)  # (ep, recipient) -> list[(sender, fs)]
            for r in cell["records"]:
                by_dp[(r["ep"], r["recipient"])].append(
                    (r["sender"], r["fs_kl_excess_fine"]["fs_kl_excess"])
                )
            n_total = len(by_dp)
            n_ge2 = sum(1 for v in by_dp.values() if len(v) >= 2)
            n_ge3 = sum(1 for v in by_dp.values() if len(v) >= 3)
            n_ge4 = sum(1 for v in by_dp.values() if len(v) >= 4)
            gaps = []
            driver_shares = []
            for v in by_dp.values():
                if len(v) < 2:
                    continue
                fs = np.array([x[1] for x in v])
                gaps.append(float(fs.max() - fs.min()))
                driver_shares.append(float((fs > 0).mean()))
            out.append({
                "arch": arch,
                "intervention": iv,
                "n_dps": n_total,
                "n_dps_ge2": n_ge2,
                "n_dps_ge3": n_ge3,
                "n_dps_ge4": n_ge4,
                "mean_rank_gap": float(np.mean(gaps)) if gaps else float("nan"),
                "mean_driver_share": float(np.mean(driver_shares)) if driver_shares else float("nan"),
            })
    return out


def write_csv(rows, path):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(
                f"{r[k]:.6f}" if isinstance(r[k], float) else str(r[k])
                for k in keys
            ) + "\n")


def main():
    mt = main_table()
    write_csv(mt, OUT_DIR / "v5_main_table.csv")
    (OUT_DIR / "v5_main_table.json").write_text(
        json.dumps(mt, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    m_dist = {a: m_distribution_for_arch(a) for a in ARCHS}
    (OUT_DIR / "v5_m_distribution.json").write_text(
        json.dumps(m_dist, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    rk = ranking_analysis()
    (OUT_DIR / "v5_ranking.json").write_text(
        json.dumps(rk, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # summary print
    print("\n=== V5 |M| DISTRIBUTION ===")
    for a, d in m_dist.items():
        if d.get("n_dps", 0) == 0:
            continue
        print(f"  {a}: n_dps={d['n_dps']} mean={d['mean']:.2f} max={d['max']} "
              f"|M|>=3:{d['ge3']}/{d['n_dps']} |M|>=4:{d['ge4']}/{d['n_dps']}")

    print("\n=== V5 MAIN TABLE (F^KL fine, mean [95% CI], n) ===")
    print(f"  {'arch':10s} {'iv':16s} {'n':>4s} {'fs_fine_mean':>14s} {'CI':>22s} {'slot_pass':>11s}")
    for r in mt:
        print(f"  {r['arch']:10s} {r['intervention']:16s} {r['n_records']:>4d} "
              f"{r['fs_fine_mean']:>+14.4f} [{r['fs_fine_lo']:+.3f},{r['fs_fine_hi']:+.3f}] "
              f"{r['slot_pass']:>4d}/{r['slot_total']:<5d}")

    print("\n=== V5 RANKING ===")
    for r in rk:
        print(f"  {r['arch']:10s} {r['intervention']:16s} "
              f"DPs={r['n_dps']:>3d} |M|>=3:{r['n_dps_ge3']:>3d} |M|>=4:{r['n_dps_ge4']:>3d} "
              f"gap={r['mean_rank_gap']:.3f} driver_share={r['mean_driver_share']:.2f}")

    # high-level rollup
    summary = {
        "n_cells": len(mt),
        "archs": ARCHS,
        "interventions": INTVS,
        "m_distribution_overall": {
            a: {k: d.get(k) for k in ("n_dps", "mean", "max", "ge3", "ge4")}
            for a, d in m_dist.items()
        },
    }
    (OUT_DIR / "v5_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nwrote {OUT_DIR}")


if __name__ == "__main__":
    main()
