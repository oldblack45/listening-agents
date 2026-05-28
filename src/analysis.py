"""Aggregate per-cell metrics into PILOT_B0_RESULT.md."""
from __future__ import annotations
import json
import math
import statistics
from pathlib import Path
from collections import defaultdict
from . import config as C


def _bootstrap_ci(values, n_boot=1000, alpha=0.05, seed=0):
    import random
    if not values:
        return (0.0, 0.0, 0.0)
    rng = random.Random(seed)
    boot_means = []
    n = len(values)
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        boot_means.append(sum(sample) / n)
    boot_means.sort()
    lo = boot_means[int(alpha / 2 * n_boot)]
    hi = boot_means[int((1 - alpha / 2) * n_boot) - 1]
    return (sum(values) / n, lo, hi)


def aggregate_cells():
    cells = {}
    for p in sorted(Path(C.METRIC_DIR).glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        cells[d["cell"]] = d
    return cells


def build_pilot_report() -> str:
    cells = aggregate_cells()
    if not cells:
        return "# Pilot B0 Result — no cells produced\n"
    lines = ["# Pilot B0 Result", "", f"Generated cells: {len(cells)}", ""]

    # Slot pass-rate
    lines.append("## Slot pass-rate per (arch × intervention)")
    lines.append("")
    lines.append("| arch | intervention | n | pass-rate |")
    lines.append("|---|---|---|---|")
    for cid in sorted(cells):
        c = cells[cid]
        sr = c["slot_records"]
        n = len(sr)
        pr = c["slot_pass_rate"]
        lines.append(f"| {c['arch']} | {c['intervention']} | {n} | {pr:.2%} |")
    lines.append("")

    # FS_KL_excess heatmap
    lines.append("## FS_KL_excess (fine-grained) per cell — mean [95% CI]")
    lines.append("")
    archs = sorted({c["arch"] for c in cells.values()})
    intvs = sorted({c["intervention"] for c in cells.values()})
    header = "| arch \ intv | " + " | ".join(intvs) + " |"
    sep = "|---" * (len(intvs) + 1) + "|"
    lines += [header, sep]
    for a in archs:
        row = [a]
        for iv in intvs:
            cid = f"{a}_{iv}"
            if cid not in cells:
                row.append("—"); continue
            vals = [r["fs_kl_excess_fine"]["fs_kl_excess"] for r in cells[cid]["records"]]
            if not vals:
                row.append("(no records)"); continue
            m, lo, hi = _bootstrap_ci(vals)
            row.append(f"{m:+.3f} [{lo:+.2f}, {hi:+.2f}]")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # FS_binary
    lines.append("## FS_binary mean per cell (action-flip rate)")
    lines.append("")
    lines += [header, sep]
    for a in archs:
        row = [a]
        for iv in intvs:
            cid = f"{a}_{iv}"
            if cid not in cells:
                row.append("—"); continue
            vals = [r["fs_binary"] for r in cells[cid]["records"]]
            if not vals:
                row.append("(none)"); continue
            row.append(f"{sum(vals)/len(vals):.2%} (n={len(vals)})")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # Pass gate
    lines.append("## Pass gate")
    lines.append("")
    overall_slot = statistics.mean([c["slot_pass_rate"] for c in cells.values()])
    pos_cells = 0
    measured = 0
    for c in cells.values():
        vals = [r["fs_kl_excess_fine"]["fs_kl_excess"] for r in c["records"]]
        if not vals:
            continue
        measured += 1
        if sum(vals) / len(vals) > 0:
            pos_cells += 1
    lines.append(f"- Slot pass-rate (mean across cells): **{overall_slot:.2%}** — gate ≥ {C.SLOT_PASS_RATE_GATE:.0%}")
    lines.append(f"- Cells with mean FS_KL_excess > 0: **{pos_cells}/{measured}** — majority gate")
    gate_ok = overall_slot >= C.SLOT_PASS_RATE_GATE and measured > 0 and pos_cells >= (measured / 2)
    lines.append("")
    lines.append(f"### Verdict: **{'PASS' if gate_ok else 'FAIL'}**")
    lines.append("")

    return "\n".join(lines)


def write_report(path: Path | None = None):
    text = build_pilot_report()
    out = Path(path) if path else (Path(C.PROJECT_ROOT) / "refine-logs" / "PILOT_B0_RESULT.md")
    out.write_text(text, encoding="utf-8")
    return out
