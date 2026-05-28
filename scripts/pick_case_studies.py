"""Pick three exemplar DPs (one per mode) for the §5.3 case studies.

For each candidate DP we require:
  - |M| >= 3 (at least three incoming senders, so a clean 3-bar picture)
  - all per-message FSKL scores recoverable (no NaN / missing)
  - trace_base and trace_intv exist for the message we will highlight

Modes:
  - SINGLE  : exactly one sender has FSKL > 0; its margin top1 - top2 is large
  - MIXED   : >=2 senders have FSKL > 0 and top1 < 2 * top2 (no dominance)
  - NO      : no sender has FSKL > 0 and the max FSKL is well below zero

We pool over counterfactual records since they carry the most readable
intervention text (negation of commitment). Score each candidate by how
"clean" the FSKL pattern is, then print the top-3 per mode for human review.

Output: data/pilot_b0/analysis/case_candidates.md
"""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
METRIC_DIR = ROOT / "data" / "pilot_b0" / "metrics"
OUT = ROOT / "data" / "pilot_b0" / "analysis" / "case_candidates.md"


def load_dps(iv: str = "counterfactual"):
    """Load every DP from every (env, arch, model) cell using a fixed IV."""
    dps = []  # list of dict
    for f in sorted(METRIC_DIR.glob(f"*_C_*_{iv}*.json")):
        # exclude gemini-tagged files until those are fixed
        if "gemini" in f.stem:
            continue
        stem = f.stem  # e.g. diplomacy_C_react_counterfactual or _haiku
        parts = stem.split("_")
        env = parts[0]
        arch = parts[2]
        tag = parts[-1] if parts[-1] in ("haiku", "gemini") else "gpt4o"
        d = json.loads(f.read_text(encoding="utf-8"))
        # group records by (ep, recipient, phase)
        by_dp = defaultdict(list)
        for r in d.get("records", []):
            key = (r["ep"], r["recipient"], r.get("phase", ""))
            by_dp[key].append(r)
        for key, recs in by_dp.items():
            if len(recs) < 3:
                continue
            scores = {r["sender"]: r["fs_kl_excess_fine"]["fs_kl_excess"] for r in recs}
            dps.append({
                "env": env, "arch": arch, "tag": tag,
                "ep": key[0], "recipient": key[1], "phase": key[2],
                "scores": scores,
                "records": recs,  # keep refs for text dump
                "n": len(recs),
                "file": f.name,
            })
    return dps


def classify(scores: dict[str, float], rho: float = 2.0):
    items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    pos = [(s, v) for s, v in items if v > 0]
    if not pos:
        return "no", items
    if len(pos) == 1:
        return "single", items
    if pos[0][1] >= rho * max(pos[1][1], 1e-9):
        return "single", items
    return "mixed", items


def pick_best(dps, mode: str, want=3):
    """Score each DP by "cleanliness" within the given mode."""
    pool = []
    for dp in dps:
        cls, items = classify(dp["scores"])
        if cls != mode:
            continue
        vals = [v for _, v in items]
        if mode == "single":
            # margin top1 - top2; want it large (clean lone driver)
            if len(vals) < 2:
                continue
            margin = vals[0] - vals[1]
            score = margin if vals[0] > 0 else -1
        elif mode == "mixed":
            # want top1 and top2 close, both > 0
            if len(vals) < 2 or vals[1] <= 0:
                continue
            ratio = vals[1] / max(vals[0], 1e-9)  # close to 1 = clean
            score = ratio * vals[1]  # high & balanced
        else:  # no
            # want max well below zero (clearly subnoise)
            score = -max(vals) if max(vals) <= 0 else -999
        if score > -10:
            pool.append((score, dp, items))
    pool.sort(key=lambda x: -x[0])
    return pool[:want]


def dump_dp(dp, items, mode, hilite_idx=0):
    """Markdown dump for one DP."""
    lines = []
    lines.append(f"### {dp['env']}/{dp['arch']}/{dp['tag']} ep{dp['ep']} -> {dp['recipient']} ({dp['phase']})")
    lines.append(f"  file: `{dp['file']}`  mode: **{mode}**")
    lines.append("")
    lines.append(f"  FSKL per sender:")
    for s, v in items:
        marker = "  <-- HILITE" if s == items[hilite_idx][0] else ""
        lines.append(f"    {s:>10s}: {v:+.3f}{marker}")
    lines.append("")
    # incoming messages
    lines.append("  Incoming messages:")
    for r in dp["records"]:
        lines.append(f"    [{r['sender']:>10s} -> {r['recipient']}] {r['content']}")
    lines.append("")
    # highlight one record's m_tilde + traces
    target = dp["records"][0]
    # pick the one matching hilite sender
    target_sender = items[hilite_idx][0]
    for r in dp["records"]:
        if r["sender"] == target_sender:
            target = r
            break
    lines.append(f"  Counterfactual highlight (sender = {target['sender']}):")
    lines.append(f"    original m: {target['content']}")
    lines.append(f"    m_tilde  : {target['m_tilde']}")
    lines.append("")
    lines.append("  Recipient trace_base (first 600 chars):")
    tb = target['trace_base']
    if isinstance(tb, str):
        for ln in tb[:600].splitlines():
            lines.append(f"    | {ln}")
    lines.append("")
    lines.append("  Recipient trace_intv after counterfactual (first 600 chars):")
    ti = target['trace_intv']
    if isinstance(ti, str):
        for ln in ti[:600].splitlines():
            lines.append(f"    | {ln}")
    lines.append("")
    lines.append("---")
    return "\n".join(lines)


def main():
    dps = load_dps(iv="counterfactual")
    print(f"loaded {len(dps)} DPs from counterfactual files (gpt4o + haiku only)")

    out_lines = ["# Case Study Candidates (Top-3 per mode)\n",
                 f"Pool: {len(dps)} DPs with |M|>=3, IV=counterfactual, gpt4o+haiku.\n"]

    for mode in ["single", "mixed", "no"]:
        out_lines.append(f"\n## {mode.upper()}-DRIVER candidates\n")
        picks = pick_best(dps, mode, want=3)
        if not picks:
            out_lines.append("(none)\n")
            continue
        for rank, (score, dp, items) in enumerate(picks, 1):
            # for SINGLE: highlight idx = 0 (the lone driver)
            # for MIXED: highlight idx = 0 (top1, but we'll mention top2 in caption)
            # for NO: highlight idx = 0 (the message closest to threshold)
            out_lines.append(f"\n#### Rank {rank}  (score={score:+.3f})\n")
            out_lines.append(dump_dp(dp, items, mode, hilite_idx=0))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
