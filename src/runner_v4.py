"""Runner v4: within-decision-point per-message attribution."""
from __future__ import annotations
import json
import os
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from . import config as C
from .env_adapter import DiplomacyEnv, orders_to_string, orders_to_coarse
from .agents import ReActAgent, AutoGenAgent, GenAgentsAgent, CamelAgent
from .slot_generator import parse_tuple, generate_interventions, validate_intervention
from .metrics import fs_kl_excess, noise_kl_samples
from . import llm_client

ARCH_FACTORY = {
    "react": ReActAgent,
    "autogen": AutoGenAgent,
    "genagents": GenAgentsAgent,
    "camel": CamelAgent,
}


def _make_env(scenario="diplomacy"):
    return DiplomacyEnv()


def _build_agent(arch, power, incentive="C"):
    cls = ARCH_FACTORY[arch]
    return cls(power, incentive=incentive)


def _sample_with_traces(arch, power, obs, n, seed_base, temperature, incentive="C"):
    """Sample N actions concurrently and ALSO collect the raw response text
    (reasoning trace) for the first sample only (to bound storage)."""
    cls = ARCH_FACTORY[arch]
    agent = cls(power, incentive=incentive)

    def one(i):
        act = agent.act(obs, seed=seed_base + i, temperature=temperature)
        return (
            orders_to_string(act["orders"]),
            orders_to_coarse(act["orders"]),
            act.get("_raw", "") if i == 0 else None,
        )

    with ThreadPoolExecutor(max_workers=C.MAX_CONCURRENT_REQUESTS) as ex:
        results = list(ex.map(one, range(n)))
    fine = [r[0] for r in results]
    coarse = [r[1] for r in results]
    trace = results[0][2] if results else ""
    return fine, coarse, trace


def play_until_target_phase(arch, episode_seed, incentive, scenario, target_phase):
    """Play episode until the env reaches target_phase. Snapshot BEFORE press.
    Then have all powers generate press messages (without injecting them into
    the snapshotted state), so we can later re-inject them selectively under
    intervention.

    Returns dict:
      snap_pre_press: snapshot at target_phase, BEFORE any press injected
      msgs_at_target: list of {sender, recipient, content} - all press at DP
    Or None if target_phase was never reached.
    """
    env = _make_env(scenario)
    env.reset(seed=episode_seed)
    agents = {p: _build_agent(arch, p, incentive) for p in env.powers}

    target_reached = False
    for t_idx in range(C.MAX_TURNS_PER_EPISODE):
        if env.is_done():
            break
        cur_phase = env.game.get_current_phase()
        if cur_phase == target_phase:
            target_reached = True
            break
        # play this phase normally
        per_orders = {}
        for power in env.powers:
            obs_p = env.extract_observation(power)
            act = agents[power].act(obs_p, seed=episode_seed * 100 + t_idx)
            for to, content in act["messages"].items():
                env.inject_message(power, to, content)
            per_orders[power] = act["orders"]
        env.step(per_orders)

    if not target_reached:
        return None

    snap_pre = env.snapshot()
    active_powers = set(env.powers)
    msgs_at_target = []
    for power in env.powers:
        obs_p = env.extract_observation(power)
        act = agents[power].act(obs_p, seed=episode_seed * 100 + 9999)
        for to, content in act["messages"].items():
            # Filter: recipient must be an active power and not self.
            # LLM occasionally addresses inactive powers (RUSSIA etc.) or
            # broadcast labels (GLOBAL/ALL) which would yield empty
            # action distributions during do-intervention sampling.
            if to not in active_powers or to == power:
                continue
            msgs_at_target.append({
                "sender": power, "recipient": to, "content": content,
            })
    return {"snap_pre_press": snap_pre, "msgs_at_target": msgs_at_target}


def _restore_and_inject(scenario, snap_pre_press, all_msgs, substitute=None):
    """Restore env to snap_pre_press, inject all_msgs in order. If substitute
    is given (a dict with 'index' and 'new_content'), the msg at that index
    is replaced by new_content before injection."""
    env = _make_env(scenario)
    env.restore(snap_pre_press)
    for i, m in enumerate(all_msgs):
        content = m["content"]
        if substitute is not None and substitute["index"] == i:
            content = substitute["new_content"]
        env.inject_message(m["sender"], m["recipient"], content)
    return env


def run_group_v4(arch, n_episodes=10, group_seed=100,
                 scenario="diplomacy", incentive="C",
                 target_phase="S1901M",
                 interventions=None,
                 progress_cb=None):
    interventions = interventions or C.INTERVENTIONS
    out_records = {iv: [] for iv in interventions}
    slot_records = {iv: [] for iv in interventions}
    cross_pool = []

    t0 = time.time()

    def log(msg):
        line = f"  [v4 t={time.time() - t0:.0f}s] {msg}"
        print(line, flush=True)
        if progress_cb:
            progress_cb(line)

    # --- Phase 1: play episodes up to target phase, collect all press msgs ---
    episode_data = []
    log(f"phase1 begin: n_episodes={n_episodes} target_phase={target_phase}")
    for ep_i in range(n_episodes):
        seed = group_seed * 1000 + ep_i
        ep_t0 = time.time()
        ep = play_until_target_phase(arch, seed, incentive, scenario, target_phase)
        if ep is None:
            log(f"phase1 ep {ep_i + 1}/{n_episodes} SKIP (target_phase not reached)")
            continue
        msgs = ep["msgs_at_target"]
        by_rec = {}
        for i, m in enumerate(msgs):
            by_rec.setdefault(m["recipient"], []).append((i, m))
        episode_data.append({
            "ep_i": ep_i, "seed": seed,
            "snap": ep["snap_pre_press"],
            "all_msgs": msgs,
            "by_recipient": by_rec,
        })
        for m in msgs:
            cross_pool.append(m["content"])
        log(f"phase1 ep {ep_i + 1}/{n_episodes} done in {time.time() - ep_t0:.0f}s "
            f"({len(msgs)} msgs, {len(by_rec)} recipients)")
    log(f"phase1 complete: {len(episode_data)} valid eps, "
        f"phase1_time={time.time() - t0:.0f}s")

    # --- Phase 2: per-msg attribution at target_phase ---
    total_attrs = sum(len(v) for ed in episode_data for v in ed["by_recipient"].values())
    log(f"phase2 begin: total_attrs={total_attrs} (each attr=base+3noise+6iv samples)")
    phase2_t0 = time.time()
    attr_idx = 0

    for ed in episode_data:
        ep_i = ed["ep_i"]
        seed = ed["seed"]
        all_msgs = ed["all_msgs"]
        for recipient, idx_msg_pairs in ed["by_recipient"].items():
            for (msg_idx, m) in idx_msg_pairs:
                attr_idx += 1
                attr_t0 = time.time()

                # Base sampling: all incoming msgs intact
                env_b = _restore_and_inject(scenario, ed["snap"], all_msgs)
                obs_base = env_b.extract_observation(recipient)
                base_fine, base_coarse, trace_base = _sample_with_traces(
                    arch, recipient, obs_base,
                    n=C.ACTION_SAMPLES_PER_DO,
                    seed_base=seed * 11 + attr_idx,
                    temperature=C.TEMPERATURE, incentive=incentive,
                )

                # Noise baseline: 3 same-temp replicates on the SAME base obs
                noise_fines, noise_coarses = [], []
                for rep in range(3):
                    nf, nc, _ = _sample_with_traces(
                        arch, recipient, obs_base,
                        n=C.ACTION_SAMPLES_PER_DO,
                        seed_base=seed * 11 + 5000 + 100 * rep + attr_idx,
                        temperature=C.TEMPERATURE, incentive=incentive,
                    )
                    noise_fines.append(nf)
                    noise_coarses.append(nc)
                nk_fine = noise_kl_samples(noise_fines, base_fine)
                nk_coarse = noise_kl_samples(noise_coarses, base_coarse)

                # Generate 6 interventions for THIS msg
                ftuple = parse_tuple(m["sender"], m["recipient"], m["content"])
                intvs = generate_interventions(
                    m["content"], ftuple, cross_pool,
                    seed_base=seed * 10 + attr_idx,
                )
                co_incoming = [
                    {"sender": all_msgs[j]["sender"], "content": all_msgs[j]["content"]}
                    for (j, mm) in idx_msg_pairs if j != msg_idx
                ]

                # Run 6 interventions on this single msg
                for iv in interventions:
                    m_tilde = intvs.get(iv, m["content"])
                    val = validate_intervention(
                        iv, m["sender"], m["recipient"],
                        m["content"], m_tilde, original_tuple=ftuple,
                    )
                    slot_records[iv].append({
                        "ep": ep_i, "recipient": recipient,
                        "sender": m["sender"], "type": iv,
                        "pass": val["pass"], "bleu": val["bleu"],
                        "cosine": val["cosine"], "tuple_match": val["tuple_match"],
                        "reason": val["reason"],
                    })
                    if not val["pass"]:
                        continue
                    env_i = _restore_and_inject(
                        scenario, ed["snap"], all_msgs,
                        substitute={"index": msg_idx, "new_content": m_tilde},
                    )
                    obs_int = env_i.extract_observation(recipient)
                    intv_fine, intv_coarse, trace_intv = _sample_with_traces(
                        arch, recipient, obs_int,
                        n=C.ACTION_SAMPLES_PER_DO,
                        seed_base=seed * 11 + 7000 + (hash(iv) % 1000) + attr_idx,
                        temperature=C.TEMPERATURE, incentive=incentive,
                    )
                    fs_f = fs_kl_excess(base_fine, intv_fine, nk_fine)
                    fs_c = fs_kl_excess(base_coarse, intv_coarse, nk_coarse)
                    out_records[iv].append({
                        "ep": ep_i,
                        "recipient": recipient,
                        "sender": m["sender"],
                        "phase": target_phase,
                        "content": m["content"],
                        "m_tilde": m_tilde,
                        "fourtuple": ftuple,
                        "fs_kl_excess_fine": fs_f,
                        "fs_kl_excess_coarse": fs_c,
                        "co_incoming_msgs": co_incoming,
                        "trace_base": (trace_base or "")[:2000],
                        "trace_intv": (trace_intv or "")[:2000],
                    })

                # Per-attribution log line with ETA
                phase2_elapsed = time.time() - phase2_t0
                eta = (total_attrs - attr_idx) * (phase2_elapsed / attr_idx) if attr_idx else 0
                log(f"attr {attr_idx}/{total_attrs} ep{ep_i} {m['sender']}->{recipient} "
                    f"in {time.time() - attr_t0:.0f}s (phase2 {phase2_elapsed:.0f}s, ETA {eta:.0f}s)")

    log(f"phase2 complete: {time.time() - phase2_t0:.0f}s")

    # --- Phase 3: write 6 metric JSON files ---
    elapsed = time.time() - t0
    written = []
    for iv in interventions:
        cell_id = f"{scenario}_{incentive}_{arch}_{iv}"
        tag = getattr(C, "MODEL_TAG", "")
        fname = f"{cell_id}_{tag}.json" if tag else f"{cell_id}.json"
        pass_n = sum(1 for r in slot_records[iv] if r["pass"])
        total_n = max(1, len(slot_records[iv]))
        out = {
            "cell": cell_id, "arch": arch, "intervention": iv,
            "scenario": scenario, "incentive": incentive,
            "n_episodes": n_episodes, "target_phase": target_phase,
            "elapsed_s": elapsed,
            "records": out_records[iv],
            "slot_records": slot_records[iv],
            "slot_pass_rate": pass_n / total_n,
            "runner_version": "v4_within_dp_ranking+reasoning_traces",
        }
        out_path = Path(C.METRIC_DIR) / fname
        out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        written.append((cell_id, pass_n, total_n, len(out_records[iv])))
    log(f"phase3 wrote {len(written)} cells")
    return {"elapsed_s": elapsed, "written": written}