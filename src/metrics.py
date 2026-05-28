"""FS metrics: FS_KL_excess, FS_logp, FS_binary."""
from __future__ import annotations
import math
from collections import Counter


def empirical_distribution(samples: list[str]) -> Counter:
    n = len(samples)
    c = Counter(samples)
    return Counter({k: v / n for k, v in c.items()}) if n else Counter()


def laplace_distribution(samples: list[str], support: list[str], alpha: float = 0.5) -> dict:
    """Laplace-smoothed distribution over a fixed support."""
    c = Counter(samples)
    K = len(support)
    total = len(samples) + alpha * K
    return {a: (c.get(a, 0) + alpha) / total for a in support}


def kl_divergence(p: dict, q: dict, eps: float = 1e-12) -> float:
    s = 0.0
    for k, pk in p.items():
        if pk <= 0:
            continue
        qk = q.get(k, eps)
        s += pk * math.log(pk / max(qk, eps))
    return max(0.0, s)


def fs_kl_excess(
    base_samples: list[str],
    inter_samples: list[str],
    noise_kl_distrib: list[float],
    support: list[str] | None = None,
    alpha: float = 0.5,
    sigma_gate: float = 1.5,
) -> dict:
    """Excess of KL(base||intervention) above noise baseline + sigma_gate * std."""
    if support is None:
        support = sorted(set(base_samples) | set(inter_samples))
    if not support:
        return {"fs_kl_excess": 0.0, "dkl": 0.0, "threshold": 0.0,
                "noise_mean": 0.0, "noise_std": 0.0, "support_size": 0}
    p_base = laplace_distribution(base_samples, support, alpha)
    p_int = laplace_distribution(inter_samples, support, alpha)
    dkl = kl_divergence(p_base, p_int)
    if noise_kl_distrib:
        m = sum(noise_kl_distrib) / len(noise_kl_distrib)
        var = sum((x - m) ** 2 for x in noise_kl_distrib) / max(1, len(noise_kl_distrib))
        std = math.sqrt(var)
    else:
        m, std = 0.0, 0.0
    threshold = m + sigma_gate * std
    return {
        "fs_kl_excess": dkl - threshold,
        "dkl": dkl, "threshold": threshold,
        "noise_mean": m, "noise_std": std,
        "support_size": len(support),
    }


def fs_logp(base_samples: list[str], inter_samples: list[str],
            chosen_action: str | None = None, alpha: float = 0.5) -> float:
    """log p_base(a) - log p_int(a) where a is the modal base action (or supplied)."""
    if not base_samples or not inter_samples:
        return 0.0
    support = sorted(set(base_samples) | set(inter_samples))
    p_b = laplace_distribution(base_samples, support, alpha)
    p_i = laplace_distribution(inter_samples, support, alpha)
    a = chosen_action if chosen_action and chosen_action in p_b else max(p_b, key=p_b.get)
    return math.log(p_b.get(a, 1e-9)) - math.log(p_i.get(a, 1e-9))


def fs_binary(base_samples: list[str], inter_samples: list[str]) -> int:
    if not base_samples or not inter_samples:
        return 0
    bm = Counter(base_samples).most_common(1)[0][0]
    im = Counter(inter_samples).most_common(1)[0][0]
    return int(bm != im)


def noise_kl_samples(noise_arr_per_temp: list[list[str]],
                     base_samples: list[str],
                     support: list[str] | None = None,
                     alpha: float = 0.5) -> list[float]:
    """Compute KL(p_base || p_at_temp) for each temperature variant."""
    if support is None:
        merged = set(base_samples)
        for arr in noise_arr_per_temp:
            merged.update(arr)
        support = sorted(merged)
    p_b = laplace_distribution(base_samples, support, alpha)
    out = []
    for arr in noise_arr_per_temp:
        p_a = laplace_distribution(arr, support, alpha)
        out.append(kl_divergence(p_b, p_a))
    return out
