"""
Stage 5 -- Consistency metrics.

Implements: total variation distance between claim distributions, noise
floor calibration (epsilon_V per variable), bootstrap violation rates, and
the confounded-vs-not-confounded contrast (H2).

Data convention:
  qhat: dict {0: p0, 1: p1, "BOT": pbot}  (output of judging.qhat)
  A "pair" = {"q_fact": qhat, "q_ctf": qhat} for a given variable.
"""
import json
from pathlib import Path

import numpy as np

KEYS = [0, 1, "BOT"]


def d_tv(q1: dict, q2: dict) -> float:
    """Total variation distance between two claim distributions."""
    return 0.5 * sum(abs(q1.get(k, 0.0) - q2.get(k, 0.0)) for k in KEYS)


# ---------------- Noise floor (Phase 2a) ----------------

def noise_floor(claims_rollouts_per_activation, variable, qhat_fn,
                n_splits=200, seed=0):
    """
    claims_rollouts_per_activation: one entry per activation, each entry a
      list of K>=10 claim sets (one per AV rollout on the SAME activation).
    Returns the null distribution of d_TV between random halves of rollouts
    of the same activation, plus epsilon = 95th percentile.
    """
    rng = np.random.default_rng(seed)
    nulls = []
    for rollouts in claims_rollouts_per_activation:
        K = len(rollouts)
        idx = np.arange(K)
        for _ in range(n_splits):
            rng.shuffle(idx)
            a = [rollouts[i] for i in idx[: K // 2]]
            b = [rollouts[i] for i in idx[K // 2:]]
            nulls.append(d_tv(qhat_fn(a, variable), qhat_fn(b, variable)))
    nulls = np.array(nulls)
    return {"null_dist": nulls, "epsilon": float(np.percentile(nulls, 95)),
            "median": float(np.median(nulls)), "n": len(nulls)}


# ---------------- Violation rates (main experiment) ----------------

def violation_rate(pairs, epsilon):
    """Fraction of pairs with d_TV(q_fact, q_ctf) > epsilon."""
    flags = np.array([d_tv(p["q_fact"], p["q_ctf"]) > epsilon for p in pairs])
    return flags


def bootstrap_ci(flags, n_boot=10000, seed=0, alpha=0.05):
    rng = np.random.default_rng(seed)
    n = len(flags)
    boots = np.array([flags[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    return {"rate": float(flags.mean()),
            "lo": float(np.percentile(boots, 100 * alpha / 2)),
            "hi": float(np.percentile(boots, 100 * (1 - alpha / 2)))}


def diff_bootstrap(flags_a, flags_b, n_boot=10000, seed=0):
    """
    Bootstrap CI of the rate DIFFERENCE between two independent conditions.
    Uses: (a) AV vs probe [localization contribution],
          (b) confounded vs not-confounded context [H2].
    Also returns a two-sided permutation p-value.
    """
    rng = np.random.default_rng(seed)
    obs = flags_a.mean() - flags_b.mean()
    na, nb = len(flags_a), len(flags_b)
    boots = np.array([
        flags_a[rng.integers(0, na, na)].mean() - flags_b[rng.integers(0, nb, nb)].mean()
        for _ in range(n_boot)])
    pooled = np.concatenate([flags_a, flags_b])
    perm = []
    for _ in range(n_boot):
        rng.shuffle(pooled)
        perm.append(pooled[:na].mean() - pooled[na:].mean())
    pval = float((np.abs(np.array(perm)) >= abs(obs)).mean())
    return {"diff": float(obs),
            "lo": float(np.percentile(boots, 2.5)),
            "hi": float(np.percentile(boots, 97.5)),
            "p_perm": pval}


def holm_correction(pvals):
    """Holm correction for the multiple cells of the main results table."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def is_informative(qhat: dict) -> bool:
    """True if at least one rollout committed to a value (not 100% BOT)."""
    return qhat.get("BOT", 1.0) < 1.0


def qhat_conditional(qhat: dict):
    """
    Renormalizes a qhat distribution EXCLUDING BOT: P(boots | a claim was
    made). Returns None if there were zero committed claims (nothing to
    condition on). This is what actually fixes the power problem: raw
    d_TV over {0,1,BOT} is dominated by the BOT-vs-BOT match whenever most
    rollouts are silent, which structurally caps the distance low even
    when the FEW committed claims flip completely.
    """
    z0, z1 = qhat.get(0, 0.0), qhat.get(1, 0.0)
    total = z0 + z1
    if total <= 0:
        return None
    return {0: z0 / total, 1: z1 / total}


def d_tv_conditional(q1: dict, q2: dict):
    """TV distance between two conditional (BOT-excluded) distributions.
    Returns None if either side has no committed claims."""
    c1, c2 = qhat_conditional(q1), qhat_conditional(q2)
    if c1 is None or c2 is None:
        return None
    return 0.5 * (abs(c1[0] - c2[0]) + abs(c1[1] - c2[1]))


def wilson_ci(k: int, n: int, z: float = 1.96) -> dict:
    """
    Wilson score interval for a binomial proportion -- the correct interval
    when k=0 or k=n, where a naive percentile bootstrap collapses to a
    zero-width [rate, rate] interval (resampling identical values can't
    produce variability). This gives an honest upper bound instead of a
    misleadingly tight one.
    """
    if n == 0:
        return {"rate": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * ((p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5) / denom
    return {"rate": p, "lo": max(0.0, center - margin), "hi": min(1.0, center + margin), "n": n}


def build_qhat_pairs(items: list, variable: str, qhat_fn) -> tuple:
    """Generic version of the pair-building loop in run() -- works for
    any tracked variable (W, Z, or Y), not just W."""
    pairs_confounded, pairs_notconf = [], []
    for item in items:
        q_fact = qhat_fn(item["claims"]["factual"], variable)
        q_ctf = qhat_fn(item["claims"]["counterfactual"], variable)
        pair = {"pair_id": item["pair_id"], "q_fact": q_fact, "q_ctf": q_ctf}
        (pairs_confounded if item["context_condition"] == "confounded"
         else pairs_notconf).append(pair)
    return pairs_confounded, pairs_notconf


def descendant_sensitivity_report(items: list, variable: str,
                                  epsilon_json_path: str = "results/epsilon.json") -> dict:
    """
    POSITIVE CONTROL: unlike W (non-descendant), Z and Y are causal
    descendants of X in the SCM, so their claims SHOULD shift under
    do(X). Reports the same conditional (BOT-excluded) shift metric as
    the W test, but frames it as "does it respond" rather than "does it
    stay put" -- a HIGH rate here is the expected/good outcome, and rules
    out the trivial alternative explanation that the AV is simply inert
    to the intervention altogether (which would make the W=0% result
    meaningless: a "dead" AV passes the invariance test for the wrong
    reason).
    """
    from src.judging import qhat as qhat_fn

    eps_data = json.load(open(epsilon_json_path)) if Path(epsilon_json_path).exists() else {}
    eps_cond = eps_data.get(f"{variable}_conditional")

    pairs_conf, pairs_notconf = build_qhat_pairs(items, variable, qhat_fn)
    all_pairs = pairs_conf + pairs_notconf

    dists = [d_tv_conditional(p["q_fact"], p["q_ctf"]) for p in all_pairs]
    comparable = [d for d in dists if d is not None]

    result = {"variable": variable, "n_total_pairs": len(all_pairs),
             "n_comparable": len(comparable)}

    if eps_cond is None:
        result["note"] = (f"no '{variable}_conditional' key in {epsilon_json_path} -- "
                          f"run scripts/compute_noise_floor.py (updated version) first")
        return result
    if len(comparable) == 0:
        result["note"] = "zero comparable pairs -- cannot assess responsiveness"
        return result

    result["epsilon_conditional"] = eps_cond
    flags = np.array([d > eps_cond for d in comparable])
    result["responsiveness_rate"] = wilson_ci(int(flags.sum()), len(flags))
    return result


def run(config: dict, items: list, epsilon: float = None) -> dict:
    """Entry point for this stage (matches generation/extraction/
    verbalization/judging's items-in-memory convention). items: output of
    judging.judge_items() -- each has item["claims"] = {"factual": [...K],
    "counterfactual": [...K]}.

    epsilon: noise-floor threshold for W (footwear). If None, loaded from
    results/epsilon.json (produced by scripts/compute_noise_floor.py --
    run that FIRST, or this raises a clear error instead of silently
    using an arbitrary threshold).

    Returns a results dict with:
      - overall violation rate (bootstrap CI)
      - violation rate split by context_condition (confounded / not)
      - H2 contrast: bootstrap difference + permutation p-value
    """
    if epsilon is None:
        eps_path = Path("results/epsilon.json")
        if not eps_path.exists():
            raise RuntimeError(
                "results/epsilon.json not found. Run scripts/noise_floor_check.py "
                "then scripts/compute_noise_floor.py first -- metrics.py refuses "
                "to guess an epsilon.")
        epsilon = json.load(open(eps_path))["W"]

    from src.judging import qhat as qhat_fn

    pairs_confounded, pairs_notconf = [], []
    for item in items:
        q_fact = qhat_fn(item["claims"]["factual"], "W")
        q_ctf = qhat_fn(item["claims"]["counterfactual"], "W")
        pair = {"pair_id": item["pair_id"], "q_fact": q_fact, "q_ctf": q_ctf}
        (pairs_confounded if item["context_condition"] == "confounded"
         else pairs_notconf).append(pair)

    all_pairs = pairs_confounded + pairs_notconf
    flags_all = violation_rate(all_pairs, epsilon)
    flags_conf = violation_rate(pairs_confounded, epsilon)
    flags_notconf = violation_rate(pairs_notconf, epsilon)

    result = {
        "epsilon_W": epsilon,
        "n_pairs_total": len(all_pairs),
        "n_pairs_confounded": len(pairs_confounded),
        "n_pairs_not_confounded": len(pairs_notconf),
        "overall_violation_rate": bootstrap_ci(flags_all, n_boot=config["metrics"]["bootstrap_n"],
                                               seed=config["metrics"]["bootstrap_seed"]),
        "confounded_violation_rate": bootstrap_ci(flags_conf, n_boot=config["metrics"]["bootstrap_n"],
                                                   seed=config["metrics"]["bootstrap_seed"]),
        "not_confounded_violation_rate": bootstrap_ci(flags_notconf, n_boot=config["metrics"]["bootstrap_n"],
                                                       seed=config["metrics"]["bootstrap_seed"]),
        "h2_contrast": diff_bootstrap(flags_conf, flags_notconf,
                                      n_boot=config["metrics"]["bootstrap_n"],
                                      seed=config["metrics"]["bootstrap_seed"]),
    }

    print(f"\nepsilon_W = {epsilon:.3f} (from results/epsilon.json)")
    print(f"Overall violation rate:        {result['overall_violation_rate']}")
    print(f"Confounded context rate:       {result['confounded_violation_rate']}")
    print(f"Not-confounded context rate:   {result['not_confounded_violation_rate']}")
    print(f"H2 contrast (confounded - not): {result['h2_contrast']}")

    # ---- Power check: among pairs where BOTH arms have at least one
    # committed claim, compare CONDITIONAL distributions (BOT excluded).
    # This is the real fix for the power problem: raw d_TV is dominated by
    # the BOT-vs-BOT match when most rollouts are silent, which caps the
    # distance low even when the few committed claims flip completely. ----
    epsilon_cond_path = Path("results/epsilon.json")
    epsilon_cond = None
    if epsilon_cond_path.exists():
        epsilon_cond = json.load(open(epsilon_cond_path)).get("W_conditional")

    def conditional_flags(pairs):
        ds = [d_tv_conditional(p["q_fact"], p["q_ctf"]) for p in pairs]
        ds = [d for d in ds if d is not None]
        return np.array(ds), np.array([d > epsilon_cond for d in ds]) if epsilon_cond is not None else (np.array(ds), None)

    dists_conf, flags_cond_conf = conditional_flags(pairs_confounded)
    dists_notconf, flags_cond_notconf = conditional_flags(pairs_notconf)

    result["n_conditional_confounded"] = len(dists_conf)
    result["n_conditional_not_confounded"] = len(dists_notconf)

    print(f"\n--- Conditional (BOT-excluded) comparison, among pairs with a "
         f"claim on both sides ---")
    print(f"n comparable: confounded={len(dists_conf)}/{len(pairs_confounded)}, "
         f"not_confounded={len(dists_notconf)}/{len(pairs_notconf)}")

    if epsilon_cond is None:
        print("results/epsilon.json has no 'W_conditional' key yet -- rerun "
             "scripts/compute_noise_floor.py with the updated version to get it. "
             "Showing raw conditional distances only (no violation flag):")
        if len(dists_conf): print(f"  confounded distances:     {sorted(dists_conf.round(2).tolist())}")
        if len(dists_notconf): print(f"  not_confounded distances: {sorted(dists_notconf.round(2).tolist())}")
    else:
        result["epsilon_W_conditional"] = epsilon_cond
        k_conf, n_conf = int(flags_cond_conf.sum()), len(flags_cond_conf)
        k_notconf, n_notconf = int(flags_cond_notconf.sum()), len(flags_cond_notconf)
        if n_conf > 0:
            result["conditional_confounded_violation_rate"] = wilson_ci(k_conf, n_conf)
            print(f"Confounded violation rate (Wilson 95% CI):     {result['conditional_confounded_violation_rate']}")
        if n_notconf > 0:
            result["conditional_not_confounded_violation_rate"] = wilson_ci(k_notconf, n_notconf)
            print(f"Not-confounded violation rate (Wilson 95% CI): {result['conditional_not_confounded_violation_rate']}")
        if n_conf > 0 and n_notconf > 0:
            result["conditional_h2_contrast"] = diff_bootstrap(
                flags_cond_conf, flags_cond_notconf, n_boot=config["metrics"]["bootstrap_n"],
                seed=config["metrics"]["bootstrap_seed"])
            print(f"H2 contrast (conditional, bootstrap diff):     {result['conditional_h2_contrast']}")
            if k_conf == 0 and k_notconf == 0:
                print(f"  (0 events on both sides -- the bootstrap diff is "
                     f"uninformative here by the same zero-variability issue; "
                     f"read the two Wilson upper bounds above instead)")

    print(f"\n=== Positive control: descendant sensitivity ===")
    print(f"(Z, Y are causal descendants of X -- they SHOULD respond to "
         f"do(X), unlike W. A high rate here rules out a 'dead' AV that "
         f"trivially passes the W invariance test by ignoring everything.)")
    result["descendant_sensitivity"] = {}
    for var in ["Z", "Y"]:
        dr = descendant_sensitivity_report(items, var)
        result["descendant_sensitivity"][var] = dr
        if "note" in dr:
            print(f"{var}: {dr['note']} (n_comparable={dr['n_comparable']}/{dr['n_total_pairs']})")
        else:
            print(f"{var}: n_comparable={dr['n_comparable']}/{dr['n_total_pairs']}, "
                 f"epsilon={dr['epsilon_conditional']:.3f}, "
                 f"responsiveness rate (Wilson 95% CI) = {dr['responsiveness_rate']}")

    return result


if __name__ == "__main__":
    # Smoke test with simulated data
    rng = np.random.default_rng(1)
    fake = lambda p1, pbot: {"BOT": pbot, 1: (1 - pbot) * p1, 0: (1 - pbot) * (1 - p1)}
    pairs_null = [{"q_fact": fake(0.7, 0.2), "q_ctf": fake(0.7 + rng.normal(0, .05), 0.2)} for _ in range(200)]
    pairs_viol = [{"q_fact": fake(0.7, 0.2), "q_ctf": fake(0.3, 0.2)} for _ in range(200)]
    eps = 0.15
    fa = violation_rate(pairs_viol, eps)
    fb = violation_rate(pairs_null, eps)
    print("violating rate:", bootstrap_ci(fa))
    print("null rate     :", bootstrap_ci(fb))
    print("difference    :", diff_bootstrap(fa, fb, n_boot=2000))
