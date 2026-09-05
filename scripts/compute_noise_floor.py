"""
Noise floor -- ANALYSIS step. CPU-only, no GPU needed.

Loads the raw texts from noise_floor_check.py and computes epsilon_W using
the EXACT SAME metric that will be used to judge real violations: keyword
classification -> qhat distribution over {0,1,BOT} -> total variation
distance between random halves of the K rollouts. This is what makes the
epsilon a fair yardstick instead of an ad-hoc majority-vote comparison.

Usage:
    python3 scripts/compute_noise_floor.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.judging import call_judge, qhat
from src.metrics import d_tv_conditional, noise_floor

VARIABLES = ["X", "W", "Z", "Y"]  # X: for the do(W) mirror test; W: non-descendant (H1); Z, Y: descendants (positive control)


def conditional_noise_floor(claims_rollouts_per_activation, variable, qhat_fn,
                            n_splits=200, seed=0):
    """Same idea as metrics.noise_floor(), but on the CONDITIONAL (BOT-
    excluded) distance -- only counts splits where BOTH halves had at
    least one committed claim (skips pairs with nothing to condition on)."""
    rng = np.random.default_rng(seed)
    nulls = []
    for rollouts in claims_rollouts_per_activation:
        K = len(rollouts)
        idx = np.arange(K)
        for _ in range(n_splits):
            rng.shuffle(idx)
            a = [rollouts[i] for i in idx[: K // 2]]
            b = [rollouts[i] for i in idx[K // 2:]]
            d = d_tv_conditional(qhat_fn(a, variable), qhat_fn(b, variable))
            if d is not None:
                nulls.append(d)
    nulls = np.array(nulls)
    if len(nulls) == 0:
        return None
    return {"epsilon": float(np.percentile(nulls, 95)),
            "median": float(np.median(nulls)), "n": len(nulls)}


def main():
    collected = json.load(open("results/noise_floor_texts.json"))

    # One judging pass per text yields claims for W, Z, AND Y together --
    # reused for all three variables' noise floor below.
    claims_rollouts_per_activation = [
        [call_judge(t, "keyword_all") for t in entry["texts"]] for entry in collected
    ]

    out = {"n_activations": len(collected)}
    for var in VARIABLES:
        result = noise_floor(claims_rollouts_per_activation, var, qhat, n_splits=200, seed=0)
        print(f"[{var}] raw: {result['n']} splits, median={result['median']:.3f}, "
             f"epsilon={result['epsilon']:.3f}")
        out[var] = result["epsilon"]
        out[f"{var}_median"] = result["median"]
        out[f"{var}_n_splits"] = result["n"]

        cond_result = conditional_noise_floor(claims_rollouts_per_activation, var, qhat, n_splits=200, seed=0)
        if cond_result is None:
            print(f"[{var}] conditional: no comparable splits found -- collect "
                 f"more activations with noise_floor_check.py before trusting "
                 f"the conditional cut for this variable.")
        else:
            print(f"[{var}] conditional: {cond_result['n']} splits, "
                 f"median={cond_result['median']:.3f}, epsilon={cond_result['epsilon']:.3f}")
            out[f"{var}_conditional"] = cond_result["epsilon"]
            out[f"{var}_conditional_median"] = cond_result["median"]
            out[f"{var}_conditional_n_splits"] = cond_result["n"]

    with open("results/epsilon.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved -> results/epsilon.json")
    print("Next: python3 main.py --from judging --to metrics --input results/results_stage2.json")


if __name__ == "__main__":
    main()
