"""
do(W) mirror experiment -- Step 5: analysis. CPU-only.

Mirrors the H1 test from metrics.py, but for X (weather) instead of W
(footwear) -- reuses the SAME generic helpers (build_qhat_pairs,
d_tv_conditional, wilson_ci), just pointed at a different variable and a
different epsilon (epsilon.json's "X_conditional", added by the updated
compute_noise_floor.py).

Also reports the H2' mirror: does X-invariance break down more under the
confounded context, when W (not X) is the one being intervened?

Usage:
    python3 scripts/analyze_do_w.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.judging import qhat
from src.metrics import build_qhat_pairs, d_tv_conditional, diff_bootstrap, wilson_ci


def main():
    items = json.load(open("results/results_stage3_doW.json"))

    eps_path = Path("results/epsilon.json")
    if not eps_path.exists():
        raise RuntimeError(
            "results/epsilon.json not found. Run scripts/noise_floor_check.py "
            "then scripts/compute_noise_floor.py first.")
    eps_data = json.load(open(eps_path))
    eps_x_cond = eps_data.get("X_conditional")
    if eps_x_cond is None:
        raise RuntimeError(
            "No 'X_conditional' key in epsilon.json -- make sure 'X' is in "
            "compute_noise_floor.py's VARIABLES list, then re-run it.")

    pairs_conf, pairs_notconf = build_qhat_pairs(items, "X", qhat)
    all_pairs = pairs_conf + pairs_notconf

    def flags_for(pairs):
        ds = [d_tv_conditional(p["q_fact"], p["q_ctf"]) for p in pairs]
        ds = [d for d in ds if d is not None]
        return np.array(ds), np.array([d > eps_x_cond for d in ds])

    dists_conf, flags_conf = flags_for(pairs_conf)
    dists_notconf, flags_notconf = flags_for(pairs_notconf)

    print(f"=== do(W) mirror test: is X (weather) invariant under do(W)? ===")
    print(f"epsilon_X_conditional = {eps_x_cond:.3f}")
    print(f"n comparable: confounded={len(flags_conf)}/{len(pairs_conf)}, "
         f"not_confounded={len(flags_notconf)}/{len(pairs_notconf)}")

    if len(flags_conf) > 0:
        k_conf = int(flags_conf.sum())
        print(f"Confounded violation rate (Wilson 95% CI):     "
             f"{wilson_ci(k_conf, len(flags_conf))}")
    if len(flags_notconf) > 0:
        k_notconf = int(flags_notconf.sum())
        print(f"Not-confounded violation rate (Wilson 95% CI): "
             f"{wilson_ci(k_notconf, len(flags_notconf))}")

    if len(flags_conf) > 0 and len(flags_notconf) > 0:
        h2_prime = diff_bootstrap(flags_conf, flags_notconf, n_boot=10000, seed=0)
        print(f"H2' contrast (confounded - not):               {h2_prime}")
        if flags_conf.sum() == 0 and flags_notconf.sum() == 0:
            print("  (0 events on both sides -- bootstrap diff uninformative; "
                 "read the two Wilson bounds above instead)")

    result = {
        "epsilon_X_conditional": eps_x_cond,
        "n_comparable_confounded": len(flags_conf),
        "n_comparable_not_confounded": len(flags_notconf),
        "confounded_violation_rate": wilson_ci(int(flags_conf.sum()), len(flags_conf)) if len(flags_conf) else None,
        "not_confounded_violation_rate": wilson_ci(int(flags_notconf.sum()), len(flags_notconf)) if len(flags_notconf) else None,
    }
    with open("results/results_do_w.json", "w") as f:
        json.dump(result, f, indent=2)
    print("\nSaved -> results/results_do_w.json")


if __name__ == "__main__":
    main()
