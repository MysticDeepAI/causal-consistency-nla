"""
AR-verified salience spot-check for footwear claims.

For a random sample of rollouts where the AV committed to a footwear
value, removes the footwear mention and, separately, a random unrelated
word (placebo), reconstructs both via the AR, and compares reconstruction
error against the REAL activation. If ARS_real consistently exceeds
ARS_placebo, footwear claims are genuinely load-bearing -- not decorative.

Usage:
    CUDA_VISIBLE_DEVICES=3 python3 scripts/ars_check.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.ars import _load_ar, compute_ars_sample

N_SAMPLES = None  # None = use ALL available (item, arm) instances, not a subsample
SEED = 0


def find_committed_rollouts(items):
    """Yields (pair_id, arm, explanation_text, real_vector) -- ONE per
    (item, arm), not one per matching rollout. Multiple K rollouts of the
    same item-arm share the SAME real activation, so testing more than one
    per item-arm would be pseudo-replication (correlated, not independent
    samples), inflating the apparent n without adding real information."""
    from src.judging import keyword_classify_footwear

    for item in items:
        for arm in ("factual", "counterfactual"):
            for text in item["explanations"][arm]:
                claim = keyword_classify_footwear(text)[0]
                if claim["value"] is not None:
                    yield item["pair_id"], arm, text, item["vectors"][arm]
                    break  # one representative rollout per item-arm, then move on


def main():
    cfg = yaml.safe_load(open(Path(__file__).resolve().parent.parent / "config.yaml"))
    items = json.load(open("results/results_stage3.json"))

    rng = np.random.default_rng(SEED)
    pool = list(find_committed_rollouts(items))
    n_target = len(pool) if N_SAMPLES is None else min(N_SAMPLES, len(pool))
    print(f"Found {len(pool)} independent (item, arm) instances with a "
         f"committed footwear claim; using {n_target}.")
    idx = rng.choice(len(pool), size=n_target, replace=False)
    sample = [pool[i] for i in idx]

    ar_model, tokenizer = _load_ar(cfg["model"]["ar_checkpoint"])
    layer = cfg["model"]["layer"]

    results = []
    for i, (pair_id, arm, text, vector) in enumerate(sample, 1):
        r = compute_ars_sample(pair_id, arm, vector, text, ar_model, tokenizer, rng, layer)
        if r is not None:
            results.append(r)
        if i % 50 == 0 or i == len(sample):
            print(f"  [{i}/{len(sample)}] processed")

    print("\nFirst 10 examples:")
    for r in results[:10]:
        print(f"  {r['pair_id']}/{r['arm']}: quote='{r['quote']}' ars_real={r['ars_real']:+.4f} | "
             f"placebo='{r['placebo_word']}' ars_placebo={r['ars_placebo']:+.4f} | "
             f"{'REAL >> PLACEBO' if r['ars_real'] > r['ars_placebo'] else 'no clear gap'}")

    real = np.array([r["ars_real"] for r in results])
    placebo = np.array([r["ars_placebo"] for r in results])
    diff = real - placebo

    print(f"\nn = {len(results)}")
    print(f"mean ARS_real    = {real.mean():+.4f}")
    print(f"mean ARS_placebo = {placebo.mean():+.4f}")
    print(f"paired diff (real - placebo): mean={diff.mean():+.4f}, "
         f"{ (diff > 0).sum() }/{len(diff)} samples favor real over placebo")

    # paired bootstrap CI on the mean difference
    n_boot = 5000
    boots = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"bootstrap 95% CI on mean diff: [{lo:+.4f}, {hi:+.4f}]")

    with open("results/ars_check.json", "w") as f:
        json.dump({"results": results, "mean_diff": float(diff.mean()),
                  "ci_lo": float(lo), "ci_hi": float(hi)}, f, indent=2)
    print("\nSaved -> results/ars_check.json")


if __name__ == "__main__":
    main()
