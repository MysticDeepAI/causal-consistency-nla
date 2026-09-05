"""
do(W) mirror experiment -- Step 1: generation. CPU-only.

Mirrors src/generation.py, but intervenes on FOOTWEAR (W) instead of
WEATHER (X). Tests the mirror of H1: does the AV's claim about weather
(a non-descendant of W) stay invariant when we flip the footwear span?

Kept as a standalone script (not wired into main.py) so nothing about the
already-working do(X) pipeline is at risk of breaking.

Usage:
    python3 scripts/generate_do_w.py
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dataset.scm_final import FROZEN, FROZEN_NOCONF, context_block, minimal_pair


def build_items_do_w(cfg: dict) -> list:
    scm_cfg = cfg["scm"]
    rng = random.Random(scm_cfg["seed"] + 1)  # different seed stream than do(X)

    items = []
    conditions = [("confounded", FROZEN), ("not_confounded", FROZEN_NOCONF)]

    for cond_name, params in conditions:
        for i in range(scm_cfg["n_pairs"]):
            tmpl_idx = i % scm_cfg["n_templates"]
            pair = minimal_pair(params, tmpl_idx, rng, do_var="W")
            ctx = context_block(params, scm_cfg["n_context_records"], rng)
            items.append({
                "pair_id": f"DOW_{cond_name[:4].upper()}_{i:04d}",
                "template": tmpl_idx,
                "context_condition": cond_name,
                "context_block": ctx,
                "text_factual": pair["text"],
                "text_counterfactual": pair["text_ctf"],
                "v_factual": list(pair["v"]),
                "v_counterfactual": list(pair["v_ctf"]),
                "do": list(pair["do"]),
            })

    print(f"[generate_do_w] built {len(items)} items (do(W) mirror experiment)")
    return items


if __name__ == "__main__":
    import yaml
    cfg = yaml.safe_load(open(Path(__file__).resolve().parent.parent / "config.yaml"))
    items = build_items_do_w(cfg)

    # Sanity check: X must be IDENTICAL across twins, W must DIFFER --
    # the exact mirror of the do(X) invariant. Fail loudly if this SCM
    # generalization has a bug, before spending any GPU time on it.
    for it in items:
        assert it["v_factual"][0] == it["v_counterfactual"][0], \
            f"X should be invariant under do(W) in {it['pair_id']}"
        assert it["v_factual"][1] != it["v_counterfactual"][1], \
            f"W should differ between twins in {it['pair_id']}"
    print(f"[generate_do_w] invariance sanity check passed on all {len(items)} items")

    out_path = "results/results_stage0_doW.json"
    Path("results").mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(items, f, indent=2)
    print(f"[generate_do_w] saved -> {out_path}")
    print("Next: python3 scripts/run_do_w_pipeline.py")
