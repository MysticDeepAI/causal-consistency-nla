"""
End-to-end smoke test on a tiny subset (2-3 pairs, K=2 rollouts), with
explicit assertions at each stage handoff. Run this before committing to a
multi-hour full run -- same role as vss-colombia's validate_all_stages.py.

Usage:
    python validate_all_stages.py
"""
import copy
import json
from pathlib import Path

import yaml

from src import generation


def make_tiny_config(cfg: dict) -> dict:
    tiny = copy.deepcopy(cfg)
    tiny["scm"]["n_pairs"] = 2
    tiny["scm"]["n_context_records"] = 3
    tiny["sampling"]["k_rollouts"] = 2
    tiny["sampling"]["k_rollouts_noise_floor"] = 4
    tiny["sampling"]["n_activations_noise_floor"] = 3
    tiny["paths"]["results_dir"] = "results/_smoke_test/"
    return tiny


def check_stage0(cfg: dict) -> str:
    items = generation.build_items(cfg)
    n_expected = cfg["scm"]["n_pairs"] * 2  # confounded + not_confounded
    assert len(items) == n_expected, \
        f"expected {n_expected} items, got {len(items)}"

    for item in items:
        # Minimal-pair sanity: the footwear span must repeat identically
        # across factual and counterfactual (do(X) only touches weather).
        shoe_word = "boots" if "boots" in item["text_factual"] else "sneakers"
        assert item["text_factual"].count(shoe_word) == item["text_counterfactual"].count(shoe_word), \
            f"footwear span mismatch in pair {item['pair_id']}"
        assert item["v_factual"][1] == item["v_counterfactual"][1], \
            f"W (footwear) must be invariant under do(X) in pair {item['pair_id']}"
        assert item["v_factual"][0] != item["v_counterfactual"][0], \
            f"X (weather) must differ between twins in pair {item['pair_id']}"

    print(f"[validate] stage0 OK -- {len(items)} pairs, all invariance checks passed")
    return items


def main():
    cfg = yaml.safe_load(open(Path(__file__).parent / "config.yaml"))
    tiny = make_tiny_config(cfg)
    Path(tiny["paths"]["results_dir"]).mkdir(parents=True, exist_ok=True)

    check_stage0(tiny)

    print("\n[validate] Stage 0 (generation) passes on the tiny subset.")
    print("[validate] Stages 1-5 require a GPU and the project's AV/AR "
          "checkpoints -- run them manually with main.py --to <stage> "
          "against results/_smoke_test/ once the harness is wired in.")


if __name__ == "__main__":
    main()
