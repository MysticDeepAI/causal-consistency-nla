#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py -- entry point for the Causal Consistency of NLA Explanations experiment.

Runs the stages in order, passing `items` (a flat list of dicts) from one
to the next in memory, and saving a checkpoint after each one. Same
convention as the companion vss-colombia repo's main.py.

Usage:
    python main.py                       # runs all stages
    python main.py --to extraction       # stops after Stage 2
    python main.py --from judging --input results/results_stage2.json
"""

import argparse
import json

import yaml

STAGES = ["generation", "extraction", "verbalization", "judging", "metrics"]


def load_config(path="config.yaml"):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(data, path):
    """numpy arrays aren't directly JSON serializable -- converted to
    lists before saving."""
    def convert(obj):
        if hasattr(obj, "tolist"):
            return obj.tolist()
        raise TypeError(f"Don't know how to serialize: {type(obj)}")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=convert)
    print(f"  -> saved to {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--from", dest="from_stage", choices=STAGES, default="generation")
    parser.add_argument("--to", dest="to_stage", choices=STAGES, default="metrics")
    parser.add_argument("--input", default=None,
                        help="partial JSON from a previous run, required if --from != generation")
    args = parser.parse_args()

    config = load_config(args.config)
    rd = config["paths"]["results_dir"]
    from_idx = STAGES.index(args.from_stage)
    to_idx = STAGES.index(args.to_stage)

    stage_out_path = {
        "generation": rd + "results_stage0.json",
        "extraction": rd + "results_stage1.json",
        "verbalization": rd + "results_stage2.json",
        "judging": rd + "results_stage3.json",
        "metrics": rd + "results.json",
    }

    # --- Stage 1: generation ---------------------------------------------
    if from_idx <= 0:
        print("\n=== Stage 1/5: prompt generation (SCM) ===")
        from src.generation import build_items
        items = build_items(config)
        save_json(items, stage_out_path["generation"])
    else:
        assert args.input, "--input is required if not starting from generation"
        with open(args.input, encoding="utf-8") as f:
            items = json.load(f)

    if to_idx == 0:
        return

    # --- Stage 2: extraction ----------------------------------------------
    if from_idx <= 1:
        print("\n=== Stage 2/5: activation extraction ===")
        from src.extraction import extract_activations
        items = extract_activations(items, config)
        save_json(items, stage_out_path["extraction"])

    if to_idx == 1:
        return

    # --- Stage 3: verbalization (AV) ---------------------------------------
    if from_idx <= 2:
        print("\n=== Stage 3/5: verbalization (AV) ===")
        from src.verbalization import verbalize
        items = verbalize(items, config)
        save_json(items, stage_out_path["verbalization"])

    if to_idx == 2:
        return

    # --- Stage 4: judging (claims extraction) -------------------------------
    if from_idx <= 3:
        print("\n=== Stage 4/5: claims judging ===")
        from src.judging import judge_items
        items = judge_items(items, config)
        save_json(items, stage_out_path["judging"])

    if to_idx == 3:
        return

    # --- Stage 5: metrics ------------------------------------------------------
    print("\n=== Stage 5/5: metrics (noise floor, violation rates, H2) ===")
    from src.metrics import run as run_metrics
    result = run_metrics(config, items)
    save_json(result, stage_out_path["metrics"])

    print(f"\nDone. Now run: python plot_results.py")


if __name__ == "__main__":
    main()
