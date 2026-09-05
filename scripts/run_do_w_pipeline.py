"""
do(W) mirror experiment -- Steps 2-4: extraction, verbalization, judging.

Reuses the EXACT SAME functions from src/ as the do(X) pipeline --
extraction.py, verbalization.py, and judging.py are all generic over
"items" and don't care which variable was intervened on. Only the
generation step (scripts/generate_do_w.py) differs.

Kept separate from main.py so a mistake here can't touch the already-
validated do(X) results.

Usage:
    CUDA_VISIBLE_DEVICES=0 python3 scripts/run_do_w_pipeline.py --to extraction
    CUDA_VISIBLE_DEVICES=0 python3 scripts/run_do_w_pipeline.py --from verbalization --to verbalization
    python3 scripts/run_do_w_pipeline.py --from judging --to judging
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

STAGES = ["extraction", "verbalization", "judging"]

PATHS = {
    "generation": "results/results_stage0_doW.json",
    "extraction": "results/results_stage1_doW.json",
    "verbalization": "results/results_stage2_doW.json",
    "judging": "results/results_stage3_doW.json",
}


def save_json(data, path):
    def convert(obj):
        if hasattr(obj, "tolist"):
            return obj.tolist()
        raise TypeError(f"Don't know how to serialize: {type(obj)}")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=convert)
    print(f"  -> saved to {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from", dest="from_stage", choices=STAGES, default="extraction")
    parser.add_argument("--to", dest="to_stage", choices=STAGES, default="judging")
    args = parser.parse_args()

    import yaml
    config = yaml.safe_load(open(Path(__file__).resolve().parent.parent / "config.yaml"))

    from_idx = STAGES.index(args.from_stage)
    to_idx = STAGES.index(args.to_stage)

    if from_idx == 0:
        items = json.load(open(PATHS["generation"]))
    else:
        prev_stage = STAGES[from_idx - 1]
        items = json.load(open(PATHS[prev_stage]))

    if from_idx <= 0:
        print("=== do(W) Stage: extraction ===")
        from src.extraction import extract_activations
        items = extract_activations(items, config)
        save_json(items, PATHS["extraction"])
    if to_idx == 0:
        return

    if from_idx <= 1:
        print("=== do(W) Stage: verbalization ===")
        from src.verbalization import verbalize
        items = verbalize(items, config)
        save_json(items, PATHS["verbalization"])
    if to_idx == 1:
        return

    print("=== do(W) Stage: judging ===")
    from src.judging import judge_items
    items = judge_items(items, config)
    save_json(items, PATHS["judging"])
    print("\nNext: python3 scripts/analyze_do_w.py")


if __name__ == "__main__":
    main()
