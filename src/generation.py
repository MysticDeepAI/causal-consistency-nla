"""
Stage 0 -- Prompt generation.

Builds the full prompt set for the main experiment from the frozen SCM
(dataset/scm_final.py): minimal pairs (do(X) on weather) crossed with two
context conditions (confounded / not confounded), rotated across the 5
surface templates. Nothing here touches the model -- this stage is pure
CPU and produces the exact prompts that Stage 1 (extraction.py) will run
through Qwen.

Output: results/results_stage0.json
  {
    "config_snapshot": {...},
    "items": [
      {
        "pair_id": "P0000",
        "template": 0,
        "context_condition": "confounded" | "not_confounded",
        "context_block": "...",
        "text_factual": "...",
        "text_counterfactual": "...",
        "v_factual": [X, W, Z, Y],
        "v_counterfactual": [X, W, Z, Y],
        "do": ["X", 0, 1]
      },
      ...
    ]
  }
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "dataset"))
from scm_final import FROZEN, FROZEN_NOCONF, minimal_pair, context_block  # noqa: E402


def build_items(cfg: dict) -> list:
    """Entry point for this stage. Returns a flat list of item dicts --
    same convention as extraction.py/verbalization.py: each stage takes
    `items` and returns `items`, main.py handles saving between stages."""
    scm_cfg = cfg["scm"]
    rng = random.Random(scm_cfg["seed"])

    items = []
    conditions = [("confounded", FROZEN), ("not_confounded", FROZEN_NOCONF)]

    for cond_name, params in conditions:
        for i in range(scm_cfg["n_pairs"]):
            tmpl_idx = i % scm_cfg["n_templates"]
            pair = minimal_pair(params, tmpl_idx, rng)
            ctx = context_block(params, scm_cfg["n_context_records"], rng)
            items.append({
                "pair_id": f"{cond_name[:4].upper()}_{i:04d}",
                "template": tmpl_idx,
                "context_condition": cond_name,
                "context_block": ctx,
                "text_factual": pair["text"],
                "text_counterfactual": pair["text_ctf"],
                "v_factual": list(pair["v"]),
                "v_counterfactual": list(pair["v_ctf"]),
                "do": list(pair["do"]),
            })

    print(f"[generation] built {len(items)} items")
    return items


if __name__ == "__main__":
    import yaml
    cfg = yaml.safe_load(open(Path(__file__).resolve().parent.parent / "config.yaml"))
    items = build_items(cfg)
    print(json.dumps(items[0], indent=2))
