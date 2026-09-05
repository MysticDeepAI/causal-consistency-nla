"""
Noise floor -- COLLECTION step.

Repeats K=10 AV rollouts on the SAME activation (no intervention at all)
and saves the raw texts to disk. Analysis (turning these into an epsilon
via qhat/d_TV -- the same metric used for real violations, not an ad-hoc
majority-vote check) happens separately in compute_noise_floor.py, so
both use the exact same yardstick.

Usage:
    CUDA_VISIBLE_DEVICES=3 python3 scripts/noise_floor_check.py
"""
import json
import re
import sys
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.verbalization import (_build_base_prompt, _load_av_config,
                               _load_embeddings, _resolve_local_path,
                               _row_with_injected_vector)

K = 10
N_ACTIVATIONS = 60  # both arms of all pilot items, for a firm epsilon


def main():
    cfg = yaml.safe_load(open(Path(__file__).resolve().parent.parent / "config.yaml"))
    items = json.load(open("results/results_stage1.json"))

    checkpoint = _resolve_local_path(cfg["model"]["av_checkpoint"])
    meta = _load_av_config(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    embed_weights = _load_embeddings(checkpoint)
    base_ids = _build_base_prompt(meta, tokenizer)
    av_model = AutoModelForCausalLM.from_pretrained(
        checkpoint, torch_dtype=torch.float16, device_map="cuda")
    av_model.eval()

    temperature = cfg["sampling"]["av_temperature"]
    max_tokens = cfg["sampling"]["max_new_tokens"]

    all_vectors = []
    for it in items:
        all_vectors.append((f"{it['pair_id']}_factual", it["vectors"]["factual"]))
        all_vectors.append((f"{it['pair_id']}_counterfactual", it["vectors"]["counterfactual"]))

    collected = []
    for label, vector in all_vectors[:N_ACTIVATIONS]:
        row = _row_with_injected_vector(embed_weights, base_ids, vector, meta)
        device = next(av_model.parameters()).device
        dtype = next(av_model.parameters()).dtype
        embeds = row.unsqueeze(0).to(device=device, dtype=dtype)

        with torch.no_grad():
            output = av_model.generate(
                inputs_embeds=embeds, max_new_tokens=max_tokens, do_sample=True,
                temperature=temperature, num_return_sequences=K,
                pad_token_id=tokenizer.eos_token_id)

        texts = []
        for j in range(K):
            text = tokenizer.decode(output[j], skip_special_tokens=False)
            m = re.search(r"<explanation>\s*(.*?)\s*</explanation>", text, re.DOTALL)
            texts.append(m.group(1).strip() if m else text)

        collected.append({"label": label, "texts": texts})
        print(f"{label}: collected {K} rollouts")

    out_path = "results/noise_floor_texts.json"
    with open(out_path, "w") as f:
        json.dump(collected, f, indent=2)
    print(f"\nSaved raw texts -> {out_path}")
    print("Next: python3 scripts/compute_noise_floor.py")


if __name__ == "__main__":
    main()
