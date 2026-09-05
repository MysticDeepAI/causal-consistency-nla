"""
Stage 2: activation extraction.

Adapted from the project's vss-colombia/src/extraction.py. Same core
recipe (chat template, hidden_states[layer] -- NOT a hook on
model.model.layers[layer], see the note below on why that's a
one-layer-off bug). Simplified: instead of a 7-position keyword window,
each item only needs the LAST TOKEN of two full prompts -- the factual
arm (context_block + text_factual) and the counterfactual arm
(context_block + text_counterfactual) -- since the design compares the
AV's explanation of one whole scene against its minimally-edited twin,
not a before/after-keyword window within a single scene.

IMPORTANT -- extraction mechanism: uses output_hidden_states=True and
reads hidden_states[layer]. hidden_states[0] is the raw embedding, before
any transformer block, so hidden_states[20] means "after 20 full layers"
-- this must match the convention the AV/AR checkpoints were trained
with (confirmed for the L20 checkpoint from the project's own debugging).
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def extract_activations(items, config):
    """Entry point for this stage.

    items: list of dicts from generation.build_items(), each with at least
           {pair_id, context_block, text_factual, text_counterfactual, ...}
    config: the dict loaded from config.yaml

    Returns: the same list of items, with a new "vectors" field per item:
        {"factual": np.array[d], "counterfactual": np.array[d]}
    """
    print(f"[extraction] loading {config['model']['qwen_path']} ...")
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["qwen_path"])
    model = AutoModelForCausalLM.from_pretrained(
        config["model"]["qwen_path"],
        torch_dtype=torch.float16,
        device_map="cuda",
    )
    model.eval()

    layer = config["model"]["layer"]

    def last_token_activation(text):
        ids = tokenizer(text, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model(**ids, output_hidden_states=True)
        # hidden_states[layer]: after `layer` full transformer blocks;
        # [0] drops the batch dim, [-1] takes the last token position.
        # NOTE (fixed): no chat template here. These are raw completion
        # texts for the BASE model, not chat turns -- wrapping them in
        # apply_chat_template() put the extraction point on the
        # "<|im_start|>assistant\n" scaffold token instead of the end of
        # the actual scene text, which made the AV describe generic
        # assistant-response formatting instead of the scene.
        return out.hidden_states[layer][0, -1].float().cpu().numpy()

    for i, item in enumerate(items, 1):
        full_factual = item["context_block"] + "\n\n" + item["text_factual"]
        full_counterfactual = item["context_block"] + "\n\n" + item["text_counterfactual"]

        item["vectors"] = {
            "factual": last_token_activation(full_factual),
            "counterfactual": last_token_activation(full_counterfactual),
        }

        if i % 20 == 0 or i == len(items):
            print(f"  [{i}/{len(items)}] processed")

    print(f"[extraction] done: {len(items)} items with activations.")
    return items
