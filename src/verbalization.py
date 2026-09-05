"""
Stage 3: verbalization.

Adapted from the project's vss-colombia/src/verbalization.py. The
injection recipe (nla_meta.yaml-driven prompt/scale/marker-token, raw
embedding-matrix loading, batching same-length rows) is copied UNCHANGED
-- it is generic over however many positions an item has. Only the
top-level verbalize() loop changed: 2 positions per item (factual /
counterfactual) instead of 7 (tw-1..tw+5).

Injection recipe: kitft/natural_language_autoencoders, docs/inference.md.
  1. tokenize the AV's fixed prompt (from nla_meta.yaml, never hardcoded)
  2. rescale the vector to injection_scale (also from nla_meta.yaml)
  3. replace a marker token's embedding with the rescaled vector, and let
     the AV generate text from there
"""

import json
import os
import re

import torch
import yaml
from huggingface_hub import snapshot_download
from safetensors import safe_open


def _resolve_local_path(checkpoint):
    """Translates a checkpoint identifier into an actual folder on disk
    (needed to open nla_meta.yaml and the raw embedding weights by hand,
    which from_pretrained() doesn't expose a path for)."""
    if os.path.isdir(checkpoint):
        return checkpoint
    return snapshot_download(repo_id=checkpoint)


def _load_av_config(checkpoint_dir):
    """Loads the AV's 'contract': prompt, special token IDs, and scale
    factor. Never hardcoded -- read fresh from the checkpoint."""
    with open(f"{checkpoint_dir}/nla_meta.yaml") as f:
        return yaml.safe_load(f)


def _load_embeddings(checkpoint_dir):
    """Loads only the AV's embedding matrix. Handles both sharded
    (model.safetensors.index.json) and single-file checkpoints."""
    index_path = os.path.join(checkpoint_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            index = json.load(f)
        weight_map = index.get("weight_map", {})
        key = next((k for k in weight_map if "embed_tokens.weight" in k), None)
        if key is None:
            raise RuntimeError(
                "Could not find 'embed_tokens.weight' in the checkpoint index. "
                f"Available keys (sample): {list(weight_map.keys())[:10]}")
        weights_file = weight_map[key]
        with safe_open(os.path.join(checkpoint_dir, weights_file), framework="pt") as f:
            return f.get_tensor(key)

    files = sorted(f for f in os.listdir(checkpoint_dir) if f.endswith(".safetensors"))
    if not files:
        raise RuntimeError(f"No .safetensors file found in {checkpoint_dir}")

    seen_keys = []
    for file in files:
        with safe_open(os.path.join(checkpoint_dir, file), framework="pt") as f:
            keys = list(f.keys())
            seen_keys.extend(keys)
            key = next((k for k in keys if "embed_tokens.weight" in k), None)
            if key is not None:
                return f.get_tensor(key)

    raise RuntimeError(
        "No key containing 'embed_tokens.weight' was found in any shard of "
        f"{checkpoint_dir}. Sample keys found: {seen_keys[:15]}")


def _build_base_prompt(meta, tokenizer):
    """Tokenizes the AV's fixed prompt ONCE. Two steps (format to string,
    then tokenize separately), not a single tokenize=True call -- see the
    original notebook's note on injection-marker misalignment."""
    content = meta["prompt_templates"]["av"].format(
        injection_char=meta["tokens"]["injection_char"])
    fmt_string = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False, add_generation_prompt=True,
    )
    return tokenizer.encode(fmt_string, add_special_tokens=False)


def _row_with_injected_vector(embed_weights, base_ids, raw_vector, meta):
    """Rescales ONE vector and returns the [T, d] embedding row with that
    vector placed at the marker token's position."""
    embeds = embed_weights[base_ids].float()

    v = torch.tensor(raw_vector, dtype=torch.float32)
    scale = meta["extraction"]["injection_scale"]
    scaled_v = v * (scale / v.norm().clamp_min(1e-12))

    injection_tok = meta["tokens"]["injection_token_id"]
    left_neighbor = meta["tokens"]["injection_left_neighbor_id"]
    right_neighbor = meta["tokens"]["injection_right_neighbor_id"]

    found = False
    for p in range(1, len(base_ids) - 1):
        if (base_ids[p] == injection_tok and base_ids[p - 1] == left_neighbor
                and base_ids[p + 1] == right_neighbor):
            embeds[p] = scaled_v
            found = True
            break
    assert found, "Could not find the injection position in the AV prompt."

    return embeds


def _build_batch(embed_weights, base_ids, vectors_by_position, meta):
    """Stacks one row per position. Generic over how many positions there
    are -- for us that's {"factual": ..., "counterfactual": ...}, i.e. 2."""
    names = list(vectors_by_position.keys())
    rows = [_row_with_injected_vector(embed_weights, base_ids, vectors_by_position[n], meta)
           for n in names]
    return names, torch.stack(rows, dim=0)


def _generate_batch(av_model, tokenizer, embeds_batch, k, temperature, max_tokens):
    """One generate() call producing K samples per row via
    num_return_sequences. Output comes back grouped contiguously by row."""
    device = next(av_model.parameters()).device
    dtype = next(av_model.parameters()).dtype
    embeds_batch = embeds_batch.unsqueeze(0) if embeds_batch.dim() == 2 else embeds_batch
    embeds_batch = embeds_batch.to(device=device, dtype=dtype)

    with torch.no_grad():
        output = av_model.generate(
            inputs_embeds=embeds_batch,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            num_return_sequences=k,
            pad_token_id=tokenizer.eos_token_id,
        )

    n_rows = embeds_batch.shape[0]
    output = output.view(n_rows, k, -1)

    texts = []
    for i in range(n_rows):
        samples = []
        for j in range(k):
            text = tokenizer.decode(output[i, j], skip_special_tokens=False)
            m = re.search(r"<explanation>\s*(.*?)\s*</explanation>", text, re.DOTALL)
            samples.append(m.group(1).strip() if m else text)
        texts.append(samples)
    return texts


def verbalize(items, config):
    """Entry point for this stage.

    items: output of extraction.extract_activations() -- each item has
           item["vectors"] = {"factual": np.array, "counterfactual": np.array}
    config: the dict loaded from config.yaml

    Returns: the same items, with a new "explanations" field per item:
        {"factual": [text_1, ..., text_K], "counterfactual": [...]}
    """
    checkpoint_id = config["model"]["av_checkpoint"]
    print(f"[verbalization] loading AV from {checkpoint_id} ...")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    checkpoint = _resolve_local_path(checkpoint_id)
    print(f"  -> resolved local path: {checkpoint}")
    meta = _load_av_config(checkpoint)

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    test_ids = tokenizer.encode(meta["tokens"]["injection_char"], add_special_tokens=False)
    assert test_ids == [meta["tokens"]["injection_token_id"]], (
        f"The injection character doesn't tokenize to the expected ID: "
        f"got {test_ids}, expected [{meta['tokens']['injection_token_id']}]")

    embed_weights = _load_embeddings(checkpoint)
    base_ids = _build_base_prompt(meta, tokenizer)
    av_model = AutoModelForCausalLM.from_pretrained(
        checkpoint, torch_dtype=torch.float16, device_map="cuda")
    av_model.eval()

    k = config["sampling"]["k_samples"]
    temperature = config["sampling"]["av_temperature"]
    max_tokens = config["sampling"]["max_new_tokens"]
    batch_positions = config["sampling"].get("batch_across_positions", True)

    total = len(items)
    verified_batch_order = False

    for i, item in enumerate(items, 1):
        item["explanations"] = {}

        if batch_positions:
            names, batch = _build_batch(embed_weights, base_ids, item["vectors"], meta)
            try:
                texts = _generate_batch(av_model, tokenizer, batch, k, temperature, max_tokens)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                raise RuntimeError(
                    "Not enough VRAM to batch factual+counterfactual together. "
                    "Set sampling.batch_across_positions: false in config.yaml.")

            if not verified_batch_order:
                assert len(texts) == len(names) == batch.shape[0], (
                    "Output row count doesn't match input positions -- "
                    "set batch_across_positions: false and report this error.")
                verified_batch_order = True

            for name, samples in zip(names, texts):
                item["explanations"][name] = samples
        else:
            for name, vector in item["vectors"].items():
                row = _row_with_injected_vector(embed_weights, base_ids, vector, meta)
                texts = _generate_batch(av_model, tokenizer, row.unsqueeze(0),
                                        k, temperature, max_tokens)
                item["explanations"][name] = texts[0]

        if i % 10 == 0 or i == total:
            print(f"  [{i}/{total}] items verbalized ({i * 2 * k}/{total * 2 * k} explanations approx.)")

    print("[verbalization] done.")
    return items
