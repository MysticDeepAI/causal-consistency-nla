"""
AR-verified salience (ARS) for footwear claims, with a placebo control.

Adapted from the project's vss-colombia/src/ars.py -- same checkpoint
facts, verified there and reused verbatim:
  - AR prompt: "Summary of the following text: <text>{explanation}</text> <summary>"
  - The AR is suffix-anchored (CRITIC_SUFFIX_IDS), not marker-token based
    like the AV -- it reads the LAST TOKEN of that exact prompt tail.
  - Reconstruction = hidden_states[layer_index] (RAW layer output), NOT
    hidden_states[-1] -- the final RMSNorm is randomly initialized in
    this checkpoint and would corrupt the comparison.

ARS_t(a) = MSE(h_real, AR(z_without_claim)) - MSE(h_real, AR(z_original))
A large positive ARS means removing the claim hurt reconstruction against
the REAL activation from Stage 1 -- evidence the claim carried genuine
information, not decorative confabulation.

WHAT'S NEW HERE (this project has no "neutral arm" control like vss did):
for every real-claim removal we ALSO remove a random unrelated word from
the SAME explanation (placebo) and compute ARS_placebo the same way. If
ARS_real doesn't clearly exceed ARS_placebo, the AR is just generically
sensitive to any edit -- not specific evidence the footwear claim mattered.
"""

import re

import numpy as np

AR_PROMPT_TEMPLATE = "Summary of the following text: <text>{explanation}</text> <summary>"
CRITIC_SUFFIX_IDS = [1318, 29, 366, 1708, 29]

FOOTWEAR_PATTERN = re.compile(r"\b(boots?|sneakers?)\b", re.IGNORECASE)


def extract_footwear_quote(text: str):
    """First literal footwear mention (case preserved), or None."""
    m = FOOTWEAR_PATTERN.search(text)
    return m.group(0) if m else None


def _remove_claim(explanation: str, quote: str) -> str:
    """Surgical substring removal -- not sentence-level, to avoid deleting
    other genuine claims that happen to share the same sentence."""
    edited = explanation.replace(quote, "")
    edited = re.sub(r"\s{2,}", " ", edited)
    edited = re.sub(r"\s+([.,;:])", r"\1", edited)
    return edited.strip()


STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "at", "in", "on", "to",
    "of", "and", "or", "it", "its", "this", "that", "with", "for", "as",
    "by", "be", "been", "has", "have", "had", "not", "but", "near", "today",
}


def pick_placebo_word(explanation: str, quote: str, rng):
    """A random CONTENT word elsewhere in the explanation, unrelated to
    the footwear claim -- used as the control removal. Excludes function
    words (stopwords): removing "the" is a much cheaper edit than removing
    a content word like "boots", which would make the placebo comparison
    unfair (biased toward showing the real claim matters more)."""
    words = re.findall(r"\b\w+\b", explanation)
    candidates = [w for w in words
                 if w.lower() not in quote.lower()
                 and w.lower() not in STOPWORDS
                 and len(w) > 2]
    if not candidates:
        return None
    return candidates[rng.integers(0, len(candidates))]


def _load_ar(checkpoint_id):
    """Loads the AR. No separate reconstruction head exists in this
    checkpoint (confirmed by inspection) -- it's the truncated model's own
    hidden state at the suffix-anchored last token position."""
    import os

    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = checkpoint_id if os.path.isdir(checkpoint_id) else snapshot_download(repo_id=checkpoint_id)
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.float16, device_map="cuda")
    model.eval()
    return model, tokenizer


def _reconstruct(ar_model, tokenizer, explanation_text: str, critic_suffix_ids, layer_index: int = 20):
    """Runs one explanation through the AR, returns the reconstructed
    activation (numpy, [d_model]). Asserts the prompt's tail tokens match
    the expected suffix -- a silent mismatch here would corrupt every
    downstream ARS value without erroring."""
    import torch

    prompt = AR_PROMPT_TEMPLATE.format(explanation=explanation_text)
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(ar_model.device)

    tail = ids[0, -len(critic_suffix_ids):].tolist()
    assert tail == critic_suffix_ids, (
        f"AR prompt does not end in the expected suffix. Got {tail}, "
        f"expected {critic_suffix_ids}.")

    with torch.no_grad():
        out = ar_model(input_ids=ids, output_hidden_states=True)
    last_token_hidden = out.hidden_states[layer_index][0, -1]
    return last_token_hidden.float().cpu().numpy()


def compute_ars_sample(pair_id, arm, real_vector, explanation_text, ar_model,
                       tokenizer, rng, layer_index: int = 20):
    """One spot-check: real-claim removal vs placebo removal, both scored
    by reconstruction MSE against the real activation. Returns None if the
    text has no footwear mention or no viable placebo word."""
    quote = extract_footwear_quote(explanation_text)
    if quote is None:
        return None
    placebo_word = pick_placebo_word(explanation_text, quote, rng)
    if placebo_word is None:
        return None

    edited_real = _remove_claim(explanation_text, quote)
    edited_placebo = _remove_claim(explanation_text, placebo_word)

    h_orig = _reconstruct(ar_model, tokenizer, explanation_text, CRITIC_SUFFIX_IDS, layer_index)
    h_real_edit = _reconstruct(ar_model, tokenizer, edited_real, CRITIC_SUFFIX_IDS, layer_index)
    h_placebo_edit = _reconstruct(ar_model, tokenizer, edited_placebo, CRITIC_SUFFIX_IDS, layer_index)

    real_vector = np.asarray(real_vector)
    mse_orig = float(np.mean((real_vector - h_orig) ** 2))
    mse_real_edit = float(np.mean((real_vector - h_real_edit) ** 2))
    mse_placebo_edit = float(np.mean((real_vector - h_placebo_edit) ** 2))

    return {"pair_id": pair_id, "arm": arm, "quote": quote, "placebo_word": placebo_word,
            "ars_real": mse_real_edit - mse_orig, "ars_placebo": mse_placebo_edit - mse_orig}
