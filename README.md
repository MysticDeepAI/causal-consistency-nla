# Causal Consistency of NLA Explanations

A quantitative audit of whether Anthropic's Natural Language Autoencoder
(NLA) explanation channel respects causal structure -- or leaks spurious
observational correlations -- when describing Qwen2.5-7B activations,
using a synthetic Structural Causal Model with known ground truth and a
claim-level total-variation-distance metric.

## The core idea, in one paragraph

The NLA's Activation Verbalizer (AV) is an encoder from activations to
text; its Activation Reconstructor (AR) is the matching decoder. Nothing
in the NLA's training objective (pure activation reconstruction)
constrains the *causal* content of its explanations. We build a synthetic
world (weather -> pace/coffee-stop <- footwear; weather and footwear
independent by construction) and check: when we intervene on weather in a
minimal text pair, does the AV's claim about footwear (a non-descendant)
stay put (H1), does that hold up worse when the context shows a spurious
weather<->footwear correlation (H2), and -- as a sanity check on the whole
design -- do claims about the true causal descendants (pace, coffee-stop)
actually respond to the intervention, ruling out an AV that's simply
inert to everything?

## Repository structure

```
causal-consistency-nla/
├── main.py                     orchestrates the 5-stage main pipeline
├── plot_results.py             reads results.json, generates figures (no GPU)
├── validate_all_stages.py      CPU-only smoke test on a tiny subset
├── config.yaml                 all experiment parameters
├── environment.yml             conda environment specification
├── dataset/
│   └── scm_final.py            frozen synthetic SCM: minimal pairs + context blocks
├── src/
│   ├── generation.py           Stage 1 -- builds items from the SCM (CPU)
│   ├── extraction.py           Stage 2 -- Qwen2.5-7B activations (GPU)
│   ├── verbalization.py        Stage 3 -- the AV describes each activation (GPU)
│   ├── judging.py              Stage 4 -- keyword classifiers for W, Z, Y (CPU)
│   ├── metrics.py              Stage 5 -- noise floor, violation rates, H1/H2, descendant control (CPU)
│   ├── probing.py              (optional) linear-probe localization, not required for the main result
│   └── ars.py                  AR-verified salience: is a claim load-bearing or decorative?
├── scripts/
│   ├── noise_floor_check.py    collects K=10 same-activation rollouts (GPU)
│   ├── compute_noise_floor.py  turns those into epsilon per variable (CPU)
│   └── ars_check.py            AR spot-check across all committed footwear claims (GPU, fast)
└── results/                    all checkpoints and final outputs land here
```

## Installation

```bash
conda env create -f environment.yml
conda activate causal-consistency-nla
python -c "import torch; print(torch.cuda.is_available())"   # sanity check
```

If `conda activate` doesn't stick in your shell, call the environment's
Python directly instead of fighting the shell hook:
```bash
/path/to/anaconda3/envs/causal-consistency-nla/bin/python3 main.py ...
```

Set `config.yaml`'s `model.qwen_path`, `model.av_checkpoint`,
`model.ar_checkpoint` to your real checkpoints before running anything.

## The four experiments, and exactly how to run each

Everything below assumes you're in the repo root with the conda env
active. `CUDA_VISIBLE_DEVICES=N` picks a free GPU on a shared cluster --
check `nvidia-smi` first.

### 1. Main pipeline (H1, H2, and the descendant-sensitivity control)

Five stages, each checkpointed to `results/results_stageN.json` so a
crash never loses completed work. Run them in order:

```bash
# Stage 1 -- build the 300 minimal-pair items from the SCM (CPU, seconds)
python3 main.py --to generation

# Stage 2 -- extract Qwen activations (GPU)
CUDA_VISIBLE_DEVICES=0 python3 main.py --from extraction --to extraction \
    --input results/results_stage0.json

# Stage 3 -- the AV describes each activation, K rollouts (GPU, the slow one)
CUDA_VISIBLE_DEVICES=0 python3 main.py --from verbalization --to verbalization \
    --input results/results_stage1.json

# Stage 4 -- classify claims for W, Z, Y from the raw explanations (CPU)
python3 main.py --from judging --to judging --input results/results_stage2.json
```

**Before Stage 5**, you need the noise-floor epsilons (next section) --
Stage 5 refuses to run without `results/epsilon.json` on disk, rather
than silently guessing a threshold.

```bash
# Stage 5 -- violation rates (H1/H2) + descendant-sensitivity control (CPU)
python3 main.py --from metrics --to metrics --input results/results_stage3.json
```

Reading the output: the "Conditional (BOT-excluded) comparison" block is
the properly-powered H1/H2 test -- read that, not the raw block above it
(the raw metric is diluted by AV silence and kept only for comparison).
The final "Positive control: descendant sensitivity" block should show a
HIGH responsiveness rate for Z and Y; if it doesn't, the W result is not
informative (see Limitations below).

### 2. Noise floor calibration (must run before Stage 5, above)

Repeats K=10 AV rollouts on the SAME activation (zero intervention) to
calibrate how much claim distributions wobble on their own -- the
yardstick every violation/responsiveness rate is measured against.

```bash
# Collect raw texts from repeated sampling of the same activations (GPU)
CUDA_VISIBLE_DEVICES=0 python3 scripts/noise_floor_check.py

# Turn them into epsilon per variable: W, Z, Y, each raw + conditional (CPU)
python3 scripts/compute_noise_floor.py
```

This writes `results/epsilon.json` with keys `W`, `W_conditional`,
`Z_conditional`, `Y_conditional`, etc. -- consumed by `metrics.py`.

### 3. AR-verified salience (is a footwear claim load-bearing or decorative?)

For every (item, arm) where the AV committed to a footwear value, removes
that claim vs. a random placebo word from the same text, reconstructs
both through the AR, and checks which hurts reconstruction more against
the real activation. No `generate()` calls -- just forward passes, so
this is fast even across hundreds of instances.

```bash
CUDA_VISIBLE_DEVICES=0 python3 scripts/ars_check.py
```

Requires `results/results_stage1.json` (has the real activations) and
`results/results_stage3.json` (has the explanations) already present --
i.e., run this after Stages 1-4 of the main pipeline.

### 4. Sanity checks (run these FIRST on a new setup, always CPU-cheap)

```bash
python3 validate_all_stages.py     # SCM invariance asserts on a tiny subset
```

If this fails, nothing downstream is trustworthy -- fix it before
spending any GPU time.

## Reading `results/results.json` (Stage 5 output)

Top-level keys: `overall_violation_rate`, `confounded_violation_rate`,
`not_confounded_violation_rate`, `h2_contrast` (all on the raw metric --
diluted, kept for comparison only); `conditional_confounded_violation_rate`,
`conditional_not_confounded_violation_rate`, `conditional_h2_contrast`
(the metric that actually matters, Wilson CIs for zero-count safety);
`descendant_sensitivity` (dict keyed `"Z"`/`"Y"`, each with a
`responsiveness_rate` -- the positive control).

## Known limitations (see also `notes/preregistro.md`)

- Footwear mention rate is the real bottleneck on statistical power --
  only pairs where the AV commits to a value on both sides of a pair are
  usable, and that fraction depends heavily on template surface form
  (prose beat structured/form-style text by a wide margin; see the
  template-iteration story in the write-up).
- H2 (does the violation rate increase under a sown spurious correlation)
  is untested rather than refuted whenever the conditional violation rate
  is 0/0 on both sides -- there's nothing to contrast.
- Single model (Qwen2.5-7B), single layer (the NLA's native training
  layer), one synthetic domain.
- The Z/Y keyword classifiers are phrase-based heuristics with known
  blind spots (documented in `src/judging.py`) -- spot-check a sample
  before trusting them at face value, same as the footwear classifier.

## References

- Fraser-Taliente et al. (2026), *Natural Language Autoencoders*, the NLA
  architecture and open checkpoints this project audits.
- Pan & Bareinboim (2024, 2025), *Counterfactual Image Editing* (I) and
  *...with Disentangled Causal Latent Space* (II) -- the counterfactual-
  consistency framework adapted here from image generative factors to
  text-realized domain SCMs.
- Bynum & Cho (2024), *Language Models as Causal Effect Generators*,
  arXiv:2411.08019 -- the LLM-realized-SCM idea behind `scm_final.py`.
- Plečko, Okanović, Havaldar, Hoefler & Bareinboim (2025), *Epidemiology
  of Large Language Models*, arXiv:2511.03070 -- motivates why an LLM-based
  explanation channel might not respect causal structure even when it
  gets the surface text right.
