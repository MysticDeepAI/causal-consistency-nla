# Pre-registration -- Counterfactual Consistency of the NLA Explanation Channel
**Version 1.0 -- [FREEZE DATE: ____] -- commit: [HASH: ____]**

Frozen BEFORE running any experiment beyond Stage 0 (generation). Any
deviation is logged in `deviations.md` with date and justification.

## 1. Hypotheses

- **H1 (channel violation):** the AV's non-descendant-invariance violation
  rate under do(X) [weather] exceeds the calibrated noise floor epsilon_W
  [footwear] (Stage: noise floor calibration).
- **H2 (spurious mechanism):** the AV violation rate is higher when the
  context exhibits the weather-footwear correlation than when it does not,
  with a significant bootstrap-permutation difference (p<0.05, Holm-corrected).
- **H3 (localization):** the AV violation rate exceeds the linear-probe
  violation rate on the same activations (the verbalization channel ADDS
  inconsistency beyond what the representation carries).

## 2. Frozen thresholds

| Quantity | Threshold | Source |
|---|---|---|
| epsilon_V (per variable) | 95th percentile of d_TV between halves of K=10 rollouts, same activation | Noise floor stage |
| Probe: "information present" | accuracy > 0.85 AND selectivity > 0.30, template split | Probing stage |
| Judge: freezing | human agreement >= 0.90 on variable AND value (30 explanations) | Judging stage |
| N per condition | see config.yaml: scm.n_pairs | Generation stage |
| K rollouts | see config.yaml: sampling.k_rollouts | Verbalization stage |
| Significance | bootstrap 10k, alpha=0.05, Holm across main-table cells | Metrics stage |

## 3. Frozen SCM (parameters numerically verified)

Switch mechanism: Z <- (~X & uz0) | (X & uz1) | (W & ub); pz0=0.35,
pz1=0.55, pb=0.45; pC=0.5, pX=pW=0.15; pd=0.55, pe=0.60, pf=0.05.
Verified: corr(X,W)=+0.49 (confounded) vs 0.00 (toggle) with identical
marginals; non-descendant bound = [1,1] exact. Code: `dataset/scm_final.py`.

## 4. Pre-registered decision tree

- GO: probe present at the native layer + AV positive control shows signal
  -> proceed with the main experiment as designed.
- PIVOT A: W is decodable at earlier layers but not at the native layer ->
  "the NLA is structurally blind to variables represented at other layers."
- PIVOT B: AV positive control fails (tried with K=20, majority vote,
  alternative extraction tokens) despite healthy probes -> characterize
  as non-transmission; the design is otherwise unchanged.

## 5. Pre-registered interpretation of the diagnostic chain

Case 2 (probe OK, AV violates): verbalization corrupts. Case 4 (probe
violates): the representation already mixes, and the AV inherits it.
Case 3 (everything consistent): pivot to "reconstruction induces implicit
ctf-consistency." Heterogeneity by variable/template is reported as a
finding, not averaged away.

## 6. What we will NOT do (anti-garden-of-forking-paths)

No adjusting thresholds after seeing results; no adding variables to the
graph; no changing the model's main layer; no excluding pairs except for
documented technical failure (judge fails to parse, malformed text),
reporting the count.
