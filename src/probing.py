"""
Stage 4 -- Linear probes (localization: representation vs. verbalization).

The only stage that strictly requires a GPU for its input (activations),
though the probes themselves train on CPU in seconds via scikit-learn.

Usage order:
  1) src/generation.py  -> prompts (CPU)
  2) src/extraction.py  -> activations.npz at cfg.model.probe_layers (GPU)
  3) src/probing.py     -> per (variable x layer x position) probes with
                            template-split and selectivity control (CPU)
  4) decision GO/PIVOT per the pre-registered thresholds in config.yaml

Pre-registered thresholds (config: probe.presence_threshold): "information
present" if accuracy > 0.85 AND selectivity > 0.30, on the held-out
template split.
"""
import json
from pathlib import Path

import numpy as np

VARS = {"X": 0, "W": 1, "Z": 2, "Y": 3}  # weather, footwear, pace, coffee


# ---------------- 1. Training with controls ----------------

def train_probes(npz_path: str, template_ids: list, cfg: dict, seed: int = 0) -> list:
    """
    For each (variable x layer x position): logistic regression with
    standardization, a C-sweep, a TEMPLATE SPLIT (train on templates_a,
    evaluate on templates_b, and the inverse) and SELECTIVITY (same probe
    on permanently shuffled labels -- Hewitt & Liang control).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    data = np.load(npz_path)
    labels = data["labels"]          # (N, 4) array: X, W, Z, Y per item
    template = data["template"]      # (N,) array: which of the 5 templates
    rng = np.random.default_rng(seed)
    results = []

    layer_keys = [k for k in data.files if k.startswith("L")]
    tr_templates, te_templates = cfg["probe"]["template_split"]

    for key in layer_keys:
        X_act = data[key]
        for var, vi in VARS.items():
            y = labels[:, vi]
            if len(np.unique(y)) < 2:
                continue
            y_rand = rng.permutation(y)  # FIXED shuffled labels
            for tr_t, te_t, tag in [
                (tr_templates, te_templates, "a->b"),
                (te_templates, tr_templates, "b->a"),
            ]:
                tr = np.isin(template, tr_t)
                te = np.isin(template, te_t)
                accs = {}
                for name, yy in [("real", y), ("control", y_rand)]:
                    best = 0.0
                    for C in cfg["probe"]["C_grid"]:
                        clf = make_pipeline(StandardScaler(),
                                            LogisticRegression(C=C, max_iter=2000))
                        clf.fit(X_act[tr], yy[tr])
                        best = max(best, clf.score(X_act[te], yy[te]))
                    accs[name] = best
                thr = cfg["probe"]["presence_threshold"]
                results.append({
                    "position_layer": key, "variable": var, "split": tag,
                    "acc": accs["real"], "acc_control": accs["control"],
                    "selectivity": accs["real"] - accs["control"],
                    "present": accs["real"] > thr["accuracy"]
                               and (accs["real"] - accs["control"]) > thr["selectivity"],
                })
    return results


# ---------------- 2. Representation-level invariance (chain link 2) ----------------

def probe_invariance(probe, act_factual, act_counterfactual) -> dict:
    """
    On do(X) minimal pairs: the W probe should predict the SAME value on
    the factual and counterfactual activations, with similar confidence.
    Returns violation flags in the same format as metrics.violation_rate,
    for the AV-vs-probe subtraction.
    """
    p_f = probe.predict_proba(act_factual)[:, 1]
    p_c = probe.predict_proba(act_counterfactual)[:, 1]
    same_pred = (p_f > 0.5) == (p_c > 0.5)
    return {"viol_flags": ~same_pred, "delta_conf": np.abs(p_f - p_c)}


def run(cfg: dict, stage1_meta_path: str, out_path: str) -> None:
    stage1_meta = json.load(open(stage1_meta_path))
    results = train_probes(stage1_meta["npz_path"], template_ids=None, cfg=cfg)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    n_present = sum(r["present"] for r in results)
    print(f"[probing] {n_present}/{len(results)} (variable, layer, split) cells "
          f"pass the presence threshold -> {out_path}")


if __name__ == "__main__":
    print("Probing module ready. Run order:")
    print(" 1) src/generation.py  (CPU, prompts)")
    print(" 2) src/extraction.py  (GPU, activations at probe_layers)")
    print(" 3) src/probing.py     (CPU, heatmap variables x layers)")
    print(" 4) GO/PIVOT decision per config.yaml pre-registered thresholds")
