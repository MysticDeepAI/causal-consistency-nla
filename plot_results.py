"""
Reads the finished results.json and produces the paper/sprint figures.
Deliberately decoupled from main.py (per vss-colombia convention):
regenerating figures never requires GPU time.

Figures produced (see plan_maestro.md for the rationale behind each):
  1. noise_floor.png       -- null d_TV distribution with epsilon marked
  2. violation_rates.png   -- AV vs probe violation rate, with bootstrap CIs
  3. h2_contrast.png       -- violation rate: confounded vs not-confounded context
  4. localization_heatmap.png -- selectivity, variables x layers (probing stage)
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml


def plot_noise_floor(null_dist: np.ndarray, epsilon: float, out_path: str):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(null_dist, bins=40, color="#888", alpha=0.8)
    ax.axvline(epsilon, color="crimson", linestyle="--",
               label=f"epsilon (95th pct) = {epsilon:.3f}")
    ax.set_xlabel("d_TV between rollout halves of the SAME activation")
    ax.set_ylabel("count")
    ax.set_title("Noise floor: AV stochasticity with no intervention")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_violation_rates(av_ci: dict, probe_ci: dict, out_path: str):
    fig, ax = plt.subplots(figsize=(5, 4))
    labels = ["Probe\n(representation)", "AV\n(verbalization)"]
    rates = [probe_ci["rate"], av_ci["rate"]]
    los = [probe_ci["rate"] - probe_ci["lo"], av_ci["rate"] - av_ci["lo"]]
    his = [probe_ci["hi"] - probe_ci["rate"], av_ci["hi"] - av_ci["rate"]]
    ax.bar(labels, rates, yerr=[los, his], capsize=6, color=["#4c72b0", "#c44e52"])
    ax.set_ylabel("non-descendant invariance violation rate")
    ax.set_title("Where does the violation come from?")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_h2_contrast(confounded_ci: dict, notconf_ci: dict, out_path: str):
    fig, ax = plt.subplots(figsize=(5, 4))
    labels = ["No correlation\nin context", "Correlation\nin context"]
    rates = [notconf_ci["rate"], confounded_ci["rate"]]
    los = [notconf_ci["rate"] - notconf_ci["lo"], confounded_ci["rate"] - confounded_ci["lo"]]
    his = [notconf_ci["hi"] - notconf_ci["rate"], confounded_ci["hi"] - confounded_ci["rate"]]
    ax.bar(labels, rates, yerr=[los, his], capsize=6, color=["#55a868", "#c44e52"])
    ax.set_ylabel("AV violation rate")
    ax.set_title("Does an in-context correlation make it worse? (H2)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_localization_heatmap(probe_results: list, out_path: str):
    variables = sorted(set(r["variable"] for r in probe_results))
    layers = sorted(set(r["position_layer"] for r in probe_results))
    grid = np.zeros((len(variables), len(layers)))
    for r in probe_results:
        i = variables.index(r["variable"])
        j = layers.index(r["position_layer"])
        grid[i, j] = max(grid[i, j], r["selectivity"])

    fig, ax = plt.subplots(figsize=(1.2 * len(layers) + 2, 1 + len(variables)))
    im = ax.imshow(grid, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layers, rotation=45, ha="right")
    ax.set_yticks(range(len(variables)))
    ax.set_yticklabels(variables)
    ax.set_title("Localization: selectivity by variable x layer/position")
    fig.colorbar(im, ax=ax, label="selectivity (acc - control)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    cfg = yaml.safe_load(open(Path(__file__).parent / "config.yaml"))
    rd = Path(cfg["paths"]["results_dir"])
    fd = Path(cfg["paths"]["figures_dir"])
    fd.mkdir(parents=True, exist_ok=True)

    results_path = rd / "results.json"
    if not results_path.exists():
        print(f"[plot_results] {results_path} not found yet -- run main.py first.")
        return

    results = json.load(open(results_path))
    # NOTE: exact key names depend on how metrics.run() assembles results.json;
    # wire the plot_* calls above to those keys once metrics.py is filled in.
    print(f"[plot_results] loaded {results_path}, figures -> {fd}")


if __name__ == "__main__":
    main()
