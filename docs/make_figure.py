"""Render the summary figure used in the project README.

Reads the per-layer *speaker-specificity index* tables in
``results/results_spec_idx/spec_idx/`` (one CSV per wav2vec 2.0 layer,
speakers x vowels) and plots how the index evolves across layers.

The specificity index is a KL-divergence ratio in [0, 1]: it asks whether
a layer's vowel representation for a talker is closer to that talker's own
acoustic vowel distribution (index -> 1) or to the talker-independent,
marginal distribution (index -> 0.5 and below). Values above 0.5 mean the
model encodes *talker-specific* detail rather than only phonetic category.

Run from the repository root:

    python docs/make_figure.py

Output: ``docs/talker_specificity_across_layers.png``
"""
import re
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # headless; no display needed
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
SPEC_DIR = REPO_ROOT / "results" / "results_spec_idx" / "spec_idx"
OUT = Path(__file__).resolve().parent / "talker_specificity_across_layers.png"


def load_layers(spec_dir):
    """Return {layer: DataFrame(speakers x vowels)} from the spec_idx CSVs."""
    layers = {}
    for csv in spec_dir.glob("layer_*_spec_idx.csv"):
        layer = int(re.search(r"layer_(\d+)_", csv.name).group(1))
        layers[layer] = pd.read_csv(csv, index_col=0)
    if not layers:
        raise FileNotFoundError(f"No layer_*_spec_idx.csv found in {spec_dir}")
    return dict(sorted(layers.items()))


def main():
    parser = argparse.ArgumentParser(description="Plot the speaker-specificity index across layers.")
    parser.add_argument("--spec-dir", type=Path, default=SPEC_DIR,
                        help="Directory of layer_<L>_spec_idx.csv files.")
    parser.add_argument("--out", type=Path, default=OUT, help="Output PNG path.")
    args = parser.parse_args()

    layers = load_layers(args.spec_dir)
    xs = list(layers.keys())
    vowels = list(next(iter(layers.values())).columns)

    # Per-layer mean over all speakers x vowels, with a 95% CI of the mean.
    means, cis = [], []
    for layer in xs:
        vals = layers[layer].to_numpy().ravel()
        vals = vals[~np.isnan(vals)]
        means.append(vals.mean())
        cis.append(1.96 * vals.std(ddof=1) / np.sqrt(vals.size))
    means, cis = np.array(means), np.array(cis)

    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Faint per-vowel curves for texture.
    for vowel in vowels:
        vy = [layers[layer][vowel].mean() for layer in xs]
        ax.plot(xs, vy, color="grey", lw=0.8, alpha=0.35)

    ax.plot(xs, means, marker="o", color="#1f77b4", lw=2.5,
            label="Mean over speakers x vowels")
    ax.fill_between(xs, means - cis, means + cis, color="#1f77b4", alpha=0.2)
    ax.axhline(0.5, color="#d62728", lw=1.2, ls="--",
               label="No talker specificity (0.5)")

    ax.set_xlabel("wav2vec 2.0 layer")
    ax.set_ylabel("Speaker-specificity index")
    ax.set_title("Talker-specific detail persists across wav2vec 2.0 layers")
    ax.set_xticks(xs)
    ax.set_ylim(0.45, 0.85)
    ax.legend(frameon=False, loc="upper center")
    ax.grid(True, lw=0.4, alpha=0.5)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
