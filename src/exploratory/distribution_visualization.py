"""Representational-distance analysis across wav2vec 2.0 layers.

Reduced from a large scratch file. It keeps the two coherent pieces:

1. ``layer_vowel_gaussians`` — fit a 3-D Gaussian to each vowel's LDA
   components within every layer (from precomputed
   ``wav_2_vec_hidden_lda/lda_acts_<layer>.pkl``).
2. ``consecutive_layer_kld`` — the *distance* measure: KL divergence between
   layer L and layer L+1 for each vowel (how much a vowel's representation
   drifts from one layer to the next), using ``kld_calculate``.

NOTE on the "new measure" of talker/phone representation relationship: that is
the **speaker-specificity index** (a KL ratio comparing model vs. speaker-
specific vs. marginal acoustics), computed by
``get_spec_idx_for_every_layer`` and shipped as the tracked
``results/results_spec_idx/`` tables + ``docs/make_figure.py``. The metric here
is a related but different quantity (layer-to-layer drift), kept for analysis.

Exploratory: expects precomputed per-layer LDA pickles; not wired into
``run_pipeline.py``.
"""
import argparse

from spec_idx import get_spec_idx_for_every_layer, kld_calculate
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import multivariate_normal


COMPONENTS = ["lda_1", "lda_2", "lda_3"]


def layer_vowel_gaussians(lda_dir, num_layers):
    """{layer: {vowel: multivariate_normal}} over LDA components."""
    dists = {}
    for layer in range(1, num_layers + 1):
        df = pd.read_pickle(Path(lda_dir) / f"lda_acts_{layer}.pkl")
        missing = [c for c in COMPONENTS if c not in df.columns]
        if missing:
            print(f"Layer {layer}: missing {missing}, skipping.")
            continue
        per_vowel = {}
        for vowel in sorted(df["phone"].dropna().unique()):
            X = df.loc[df["phone"] == vowel, COMPONENTS].dropna().to_numpy()
            if len(X) < 4:
                continue
            per_vowel[vowel] = multivariate_normal(
                X.mean(0), np.cov(X, rowvar=False), allow_singular=True)
        dists[layer] = per_vowel
    return dists


def consecutive_layer_kld(gaussians):
    """DataFrame of KL(layer L || layer L+1) per vowel."""
    rows = []
    layers = sorted(gaussians)
    for l in layers[:-1]:
        d1, d2 = gaussians[l], gaussians.get(l + 1, {})
        for vowel in sorted(set(d1) & set(d2)):
            rows.append({
                "layer_from": l,
                "layer_to": l + 1,
                "vowel": vowel,
                "kl": float(kld_calculate(d1[vowel], d2[vowel])),
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["layer_pair"] = df["layer_from"].astype(str) + "_" + df["layer_to"].astype(str)
    return df


def plot_consecutive_layer_kld(kl_df, out_path):
    plt.figure(figsize=(8, 5))
    for vowel, sub in kl_df.groupby("vowel"):
        plt.plot(sub["layer_pair"], sub["kl"], marker="o", label=vowel)
    plt.xlabel("Layer pair (L -> L+1)")
    plt.ylabel("KL divergence")
    plt.title("Representation drift between consecutive layers, per vowel")
    plt.legend(title="Vowel", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Cross-layer KLD distance per vowel.")
    parser.add_argument("--lda-dir", default="./wav_2_vec_hidden_lda",
                        help="Folder of per-layer lda_acts_<layer>.pkl files.")
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--out", default="./figures/consecutive_layer_kld.png")
    args = parser.parse_args()

    gaussians = layer_vowel_gaussians(args.lda_dir, args.num_layers)
    kl_df = consecutive_layer_kld(gaussians)
    if kl_df.empty:
        print("No overlapping vowels across layers; nothing to plot.")
        return
    plot_consecutive_layer_kld(kl_df, args.out)


if __name__ == "__main__":
    main()
