"""Compute the speaker-specificity index (KLD ratio) for every model layer.

This is the entry point for the project's headline talker/phone measure. For
each wav2vec 2.0 layer it LDA-reduces the hidden activations, fits per-(speaker,
vowel) Gaussians, and forms the KL-divergence ratio

    idx = KL(model || acoustic_speaker)
          / ( KL(model || acoustic_marginal) + KL(model || acoustic_speaker) )

writing one ``layer_<L>_spec_idx.csv`` (speakers x vowels) per layer — the same
format shipped under ``results/results_spec_idx/``. Plot with
``docs/make_figure.py``.

Run from inside ``src/``:

    python compute_spec_idx.py \\
        --formant_df ../data/processed/formant.pkl \\
        --hid-act-folder ../data/processed/hidden_states \\
        --out-dir ../data/processed/spec_idx

The formant table must carry per-(filename, time) f1/f2/f3 + phone/speaker/rep
(see get_phone_tier.py / acoustics.get_formant). pca_f1..3 columns are derived
here if absent.
"""
import argparse

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from acoustics import filter_out_irrelevant_rows
from spec_idx import get_spec_idx_for_every_layer


def ensure_formant_pca(formant_df):
    """Add pca_f1..3 (PCA of f1/f2/f3) if they are not already present."""
    if all(c in formant_df.columns for c in ["pca_f1", "pca_f2", "pca_f3"]):
        return formant_df
    df = filter_out_irrelevant_rows(formant_df)
    X = StandardScaler().fit_transform(df[["f1", "f2", "f3"]])
    X_pca = PCA(n_components=3).fit_transform(X)
    for i in range(3):
        df[f"pca_f{i + 1}"] = X_pca[:, i]
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Compute speaker-specificity index (KLD ratio) per layer.")
    parser.add_argument("--formant_df", required=True,
                        help="Pickle with per-frame f1/f2/f3 + phone/speaker/rep annotation.")
    parser.add_argument("--hid-act-folder", required=True,
                        help="Folder of per-layer hidden activations (*.pt).")
    parser.add_argument("--num-layers", type=int, default=12,
                        help="Transformer layers (wav2vec2-base = 12).")
    parser.add_argument("--out-dir", default="../data/processed/spec_idx",
                        help="Directory for the per-layer spec_idx CSVs.")
    parser.add_argument("--rerun", action="store_true",
                        help="Recompute even if the output CSVs already exist.")
    args = parser.parse_args()

    formant_df = ensure_formant_pca(pd.read_pickle(args.formant_df))

    spec_idx = get_spec_idx_for_every_layer(
        layers=list(range(1, args.num_layers + 1)),
        hid_act_folder=args.hid_act_folder,
        formant_df=formant_df,
        path=args.out_dir,
        dont_run_again=not args.rerun,
    )
    print(f"Done. Wrote {len(spec_idx)} layer table(s) to {args.out_dir}")


if __name__ == "__main__":
    main()
