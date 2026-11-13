"""RSA: do model-layer RDMs track speaker-specific or marginal acoustics?

For each wav2vec 2.0 layer this builds a Mahalanobis phone x phone RDM per
speaker from the hidden activations, and correlates it with

  * the pooled, all-speaker formant RDM   -> "marginal"   (talker-independent)
  * that speaker's own formant RDM         -> "speaker-specific"

Writes ``marginal_cors_across_layers.csv`` and ``spec_cors_across_layers.csv``
(speaker, corr, layer) plus a comparison figure — the two tables shipped under
``results/``.

Run from inside ``src/`` (after the readout stage):

    python compute_rsa.py \\
        --formant_df ../data/processed/formant.pkl \\
        --hid-pkl-folder ../data/processed/hidden_acts_pkl \\
        --out-dir ../data/processed/rsa

Requires the ``rsatoolbox`` package.
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import rsatoolbox

from acoustics import filter_out_irrelevant_rows


def _precision_by_speaker(dataset):
    cov = rsatoolbox.data.noise.cov_from_unbalanced(
        dataset, obs_desc="phone", dof=None, method="shrinkage_diag")
    return np.linalg.pinv(np.asarray(cov))


def _speaker_rdms(dataset, phones):
    """One Mahalanobis phone RDM per speaker from a channel dataset."""
    rdms = {}
    for speaker_df in dataset.split_obs("speaker"):
        spk = str(np.unique(speaker_df.obs_descriptors["speaker"])[0])
        rdm = rsatoolbox.rdm.calc.calc_rdm_mahalanobis(
            dataset=speaker_df, descriptor="phone",
            noise=_precision_by_speaker(speaker_df), remove_mean=True)
        rdm.rdm_descriptors["speaker"] = [spk]
        rdm.pattern_descriptors["phone"] = phones
        rdms[spk] = rdm
    return rdms


def build_formant_rdms(formant_df, phones):
    """(all-speaker marginal RDM, {speaker: speaker-specific RDM}) from formants."""
    data = rsatoolbox.data.Dataset(
        formant_df[["f1", "f2", "f3"]].to_numpy(dtype=float),
        obs_descriptors={
            "phone": formant_df["phone"].tolist(),
            "speaker": formant_df["speaker"].tolist(),
            "filename": formant_df["filename"].tolist(),
        },
        channel_descriptors={"feature": ["f1", "f2", "f3"]},
    )
    cov = rsatoolbox.data.noise.cov_from_unbalanced(data, obs_desc="phone", dof=None, method="full")
    marginal_rdm = rsatoolbox.rdm.calc.calc_rdm_mahalanobis(
        data, descriptor="phone", noise=np.linalg.inv(cov), remove_mean=True)
    return marginal_rdm, _speaker_rdms(data, phones)


def layer_rsa(act_df, phones, marginal_rdm, formant_rdm_by_speaker):
    """Per-speaker (marginal corr, speaker-specific corr) for one layer's acts."""
    data = rsatoolbox.data.Dataset(
        np.stack(act_df["act"].to_numpy()),
        obs_descriptors={
            "phone": act_df["phone"].tolist(),
            "speaker": act_df["speaker"].tolist(),
            "filename": act_df["filename"].tolist(),
        },
        channel_descriptors={
            "feature": [f"dim_{d}" for d in range(1, act_df["act"].iloc[0].shape[0] + 1)]},
    )
    marginal, specific = {}, {}
    for spk, rdm in _speaker_rdms(data, phones).items():
        marginal[spk] = float(rsatoolbox.rdm.compare_correlation(rdm1=marginal_rdm, rdm2=rdm).item())
        formant_rdm = formant_rdm_by_speaker.get(spk)
        if formant_rdm is not None:
            specific[spk] = float(rsatoolbox.rdm.compare_correlation(rdm1=formant_rdm, rdm2=rdm).item())
    return marginal, specific


def rsa_across_layers(formant_df, hid_pkl_folder, num_layers):
    formant_df = filter_out_irrelevant_rows(formant_df)
    phones = sorted(formant_df["phone"].unique())
    marginal_rdm, formant_rdm_by_speaker = build_formant_rdms(formant_df, phones)

    marginal_rows, spec_rows = [], []
    for layer in range(1, num_layers + 1):
        act_df = pd.read_pickle(os.path.join(hid_pkl_folder, f"hidden_acts_{layer}.pkl"))
        act_df = filter_out_irrelevant_rows(act_df)
        marginal, specific = layer_rsa(act_df, phones, marginal_rdm, formant_rdm_by_speaker)
        for spk, corr in marginal.items():
            marginal_rows.append({"speaker": spk, "corr": corr, "layer": layer})
        for spk, corr in specific.items():
            spec_rows.append({"speaker": spk, "corr": corr, "layer": layer})
        print(f"RSA layer {layer} done.")
    return pd.DataFrame(marginal_rows), pd.DataFrame(spec_rows)


def plot_comparison(marginal_df, spec_df, out_path):
    results = pd.concat([marginal_df.assign(type="all-speakers"),
                         spec_df.assign(type="speaker-specific")], ignore_index=True)
    stats = (results.groupby(["layer", "type"])["corr"]
             .agg(mean="mean", sem=lambda s: s.std() / np.sqrt(len(s))).reset_index())
    palette = {"all-speakers": "#f1223e", "speaker-specific": "#00b4d8"}

    plt.figure(figsize=(7, 4.5))
    for t, sub in stats.groupby("type"):
        plt.errorbar(sub["layer"], sub["mean"], yerr=sub["sem"], marker="o",
                     color=palette[t], label=t, lw=2, capsize=3)
    plt.axhline(0, color="grey", lw=0.8, ls="--")
    plt.xlabel("wav2vec 2.0 layer")
    plt.ylabel("RDM correlation (r)")
    plt.title("Model RDMs vs. speaker-specific and marginal acoustics")
    plt.legend(frameon=False)
    plt.grid(True, lw=0.4, alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")


def main():
    parser = argparse.ArgumentParser(description="RSA of model layers vs. acoustic RDMs.")
    parser.add_argument("--formant_df", required=True,
                        help="Pickle with per-frame f1/f2/f3 + phone/speaker/filename.")
    parser.add_argument("--hid-pkl-folder", required=True,
                        help="Folder of per-layer joined activation pickles (readout output).")
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--out-dir", default="../data/processed/rsa")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    formant_df = pd.read_pickle(args.formant_df)

    marginal_df, spec_df = rsa_across_layers(formant_df, args.hid_pkl_folder, args.num_layers)
    marginal_df.to_csv(os.path.join(args.out_dir, "marginal_cors_across_layers.csv"))
    spec_df.to_csv(os.path.join(args.out_dir, "spec_cors_across_layers.csv"))
    plot_comparison(marginal_df, spec_df, os.path.join(args.out_dir, "comparison_across_layers.png"))
    print(f"Done. Wrote RSA tables + figure to {args.out_dir}")


if __name__ == "__main__":
    main()
