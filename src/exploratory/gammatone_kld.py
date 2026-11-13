"""Gammatone-baseline version of the speaker-specificity KLD metric.

Saved from the former ``get_gammatone_and_decode.py`` (the plotting / top-level
scratch there was dropped). This keeps only the part that *computes the KLD
metric*: instead of comparing model representations against formant
distributions, it compares them against an LDA projection of a gammatone
filterbank — an alternative acoustic reference.

For each (speaker, vowel) it fits a 3-D Gaussian to the model's LDA components
and to the gammatone LDA components, then forms the speaker-specificity index

    idx = KL(model || acoustic_speaker)
          / ( KL(model || acoustic_marginal) + KL(model || acoustic_speaker) )

reusing ``kld_calculate`` and ``speaker_specificity_kld``.

Exploratory: expects precomputed ``wav_2_vec_hidden_lda/lda_acts_<layer>.pkl``
and a gammatone-LDA dataframe; paths are not wired to the current data/ layout.
See ``get_spec_idx_for_every_layer`` for the formant-based version that
produces the tracked ``results/results_spec_idx/`` tables.
"""
from pathlib import Path

from acoustics import filter_out_irrelevant_rows
from spec_idx import get_spec_idx_for_every_layer, kld_calculate, speaker_specificity_kld
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal


def calculate_speaker_marginal_kld_lda_from_gamma(df):
    """Speaker-specificity index per (phone, speaker) using gammatone-LDA as the
    acoustic reference. ``df`` must carry gamma_lda_1..3 and lda_1..3 columns."""
    # speaker-specific distributions
    speaker_spec_results = []
    for (speaker, phone), group in df.groupby(["speaker", "phone"]):
        acous = group[["gamma_lda_1", "gamma_lda_2", "gamma_lda_3"]].to_numpy()
        model = group[["lda_1", "lda_2", "lda_3"]].to_numpy()
        speaker_spec_results.append({
            "speaker": speaker,
            "phone": phone,
            "acoustic_mvg": multivariate_normal(acous.mean(0), np.cov(acous, rowvar=False), allow_singular=True),
            "model_mvg": multivariate_normal(model.mean(0), np.cov(model, rowvar=False), allow_singular=True),
        })
    speaker_spec_results = pd.DataFrame.from_dict(speaker_spec_results)

    # marginal (talker-independent) distributions
    marginal_results = []
    for phone, group in df.groupby(["phone"]):
        acous = group[["gamma_lda_1", "gamma_lda_2", "gamma_lda_3"]].to_numpy()
        marginal_results.append({
            "phone": phone,
            "acoustic_mvg": multivariate_normal(acous.mean(0), np.cov(acous, rowvar=False), allow_singular=True),
        })
    marginal_results = pd.DataFrame.from_dict(marginal_results)
    marginal_results["phone"] = marginal_results["phone"].apply(
        lambda x: x[0] if isinstance(x, tuple) else x)

    phones = marginal_results["phone"].to_list()
    speakers = speaker_spec_results["speaker"].unique().tolist()

    spec_idx = {}
    for phone in phones:
        acous_marg = marginal_results.loc[marginal_results["phone"] == phone, "acoustic_mvg"].iloc[0]
        phone_spec_idx = {}
        for speaker in speakers:
            mask = (speaker_spec_results["phone"] == phone) & (speaker_spec_results["speaker"] == speaker)
            subset = speaker_spec_results.loc[mask, ["model_mvg", "acoustic_mvg"]]
            if subset.empty:
                phone_spec_idx[speaker] = np.nan
                continue
            model_spec = subset["model_mvg"].iloc[0]
            acous_spec = subset["acoustic_mvg"].iloc[0]

            kld_model_marginal = kld_calculate(model_spec, acous_marg)
            kld_model_spec = kld_calculate(model_spec, acous_spec)
            phone_spec_idx[speaker] = speaker_specificity_kld(
                kld_model_marginal=kld_model_marginal, kld_model_speaker=kld_model_spec)
        spec_idx[phone] = phone_spec_idx

    return spec_idx


def spec_idx_across_layers_from_gammatone(num_layers, gammatone_lda_df,
                                          lda_dir="./wav_2_vec_hidden_lda", out_dir="./spec_idx_gamma"):
    """Compute the gammatone-baseline specificity index for each layer.

    ``gammatone_lda_df`` has columns filename, time, gamma_lda_1..3.
    ``lda_dir`` holds per-layer model-LDA pickles (lda_acts_<layer>.pkl with
    lda_1..3 + phone/speaker/filename/time/rep). Returns {layer: DataFrame}.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    results = {}
    for layer in range(1, num_layers + 1):
        act = pd.read_pickle(Path(lda_dir) / f"lda_acts_{layer}.pkl")
        act = filter_out_irrelevant_rows(act)
        act = act.merge(gammatone_lda_df, how="left", on=["filename", "time"])

        pooled = (act.groupby(["filename", "rep", "phone", "speaker"])
                     .agg({c: "mean" for c in
                           ["gamma_lda_1", "gamma_lda_2", "gamma_lda_3", "lda_1", "lda_2", "lda_3"]})
                     .round(2))

        df = pd.DataFrame(calculate_speaker_marginal_kld_lda_from_gamma(pooled))
        df.to_csv(out / f"layer_{layer}_spec_idx.csv")
        results[layer] = df
        print(f"gammatone KLD for layer {layer} done.")
    return results
