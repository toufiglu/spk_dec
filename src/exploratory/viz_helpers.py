"""Plotting helpers for distributions and KL-ratio summaries. Exploratory."""


from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def plot_gaussian_ellipsoid(mean, cov, ax, n_std=2.0, **kwargs):
    # n_std: 1≈68%, 2≈95%
    vals, vecs = np.linalg.eigh(cov)
    radii = n_std * np.sqrt(vals)

    # unit sphere: Yiming-learn what this is doing
    u = np.linspace(0, 2*np.pi, 40)
    v = np.linspace(0, np.pi, 20)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    sphere = np.stack([x, y, z], axis=-1)                        # (40,20,3)

    A = vecs @ np.diag(radii)                                    # (3,3)
    E = np.einsum('ij,uvj->uvi', A, sphere) + mean               # (40,20,3)

    ax.plot_surface(E[:,:,0], E[:,:,1], E[:,:,2],
                    rstride=1, cstride=1, linewidth=0, shade=False, **kwargs)


def visualization_3d_spec_distributions(num_layer, formant_df):
    speaker_specs = {}
    marginals = {}
    for layer in range(1, num_layer+1): 
        # read out the hid act for that layer first
        filtered_act_path = Path("./wav_2_vec_hidden_lda")
        filtered_act_path.mkdir(parents=True, exist_ok=True)
        lda_act_filename = filtered_act_path / f"lda_acts_{layer}.pkl"

        act_filtered = pd.read_pickle(lda_act_filename)

        X = act_filtered[["lda_1", "lda_2", "lda_3"]]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pca = PCA(n_components=3)
        X_pca = pca.fit_transform(X_scaled)

        pca_cols = [f"pca_{i+1}" for i in range(X_pca.shape[1])]
        act_filtered[pca_cols] = X_pca

        missing = [c for c in pca_cols if c not in act_filtered.columns]
        if missing:
            raise RuntimeError(f"Missing PCA cols {missing} for layer {layer}. Recompute or check LDA step.")
        
        formant_df_cols = formant_df[["time", "filename", "pca_f1", "pca_f2", "pca_f3"]]
        act_filtered = pd.merge(act_filtered, formant_df_cols, on=["time", "filename"])
        # average across vowels
        act_vowel_mean_pooled = (act_filtered
                                        .groupby(["filename", "rep", "phone", "speaker"])
                                        .agg(
                                            {
                                                "pca_f1": "mean",
                                                "pca_f2": "mean",
                                                "pca_f3": "mean",
                                                "pca_1": "mean",
                                                "pca_2": "mean",
                                                "pca_3": "mean",
                                            }
                                        )
                                        .round(2))
            # get speaker specific distr
        speaker_spec_results = []
        for (speaker, phone), group in act_filtered.groupby(["speaker", "phone"]):
            acous = group[["pca_f1", "pca_f2", "pca_f3"]].to_numpy()
            model = group[["pca_1", "pca_2", "pca_3"]].to_numpy()

            acous_mean = acous.mean(axis=0)
            acous_var = np.cov(acous, rowvar=False)
            acous_mvg = multivariate_normal(mean=acous_mean, cov=acous_var)

            model_mean = model.mean(axis=0)
            model_var = np.cov(model, rowvar=False)
            model_mvg = multivariate_normal(mean=model_mean, cov=model_var)

            speaker_spec_results.append(
                {
                    "speaker": speaker,
                    "phone": phone,
                    "acoustic_mvg": acous_mvg,
                    "model_mvg": model_mvg
                }
            )

        speaker_spec_results = pd.DataFrame.from_dict(speaker_spec_results)
        speaker_specs[layer] = speaker_spec_results

        # get marginal distr
        marginal_results = []
        for phone, group in act_filtered.groupby(["phone"]):
            acous = group[["pca_f1", "pca_f2", "pca_f3"]].to_numpy()
            model = group[["pca_1", "pca_2", "pca_3"]].to_numpy()

            acous_mean = acous.mean(axis=0)
            acous_var = np.cov(acous, rowvar=False)
            acous_mvg = multivariate_normal(mean=acous_mean, cov=acous_var)

            model_mean = model.mean(axis=0)
            model_var = np.cov(model, rowvar=False)
            model_mvg = multivariate_normal(mean=model_mean, cov=model_var)

            marginal_results.append(
                {
                    "speaker": speaker,
                    "phone": phone,
                    "acoustic_mvg": acous_mvg,
                    "model_mvg": model_mvg
                }
            )
        marginal_results = pd.DataFrame.from_dict(marginal_results).drop(columns="speaker")
        marginal_results["phone"] = marginal_results["phone"].apply(lambda x: x[0])

        marginals[layer] = marginal_results
    return speaker_specs, marginals


def mean_line_and_points(data, **kwargs):
    m = data.groupby("layer", as_index=False)["r"].mean()
    sns.lineplot(data=m, x="layer", y="r", lw=1, **kwargs)
    sns.scatterplot(data=m, x="layer", y="r", s=30, **kwargs)
