# get waveform and gammatone
import importlib
import helpers
importlib.reload(helpers)
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from transformers import Wav2Vec2Model, Wav2Vec2Processor
from pathlib import Path
from scipy.stats import multivariate_normal

ROOT="./dev-clean"

formant_df = pd.read_pickle("formant.pkl")
hid_act_folder = "./wav_2_vec_hidden_states"
# read activations, left join with formant table 

model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
num_layers = model.config.num_hidden_layers + 1


gammatone = helpers.get_waveform_and_gammatone(ROOT=ROOT, dont_run_again=True)
gammatone_longform = helpers.gammatone_long(gammatone, formant_df=formant_df)
gammatone_longform["phone"] = gammatone_longform["phone"].str.replace(r"\d+", "", regex=True)


gt_with_lda = helpers.lda_gammatone(gt_df=gammatone_longform)
gt_with_lda = gt_with_lda.rename(columns={
    "lda_1": "gamma_lda_1",
    "lda_2": "gamma_lda_2",
    "lda_3": "gamma_lda_3"
})

gt_with_lda = helpers.filter_out_irrelevant_rows(gt_with_lda)

cols = ["filename", "time", "gamma_lda_1", "gamma_lda_2", "gamma_lda_3"]
gt_with_lda = gt_with_lda[cols]

def get_spec_idx_for_every_layer_lda_from_gamma(
    num_layer, hid_act_folder, formant, dont_run_again=True, gammatone=None
):  # dont_run_again is a toggle: if T we don't re-compute
    '''
    return a dictionary of dfs, one for each layer

    Questions:
    lda is initiated once for each layer; random number generators are initiated once for each layer
    '''
    spec_idx = {}
    p = Path("./spec_idx")

    # If reuse & files exist, load them
    if dont_run_again and check_folder_files_exist(folder=p, num_layer=num_layer):
        for file in p.glob("layer_*_spec_idx.csv"):
            i = file.stem.split("_")[1]
            df = pd.read_csv(file)
            spec_idx[i] = df

    else:
        p.mkdir(parents=True, exist_ok=True)

        for layer in range(1, num_layer + 1):

            # read out the hid act for that layer first
            filtered_act_path = Path("./wav_2_vec_hidden_lda")
            filtered_act_path.mkdir(parents=True, exist_ok=True)
            lda_act_filename = filtered_act_path / f"lda_acts_{layer}.pkl"

            if lda_act_filename.exists():
                act_filtered = pd.read_pickle(lda_act_filename)
                act_filtered = helpers.filter_out_irrelevant_rows(act_filtered)
            else:
                acts = read_activation_and_left_join_with_acoustics(
                    hid_act_folder=hid_act_folder,
                    hid_pkl_folder="wav_2_vec_hidden_pkl",
                    num_layers=num_layer,
                    formant_df=formant,
                    which_layer=layer
                )

                act_filtered = filter_out_irrelevant_rows(acts)

                lda = LinearDiscriminantAnalysis(n_components=3)
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

                X = np.stack(act_filtered['act'].to_numpy())
                y = act_filtered['phone']

                le = LabelEncoder()
                y = le.fit_transform(y)

                lda_X = lda_reduction(X=X, y=y, lda=lda, cv=cv)
                lda_cols = [f"lda_{i+1}" for i in range(lda_X.shape[1])]
                act_filtered[lda_cols] = lda_X
                act_filtered.to_pickle(lda_act_filename)

            # left join with gammatone
            act_filtered = act_filtered.merge(
                gammatone,
                how="left",
                on=["filename", "time"]
            )

            # average across vowels
            act_vowel_mean_pooled = (
                act_filtered
                .groupby(["filename", "rep", "phone", "speaker"])
                .agg({
                    "gamma_lda_1": "mean",
                    "gamma_lda_2": "mean",
                    "gamma_lda_3": "mean",
                    "lda_1": "mean",
                    "lda_2": "mean",
                    "lda_3": "mean",
                })
                .round(2)
            )

            act_spec_idx = calculate_speaker_marginal_kld_lda_from_gamma(df=act_vowel_mean_pooled)
            act_spec_idx_df = pd.DataFrame(act_spec_idx)

            output_path = p / f"layer_{layer}_spec_idx.csv"
            act_spec_idx_df.to_csv(output_path)

            spec_idx[layer] = act_spec_idx_df
            print(f'The kld for Layer {layer} is done.')

    return spec_idx


def calculate_speaker_marginal_kld_lda_from_gamma(df):
    # get speaker specific distr
    speaker_spec_results = []
    for (speaker, phone), group in df.groupby(["speaker", "phone"]):
        acous = group[["gamma_lda_1", "gamma_lda_2", "gamma_lda_3"]].to_numpy()
        model = group[["lda_1", "lda_2", "lda_3"]].to_numpy()

        acous_mean = acous.mean(axis=0)
        acous_var = np.cov(acous, rowvar=False)
        acous_mvg = multivariate_normal(mean=acous_mean, cov=acous_var, allow_singular=True)

        model_mean = model.mean(axis=0)
        model_var = np.cov(model, rowvar=False)
        model_mvg = multivariate_normal(mean=model_mean, cov=model_var, allow_singular=True)

        speaker_spec_results.append(
            {
                "speaker": speaker,
                "phone": phone,
                "acoustic_mvg": acous_mvg,
                "model_mvg": model_mvg
            }
        )

    speaker_spec_results = pd.DataFrame.from_dict(speaker_spec_results)

    # get marginal distr
    marginal_results = []
    for phone, group in df.groupby(["phone"]):
        acous = group[["gamma_lda_1", "gamma_lda_2", "gamma_lda_3"]].to_numpy()
        model = group[["lda_1", "lda_2", "lda_3"]].to_numpy()

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

    phones = marginal_results['phone'].to_list()
    speakers = speaker_spec_results['speaker'].unique().tolist()

    spec_idx = {}
    for phone in phones:
        acous_marg = marginal_results.loc[marginal_results['phone']==phone, "acoustic_mvg"].iloc[0]
        phone_spec_idx = {}
        for speaker in speakers:
            mask = ( # if the speaker never said that phone (but is this really possible?)
                (speaker_spec_results['phone'] == phone) &
                (speaker_spec_results['speaker'] == speaker)
                )
            subset = speaker_spec_results.loc[mask, ["model_mvg", "acoustic_mvg"]]

            if subset.empty:
                # no matching rows for this phone–speaker pair
                phone_spec_idx[speaker] = np.nan
                continue

            model_spec = subset["model_mvg"].iloc[0]
            acous_spec = subset["acoustic_mvg"].iloc[0]

            if model_spec is None or acous_spec is None:
                phone_spec_idx[speaker] = np.nan
                continue

            kld_model_marginal = kld_calculate(model_spec, acous_marg)
            kld_model_spec = kld_calculate(model_spec, acous_spec)

            specificity_index = speaker_specificity_kld(kld_model_marginal, kld_model_spec)
            phone_spec_idx[speaker] = specificity_index

        spec_idx[phone] = phone_spec_idx

    return spec_idx




def kld_calculate(dist1, dist2):
    """
    KL divergence between two multivariate normals.
    dist1: model mvg distribution,
    dist2: acoustic mvg distribution
    """
    d = dist1.mean.shape[0]
    try:
        kld=0.5 * (
        np.trace(np.linalg.inv(dist2.cov) @ dist1.cov)
        + (dist2.mean - dist1.mean).T @ np.linalg.inv(dist2.cov) @ (dist2.mean - dist1.mean)
        - d
        + np.log(np.linalg.det(dist2.cov) / np.linalg.det(dist1.cov))
    )
        return kld
    except np.linalg.LinAlgError:
        return np.nan
        
def speaker_specificity_kld(kld_model_marginal, kld_model_speaker):
    if (
        kld_model_marginal is None or kld_model_speaker is None or
        np.isnan(kld_model_marginal) or np.isnan(kld_model_speaker)
    ):
        return np.nan
    else:
        index = kld_model_speaker / (kld_model_marginal  + kld_model_speaker) 
        return index

gamma_lda_comparison = get_spec_idx_for_every_layer_lda_from_gamma(num_layer = num_layers-1,
                                            hid_act_folder=hid_act_folder, 
                                            formant=formant_df,
                                            dont_run_again=False,
                                            gammatone=gt_with_lda)
#######################################################################
# something in the get_spec_idx_for_every_layer_lda must be wrong: check again and clean it

all_results = []
for layer in range(1, num_layers):
    results = gamma_lda_comparison[layer]
    results_long = (
        results
        .reset_index(names="speaker")        # move the index (speaker IDs) into a column
        .melt(id_vars="speaker",             # columns to keep fixed
            var_name="vowel",              # new column name for former column headers
            value_name="r")                # new column name for the numeric values
    )
    results_long["layer"] = layer
    all_results.append(results_long)
all_results = pd.concat(all_results)
sns.set(style="white")  # clean background
# do some plotting:
g = sns.FacetGrid(all_results, col="vowel", col_wrap=3, sharex=True, sharey=True, height=3, margin_titles=True)

# raw points
g.map_dataframe(sns.scatterplot, x="layer", y="r", alpha=0.2, s=12, color=None)

# mean line + mean points (per-vowel facet)
g.map_dataframe(helpers.mean_line_and_points, color=None)

# hline, ticks, limits, grid
for ax in g.axes.flatten():
    ax.axhline(0.5, ls="--", lw=2, color="red")
    ax.set_xticks(range(1, 13))
    ax.set_ylim(0, 1)
    ax.set_yticks(np.linspace(0, 1, 11))
    ax.grid(True, axis="both", which="major", linewidth=0.5, color="grey")
    ax.set_facecolor("white")

g.set_axis_labels("Layers", "KL ratio")
g.fig.tight_layout()
plt.show()
plt.savefig("./figures/gammatone_lda/redo_lda_gammatone_kl_ratio_plot", dpi=600, bbox_inches="tight")


all_results["speaker"] = all_results["speaker"].astype(str)

plt.figure(figsize=(10, 6))
sns.set(style="white")

# raw points (colored by speaker, like your ggplot)
sns.scatterplot(data=all_results, x="layer", y="r", hue="speaker", alpha=0.2, s=12, linewidth=0, legend=False)

# compute mean across vowels per speaker × layer
m = all_results.groupby(["speaker", "layer"], as_index=False)["r"].mean()

# mean line + mean points per speaker
sns.lineplot(data=m, x="layer", y="r", hue="speaker", lw=1, legend=False)
sns.scatterplot(data=m, x="layer", y="r", hue="speaker", s=60, legend=True)

# styling to match R
plt.axhline(0.5, ls="--", lw=2, color="red")
plt.xticks(range(1, 13))
plt.yticks(np.linspace(0, 1, 11))
plt.ylim(0, 1)
plt.xlabel("Layers", fontsize=14, fontweight="bold")
plt.ylabel("KL ratop", fontsize=14, fontweight="bold")
plt.grid(True, which="major", linewidth=0.5, color="grey")
plt.gca().set_facecolor("white")

# legend on the right
plt.legend(title="speaker", bbox_to_anchor=(1, 0.75), loc="upper left", borderaxespad=0., ncol=2)
plt.tight_layout()
plt.show()



#######################################################################

# below is the visualization for each vowel for each formant: migrate to another file here
ldas = ['gamma_lda_1', 'gamma_lda_2', 'gamma_lda_3']
vowels = sorted(gt_with_lda['phone'].unique())

fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
for ax, lda in zip(axes, ldas):
    for v in vowels:
        subset = gt_with_lda.loc[gt_with_lda['phone'] == v, lda]
        ax.hist(subset, bins=30, alpha=0.5, label=v, density=True)
    ax.set_title(lda.upper())
    ax.set_xlabel("unit")
    ax.legend(fontsize=8, ncol=2)

fig.suptitle("Component Distributions by Vowel")
plt.tight_layout()
plt.show()
fig.savefig("./figures/gammatone_lda_distributions/formant_distributions_by_vowel.png", dpi=300, bbox_inches="tight")


for v in vowels:
    # one row × 3 columns for F1/F2/F3
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

    # plot each formant distribution for this vowel
    for ax, formant in zip(axes, formants):
        subset = gammatone_longform.loc[gammatone_longform['phone'] == v, formant]
        ax.hist(subset, bins=30, color="steelblue", alpha=0.6, density=True)
        ax.set_title(formant.upper())
        ax.set_xlabel("Hz")
        ax.set_ylabel("Count")

    # overall title and layout
    fig.suptitle(f"Formant distributions for vowel {v}", fontsize=14)
    plt.tight_layout()

    # save and close
    out_path = f"./figures/formant_distributions/{v}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# count distribution of speakers and create histogram
gt_with_lda["speaker"] = gt_with_lda["speaker"].astype(str)

counts = gt_with_lda["speaker"].value_counts().sort_index()

# bar plot
plt.figure(figsize=(10,5))
counts.plot(kind="bar")
plt.xlabel("Speaker")
plt.ylabel("Number of rows")
plt.title("Rows per Speaker")
plt.tight_layout()
plt.show()
#######################################################################
# formant and lda from spectrogram: KL must be very small; check this


X = formant_df[["f1", "f2", "f3"]]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

pca_cols = [f"pca_f{i+1}" for i in range(X_pca.shape[1])] # create a bunch of colnames for the 3d vector output
formant_df[pca_cols] = X_pca

formant_df
gt_with_lda

formant_and_gt_lda = formant_df.merge(
                gt_with_lda,
                how="left",
                on=["filename", "time"]
            )


X = formant_and_gt_lda[["gamma_lda_1", "gamma_lda_2", "gamma_lda_3"]]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

pca_cols = [f"pca_lda_{i+1}" for i in range(X_pca.shape[1])] # create a bunch of colnames for the 3d vector output
formant_and_gt_lda[pca_cols] = X_pca

speaker_kld_formant_gammatone = {}
for speaker, speaker_df in formant_and_gt_lda.groupby("speaker"):
    vowel_kld_formant_gammatone = {}
    for phone, group in speaker_df.groupby("phone"):
        X_formant = group[["pca_f1", "pca_f2", "pca_f3"]].to_numpy()
        mean_formant = np.mean(X_formant, axis=0)
        cov_formant  = np.cov(X_formant, rowvar=False)

        formant_mvg = multivariate_normal(mean=mean_formant, cov=cov_formant)

        X_lda = group[["pca_lda_1", "pca_lda_2", "pca_lda_3"]].to_numpy()
        mean_lda = np.mean(X_lda, axis=0)
        cov_lda  = np.cov(X_lda, rowvar=False)

        lda_gammatone_mvg = multivariate_normal(mean=mean_lda, cov=cov_lda)

        kld = helpers.kld_calculate(formant_mvg, lda_gammatone_mvg)
        vowel_kld_formant_gammatone[phone] = kld

    speaker_kld_formant_gammatone[speaker] = vowel_kld_formant_gammatone

kld_df = pd.DataFrame.from_dict(speaker_kld_formant_gammatone, orient="index")

# plot a histogram for the kld distribution
plt.figure(figsize=(8, 5))

# loop over each vowel column
for vowel in kld_df.columns:
    plt.hist(
        kld_df[vowel].dropna(),    # drop NaN if any
        bins=5,
        alpha=0.5,                 # transparency
        label=vowel,
        density=True
    )

# vertical line at 0
plt.axvline(0, color="black", linestyle="--", linewidth=1)

# labels and legend
plt.xlabel("KLD")
plt.ylabel("Count")
plt.title("Distribution of KLD for each vowel")
plt.legend(title="Vowel", fontsize=9)
plt.tight_layout()
plt.savefig("./figures/gammatone_lda/kl_hist_formant_gamma_lda_each_vowel_density", dpi=600, bbox_inches="tight")



# over all vowels
all_klds = kld_df.values.flatten()   # or kld_df.to_numpy().ravel()

plt.figure(figsize=(7, 4))
plt.hist(all_klds, bins=5, color="steelblue", alpha=0.7, edgecolor="black", density=True)

# vertical line at 0
plt.axvline(0, color="black", linestyle="--", linewidth=1)

plt.xlabel("KLD")
plt.ylabel("Count")
plt.title("Distribution of KLD for all vowels")
plt.tight_layout()
plt.savefig("./figures/gammatone_lda/kl_hist_formant_gamma_lda_all_vowels_density", dpi=600, bbox_inches="tight")
