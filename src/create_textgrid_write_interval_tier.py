import os
import glob
import torch
import parselmouth
import numpy as np
import pandas as pd
import pickle
import soundfile as sf
import re 

import importlib
import helpers
importlib.reload(helpers)

from praatio import textgrid
from praatio.audio import getDuration
from pathlib import Path
from scipy.stats import multivariate_normal
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.base import clone
from transformers import Wav2Vec2Processor, Wav2Vec2Model

arpabet_vowels = [
    "AA", "AA0", "AA1", "AA2",
    "AE", "AE0", "AE1", "AE2",
    "AH", "AH0", "AH1", "AH2",
    "AO", "AO0", "AO1", "AO2",
    "AW", "AW0", "AW1", "AW2",
    "AY", "AY0", "AY1", "AY2",
    "EH", "EH0", "EH1", "EH2",
    "ER", "ER0", "ER1", "ER2",
    "EY", "EY0", "EY1", "EY2",
    "IH", "IH0", "IH1", "IH2",
    "IY", "IY0", "IY1", "IY2",
    "OW", "OW0", "OW1", "OW2",
    "OY", "OY0", "OY1", "OY2",
    "UH", "UH0", "UH1", "UH2",
    "UW", "UW0", "UW1", "UW2",
]

ROOT = "./dev-clean"

hid_act_folder = "./wav_2_vec_hidden_states"


# create a tier for sentence transcription; prepare for mfa later
for speaker in os.listdir(ROOT):
    speaker_dir = os.path.join(ROOT, speaker)
    if not os.path.isdir(speaker_dir):
        continue

    for rec_folder in os.listdir(speaker_dir):
        rec_dir = os.path.join(speaker_dir, rec_folder)
        if not os.path.isdir(rec_dir):
            continue

        trans_map = helpers.read_transcript_map(rec_dir)

        wav_paths = glob.glob(os.path.join(rec_dir, "*.wav"))
        # Report which WAVs currently lack a TextGrid (before creating)
        missing = []
        for wp in wav_paths:
            stem = os.path.splitext(os.path.basename(wp))[0]
            tg_path = os.path.join(rec_dir, f"{stem}.TextGrid")
            if not os.path.exists(tg_path):
                missing.append(stem)
        if missing:
            print(f"[INFO] Missing TextGrids in {rec_dir}: {len(missing)} → {missing[:5]}{' ...' if len(missing)>5 else ''}")

        for wp in wav_paths:
            stem = os.path.splitext(os.path.basename(wp))[0]
            transcription = trans_map.get(stem, "")  
            dur = getDuration(wp)                    # WAV duration

            tg = textgrid.Textgrid()
            tier = textgrid.IntervalTier("Sentence", [(0.0, dur, transcription)], 0.0, dur) 
            tg.addTier(tier)

            tg_path = os.path.join(rec_dir, f"{stem}.TextGrid")
            tg.save(tg_path, format="long_textgrid", includeBlankSpaces=False)

'''
 Ming to himself: split the script here, create a new one for below; after this line, you've done mfa, and have all files aligned.    
'''
# after mfa, extract formants time aligned to the vowels
mfa_folder = "./mfa_align/"
sound_folder = mfa_folder + "corpus_before_align/"
formant_df = helpers.get_formant(sound_folder=sound_folder)
formant_df["phone"] = formant_df["phone"].str.replace(r"\d+", "", regex=True) # remove stress information of the vowels,

formant_df = helpers.filter_out_irrelevant_rows(formant_df)
# read activations for each layer separately, do lda 

hid_act_folder = "./wav_2_vec_hidden_states"
# read activations, left join with formant table 

model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
num_layers = model.config.num_hidden_layers + 1

formant_df = helpers.filter_out_irrelevant_rows(formant_df)
spec_idx = helpers.get_spec_idx_for_every_layer(num_layer=12, re_use_existing=False, hid_act_folder=hid_act_folder, formant=formant_df)



# below were the codes written to create 3d visualizations for the pcas of the formants and the lda-ed hidden activations. they worked but to see clearly, we will need to rotate the figures individually.

'''
outdir = "./figures/distribution_plots/"
os.makedirs(outdir, exist_ok=True)

speaker_specs, marginals = helpers.visualization_3d_spec_distributions(num_layer=12, formant_df = formant_df)

for layer in range(1, 13):
    vowels = speaker_specs[layer]["phone"].unique()
    for vowel in vowels:
        speakers = speaker_specs[layer].loc[speaker_specs[layer]["phone"] == vowel, "speaker"].unique()
        acoustic_marginal = marginals[layer][(marginals[layer]["phone"] == vowel)].iloc[0, 1]

        for sp in speakers:
            acoustic_speaker = speaker_specs[layer][
                (speaker_specs[layer]["speaker"] == sp) &
                (speaker_specs[layer]["phone"] == vowel)
            ].iloc[0, 2]

            model_speaker = speaker_specs[layer][
                (speaker_specs[layer]["speaker"] == sp) &
                (speaker_specs[layer]["phone"] == vowel)
            ].iloc[0, 3]

            d_model_speaker_acous_speaker = kld_calculate(model_speaker, acoustic_speaker)
            d_model_speaker_acous_marginal = kld_calculate(model_speaker, acoustic_marginal)
            r = d_model_speaker_acous_speaker / (d_model_speaker_acous_speaker + d_model_speaker_acous_marginal)

            # --- collect result ---
            results.append({
                "speaker": sp,
                "vowel": vowel,
                "layer": layer,
                "r": r
            })

            # --- sample & plot ---
            dist_acoustic_speaker = acoustic_speaker.rvs(size=100)
            dist_acoustic_marginal = acoustic_marginal.rvs(size=100)
            dist_model_speaker = model_speaker.rvs(size=100)

            fig = plt.figure(figsize=(7, 4))
            ax = fig.add_subplot(111, projection='3d')

            ax.scatter(dist_acoustic_speaker[:, 0], dist_acoustic_speaker[:, 1], dist_acoustic_speaker[:, 2],
                       s=12, alpha=0.8, label='Acoustic-speaker specific')
            ax.scatter(dist_acoustic_marginal[:, 0], dist_acoustic_marginal[:, 1], dist_acoustic_marginal[:, 2],
                       s=12, alpha=0.8, label='Acoustic-marginal')
            ax.scatter(dist_model_speaker[:, 0], dist_model_speaker[:, 1], dist_model_speaker[:, 2],
                       s=12, alpha=0.8, label='Model-speaker specific')

            plot_gaussian_ellipsoid(dist_acoustic_speaker.mean(0),
                                    np.cov(dist_acoustic_speaker, rowvar=False),
                                    ax, n_std=2.0, alpha=0.25)
            plot_gaussian_ellipsoid(dist_acoustic_marginal.mean(0),
                                    np.cov(dist_acoustic_marginal, rowvar=False),
                                    ax, n_std=2.0, alpha=0.25)
            plot_gaussian_ellipsoid(dist_model_speaker.mean(0),
                                    np.cov(dist_model_speaker, rowvar=False),
                                    ax, n_std=2.0, alpha=0.25)

            ax.set_xlabel('pca_1')
            ax.set_ylabel('pca_2')
            ax.set_zlabel('pca_3')
            ax.set_box_aspect([1, 1, 1])
            ax.legend(loc='upper left')
            ax.view_init(elev=15, azim=5)
            plt.tight_layout()

            text_str = (
                f"Layer {layer} | Speaker {sp} | Vowel {vowel}\n"
                f"KL(model‖acous_speaker): {d_model_speaker_acous_speaker:.3f}\n"
                f"KL(model‖acous_marginal): {d_model_speaker_acous_marginal:.3f}\n"
                f"KL ratio: {r:.3f}"
            )
            ax.text2D(0.60, 0.80, text_str, transform=ax.transAxes,
                      fontsize=10, va='top',
                      bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.6))

            combo_dir = os.path.join(outroot, f"{sp}_{vowel}")
            os.makedirs(combo_dir, exist_ok=True)
            save_path = os.path.join(combo_dir, f"layer_{layer:02d}_{sp}_{vowel}.png")
            plt.savefig(save_path, dpi=300)
            plt.close(fig)

            print(f"Saved: {save_path}")

# --- save summary DataFrame ---
results_df = pd.DataFrame(results)
results_path = os.path.join(outroot, "kl_ratio_summary.csv")
results_df.to_csv(results_path, index=False)


sns.set(style="white")  # clean background



g = sns.FacetGrid(results_df, col="vowel", col_wrap=3, sharex=True, sharey=True, height=3, margin_titles=True)

# raw points
g.map_dataframe(sns.scatterplot, x="layer", y="r", alpha=0.2, s=12, color=None)

# mean line + mean points (per-vowel facet)
g.map_dataframe(mean_line_and_points, color=None)

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



results_df["speaker"] = results_df["speaker"].astype(str)

plt.figure(figsize=(10, 6))
sns.set(style="white")

# raw points (colored by speaker, like your ggplot)
sns.scatterplot(data=results_df, x="layer", y="r", hue="speaker", alpha=0.2, s=12, linewidth=0, legend=False)

# compute mean across vowels per speaker × layer
m = results_df.groupby(["speaker", "layer"], as_index=False)["r"].mean()

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

plt.figure(figsize=(10, 6))
sns.set(style="white")

# raw points (colored by speaker, like your ggplot)
sns.scatterplot(data=results_df, x="layer", y="r", hue="vowel", alpha=0.2, s=40, linewidth=0, legend=False)

# compute mean across vowels per speaker × layer
m = results_df.groupby(["vowel", "layer"], as_index=False)["r"].mean()

# mean line + mean points per speaker
sns.lineplot(data=m, x="layer", y="r", hue="vowel", lw=1, legend=False)
sns.scatterplot(data=m, x="layer", y="r", hue="vowel", s=120, legend=True)

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
plt.legend(title="vowels", bbox_to_anchor=(1, 0.75), loc="upper left", borderaxespad=0.)
plt.tight_layout()
plt.show()



'''