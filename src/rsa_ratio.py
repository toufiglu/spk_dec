import rsatoolbox
import importlib
import helpers
import numpy as np
import pandas as pd
import os
import seaborn as sns
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import matplotlib.pyplot as plt
importlib.reload(helpers)
import matplotlib
matplotlib.use("Agg")

def calculate_precision_by_speaker(df):
    cov = rsatoolbox.data.noise.cov_from_unbalanced(df, obs_desc='phone', dof=None, method='shrinkage_diag')
    cov = np.asarray(cov)
    prec = np.linalg.pinv(cov)
    return prec

def get_speaker_index(rdm):
    return str(rdm.rdm_descriptors['speaker'][0])

formant_df = helpers.get_formant_with_pca_cols()
phones = formant_df["phone"].unique().tolist()

model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
num_layers = model.config.num_hidden_layers + 1

#######################################################################
formants = formant_df[["f1","f2","f3"]].to_numpy(dtype=float)  # (n_obs, 3)


formant_obs_desc = {
    "phone":   formant_df["phone"].tolist(),
    "speaker": formant_df["speaker"].tolist(),
    "filename":formant_df["filename"].tolist(),  # optional
}

formant_chan_desc = {
    "feature": ["f1","f2","f3"]  # length = 3 = n_channels
}

formant_data = rsatoolbox.data.Dataset(
    formants,
    obs_descriptors=formant_obs_desc,
    channel_descriptors=formant_chan_desc
)
# get formant marginal rdms
all_speaker_formant_cov_matrix = rsatoolbox.data.noise.cov_from_unbalanced(formant_data, obs_desc="phone", dof=None, method='full') 
all_speaker_formant_prec_matrix = np.linalg.inv(all_speaker_formant_cov_matrix)

all_formant_rdm = rsatoolbox.rdm.calc.calc_rdm_mahalanobis(formant_data, descriptor="phone", noise=all_speaker_formant_prec_matrix, remove_mean=True)
fig, axes, _ = rsatoolbox.vis.show_rdm(all_formant_rdm, cmap='RdYlBu_r') # need to know how to control this
fig.savefig("./all_formant_rdm.png", dpi=300, bbox_inches="tight")


# speaker-specific rdms

formant_data_by_speaker = formant_data.split_obs("speaker")

formant_rdms_by_speaker = []

for speaker_df in formant_data_by_speaker:
    spk = np.unique(speaker_df.obs_descriptors['speaker'])[0].item()
    prec = calculate_precision_by_speaker(speaker_df)
    rdm = rsatoolbox.rdm.calc.calc_rdm_mahalanobis(
        dataset   = speaker_df,
        descriptor= "phone",          # averages replicates per phone
        noise     = prec,             # (n_chan x n_chan) precision
        remove_mean = True
    )
    rdm.rdm_descriptors['speaker'] = [spk] # the function might anticipate a list; need to check if this works.
    rdm.pattern_descriptors['phone'] = phones
    formant_rdms_by_speaker.append(rdm)

for rdm in formant_rdms_by_speaker:
    os.makedirs("./figures/rsa/formant_spec/", exist_ok=True)
    M = rdm.get_matrices()[0]
    speaker = str(rdm.rdm_descriptors['speaker'][0])

    fig, axes, _ = rsatoolbox.vis.show_rdm(
        rdm,
        pattern_descriptor='phone',
        cmap='RdYlBu_r',
        rdm_descriptor='speaker',
        show_colorbar='panel',
        figsize=(6,6),
        vmin=np.nanmin(M),
        vmax=np.nanmax(M),
    )
    fig.savefig(f"./figures/rsa/formant_spec/formant_speaker_{speaker}_rdm.png",
                dpi=300, bbox_inches="tight")

#######################################################################

formant_rdm_with_index = { get_speaker_index(rdm): rdm for rdm in formant_rdms_by_speaker }

layers_model_spec_speaker_marginal_cors = {}
layers_model_spec_speaker_spec_cors = {}


for layer in range(1, num_layers):

    act_layer = pd.read_pickle(f"./wav_2_vec_hidden_lda/lda_acts_{layer}.pkl")
    act = np.stack(act_layer["act"])

    act_obs_desc = {
        "phone":   act_layer["phone"].tolist(),
        "speaker": act_layer["speaker"].tolist(),
        "filename": act_layer["filename"].tolist(),  # optional
    }

    dim = np.arange(1, act_layer["act"].iloc[0].shape[0]+1) # define # of channels
    act_col_names = [f"dim_{d}" for d in dim]
    act_chan_desc = {
        "feature": act_col_names
    }

    act_data = rsatoolbox.data.Dataset(
        act,
        obs_descriptors=act_obs_desc,
        channel_descriptors=act_chan_desc
    )

    act_data_by_speaker = act_data.split_obs("speaker")

    act_rdms_by_speaker = []

    for speaker_df in act_data_by_speaker:
        spk = np.unique(speaker_df.obs_descriptors['speaker'])[0].item()
        prec = calculate_precision_by_speaker(speaker_df)
        rdm = rsatoolbox.rdm.calc.calc_rdm_mahalanobis(
            dataset   = speaker_df,
            descriptor= "phone",          
            noise     = prec,             
            remove_mean = True
        )
        rdm.rdm_descriptors['speaker'] = [spk] # the function might anticipate a list; need to check if this works.
        rdm.pattern_descriptors['phone'] = phones
        act_rdms_by_speaker.append(rdm)

    # should store this somewhere, it takes long to compute

    model_spec_speaker_marginal_cors = {} 
    model_spec_speaker_spec_cors = {} 

    os.makedirs(f"./figures/rsa/model_spec/layer_{layer}", exist_ok=True)

    for rdm in act_rdms_by_speaker:
        M = rdm.get_matrices()[0]
        speaker = str(rdm.rdm_descriptors['speaker'][0])

        fig, axes, _ = rsatoolbox.vis.show_rdm(
            rdm,
            pattern_descriptor='phone',
            cmap='RdYlBu_r',
            rdm_descriptor='speaker',
            show_colorbar='panel',
            figsize=(6,6),
            vmin=np.nanmin(M),
            vmax=np.nanmax(M),
        )

        fig.savefig(f"./figures/rsa/model_spec/layer_{layer}/hidden_layer_1_speaker_{speaker}_rdm.png",
                    dpi=300, bbox_inches="tight")

        marginal_corr = rsatoolbox.rdm.compare_correlation(rdm1=all_formant_rdm, rdm2=rdm)

        model_spec_speaker_marginal_cors[speaker] = float(marginal_corr.item())

        # calculate speaker specific cor
        formant_rdm = formant_rdm_with_index.get(speaker)

        spec_corr = rsatoolbox.rdm.compare_correlation(rdm1=formant_rdm, rdm2=act_rdm)

        model_spec_speaker_spec_cors[speaker] = float(spec_corr.item())

    marginal_cors = pd.DataFrame(list(model_spec_speaker_marginal_cors.items()), columns=['speaker', 'corr'])
    layers_model_spec_speaker_marginal_cors[layer] = marginal_cors

    spec_cors = pd.DataFrame(list(model_spec_speaker_spec_cors.items()), columns=['speaker', 'corr'])
    layers_model_spec_speaker_spec_cors[layer] = spec_cors

def merge_results_across_layers(df):
    return pd.concat([df.assign(layer=key) for key, df in df.items()],
                ignore_index=True)

all_layers_marginal_cors = merge_results_across_layers(layers_model_spec_speaker_marginal_cors)
all_layers_spec_cors = merge_results_across_layers(layers_model_spec_speaker_spec_cors)

all_layers_marginal_cors.to_csv("./marginal_cors_across_layers.csv")            
all_layers_spec_cors.to_csv("./spec_cors_across_layers.csv")

all_layers_marginal_cors = pd.read_csv("./marginal_cors_across_layers.csv")
all_layers_spec_cors = pd.read_csv("./spec_cors_across_layers.csv")

all_layers_marginal_cors["type"] = "all-speakers"
all_layers_spec_cors["type"] = "speaker-specific"

all_layers_results = pd.concat(
    [all_layers_marginal_cors, all_layers_spec_cors],
    axis=0,          # rows
    ignore_index=True,
    sort=False
)

type_order  = ["all-speakers", "speaker-specific"]
layer_order = [i for i in range(1, 13)]

all_layers_results = all_layers_results.copy()
all_layers_results["type"]  = pd.Categorical(all_layers_results["type"],  categories=type_order,  ordered=True)
all_layers_results["layer"] = pd.Categorical(all_layers_results["layer"], categories=layer_order, ordered=True)

palette = {
    "all-speakers":    "#f1223e",  
    "speaker-specific":"#00b4d8",  
}

g = sns.catplot(
    data=all_layers_results, kind="strip",
    x="type", y="corr", hue="type",
    palette=palette, order=type_order,
    col="layer", col_wrap=4, height=3, sharey=True,
    jitter=0.15, dodge=True, alpha=0.4,
    edgecolor="none", linewidth=0, legend=False
)

# precompute mean ± SEM per facet/type
stats = (all_layers_results
         .groupby(['layer','type'])['corr']
         .agg(mean='mean', count='count', std='std')
         .reset_index())
stats['sem'] = stats['std'] / np.sqrt(stats['count'])

xpos = {t:i for i,t in enumerate(type_order)}

# overlay ONLY mean ± SEM (opaque)
for ax, lyr in zip(g.axes.flat, layer_order):
    sub = stats[stats['layer'] == lyr]
    for t in type_order:
        row = sub[sub['type'] == t]
        if row.empty: 
            continue
        ax.errorbar(
            x=[xpos[t]], y=row['mean'].values, yerr=row['sem'].values,
            fmt='o', markersize=7, linewidth=1.2, capsize=3,
            color=palette[t], alpha=1.0, zorder=3, markeredgewidth=0,
            label='_nolegend_'
        )
    ax.grid(True, axis="y", linewidth=0.5)

# clean legend (just the two types)
handles = [plt.Line2D([0],[0], marker='o', linestyle='', color=palette[k], label=k)
           for k in type_order]
g.fig.legend(handles, type_order, loc="upper center", ncol=2, frameon=False)

plt.tight_layout()

plt.savefig('./figures/rsa/comparison_across_layers.png')



