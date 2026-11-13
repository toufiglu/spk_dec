 
import os
import torch
import pandas as pd
import h5py
import numpy as np
import matplotlib.pyplot as plt
import hoggorm as ho

from acoustics import filter_out_irrelevant_rows
from decoding import decode_phone_COEF_from_one_layer, decode_talker_COEF_from_one_layer, phone_find_best_C, phone_log_reg, split_audio_files_by_filename, split_audio_files_stratified, talker_find_best_C, talker_log_reg
from transformers import Wav2Vec2Model
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from plotnine import *


def get_talker_coef_across_layers(
    num_layers,
    hid_act_folder,
    hid_pkl_folder,
    formant_df,
    storage="./coef_talker_log_reg",
    fname="talker_coef.pkl",
    dont_run_again=True,
    dont_read_again=True,
):
    os.makedirs(storage, exist_ok=True)
    coef_path = os.path.join(storage, fname)
    # load if ready
    if os.path.exists(coef_path):
        return pd.read_pickle(coef_path)
    # run if not
    talker_coef_across_layers = {}

    for layer in range(1, num_layers):
        layer_coef = decode_talker_COEF_from_one_layer(
            layer=layer,
            hid_act_folder=hid_act_folder,
            hid_pkl_folder=hid_pkl_folder,
            num_layers=num_layers - 1,
            formant_df=formant_df,
            dont_run_again=dont_run_again,
            dont_read_again=dont_read_again,
            storage=storage,
        )
        talker_coef_across_layers[f"{layer:02d}"] = layer_coef

    talker_decoding_coef_across_layers = pd.DataFrame(talker_coef_across_layers)
    talker_coef = talker_decoding_coef_across_layers.T

    talker_coef.to_pickle(coef_path)
    return talker_coef

def get_phone_coef_across_layers(
    num_layers,
    hid_act_folder,
    hid_pkl_folder,
    formant_df,
    storage="./coef_phone_log_reg",
    fname="phone_coef.pkl",
    dont_run_again=True,
    dont_read_again=True,
):
    os.makedirs(storage, exist_ok=True)
    coef_path = os.path.join(storage, fname)

    if os.path.exists(coef_path):
        return pd.read_pickle(coef_path)

    phone_coef_across_layers = {}

    for layer in range(1, num_layers):
        layer_coef = decode_phone_COEF_from_one_layer(
            layer=layer,
            hid_act_folder=hid_act_folder,
            hid_pkl_folder=hid_pkl_folder,
            num_layers=num_layers - 1,
            formant_df=formant_df,
            dont_run_again=dont_run_again,
            dont_read_again=dont_read_again,
            storage=storage,
        )
        phone_coef_across_layers[f"{layer:02d}"] = layer_coef

    phone_decoding_coef_across_layers = pd.DataFrame(phone_coef_across_layers)
    phone_coef = phone_decoding_coef_across_layers.T

    phone_coef.to_pickle(coef_path)
    return phone_coef

def plot_zero_count_per_layer(df, percent):
    plt.figure()
    plt.plot(df["layer"], df["layer_count"], marker="o", color="#A80874")
    plt.xticks(np.arange(1, 13, step=1))
    plt.xlabel("Layer")
    plt.ylabel("Number of zero out neurons")
    plt.title(f"The distribution of top {percent}% of talker neurons")
    plt.grid(True)
    plt.savefig("talker_zero_out_per_layer.png", dpi=300, bbox_inches="tight")


processed_dir = "./data/processed/"

formant_df = pd.read_pickle(os.path.join(processed_dir, "formant.pkl")) # for full recordings

hid_act_folder = os.path.join(processed_dir, "./wav_2_vec_hidden_lda")
hid_pkl_folder = os.path.join(processed_dir, "./wav_2_vec_hidden_pkl")

model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
num_layers = model.config.num_hidden_layers + 1

talker_coef = get_talker_coef_across_layers(num_layers=num_layers,
                                            hid_act_folder=hid_act_folder,
                                            hid_pkl_folder=hid_pkl_folder,
                                            formant_df=formant_df)

phone_coef = get_phone_coef_across_layers(num_layers=num_layers,
                            hid_act_folder=hid_act_folder,
                            hid_pkl_folder=hid_pkl_folder,
                            formant_df=formant_df)

phone_mat = np.stack(phone_coef["coef"].to_numpy())
talker_mat = np.stack(talker_coef["coef"].to_numpy())

# ---- cosine similarity ----
cosine = [cosine_similarity(talker_mat[i], phone_mat[i]) for i in range(phone_mat.shape[0])]
cosine_results_dfs = [
    pd.DataFrame({
        "cosine": cosine[layer].ravel(),
        "layer": layer
    })
    for layer in range(len(cosine))
]
cosine_results = pd.concat(cosine_results_dfs, ignore_index=True)
cosine_results["layer"] = cosine_results["layer"] + 1

p = (
    ggplot(cosine_results, aes(x="layer", y="cosine")) 
    + geom_point(alpha=0.4)
    + stat_summary( # if we want to run stats in the 
        fun_y=np.mean,
        geom="point",
        color="red",
        size=2
    )
    + stat_summary(
        fun_data="mean_se",
        geom="errorbar",
        color="red",
        width=0.2
    )
    + scale_x_continuous(breaks=range(1, 12), labels=[f"{i}" for i in range(1,12)])
    + labs(x="Layer", y="Cosine similarity", title="Talker/Phone cosine similarity")
    + theme(
        axis_text_x=element_text(size=20, weight="bold"),
        axis_text_y=element_text(size=20),
        axis_title_x=element_text(size=15, weight="bold"),
        axis_title_y=element_text(size=20, weight="bold"),
        plot_title=element_text(size=20),
        panel_background=element_rect(fill="white"),
        panel_grid_major=element_line(color="grey", size=0.5),
    )
)

p.save(
    "cosine_similarity_by_layer_gg.png",
    width=20,
    height=10,
    dpi=300,
    verbose=True
)


# ---- RV coef ----
'''
but we will need an error bar for the coefficient
'''
rv = [
    ho.RVcoeff([phone_mat[i].T, talker_mat[i].T])[0, 1]
    for i in range(phone_mat.shape[0])
]

rv_coef = pd.Series(rv, index=phone_coef.index, name="RV")
rv_coef = rv_coef.rename_axis("layer").reset_index()
rv_coef = rv_coef.sort_values("layer")

p = (
    ggplot(rv_coef, aes(x="layer", y="RV")) 
    + geom_point()
    + geom_line()
    + scale_x_continuous(breaks=range(1, 13), labels=[f"{i:02d}" for i in range(1,13)])
    + labs(x="Layer", y="RV", title="Talker/Phone RV Coefficient")
    + theme_minimal()
)

# ---- zero-out ----
# for now, try removing neurons with a coef higher than 2

talker_flat = np.concatenate([np.ravel(coef) for coef in talker_coef["coef"]])
abs_talker_flat = np.abs(talker_flat)
percentage = 1
threshold = np.quantile(abs_talker_flat, 1-(percentage*0.01)) # top-k of talker sensitive units
top10_mask = (np.abs(talker_mat) >= threshold).astype(int)
count_mask_per_neuron = (top10_mask == 1).sum(axis=1)
count_per_layer = (count_mask_per_neuron == 1).sum(axis=1)

df_count_zero = pd.DataFrame({"layer_count": count_per_layer,
                              "layer": np.arange(1, len(count_per_layer) + 1)})

plot_zero_count_per_layer(df_count_zero, percent=str(percentage)) # plot how many neurons are zeroed-out in that layer

zero_out_toggle = (count_mask_per_neuron==0).astype(int)
talker_auc_across_layers = {}
phone_auc_across_layers = {}

for layer in range(1, num_layers):
    act_layer = pd.read_pickle(f"./data/processed/wav_2_vec_hidden_pkl/hidden_acts_{layer}.pkl")
    act_layer = filter_out_irrelevant_rows(act_layer)
    mask = zero_out_toggle[layer-1]

    X = np.stack(act_layer["act"].to_numpy())   
    X_masked = X * mask
    act_layer["act"] = list(X_masked)

    train_df, test_df = split_audio_files_stratified(   data=act_layer,
                                                                test_size=0.1,
                                                                group_by="speaker" )

    best_C, C_auc_scores = talker_find_best_C(path="./zero_out_experiments/talker_log_reg", data=test_df, source=f"model_{layer:02d}", dont_run_again=False)

    preds, auc = talker_log_reg(act_layer, best_C, source=f"layer_{layer:02d}", path="./zero_out_experiments/talker_log_reg", dont_run_again=False)

    talker_auc_across_layers[f"{layer:02d}"] = auc

    train_df, test_df = split_audio_files_by_filename(data=act_layer, test_size=0.1) # take 10% of the data to get best C; the naming of the var test_df really isn't great

    best_C, C_auc_scores = phone_find_best_C(path="./zero_out_experiments/phone_log_reg", data=test_df, source=f"model_{layer:02d}", dont_run_again=False)

    preds, auc = phone_log_reg(path="./zero_out_experiments/phone_log_reg", data=act_layer, best_C=best_C, source=f"layer_{layer:02d}", dont_run_again=False)
    
    phone_auc_across_layers[f"{layer:02d}"] = auc

talker_decoding_auc_across_layers = pd.DataFrame.from_dict(talker_auc_across_layers, orient="index", columns=["auc_weighted"]).rename_axis("layer")
phone_decoding_auc_across_layers = pd.DataFrame.from_dict(phone_auc_across_layers, orient="index", columns=["auc_weighted"]).rename_axis("layer")

def process_decoding_result_df(df, condition):
    df = df.copy().reset_index()
    df["layer"] = df["layer"].astype(float)
    df["auc_weighted"] = df["auc_weighted"].astype(float)
    df['condition'] = condition
    df = df.sort_values("layer")
    return df

# visualization
def get_talker_auc_plot_ggplot(talker_results, phone_results, plot_dir):
    df_talker = process_decoding_result_df(talker_results, condition="talker")
    df_phone = process_decoding_result_df(phone_results, condition="phone")
    df = pd.concat([df_talker, df_phone], ignore_index=True)

    p = (
        ggplot(df, aes(x="layer", y="auc_weighted", group='condition', color='condition'))
        + geom_line()
        + geom_point(size=2)
        + scale_x_continuous(breaks=np.arange(1, 13, 1))
        + labs(
            x="Layer",
            y="AUC (weighted)",
            title="Talker/phone Decoding AUC Across Layers"
        ) 
        + theme_bw()
        + theme(
            figure_size=(6, 4),
            axis_text_x=element_text(size=20, weight="bold"),
            axis_text_y=element_text(size=20),
            axis_title_x=element_text(size=15, weight="bold"),
            axis_title_y=element_text(size=20, weight="bold"),
            plot_title=element_text(size=20),
            panel_background=element_rect(fill="white"),
            panel_grid_major=element_line(color="grey", size=0.5),
            )
    )

    p.save(plot_dir, dpi=300)
    return p

get_talker_auc_plot_ggplot(talker_results=talker_decoding_auc_across_layers, phone_results=phone_decoding_auc_across_layers, plot_dir="./top_1_ablation_decoding.png")
# ---- rotate representation----
