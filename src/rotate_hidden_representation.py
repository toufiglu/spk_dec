
import importlib
import os
import torch
import pandas as pd
import h5py
import numpy as np
import matplotlib.pyplot as plt
import hoggorm as ho

import helpers
from transformers import Wav2Vec2Model
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from plotnine import *

importlib.reload(helpers)

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
        layer_coef = helpers.decode_talker_COEF_from_one_layer(
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
        layer_coef = helpers.decode_phone_COEF_from_one_layer(
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

talker_mat = np.stack(talker_coef["coef"].to_numpy())

talker_coef_2spk = talker_coef.copy()

talker_coef_2spk["coef"] = talker_coef_2spk["coef"].apply(
    lambda x: np.asarray(x)[:2, :]
)

hid_act_1 = pd.read_pickle("./data/processed/wav_2_vec_hidden_pkl/hidden_acts_1.pkl")
spk = hid_act_1['speaker'].to_numpy() # before encoding
le = LabelEncoder()
spk_enc = le.fit_transform(spk)

classes = list(range(0, talker_coef_2spk["coef"][0].shape[0])) 
speakers = le.inverse_transform(np.array(classes))


talker_coef_2spk_layer1 = talker_coef_2spk[talker_coef_2spk['layer']==1]
diff = talker_coef_2spk_layer1['coef'][0][1] - talker_coef_2spk_layer1['coef'][0][0]
k=0.5

hid_act_1_2spk = hid_act_1[hid_act_1['speaker'].isin(speakers)]
hid_act_1_2spk["class"] = le.transform(hid_act_1_2spk["speaker"])


mask = hid_act_1_2spk["class"] == 0

hid_act_1_2spk.loc[mask, 'act'] = (
    hid_act_1_2spk.loc[mask, "act"]
    .apply(lambda x: add_diff(x, ratio=k, diff=diff))
)


def add_diff(x, ratio, diff):
    return x + ratio * diff


def rotate_hid_2spk_and_decode_phone_from_one_layer(layer,
                                hid_act_folder,
                                    hid_pkl_folder,
                                       num_layers,
                                         formant_df,
                                           dont_run_again,
                                             dont_read_again,
                                               k,
                                               talker_coef_2spk,
                                               path
                                                ):
    
    row = talker_coef_2spk.loc[talker_coef_2spk["layer"] == layer]

    if row.empty:
        raise ValueError(f"No talker coef for layer {layer}")

    coef_mat = np.asarray(row["coef"].iloc[0])
    diff = coef_mat[1] - coef_mat[0]

    act_layer = helpers.read_activation_and_left_join_with_acoustics(
        hid_act_folder=hid_act_folder,
        hid_pkl_folder=hid_pkl_folder,
        num_layers=num_layers-1,
        formant_df=formant_df,
        which_layer=layer,
        dont_read_again=dont_read_again
    )

    spk = act_layer['speaker'].to_numpy() # before encoding
    le = LabelEncoder()
    spk_enc = le.fit_transform(spk)

    classes = np.array([0, 1])
    speakers = le.inverse_transform(classes)


    act_layer_2spk = act_layer.loc[act_layer["speaker"].isin(speakers)].copy()
    act_layer_2spk["class"] = le.transform(act_layer_2spk["speaker"])

    mask = act_layer_2spk["class"] == 0

    act_layer_2spk.loc[mask, 'act'] = (
        act_layer_2spk.loc[mask, "act"]
        .apply(lambda x: add_diff(x, ratio=k, diff=diff))
    )
    
    act_layer_spk0 = act_layer_2spk[act_layer_2spk['class']==0]

    act_layer_spk0 = helpers.filter_out_irrelevant_rows(act_layer_spk0)

    train_df, test_df = helpers.split_audio_files_by_filename(data=act_layer_spk0, test_size=0.1)

    best_C, C_auc_scores = helpers.phone_find_best_C(path=path, data=test_df, source=f"model_{layer:02d}", dont_run_again=dont_run_again)

    preds, auc = helpers.phone_log_reg(path=path, data=act_layer_spk0, best_C=best_C, source=f"layer_{layer:02d}", dont_run_again=dont_run_again)

    return {
         "layer": layer,
        "best_C": best_C,
        "auc_weighted": auc
    }


processed_dir = "./data/processed/"

formant_df = pd.read_pickle(os.path.join(processed_dir, "formant.pkl")) # for full recordings

hid_act_folder = os.path.join(processed_dir, "./wav_2_vec_hidden_lda")
hid_pkl_folder = os.path.join(processed_dir, "./wav_2_vec_hidden_pkl")

model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
num_layers = model.config.num_hidden_layers + 1

def process_decoding_result_df(df):
    df = df.copy()
    df["layer"] = df["layer"].astype(float)
    df["auc_weighted"] = df["auc_weighted"].astype(float)
    df['condition'] = df["k"]
    df = df.sort_values("layer")
    return df

def get_auc_plot_ggplot(phone_results, plot_dir):
    df_phone = process_decoding_result_df(phone_results)

    p = (
        ggplot(df_phone, aes(x="layer", y="auc_weighted", group='condition', color='condition'))
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


k_values = np.linspace(0, 1, 11)

def sweep_k_rotate_phone_decoding(
    k_values,
    num_layers,
    hid_act_folder,
    hid_pkl_folder,
    formant_df,
    talker_coef_2spk,
    base_path,
    dont_run_again=True,
    dont_read_again=True,
):
    rows = []

    for k in k_values:
        print(f"\n=== Running k = {k} ===")

        for layer in range(1, num_layers + 1):
            out = rotate_hid_2spk_and_decode_phone_from_one_layer(
                layer=layer,
                hid_act_folder=hid_act_folder,
                hid_pkl_folder=hid_pkl_folder,
                num_layers=num_layers,
                formant_df=formant_df,
                dont_run_again=dont_run_again,
                dont_read_again=dont_read_again,
                k=k,
                talker_coef_2spk=talker_coef_2spk,
                path=f"{base_path}/{k}"
            )

            rows.append({
                "k": k,
                "layer": out["layer"],
                "auc_weighted": out["auc_weighted"],
                "best_C": out["best_C"],
            })

    return pd.DataFrame(rows)

rotate_phone_results = sweep_k_rotate_phone_decoding(
    k_values=k_values,
    num_layers=num_layers-1,
    hid_act_folder=hid_act_folder,
    hid_pkl_folder=hid_pkl_folder,
    formant_df=formant_df,
    talker_coef_2spk=talker_coef_2spk,
    base_path="./rotate_space/phone_log_reg",
    dont_run_again=True,
    dont_read_again=True,
)

rotate_phone_results["k"] = rotate_phone_results["k"].round(1)

get_auc_plot_ggplot(phone_results=rotate_phone_results, plot_dir="./rotation_all_k_results.png")
