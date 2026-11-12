
import importlib
import helpers
import argparse
import os
import torch
import pandas as pd
import h5py
import numpy as np
import matplotlib.pyplot as plt
from transformers import Wav2Vec2Model
from sklearn.preprocessing import LabelEncoder,StandardScaler

importlib.reload(helpers)

def get_talker_auc_plot(results, plot_dir):
    df = results.copy()
    df = df.T
    df["layer"] = df["layer"].astype(float)
    df["auc_weighted"] = df["auc_weighted"].astype(float)
    df = df.sort_values("layer")

    plt.figure()
    plt.plot(df["layer"], df["auc_weighted"], marker="o")
    plt.xticks(np.arange(1, 13, step=1))
    plt.xlabel("Layer")
    plt.ylabel("AUC (weighted)")
    plt.title("Talker Decoding AUC Across Layers")
    plt.grid(True)
    plt.savefig(f"{plot_dir}", dpi=300, bbox_inches='tight')

def main():
    parser = argparse.ArgumentParser(description="decode talker from model by logistic regression and plot auc")

    parser.add_argument(
        "--formant_df",
        type=str,
        required=True,
        help="the formant df extracted",
    )

    parser.add_argument(
        "--hid-act-folder",
        type=str,
        required=True,
        help="Folder containing wav2vec hidden activations (per layer) for each file.",
    )
    parser.add_argument(
        "--hid-pkl-folder",
        type=str,
        required=True,
        help="Folder that stores one pkl file for each layer",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Name or path of used models",
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Force rerun of decoding (override dont_run_again).",
    )
    parser.add_argument(
        "--reread",
        action="store_true",
        help="Force reread of hid_act pkl file(override dont_read_again).",
    )
    parser.add_argument(
        "--plot_dir",
        type=str,
        required=True,
        help="the location to store the plot",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        required=True,
        help="path to save AUC results as CSV.",
    )

    args = parser.parse_args()

    formant_df = pd.read_pickle(args.formant_df)

    model = Wav2Vec2Model.from_pretrained(args.model) # I will need to change this if we have another model
    num_layers = model.config.num_hidden_layers + 1  

    auc_across_layers = {}
    for layer in range(1, num_layers):
        layer_auc = helpers.decode_talker_from_one_layer(
            layer=layer,
            hid_act_folder=args.hid_act_folder,
            hid_pkl_folder=args.hid_pkl_folder,
            num_layers=num_layers - 1,  
            formant_df=formant_df,
            dont_run_again=not args.rerun,
            dont_read_again=not args.reread,
        )
        auc_across_layers[f"{layer:02d}"] = layer_auc

    talker_decoding_auc_across_layers = pd.DataFrame(auc_across_layers)

    if args.out_csv is not None:
        out_dir = os.path.dirname(args.out_csv)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        talker_decoding_auc_across_layers.to_csv(args.out_csv)

    get_talker_auc_plot(talker_decoding_auc_across_layers, args.plot_dir)

if __name__ == "__main__":
    main()

'''
waveform_gammatone = pd.read_pickle("waveform_gammatone.pkl")
gammatone_longform = helpers.gammatone_long(waveform_gammatone, formant_df=formant_df)
gammatone_longform["phone"] = gammatone_longform["phone"].str.replace(r"\d+", "", regex=True)


# decode from raw spectrogram
# there is not much of a point doing multiprocessing here: it's quite fast; I changed the function here, so change the codes below
if __name__ == "__main__":
    
    gammatone_longform = helpers.filter_out_irrelevant_rows(gammatone_longform) # remove irrelevant rows

    best_C_acou, C_auc_scores_acou = helpers.talker_find_best_C(data=gammatone_longform, source="gt", dont_run_again=True) 

    preds_acou, auc_acou = helpers.talker_log_reg(data=gammatone_longform, best_C=best_C_acou, source="gt", dont_run_again=True) 
    
# get bayes factor
le = LabelEncoder()
y_enc = le.fit_transform(gammatone_longform["speaker"])
speaker_map = dict(zip(le.classes_, le.transform(le.classes_)))

speaker_map = {k:int(v) for k, v in speaker_map.items()}

gammatone_longform["speaker_idx"] = gammatone_longform["speaker"].map(speaker_map)

gammatone_longform["speaker_prob"] = preds_acou[
    np.arange(len(preds_acou)), 
    gammatone_longform["speaker_idx"].values
]

eps = 1e-12 # some row returned 0 probability, log prob will blow up to -inf
gammatone_logprob = np.log(gammatone_longform["speaker_prob"].clip(lower=eps)).sum()

gammatone_logprob = np.log(gammatone_longform["speaker_prob"]).sum()

ratio_dict = {}

for layer in range(1, 13):

    # load layer probabilities
    file = f"./log_reg/layer_{layer:02d}_pred_prob.h5"
    with h5py.File(file, "r") as f:
        preds_layer = f["predictions"][:]   # (N, speakers)

    # load layer dataframe
    layer_df = pd.read_pickle(f"./context_size/single_vowels/w2v_hid_act_formant_pkl/hidden_acts_{layer}.pkl")

    # map speaker to index
    layer_df["speaker_idx"] = layer_df["speaker"].map(speaker_map)

    layer_df = helpers.filter_out_irrelevant_rows(layer_df) 

    # extract probability of true speaker
    layer_df["speaker_prob"] = preds_layer[
        np.arange(len(preds_layer)),
        layer_df["speaker_idx"].values
    ]

    # compute log-prob
    model_logprob = np.log(layer_df["speaker_prob"]).sum()

    # ratio in log-space
    ratio = model_logprob - gammatone_logprob

    ratio_dict[layer] = ratio
    print(f"Layer {layer}: ratio = {ratio}")

ratio_df = pd.DataFrame({
    "layer": list(ratio_dict.keys()),
    "log_prob_ratio": list(ratio_dict.values())
})

# Plot
plt.figure()
plt.plot(ratio_df["layer"], ratio_df["log_prob_ratio"], marker="o")
plt.xlabel("Layer")
plt.ylabel("log Bayes factor")
plt.title("Bayes factor across layers")
plt.grid(True)
plt.show()
'''

