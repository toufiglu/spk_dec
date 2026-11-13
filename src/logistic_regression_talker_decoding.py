
import argparse
import os
import torch
import pandas as pd
import h5py
import numpy as np
import matplotlib.pyplot as plt
from transformers import Wav2Vec2Model
from sklearn.preprocessing import LabelEncoder,StandardScaler

from decoding import decode_talker_from_one_layer

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
    parser.add_argument(
        "--storage",
        type=str,
        default="./talker_decoding",
        help="directory for per-layer best-C tables and prediction probabilities.",
    )

    args = parser.parse_args()

    formant_df = pd.read_pickle(args.formant_df)

    model = Wav2Vec2Model.from_pretrained(args.model) # I will need to change this if we have another model
    num_layers = model.config.num_hidden_layers + 1  

    auc_across_layers = {}
    for layer in range(1, num_layers):
        layer_auc = decode_talker_from_one_layer(
            layer=layer,
            hid_act_folder=args.hid_act_folder,
            hid_pkl_folder=args.hid_pkl_folder,
            num_layers=num_layers - 1,
            formant_df=formant_df,
            path=args.storage,
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
