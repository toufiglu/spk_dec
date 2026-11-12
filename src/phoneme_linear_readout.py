
import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import h5py

from transformers import Wav2Vec2Model
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from readout_activation_leftjoin_annotation import read_activation_and_left_join_with_acoustics
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score, cross_val_predict, GroupShuffleSplit, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed



def multiprocess_C(X_sub, y_sub, C):
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            random_state=16,
            penalty="l2",
            C=C,
            solver="lbfgs",
            max_iter=10000
        )
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(
            model,
            X_sub,
            y_sub,
            scoring="roc_auc_ovr",
            cv=cv,
            n_jobs=1
        )
    
    return C, float(np.mean(scores)), float(np.std(scores))
    
def decode_phone_from_one_layer(layer, hid_act_folder, hid_pkl_folder, num_layers, formant_df, dont_read_again, path, dont_run_again):
    act_layer = read_activation_and_left_join_with_acoustics(
        hid_act_folder=hid_act_folder,
        hid_pkl_folder=hid_pkl_folder,
        num_layers=num_layers,
        formant_df=formant_df,
        which_layer=layer,
        dont_read_again=dont_read_again
    )

    # act_layer['phone'] = act_layer['filename'].str.split("_").str[1]
    # act_layer['speaker'] = act_layer['filename'].str.split("_").str[0].str.split("-").str[0]

    train_df, test_df = split_audio_files_by_filename(data=act_layer, test_size=0.1) # take 10% of the data to get best C; the naming of the var test_df really isn't great

    best_C, C_auc_scores = phone_find_best_C(path=path, data=test_df, source=f"model_{layer:02d}", dont_run_again=dont_run_again)

    preds, auc = phone_log_reg(act_layer, best_C, path=path, source=f"layer_{layer:02d}", dont_run_again=dont_run_again)

    return {
        "layer": layer,
        "best_C": best_C,
        "auc_weighted": auc
    }



def split_audio_files_by_filename(data, test_size):
    '''
    stratified by this variable, and split all sound files, so that the one appearing in training won't appear in testing.
    '''
    groups = data["filename"]
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    train_idx, test_idx = next(gss.split(data, groups=groups))
    return (data.iloc[train_idx].reset_index(drop=True),
            data.iloc[test_idx].reset_index(drop=True))  


def phone_find_best_C(path, data, source, dont_run_again): # what if the folder doesn't exist? should makedirs
    
    os.makedirs(path, exist_ok=True)

    filepath = os.path.join(path, f"best_C_{source}.csv")

    if dont_run_again and os.path.exists(filepath):
        C_auc_scores = pd.read_csv(filepath)
    else:
        
        C_values = [0.001, 0.01, 0.1, 1, 10, 100]
        
        X=np.stack(data["act"].to_numpy())
        y=data["phone"].to_numpy() 

        assert len(X)==len(y)
        le = LabelEncoder()
        y = le.fit_transform(y) 


        results = Parallel(n_jobs=2, verbose=10)(
            delayed(multiprocess_C)(X, y, C)
            for C in C_values
        )

        C_auc_scores = pd.DataFrame(results, columns=["C", "mean_auc", "std_auc"]).sort_values("mean_auc", ascending=False)

        C_auc_scores.to_csv(filepath, index=False)

    best_C_row = C_auc_scores.loc[C_auc_scores["mean_auc"].idxmax()]
    best_C = float(best_C_row["C"])
    print(f"The best parameter for {source} was {best_C}")
    return best_C, C_auc_scores

def phone_log_reg(path, data, best_C, source, dont_run_again):
    file = os.path.join(path, f"{source}_pred_prob.h5")

    if dont_run_again and os.path.isfile(file):
        with h5py.File(file, "r") as f:
            preds = f["predictions"][:]
            
        y = data["phone"].to_numpy()  
        le = LabelEncoder()
        y_enc = le.fit_transform(y)   # otherwise y will be missing          

    else:
        
        X = np.stack(data["act"].to_numpy())         # shape (N, D)
        y = data["phone"].to_numpy()               # shape (N,) 

        le = LabelEncoder()
        y_enc = le.fit_transform(y)

        logreg = LogisticRegression(
            random_state=16, penalty="l2", solver="lbfgs", C=best_C, max_iter=10000
        )

        model = make_pipeline(StandardScaler(), logreg)
        groups = data["filename"]
        cv = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)

        preds = cross_val_predict(model, X, y_enc, cv=cv, method="predict_proba", n_jobs=2, groups=groups) # this is the same as the manual function below

        os.makedirs(os.path.dirname(file), exist_ok=True)
        with h5py.File(file, "w") as f:
            f.create_dataset("predictions", data=preds) # save it, so that we can calculate Bayes factors later
    
    auc_weighted = roc_auc_score(y_enc, preds, multi_class="ovr", average="weighted")

    return preds, float(auc_weighted)


def get_phone_auc_plot(df, plot_path):
    df = df.copy().T               # layers as rows
    df["layer"] = df["layer"].astype(float)
    df["auc_weighted"] = df["auc_weighted"].astype(float)
    df = df.sort_values("layer")

    plt.figure()
    plt.plot(df["layer"], df["auc_weighted"], marker="o")
    plt.xticks(np.arange(1, df["layer"].max() + 1, step=1))
    plt.xlabel("Layer")
    plt.ylabel("AUC (weighted)")
    plt.title("Phone Decoding AUC Across Layers")
    plt.grid(True)

    if plot_path is not None:
        plot_dir = os.path.dirname(plot_path)
        if plot_dir:
            os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Decode phones from wav2vec2 hidden activations and plot AUC across layers."
    )

    parser.add_argument(
        "--formant_df",
        type=str,
        required=True,
        help="Path to formant dataframe pickle."
    )

    parser.add_argument(
        "--hid_act_folder",
        type=str,
        required=True,
        help="Folder containing wav2vec hidden activations (per layer)."
    )

    parser.add_argument(
        "--hid_pkl_folder",
        type=str,
        required=True,
        help="Folder storing merged hidden-act + formant PKLs for each layer."
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Name or path of wav2vec2 model."
    )

    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Force rerun logistic regression"
    )

    parser.add_argument(
        "--reread",
        action="store_true",
        help="Force reread of hid_act pickle files."
    )

    parser.add_argument(
        "--plot_path",
        type=str,
        default=None,
        help="Where to save the AUC plot"
    )

    parser.add_argument(
        "--out_csv",
        type=str,
        default=None,
        help="Path to save the AUC results CSV."
    )

    args = parser.parse_args()

    # -------------------------------------
    # Load data and model
    # -------------------------------------
    formant_df = pd.read_pickle(args.formant_df)

    model = Wav2Vec2Model.from_pretrained(args.model)
    num_layers = model.config.num_hidden_layers + 1  

    # -------------------------------------
    # Run decoding for each layer
    # -------------------------------------
    auc_across_layers = {}
    for layer in range(1, num_layers):
        layer_auc = decode_phone_from_one_layer(
            layer=layer,
            hid_act_folder=args.hid_act_folder,
            hid_pkl_folder=args.hid_pkl_folder,
            num_layers=num_layers - 1, 
            formant_df=formant_df,
            dont_run_again=not args.rerun,
            dont_read_again=not args.reread,
        )
        auc_across_layers[f"{layer:02d}"] = layer_auc

    phone_decoding_auc_across_layers = pd.DataFrame(auc_across_layers)

    # -------------------------------------
    # Save CSV
    # -------------------------------------
    if args.out_csv is not None:
        csv_dir = os.path.dirname(args.out_csv)
        if csv_dir:
            os.makedirs(csv_dir, exist_ok=True)
        phone_decoding_auc_across_layers.to_csv(args.out_csv)
        print(f"Saved AUC CSV to {args.out_csv}")

    # -------------------------------------
    # Plot
    # -------------------------------------
    get_phone_auc_plot(phone_decoding_auc_across_layers, args.plot_path)


if __name__ == "__main__":
    main()
