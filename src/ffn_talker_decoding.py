"""Decode talker identity from model hidden states with a feed-forward net.

This is the FFN counterpart to ``logistic_regression_talker_decoding.py`` (which uses
logistic regression). For each transformer layer it trains a small MLP on the
per-frame hidden activations to classify the *speaker*, then reports the
held-out weighted one-vs-rest ROC-AUC. Output format (CSV + plot) matches the
logistic-regression script, so the two are drop-in interchangeable in
``run_pipeline.py`` via ``--decoder {logreg,ffn}``.

Run from inside ``src/`` (sibling modules import directly):

    python ffn_talker_decoding.py \\
        --formant_df ../data/processed/formant.pkl \\
        --hid-act-folder ../data/processed/hidden_states \\
        --hid-pkl-folder ../data/processed/hidden_acts_pkl \\
        --model facebook/wav2vec2-base \\
        --out-csv ../data/processed/talker_auc_ffn.csv \\
        --plot_dir ../data/processed/talker_auc_ffn.png
"""
import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from transformers import Wav2Vec2Model
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score

from acoustics import filter_out_irrelevant_rows
from activations import read_activation_and_left_join_with_acoustics
from decoding import split_audio_files_stratified


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class TalkerFFN(nn.Module):
    """One hidden layer + ReLU/dropout -> speaker logits (log-softmax)."""

    def __init__(self, input_dim, n_classes, hidden=256, p_drop=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return torch.log_softmax(self.net(x), dim=1)


def _loaders(train_df, test_df, le, batch_size=64):
    """z-score on train, apply to test; return (train_loader, test_loader, y_test)."""
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(np.vstack(train_df["act"].values)).astype("float32")
    Xte = scaler.transform(np.vstack(test_df["act"].values)).astype("float32")
    ytr = le.transform(train_df["speaker"].to_numpy())
    yte = le.transform(test_df["speaker"].to_numpy())

    train_ds = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr).long())
    test_ds = TensorDataset(torch.from_numpy(Xte), torch.from_numpy(yte).long())
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=4096, shuffle=False)
    return train_loader, test_loader, yte


def _train(model, train_loader, device, epochs, lr):
    model.to(device).train()
    lossfun = nn.NLLLoss()
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-3)
    for epoch in range(1, epochs + 1):
        running = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optim.zero_grad()
            loss = lossfun(model(X), y)
            loss.backward()
            optim.step()
            running += loss.item()
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f"    epoch {epoch:3d}/{epochs}  loss={running / len(train_loader):.4f}")


@torch.no_grad()
def _predict_proba(model, test_loader, device):
    model.eval()
    probs = [torch.exp(model(X.to(device))).cpu().numpy() for X, _ in test_loader]
    return np.concatenate(probs, axis=0)


def decode_talker_ffn_one_layer(layer, hid_act_folder, hid_pkl_folder, num_layers,
                                formant_df, device, epochs, lr, dont_read_again):
    act_layer = read_activation_and_left_join_with_acoustics(
        hid_act_folder=hid_act_folder,
        hid_pkl_folder=hid_pkl_folder,
        num_layers=num_layers,
        which_layer=layer,
        formant_df=formant_df,
        dont_read_again=dont_read_again,
    )
    act_layer = filter_out_irrelevant_rows(act_layer)

    # Group-aware split so no utterance leaks across train/test, stratified by speaker.
    train_df, test_df = split_audio_files_stratified(
        data=act_layer, test_size=0.2, group_by="speaker")

    le = LabelEncoder().fit(act_layer["speaker"].to_numpy())
    input_dim = np.vstack(act_layer["act"].values[:1]).shape[1]
    n_classes = len(le.classes_)

    train_loader, test_loader, y_test = _loaders(train_df, test_df, le)
    model = TalkerFFN(input_dim=input_dim, n_classes=n_classes)
    print(f"  layer {layer:02d}: input_dim={input_dim} n_speakers={n_classes} "
          f"train={len(train_df)} test={len(test_df)}")
    _train(model, train_loader, device, epochs=epochs, lr=lr)

    probs = _predict_proba(model, test_loader, device)
    auc = roc_auc_score(y_test, probs, multi_class="ovr", average="weighted",
                        labels=np.arange(n_classes))
    return {"layer": layer, "auc_weighted": float(auc)}


def get_talker_auc_plot(results, plot_dir):
    df = results.copy().T
    df["layer"] = df["layer"].astype(float)
    df["auc_weighted"] = df["auc_weighted"].astype(float)
    df = df.sort_values("layer")

    plt.figure()
    plt.plot(df["layer"], df["auc_weighted"], marker="o")
    plt.xticks(np.arange(1, len(df) + 1, step=1))
    plt.xlabel("Layer")
    plt.ylabel("AUC (weighted)")
    plt.title("Talker Decoding AUC Across Layers (FFN)")
    plt.grid(True)
    os.makedirs(os.path.dirname(plot_dir) or ".", exist_ok=True)
    plt.savefig(plot_dir, dpi=300, bbox_inches="tight")


def main():
    parser = argparse.ArgumentParser(
        description="Decode talker from model hidden states with an FFN; plot AUC by layer.")
    parser.add_argument("--formant_df", type=str, required=True,
                        help="Pickle with per-frame phone/speaker/f1-f3 annotation.")
    parser.add_argument("--hid-act-folder", type=str, required=True,
                        help="Folder of per-layer hidden activations (*.pt).")
    parser.add_argument("--hid-pkl-folder", type=str, required=True,
                        help="Folder of per-layer joined activation pickles.")
    parser.add_argument("--model", type=str, required=True,
                        help="Model name/path (used only to infer layer count).")
    parser.add_argument("--out-csv", type=str, required=True,
                        help="Path to save AUC-by-layer CSV.")
    parser.add_argument("--plot_dir", type=str, required=True,
                        help="Path to save the AUC plot.")
    parser.add_argument("--storage", type=str, default="./talker_decoding_ffn",
                        help="Unused by the FFN path; accepted for CLI parity with the logreg script.")
    parser.add_argument("--epochs", type=int, default=60, help="Training epochs per layer.")
    parser.add_argument("--lr", type=float, default=5e-4, help="Adam learning rate.")
    parser.add_argument("--reread", action="store_true",
                        help="Force rebuild of per-layer activation pickles.")
    args = parser.parse_args()

    set_seed(42)
    device = get_device()
    print(f"Using device: {device}")

    formant_df = pd.read_pickle(args.formant_df)
    model = Wav2Vec2Model.from_pretrained(args.model)
    num_layers = model.config.num_hidden_layers

    auc_across_layers = {}
    for layer in range(1, num_layers + 1):
        auc_across_layers[f"{layer:02d}"] = decode_talker_ffn_one_layer(
            layer=layer,
            hid_act_folder=args.hid_act_folder,
            hid_pkl_folder=args.hid_pkl_folder,
            num_layers=num_layers,
            formant_df=formant_df,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            dont_read_again=not args.reread,
        )

    results = pd.DataFrame(auc_across_layers)
    out_dir = os.path.dirname(args.out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    results.to_csv(args.out_csv)
    get_talker_auc_plot(results, args.plot_dir)
    print(f"Done. Wrote {args.out_csv} and {args.plot_dir}")


if __name__ == "__main__":
    main()
