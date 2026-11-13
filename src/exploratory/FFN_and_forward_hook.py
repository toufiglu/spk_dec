

import matplotlib.pyplot as plt
from acoustics import get_formant_with_pca_cols
from decoding import phone_find_best_C, phone_log_reg, split_audio_files_by_filename, split_audio_files_stratified, talker_find_best_C, talker_log_reg
from spec_idx import calculate_speaker_marginal_kld, check_folder_files_exist, create_lda_from_hidden_acts, create_pca_components
from viz_helpers import mean_line_and_points
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import pandas as pd
import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from pathlib import Path
from torch.utils.data import DataLoader,TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


# load the last layer, but this is not ideal
# hid_act_layer12 = pd.read_pickle("./wav_2_vec_hidden_lda/lda_acts_12.pkl")

hid_act_layer9 = pd.read_pickle("./wav_2_vec_hidden_lda/lda_acts_9.pkl")


## format the data: from Mike X Cohen
# split the data, each recording should only occur in either training or testing
train_df, test_df = split_audio_files_by_filename(data=hid_act_layer9, test_size=.1)
# train_all = hid_act_layer12 # the version of experiment where I used all training data

# do z-score normalization for hidden activations, format it into pytorch dataloader
def ffn_data_normalization_and_formatting(train_df, test_df):
    train_data = train_df['act']
    test_data = test_df['act']
    train_labels = train_df['phone']
    test_labels = test_df['phone']

    # z-score normalize the hidden activation
    scaler = StandardScaler()

    train_data = np.vstack(train_data.values)
    train_data_zscored = scaler.fit_transform(train_data)

    test_data = np.vstack(test_data.values)
    test_data_zscored = scaler.transform(test_data)

    # format it so that pytorch works
    train_data   = torch.tensor(train_data_zscored).float()
    test_data   = torch.tensor(test_data_zscored).float()

    le = LabelEncoder()
    train_labels = le.fit_transform(train_labels)
    test_labels = le.transform(test_labels)

    train_labels = torch.tensor(train_labels).long()
    test_labels = torch.tensor(test_labels).long()

    train_data = TensorDataset(train_data, train_labels)
    test_data  = TensorDataset(test_data, test_labels)

    batchsize    = 32
    train_loader = DataLoader(train_data, batch_size=batchsize, shuffle=True, drop_last=True)
    test_loader  = DataLoader(test_data, batch_size=test_data.tensors[0].shape[0])

    return train_loader, test_loader

# Notice: this is the latest version (v3) of the model: all rows were used for training 
train_loader, test_loader = ffn_data_normalization_and_formatting(train_df=train_df, test_df=test_df)

## create model
def phoneme_decoding_ffn():

    class phone_nn(nn.Module):
        def __init__(self):
            super().__init__()

            self.input  = nn.Linear(768, 668)
            self.bnorm1 = nn.BatchNorm1d(668)

            self.fc1    = nn.Linear(668, 568)
            self.bnorm2 = nn.BatchNorm1d(568)

            self.fc2    = nn.Linear(568, 468)
            self.bnorm3 = nn.BatchNorm1d(468)

            self.output = nn.Linear(468, 10)

        def forward(self, x):
            x = self.input(x)
            x = self.bnorm1(x)
            x = F.relu(x)

            x = self.fc1(x)
            x = self.bnorm2(x)
            x = F.relu(x)

            x = self.fc2(x)
            x = self.bnorm3(x)
            x = F.relu(x)

            return torch.log_softmax(self.output(x), dim=1)

    # model, loss, optimizer
    net = phone_nn()
    lossfun = nn.NLLLoss()
    optimizer = torch.optim.Adam(net.parameters(), weight_decay=0.0055, lr=0.00005)
    # l2 norm between 1e-2 and 1e-3, learning rate halved from 1e-4
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)


    return net, lossfun, optimizer, scheduler

# randomly initiate weights
def kaiming_init(net):
    if isinstance(net, nn.Linear):
        nn.init.kaiming_normal_(net.weight, mode='fan_in', nonlinearity='relu')
        if net.bias is not None:
           nn.init.zeros_(net.bias)
    elif isinstance(net, nn.BatchNorm1d):
        nn.init.ones_(net.weight)
        nn.init.zeros_(net.bias)

# a function that trains the model
def function2trainTheModel():

  # number of epochs
  numepochs = 200
  device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

  # create a new model
  net,lossfun,optimizer,scheduler = phoneme_decoding_ffn()
  net.to(device)

  net.apply(kaiming_init)
        
  # initialize losses
  losses    = torch.zeros(numepochs)
  trainBatchAcc  = []
  testAcc   = []
  trainEvalAcc = [] # get accuracy over entire epoch

  # loop over epochs
  for epochi in range(numepochs):

    net.train() 
    # loop over training data batches
    batchAcc  = []
    batchLoss = []
    for X,y in train_loader:
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
      # forward pass and loss
        yHat = net(X)
        loss = lossfun(yHat,y)

        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # loss from this batch
        batchLoss.append(loss.item())

        # compute accuracy
        matches = torch.argmax(yHat,dim=1) == y     # booleans (false/true)
        matchesNumeric = matches.float()             # convert to numbers (0/1)
        accuracyPct = matchesNumeric.mean().mul(100).item() # average and x100
        batchAcc.append( accuracyPct )               # add to list of accuracies
    # end of batch loop...
    # now that we've trained through the batches, get their average training accuracy
    trainBatchAcc.append( float(np.mean(batchAcc)) )

    # and get average losses across the batches
    losses[epochi] = float(np.mean(batchLoss))

    # test accuracy
    net.eval()
    scheduler.step() 
    X,y = next(iter(test_loader)) # extract X,y from test dataloader
    X = X.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)

    with torch.no_grad(): # deactivates autograd
        yHat = net(X)

    # compare the following really long line of code to the training accuracy lines
    accuracyPct = (torch.argmax(yHat, dim=1) == y).float().mean().mul(100).item()
    testAcc.append(accuracyPct) 

    with torch.no_grad():
        correct, total = 0, 0
        for X,y in train_loader:
            yHat = net(X.to(device, non_blocking=True)) 
            preds = yHat.argmax(dim=1)
            correct += (preds == y.to(device, non_blocking=True)).sum().item()
            total   += y.size(0)
        trainEvalAcc.append(100.0 * (correct / total)) 
  # end epochs

  # function output
  return trainBatchAcc, trainEvalAcc, testAcc, losses, net


trainAcc,trainEvalAcc, testAcc,losses,net = function2trainTheModel()

torch.save(net.state_dict(), "./ffn_saved_models/ffn_model_v5_epoch200.pth")
net.load_state_dict(torch.load("./ffn_saved_models/ffn_model_v2_epoch100_l2_1e-4.pth", map_location="cpu")) # load it later


def performance_make_plot_ffn(trainAcc, testAcc, losses, acc_plot_name, loss_plot_name):
    # plot performance across epochs
    epochs = np.arange(1, len(trainAcc) + 1)

    # convert losses tensor to numpy
    loss_vals = losses.cpu().numpy() if hasattr(losses, "cpu") else np.array(losses)

    plt.figure(figsize=(8,5))
    plt.plot(epochs, trainAcc, label='Train Accuracy', color='tab:blue', linewidth=2)
    plt.plot(epochs, testAcc,  label='Test Accuracy',  color='tab:orange', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"./figures/ffn/{acc_plot_name}.png", dpi=300, bbox_inches="tight")

    plt.figure(figsize=(8,5))
    plt.plot(epochs, loss_vals, label='(Averaged) training loss per mini-batch across training epochs', color='tab:blue', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"./figures/ffn/{loss_plot_name}.png", dpi=300, bbox_inches="tight")

performance_make_plot_ffn(trainAcc=trainEvalAcc, testAcc=testAcc, losses=losses, 
                          acc_plot_name="epoch_v5_trainAll_train_test_acc", loss_plot_name="epoch_v5_trainAll_loss")

#######################################################################
# extract hid act; get talker and phone decoding
def extract_ffn_hidden_activation(net, data_loader, device="cpu"):
    net.eval()
    activations = {"input": [], "fc1": [], "fc2": [], "output": []}
    
    # define hooks to grab intermediate outputs
    def get_activation(name):
        def hook(model, input, output):
            activations[name].append(output.detach().cpu())
        return hook
    
    hooks = []
    hooks.append(net.input.register_forward_hook(get_activation("input")))
    hooks.append(net.fc1.register_forward_hook(get_activation("fc1")))
    hooks.append(net.fc2.register_forward_hook(get_activation("fc2")))
    hooks.append(net.output.register_forward_hook(get_activation("output")))

    with torch.no_grad():
        for X, _ in data_loader:
            X = X.to(device)
            _ = net(X)  # forward pass triggers hooks

    # concatenate activations per layer
    for key in activations:
        activations[key] = torch.cat(activations[key], dim=0)

    # remove hooks
    for h in hooks:
        h.remove()

    return activations

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
net.to(device)

# get activations from training or test data
train_acts = extract_ffn_hidden_activation(net, train_loader, device)
test_acts  = extract_ffn_hidden_activation(net, test_loader, device)

def get_batched_rows(df, loader):
    num_total = len(df)
    batch_size = train_loader.batch_size
    num_used = (num_total // batch_size) * batch_size
    num_dropped = num_total - num_used
    print(num_total, num_used, num_dropped)

    used_idx = np.arange(num_used)
    dropped_idx = np.arange(num_used, num_total)
    print(dropped_idx) 

    df_used = df.iloc[:num_used].copy()

    return df_used

train_df_used = get_batched_rows(df=train_df, loader=train_loader)

# test_loader only had 1 mini-batch.

for key, tensor in train_acts.items():
    train_df_used[key] = tensor.cpu().numpy().tolist()

for key, tensor in test_acts.items():
    test_df[key] = tensor.cpu().numpy().tolist()

# this version, all rows were used in training
hid_act_layer9_with_ffn_act = pd.concat([train_df_used, test_df], axis=0, ignore_index=True, sort=False)
hid_act_layer9_with_ffn_act = hid_act_layer9_with_ffn_act.rename(columns={"act": "layer_9"})
cols_to_remove = ["lda_1", "lda_2", "lda_3"]
hid_act_layer9_with_ffn_act = hid_act_layer9_with_ffn_act.drop(columns=cols_to_remove)

with open("./ffn_saved_models/ffn_v5_epoch200_hidden_acts.pkl", "wb") as f: 
    pickle.dump(hid_act_layer9_with_ffn_act, f)

def decode_talker_from_ffn_one_layer(df, layer, dont_run_again=True):

    df_copy = df.copy()

    df_copy.rename(columns={layer: 'act'}, inplace=True)

    train_df, test_df = split_audio_files_stratified(data=df_copy, test_size=0.1, group_by="speaker") 

    best_C, C_auc_scores = talker_find_best_C(data=test_df, source=f"ffn_{layer}", dont_run_again=dont_run_again)

    preds, auc = talker_log_reg(df_copy, best_C, source=f"ffn_{layer}", dont_run_again=dont_run_again)

    return {
        "layer": layer,
        "best_C": best_C,
        "auc_weighted": auc
    }

layers = ["layer_9", "input", "fc1", "fc2", "output"]

ffn_talker_auc_across_layers = {}
for layer in layers:
    layer_auc = decode_talker_from_ffn_one_layer(layer=layer, df=hid_act_layer9_with_ffn_act, dont_run_again=False)
    ffn_talker_auc_across_layers[f"{layer}"] = layer_auc
talker_decoding_auc_across_layers = pd.DataFrame(ffn_talker_auc_across_layers)


def decode_phone_from_ffn_one_layer(df, layer, dont_run_again=True):
    df_copy = df.copy()

    df_copy.rename(columns={layer: 'act'}, inplace=True)

    train_df, test_df = split_audio_files_by_filename(data=df_copy, test_size=0.1) 

    best_C, C_auc_scores = phone_find_best_C(data=test_df, source=f"ffn_{layer}", dont_run_again=dont_run_again)

    preds, auc = phone_log_reg(df_copy, best_C, source=f"ffn_{layer}", dont_run_again=dont_run_again)

    return {
        "layer": layer,
        "best_C": best_C,
        "auc_weighted": auc
    }

ffn_phone_auc_across_layers = {}
for layer in layers:
    layer_auc = decode_phone_from_ffn_one_layer(layer=layer, df=hid_act_layer9_with_ffn_act, dont_run_again=False)
    ffn_phone_auc_across_layers[f"{layer}"] = layer_auc
phone_decoding_auc_across_layers = pd.DataFrame(ffn_phone_auc_across_layers)


def decoding_make_plot_ffn(result_df, plot_name, layers=["layer_9", "input", "fc1", "fc2", "output"]):
    df = result_df.copy()

    df = df.T 
    df["layer"] = pd.Categorical(df["layer"], categories=layers, ordered=True) 
    # sort out the order of the x axis in the plot

    df["auc_weighted"] = df["auc_weighted"].astype(float)

    df = df.sort_values("layer")

    # Plot
    plt.figure()
    plt.plot(df["layer"], df["auc_weighted"], marker="o")
    plt.xlabel("Layer")
    plt.ylabel("AUC (weighted)")
    plt.axhline(y=0.5, linestyle="--", linewidth=1, alpha=0.7, color="red")
    plt.title(f"{plot_name}")
    plt.grid(True)
    plt.savefig(f"./figures/ffn/{plot_name}.png", dpi=300, bbox_inches="tight")
    plt.close()

decoding_make_plot_ffn(result_df=talker_decoding_auc_across_layers, plot_name="ffn_talker_decoding")
decoding_make_plot_ffn(result_df=phone_decoding_auc_across_layers, plot_name="ffn_phone_decoding")
#######################################################################
# get KL ratio


def get_spec_idx_for_every_ffn_layer(ffn_act_df, path, formant_df, dont_run_again=True):
    spec_idx = {}
    p = Path(path) # we will save results to this directory.
    os.makedirs(p, exist_ok=True)
    if dont_run_again and check_folder_files_exist(folder=p, layers=layers): # check if dir exists, all csv files were already computed.
        for file in p.glob("layer_*_spec_idx.csv"):
            i = file.stem.split("_")[1]
            df = pd.read_csv(file)
            spec_idx[i] = df
    else:
        for layer in layers:
            df_copy = ffn_act_df.copy()
            cols_to_drop = []

            if 'act' in df_copy.columns:
                cols_to_drop.append('act')

            other_layers = [l for l in layers if l != layer]
            cols_to_drop += other_layers

            df_copy = df_copy.drop(columns=cols_to_drop)

            act = create_lda_from_hidden_acts(layer=layer, df=df_copy, hid_act_col=layer, lda_act_filename=f"./ffn_saved_models/ffn_with_lda_{layer}.pkl")

            act = create_pca_components(df=act, lda_components=[f"{layer}_lda_1", f"{layer}_lda_2", f"{layer}_lda_3"])

            # merge formant_df cols with the act
            formant_df_cols = formant_df[["time", "filename", "pca_f1", "pca_f2", "pca_f3"]] # but I need to make sure pca is done for the formant_df
            act = pd.merge(act, formant_df_cols, on=["time", "filename"])

            # average across vowels
            act_vowel_mean_pooled = (act
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
            # calculate kld
            act_spec_idx = calculate_speaker_marginal_kld(df=act_vowel_mean_pooled)
            act_spec_idx_df = pd.DataFrame(act_spec_idx)
            
            # name the output file
            output_path = p / f"layer_{layer}_spec_idx.csv"
            # save it
            act_spec_idx_df.to_csv(output_path) # save the results

            spec_idx[layer] = act_spec_idx_df
            print(f'The kld for Layer {layer} is done.')
        
    return spec_idx

kl_ratio_ffn = get_spec_idx_for_every_ffn_layer(ffn_act_df=hid_act_layer9_with_ffn_act, formant_df=get_formant_with_pca_cols(), dont_run_again=False, path="./ffn_saved_models")

#######################################################################
# visualization
all_results = []
for layer in layers:
    results = kl_ratio_ffn[layer]
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

order = ["layer_12","input","fc1","fc2","output"]
all_results["layer"] = pd.Categorical(all_results["layer"], categories=order, ordered=True)

# do some plotting:
g = sns.FacetGrid(all_results, col="vowel", col_wrap=3, sharex=True, sharey=True, height=3, margin_titles=True)

# raw points
g.map_dataframe(sns.scatterplot, x="layer", y="r", alpha=0.2, s=12, color=None, ax=ax)

# mean line + mean points (per-vowel facet)
g.map_dataframe(mean_line_and_points, color=None)

# hline, ticks, limits, grid
for ax in g.axes.flatten():
    ax.axhline(0.5, ls="--", lw=2, color="red")
    ax.set_xticks(range(1, 4))
    ax.set_ylim(0, 1)
    ax.set_yticks(np.linspace(0, 1, 11))
    ax.grid(True, axis="both", which="major", linewidth=0.5, color="grey")
    ax.set_facecolor("white")

g.set_axis_labels("Layers", "KL ratio")
g.fig.tight_layout()
plt.savefig("./figures/ffn/ffn_v4_kl_ratio_plot.png", dpi=600, bbox_inches="tight")


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
plt.xticks(range(1, 4))
plt.yticks(np.linspace(0, 1, 11))
plt.ylim(0, 1)
plt.xlabel("Layers", fontsize=14, fontweight="bold")
plt.ylabel("KL ratio", fontsize=14, fontweight="bold")
plt.grid(True, which="major", linewidth=0.5, color="grey")
plt.gca().set_facecolor("white")

# legend on the right
plt.legend(title="speaker", bbox_to_anchor=(1, 0.75), loc="upper left", borderaxespad=0., ncol=2)
plt.tight_layout()
plt.savefig("./figures/ffn/ffn_v4_kl_ratio_by_speaker.png", dpi=600, bbox_inches="tight")

#######################################################################
# count phone distributions
counts = train_df["phone"].value_counts()          # counts per phone
ax = counts.plot(kind="bar", figsize=(8,4))
ax.set_xlabel("Phone")
ax.set_ylabel("Count")
ax.set_title("Phone distribution (train)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()
plt.close()

counts = test_df["phone"].value_counts()          # counts per phone
ax = counts.plot(kind="bar", figsize=(8,4))
ax.set_xlabel("Phone")
ax.set_ylabel("Count")
ax.set_title("Phone distribution (test)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()
plt.close()
