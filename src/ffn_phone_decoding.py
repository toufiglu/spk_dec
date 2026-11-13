import matplotlib.pyplot as plt
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pickle
import pandas as pd
import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from pathlib import Path
from torch.utils.data import DataLoader,TensorDataset
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder


def split_audio_files_by_filename(data, test_size):
    '''
    stratified by this variable, and split all sound files, so that the one appearing in training won't appear in testing.
    '''
    groups = data["filename"]
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    train_idx, test_idx = next(gss.split(data, groups=groups))
    return (data.iloc[train_idx].reset_index(drop=True),
            data.iloc[test_idx].reset_index(drop=True))  

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

## create model
def phoneme_decoding_ffn():

    class phone_nn(nn.Module):
        def __init__(self):
            super().__init__()

            self.output = nn.Linear(768, 40)

        def forward(self, x):

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
def function2trainTheModel(train_loader, test_loader):

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


def main():
    parser = argparse.ArgumentParser(description="decode phone from hidden activation")

    parser.add_argument("--input_folder", type=str, required=True, help="hidden activation folder")

    parser.add_argument("--model_name", type=str, required=True, help="Model name (to be used for folder naming)")

    args = parser.parse_args()
    folder = Path(args.input_folder)
    model_ffn_dir = os.path.join("ffn_saved_models/", args.model_name)

    for pkl_path in sorted(folder.glob("*.pkl")):
        with open(pkl_path, "rb") as f:
            hid_act = pd.read_pickle(f)
            train_df, test_df = split_audio_files_by_filename(data=hid_act, test_size=.1)
            train_loader, test_loader = ffn_data_normalization_and_formatting(train_df=train_df, test_df=test_df)
            trainAcc,trainEvalAcc, testAcc,losses,net = function2trainTheModel(train_loader=train_loader, test_loader=test_loader)
            save_dir = os.path.join(model_ffn_dir, pkl_path + ".pth")


            torch.save(net.state_dict(), save_dir)





