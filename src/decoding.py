"""Linear-decoding utilities: splits, C-search, talker/phone logistic regression."""


import os
import numpy as np
import pandas as pd
import h5py
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (StratifiedKFold, StratifiedGroupKFold, GroupShuffleSplit,
                                     cross_val_predict, cross_val_score, train_test_split)
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

from acoustics import filter_out_irrelevant_rows
from activations import read_activation_and_left_join_with_acoustics


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


def split_audio_files_stratified(data, test_size, group_by):
    '''
    stratified by this variable, and split all sound files, so that the one appearing in training won't appear in testing.
    '''
    train_audios = []
    test_audios = []
    
    for group, othercols in data.groupby(group_by): 
        filenames = othercols["filename"].unique()
        train_files, test_files = train_test_split(filenames, test_size = test_size, random_state=42)
        train_audios.extend(train_files)
        test_audios.extend(test_files)

    train_df = data[data["filename"].isin(train_audios)].reset_index(drop=True)
    test_df  = data[data["filename"].isin(test_audios)].reset_index(drop=True)    

    return train_df, test_df


def split_audio_files_by_filename(data, test_size):
    '''
    stratified by this variable, and split all sound files, so that the one appearing in training won't appear in testing.
    '''
    groups = data["filename"]
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    train_idx, test_idx = next(gss.split(data, groups=groups))
    return (data.iloc[train_idx].reset_index(drop=True),
            data.iloc[test_idx].reset_index(drop=True))


def talker_find_best_C(path, data, source, dont_run_again=True): # what if the folder doesn't exist? should makedirs
    
    os.makedirs(path, exist_ok=True)
    
    filepath = os.path.join(path, f"best_C_{source}.csv")

    if dont_run_again and os.path.exists(filepath):
        C_auc_scores = pd.read_csv(filepath)
    else:
        C_values = [0.001, 0.01, 0.1, 1, 10, 100]
        
        X=np.stack(data["act"].to_numpy())
        y=data["speaker"].to_numpy() 

        assert len(X)==len(y)
        le = LabelEncoder()
        y = le.fit_transform(y) 


        results = Parallel(n_jobs=7, verbose=10)(
            delayed(multiprocess_C)(X, y, C)
            for C in C_values
        )

        C_auc_scores = pd.DataFrame(results, columns=["C", "mean_auc", "std_auc"]).sort_values("mean_auc", ascending=False)

        C_auc_scores.to_csv(filepath, index=False)

    best_C_row = C_auc_scores.loc[C_auc_scores["mean_auc"].idxmax()]
    best_C = float(best_C_row["C"])
    print(f"The best parameter for {source} was {best_C}")
    return best_C, C_auc_scores


def talker_log_reg(data, best_C, source, path, dont_run_again=True):
    file = os.path.join(path, f"{source}_pred_prob.h5")
    if dont_run_again and os.path.isfile(file):
        with h5py.File(file, "r") as f:
            preds = f["predictions"][:]
            
        y = data["speaker"].to_numpy()  
        le = LabelEncoder()
        y_enc = le.fit_transform(y)   # otherwise y will be missing          

    else:
        X = np.stack(data["act"].to_numpy())         # shape (N, D)
        y = data["speaker"].to_numpy()               # shape (N,) 

        le = LabelEncoder()
        y_enc = le.fit_transform(y)

        logreg = LogisticRegression(
            random_state=16, penalty="l2", solver="lbfgs", C=best_C, max_iter=10000
        )

        model = make_pipeline(StandardScaler(), logreg)
        groups = data["filename"]
        cv = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)

        preds = cross_val_predict(model, X, y_enc, cv=cv, method="predict_proba", n_jobs=7, groups=groups) # this is the same as the manual function below

        os.makedirs(os.path.dirname(file), exist_ok=True)
        with h5py.File(file, "w") as f:
            f.create_dataset("predictions", data=preds) # save it, so that we can calculate Bayes factors later
    
    auc_weighted = roc_auc_score(y_enc, preds, multi_class="ovr", average="weighted")

    return preds, float(auc_weighted)


def decode_talker_from_one_layer(layer, hid_act_folder, hid_pkl_folder, num_layers, formant_df, path, dont_run_again, dont_read_again):
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

    act_layer = filter_out_irrelevant_rows(act_layer)

    train_df, test_df = split_audio_files_stratified(data=act_layer, test_size=0.1, group_by="speaker") # take 10% of the data to get best C; the naming of the var test_df really isn't great

    best_C, C_auc_scores = talker_find_best_C(path=path, data=test_df, source=f"model_{layer:02d}", dont_run_again=dont_run_again)

    preds, auc = talker_log_reg(act_layer, best_C, source=f"layer_{layer:02d}", path=path, dont_run_again=dont_run_again)

    return {
        "layer": layer,
        "best_C": best_C,
        "auc_weighted": auc
    }


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

    act_layer = filter_out_irrelevant_rows(act_layer)

    train_df, test_df = split_audio_files_by_filename(data=act_layer, test_size=0.1) # take 10% of the data to get best C; the naming of the var test_df really isn't great

    best_C, C_auc_scores = phone_find_best_C(path=path, data=test_df, source=f"model_{layer:02d}", dont_run_again=dont_run_again)

    preds, auc = phone_log_reg(act_layer, best_C, path=path, source=f"layer_{layer:02d}", dont_run_again=dont_run_again)

    return {
        "layer": layer,
        "best_C": best_C,
        "auc_weighted": auc
    }


def talker_logreg_coef(data, best_C, coef_file_path, source, dont_run_again=True):
    
    if dont_run_again and os.path.isfile(coef_file_path):  
        return pd.read_csv(coef_file_path).to_numpy()      
    
    else:
        X = np.stack(data["act"].to_numpy())         # shape (N, D)
        y = data["speaker"].to_numpy()               # shape (N,) 

        le = LabelEncoder()
        y_enc = le.fit_transform(y)

        logreg = LogisticRegression(
            random_state=16, penalty="l2", solver="lbfgs", C=best_C, max_iter=10000
        )

        model = make_pipeline(StandardScaler(), logreg)
        
        model.fit(X, y_enc)

        coef = model.named_steps["logisticregression"].coef_

        os.makedirs(os.path.dirname(coef_file_path), exist_ok=True)
        pd.DataFrame(coef).to_csv(coef_file_path, index=False)

    
    return coef


def phone_logreg_coef(data, best_C, coef_file_path, source, dont_run_again=True):
    if dont_run_again and os.path.isfile(coef_file_path):  
        return pd.read_csv(coef_file_path).to_numpy()      
    else:
        X = np.stack(data["act"].to_numpy())         # shape (N, D)
        y = data["phone"].to_numpy()               # shape (N,) 

        le = LabelEncoder()
        y_enc = le.fit_transform(y)

        logreg = LogisticRegression(
            random_state=16, penalty="l2", solver="lbfgs", C=best_C, max_iter=10000
        )

        model = make_pipeline(StandardScaler(), logreg)
        
        model.fit(X, y_enc)

        coef = model.named_steps["logisticregression"].coef_

        os.makedirs(os.path.dirname(coef_file_path), exist_ok=True)
        pd.DataFrame(coef).to_csv(coef_file_path, index=False)

    
    return coef


def decode_talker_COEF_from_one_layer(layer, hid_act_folder, hid_pkl_folder, num_layers, formant_df, dont_run_again, dont_read_again, storage):
    act_layer = read_activation_and_left_join_with_acoustics(
        hid_act_folder=hid_act_folder,
        hid_pkl_folder=hid_pkl_folder,
        num_layers=num_layers,
        formant_df=formant_df,
        which_layer=layer,
        dont_read_again=dont_read_again
    )

    act_layer = filter_out_irrelevant_rows(act_layer)
    train_df, test_df = split_audio_files_stratified(data=act_layer, test_size=0.1, group_by="speaker")
    best_C, C_auc_scores = talker_find_best_C(data=test_df, source=f"model_{layer:02d}", path=storage, dont_run_again=dont_run_again)
    coef = talker_logreg_coef(act_layer, best_C, coef_file_path=f"{storage}/layer_{layer:02d}_coef.csv", source=f"layer_{layer:02d}", dont_run_again=dont_run_again)

    return {
        "layer": layer,
        "coef": coef
    }


def decode_phone_COEF_from_one_layer(layer, hid_act_folder, hid_pkl_folder, num_layers, formant_df, dont_run_again, dont_read_again, storage):
    act_layer = read_activation_and_left_join_with_acoustics(
        hid_act_folder=hid_act_folder,
        hid_pkl_folder=hid_pkl_folder,
        num_layers=num_layers,
        formant_df=formant_df,
        which_layer=layer,
        dont_read_again=dont_read_again
    )

    act_layer = filter_out_irrelevant_rows(act_layer)
    train_df, test_df = split_audio_files_stratified(data=act_layer, test_size=0.1, group_by="speaker")
    best_C, C_auc_scores = phone_find_best_C(path=storage, data=test_df, source=f"model_{layer:02d}", dont_run_again=dont_run_again)
    coef = phone_logreg_coef(act_layer, best_C, coef_file_path=f"{storage}/layer_{layer:02d}_coef.csv", source=f"layer_{layer:02d}", dont_run_again=dont_run_again)

    return {
        "layer": layer,
        "coef": coef
    }
