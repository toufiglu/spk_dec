"""Speaker-specificity index: KL-divergence ratio between model and acoustic vowel distributions."""


import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.base import clone

from acoustics import filter_out_irrelevant_rows
from activations import read_activation_and_left_join_with_acoustics


def kld_calculate(dist1, dist2):
    """
    KL divergence between two multivariate normals.
    dist1: model mvg distribution,
    dist2: acoustic mvg distribution
    """
    d = dist1.mean.shape[0]
    try:
        kld=0.5 * (
        np.trace(np.linalg.inv(dist2.cov) @ dist1.cov)
        + (dist2.mean - dist1.mean).T @ np.linalg.inv(dist2.cov) @ (dist2.mean - dist1.mean)
        - d
        + np.log(np.linalg.det(dist2.cov) / np.linalg.det(dist1.cov))
    )
        return kld
    except np.linalg.LinAlgError:
        return np.nan


def speaker_specificity_kld(kld_model_marginal, kld_model_speaker):
    if (
        kld_model_marginal is None or kld_model_speaker is None or
        np.isnan(kld_model_marginal) or np.isnan(kld_model_speaker)
    ):
        return np.nan
    else:
        index = kld_model_speaker / (kld_model_marginal  + kld_model_speaker) 
        return index


def calculate_speaker_marginal_kld(df):
    # get speaker specific distr
    speaker_spec_results = []
    for (speaker, phone), group in df.groupby(["speaker", "phone"]):
        acous = group[["pca_f1", "pca_f2", "pca_f3"]].to_numpy()
        model = group[["pca_1", "pca_2", "pca_3"]].to_numpy()

        acous_mean = acous.mean(axis=0)
        acous_var = np.cov(acous, rowvar=False)
        acous_mvg = multivariate_normal(mean=acous_mean, cov=acous_var)

        model_mean = model.mean(axis=0)
        model_var = np.cov(model, rowvar=False)
        model_mvg = multivariate_normal(mean=model_mean, cov=model_var)

        speaker_spec_results.append(
            {
                "speaker": speaker,
                "phone": phone,
                "acoustic_mvg": acous_mvg,
                "model_mvg": model_mvg
            }
        )

    speaker_spec_results = pd.DataFrame.from_dict(speaker_spec_results)

    # get marginal distr
    marginal_results = []
    for phone, group in df.groupby(["phone"]):
        acous = group[["pca_f1", "pca_f2", "pca_f3"]].to_numpy()
        model = group[["pca_1", "pca_2", "pca_3"]].to_numpy()

        acous_mean = acous.mean(axis=0)
        acous_var = np.cov(acous, rowvar=False)
        acous_mvg = multivariate_normal(mean=acous_mean, cov=acous_var)

        model_mean = model.mean(axis=0)
        model_var = np.cov(model, rowvar=False)
        model_mvg = multivariate_normal(mean=model_mean, cov=model_var)

        marginal_results.append(
            {
                "speaker": speaker,
                "phone": phone,
                "acoustic_mvg": acous_mvg,
                "model_mvg": model_mvg
            }
        )
    marginal_results = pd.DataFrame.from_dict(marginal_results).drop(columns="speaker")
    marginal_results["phone"] = marginal_results["phone"].apply(lambda x: x[0])

    phones = marginal_results['phone'].to_list()
    speakers = speaker_spec_results['speaker'].unique().tolist()

    spec_idx = {}
    for phone in phones:
        acous_marg = marginal_results.loc[marginal_results['phone']==phone, "acoustic_mvg"].iloc[0]
        phone_spec_idx = {}
        for speaker in speakers:
            mask = ( # if the speaker never said that phone (but is this really possible?)
                (speaker_spec_results['phone'] == phone) &
                (speaker_spec_results['speaker'] == speaker)
                )
            subset = speaker_spec_results.loc[mask, ["model_mvg", "acoustic_mvg"]]

            if subset.empty:
                # no matching rows for this phone–speaker pair
                phone_spec_idx[speaker] = np.nan
                continue
            model_spec = subset["model_mvg"].iloc[0]
            acous_spec = subset["acoustic_mvg"].iloc[0]

            if model_spec is None or acous_spec is None:
                phone_spec_idx[speaker] = np.nan
                continue

            kld_model_marginal = kld_calculate(model_spec, acous_marg)
            kld_model_spec = kld_calculate(model_spec, acous_spec)

            specificity_index = speaker_specificity_kld(kld_model_marginal=kld_model_marginal, kld_model_speaker=kld_model_spec)
            phone_spec_idx[speaker] = specificity_index

        spec_idx[phone] = phone_spec_idx

    return spec_idx


def lda_reduction(X, y, lda, cv):
    n = X.shape[0]
    z = None
    splits = cv.split(X, y)
    
    for idx, (train, test) in enumerate(splits, start=1):
        pipe = make_pipeline(StandardScaler(), clone(lda))
        pipe.fit(X[train], y[train])

        z_test = pipe.transform(X[test])
        
        if z is None:
            z = np.empty((n, z_test.shape[1]), dtype=z_test.dtype)
        
        z[test] = z_test
        print(f'the {idx} fold is done')
    
    return z


def check_folder_files_exist(folder, layers): # the file checking is more complex than I thought. safer to write a function
    if not folder.is_dir():
        return False

    expected = [folder / f"layer_{layer}_spec_idx.csv" for layer in layers] # all files we should have in that folder
    return all(f.exists() for f in expected)


def create_lda_from_hidden_acts(df, hid_act_col, lda_act_filename):
    '''
    create 3 LDA components that maximally discriminate phones from NN hidden activation
    '''

    lda = LinearDiscriminantAnalysis(n_components=3) # reduce it to 3d, compatile with formants
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) # stratified 5-fold, as some vowels may have more examples than the others.

    X = np.stack(df[hid_act_col].to_numpy()) # pandas-series -> np array 1d -> stack, so that it becomes a 2d array, compatible with sklearn input requirement
    y = df['phone']

    le = LabelEncoder()
    y = le.fit_transform(y) # the phones were symbols (phonemes). Recode them into numbers.

    lda_X = lda_reduction(X=X, y=y, lda=lda, cv=cv)
    lda_cols = [f"lda_{i+1}" for i in range(lda_X.shape[1])] # create a bunch of colnames for the 3d vector output
    df[lda_cols] = lda_X
    df.to_pickle(lda_act_filename)

    return df


def create_pca_components(df, lda_components):
    '''
    project lda and formants to a 3d space, where each dimension reflects distribution of variance, making sure that the two components are comparible
    '''
    Missing = [c for c in lda_components if c not in df.columns]
    if Missing:
        raise KeyError(f"Missing columns: {Missing}")
    
    X = df[lda_components].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)

    pca_cols = [f"pca_{i+1}" for i in range(X_pca.shape[1])]
    df[pca_cols] = X_pca

    return df


def create_or_load_acts_with_lda(lda_act_filename, hid_act_folder, layer, num_layers, formant_df):
    '''
    lda projection to 3d for a 768d hid activation
    '''
    if lda_act_filename.exists():
        act_filtered = pd.read_pickle(lda_act_filename)

    else:
        acts = read_activation_and_left_join_with_acoustics(
            hid_act_folder=hid_act_folder,
            hid_pkl_folder="wav_2_vec_hidden_pkl",
            num_layers=num_layers,
            formant_df=formant_df,
            which_layer=layer,
            dont_read_again=True
            )

        act_filtered = filter_out_irrelevant_rows(acts)

        act_filtered = create_lda_from_hidden_acts(df=act_filtered, hid_act_col="act", lda_act_filename=lda_act_filename)

    return act_filtered


def get_spec_idx_for_every_layer(layers, hid_act_folder, formant_df, path, dont_run_again=True): # re-use existing is a toggle: if T we don't re-compute
    '''
    return a dictionary of dfs, one for each layer

    Questions:
    1. lda is initiated once for each layer; random number generators are initiated once for each layer.
    2. we need to make this function fit more general purposes; currently, it only works for the 13 hidden-layer wav-2-vec model
    '''
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
            # read out the hid act for that layer first
            filtered_act_path = Path("./wav_2_vec_hidden_lda")
            filtered_act_path.mkdir(parents=True, exist_ok=True)
            lda_act_filename = filtered_act_path / f"lda_acts_{layer}.pkl"

            act_filtered = create_or_load_acts_with_lda(lda_act_filename=lda_act_filename, hid_act_folder=hid_act_folder, layer=layer, num_layers=len(layers), formant_df=formant_df)
            act_filtered = create_pca_components(df=act_filtered, lda_components=["lda_1", "lda_2", "lda_3"])
            
            # merge formant_df cols with the act
            formant_df_cols = formant_df[["time", "filename", "pca_f1", "pca_f2", "pca_f3"]] # but I need to make sure pca is done for the formant_df
            act_filtered = pd.merge(act_filtered, formant_df_cols, on=["time", "filename"])

            # average across vowels
            act_vowel_mean_pooled = (act_filtered
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
