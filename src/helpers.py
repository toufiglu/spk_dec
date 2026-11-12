# eelbrain requires building a conda environment; making it difficult to replicate on other people's computer
# import eelbrain
import os
import pandas as pd
import glob
import soundfile as sf
import torch
import numpy as np
import h5py
import pickle
import re 
import parselmouth
import matplotlib.pyplot as plt
import seaborn as sns
import pyphen


from praatio import textgridf
from praatio.audio import getDuration
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedGroupKFold, cross_val_predict, cross_val_score, train_test_split, GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from sklearn.base import clone
from transformers import Wav2Vec2Model, Wav2Vec2Processor
from joblib import Parallel, delayed
from scipy.stats import multivariate_normal
from sklearn.decomposition import PCA


arpabets = [
    "AA", "AA0", "AA1", "AA2",
    "AE", "AE0", "AE1", "AE2",
    "AH", "AH0", "AH1", "AH2",
    "AO", "AO0", "AO1", "AO2",
    "AW", "AW0", "AW1", "AW2",
    "AY", "AY0", "AY1", "AY2",
    "EH", "EH0", "EH1", "EH2",
    "ER", "ER0", "ER1", "ER2",
    "EY", "EY0", "EY1", "EY2",
    "IH", "IH0", "IH1", "IH2",
    "IY", "IY0", "IY1", "IY2",
    "OW", "OW0", "OW1", "OW2",
    "OY", "OY0", "OY1", "OY2",
    "UH", "UH0", "UH1", "UH2",
    "UW", "UW0", "UW1", "UW2",
]

arpabet_vowels = list(dict.fromkeys([re.sub(r"\d+", "", p) for p in arpabets]))

diphthongs = ["AW", "AY", "EY", "OW", "OY"]

target_vowels = [v for v in arpabet_vowels if v not in diphthongs]


def get_waveform_and_gammatone(ROOT, dont_run_again=True): 
    speakers = [f for f in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, f))]
    if dont_run_again and os.path.exists("./waveform_gammatone.pkl"):
        audio_df = pd.read_pickle("waveform_gammatone.pkl")
    else:
        audio_df_list = []
        for idx, speaker in enumerate(speakers, start=1):
            speaker_dir = os.path.join(ROOT, speaker)
            if not os.path.isdir(speaker_dir):
                continue

            for rec_folder in os.listdir(speaker_dir):
                rec_dir = os.path.join(speaker_dir, rec_folder)
                if not os.path.isdir(rec_dir):
                    continue

                trans_map = read_transcript_map(rec_dir)
                
                trans_map = pd.DataFrame(
                    [{"FILENAME": k, "SENTENCE": v} for k, v in trans_map.items()],
                    columns=["FILENAME", "SENTENCE"]
                )

                wav_paths = glob.glob(os.path.join(rec_dir, "*.wav"))

                tmp = []
                for wav in wav_paths:
                    FILENAME = wav.split("/")[-1].split(".")[0]
                    AUDIO, sampling_rate = sf.read(wav, dtype="float32") # read the sound file using sf, float32
                    waveform = eelbrain.load.wav(wav) # read soundfile using eelbrain, then turn it into a gammatone filterbank
                    gt = eelbrain.gammatone_bank(waveform, f_min = 50, f_max = 15000, n = 128, location='left', tstep=0.02, integration_window = 0.01)
                    tmp.append({'FILENAME': FILENAME, "AUDIO": AUDIO, "gammatone": gt.x})

                tmp=pd.DataFrame(tmp)

                audio_df = pd.merge(trans_map, tmp, how="left", on="FILENAME")
                audio_df_list.append(audio_df)
            print(f"Speaker {speaker} is done. This is {idx} out of {len(speakers)} speakers.")

        audio_df = pd.concat(audio_df_list, ignore_index=True)
        audio_df.to_pickle("waveform_gammatone.pkl")

    return audio_df


# make this a function: read from all activation folders
def read_activation_and_left_join_with_acoustics(hid_act_folder, hid_pkl_folder, num_layers, which_layer, formant_df, dont_read_again):
    os.makedirs(hid_pkl_folder, exist_ok=True)

    expected_pkls = [os.path.join(hid_pkl_folder, f"hidden_acts_{layer}.pkl") for layer in range(1, num_layers + 1)]
    if all(os.path.exists(pkl) for pkl in expected_pkls) and dont_read_again:
        print("All layer pickle files found, will load the target layer.")
        acts = pd.read_pickle(f"./{hid_pkl_folder}/hidden_acts_{which_layer}.pkl")
        return acts

    else:
        formant_idx = formant_df.set_index(["filename", "time"]) # vowel formant df is left-joined to the hidden activation df

        for layer in range(1, num_layers + 1): 
            layer_pkl = os.path.join(hid_pkl_folder, f"hidden_acts_{layer}.pkl")
            if os.path.exists(layer_pkl) and dont_read_again:
                print(f"Layer {layer} already exists — skipping.")
                continue
            else: 
                if layer < 10:
                    folder = hid_act_folder + "/layer_0" + str(layer) # naming of the folder requires, not hugely important though
                else:
                    folder = hid_act_folder + "/layer_" + str(layer)
                
                files = sorted(f for f in os.listdir(folder) if f.endswith(".pt"))


                all_files_act_df = []
                for idx, file in enumerate(files):
                    rows = []
                    file_path = os.path.join(folder, file)
                    filename = os.path.splitext(file)[0]
                    file_act = torch.load(file_path, map_location="cpu")

                    T, D = file_act.shape
                    times = np.arange(T, dtype=float)*0.02 # every 20 ms get a frame so 0.02
                    act = file_act.cpu().numpy()
                    
                    for t in range(T):
                        rows.append(
                            {
                                'filename': filename,
                                'time': times[t],
                                'act': act[t]
                            }
                        )

                    
                    file_act_df = pd.DataFrame(rows)
                    file_act_with_phone = file_act_df.join(formant_idx, on=["filename", "time"], how='left')


                    all_files_act_df.append(file_act_with_phone)

                    print(f"Layer {layer}: file {file} is done. This is the {idx} out of 2702 files.")
                
                hid_act_df = pd.concat(all_files_act_df, ignore_index=True)
                hid_act_df.to_pickle(f"./{hid_pkl_folder}/hidden_acts_{layer}.pkl")

                print(f"Layer {layer} is done")

        acts = pd.read_pickle(f"./{hid_pkl_folder}/hidden_acts_{which_layer}.pkl")

    return acts

def gammatone_long(df_with_gammatone, formant_df):
    rows_long = []

    formant_idx = formant_df.set_index(["filename", "time"]) # vowel formant df is left-joined to gammatone, we might need this later

    for _, row in df_with_gammatone.iterrows():
        filename = row['FILENAME']
        gt = row['gammatone']

        D, T = gt.shape
        times = np.arange(T, dtype=float) * 0.02

        for t in range(T):
            rows_long.append({
                'filename': filename,
                'time': times[t],
                'gammatone': gt[:, t]
            })
        
    rows_long_df = pd.DataFrame(rows_long)
    rows_long_with_phone = rows_long_df.join(formant_idx, on=["filename", "time"], how='left')

    return rows_long_with_phone


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


def filter_out_irrelevant_rows(df):
    filtered = df[
        df["phone"].isin(arpabet_vowels) 
        & ~df["phone"].isin(diphthongs)
        & df["f1"].notna()
        & df["f2"].notna()
        & df["f3"].notna()
        & df["phone"].notna()
        & (df["phone"].str.strip() != "")
    ].copy()

    return filtered



def decode_talker_from_one_layer(layer, hid_act_folder, hid_pkl_folder, num_layers, formant_df, dont_run_again, dont_read_again):
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

    best_C, C_auc_scores = talker_find_best_C(data=test_df, source=f"model_{layer:02d}", dont_run_again=dont_run_again)

    preds, auc = talker_log_reg(act_layer, best_C, source=f"layer_{layer:02d}", dont_run_again=dont_run_again)

    return {
        "layer": layer,
        "best_C": best_C,
        "auc_weighted": auc
    }



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

    expected = [folder / f"layer_{layers}_spec_idx.csv" for layer in layers] # all files we should have in that folder
    return all(f.exists() for f in expected)


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

            act_filtered = create_or_load_acts_with_lda(lda_act_filename=lda_act_filename, hid_act_folder=hid_act_folder, formant_df=formant_df, layers=layers)
            act_filtered = create_pca_components(df=act_filtered, lda_components=["lda_1, lda_2, lda_3"])
            
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
    


def label_at_time(time, entries):
    for start, end, label, rep in entries:
        if start <= time <= end:
            return label, rep

# get formant from recordings: should put this in another file
def get_formant(sound_folder, formant_fname):
    f_path_name = f"./{formant_fname}"
    if os.path.exists(f_path_name):
        with open(f_path_name, "rb") as f:
            formant_df=pickle.load(f) 
    else:  
        entire_df = [] 
        for speaker in os.listdir(sound_folder):
            speaker_dir = os.path.join(sound_folder, speaker)
            
            if not os.path.isdir(speaker_dir):
                continue
            for rec_folder in os.listdir(speaker_dir):
                rec_dir = os.path.join(speaker_dir, rec_folder)
                if not os.path.isdir(rec_dir):
                    continue
                
                wav_paths = glob.glob(os.path.join(rec_dir, "*.wav"))
                for wav in wav_paths:
                    # get phone tier from the textgrid
                    txtgrid_loc = wav.replace("corpus_before_align", "my_corpus_aligned").replace(".wav", ".TextGrid") # get location of txtgrid file
                    # replace the first strings with the second

                    txtgrid = textgrid.openTextgrid(txtgrid_loc, includeEmptyIntervals=True)
                    phone_tier = txtgrid.getTier(tierName="phones")

                    entries = phone_tier.entries
                    counts={}
                    intervals = []
                    for start, end, lab in entries:
                        if lab=="" or lab is None:
                            rep=0
                        else:
                            rep=counts.get(lab, 0) + 1
                            counts[lab] = rep
                        intervals.append((start, end, lab, rep))

                    # get speaker and filename, later as columns
                    filename = txtgrid_loc.split(".")[1].split("/")[-1]
                    speaker = txtgrid_loc.split(".")[1].split("/")[3]

                    # get formants
                    audio = parselmouth.Sound(wav)
                    formant = audio.to_formant_burg(time_step=0.02, # in keeping with the sampling rate of the ssl model, we also used 20 ms here.
                                        max_number_of_formants=5, # select 5 formants but later use only the first 3.
                                        maximum_formant=5500, # female max formant is suited to be 5500, we keep it this way for all speakers
                                        window_length=0.025) 
                    n_frame = formant.get_number_of_frames()
                    # make a tuple that store all info
                    row = []
                    t0 = formant.get_start_time()
                    dt = formant.get_time_step()
                    times = t0 + dt*np.arange(n_frame)
                    for time in times: 
                        f1 = formant.get_value_at_time(1, time)
                        f2 = formant.get_value_at_time(2, time)
                        f3 = formant.get_value_at_time(3, time)
                        phone, rep = label_at_time(time, intervals)
                        row.append((time, f1, f2, f3, phone, rep, speaker, filename))
                    df = pd.DataFrame(row, columns=["time", "f1", "f2", "f3", "phone", "rep", "speaker", "filename"])
                    entire_df.append(df)
                print(f"The folder of {rec_dir} is sorted.")
        formant_df = pd.concat(entire_df, ignore_index=True) 
        formant_df.to_pickle("formant.pkl")

    return formant_df


def lda_gammatone(gt_df):
    lda = LinearDiscriminantAnalysis(n_components=3) # reduce it to 3d, compatile with formants
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) # stratified 5-fold, as some vowels may have more examples than the others.

    X = np.stack(gt_df['gammatone'].to_numpy()) # pandas-series -> np array 1d -> stack, so that it becomes a 2d array, compatible with sklearn input requirement
    y = gt_df['phone']

    le = LabelEncoder()
    y = le.fit_transform(y) # the phones were symbols (phonemes). Recode them into numbers.

    lda_X = lda_reduction(X=X, y=y, lda=lda, cv=cv)
    lda_cols = [f"lda_{i+1}" for i in range(lda_X.shape[1])] # create a bunch of colnames for the 3d vector output
    gt_df[lda_cols] = lda_X

    gt_df.to_pickle("waveform_gammatone_with_lda_components.pkl")
    return gt_df

# below were the codes written to create 3d visualizations for the pcas of the formants and the lda-ed hidden activations. they worked but to see clearly, we will need to rotate the figures individually.

def plot_gaussian_ellipsoid(mean, cov, ax, n_std=2.0, **kwargs):
    # n_std: 1≈68%, 2≈95%
    vals, vecs = np.linalg.eigh(cov)
    radii = n_std * np.sqrt(vals)

    # unit sphere: Yiming-learn what this is doing
    u = np.linspace(0, 2*np.pi, 40)
    v = np.linspace(0, np.pi, 20)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    sphere = np.stack([x, y, z], axis=-1)                        # (40,20,3)

    A = vecs @ np.diag(radii)                                    # (3,3)
    E = np.einsum('ij,uvj->uvi', A, sphere) + mean               # (40,20,3)

    ax.plot_surface(E[:,:,0], E[:,:,1], E[:,:,2],
                    rstride=1, cstride=1, linewidth=0, shade=False, **kwargs)



def visualization_3d_spec_distributions(num_layer, formant_df):
    speaker_specs = {}
    marginals = {}
    for layer in range(1, num_layer+1): 
        # read out the hid act for that layer first
        filtered_act_path = Path("./wav_2_vec_hidden_lda")
        filtered_act_path.mkdir(parents=True, exist_ok=True)
        lda_act_filename = filtered_act_path / f"lda_acts_{layer}.pkl"

        act_filtered = pd.read_pickle(lda_act_filename)

        X = act_filtered[["lda_1", "lda_2", "lda_3"]]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pca = PCA(n_components=3)
        X_pca = pca.fit_transform(X_scaled)

        pca_cols = [f"pca_{i+1}" for i in range(X_pca.shape[1])]
        act_filtered[pca_cols] = X_pca

        missing = [c for c in pca_cols if c not in act_filtered.columns]
        if missing:
            raise RuntimeError(f"Missing PCA cols {missing} for layer {layer}. Recompute or check LDA step.")
        
        formant_df_cols = formant_df[["time", "filename", "pca_f1", "pca_f2", "pca_f3"]]
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
            # get speaker specific distr
        speaker_spec_results = []
        for (speaker, phone), group in act_filtered.groupby(["speaker", "phone"]):
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
        speaker_specs[layer] = speaker_spec_results

        # get marginal distr
        marginal_results = []
        for phone, group in act_filtered.groupby(["phone"]):
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

        marginals[layer] = marginal_results
    return speaker_specs, marginals

def mean_line_and_points(data, **kwargs):
    m = data.groupby("layer", as_index=False)["r"].mean()
    sns.lineplot(data=m, x="layer", y="r", lw=1, **kwargs)
    sns.scatterplot(data=m, x="layer", y="r", s=30, **kwargs)


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

def split_audio_files_by_filename(data, test_size):
    '''
    stratified by this variable, and split all sound files, so that the one appearing in training won't appear in testing.
    '''
    groups = data["filename"]
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    train_idx, test_idx = next(gss.split(data, groups=groups))
    return (data.iloc[train_idx].reset_index(drop=True),
            data.iloc[test_idx].reset_index(drop=True))  

    
def create_lda_from_hidden_acts(layer, df, hid_act_col, lda_act_filename):
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
    lda_cols = [f"{layer}_lda_{i+1}" for i in range(lda_X.shape[1])] # create a bunch of colnames for the 3d vector output
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


def create_or_load_acts_with_lda(lda_act_filename, hid_act_folder, layers, formant_df):
    '''
    lda projection to 3d for a 768d hid activation
    '''
    if lda_act_filename.exists():
        act_filtered = pd.read_pickle(lda_act_filename)

    else:
        acts = read_activation_and_left_join_with_acoustics(
            hid_act_folder=hid_act_folder,
            hid_pkl_folder="wav_2_vec_hidden_pkl",
            num_layers=len(layers),
            formant_df=formant_df,
            which_layer=layer
            )

        act_filtered = filter_out_irrelevant_rows(acts)

        act_filtered = create_lda_from_hidden_acts(df=act_filtered, hid_act_col="act", lda_act_filename=lda_act_filename)

    return act_filtered

def get_formant_with_pca_cols():
    mfa_folder = "./mfa_align/"
    sound_folder = mfa_folder + "corpus_before_align/"
    formant_df = get_formant(sound_folder=sound_folder)
    formant_df["phone"] = formant_df["phone"].str.replace(r"\d+", "", regex=True)
    formant_df = filter_out_irrelevant_rows(formant_df)

    X = formant_df[["f1", "f2", "f3"]]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)

    pca_cols = [f"pca_f{i+1}" for i in range(X_pca.shape[1])] # create a bunch of colnames for the 3d vector output
    formant_df[pca_cols] = X_pca
    
    return formant_df


# some codes for cka: taken from https://cka-similarity.github.io/


# logistic regression without cross validation, get coefficients for talker and phone decoders, and calculate cosine similarity


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


def read_transcript_map(rec_dir):
    """
    Extract raw audio waveform and transcription

    Args:
    rec_dir: recording directory
    
    Explainations:
    LibriSpeech chapter file has a txt file that contains the transcription. We are gonna get it and write onto the textgrid file later.

    Returns:
        a dataframe with filename (speaker, recording index), sentence transcription, and waveform.
    """
    # Find the transcript file if present
    txts = glob.glob(os.path.join(rec_dir, "*.trans.txt"))
    
    if not txts:
        return pd.DataFrame(columns=["FILENAME", "SENTENCE"])
    
    txt_path = txts[0]
    rows = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                rid, sent = parts
            else:
                rid, sent = parts[0], ""
            rows.append({"FILENAME": rid, "SENTENCE": sent})
    return pd.DataFrame(rows)