"""Phone/vowel constants and acoustic annotation: formants, transcripts, row filtering."""


import os
import re
import glob
import pickle
import numpy as np
import pandas as pd
import parselmouth
from praatio import textgrid
from sklearn.preprocessing import StandardScaler
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


def label_at_time(time, entries):
    for start, end, label, rep in entries:
        if start <= time <= end:
            return label, rep


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
