"""Gammatone-filterbank helpers (needs eelbrain, conda-only). Exploratory."""


import os
import glob
import numpy as np
import pandas as pd
import soundfile as sf
# import eelbrain  # required at runtime; conda-only
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

from acoustics import read_transcript_map
from spec_idx import lda_reduction


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
