import importlib
import helpers
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from praatio import textgrid
import glob
import soundfile as sf
import praatio.audio
import re
import parselmouth 
importlib.reload(helpers)
        

vowels = helpers.target_vowels

ROOT = "./mfa_align"
path_audios = os.path.join(ROOT, "corpus_before_align")
path_tg = os.path.join(ROOT, "my_corpus_aligned")
row = []
for speaker in os.listdir(path_audios):
    speaker_dir = os.path.join(path_audios, speaker)
    if not os.path.isdir(speaker_dir):
        continue

    for rec_folder in os.listdir(speaker_dir):
        rec_dir = os.path.join(speaker_dir, rec_folder)
        if not os.path.isdir(rec_dir):
            continue

        wav_paths = glob.glob(os.path.join(rec_dir, "*.wav"))
        tg_paths = [wav.replace("corpus_before_align", "my_corpus_aligned").replace(".wav", ".TextGrid") for wav in wav_paths]
        

        for wav_i, wav in enumerate(wav_paths):
            # get formants first
            audio = parselmouth.Sound(wav)
            formant = audio.to_formant_burg(time_step=0.02, # in keeping with the sampling rate of the ssl model, we also used 20 ms here.
                                max_number_of_formants=5, # select 5 formants but later use only the first 3.
                                maximum_formant=5500, # female max formant is suited to be 5500, we keep it this way for all speakers
                                window_length=0.025)
            f1_values = []
            f2_values = []
            f3_values = []
            n_frames = formant.get_number_of_frames()
            for frame_i in range(1, n_frames + 1):  
                t = formant.get_time_from_frame_number(frame_i)
                f1 = formant.get_value_at_time(1, t)
                f2 = formant.get_value_at_time(2, t)
                f3 = formant.get_value_at_time(3, t)
                f1_values.append(f1)
                f2_values.append(f2)
                f3_values.append(f3)

            # then do clipping
            audio, sr = sf.read(wav)
            tg_path = tg_paths[wav_i]
            tg = textgrid.openTextgrid(tg_path, includeEmptyIntervals=True)
            phone_tiers = tg._tierDict["phones"].entries

            vowel_counter = {} 

            filename = wav.split("/")[-1].split(".")[0]

            for i, phone_tier in enumerate(phone_tiers):
                phone = re.sub(r"\d+", "", phone_tier[2])

                if i==0:
                    prev_phone = None
                else:
                    prev_phone_raw = phone_tiers[i-1][2] or ""
                    prev_phone = re.sub(r"\d+", "", prev_phone_raw).strip()
                    if prev_phone == "":
                        prev_phone = None
                
                if prev_phone is None:
                    continue

                rep = vowel_counter.get(phone, 0) + 1
                vowel_counter[phone] = rep

                if phone in vowels:
                    vowel_t_start = phone_tier[0]
                    vowel_t_end = phone_tier[1]
                    dur = (vowel_t_end - vowel_t_start)*1000

                    if dur < 40:
                        continue
                    
                    t_start = phone_tiers[i-1][0]
                    t_end = phone_tier[1]

                    idx_start = int(t_start * sr)
                    idx_end = int(t_end * sr)
                    
                    clip = audio[idx_start:idx_end].astype(np.float32, copy=True)
                    
                    row.append(
                        {
                            "phone": phone,
                            "prev_phone": prev_phone,
                            "FILENAME": filename,
                            "rep": rep,
                            "start_time": t_start,
                            "end_time": t_end,
                            "AUDIO": clip,
                            "dur": dur,
                            "f1": f1_values,
                            "f2": f2_values,
                            "f3": f3_values,
                            'speaker': speaker
                        }
                    )
            print(f"{filename} is done.")
vowel_one_phone_info = pd.DataFrame(row)
vowel_one_phone_info.to_pickle("./context_size/one_previous_phone/vowel_one_phone_info.pkl")

vowel_one_phone_info = pd.read_pickle("./context_size/one_previous_phone/vowel_one_phone_info.pkl")

def one_time_point_per_row(df):
    """
    df: info file with raw audio + f1/f2/f3 per vowel token
    Returns: long-format dataframe (one row per 20ms time point)
    """
    rows_long = []

    for _, row in df.iterrows():
        speaker = row["speaker"]
        phone = row["phone"]
        prev_phone = row["prev_phone"]
        filename = row['FILENAME']
        f1 = row['f1']
        f2 = row['f2']
        f3 = row['f3']
        start_time = row['start_time']
        end_time = row['end_time']

        times = np.arange(start_time, end_time, 0.02)

        for frame_idx, t in enumerate(times):
            rows_long.append({
                'filename': filename,
                'time': t,
                'f1': f1[frame_idx],
                'f2': f2[frame_idx],
                'f3': f3[frame_idx],
                'phone': phone,
                'prev_phone': prev_phone,
                'speaker': speaker
            })

    return pd.DataFrame(rows_long)

vowel_one_phone_formant_df = one_time_point_per_row(vowel_one_phone_info)
vowel_one_phone_formant_df.to_pickle("./context_size/one_previous_phone/vowel_one_phone_formant_df.pkl")

