# %%
import torch
import pandas as pd
import numpy as np
import soundfile as sf
import os
import ast
import sys

from pathlib import Path
from praatio.audio import getDuration
from praatio import textgrid

p = Path.cwd()
stimuli_dir = p.parent.parent / "stimuli"
WAV_DIR = stimuli_dir / "after_alignment"

# functions
def stimuli_value_to_wav_filename(row, wav_dir=WAV_DIR):
    fname = f"beer_pier_F0_{int(row['f0'])}hz_VOT_{int(row['VOT'])}ms.wav"
    return wav_dir / fname

def concat_wavs_with_silence(df, out_path, silence_ms=300, wav_dir=WAV_DIR):
    # turn stimuli values to corresponding recording filenames
    paths = [stimuli_value_to_wav_filename(r, wav_dir=wav_dir) for _, r in df.iterrows()]

    audio_list = []

    for p in paths:

        x, sr = sf.read(p, always_2d=True)  # make sure we always get 2d, even if the sound file is mono
        audio_list.append(x)

    # Create silence interval: 300 ms in Murphy et al., 2025
    silence = np.zeros((int(round(sr * silence_ms / 1000.0)), 1), dtype=audio_list[0].dtype) 
    # keep the same data type as sf output, in this case, float64

    # stack them together
    out = audio_list[0]
    for x in audio_list[1:]:
        out = np.vstack([out, silence, x])

    # Write it as a wav file
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_path, out, sr)

    return paths, sr, out.shape

def concat_two_wavs_with_silence(sf1, sf2, outfname, silence_ms=600, wav_dir= p/"adaptation_exps/stimuli/cue_reweighting_recordings"):
    wav_dir = Path(wav_dir)

    p1 = wav_dir / sf1
    p2 = wav_dir / sf2
    outpath = wav_dir / outfname

    x1, sr1 = sf.read(p1, always_2d=True)
    x2, sr2 = sf.read(p2, always_2d=True)

    if sr1 != sr2:
        raise ValueError("the two sound files have different sampling rate. It won't work!")
    n_sil = int(round(sr1 * silence_ms / 1000.0))
    silence = np.zeros((n_sil, x1.shape[1]), dtype=x1.dtype)

    out = np.vstack([x1, silence, x2])

    outpath.parent.mkdir(parents=True, exist_ok=True)

    sf.write(outpath, out, sr1)
    return outpath

def concat_textgrid_based_on_shuffling(out_dir, df, tg_dir, tg_name, silence_dur=300):
    '''
    Docstring for concat_textgrid_based_on_shuffling
    
    :param out_dir: Description
    :param df: Description
    :param tg_dir: Description
    :param tg_name: Description
    :param silence_dur: Description
    '''
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tg_paths = [Path(stimuli_value_to_textgrid_filename(r, tg_dir=tg_dir))
                for _, r in df.iterrows()]

    merged_tier_entries = {}
    offset = 0.0

    for i, tg_path in enumerate(tg_paths):
        tg = textgrid.openTextgrid(str(tg_path), includeEmptyIntervals=True)

        interval_tiers = [t for t in tg.tiers if isinstance(t, textgrid.IntervalTier)]

        for tier in interval_tiers:
            shifted = [(s + offset, e + offset, lab) for (s, e, lab) in tier.entries]
            merged_tier_entries.setdefault(tier.name, []).extend(shifted)

        token_dur = float(tg.maxTimestamp)
        offset += token_dur

        if i != len(tg_paths) - 1 and silence_dur and silence_dur > 0:
            sil_dur = silence_dur / 1000.0
            sil_start = offset
            sil_end = offset + sil_dur

            for tier in interval_tiers:
                merged_tier_entries.setdefault(tier.name, []).append((sil_start, sil_end, ""))

            offset = sil_end

    total_dur = offset

    out_tg = textgrid.Textgrid()
    for name, entries in merged_tier_entries.items():
        entries = sorted(entries, key=lambda z: (z[0], z[1]))
        out_tg.addTier(textgrid.IntervalTier(name, entries, 0.0, total_dur))

    out_tg_path = out_dir / f"{tg_name}.TextGrid"
    out_tg.save(str(out_tg_path), format="long_textgrid", includeBlankSpaces=True)

    return tg_paths, total_dur

def combine_textgrids_and_add_silence(first_tg_name, sec_tg_name, input_dir, output_dir, silence_ms, tg_name):
    '''
    Docstring for combine_textgrids_and_add_silence
    
    :param first_rec_name: the first recording name
    :param sec_rec_name: the second recording name
    :param input_dir: where the two textgrid files are
    :param output_dir: where we want the concatenated textgrid to be
    :param silence_ms: silence duration between the two textgrids, same as recordings
    :param tg_name: what we want to call it
    '''

    first_path = Path(input_dir / f"{first_tg_name}.TextGrid")
    sec_path = Path(input_dir / f"{sec_tg_name}.TextGrid")

    tg_paths = [first_path, sec_path]

    merged_tier_entries = {}
    offset = 0.0
    for i, tg_path in enumerate(tg_paths):
        tg = textgrid.openTextgrid(str(tg_path), includeEmptyIntervals=True)

        interval_tiers = [t for t in tg.tiers if isinstance(t, textgrid.IntervalTier)]

        for tier in interval_tiers:
            shifted = [(s + offset, e + offset, lab) for (s, e, lab) in tier.entries]
            merged_tier_entries.setdefault(tier.name, []).extend(shifted)

        token_dur = float(tg.maxTimestamp)
        offset += token_dur

        # only between the first and second textgrids
        if i == 0:
            sil_dur = silence_ms / 1000.0
            sil_start = offset
            sil_end = offset + sil_dur

            for tier in interval_tiers:
                merged_tier_entries.setdefault(tier.name, []).append((sil_start, sil_end, ""))

            offset = sil_end

    total_dur = offset

    out_tg = textgrid.Textgrid()
    for name, entries in merged_tier_entries.items():
        entries = sorted(entries, key=lambda z: (z[0], z[1]))
        out_tg.addTier(textgrid.IntervalTier(name, entries, 0.0, total_dur))

    out_tg_path = output_dir / f"{tg_name}.TextGrid"
    out_tg.save(str(out_tg_path), format="long_textgrid", includeBlankSpaces=True)

    return out_tg_path, total_dur


def add_words(df):
    df = df.copy()
    df["words"] = np.where(df["category"] == "b", "beer", "pier")
    return df


def stimuli_value_to_textgrid_filename(row, tg_dir):
    '''
    Docstring for stimuli_value_to_textgrid_filename
    
    :param row: what is the f0 and vot of that stimuli; based on sequence after shuffling
    :param tg_dir: where is the sound file? you will find them in "after_alignment"
    '''
    return Path(tg_dir) / f"beer_pier_F0_{int(row['f0'])}hz_VOT_{int(row['VOT'])}ms.TextGrid"

def read_activation_and_left_join_with_acoustics(hid_act_folder, hid_pkl_folder, num_layers, which_layer, time_df, dont_read_again):
    os.makedirs(hid_pkl_folder, exist_ok=True)

    expected_pkls = [os.path.join(hid_pkl_folder, f"hidden_acts_{layer}.pkl") for layer in range(1, num_layers + 1)]
    if all(os.path.exists(pkl) for pkl in expected_pkls) and dont_read_again:
        print("All layer pickle files found, will load the target layer.")
        acts = pd.read_pickle(f"./{hid_pkl_folder}/hidden_acts_{which_layer}.pkl")
        return acts

    else:
        time_idx = time_df.set_index(["filename", "time"])

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
                    filename = file.split(".")[0]
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
                    file_act_with_phone = file_act_df.join(time_idx, on=["filename", "time"], how='left')


                    all_files_act_df.append(file_act_with_phone)

                    print(f"Layer {layer}: file {file} is done. This is the {idx} out of 2702 files.")
                
                hid_act_df = pd.concat(all_files_act_df, ignore_index=True)
                hid_act_df.to_pickle(f"./{hid_pkl_folder}/hidden_acts_{layer}.pkl")

                print(f"Layer {layer} is done")

        acts = pd.read_pickle(f"./{hid_pkl_folder}/hidden_acts_{which_layer}.pkl")

    return acts