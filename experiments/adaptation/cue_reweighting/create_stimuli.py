# %%
%load_ext autoreload
%autoreload 2

import pandas as pd
import numpy as np

import soundfile as sf
import os
import ast
import sys
import shutil


from pathlib import Path
from praatio.audio import getDuration
from praatio import textgrid
from functions import *
# %% paths 
p = Path.cwd()
stimuli_dir = p.parent.parent / "stimuli"
long_rec_dir = stimuli_dir / "cue_reweighting_recordings"
WAV_DIR = stimuli_dir / "after_alignment"
np.random.seed(134)

# %% 
# Create TextGrid files and do mfa
# Note: even in a virtual env, use full location /Users/yiminglu/miniforge3/envs/aligner/bin/mfa 
trial_rec_folder = Path(stimuli_dir / "before_alignment")

for file in trial_rec_folder.rglob("*.wav"):
    f0 = int(file.name.split(".")[0].split("_")[3].removesuffix("hz"))
    VOT = int(file.name.split(".")[0].split("_")[5].removesuffix("ms"))
    if VOT < 25:
        transcription = "beer"
    elif VOT > 25: 
        transcription = "pier"
    else:
        if f0 < 250:
            transcription = "beer"
        else:
            transcription = "pier"
    dur = getDuration(str(file))

    tg = textgrid.Textgrid()
    tier = textgrid.IntervalTier("transcription", [(0.0, dur, transcription)], 0.0, dur) 
    tg.addTier(tier)

    tg_path = file.with_suffix(".TextGrid")
    tg.save(str(tg_path), format="long_textgrid", includeBlankSpaces=False)
# %%
# after mfa,
# encode stimuli values
canonical = pd.DataFrame(

    [
        [220, 10, "b"],
        [230, 5, "b"],
        [230, 10, "b"],
        [230, 15, "b"],
        [240, 10, "b"],
        [300, 40, "p"],
        [310, 35, "p"],
        [310, 40, "p"],
        [310, 45, "p"],
        [320, 40, "p"]
    ],

    columns = ['f0', 'VOT', "category"]
)

reversed = pd.DataFrame(
    [
        [320, 10, "b"],
        [310, 5, "b"],
        [310, 10, "b"],
        [310, 15, "b"],
        [300, 10, "b"],
        [240, 40, "p"],
        [230, 35, "p"],
        [230, 40, "p"],
        [230, 45, "p"],
        [220, 40, "p"]
    ],

    columns = ['f0', 'VOT', "category"]
)

test = pd.DataFrame(
    [
        [230, 25, "b"],
        [310, 25, "p"]
    ],

    columns = ['f0', 'VOT', "category"]
) # but is it ok to assign category this way?

# %%
# create stimuli: 24 trials (3 iterations were used for Murphy25)

# 1 iteration

# randomly select 8 from each category, and re-shuffle

canonical_8 = canonical.groupby("category", group_keys=False).sample(n=4)
canonical_8 = canonical_8.sample(frac=1).reset_index(drop=True)

reversed_8 = reversed.groupby("category", group_keys=False).sample(n=4)
reversed_8 = reversed_8.sample(frac=1).reset_index(drop=True)

## canonical
concat_wavs_with_silence(
    canonical_8[["f0", "VOT"]],
    out_path= long_rec_dir / "canonical_8_concat.wav",
    silence_ms=300
)

## reversed
concat_wavs_with_silence(
    reversed_8[["f0", "VOT"]],
    out_path= long_rec_dir / "reversed_8_concat.wav",
    silence_ms=300
)

concat_wavs_with_silence(
    test[["f0", "VOT"]],
    out_path= long_rec_dir / "test.wav",
    silence_ms=300
)

concat_two_wavs_with_silence(sf1="canonical_8_concat.wav", sf2="test.wav", outfname="canonical_8_with_test.wav")
concat_two_wavs_with_silence(sf1="reversed_8_concat.wav", sf2="test.wav", outfname="reversed_8_with_test.wav")

# 2 iterations

canonical_8_iter2 = canonical.groupby("category", group_keys=False).sample(n=4)
canonical_8_iter2 = canonical_8_iter2.sample(frac=1).reset_index(drop=True)

reversed_8_iter2 = reversed.groupby("category", group_keys=False).sample(n=4)
reversed_8_iter2 = reversed_8_iter2.sample(frac=1).reset_index(drop=True)

concat_wavs_with_silence(
    canonical_8_iter2[["f0", "VOT"]],
    out_path= long_rec_dir / "canonical_8_concat_iter2.wav",
    silence_ms=300
)

concat_wavs_with_silence(
    reversed_8_iter2[["f0", "VOT"]],
    out_path= long_rec_dir / "reversed_8_concat_iter2.wav",
    silence_ms=300
)

# concat 2 iterations of exposure first
concat_two_wavs_with_silence(sf1="canonical_8_concat.wav", sf2="canonical_8_concat_iter2.wav", silence_ms=300, outfname="canonical_16_concat.wav")
concat_two_wavs_with_silence(sf1="reversed_8_concat.wav", sf2="reversed_8_concat_iter2.wav", silence_ms=300, outfname="reversed_16_concat.wav")

# then concat with test trials
concat_two_wavs_with_silence(sf1="canonical_16_concat.wav", sf2="test.wav", silence_ms=600, outfname="canonical_16_with_test.wav")
concat_two_wavs_with_silence(sf1="reversed_16_concat.wav", sf2="test.wav", silence_ms=600, outfname="reversed_16_with_test.wav")

# 3 iterations
canonical_8_iter3 = canonical.groupby("category", group_keys=False).sample(n=4)
canonical_8_iter3 = canonical_8_iter3.sample(frac=1).reset_index(drop=True)

reversed_8_iter3 = reversed.groupby("category", group_keys=False).sample(n=4)
reversed_8_iter3 = reversed_8_iter3.sample(frac=1).reset_index(drop=True)

concat_wavs_with_silence(
    canonical_8_iter3[["f0", "VOT"]],
    out_path= long_rec_dir / "canonical_8_concat_iter3.wav",
    silence_ms=300
)

concat_wavs_with_silence(
    reversed_8_iter3[["f0", "VOT"]],
    out_path= long_rec_dir / "reversed_8_concat_iter3.wav",
    silence_ms=300
)

# concat iter1+2+3 exposure first
concat_two_wavs_with_silence(sf1="canonical_16_concat.wav", sf2="canonical_8_concat_iter3.wav", silence_ms=300, outfname="canonical_24_concat.wav")
concat_two_wavs_with_silence(sf1="reversed_16_concat.wav", sf2="reversed_8_concat_iter3.wav", silence_ms=300, outfname="reversed_24_concat.wav")

# then concat with test trials
concat_two_wavs_with_silence(sf1="canonical_24_concat.wav", sf2="test.wav", silence_ms=600, outfname="canonical_24_with_test.wav")
concat_two_wavs_with_silence(sf1="reversed_24_concat.wav", sf2="test.wav", silence_ms=600, outfname="reversed_24_with_test.wav")

# %% 
# Create audio_df for all with_test files -> necessary to get deep model hid state

rows = []
for file in Path(long_rec_dir).rglob("*"):
    if "with_test" in file.name or file.name=="test.wav":
        audio, sr = sf.read(file, always_2d=False)
        rows.append(
            {
            "FILENAME": file.name,
            "AUDIO": audio.astype("float32")
            }
        )

audio_df = pd.DataFrame(rows)
audio_df.to_pickle(f"{long_rec_dir}/cue_reweighting_stimuli_audio_df.pkl")

# %% create long textgrid
# Create transcription: concatenate textgrid files, create time-resolved df based on concatenated textgrid files
concat_textgrid_based_on_shuffling(out_dir=long_rec_dir, df=canonical_8, tg_dir=WAV_DIR, tg_name="canonical_8")
concat_textgrid_based_on_shuffling(out_dir=long_rec_dir, df=reversed_8, tg_dir=WAV_DIR, tg_name="reversed_8")
concat_textgrid_based_on_shuffling(out_dir=long_rec_dir, df=test, tg_dir=WAV_DIR, tg_name="test")
concat_textgrid_based_on_shuffling(out_dir=long_rec_dir, df=canonical_8_iter2, tg_dir=WAV_DIR, tg_name="canonical_8_iter_2")
concat_textgrid_based_on_shuffling(out_dir=long_rec_dir, df=reversed_8_iter2, tg_dir=WAV_DIR, tg_name="reversed_8_iter_2")
concat_textgrid_based_on_shuffling(out_dir=long_rec_dir, df=canonical_8_iter3, tg_dir=WAV_DIR, tg_name="canonical_8_iter_3")
concat_textgrid_based_on_shuffling(out_dir=long_rec_dir, df=reversed_8_iter3, tg_dir=WAV_DIR, tg_name="reversed_8_iter_3")

# create iter-1 full textgrids
combine_textgrids_and_add_silence(first_tg_name="canonical_8", sec_tg_name="test", input_dir=Path(long_rec_dir), output_dir=Path(long_rec_dir), silence_ms=600, tg_name="canonical_8_with_test")
combine_textgrids_and_add_silence(first_tg_name="reversed_8", sec_tg_name="test", input_dir=Path(long_rec_dir), output_dir=Path(long_rec_dir), silence_ms=600, tg_name="reversed_8_with_test")

# create iter-2 full textgrids
combine_textgrids_and_add_silence(first_tg_name="canonical_8", sec_tg_name="canonical_8_iter_2", input_dir=Path(long_rec_dir), output_dir=Path(long_rec_dir), silence_ms=300, tg_name="canonical_16")
combine_textgrids_and_add_silence(first_tg_name="reversed_8", sec_tg_name="reversed_8_iter_2", input_dir=Path(long_rec_dir), output_dir=Path(long_rec_dir), silence_ms=300, tg_name="reversed_16")
combine_textgrids_and_add_silence(first_tg_name="canonical_16", sec_tg_name="test", input_dir=Path(long_rec_dir), output_dir=Path(long_rec_dir), silence_ms=600, tg_name="canonical_16_with_test")
combine_textgrids_and_add_silence(first_tg_name="reversed_16", sec_tg_name="test", input_dir=Path(long_rec_dir), output_dir=Path(long_rec_dir), silence_ms=600, tg_name="reversed_16_with_test")

# create iter-3 full textgrids
combine_textgrids_and_add_silence(first_tg_name="canonical_16", sec_tg_name="canonical_8_iter_3", input_dir=Path(long_rec_dir), output_dir=Path(long_rec_dir), silence_ms=300, tg_name="canonical_24")
combine_textgrids_and_add_silence(first_tg_name="reversed_16", sec_tg_name="reversed_8_iter_3", input_dir=Path(long_rec_dir), output_dir=Path(long_rec_dir), silence_ms=300, tg_name="reversed_24")
combine_textgrids_and_add_silence(first_tg_name="canonical_24", sec_tg_name="test", input_dir=Path(long_rec_dir), output_dir=Path(long_rec_dir), silence_ms=600, tg_name="canonical_24_with_test")
combine_textgrids_and_add_silence(first_tg_name="reversed_24", sec_tg_name="test", input_dir=Path(long_rec_dir), output_dir=Path(long_rec_dir), silence_ms=600, tg_name="reversed_24_with_test")

# %% create a time annotated df
# Note: even in a virtual env, use full location: /Users/yiminglu/miniforge3/envs/aligner/bin/mfa 
exp_rec = Path(stimuli_dir / "cue_reweighting_experiment")
files = os.listdir(exp_rec)


rows = []
step = 0.02

for file in files:
    ext = os.path.splitext(file)[1].lower()
    if ext != ".textgrid":
        continue

    tg_path = exp_rec / file

    filename = os.path.splitext(file)[0]

    tg = textgrid.openTextgrid(str(tg_path), includeEmptyIntervals=True)
    phone_tier = tg.getTier(tierName="phones")

    entries = phone_tier.entries

    counts={}
    intervals = []
    for start, end, lab in entries:
        if lab not in {'P', 'B'}:
            continue
        
        rep=counts.get(lab, 0) + 1
        counts[lab] = rep

        intervals.append((start, end, lab, rep))

    intervals.sort(key=lambda x: x[0])
    n = len(intervals)
    j = 0

    t0 = tg.minTimestamp
    t1 = tg.maxTimestamp
    times = np.arange(t0, t1, step, dtype=float)

    
    for t in times:
        bin_start = t
        bin_end = t + step


        while j < n and intervals[j][1] <= bin_start:
            j += 1

        if j < n:
            s, e, lab, rep = intervals[j]
            overlaps = (bin_start < s < bin_end)
        else:
            overlaps = False

        if overlaps:
            phone = lab
            rep_out = rep
            s_out, e_out = s, e
        else:
            phone = ""
            rep_out = 0
            s_out, e_out = np.nan, np.nan

        rows.append({
            "filename": filename,
            "time": float(t),
            "phone": phone,
            "rep": int(rep_out),
            "start": float(s_out) if s_out == s_out else np.nan,
            "end": float(e_out) if e_out == e_out else np.nan,
        })

time_df = pd.DataFrame(rows)
time_df.to_pickle(exp_rec / "cue_reweighting_time_annotated.pkl")
tryout_time_df = time_df[time_df['filename']=="canonical_16_with_test"]
# %% create time-annotated hidden act df
time_df = pd.read_pickle(exp_rec / "cue_reweighting_time_annotated.pkl")

hid_act_layer1 = read_activation_and_left_join_with_acoustics(hid_act_folder="../../embeddings/wav2vec2_base",
                                                       hid_pkl_folder="../../embeddings/wav2vec2_base_pkl",
                                                       num_layers=12,
                                                       which_layer=1,
                                                       time_df=time_df, 
                                                       dont_read_again=False)

# %% get the test trials from each recording

def take_last2rep_phone_window(g, N=9):
    '''
    
    :param g: the df
    :param N: number of following frames we want to take
    '''
    reps = g["rep"].dropna().unique()
    reps = reps[reps != 0]

    last2 = np.sort(reps)[-1:]  # last rep 
    rep_arr = g["rep"].to_numpy()
    nonzero_rep_mask = (~pd.isna(rep_arr)) & (rep_arr != 0)

    out_parts = []
    for r in last2:
        start_positions = np.flatnonzero(rep_arr == r)
        if start_positions.size == 0:
            continue
        for start_pos in start_positions:
            start_pos = int(start_pos)

            next_nonzero_positions = np.flatnonzero(nonzero_rep_mask & (np.arange(len(g)) > start_pos))
            end_pos = int(next_nonzero_positions[0]) if next_nonzero_positions.size else len(g)

            gr = g.iloc[start_pos:end_pos]

            out = gr.iloc[: (N + 1)].copy()
            out["rep_selected"] = r
            out["rep_start_pos"] = start_pos  
            out_parts.append(out)

    if not out_parts:
        return g.iloc[0:0]

    return pd.concat(out_parts, axis=0)

lda_windows = (
    hid_act_layer1.groupby("filename", group_keys=False, sort=False)
      .apply(take_last2rep_phone_window)
      .reset_index(drop=True)
)

stimuli_dir = p.parent.parent / "embeddings/wav2vec2_before_lda"
stimuli_dir.mkdir(exist_ok=True)

lda_windows.to_pickle(stimuli_dir / "lda_window_layer1.pkl")

# %% make into functions
def make_lda_windows_for_layer(
    *,
    layer: int,
    time_df_path: Path,
    hid_act_folder: str | Path,
    hid_pkl_folder: str | Path,
    out_dir: Path,
    out_fname: str | None = None,
    num_layers: int = 12,
    N: int = 9,
    take_last_k_reps: int = 1,   # 1 = last rep, 2 = last two reps, etc.
    dont_read_again: bool = False,
):
    time_df = pd.read_pickle(time_df_path)

    hid_act = read_activation_and_left_join_with_acoustics(
        hid_act_folder=str(hid_act_folder),
        hid_pkl_folder=str(hid_pkl_folder),
        num_layers=num_layers,
        which_layer=layer,
        time_df=time_df,
        dont_read_again=dont_read_again,
    )

    def take_lastkrep_phone_window(g, N=N):
        reps = g["rep"].dropna().unique()
        reps = reps[reps != 0]
        if len(reps) == 0:
            return g.iloc[0:0]

        lastk = np.sort(reps)[-take_last_k_reps:]  # last K reps

        rep_arr = g["rep"].to_numpy()
        nonzero_rep_mask = (~pd.isna(rep_arr)) & (rep_arr != 0)

        out_parts = []
        idx = np.arange(len(g))

        for r in lastk:
            start_positions = np.flatnonzero(rep_arr == r)
            if start_positions.size == 0:
                continue

            for start_pos in start_positions:
                start_pos = int(start_pos)

                next_nonzero_positions = np.flatnonzero(nonzero_rep_mask & (idx > start_pos))
                end_pos = int(next_nonzero_positions[0]) if next_nonzero_positions.size else len(g)

                gr = g.iloc[start_pos:end_pos]

                out = gr.iloc[: (N + 1)].copy()
                out["rep_selected"] = r
                out["rep_start_pos"] = start_pos
                out_parts.append(out)

        if not out_parts:
            return g.iloc[0:0]

        return pd.concat(out_parts, axis=0)

    lda_windows = (
        hid_act.groupby("filename", group_keys=False, sort=False)
        .apply(take_lastkrep_phone_window)
        .reset_index(drop=True)
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    if out_fname is None:
        out_fname = f"lda_window_layer{layer}.pkl"

    saved_path = out_dir / out_fname
    lda_windows.to_pickle(saved_path)

    return lda_windows, saved_path
from pathlib import Path

def make_lda_windows_all_layers(
    *,
    layers=range(1, 13),
    time_df_path: Path,
    hid_act_folder: str | Path,
    hid_pkl_folder: str | Path,
    out_dir: Path,
    num_layers: int = 12,
    N: int = 9,
    take_last_k_reps: int = 1,
    dont_read_again: bool = False,
):
    """
    Create and save lda_window_layer{L}.pkl for all layers in `layers`.

    Returns: dict {layer: saved_path}
    """
    saved = {}
    for layer in layers:
        _, saved_path = make_lda_windows_for_layer(
            layer=layer,
            time_df_path=time_df_path,
            hid_act_folder=hid_act_folder,
            hid_pkl_folder=hid_pkl_folder,
            out_dir=out_dir,
            out_fname=f"lda_window_layer{layer}.pkl",
            num_layers=num_layers,
            N=N,
            take_last_k_reps=take_last_k_reps,
            dont_read_again=dont_read_again,
        )
        saved[layer] = saved_path
    return saved


saved_paths = make_lda_windows_all_layers(
    layers=range(1, 13),
    time_df_path=exp_rec / "cue_reweighting_time_annotated.pkl",
    hid_act_folder="../../embeddings/wav2vec2_base",
    hid_pkl_folder="../../embeddings/wav2vec2_base_pkl",
    out_dir=p.parent.parent / "embeddings/wav2vec2_before_lda",
    N=9,
    take_last_k_reps=1,
    dont_read_again=False,
)

print(saved_paths)  
# %% move canonical files to another folder
paths = [stimuli_value_to_wav_filename(r, wav_dir=WAV_DIR) for _, r in canonical.iterrows()]

canonical_dir = stimuli_dir / "canonical"
canonical_dir.mkdir(parents=True, exist_ok=True)
for path in paths:
    fn = Path(path)
    base = fn.stem

    match = [p for p in WAV_DIR.iterdir()
                 if p.is_file()
                 and base in p.stem
                 ]
   

    tg_path = next((p for p in match if p.suffix == ".TextGrid"), None)
    
    
    wav_path = next((p for p in match if p.suffix == ".wav"), None)

    shutil.copy2(wav_path, canonical_dir / wav_path.name)
    shutil.copy2(tg_path, canonical_dir / tg_path.name)

# %% instead, we train the lda model on expriment (canonical) tokens

rows = []
for path in paths:
    
    fn = Path(path)
    audio, sr = sf.read(fn, dtype="float32")
    
    filename = fn.stem

    rows.append({
        "AUDIO": audio,
        "FILENAME": filename
    })

audio_df = pd.DataFrame(rows)
audio_df.to_pickle(canonical_dir / "canonical_words_df.pkl")
# %% create time df for the lda training data
can_rec = Path(stimuli_dir / "canonical")
files = os.listdir(can_rec)


rows = []
step = 0.02

for file in files:
    ext = os.path.splitext(file)[1].lower()
    if ext != ".textgrid":
        continue

    tg_path = can_rec / file

    filename = os.path.splitext(file)[0]

    tg = textgrid.openTextgrid(str(tg_path), includeEmptyIntervals=True)
    phone_tier = tg.getTier(tierName="phones")

    entries = phone_tier.entries

    counts={}
    intervals = []
    for start, end, lab in entries:
        if lab not in {'P', 'B'}:
            continue
        
        rep=counts.get(lab, 0) + 1
        counts[lab] = rep

        intervals.append((start, end, lab, rep))

    intervals.sort(key=lambda x: x[0])
    n = len(intervals)
    j = 0

    t0 = tg.minTimestamp
    t1 = tg.maxTimestamp
    times = np.arange(t0, t1, step, dtype=float)

    
    for t in times:
        bin_start = t
        bin_end = t + step


        while j < n and intervals[j][1] <= bin_start:
            j += 1

        if j < n:
            s, e, lab, rep = intervals[j]
            overlaps = (bin_start < s < bin_end)
        else:
            overlaps = False

        if overlaps:
            phone = lab
            rep_out = rep
            s_out, e_out = s, e
        else:
            phone = ""
            rep_out = 0
            s_out, e_out = np.nan, np.nan

        rows.append({
            "filename": filename,
            "time": float(t),
            "phone": phone,
            "rep": int(rep_out),
            "start": float(s_out) if s_out == s_out else np.nan,
            "end": float(e_out) if e_out == e_out else np.nan,
        })

time_df = pd.DataFrame(rows)
time_df.to_pickle(can_rec / "canonical_time_annotated.pkl")

# %% after getting embedding, 

hid_act_lda_train_layer1 = read_activation_and_left_join_with_acoustics(hid_act_folder="../../canonical_embedding/wav2vec2_base",
                                                       hid_pkl_folder="../../canonical_embedding/wav2vec2_base_pkl",
                                                       num_layers=12,
                                                       which_layer=1,
                                                       time_df=time_df, 
                                                       dont_read_again=False)
# %%
