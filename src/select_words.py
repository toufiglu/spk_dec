# %% import 
import importlib
import helpers
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from praatio import textgrid
import glob
import soundfile as sf
from praatio.audio import getDuration
import pyphen
import shutil
import re
import joblib

from pathlib import Path
importlib.reload(helpers)

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression


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

def get_num_syllables(word: str) -> int:
    '''
    get number of syllable 
    '''
    parts = dic.inserted(word).split("-")
    return len([p for p in parts if p])

def take_window(df, N=9):
    v = df["vot"]

    mask = v.astype(str).str.strip().ne("")
    if not mask.any():
        return df.iloc[0:0]  
    first_pos = mask.to_numpy().argmax()
    out = df.iloc[first_pos: first_pos+(N+1)].copy()
    return out

def concat_activation(df):
    return np.stack(df["act"].to_numpy()).reshape(-1)

def create_category_label(df):
    word = df['filename'].str.split("_").str.get(1)
    df["voice"] = np.where(word.str.startswith("b"), "voiced", "voiceless")
    return df

def concat_activation_output_df(df):
    return pd.Series({
        "act": np.stack(df["act"].to_numpy()).reshape(-1),
        "phone": df['phone'].iloc[0]
    })

# all vowels in arpabet
vowels = [
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


# %% find monosyllabic words in Librispeech dev set
p = Path.cwd()
tg_dirs = p / "../data/raw/mfa_align/my_corpus_aligned"
words = []
dic = pyphen.Pyphen(lang="en_US")
# below are words that passed the automatic detection but don't fit the criteria. 
# Some were not English words. Others have more than 1 syllable.
removed = ["bui", "basil", "pavel", "porter", "body", "parish", "baron", "poniard", "p", "putting", "pieces"]
targets = {"B", "P"}

for speaker in os.listdir(tg_dirs):
    speaker_dir = os.path.join(tg_dirs, speaker)
    if not os.path.isdir(speaker_dir):
        continue

    for rec_folder in os.listdir(speaker_dir):
        rec_dir = os.path.join(speaker_dir, rec_folder)
        if not os.path.isdir(rec_dir):
            continue

        tg_paths = glob.glob(os.path.join(rec_dir, "*.TextGrid"))
        for tg_path in tg_paths:
             tg = textgrid.openTextgrid(str(tg_path), includeEmptyIntervals=True)
             phone_tier = tg.getTier(tierName="phones")                 
             word_tier = tg.getTier(tierName="words")
             phone_entries = phone_tier.entries
             word_entries = word_tier.entries

             for i, (ps, pe, plab) in enumerate(phone_entries):
                if plab not in targets:
                    continue
                if i != 0:
                    prev_lab = phone_entries[i - 1][2]
                    if prev_lab not in ("", None):
                        continue
                if i + 1 >= len(phone_entries):
                    continue

                foll_lab = phone_entries[i + 1][2]
                if foll_lab not in vowels:
                    continue
                
                t = (ps + pe) / 2.0
                for ws, we, wlab in word_entries:
                    if ws <= t < we and wlab not in ("", None):
                        if get_num_syllables(wlab) != 1:
                            break
                        if wlab in removed:
                            break
                        words.append(wlab)
                        break

unique_words = list(set(words))

# %% chop corresponding recordings, create sound files, create textgrid with word
word_rec_out_dir = p / "../adaptation_exps/stimuli/librispeech_bp_words"
os.makedirs(word_rec_out_dir, exist_ok=True)
rows = []
for speaker in os.listdir(tg_dirs):
    speaker_dir = os.path.join(tg_dirs, speaker)
    if not os.path.isdir(speaker_dir):
        continue

    for rec_folder in os.listdir(speaker_dir):
        rec_dir = os.path.join(speaker_dir, rec_folder)
        if not os.path.isdir(rec_dir):
            continue

        tg_paths = glob.glob(os.path.join(rec_dir, "*.TextGrid"))

        wav_paths = [tg.replace("my_corpus_aligned", "corpus_before_align").replace(".TextGrid", ".wav") for tg in tg_paths]

        for j, tg_path in enumerate(tg_paths):
             audio, sr = sf.read(wav_paths[j]) 
             tg = textgrid.openTextgrid(str(tg_path), includeEmptyIntervals=True)
             word_tier = tg.getTier(tierName="words")
             word_entries = word_tier.entries

             for i, (ws, we, wlab) in enumerate(word_entries):
                 wlab_clean = wlab.strip() # if the label has space, strip it
                 if wlab_clean in unique_words:
                    raw_dur = we - ws 
                    if raw_dur < 0.128: # requirement of Dr.VOT later
                        continue
                    
                    idx_start = int(ws * sr)
                    idx_end = int(we * sr)
                    clip = audio[idx_start:idx_end].astype(np.float32, copy=True) 

                    rec_basename = os.path.basename(tg_path).split(".")[0]
                    word_fname = f"{rec_basename}_{wlab_clean}_{i}_{ws:.3f}-{we:.3f}.wav" 
                    # add i (sequence) and start/end time, in case this word appears multiple times in this recording

                    word_rec_wav_path = word_rec_out_dir / word_fname
                    sf.write(str(word_rec_wav_path), clip, sr, subtype="PCM_16") 

                    # create tg files for each word
                    dur = (idx_end - idx_start) / sr

                    tg = textgrid.Textgrid()
                    tier = textgrid.IntervalTier("transcription", [(0.0, dur, wlab_clean)], 0.0, dur) 
                    tg.addTier(tier)

                    out_tg_path = word_rec_wav_path.with_suffix(".TextGrid")
                    tg.save(str(out_tg_path), format="long_textgrid", includeBlankSpaces=False)

                    rows.append({
                        "FILENAME": word_fname,
                        "AUDIO": clip
                    })
audio_df = pd.DataFrame(rows)
audio_df.to_pickle(word_rec_out_dir / "single_word_audio_df.pickle")

# %% check VOT summary
VOT_summary_dir = Path(p / "../adaptation_exps/stimuli/librispeech_VOT_tg/summary.csv")
summary = pd.read_csv(VOT_summary_dir)
summary['VOT'] = np.where(summary[" type"]=="POS_VOT", summary[" duration(msec)"], -summary[" duration(msec)"])

summary["word"] = summary["filename"].str.split("_").str[1]
summary['voice'] = np.where(summary['word'].str.startswith("b"), "voiced", "voiceless")
summary.groupby("voice").size()

# filtered = summary[
#     summary["VOT"].sub(summary.groupby("voice")["VOT"].transform("mean")).abs()
#     <= 3 * summary.groupby("voice")["VOT"].transform("std")
# ].copy()


# %% draw random number of samples from the voiced category
# np.random.seed(123)

# n = filtered.groupby("voice").size().min()

# balanced_sample = filtered.groupby("voice").sample(n=n)

# balanced_sample.groupby("voice")["VOT"].min()
# balanced_sample.groupby("voice")['VOT'].max()
# balanced_sample.groupby("voice").size()
# %% plot the distribution
tmp = summary[["VOT", "voice"]].dropna()

plt.figure()
for grp, d in tmp.groupby("voice"):
    d["VOT"].plot(kind="kde", label=grp)  # KDE density

plt.xlabel("VOT (msec)")
plt.ylabel("Density")
plt.title("VOT density: voiced vs voiceless")
plt.legend(title="voice")
plt.tight_layout()
plt.show()

# in all cases, we can use 10 frames from the start of 
# %% copy sampled wav files to another folder
src_wav_folder = Path(p / "../adaptation_exps/stimuli/librispeech_bp_recs_with_VOT_tgs")
dst_wav_folder = Path(p / "../adaptation_exps/stimuli/selected_bp_wav_tg")
dst_wav_folder.mkdir(parents=True, exist_ok=True)

filenames = set(summary["filename"].astype(str))

for fn in filenames:
    fn = Path(fn)
    wav_path = src_wav_folder / fn
    base = fn.stem

    match = [p for p in src_wav_folder.iterdir()
                 if p.is_file()
                 and base in p.stem
                 and p.suffix.lower() == ".textgrid"]

    
    tg_path = match[0]
    if tg_path is None:
        print("missing tg for file", base) 

    shutil.copy2(wav_path, dst_wav_folder / wav_path.name)
    shutil.copy2(tg_path, dst_wav_folder / tg_path.name)

# %% get time df for words
files = os.listdir(dst_wav_folder)
rows = []
for file in files:
    ext = os.path.splitext(file)[1].lower()
    if ext != ".textgrid":
        continue

    tg_path = dst_wav_folder / file

    filename = os.path.splitext(file)[0]
    filename = re.sub(r"_pred.*$", "", filename)

    tg = textgrid.openTextgrid(str(tg_path), includeEmptyIntervals=True)
    VOT_tier = tg.getTier(tierName="VOT")

    entries = VOT_tier.entries
    intervals = []
    for start, end, lab in entries:
        intervals.append((start, end, lab))
    
    dur = tg.maxTimestamp - tg.minTimestamp

    step = .02  # 20 ms per frame
    times = np.arange(0.0, dur, step, dtype=float)

    j = 0
    n = len(intervals)
 
    for t in times:
        while j < n and t >= intervals[j][1]:
            j += 1

        if j < n and intervals[j][0] <= t < intervals[j][1]:
            start, end, lab = intervals[j]
        else:
            start, end, lab = (np.nan, np.nan, "")

        rows.append({
            "filename": filename,
            "time": float(t),
            "vot": lab,
            "start": float(start) if start == start else np.nan,  # keep NaN if nan
            "end": float(end) if end == end else np.nan,
        })

time_df = pd.DataFrame(rows)
time_df.to_pickle(dst_wav_folder / "word_time_df.pickle")
# %% create time-annotated hidden act df
time_df = pd.read_pickle(dst_wav_folder / "word_time_df.pickle")
hid_act_layer1 = helpers.read_activation_and_left_join_with_acoustics(hid_act_folder="../adaptation_exps/single_word_embeddings/wav2vec2_base",
                                                       hid_pkl_folder="../adaptation_exps/single_word_embeddings/wav2vec2_base_pkl",
                                                       num_layers=12,
                                                       which_layer=6,
                                                       formant_df=time_df, 
                                                       dont_read_again=False)

hid_act_layer1_filtered = hid_act_layer1.groupby("filename").filter(lambda g: g['vot'].notna().any()).copy()
# %% LDA
hid_act_1_lda_window = (
    hid_act_layer1_filtered.groupby("filename", group_keys=False, sort=False)
    .apply(take_window)
)

files = hid_act_1_lda_window.groupby("filename").filter(lambda g: len(g) == 10)

hid_act_1_act_flattened = (
    files.groupby("filename", group_keys=False, sort=False)
    .apply(concat_activation)
    .reset_index(name="act_vec")
)

hid_act_1_flat_with_label = (hid_act_1_act_flattened.groupby("filename", group_keys=False, sort=False)
                             .apply(create_category_label))
# %% write an LDA function here
lda_saved_path = p.parent / "data/processed/adaptation_lda"
lda_saved_path.mkdir(exist_ok=True)

pipeline = Pipeline(
    [
        ('scaler', StandardScaler()),
        ('lda', LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))
    ]
)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

X = np.stack(hid_act_1_flat_with_label['act_vec'].to_numpy())
y = hid_act_1_flat_with_label['voice'].to_numpy()

assert len(X)==len(y)
le = LabelEncoder()
y = le.fit_transform(y)
pipeline.fit(X, y)

out_path = Path(lda_saved_path / "lda_pipeline.joblib")
joblib.dump(
    {
        "model": pipeline,
        "label_encoder": le,   # save this too, so class mapping is consistent
    },
    out_path,
)

# %% prediction
# re-load saved lda
model_saved = joblib.load(out_path)
pipeline_saved = model_saved["model"]
encoder_saved = model_saved["label_encoder"]

# read saved test trial hidden activations
test_trial_path = p.parent / "adaptation_exps/embeddings/wav2vec2_before_lda"
test_trial_hid_df = pd.read_pickle(test_trial_path / "lda_window_layer1.pkl")


test_trial_concat_df = (
    test_trial_hid_df
    .groupby(["filename", "rep_start_pos"], group_keys=False)
    .apply(concat_activation_output_df)
    .reset_index()
    )

test_trial_concat_df["voice"] = np.where(test_trial_concat_df["phone"]=="B", "voiced", "voiceless")

X_test = np.stack(test_trial_concat_df['act'].to_numpy())
pred_int = pipeline_saved.predict(X_test)
pred_label = encoder_saved.inverse_transform(pred_int)

proba = pipeline_saved.predict_proba(X_test)
conf = proba.max(axis=1) # for each row, take the prob of the most probable class

test_trial_concat_df["pred_voice"] = pred_label
test_trial_concat_df["pred_voice_prob"] = conf

# %%
def train_model_for_layer(
    *,
    layer: int,
    time_df: pd.DataFrame,
    saved_path: Path,
    hid_act_folder: str,
    hid_pkl_folder: str,
    num_layers: int = 12,
    model: str,
    training_data: str
):
    hid_act = helpers.read_activation_and_left_join_with_acoustics(
        hid_act_folder=hid_act_folder,
        hid_pkl_folder=hid_pkl_folder,
        num_layers=num_layers,
        which_layer=layer,
        formant_df=time_df,
        dont_read_again=True,
    )

    if training_data=="libri":
        hid_act_filtered = (
            hid_act.groupby("filename")
            .filter(lambda g: g["vot"].notna().any())
            .copy()
        )

        lda_window = (
            hid_act_filtered.groupby("filename", group_keys=False, sort=False)
            .apply(take_window)
        )

        files = lda_window.groupby("filename").filter(lambda g: len(g) == 10)

        act_flat = (
            files.groupby("filename", group_keys=False, sort=False)
            .apply(concat_activation)
            .reset_index(name="act_vec")
        )

        flat_with_label = (
            act_flat.groupby("filename", group_keys=False, sort=False)
            .apply(create_category_label)
        )
    else:
        lda_windows = (
            hid_act.groupby("filename", group_keys=False, sort=False)
            .apply(take_last2rep_phone_window)
            .reset_index(drop=True)
        )

        act_flat = (
            lda_windows.groupby("filename", group_keys=False, sort=False)
            .apply(concat_activation_output_df)
            .reset_index(drop=True)
            .rename(columns={"act": "act_vec"})
            
        )
        act_flat['voice'] = np.where(act_flat['phone']=="B", "voiced", "voiceless")
        flat_with_label = act_flat.copy()



    # ---- train ----
    X = np.stack(flat_with_label["act_vec"].to_numpy())
    y = flat_with_label["voice"].to_numpy()
    assert len(X) == len(y)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    if model=="lda":
        pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lda", LinearDiscriminantAnalysis(solver="lsqr", shrinkage='auto')),
        ]
        )
    else:
        pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            solver="lbfgs",
            penalty="l2",
            max_iter=2000,
        ))
        ])

    
    pipeline.fit(X, y_enc)

    # ---- save model ----
    out_path = saved_path / f"decoding_pipeline_layer{layer}_{model}.joblib"
    joblib.dump({"model": pipeline, "label_encoder": le}, out_path)

    return out_path

def predict_model_for_layer(
    *,
    layer: int,
    model_path: Path,
    test_trial_path: Path,
):
    bundle = joblib.load(model_path)
    pipeline = bundle["model"]
    le = bundle["label_encoder"]

    test_df = pd.read_pickle(test_trial_path / f"lda_window_layer{layer}.pkl")

    test_concat = (
        test_df.groupby(["filename", "rep_start_pos"], group_keys=False)
        .apply(concat_activation_output_df)
        .reset_index()
    )

    test_concat["voice"] = np.where(test_concat["phone"] == "B", "voiced", "voiceless")

    X_test = np.stack(test_concat["act"].to_numpy())

    pred_int = pipeline.predict(X_test)
    test_concat["pred_voice"] = le.inverse_transform(pred_int)

    proba = pipeline.predict_proba(X_test)

    test_concat["pred_voice_prob"] = proba.max(axis=1)
    test_concat["layer"] = layer
    return test_concat
# %% run for all layers
saved_path = p.parent / "data/processed/adaptation_lda_experiment_data"
saved_path.mkdir(exist_ok=True, parents=True)

can_rec = Path(p.parent / "adaptation_exps/stimuli/canonical")
time_df = pd.read_pickle(can_rec / "canonical_time_annotated.pkl")
test_trial_path = p.parent / "adaptation_exps/embeddings/wav2vec2_before_lda"

pred_dfs = []

for layer in range(1, 13):
    model_path = train_model_for_layer(
        layer=layer,
        time_df=time_df,
        saved_path=saved_path,
        hid_act_folder="../adaptation_exps/canonical_embedding/wav2vec2_base",
        hid_pkl_folder="../adaptation_exps/canonical_embedding/wav2vec2_base_pkl",
        num_layers=12,
        model="logreg",
        training_data="experiment"
    )

    pred_df = predict_model_for_layer(
        layer=layer,
        model_path=model_path,
        test_trial_path=test_trial_path,
    )
    pred_dfs.append(pred_df)

all_preds = pd.concat(pred_dfs, ignore_index=True)
all_preds.to_csv(saved_path / "lda_model_pred.csv")

# %%
