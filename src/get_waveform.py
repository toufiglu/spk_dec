import os
import pandas as pd
import glob
import soundfile as sf
import argparse

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

def get_waveform_and_gammatone(ROOT, save_dir, dont_run_again=True): 
    speakers = [f for f in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, f))]
    if dont_run_again and os.path.exists(save_dir):
        audio_df = pd.read_pickle(save_dir)
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
                    AUDIO, _ = sf.read(wav, dtype="float32") # read the sound file using sf, float32
                    tmp.append({'FILENAME': FILENAME, "AUDIO": AUDIO})

                tmp=pd.DataFrame(tmp)

                audio_df = pd.merge(trans_map, tmp, how="left", on="FILENAME")
                audio_df_list.append(audio_df)
            print(f"Speaker {speaker} is done. This is {idx} out of {len(speakers)} speakers.")

        audio_df = pd.concat(audio_df_list, ignore_index=True)
        audio_df.to_pickle(save_dir)

    return audio_df

def main():
    parser = argparse.ArgumentParser(description="get waveform from audio and create a df")

    parser.add_argument("--fname", type=str, required=True, help="Root folder of the sound files")
    parser.add_argument("--repeat", type=bool, required=True, help="Don't run again?")
    parser.add_argument("--save_dir", type=str, required=True, help="filename of the saved output")

    args = parser.parse_args()

    get_waveform_and_gammatone(ROOT=args.fname, save_dir= args.save_dir, dont_run_again=args.repeat)

if __name__ == "__main__":
    main()





