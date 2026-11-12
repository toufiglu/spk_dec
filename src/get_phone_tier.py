
import os
import pandas as pd
import numpy as np
import glob
import soundfile as sf
import pickle
import parselmouth
import argparse
from praatio import textgrid


def label_at_time(time, entries):
    for start, end, label, rep in entries:
        if start <= time <= end:
            return label, rep
        
def get_phone(sound_folder, phone_fname):
    '''
    after forced alignment, read out the df with phone annotation (every 20 ms)
    '''
    f_path_name = f"./{phone_fname}"
    if os.path.exists(f_path_name):
        with open(f_path_name, "rb") as f:
            phone_df=pickle.load(f) 
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

                    # get phone tier annotation
                    audio = parselmouth.Sound(wav)
                    obj = audio.to_formant_burg(time_step=0.02, # in keeping with the sampling rate of the ssl model, we also used 20 ms here.
                                        max_number_of_formants=5, # select 5 formants but later use only the first 3.
                                        maximum_formant=5500, # female max formant is suited to be 5500, we keep it this way for all speakers
                                        window_length=0.025) 
                    n_frame = obj.get_number_of_frames()
                    # make a tuple that store all info
                    row = []
                    t0 = obj.get_start_time()
                    dt = obj.get_time_step()
                    times = t0 + dt*np.arange(n_frame)
                    for time in times:
                        phone, rep = label_at_time(time, intervals)
                        row.append((time, phone, rep, speaker, filename))
                    df = pd.DataFrame(row, columns=["time", "phone", "rep", "speaker", "filename"])
                    entire_df.append(df)
                print(f"The folder of {rec_dir} is sorted.")
        phone_df = pd.concat(entire_df, ignore_index=True) 
        phone_df.to_pickle(f_path_name)

    return phone_df

def main():
    parser = argparse.ArgumentParser(description="Get annotation from textgrid files after forced alignment")

    parser.add_argument("--sfolder", type=str, required=True, help="sound folder where the textgrid files are located.")

    parser.add_argument("--fname", type=str, required=True, help="the file name for the output.")

    args = parser.parse_args()

    get_phone(args.sfolder, args.fname)

if __name__ == "__main__":
    main()
