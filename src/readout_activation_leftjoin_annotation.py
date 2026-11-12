import os
import pandas as pd
import torch
import numpy as np
import argparse

def read_activation_and_left_join_with_acoustics(hid_act_folder, hid_pkl_folder, num_layers, which_layer, phone_df, overwrite):
    os.makedirs(hid_pkl_folder, exist_ok=True)

    if not 1 <= which_layer <= num_layers:
        raise ValueError("the layer number is not valid. check again.")
     
    expected_pkls = [os.path.join(hid_pkl_folder, f"hidden_acts_{layer}.pkl") for layer in range(1, num_layers + 1)]

    target_pkl = os.path.join(hid_pkl_folder, f"hidden_acts_{which_layer}.pkl")   
    
    if all(os.path.exists(pkl) for pkl in expected_pkls) and not overwrite:
        print("All layer pickle files found, will load the target layer.")
        acts = pd.read_pickle(target_pkl)
        return acts

    else:
        phone_idx = phone_df.set_index(["filename", "time"]) # vowel formant df is left-joined to the hidden activation df

        for layer in range(1, num_layers + 1): 
            layer_pkl = os.path.join(hid_pkl_folder, f"hidden_acts_{layer}.pkl")
            if os.path.exists(layer_pkl) and not overwrite:
                print(f"Layer {layer} already exists — skipping.")
                continue
            else: 
                folder = os.path.join(hid_act_folder, f"layer_{layer:02d}")
                
                files = sorted(f for f in os.listdir(folder) if f.endswith(".pt"))

                if not files:
                    raise FileNotFoundError(f"No .pt files found in {folder}")

                all_files_act_df = []
                for idx, file in enumerate(files, start=1):
                    file_path = os.path.join(folder, file)
                    filename = os.path.splitext(file)[0]
                    file_act = torch.load(file_path, map_location="cpu")

                    T, D = file_act.shape
                    times = np.arange(T, dtype=float)*0.02 # every 20 ms get a frame so 0.02
                    act = file_act.cpu().numpy()
                    
                    file_act_df = pd.DataFrame({
                        "filename": filename,
                        "time": times,
                        "act": list(act),
                    })

                    file_act_with_phone = file_act_df.join(phone_idx,
                                                           on=["filename", "time"], 
                                                           how='left',
                                                           validate="many_to_one")

                    all_files_act_df.append(file_act_with_phone)

                    print(f"Layer {layer}: file {idx}/{len(files)} completed")
                
                hid_act_df = pd.concat(all_files_act_df, ignore_index=True)

                hid_act_df.to_pickle(layer_pkl)

                print(f"Layer {layer} is done")

        acts = pd.read_pickle(target_pkl)

    return acts

def main():
    parser = argparse.ArgumentParser(description="read out the activation and left join with phone df")

    parser.add_argument("--input_folder", type=str, required=True, help="location of hidden activation folder (input)")

    parser.add_argument("--output_folder", type=str, required=True, help="location of output pkl files")

    parser.add_argument("--nlayer", type=int, required=True, help="How many layers in this model")

    parser.add_argument("--which_layer", type=int, required=True, help="Which layer should we read out?")

    parser.add_argument("--loc_phone_df", type=str, required=True, help="Location of the phone df")

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild existing layer pickle files"
    )

    args = parser.parse_args()
    phone_df = pd.read_pickle(args.loc_phone_df)


    read_activation_and_left_join_with_acoustics(hid_act_folder=args.input_folder,
                                                 hid_pkl_folder=args.output_folder, 
                                                 num_layers=args.nlayer, 
                                                 which_layer=args.which_layer,
                                                 phone_df=phone_df,
                                                 overwrite = args.overwrite)

if __name__ == "__main__":
    main()

