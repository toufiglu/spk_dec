"""Load per-layer model activations and left-join acoustic annotation.

Single home for ``read_activation_and_left_join_with_acoustics``: imported by the
decoders (decoding.py, spec_idx.py, ffn_talker_decoding.py) and wrapped as a CLI
by readout_activation_leftjoin_annotation.py.
"""
import os

import numpy as np
import pandas as pd
import torch


def read_activation_and_left_join_with_acoustics(hid_act_folder, hid_pkl_folder, num_layers,
                                                  which_layer, formant_df, dont_read_again):
    os.makedirs(hid_pkl_folder, exist_ok=True)

    if not 1 <= which_layer <= num_layers:
        raise ValueError("the layer number is not valid. check again.")

    expected_pkls = [os.path.join(hid_pkl_folder, f"hidden_acts_{layer}.pkl")
                     for layer in range(1, num_layers + 1)]
    target_pkl = os.path.join(hid_pkl_folder, f"hidden_acts_{which_layer}.pkl")

    if all(os.path.exists(pkl) for pkl in expected_pkls) and dont_read_again:
        print("All layer pickle files found, will load the target layer.")
        return pd.read_pickle(target_pkl)

    formant_idx = formant_df.set_index(["filename", "time"])  # left-joined onto the activation df

    for layer in range(1, num_layers + 1):
        layer_pkl = os.path.join(hid_pkl_folder, f"hidden_acts_{layer}.pkl")
        if os.path.exists(layer_pkl) and dont_read_again:
            print(f"Layer {layer} already exists — skipping.")
            continue

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
            times = np.arange(T, dtype=float) * 0.02  # every 20 ms -> one frame
            act = file_act.cpu().numpy()

            file_act_df = pd.DataFrame({
                "filename": filename,
                "time": times,
                "act": list(act),
            })
            file_act_with_phone = file_act_df.join(
                formant_idx, on=["filename", "time"], how="left", validate="many_to_one")
            all_files_act_df.append(file_act_with_phone)

            print(f"Layer {layer}: file {idx}/{len(files)} completed")

        hid_act_df = pd.concat(all_files_act_df, ignore_index=True)
        hid_act_df.to_pickle(layer_pkl)
        print(f"Layer {layer} is done")

    return pd.read_pickle(target_pkl)
