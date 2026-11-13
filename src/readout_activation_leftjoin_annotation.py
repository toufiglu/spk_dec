"""CLI: read out per-layer activations and left-join the phone/acoustic table.

Thin wrapper around ``activations.read_activation_and_left_join_with_acoustics``
(the single implementation). Builds one ``hidden_acts_<layer>.pkl`` per layer.
"""
import argparse

import pandas as pd

from activations import read_activation_and_left_join_with_acoustics


def main():
    parser = argparse.ArgumentParser(description="read out the activation and left join with phone df")
    parser.add_argument("--input_folder", type=str, required=True, help="location of hidden activation folder (input)")
    parser.add_argument("--output_folder", type=str, required=True, help="location of output pkl files")
    parser.add_argument("--nlayer", type=int, required=True, help="How many layers in this model")
    parser.add_argument("--which_layer", type=int, required=True, help="Which layer should we read out?")
    parser.add_argument("--loc_phone_df", type=str, required=True, help="Location of the phone/formant df")
    parser.add_argument("--overwrite", action="store_true", help="Rebuild existing layer pickle files")

    args = parser.parse_args()
    phone_df = pd.read_pickle(args.loc_phone_df)

    read_activation_and_left_join_with_acoustics(
        hid_act_folder=args.input_folder,
        hid_pkl_folder=args.output_folder,
        num_layers=args.nlayer,
        which_layer=args.which_layer,
        formant_df=phone_df,
        dont_read_again=not args.overwrite,
    )


if __name__ == "__main__":
    main()
