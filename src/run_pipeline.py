"""End-to-end driver for the talker-decoding pipeline.

Chains the four self-contained command-line stages into a single run:

    raw audio ──get_waveform──▶ audio_df.pkl
              ──extract_embeddings──▶ per-layer hidden states (*.pt)
              ──readout_activation_leftjoin_annotation──▶ per-layer joined *.pkl
              ──logistic_regression_talker_decoding──▶ talker-AUC-by-layer CSV + plot

PREREQUISITE — annotation table
    The decoding stage joins model activations against a per-frame acoustic
    annotation table (``--annotation``, a pickle) that carries, per
    (filename, time): ``phone``, ``speaker``, and the first three formants
    ``f1/f2/f3``. Producing it needs Montreal Forced Aligner TextGrids plus
    formant extraction; see ``get_phone_tier.py`` and ``acoustics.get_formant``.
    This driver does NOT run forced alignment.

Examples
    # See exactly what would run, without touching anything:
    python run_pipeline.py --audio-root ../data/raw/dev-clean \\
        --annotation ../data/processed/formant.pkl --dry-run

    # Run only the two acoustic/model stages:
    python run_pipeline.py --audio-root ../data/raw/dev-clean \\
        --annotation ../data/processed/formant.pkl --stages waveform,embeddings
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
ALL_STAGES = ["waveform", "embeddings", "readout", "decode", "spec_idx", "figure", "rsa"]


def build_stages(args):
    """Return [(name, argv, primary_output_path_or_None), ...] in run order."""
    py = sys.executable
    stages = {
        "waveform": (
            [py, "get_waveform.py",
             "--fname", args.audio_root,
             "--repeat", "True",
             "--save_dir", args.audio_df],
            args.audio_df,
        ),
        "embeddings": (
            [py, "extract_embeddings.py",
             "--audio-df", args.audio_df,
             "--output-dir", args.hidden_dir,
             "--model", args.model],
            os.path.join(args.hidden_dir, "layer_01"),
        ),
        "readout": (
            [py, "readout_activation_leftjoin_annotation.py",
             "--input_folder", args.hidden_dir,
             "--output_folder", args.readout_dir,
             "--nlayer", str(args.num_layers),
             # the readout builds every layer on first call; layer 1 triggers it
             "--which_layer", "1",
             "--loc_phone_df", args.annotation],
            os.path.join(args.readout_dir, "hidden_acts_1.pkl"),
        ),
        "decode": (
            [py,
             "logistic_regression_talker_decoding.py" if args.decoder == "logreg"
             else "ffn_talker_decoding.py",
             "--formant_df", args.annotation,
             "--hid-act-folder", args.hidden_dir,
             "--hid-pkl-folder", args.readout_dir,
             "--model", args.model,
             "--storage", args.storage,
             "--plot_dir", args.plot,
             "--out-csv", args.out_csv],
            args.out_csv,
        ),
        # Speaker-specificity index (KLD ratio) by layer.
        "spec_idx": (
            [py, "compute_spec_idx.py",
             "--formant_df", args.annotation,
             "--hid-act-folder", args.hidden_dir,
             "--num-layers", str(args.num_layers),
             "--out-dir", args.spec_dir],
            os.path.join(args.spec_dir, "layer_1_spec_idx.csv"),
        ),
        # Visualize the spec_idx just computed.
        "figure": (
            [py, "../docs/make_figure.py",
             "--spec-dir", args.spec_dir,
             "--out", args.figure],
            args.figure,
        ),
        # RSA of model layers vs. speaker-specific / marginal acoustics.
        "rsa": (
            [py, "compute_rsa.py",
             "--formant_df", args.annotation,
             "--hid-pkl-folder", args.readout_dir,
             "--num-layers", str(args.num_layers),
             "--out-dir", args.rsa_dir],
            os.path.join(args.rsa_dir, "spec_cors_across_layers.csv"),
        ),
    }
    return [(name, *stages[name]) for name in args.stages]


def main():
    p = argparse.ArgumentParser(
        description="Run the talker-decoding pipeline end to end.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--audio-root", required=True,
                   help="Root of 16 kHz .wav files (LibriSpeech dev-clean layout).")
    p.add_argument("--annotation", required=True,
                   help="Pickle with per-frame phone/speaker/f1-f3 annotation (prerequisite).")
    p.add_argument("--model", default="facebook/wav2vec2-base",
                   help="HF model name or path.")
    p.add_argument("--num-layers", type=int, default=12,
                   help="Transformer layers to read out (wav2vec2-base = 12).")
    p.add_argument("--decoder", choices=["logreg", "ffn"], default="logreg",
                   help="Talker decoder for the 'decode' stage: logistic regression or FFN.")

    # Intermediate / output artifacts (all under data/processed by default).
    proc = "../data/processed"
    p.add_argument("--audio-df", default=f"{proc}/audio_df.pkl")
    p.add_argument("--hidden-dir", default=f"{proc}/hidden_states")
    p.add_argument("--readout-dir", default=f"{proc}/hidden_acts_pkl")
    p.add_argument("--storage", default=f"{proc}/talker_decoding")
    # Default output names carry the decoder, so logreg/ffn runs don't collide.
    p.add_argument("--out-csv", default=None,
                   help=f"default: {proc}/talker_auc_<decoder>.csv")
    p.add_argument("--plot", default=None,
                   help=f"default: {proc}/talker_auc_<decoder>.png")
    p.add_argument("--spec-dir", default=f"{proc}/spec_idx",
                   help="Output dir for per-layer spec_idx CSVs.")
    p.add_argument("--figure", default=f"{proc}/spec_idx_across_layers.png",
                   help="Output path for the spec_idx visualization.")
    p.add_argument("--rsa-dir", default=f"{proc}/rsa",
                   help="Output dir for RSA correlation tables + figure.")

    p.add_argument("--stages", default=",".join(ALL_STAGES),
                   help="Comma-separated subset of: " + ",".join(ALL_STAGES))
    p.add_argument("--force", action="store_true",
                   help="Run stages even if their output already exists.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the commands without executing them.")
    args = p.parse_args()

    if args.out_csv is None:
        args.out_csv = f"{proc}/talker_auc_{args.decoder}.csv"
    if args.plot is None:
        args.plot = f"{proc}/talker_auc_{args.decoder}.png"

    args.stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    bad = [s for s in args.stages if s not in ALL_STAGES]
    if bad:
        p.error(f"unknown stage(s): {', '.join(bad)}; choose from {', '.join(ALL_STAGES)}")

    for name, argv, output in build_stages(args):
        exists = output is not None and os.path.exists(output)
        if exists and not args.force and not args.dry_run:
            print(f"[skip] {name}: output exists ({output})")
            continue

        printable = " ".join(argv)
        if args.dry_run:
            print(f"[dry-run] {name}:\n    {printable}")
            continue

        print(f"[run ] {name}:\n    {printable}")
        result = subprocess.run(argv, cwd=SRC_DIR)
        if result.returncode != 0:
            sys.exit(f"[fail] stage '{name}' exited with code {result.returncode}")

    print("Pipeline finished.")


if __name__ == "__main__":
    main()
