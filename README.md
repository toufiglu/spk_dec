# Talker Decoding

Probing how self-supervised speech models (wav2vec 2.0, WavLM, HuBERT, Whisper)
represent **talker identity** vs. **phonetic content** across their layers.
The pipeline extracts per-layer hidden states for LibriSpeech utterances, aligns
them with phone- and formant-level acoustic annotations, and trains linear /
FFN decoders plus RSA and representational-rotation analyses.

## Repository layout

```
talker-decoding/
├── README.md              This file
├── environment.yml        Conda environment (Python 3.12 + torch/transformers/sklearn)
├── .gitignore
├── scripts/
│   └── download_data.sh   Fetch LibriSpeech dev-clean from OpenSLR
├── src/                   Python pipeline (run from inside this directory)
├── R/                     R analysis (KLD visualization, helpers)
├── experiments/
│   └── adaptation/        Adaptation / cue-reweighting stimulus + embedding code
├── results/               Small tracked result tables (CSV)
└── data/                  Git-ignored; created by download_data.sh + the pipeline
    ├── raw/               LibriSpeech dev-clean (.wav + MFA TextGrids)
    └── processed/         Hidden states, decoders, intermediate tables
```

Large, regenerable artifacts are intentionally **not** stored in this repo:
the raw corpus is downloaded, and all hidden states, embeddings, decoder
coefficients, and figures are produced by running the code.

## Setup

```bash
conda env create -f environment.yml
conda activate audio
```

## Getting the data

The project uses **LibriSpeech dev-clean** (https://www.openslr.org/12/).

```bash
bash scripts/download_data.sh
```

LibriSpeech ships as `.flac`. The pipeline expects 16 kHz `.wav` plus
Montreal Forced Aligner (MFA) `.TextGrid` files per utterance. To reproduce
the `data/raw/dev-clean/<speaker>/<chapter>/*.wav + *.TextGrid` layout:

1. Convert `.flac` → `.wav` (see `experiments/adaptation/convert_flac_to_wav.sh`).
2. Force-align with MFA to generate the `.TextGrid` files.

## Running the pipeline

> **Convention:** the `src/` scripts use bare imports (`import helpers`) and
> write intermediate files to paths relative to the working directory, so run
> them from **inside `src/`** (`cd src`). Processed outputs land under the
> project's `data/processed/` tree.

Rough order:

1. **Audio & acoustics** — `get_waveform.py`, `get_gammatone_and_decode.py`
   build waveforms, gammatone filterbanks, and formant tables.
2. **Phone annotation** — `get_phone_tier.py`,
   `create_textgrid_write_interval_tier.py` turn MFA TextGrids into
   20 ms-frame phone labels.
3. **Embeddings** — `extract_embeddings.py` (full batched extraction) and
   `model_embedding.py` (single-utterance example) dump per-layer hidden
   states for wav2vec 2.0 / WavLM / HuBERT / Whisper.
4. **Join** — `readout_activation_leftjoin_annotation.py` left-joins
   activations with phone/acoustic annotations.
5. **Decoding** — `direct_talker_decoding.py` (talker AUC by layer),
   `phoneme_linear_readout.py` (phone logistic readout),
   `cosine_phone_talker_decoders.py` (talker vs. phone decoder geometry),
   `ffn_decoding.py` / `FFN_and_forward_hook.py` (FFN decoding & activation hooks).
6. **Analysis** — `rsa_ratio.py` (RSA), `rotate_hidden_representation.py`
   (representational rotation), `distribution_visualization.py`,
   `select_words.py`, `get_individual_vowels.py`,
   `get_vowel_plus_previous_consonant.py`.

### R analysis

`R/visualize_kld.Rmd` (with `R/helpers.R`) produces the KLD-related
visualizations from the exported result tables.

## External tools

- **Montreal Forced Aligner (MFA)** — phone-level forced alignment.
- **Dr.VOT** (https://github.com/MLSpeech/Dr.VOT) — used in the adaptation
  experiments for VOT measurement. Install it separately; it is not vendored
  here (it ships large model checkpoints and a `praat` binary).
