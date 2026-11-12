!/usr/bin/env bash

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

cd "$REPO_ROOT/src"


# python get_wav2vec_embeddings.py \
#   --audio-df "$REPO_ROOT/adaptation_exps/stimuli/cue_reweighting_recordings/cue_reweighting_stimuli_audio_df.pkl" \
#   --output-dir "$REPO_ROOT/adaptation_exps/embeddings/wav2vec2_base" \
#   --model facebook/wav2vec2-base \
#   --processor facebook/wav2vec2-base


# python get_wav2vec_embeddings.py \
#   --audio-df "$REPO_ROOT/adaptation_exps/stimuli/librispeech_bp_words/single_word_audio_df.pickle" \
#   --output-dir "$REPO_ROOT/adaptation_exps/single_word_embeddings/wav2vec2_base" \
#   --model facebook/wav2vec2-base \
#   --processor facebook/wav2vec2-base


python get_wav2vec_embeddings.py \
  --audio-df "$REPO_ROOT/adaptation_exps/stimuli/canonical/canonical_words_df.pkl" \
  --output-dir "$REPO_ROOT/adaptation_exps/canonical_embedding/wav2vec2_base" \
  --model facebook/wav2vec2-base \
  --processor facebook/wav2vec2-base
