"""Minimal example: extract a single hidden-state layer for one utterance.

Loads one LibriSpeech dev-clean utterance, runs it through a pretrained
speech model (WavLM by default), and saves one layer's hidden states to
``emb.pt``. See ``extract_embeddings.py`` for the full batched pipeline.
"""
import os

import torch
import librosa
from transformers import AutoFeatureExtractor, WavLMModel

# Pick the best available device. (Note: the original version of this file had
# the cuda/mps branches inverted and a bogus 'gpu' fallback; fixed here.)
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

# Path relative to the project root; populate data/ via scripts/download_data.sh
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
audio_file_path = os.path.join(
    REPO_ROOT, "data", "raw", "dev-clean", "1272", "128104", "1272-128104-0000.wav"
)
audio, sr = librosa.load(audio_file_path, sr=16000)

model_name = "microsoft/wavlm-base"
extractor = AutoFeatureExtractor.from_pretrained(model_name)
model = WavLMModel.from_pretrained(model_name)
model.eval().to(device)

inputs = extractor(audio, sampling_rate=16000, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.no_grad():
    output = model(**inputs, output_hidden_states=True)

hidden_states = output.hidden_states[1]
torch.save(hidden_states.cpu(), "emb.pt")
