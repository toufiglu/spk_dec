#!/usr/bin/env bash
#
# Convert LibriSpeech .flac files to 16 kHz mono .wav in place, recursively,
# preserving the data/raw/dev-clean/<speaker>/<chapter>/ layout the pipeline
# expects. Safe to re-run: existing .wav files are skipped.
#
# Usage:
#   bash scripts/convert_flac_to_wav.sh [ROOT]
# ROOT defaults to data/raw/dev-clean (as produced by download_data.sh).
#
# Requires ffmpeg (or swap the conversion line for sox if you prefer).

set -euo pipefail

ROOT="${1:-data/raw/dev-clean}"

if [ ! -d "${ROOT}" ]; then
  echo "No such directory: ${ROOT}" >&2
  echo "Run scripts/download_data.sh first, or pass the audio root as an argument." >&2
  exit 1
fi

command -v ffmpeg >/dev/null 2>&1 || {
  echo "ffmpeg not found. Install it (e.g. 'conda install -c conda-forge ffmpeg') and retry." >&2
  exit 1
}

count=0
while IFS= read -r -d '' flac; do
  wav="${flac%.flac}.wav"
  if [ -f "${wav}" ]; then
    continue
  fi
  ffmpeg -nostdin -loglevel error -i "${flac}" -ar 16000 -ac 1 "${wav}"
  count=$((count + 1))
done < <(find "${ROOT}" -type f -name '*.flac' -print0)

echo "Done. Converted ${count} new .flac file(s) to 16 kHz mono .wav under ${ROOT}."
echo "Next: force-align with MFA to produce the per-utterance .TextGrid files (see README.md)."
