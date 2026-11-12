#!/usr/bin/env bash
#
# Download the raw speech corpus used by this project: LibriSpeech dev-clean,
# from OpenSLR (https://www.openslr.org/12/). Roughly 337 MB download.
#
# Usage:
#   bash scripts/download_data.sh
#
# After downloading, LibriSpeech ships as .flac. The pipeline expects 16 kHz
# .wav alongside Montreal Forced Aligner (MFA) TextGrids. To reproduce the
# exact data/raw/dev-clean layout:
#   1. Convert .flac -> .wav   (see experiments/adaptation/convert_flac_to_wav.sh)
#   2. Force-align with MFA     (produces the per-utterance .TextGrid files)
# See README.md for details.

set -euo pipefail

URL="https://www.openslr.org/resources/12/dev-clean.tar.gz"
RAW_DIR="data/raw"
TARBALL="${RAW_DIR}/dev-clean.tar.gz"

mkdir -p "${RAW_DIR}"

if [ ! -f "${TARBALL}" ]; then
  echo "Downloading LibriSpeech dev-clean from OpenSLR ..."
  curl -L -o "${TARBALL}" "${URL}"
else
  echo "Tarball already present: ${TARBALL}"
fi

echo "Extracting ..."
tar -xzf "${TARBALL}" -C "${RAW_DIR}"

# OpenSLR extracts to data/raw/LibriSpeech/dev-clean; flatten to data/raw/dev-clean
if [ -d "${RAW_DIR}/LibriSpeech/dev-clean" ]; then
  mv "${RAW_DIR}/LibriSpeech/dev-clean" "${RAW_DIR}/dev-clean"
  rmdir "${RAW_DIR}/LibriSpeech" 2>/dev/null || true
fi

echo "Done. Raw audio is in ${RAW_DIR}/dev-clean (as .flac)."
echo "Next: convert to .wav and run MFA alignment (see README.md)."
