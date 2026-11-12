#!/usr/bin/env bash


outdir=../stimuli/beer_pier_wav
mkdir -p "$outdir"

for f in ../stimuli/beer_pier/*.flac; do
  sox "$f" -r 16000 "$outdir/$(basename "${f%.flac}").wav"
done



