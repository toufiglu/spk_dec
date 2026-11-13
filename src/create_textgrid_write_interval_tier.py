"""Write one sentence-level TextGrid per utterance, in preparation for MFA.

For every ``.wav`` under the corpus root this creates a matching
``<stem>.TextGrid`` containing a single interval tier ("Sentence") that spans
the whole file and holds the LibriSpeech transcription. Montreal Forced Aligner
consumes these to produce the per-phone alignment used by ``get_phone_tier.py``.

Trivial, but it is a real pipeline step: it sits between
``convert_flac_to_wav.sh`` and MFA alignment. Run from inside ``src/``:

    python create_textgrid_write_interval_tier.py --corpus-root ../data/raw/dev-clean

Idempotent by default: existing TextGrids are left alone (use --overwrite to
rebuild them).
"""
import argparse
import glob
import os

from praatio import textgrid
from praatio.audio import getDuration

from acoustics import read_transcript_map


def transcript_lookup(rec_dir):
    """filename-stem -> transcription for one recording folder."""
    df = read_transcript_map(rec_dir)
    return dict(zip(df["FILENAME"], df["SENTENCE"]))


def write_sentence_textgrids(corpus_root, overwrite=False):
    created = skipped = 0
    for speaker in sorted(os.listdir(corpus_root)):
        speaker_dir = os.path.join(corpus_root, speaker)
        if not os.path.isdir(speaker_dir):
            continue

        for rec_folder in sorted(os.listdir(speaker_dir)):
            rec_dir = os.path.join(speaker_dir, rec_folder)
            if not os.path.isdir(rec_dir):
                continue

            trans_map = transcript_lookup(rec_dir)

            for wav in sorted(glob.glob(os.path.join(rec_dir, "*.wav"))):
                stem = os.path.splitext(os.path.basename(wav))[0]
                tg_path = os.path.join(rec_dir, f"{stem}.TextGrid")
                if os.path.exists(tg_path) and not overwrite:
                    skipped += 1
                    continue

                transcription = trans_map.get(stem, "")
                if not transcription:
                    print(f"[warn] no transcription for {stem}; writing empty tier")

                dur = getDuration(wav)
                tg = textgrid.Textgrid()
                tier = textgrid.IntervalTier(
                    "Sentence", [(0.0, dur, transcription)], 0.0, dur)
                tg.addTier(tier)
                tg.save(tg_path, format="long_textgrid", includeBlankSpaces=False)
                created += 1

    print(f"Done. Wrote {created} TextGrid(s), skipped {skipped} existing.")


def main():
    parser = argparse.ArgumentParser(
        description="Write sentence-tier TextGrids for MFA alignment.")
    parser.add_argument("--corpus-root", required=True,
                        help="Root of <speaker>/<chapter>/*.wav (+ *.trans.txt).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Rebuild TextGrids that already exist.")
    args = parser.parse_args()
    write_sentence_textgrids(args.corpus_root, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
