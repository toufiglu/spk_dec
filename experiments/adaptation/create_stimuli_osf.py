from pathlib import Path
import random
import subprocess
import sys

def run(cmd):
    subprocess.run(cmd, check=True)

def main(folder, out_wav="combined.wav", gap_sec=1.0, seed=None):
    folder = Path(folder).expanduser().resolve()

    mp3s = sorted(folder.glob("*.mp3"))
    random.seed(seed)

    # Convert to WAVs in a subfolder
    wav_dir = folder / "_wav_converted"
    wav_dir.mkdir(exist_ok=True)

    converted = []
    for mp3 in mp3s:
        wav = wav_dir / (mp3.stem + ".wav")
        # Force same format for all files (important for concat-copy)
        run([
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(mp3),
             "-ar", "44100", "-c:a", "pcm_s16le",
            str(wav)
        ])
        converted.append(wav)

    random.shuffle(converted)

    silence = wav_dir / f"_silence_{gap_sec:.3f}s.wav"
    run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
        "-t", str(gap_sec),
        "-c:a", "pcm_s16le",
        str(silence)
    ])

    list_file = wav_dir / "_concat_list.txt"
    with list_file.open("w", encoding="utf-8") as f:
        for i, wav in enumerate(converted):
            f.write(f"file '{wav.as_posix()}'\n")
            if i != len(converted) - 1:  # no trailing silence after last track
                f.write(f"file '{silence.as_posix()}'\n")

    # Concatenate
    out_path = Path(out_wav).expanduser().resolve()
    run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "concat", "-safe", "0",
        "-i", str(list_file),
        "-c", "copy",
        str(out_path)
    ])

    print("Shuffled order:")
    for p in converted:
        print(" -", p.name)
    print("\nWrote:", out_path)

if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else "."
    out_wav = sys.argv[2] if len(sys.argv) > 2 else "combined.wav"
    gap = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else None
    main(folder, out_wav=out_wav, gap_sec=gap, seed=seed)
