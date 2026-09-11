"""
Phase 3 - Step 1: Record "yes"/"no" from multiple real speakers.

(silence/unknown are already well covered by Speech Commands + background
noise from Phase 1, so we only need real recordings of the target words
here -- that's where a synthetic/public-dataset-only model is weakest.)

Saves to:
    data/speakers/<speaker_id>/<word>/<speaker_id>_<word>_<NNN>.wav

Run once per speaker (it'll ask for their ID so you can run this on
different people's machines, or just re-run locally and change --speaker_id):

    python record_speakers.py --speaker_id alice --reps 15
"""

import argparse
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

SAMPLE_RATE = 16000
CLIP_SECONDS = 1.0
WORDS = ["yes", "no"]


def record_clip(seconds=CLIP_SECONDS) -> np.ndarray:
    audio = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                    channels=1, dtype="float32")
    sd.wait()
    return audio[:, 0]


def main(args):
    out_root = Path(args.data_dir) / "speakers" / args.speaker_id
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Recording for speaker '{args.speaker_id}'. "
          f"{args.reps} reps x {len(WORDS)} words. Ctrl+C to abort.\n")

    for word in WORDS:
        word_dir = out_root / word
        word_dir.mkdir(parents=True, exist_ok=True)
        existing = len(list(word_dir.glob("*.wav")))

        for i in range(args.reps):
            idx = existing + i
            input(f"[{word}] rep {i+1}/{args.reps} -- press Enter, then say '{word}' "
                  f"clearly within {CLIP_SECONDS}s...")
            time.sleep(0.15)  # small buffer so the Enter keypress sound isn't captured
            audio = record_clip()
            path = word_dir / f"{args.speaker_id}_{word}_{idx:03d}.wav"
            sf.write(path, audio, SAMPLE_RATE)
            peak = np.max(np.abs(audio))
            level_flag = " <- quiet, consider re-recording" if peak < 0.05 else ""
            print(f"  saved {path.name}  (peak level={peak:.3f}){level_flag}")

    print(f"\nDone. Recordings under {out_root}")
    print("Repeat this script with a different --speaker_id for each of your "
          "~8 speakers. Vary distance from the mic, pace, and volume across "
          "speakers/sessions -- that variation is the point of Phase 3.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--speaker_id", type=str, required=True,
                         help="Short unique name/id for this speaker, e.g. 'alice'")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--reps", type=int, default=15,
                         help="Repetitions per word for this speaker")
    args = parser.parse_args()
    main(args)
