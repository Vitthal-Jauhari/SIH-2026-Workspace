"""
Phase 3 - Step 3: Split augmented speaker recordings into train/validation/
testing by SPEAKER, not by clip -- some speakers are held out completely so
"testing" measures generalization to a genuinely unseen voice, matching
Phase 1's methodology but applied to your real speakers instead of the
Speech Commands crowd.

Output layout matches Phase 1's `processed/` folder exactly (training/
validation/testing/<label>/*.wav), so you can point train.py at either one,
or merge them (see merge_with_phase1.py) to train on everything at once.

Run:
    python speaker_split.py --in_dir ./data/speakers_augmented \
                             --out_dir ./data/speakers_processed \
                             --held_out_speakers charlie dana
"""

import argparse
import random
import shutil
from pathlib import Path


def main(args):
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)

    all_speakers = sorted([d.name for d in in_dir.iterdir() if d.is_dir()])
    held_out = set(args.held_out_speakers)
    missing = held_out - set(all_speakers)
    if missing:
        print(f"WARNING: held-out speakers not found in data: {missing}")

    remaining = [s for s in all_speakers if s not in held_out]
    random.seed(args.seed)
    random.shuffle(remaining)

    n_val = max(1, int(len(remaining) * args.val_fraction)) if remaining else 0
    val_speakers = set(remaining[:n_val])
    train_speakers = set(remaining[n_val:])

    print(f"Speakers -- train: {sorted(train_speakers)}")
    print(f"Speakers -- validation: {sorted(val_speakers)}")
    print(f"Speakers -- testing (fully unseen): {sorted(held_out)}")

    def speaker_split(speaker_id: str) -> str:
        if speaker_id in held_out:
            return "testing"
        if speaker_id in val_speakers:
            return "validation"
        return "training"

    n_copied = 0
    for speaker_dir in in_dir.iterdir():
        if not speaker_dir.is_dir():
            continue
        split = speaker_split(speaker_dir.name)
        for word_dir in speaker_dir.iterdir():
            if not word_dir.is_dir():
                continue
            label = word_dir.name
            dest_dir = out_dir / split / label
            dest_dir.mkdir(parents=True, exist_ok=True)
            for wav in word_dir.glob("*.wav"):
                dest = dest_dir / f"{speaker_dir.name}_{wav.name}"
                if not dest.exists():
                    shutil.copy(wav, dest)
                    n_copied += 1

    print(f"\nCopied {n_copied} files into {out_dir}")
    for split in ("training", "validation", "testing"):
        split_dir = out_dir / split
        if split_dir.exists():
            counts = {d.name: len(list(d.glob("*.wav"))) for d in split_dir.iterdir()}
            print(f"  {split}: {counts}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--held_out_speakers", nargs="+", default=[],
                         help="Speaker IDs to exclude entirely from train/val, "
                              "used only as the unseen-speaker test set")
    parser.add_argument("--val_fraction", type=float, default=0.2,
                         help="Fraction of the remaining (non-held-out) speakers "
                              "to use for validation")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
