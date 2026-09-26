"""
Prepare Vaani wake-word dataset for 3-class KWS training (silence/unknown/vaani).

Organizes custom Vaani recordings (<speaker_id>/*.wav) into training, validation,
and testing splits with speaker holdout, and pulls 'unknown' and 'silence' classes
from pre-processed Google Speech Commands data (./data/processed).

Usage:
    python prepare_vaani_data.py --vaani_dir ./data/vaani_raw \
                                 --out_dir ./data/vaani_processed \
                                 --held_out_speakers speaker_a speaker_b
"""

import argparse
import os
import random
import shutil
import sys
from pathlib import Path


def link_or_copy(src: Path, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    try:
        os.symlink(src.resolve(), dest)
    except OSError:
        try:
            os.link(src.resolve(), dest)
        except OSError:
            shutil.copy(src, dest)


def pull_negative_classes(sc_dir: Path, out_dir: Path, classes=("unknown", "silence")):
    """Pulls existing unknown and silence samples from Speech Commands processed data."""
    if not sc_dir.exists():
        print(f"WARNING: Speech Commands directory '{sc_dir}' not found.")
        print("Run 'python data_prep.py --data_dir ./data' first to generate silence/unknown data.")
        return 0

    n_pulled = 0
    for split in ("training", "validation", "testing"):
        split_dir = sc_dir / split
        if not split_dir.exists():
            continue
        for label in classes:
            label_dir = split_dir / label
            if not label_dir.exists():
                continue
            dest_label_dir = out_dir / split / label
            for wav in label_dir.glob("*.wav"):
                dest = dest_label_dir / wav.name
                link_or_copy(wav, dest)
                n_pulled += 1
    return n_pulled


def print_summary(out_dir: Path):
    print("\nDataset summary:")
    for split in ("training", "validation", "testing"):
        split_dir = out_dir / split
        if split_dir.exists():
            counts = {
                d.name: len(list(d.glob("*.wav")))
                for d in sorted(split_dir.iterdir())
                if d.is_dir()
            }
            print(f"  {split}: {counts}")


def main(args):
    vaani_dir = Path(args.vaani_dir)
    out_dir = Path(args.out_dir)
    sc_dir = Path(args.speech_commands_dir)

    if not vaani_dir.exists():
        print(f"Vaani directory '{vaani_dir}' does not exist yet.")
        print("Please place your Vaani recordings into this directory organized as:")
        print(f"    {vaani_dir}/<speaker_id>/*.wav")
        print("Then re-run this script to build the 3-class dataset.")
        sys.exit(0)

    speaker_dirs = sorted([d for d in vaani_dir.iterdir() if d.is_dir()])
    if not speaker_dirs:
        print(f"No speaker directories found inside '{vaani_dir}'.")
        print("Expected structure: <vaani_dir>/<speaker_id>/*.wav")
        sys.exit(0)

    all_speakers = [d.name for d in speaker_dirs]
    held_out = set(args.held_out_speakers) if args.held_out_speakers else set()
    missing_held_out = held_out - set(all_speakers)
    if missing_held_out:
        print(f"WARNING: Held-out speakers not found in data: {missing_held_out}")
        held_out = held_out & set(all_speakers)

    remaining = [s for s in all_speakers if s not in held_out]
    random.seed(args.seed)
    random.shuffle(remaining)

    if held_out:
        test_speakers = held_out
    else:
        n_test = max(1, int(len(remaining) * args.test_fraction)) if len(remaining) > 1 else 0
        test_speakers = set(remaining[:n_test])
        remaining = remaining[n_test:]

    n_val = max(1, int(len(remaining) * args.val_fraction)) if len(remaining) > 1 else 0
    val_speakers = set(remaining[:n_val])
    train_speakers = set(remaining[n_val:])

    print(f"Speakers -- training: {sorted(train_speakers)}")
    print(f"Speakers -- validation: {sorted(val_speakers)}")
    print(f"Speakers -- testing (unseen / held-out): {sorted(test_speakers)}")

    def get_split(speaker_id: str) -> str:
        if speaker_id in test_speakers:
            return "testing"
        if speaker_id in val_speakers:
            return "validation"
        return "training"

    # 1. Process Vaani recordings
    out_dir.mkdir(parents=True, exist_ok=True)
    n_vaani = 0
    for speaker_dir in speaker_dirs:
        split = get_split(speaker_dir.name)
        dest_dir = out_dir / split / "vaani"
        wav_files = list(speaker_dir.glob("*.wav"))
        if not wav_files:
            wav_files = list(speaker_dir.rglob("*.wav"))
        for wav in wav_files:
            dest = dest_dir / f"{speaker_dir.name}_{wav.name}"
            link_or_copy(wav, dest)
            n_vaani += 1

    print(f"\nLinked/copied {n_vaani} Vaani samples into {out_dir}")

    # 2. Pull unknown and silence classes from processed Speech Commands data
    print(f"Pulling 'unknown' and 'silence' from '{sc_dir}'...")
    n_neg = pull_negative_classes(sc_dir, out_dir, classes=("unknown", "silence"))
    print(f"Linked/copied {n_neg} negative samples (unknown/silence).")

    print_summary(out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Vaani 3-class dataset.")
    parser.add_argument("--vaani_dir", type=str, required=True,
                        help="Path to Vaani recordings organized as <speaker_id>/*.wav")
    parser.add_argument("--speech_commands_dir", type=str, default="./data/processed",
                        help="Path to existing processed Speech Commands data (source for unknown/silence)")
    parser.add_argument("--out_dir", type=str, default="./data/vaani_processed",
                        help="Output directory for the 3-class dataset")
    parser.add_argument("--held_out_speakers", nargs="+", default=[],
                        help="Speaker IDs to exclude entirely from train/val, used only for testing")
    parser.add_argument("--val_fraction", type=float, default=0.2,
                        help="Fraction of non-held-out speakers for validation")
    parser.add_argument("--test_fraction", type=float, default=0.1,
                        help="Fraction of speakers for test if --held_out_speakers not specified")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
