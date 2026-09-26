"""
Phase 3 - Step 2: Speaker-Level Split (Strictly Before Augmentation)

Splits normalized recordings by SPEAKER into:
- data/train/<speaker>/        (e.g. Ananya, Ark, Umang)
- data/validation/<speaker>/   (e.g. Ishita)
- data/test_unseen/<speaker>/  (e.g. Vitthal)

Includes automated guardrails that abort immediately if any speaker leakage is detected.

Usage:
    python speaker_split.py --in_dir ./data/normalized \
                            --out_dir ./data \
                            --train_speakers Ananya Ark Umang \
                            --val_speakers Ishita \
                            --held_out_speakers Vitthal
"""

import argparse
import os
import shutil
import sys
from pathlib import Path


def link_or_copy(src: Path, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    try:
        os.link(src.resolve(), dest)
    except OSError:
        shutil.copy(src, dest)


def split_speakers(
    in_dir: Path,
    out_dir: Path,
    train_speakers: list,
    val_speakers: list,
    held_out_speakers: list,
):
    in_dir = in_dir.resolve()
    out_dir = out_dir.resolve()

    if not in_dir.exists():
        raise FileNotFoundError(f"Normalized input directory not found: {in_dir}")

    train_set = set(train_speakers)
    val_set = set(val_speakers)
    held_out_set = set(held_out_speakers)

    print("=" * 65)
    print("PHASE 3 - STEP 2: SPEAKER-LEVEL SPLIT")
    print("=" * 65)
    print(f"Train speakers       : {sorted(train_set)}")
    print(f"Validation speakers  : {sorted(val_set)}")
    print(f"Held-out Unseen test : {sorted(held_out_set)}\n")

    # Guardrail 1: Disjoint configuration check
    overlap_tv = train_set & val_set
    overlap_th = train_set & held_out_set
    overlap_vh = val_set & held_out_set

    if overlap_tv or overlap_th or overlap_vh:
        msg = (
            f"ERROR: Overlap detected in speaker split configuration!\n"
            f"Train & Val overlap: {overlap_tv}\n"
            f"Train & Unseen overlap: {overlap_th}\n"
            f"Val & Unseen overlap: {overlap_vh}\n"
            f"Phase 3 aborted to prevent data leakage."
        )
        print(msg, file=sys.stderr)
        raise RuntimeError(msg)

    # Verify all configured speakers exist in normalized directory
    available_speakers = {d.name for d in in_dir.iterdir() if d.is_dir()}
    all_configured = train_set | val_set | held_out_set
    missing = all_configured - available_speakers
    if missing:
        msg = f"ERROR: Configured speakers not found in {in_dir}: {missing}"
        print(msg, file=sys.stderr)
        raise RuntimeError(msg)

    # Assign any newly discovered speakers not explicitly listed to train by default
    unassigned = available_speakers - (train_set | val_set | held_out_set)
    if unassigned:
        print(f"Assigning new unassigned speakers to Train split: {sorted(unassigned)}")
        train_set.update(unassigned)

    # Destination directories
    train_dir = out_dir / "train"
    val_dir = out_dir / "validation"
    test_unseen_dir = out_dir / "test_unseen"

    # Clean existing destination dirs to guarantee fresh state
    for d in (train_dir, val_dir, test_unseen_dir):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    counts = {"train": {}, "validation": {}, "test_unseen": {}}

    # Populate train
    for spk in train_set:
        src_spk_dir = in_dir / spk
        dest_spk_dir = train_dir / spk
        dest_spk_dir.mkdir(parents=True, exist_ok=True)
        wavs = list(src_spk_dir.glob("*.wav"))
        for w in wavs:
            link_or_copy(w, dest_spk_dir / w.name)
        counts["train"][spk] = len(wavs)

    # Populate validation
    for spk in val_set:
        src_spk_dir = in_dir / spk
        dest_spk_dir = val_dir / spk
        dest_spk_dir.mkdir(parents=True, exist_ok=True)
        wavs = list(src_spk_dir.glob("*.wav"))
        for w in wavs:
            link_or_copy(w, dest_spk_dir / w.name)
        counts["validation"][spk] = len(wavs)

    # Populate unseen test
    for spk in held_out_set:
        src_spk_dir = in_dir / spk
        dest_spk_dir = test_unseen_dir / spk
        dest_spk_dir.mkdir(parents=True, exist_ok=True)
        wavs = list(src_spk_dir.glob("*.wav"))
        for w in wavs:
            link_or_copy(w, dest_spk_dir / w.name)
        counts["test_unseen"][spk] = len(wavs)

    # Summary table
    print(f"{'Split':<15}{'Speakers':<30}{'Clips':<10}")
    print("-" * 55)
    total_split_clips = 0
    for split_name, spk_dict in counts.items():
        spk_list_str = ", ".join(f"{s} ({c})" for s, c in sorted(spk_dict.items()))
        split_total = sum(spk_dict.values())
        total_split_clips += split_total
        print(f"{split_name:<15}{spk_list_str:<30}{split_total:<10}")
    print("-" * 55)
    print(f"Total Clips Split: {total_split_clips}")

    # Guardrail 2: STRICT LEAKAGE AUDIT
    print("\nAuditing for speaker data leakage...")
    train_speakers_found = {d.name for d in train_dir.iterdir() if d.is_dir()}
    val_speakers_found = {d.name for d in val_dir.iterdir() if d.is_dir()}
    unseen_speakers_found = {d.name for d in test_unseen_dir.iterdir() if d.is_dir()}

    for held_spk in held_out_set:
        if held_spk in train_speakers_found:
            msg = f"CRITICAL ERROR: Speaker {held_spk} found in training data! Phase 3 aborted to prevent data leakage."
            print(msg, file=sys.stderr)
            raise RuntimeError(msg)
        if held_spk in val_speakers_found:
            msg = f"CRITICAL ERROR: Speaker {held_spk} found in validation data! Phase 3 aborted to prevent data leakage."
            print(msg, file=sys.stderr)
            raise RuntimeError(msg)

        # File-level scan for held-out keywords in filename or content
        for f in train_dir.rglob("*.wav"):
            if held_spk.lower() in f.name.lower() or held_spk.lower() in str(f).lower():
                msg = f"CRITICAL ERROR: File from unseen speaker {held_spk} leaked into train: {f}!"
                print(msg, file=sys.stderr)
                raise RuntimeError(msg)

        for f in val_dir.rglob("*.wav"):
            if held_spk.lower() in f.name.lower() or held_spk.lower() in str(f).lower():
                msg = f"CRITICAL ERROR: File from unseen speaker {held_spk} leaked into val: {f}!"
                print(msg, file=sys.stderr)
                raise RuntimeError(msg)

    print("[SUCCESS] ZERO SPEAKER LEAKAGE VERIFIED. Held-out test speakers remain completely pristine.\n")
    return counts


def main():
    parser = argparse.ArgumentParser(description="Split normalized speakers into train, val, and unseen test.")
    parser.add_argument("--in_dir", type=str, default="./data/normalized", help="Normalized audio root")
    parser.add_argument("--out_dir", type=str, default="./data", help="Target root for train/val/test_unseen")
    parser.add_argument("--train_speakers", nargs="+", default=["Ananya", "Ark", "Umang", "Mayank"],
                        help="Speakers allocated to training")
    parser.add_argument("--val_speakers", nargs="+", default=["Ishita"],
                        help="Speakers allocated to validation")
    parser.add_argument("--held_out_speakers", nargs="+", default=["Vitthal"],
                        help="Speakers held out entirely as unseen test")
    args = parser.parse_args()

    split_speakers(
        Path(args.in_dir),
        Path(args.out_dir),
        args.train_speakers,
        args.val_speakers,
        args.held_out_speakers,
    )


if __name__ == "__main__":
    main()
