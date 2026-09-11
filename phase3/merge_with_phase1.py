"""
Phase 3 - Step 4: Merge Phase 1's Speech Commands split with your real-speaker
split into one combined dataset, so Phase 1's train.py can train on both at
once. Uses symlinks (falls back to copy on filesystems/OSes that block them,
e.g. Windows without dev mode / admin rights).

Run:
    python merge_with_phase1.py \
        --phase1_dir ../vikramedge_phase1/data/processed \
        --phase3_dir ./data/speakers_processed \
        --out_dir ./data/combined
"""

import argparse
import shutil
from pathlib import Path


def link_or_copy(src: Path, dest: Path):
    if dest.exists():
        return
    try:
        import os
        os.symlink(src.resolve(), dest)
    except OSError:
        shutil.copy(src, dest)


def merge_dir(src_dir: Path, out_dir: Path, prefix: str):
    if not src_dir.exists():
        return
    for split_dir in src_dir.iterdir():
        if not split_dir.is_dir():
            continue
        for label_dir in split_dir.iterdir():
            if not label_dir.is_dir():
                continue
            dest_label_dir = out_dir / split_dir.name / label_dir.name
            dest_label_dir.mkdir(parents=True, exist_ok=True)
            for wav in label_dir.glob("*.wav"):
                dest = dest_label_dir / f"{prefix}_{wav.name}"
                link_or_copy(wav, dest)


def main(args):
    out_dir = Path(args.out_dir)
    merge_dir(Path(args.phase1_dir), out_dir, prefix="sc")
    merge_dir(Path(args.phase3_dir), out_dir, prefix="spk")

    print("Combined dataset:")
    for split in ("training", "validation", "testing"):
        split_dir = out_dir / split
        if split_dir.exists():
            counts = {d.name: len(list(d.glob("*.wav"))) for d in split_dir.iterdir()}
            print(f"  {split}: {counts}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase1_dir", type=str, required=True)
    parser.add_argument("--phase3_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()
    main(args)
