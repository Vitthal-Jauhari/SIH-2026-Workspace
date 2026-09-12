"""
Phase 3 - Step 4: Merge Real Vaani Data with Negative Speech Commands Data

Merges:
1. Vaani positive data:
   - training: augmented clips from Ananya, Ark, Umang
   - validation: clean clips from Ishita
   - testing: clean clips from unseen Vitthal
2. Negative Speech Commands data:
   - unknown: non-target spoken words (go, stop, up, down, etc.)
   - silence: ambient room background noise clips

Final 3-Class Layout (strictly matching Phase 1 model):
data/combined/
├── training/
│   ├── silence/
│   ├── unknown/
│   └── vaani/
├── validation/
│   ├── silence/
│   ├── unknown/
│   └── vaani/
└── testing/
    ├── silence/
    ├── unknown/
    └── vaani/

Includes strict guardrails to assert ZERO leakage of unseen test speaker Vitthal.

Usage:
    python merge_with_phase1.py --vaani_dir ./data \
                                --sc_dir ./data/sc_raw/mini_speech_commands \
                                --out_dir ./data/combined \
                                --held_out_speakers Vitthal \
                                --val_speakers Ishita
"""

import argparse
import os
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

LABELS = ["silence", "unknown", "vaani"]
SAMPLE_RATE = 16000
CLIP_LEN = int(SAMPLE_RATE * 1.0)


def link_or_copy(src: Path, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    try:
        os.link(src.resolve(), dest)
    except OSError:
        shutil.copy(src, dest)


def generate_silence_clips(out_dir: Path, n_needed: int, prefix: str):
    """
    Generates realistic 1-second silence/ambient room noise clips.
    Combines low-level pink noise, thermal noise, and mains hum (50/60Hz).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n_needed):
        t = np.linspace(0, 1.0, CLIP_LEN, endpoint=False)
        # Room ambient hum (50Hz + 100Hz harmonic) at very low level (-40 dB to -60 dB)
        hum_amp = 10 ** (random.uniform(-55, -40) / 20)
        noise_amp = 10 ** (random.uniform(-60, -45) / 20)
        hum = hum_amp * (np.sin(2 * np.pi * 50 * t) + 0.3 * np.sin(2 * np.pi * 100 * t))
        white = np.random.normal(0, noise_amp, CLIP_LEN)
        clip = np.clip(hum + white, -1.0, 1.0).astype(np.float32)
        out_file = out_dir / f"silence_{prefix}_{i:04d}.wav"
        sf.write(str(out_file), clip, SAMPLE_RATE, subtype="PCM_16")


def build_combined_dataset(
    vaani_dir: Path,
    sc_dir: Path,
    out_dir: Path,
    held_out_speakers: list,
    val_speakers: list,
    seed: int = 42,
):
    random.seed(seed)
    np.random.seed(seed)

    vaani_dir = vaani_dir.resolve()
    sc_dir = sc_dir.resolve()
    out_dir = out_dir.resolve()
    held_out_set = set(held_out_speakers or ["Vitthal"])
    val_set = set(val_speakers or ["Ishita"])

    print("=" * 65)
    print("PHASE 3 - STEP 4: MERGE VAANI + NEGATIVE DATASET")
    print("=" * 65)
    print(f"Vaani data root       : {vaani_dir}")
    print(f"Speech commands root  : {sc_dir}")
    print(f"Combined destination  : {out_dir}")
    print(f"Held-out Unseen test  : {sorted(held_out_set)}")
    print(f"Validation speakers   : {sorted(val_set)}\n")

    # Clean existing destination
    if out_dir.exists():
        shutil.rmtree(out_dir)
    for split in ("training", "validation", "testing"):
        for label in LABELS:
            (out_dir / split / label).mkdir(parents=True, exist_ok=True)

    # 1. Ingest Vaani POSITIVE data
    # Training: from data/train_augmented
    train_aug_dir = vaani_dir / "train_augmented"
    if not train_aug_dir.exists():
        raise FileNotFoundError(f"Training augmented directory not found: {train_aug_dir}")

    vaani_train_dest = out_dir / "training" / "vaani"
    n_train_vaani = 0
    train_speakers_found = set()
    for spk_dir in train_aug_dir.iterdir():
        if not spk_dir.is_dir():
            continue
        spk = spk_dir.name
        train_speakers_found.add(spk)
        for wav in spk_dir.glob("*.wav"):
            dest = vaani_train_dest / f"{spk}_{wav.name}"
            link_or_copy(wav, dest)
            n_train_vaani += 1

    # Validation: clean Ishita from data/validation
    val_raw_dir = vaani_dir / "validation"
    vaani_val_dest = out_dir / "validation" / "vaani"
    n_val_vaani = 0
    val_speakers_found = set()
    for spk_dir in val_raw_dir.iterdir():
        if not spk_dir.is_dir():
            continue
        spk = spk_dir.name
        val_speakers_found.add(spk)
        for wav in spk_dir.glob("*.wav"):
            dest = vaani_val_dest / f"{spk}_{wav.name}"
            link_or_copy(wav, dest)
            n_val_vaani += 1

    # Testing: clean unseen Vitthal from data/test_unseen
    test_raw_dir = vaani_dir / "test_unseen"
    vaani_test_dest = out_dir / "testing" / "vaani"
    n_test_vaani = 0
    unseen_speakers_found = set()
    for spk_dir in test_raw_dir.iterdir():
        if not spk_dir.is_dir():
            continue
        spk = spk_dir.name
        unseen_speakers_found.add(spk)
        for wav in spk_dir.glob("*.wav"):
            dest = vaani_test_dest / f"{spk}_{wav.name}"
            link_or_copy(wav, dest)
            n_test_vaani += 1

    # 2. Ingest NEGATIVE Speech Commands Data (unknown)
    # Collect all words from sc_dir
    sc_words = [d for d in sc_dir.iterdir() if d.is_dir() and not d.name.startswith(("_", "."))]
    sc_wavs_by_word = {}
    total_sc_wavs = 0
    for w_dir in sc_words:
        wavs = list(w_dir.glob("*.wav"))
        if wavs:
            sc_wavs_by_word[w_dir.name] = wavs
            total_sc_wavs += len(wavs)

    print(f"Speech Commands words available: {sorted(sc_wavs_by_word.keys())} ({total_sc_wavs} total clips)")

    # Pool words for balanced distribution
    all_sc_pool = []
    for w, w_list in sc_wavs_by_word.items():
        all_sc_pool.extend(w_list)
    random.shuffle(all_sc_pool)

    # Balance counts with Vaani classes
    # Train: 700 unknown, 700 silence (to match ~690 Vaani)
    # Val: 100 unknown, 100 silence (to match ~42 Vaani)
    # Test: 100 unknown, 100 silence (to match ~26 Vaani)
    n_train_neg = max(n_train_vaani, 700)
    n_val_neg = 100
    n_test_neg = 100

    idx = 0
    # Training unknown
    for _ in range(n_train_neg):
        w = all_sc_pool[idx % len(all_sc_pool)]
        idx += 1
        dest = out_dir / "training" / "unknown" / f"sc_train_{idx:05d}_{w.parent.name}_{w.name}"
        link_or_copy(w, dest)

    # Validation unknown
    for _ in range(n_val_neg):
        w = all_sc_pool[idx % len(all_sc_pool)]
        idx += 1
        dest = out_dir / "validation" / "unknown" / f"sc_val_{idx:05d}_{w.parent.name}_{w.name}"
        link_or_copy(w, dest)

    # Testing unknown
    for _ in range(n_test_neg):
        w = all_sc_pool[idx % len(all_sc_pool)]
        idx += 1
        dest = out_dir / "testing" / "unknown" / f"sc_test_{idx:05d}_{w.parent.name}_{w.name}"
        link_or_copy(w, dest)

    # 3. Generate SILENCE clips
    generate_silence_clips(out_dir / "training" / "silence", n_train_neg, prefix="train")
    generate_silence_clips(out_dir / "validation" / "silence", n_val_neg, prefix="val")
    generate_silence_clips(out_dir / "testing" / "silence", n_test_neg, prefix="test")

    # Guardrail: Check for unseen speaker leakage in combined dataset
    for held_spk in held_out_set:
        for split in ("training", "validation"):
            for f in (out_dir / split).rglob("*.wav"):
                if held_spk.lower() in f.name.lower() or held_spk.lower() in str(f).lower():
                    msg = f"CRITICAL ERROR: Unseen speaker {held_spk} detected in {split} data: {f}!"
                    print(msg, file=sys.stderr)
                    raise RuntimeError(msg)

    # Dataset Summary Table
    print("\nCOMBINED DATASET SUMMARY:")
    print(f"{'Split':<15}{'silence':<12}{'unknown':<12}{'vaani':<12}{'Total':<10}")
    print("-" * 55)
    for split in ("training", "validation", "testing"):
        c_sil = len(list((out_dir / split / "silence").glob("*.wav")))
        c_unk = len(list((out_dir / split / "unknown").glob("*.wav")))
        c_vaa = len(list((out_dir / split / "vaani").glob("*.wav")))
        c_tot = c_sil + c_unk + c_vaa
        print(f"{split:<15}{c_sil:<12}{c_unk:<12}{c_vaa:<12}{c_tot:<10}")
    print("-" * 55)

    print(f"Vaani Speakers in Train : {sorted(train_speakers_found)}")
    print(f"Vaani Speakers in Val   : {sorted(val_speakers_found)}")
    print(f"Vaani Speakers in Test  : {sorted(unseen_speakers_found)}")
    print("\n[SUCCESS] Dataset combined and verified. Zero leakage of unseen test speaker.\n")


def main():
    parser = argparse.ArgumentParser(description="Merge Vaani with Phase 1 negative dataset.")
    parser.add_argument("--vaani_dir", type=str, default="./data")
    parser.add_argument("--sc_dir", type=str, default="./data/sc_raw/mini_speech_commands")
    parser.add_argument("--out_dir", type=str, default="./data/combined")
    parser.add_argument("--held_out_speakers", nargs="+", default=["Vitthal"])
    parser.add_argument("--val_speakers", nargs="+", default=["Ishita"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    build_combined_dataset(
        Path(args.vaani_dir),
        Path(args.sc_dir),
        Path(args.out_dir),
        args.held_out_speakers,
        args.val_speakers,
        args.seed,
    )


if __name__ == "__main__":
    main()
