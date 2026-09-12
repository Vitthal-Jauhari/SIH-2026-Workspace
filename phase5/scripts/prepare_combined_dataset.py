"""
Phase 5 - Step 6: Dataset Balancing & 3-Class Combined Assembly

Assembles the 3-class dataset:
- "vaani": Positive wake-word clips
- "unknown": Negative speech (mini_speech_commands + any hard negatives: pani, rani, etc.)
- "silence": Calibrated ambient room noise & silence

Dataset Dependency Rule:
- Strictly checks that mini_speech_commands is available in the project.
- If missing, aborts immediately and reports the missing dependency.

Zero-Leakage Guardrail:
- Verifies that held-out unseen test speaker (Vitthal) never appears in training or validation splits.
"""

import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path
from typing import List, Set

import numpy as np
import soundfile as sf

LABELS = ["silence", "unknown", "vaani"]
TARGET_SAMPLE_RATE = 16000
CLIP_LEN = int(TARGET_SAMPLE_RATE * 1.0)


def link_or_copy(src: Path, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    try:
        os.link(src.resolve(), dest)
    except OSError:
        shutil.copy(src, dest)


def generate_silence_clips(out_dir: Path, n_needed: int, prefix: str):
    """Generates realistic calibrated ambient room noise clips."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n_needed):
        t = np.linspace(0, 1.0, CLIP_LEN, endpoint=False)
        hum_amp = 10 ** (random.uniform(-55, -40) / 20)
        noise_amp = 10 ** (random.uniform(-60, -45) / 20)
        hum = hum_amp * (np.sin(2 * np.pi * 50 * t) + 0.3 * np.sin(2 * np.pi * 100 * t))
        white = np.random.normal(0, noise_amp, CLIP_LEN)
        clip = np.clip(hum + white, -1.0, 1.0).astype(np.float32)
        out_file = out_dir / f"silence_{prefix}_{i:04d}.wav"
        sf.write(str(out_file), clip, TARGET_SAMPLE_RATE, subtype="PCM_16")


def build_combined_dataset(
    phase5_root: Path,
    sc_dir: Path,
    held_out_speakers: List[str],
    val_speakers: List[str],
    seed: int = 42,
):
    random.seed(seed)
    np.random.seed(seed)

    phase5_root = phase5_root.resolve()
    sc_dir = sc_dir.resolve()
    data_dir = phase5_root / "data"
    splits_dir = data_dir / "splits"
    train_aug_dir = data_dir / "train_augmented"
    combined_dir = data_dir / "combined"
    artifacts_dir = phase5_root / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    held_out_set = set(held_out_speakers or ["Vitthal"])
    val_set = set(val_speakers or ["Ishita"])

    print("=" * 70)
    print("PHASE 5 - STEP 6: DATASET BALANCING & COMBINED ASSEMBLY")
    print("=" * 70)
    print(f"Phase 5 Root       : {phase5_root}")
    print(f"Speech Commands Dir: {sc_dir}")
    print(f"Combined Target    : {combined_dir}")
    print(f"Held-out Unseen    : {sorted(held_out_set)}")
    print(f"Validation Speakers: {sorted(val_set)}\n")

    # Dependency check: mini_speech_commands
    if not sc_dir.exists():
        msg = (
            f"FATAL ERROR: Required dataset dependency 'mini_speech_commands' not found at:\n"
            f"{sc_dir}\n"
            f"Per Non-Negotiable Rule, the pipeline will not invent or silently download unknown data.\n"
            f"Please ensure mini_speech_commands is available in phase3/data/sc_raw/mini_speech_commands."
        )
        print(msg, file=sys.stderr)
        raise FileNotFoundError(msg)

    # Clean previous combined dataset
    if combined_dir.exists():
        shutil.rmtree(combined_dir)
    for split in ("training", "validation", "testing"):
        for lbl in LABELS:
            (combined_dir / split / lbl).mkdir(parents=True, exist_ok=True)

    # 1. Ingest Vaani Positives
    # Train: from train_augmented
    train_vaani_dest = combined_dir / "training" / "vaani"
    n_train_vaani = 0
    for spk_dir in sorted(train_aug_dir.iterdir()):
        if not spk_dir.is_dir() or spk_dir.name in held_out_set or spk_dir.name in val_set:
            continue
        for wav in spk_dir.glob("*.wav"):
            dest = train_vaani_dest / f"{spk_dir.name}_{wav.name}"
            link_or_copy(wav, dest)
            n_train_vaani += 1

    # Validation: clean from splits/validation
    val_vaani_dest = combined_dir / "validation" / "vaani"
    n_val_vaani = 0
    val_dir = splits_dir / "validation"
    if val_dir.exists():
        for spk_dir in sorted(val_dir.iterdir()):
            if not spk_dir.is_dir():
                continue
            for wav in spk_dir.glob("*.wav"):
                dest = val_vaani_dest / f"{spk_dir.name}_{wav.name}"
                link_or_copy(wav, dest)
                n_val_vaani += 1

    # Testing: clean from splits/test_unseen
    test_vaani_dest = combined_dir / "testing" / "vaani"
    n_test_vaani = 0
    test_dir = splits_dir / "test_unseen"
    if test_dir.exists():
        for spk_dir in sorted(test_dir.iterdir()):
            if not spk_dir.is_dir():
                continue
            for wav in spk_dir.glob("*.wav"):
                dest = test_vaani_dest / f"{spk_dir.name}_{wav.name}"
                link_or_copy(wav, dest)
                n_test_vaani += 1

    print(f"Positive Vaani counts: Train={n_train_vaani}, Val={n_val_vaani}, Test={n_test_vaani}")

    # 2. Ingest Hard Negatives (if any exist in data/normalized/hard_negatives)
    hard_neg_dir = data_dir / "normalized" / "hard_negatives"
    hard_neg_files = []
    if hard_neg_dir.exists():
        for cat_dir in hard_neg_dir.iterdir():
            if cat_dir.is_dir():
                hard_neg_files.extend(list(cat_dir.glob("*.wav")))

    n_hard_neg = len(hard_neg_files)
    print(f"Discovered {n_hard_neg} hard negative files.")

    # 3. Ingest Speech Commands Negatives (unknown class)
    sc_words = [d for d in sc_dir.iterdir() if d.is_dir() and not d.name.startswith(("_", "."))]
    all_sc_pool = []
    for w_dir in sc_words:
        all_sc_pool.extend(list(w_dir.glob("*.wav")))
    random.shuffle(all_sc_pool)
    print(f"Speech Commands available clips: {len(all_sc_pool)} across {len(sc_words)} words")

    # Target class balance
    # Train: balanced with positive Vaani count (~1000 - 2000)
    n_train_neg = n_train_vaani
    n_val_neg = max(100, n_val_vaani * 2)
    n_test_neg = max(100, n_test_vaani * 3)

    # Ingest hard negatives into training unknown first
    idx = 0
    for hn in hard_neg_files:
        dest = combined_dir / "training" / "unknown" / f"hardneg_{hn.parent.name}_{hn.name}"
        link_or_copy(hn, dest)
        idx += 1

    # Fill remainder of training unknown with Speech Commands
    sc_idx = 0
    while idx < n_train_neg:
        w = all_sc_pool[sc_idx % len(all_sc_pool)]
        sc_idx += 1
        dest = combined_dir / "training" / "unknown" / f"sc_train_{idx:05d}_{w.parent.name}_{w.name}"
        link_or_copy(w, dest)
        idx += 1

    # Validation unknown
    for v_i in range(n_val_neg):
        w = all_sc_pool[sc_idx % len(all_sc_pool)]
        sc_idx += 1
        dest = combined_dir / "validation" / "unknown" / f"sc_val_{v_i:05d}_{w.parent.name}_{w.name}"
        link_or_copy(w, dest)

    # Testing unknown
    for t_i in range(n_test_neg):
        w = all_sc_pool[sc_idx % len(all_sc_pool)]
        sc_idx += 1
        dest = combined_dir / "testing" / "unknown" / f"sc_test_{t_i:05d}_{w.parent.name}_{w.name}"
        link_or_copy(w, dest)

    # 4. Generate Calibrated Ambient Silence
    generate_silence_clips(combined_dir / "training" / "silence", n_train_neg, "train")
    generate_silence_clips(combined_dir / "validation" / "silence", n_val_neg, "val")
    generate_silence_clips(combined_dir / "testing" / "silence", n_test_neg, "test")

    # 5. Strict Zero-Leakage Guardrail on Combined Dataset
    for held_spk in held_out_set:
        for split in ("training", "validation"):
            for f in (combined_dir / split).rglob("*.wav"):
                if held_spk.lower() in f.name.lower():
                    raise RuntimeError(f"FATAL LEAKAGE: Unseen speaker '{held_spk}' found in {f}!")

    stats = {
        "training": {
            "vaani": len(list((combined_dir / "training" / "vaani").glob("*.wav"))),
            "unknown": len(list((combined_dir / "training" / "unknown").glob("*.wav"))),
            "silence": len(list((combined_dir / "training" / "silence").glob("*.wav"))),
        },
        "validation": {
            "vaani": len(list((combined_dir / "validation" / "vaani").glob("*.wav"))),
            "unknown": len(list((combined_dir / "validation" / "unknown").glob("*.wav"))),
            "silence": len(list((combined_dir / "validation" / "silence").glob("*.wav"))),
        },
        "testing": {
            "vaani": len(list((combined_dir / "testing" / "vaani").glob("*.wav"))),
            "unknown": len(list((combined_dir / "testing" / "unknown").glob("*.wav"))),
            "silence": len(list((combined_dir / "testing" / "silence").glob("*.wav"))),
        },
    }
    for s in ("training", "validation", "testing"):
        stats[s]["total"] = sum(stats[s].values())

    stats["total_samples"] = sum(stats[s]["total"] for s in ("training", "validation", "testing"))
    stats["verified_zero_leakage"] = True

    stats_file = artifacts_dir / "dataset_stats.json"
    with open(stats_file, "w") as fp:
        json.dump(stats, fp, indent=2)

    print("\nCOMBINED DATASET STATS:")
    print(f"  Training Split   : {stats['training']['total']} (vaani={stats['training']['vaani']}, unknown={stats['training']['unknown']}, silence={stats['training']['silence']})")
    print(f"  Validation Split : {stats['validation']['total']} (vaani={stats['validation']['vaani']}, unknown={stats['validation']['unknown']}, silence={stats['validation']['silence']})")
    print(f"  Testing Split    : {stats['testing']['total']} (vaani={stats['testing']['vaani']}, unknown={stats['testing']['unknown']}, silence={stats['testing']['silence']})")
    print(f"  Total Dataset    : {stats['total_samples']} samples")
    print(f"\n[PASSED] Zero-leakage verified. Stats saved to: {stats_file}\n")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Build Combined 3-Class Dataset")
    parser.add_argument(
        "--phase5_root",
        type=str,
        default=str(Path(__file__).resolve().parent.parent),
    )
    parser.add_argument(
        "--sc_dir",
        type=str,
        default=str(Path(__file__).resolve().parent.parent.parent / "phase3" / "data" / "sc_raw" / "mini_speech_commands"),
    )
    parser.add_argument(
        "--held_out_speakers",
        nargs="+",
        default=["Vitthal"],
    )
    parser.add_argument(
        "--val_speakers",
        nargs="+",
        default=["Ishita"],
    )
    args = parser.parse_args()

    build_combined_dataset(
        Path(args.phase5_root),
        Path(args.sc_dir),
        held_out_speakers=args.held_out_speakers,
        val_speakers=args.val_speakers,
    )


if __name__ == "__main__":
    main()
