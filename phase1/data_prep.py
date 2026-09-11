"""
Phase 1 - Step 1: Download Google Speech Commands v0.02 and organize negative classes
(unknown and silence) for VikramEdge prototyping.

"unknown" is built by sampling across Speech Commands words so the model learns
a broad "not wake-word" boundary.

"silence" is built by chopping up the background_noise/ recordings that ship
with the dataset into 1-second clips.

(Target keyword extraction for yes/no has been commented out; custom wake words
like Vaani are ingested via prepare_vaani_data.py).

Run:
    python data_prep.py --data_dir ./data
"""

import argparse
import hashlib
import os
import random
import shutil
import tarfile
import urllib.request
from pathlib import Path

DATASET_URL = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
# TARGET_WORDS = ["yes", "no"]  # Commented out: target keyword now handled by prepare_vaani_data.py
SILENCE_LABEL = "silence"
UNKNOWN_LABEL = "unknown"
SAMPLE_RATE = 16000
CLIP_SECONDS = 1.0


def download_and_extract(data_dir: Path):
    archive_path = data_dir / "speech_commands_v0.02.tar.gz"
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    if not archive_path.exists():
        print(f"Downloading {DATASET_URL} ...")
        urllib.request.urlretrieve(DATASET_URL, archive_path)
    else:
        print("Archive already downloaded, skipping.")

    if not any(raw_dir.iterdir()):
        print("Extracting...")
        with tarfile.open(archive_path) as tar:
            tar.extractall(raw_dir)
    else:
        print("Already extracted, skipping.")

    return raw_dir


def which_set(filename: str, validation_pct=10, testing_pct=10) -> str:
    """
    Reproduces Google's official deterministic train/val/test split so a
    given speaker's utterances always land in the same split (no leakage).
    Based on the hashing scheme from the original Speech Commands paper.
    """
    base_name = os.path.basename(filename)
    # Strip anything after '_nohash_' so the same speaker always hashes the same way
    hash_name = base_name.split("_nohash_")[0]
    h = hashlib.sha1(hash_name.encode("utf-8")).hexdigest()
    percentage_hash = (int(h, 16) % (2 ** 27 - 1)) * (100.0 / (2 ** 27 - 1))
    if percentage_hash < validation_pct:
        return "validation"
    elif percentage_hash < (testing_pct + validation_pct):
        return "testing"
    else:
        return "training"


def build_dataset(raw_dir: Path, out_dir: Path, unknown_per_split_ratio=1.0, seed=42):
    random.seed(seed)
    word_dirs = [
        d for d in raw_dir.iterdir()
        if d.is_dir() and d.name not in ("_background_noise_",)
    ]
    # All Speech Commands words serve as pool for "unknown"
    other_words = [d.name for d in word_dirs]

    splits = {"training": [], "validation": [], "testing": []}

    # 1. Target words (yes/no) -- commented out; target wake word is now handled by prepare_vaani_data.py
    # for word in TARGET_WORDS:
    #     word_dir = raw_dir / word
    #     for wav in word_dir.glob("*.wav"):
    #         split = which_set(wav.name)
    #         splits[split].append((wav, word))

    # 2. unknown -- sample evenly across non-target words
    unknown_candidates = {"training": [], "validation": [], "testing": []}
    for word in other_words:
        word_dir = raw_dir / word
        for wav in word_dir.glob("*.wav"):
            split = which_set(wav.name)
            unknown_candidates[split].append(wav)

    # Negative-class sample counts per split matching standard Speech Commands scale
    target_counts = {"training": 6367, "validation": 744, "testing": 874}
    for split in ("training", "validation", "testing"):
        n_unknown = int(target_counts[split] * unknown_per_split_ratio)
        pool = unknown_candidates[split]
        random.shuffle(pool)
        chosen = pool[:n_unknown]
        for wav in chosen:
            splits[split].append((wav, UNKNOWN_LABEL))

    # 3. silence -- handled separately by chop_silence(), just record the count needed
    for split in ("training", "validation", "testing"):
        splits[split + "_silence_count"] = int(target_counts[split] * unknown_per_split_ratio)

    # Materialize into out_dir/<split>/<label>/*.wav via symlinks (cheap, no copy)
    for split in ("training", "validation", "testing"):
        for wav, label in splits[split]:
            label_dir = out_dir / split / label
            label_dir.mkdir(parents=True, exist_ok=True)
            dest = label_dir / f"{wav.parent.name}_{wav.name}"
            if not dest.exists():
                try:
                    os.symlink(wav.resolve(), dest)
                except OSError:
                    shutil.copy(wav, dest)

    chop_silence(raw_dir, out_dir, splits)
    print_summary(out_dir)


def chop_silence(raw_dir: Path, out_dir: Path, splits):
    import soundfile as sf
    import numpy as np

    bg_dir = raw_dir / "_background_noise_"
    bg_files = list(bg_dir.glob("*.wav")) if bg_dir.exists() else []
    if not bg_files:
        print("WARNING: no _background_noise_ directory found; skipping silence class.")
        return

    clip_len = int(SAMPLE_RATE * CLIP_SECONDS)
    for split in ("training", "validation", "testing"):
        n_needed = splits[split + "_silence_count"]
        out_label_dir = out_dir / split / SILENCE_LABEL
        out_label_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n_needed):
            bg_path = random.choice(bg_files)
            audio, sr = sf.read(bg_path)
            if len(audio) <= clip_len:
                clip = np.pad(audio, (0, clip_len - len(audio)))
            else:
                start = random.randint(0, len(audio) - clip_len)
                clip = audio[start:start + clip_len]
            sf.write(out_label_dir / f"silence_{split}_{i:05d}.wav", clip, sr)


def print_summary(out_dir: Path):
    print("\nDataset summary:")
    for split in ("training", "validation", "testing"):
        split_dir = out_dir / split
        if not split_dir.exists():
            continue
        counts = {
            label_dir.name: len(list(label_dir.glob("*.wav")))
            for label_dir in split_dir.iterdir() if label_dir.is_dir()
        }
        print(f"  {split}: {counts}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = data_dir / "processed"

    raw_dir = download_and_extract(data_dir)
    build_dataset(raw_dir, out_dir)
