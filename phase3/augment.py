"""
Phase 3 - Step 3: Augment Training Data ONLY

Applies realistic, gentle acoustic augmentations to TRAINING speakers only:
- Volume scaling (+/- 6 dB)
- Speed / time stretch (0.92x to 1.08x)
- Pitch shift (+/- 1.5 semitones)
- Background room / ambient noise injection (SNR 10 dB to 25 dB)

Validation and unseen test speakers are STRICTLY EXCLUDED and NEVER augmented.

Usage:
    python augment.py --in_dir ./data/train \
                      --out_dir ./data/train_augmented \
                      --copies_per_file 4 \
                      --held_out_speakers Vitthal \
                      --val_speakers Ishita
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf

SAMPLE_RATE = 16000
CLIP_SECONDS = 1.0
CLIP_LEN = int(SAMPLE_RATE * CLIP_SECONDS)


def fit_length(audio: np.ndarray, target_len: int = CLIP_LEN) -> np.ndarray:
    """Pad or center-crop/truncate audio to target_len."""
    if len(audio) < target_len:
        pad_len = target_len - len(audio)
        return np.pad(audio, (0, pad_len), mode="constant")
    elif len(audio) > target_len:
        # If audio is longer than 1s, keep the primary energy region (wake-word)
        # Find highest energy 1s window
        window = target_len
        step = target_len // 4
        best_energy = -1.0
        best_start = 0
        for start in range(0, len(audio) - window + 1, max(1, step)):
            segment = audio[start:start + window]
            energy = np.sum(segment ** 2)
            if energy > best_energy:
                best_energy = energy
                best_start = start
        return audio[best_start:best_start + window]
    return audio


def add_noise(audio: np.ndarray, snr_db: float, noise_type: str = "ambient") -> np.ndarray:
    """Generate and mix realistic background noise at target SNR."""
    length = len(audio)
    if noise_type == "pink":
        # Pink noise (1/f)
        white = np.random.normal(0, 1, length)
        # Approximate 1/f filter
        b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
        a = [1, -2.494956002, 2.017265875, -0.522189400]
        from scipy.signal import lfilter
        noise = lfilter(b, a, white)
    elif noise_type == "ambient":
        # Low-frequency ambient rumble + gentle hiss
        t = np.linspace(0, length / SAMPLE_RATE, length)
        hum = 0.3 * np.sin(2 * np.pi * 50 * t) + 0.2 * np.sin(2 * np.pi * 100 * t)
        hiss = np.random.normal(0, 0.5, length)
        noise = hum + hiss
    else:
        noise = np.random.normal(0, 1, length)

    sig_power = np.mean(audio ** 2) + 1e-10
    noise_power = np.mean(noise ** 2) + 1e-10
    target_noise_power = sig_power / (10 ** (snr_db / 10))
    scaled_noise = noise * np.sqrt(target_noise_power / noise_power)
    return np.clip(audio + scaled_noise, -1.0, 1.0)


def time_stretch(audio: np.ndarray, rate: float) -> np.ndarray:
    stretched = librosa.effects.time_stretch(audio, rate=rate)
    return fit_length(stretched)


def pitch_shift(audio: np.ndarray, n_steps: float) -> np.ndarray:
    shifted = librosa.effects.pitch_shift(audio, sr=SAMPLE_RATE, n_steps=n_steps)
    return fit_length(shifted)


def change_volume(audio: np.ndarray, gain_db: float) -> np.ndarray:
    gain = 10 ** (gain_db / 20)
    return np.clip(audio * gain, -1.0, 1.0)


def random_augment(audio: np.ndarray) -> np.ndarray:
    out = audio.copy()

    # 1. Volume scaling (70% chance): gentle +/- 5 dB
    if random.random() < 0.7:
        out = change_volume(out, gain_db=random.uniform(-5.0, 5.0))

    # 2. Time stretch / speed change (50% chance): 0.92x to 1.08x
    if random.random() < 0.5:
        out = time_stretch(out, rate=random.uniform(0.92, 1.08))

    # 3. Pitch shift (50% chance): +/- 1.5 semitones
    if random.random() < 0.5:
        out = pitch_shift(out, n_steps=random.uniform(-1.5, 1.5))

    # 4. Realistic background / room noise (75% chance): 12 dB to 25 dB SNR
    if random.random() < 0.75:
        ntype = random.choice(["ambient", "pink", "white"])
        out = add_noise(out, snr_db=random.uniform(12.0, 25.0), noise_type=ntype)

    return fit_length(out)


def augment_training_speakers(
    in_dir: Path,
    out_dir: Path,
    copies_per_file: int = 4,
    held_out_speakers: list = None,
    val_speakers: list = None,
    seed: int = 42,
):
    random.seed(seed)
    np.random.seed(seed)

    in_dir = in_dir.resolve()
    out_dir = out_dir.resolve()

    if not in_dir.exists():
        raise FileNotFoundError(f"Training input dir {in_dir} does not exist!")

    held_out_set = set(held_out_speakers or [])
    val_set = set(val_speakers or [])

    # Guardrail: Check for held-out or validation speakers
    input_speakers = {d.name for d in in_dir.iterdir() if d.is_dir()}
    leaked_held_out = input_speakers & held_out_set
    leaked_val = input_speakers & val_set

    if leaked_held_out:
        msg = f"CRITICAL ERROR: Held-out unseen speaker(s) {leaked_held_out} detected in training augmentation input!"
        print(msg, file=sys.stderr)
        raise RuntimeError(msg)

    if leaked_val:
        msg = f"CRITICAL ERROR: Validation speaker(s) {leaked_val} detected in training augmentation input!"
        print(msg, file=sys.stderr)
        raise RuntimeError(msg)

    print("=" * 65)
    print("PHASE 3 - STEP 3: TRAIN-ONLY DATA AUGMENTATION")
    print("=" * 65)
    print(f"Training speakers to augment : {sorted(input_speakers)}")
    print(f"Copies per file              : {copies_per_file} (+1 clean = {copies_per_file + 1}x)")
    print(f"Destination                  : {out_dir}\n")

    if out_dir.exists():
        import shutil
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total_sources = 0
    total_written = 0
    per_speaker_counts = {}

    for spk_dir in sorted(in_dir.iterdir()):
        if not spk_dir.is_dir():
            continue
        spk_name = spk_dir.name
        wav_files = sorted(list(spk_dir.glob("*.wav")))
        total_sources += len(wav_files)

        dest_spk_dir = out_dir / spk_name
        dest_spk_dir.mkdir(parents=True, exist_ok=True)

        spk_written = 0
        for wav in wav_files:
            audio, _ = librosa.load(str(wav), sr=SAMPLE_RATE, mono=True)
            clean_audio = fit_length(audio)

            # 1 clean copy
            clean_dest = dest_spk_dir / f"{wav.stem}_clean.wav"
            sf.write(str(clean_dest), clean_audio, SAMPLE_RATE, subtype="PCM_16")
            spk_written += 1

            # N augmented copies
            for c in range(copies_per_file):
                aug_audio = random_augment(clean_audio)
                aug_dest = dest_spk_dir / f"{wav.stem}_aug{c:02d}.wav"
                sf.write(str(aug_dest), aug_audio, SAMPLE_RATE, subtype="PCM_16")
                spk_written += 1

        per_speaker_counts[spk_name] = {"source": len(wav_files), "augmented": spk_written}
        total_written += spk_written

    print(f"{'Speaker':<15}{'Source Clips':<15}{'Total Output Clips':<20}")
    print("-" * 50)
    for spk, counts in per_speaker_counts.items():
        print(f"{spk:<15}{counts['source']:<15}{counts['augmented']:<20}")
    print("-" * 50)
    print(f"Total Source Clips : {total_sources}")
    print(f"Total Clips Written: {total_written}")

    # Guardrail: Verify that unseen speakers are NOT in augmented output
    out_speakers = {d.name for d in out_dir.iterdir() if d.is_dir()}
    leakage = out_speakers & (held_out_set | val_set)
    if leakage:
        msg = f"CRITICAL ERROR: Unseen/val speakers {leakage} found in augmented output!"
        print(msg, file=sys.stderr)
        raise RuntimeError(msg)

    print(f"[SUCCESS] Training data augmented successfully ({total_written} clips). Zero test leakage.\n")
    return per_speaker_counts


def main():
    parser = argparse.ArgumentParser(description="Augment training speakers only.")
    parser.add_argument("--in_dir", type=str, default="./data/train", help="Path to data/train")
    parser.add_argument("--out_dir", type=str, default="./data/train_augmented", help="Path to data/train_augmented")
    parser.add_argument("--copies_per_file", type=int, default=4, help="Number of augmented copies per file")
    parser.add_argument("--held_out_speakers", nargs="+", default=["Vitthal"], help="Held-out unseen test speakers")
    parser.add_argument("--val_speakers", nargs="+", default=["Ishita"], help="Validation speakers")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    augment_training_speakers(
        Path(args.in_dir),
        Path(args.out_dir),
        args.copies_per_file,
        args.held_out_speakers,
        args.val_speakers,
        args.seed,
    )


if __name__ == "__main__":
    main()
