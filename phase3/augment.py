"""
Phase 3 - Step 2: AI/DSP-based augmentation to multiply your real-speaker
recordings and cover the acoustic variation an unseen speaker/environment
will introduce: noise, speed, pitch, volume.

For each input clip, generates several augmented copies by combining random
draws from each augmentation type. Applied only to the *training* split
later (never to validation/testing -- those must stay clean to measure
real generalization).

Run:
    python augment.py --in_dir ./data/speakers --out_dir ./data/speakers_augmented \
                       --noise_dir ../phase1/data/raw/_background_noise_ \
                       --copies_per_file 4
"""

import argparse
import random
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf

SAMPLE_RATE = 16000
CLIP_SECONDS = 1.0
CLIP_LEN = int(SAMPLE_RATE * CLIP_SECONDS)


def add_noise(audio: np.ndarray, noise_clip: np.ndarray, snr_db: float) -> np.ndarray:
    if len(noise_clip) < len(audio):
        reps = int(np.ceil(len(audio) / len(noise_clip)))
        noise_clip = np.tile(noise_clip, reps)
    start = random.randint(0, max(len(noise_clip) - len(audio), 0))
    noise = noise_clip[start:start + len(audio)]

    sig_power = np.mean(audio ** 2) + 1e-10
    noise_power = np.mean(noise ** 2) + 1e-10
    target_noise_power = sig_power / (10 ** (snr_db / 10))
    noise = noise * np.sqrt(target_noise_power / noise_power)
    return audio + noise


def time_stretch(audio: np.ndarray, rate: float) -> np.ndarray:
    stretched = librosa.effects.time_stretch(audio, rate=rate)
    return fit_length(stretched)


def pitch_shift(audio: np.ndarray, n_steps: float) -> np.ndarray:
    shifted = librosa.effects.pitch_shift(audio, sr=SAMPLE_RATE, n_steps=n_steps)
    return fit_length(shifted)


def change_volume(audio: np.ndarray, gain_db: float) -> np.ndarray:
    gain = 10 ** (gain_db / 20)
    return np.clip(audio * gain, -1.0, 1.0)


def fit_length(audio: np.ndarray, target_len=CLIP_LEN) -> np.ndarray:
    if len(audio) < target_len:
        return np.pad(audio, (0, target_len - len(audio)))
    return audio[:target_len]


def random_augment(audio: np.ndarray, noise_clips) -> np.ndarray:
    out = audio.copy()

    if noise_clips and random.random() < 0.7:
        out = add_noise(out, random.choice(noise_clips), snr_db=random.uniform(3, 20))

    if random.random() < 0.5:
        out = time_stretch(out, rate=random.uniform(0.85, 1.15))

    if random.random() < 0.5:
        out = pitch_shift(out, n_steps=random.uniform(-2.5, 2.5))

    if random.random() < 0.6:
        out = change_volume(out, gain_db=random.uniform(-8, 8))

    return fit_length(out)


def load_noise_clips(noise_dir: Path):
    if not noise_dir or not noise_dir.exists():
        print(f"WARNING: noise_dir {noise_dir} not found -- skipping noise augmentation.")
        return []
    clips = []
    for wav in noise_dir.glob("*.wav"):
        audio, _ = librosa.load(wav, sr=SAMPLE_RATE, mono=True)
        clips.append(audio)
    return clips


def main(args):
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    noise_clips = load_noise_clips(Path(args.noise_dir)) if args.noise_dir else []

    wav_files = list(in_dir.rglob("*.wav"))
    print(f"Found {len(wav_files)} source clips under {in_dir}")

    n_written = 0
    for wav in wav_files:
        rel = wav.relative_to(in_dir)  # <speaker_id>/<word>/<file>.wav
        audio, _ = librosa.load(wav, sr=SAMPLE_RATE, mono=True)
        audio = fit_length(audio)

        # copy the clean original too, so augmentation adds coverage without
        # discarding the ground-truth recording
        clean_dest = out_dir / rel.parent / f"{wav.stem}_clean.wav"
        clean_dest.parent.mkdir(parents=True, exist_ok=True)
        sf.write(clean_dest, audio, SAMPLE_RATE)
        n_written += 1

        for i in range(args.copies_per_file):
            aug_audio = random_augment(audio, noise_clips)
            dest = out_dir / rel.parent / f"{wav.stem}_aug{i:02d}.wav"
            sf.write(dest, aug_audio, SAMPLE_RATE)
            n_written += 1

    print(f"Wrote {n_written} clips ({args.copies_per_file} augmented + 1 clean per source) "
          f"to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", type=str, required=True,
                         help="Root of speaker recordings, e.g. ./data/speakers")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--noise_dir", type=str, default=None,
                         help="Folder of background noise WAVs to mix in "
                              "(reuse Phase 1's _background_noise_ folder)")
    parser.add_argument("--copies_per_file", type=int, default=4)
    args = parser.parse_args()
    main(args)
