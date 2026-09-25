#!/usr/bin/env python3
"""
trim_speech.py

Batch-trims a folder of audio clips down to a 1-second window that contains
the spoken word utterance, discarding leading/trailing silence and noise.

How it distinguishes speech from noise
---------------------------------------
Instead of a naive volume/energy threshold (which treats any loud noise as
"speech"), this uses Silero VAD — a small neural voice-activity-detection
model trained specifically to separate human speech from silence, background
noise, clicks, static, etc. This is far more reliable than energy-based
gating for real-world recordings.

What it does per file
----------------------
1. Load the audio, downmix to mono, resample to 16kHz (what the VAD expects).
2. Run VAD to get speech timestamp(s) inside the clip.
3. Pick the region of interest:
     - "longest" (default): the single longest continuous speech segment
       (robust against stray noise blips being misdetected far from the
       real word).
     - "span": from the start of the first speech segment to the end of the
       last one (use this if a word might legitimately be split into two
       detected segments).
4. Build a 1-second window around that region:
     - If the detected speech is shorter than 1s, pad symmetrically with
       audio on both sides (falling back to trailing/leading silence at
       the edges of the clip, exactly like your "word spoken at the end
       -> keep the last 1s" example).
     - If the detected speech is longer than 1s, center-crop it to 1s.
5. Write the result to the output folder as a 16kHz mono WAV.
6. Files with no detected speech (pure noise / silence) are skipped and
   reported, not silently dropped.

Install
-------
    pip install torch torchaudio soundfile numpy librosa pydub

(torch.hub downloads the small Silero VAD model on first run — a few MB,
cached locally after that.)

M4A / AAC / MP3 support
------------------------
`soundfile` (libsndfile) only natively decodes WAV/FLAC/OGG. For M4A, AAC,
MP3, etc. this script automatically falls back to `pydub`, which shells out
to `ffmpeg`. You need ffmpeg installed and on your PATH:
    Windows (with winget):  winget install Gyan.FFmpeg
    Windows (with choco):   choco install ffmpeg
    macOS:                  brew install ffmpeg
    Linux:                  sudo apt install ffmpeg
Verify with `ffmpeg -version` in a new terminal after installing.

Usage
-----
    python trim_speech.py --input_dir ./raw_clips --output_dir ./trimmed_clips

    # tune sensitivity / behavior
    python trim_speech.py --input_dir raw --output_dir trimmed \
        --threshold 0.5 --min_speech_ms 100 --strategy longest
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

TARGET_SR = 16000
TARGET_DURATION = 1.0  # seconds
TARGET_SAMPLES = int(TARGET_SR * TARGET_DURATION)


def load_vad_model():
    """Loads Silero VAD via torch.hub (cached after first download)."""
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        onnx=False,
        trust_repo=True,
    )
    get_speech_timestamps = utils[0]
    return model, get_speech_timestamps


def _load_with_pydub(filepath):
    """Fallback loader for formats libsndfile can't read (M4A, AAC, MP3, ...).
    Requires ffmpeg on PATH. Returns (float32 mono array in [-1, 1], sample rate)."""
    from pydub import AudioSegment

    seg = AudioSegment.from_file(str(filepath))
    sr = seg.frame_rate
    samples = np.array(seg.get_array_of_samples())

    if seg.channels > 1:
        samples = samples.reshape((-1, seg.channels)).mean(axis=1)

    # Normalize integer PCM samples to float32 range [-1, 1].
    max_val = float(1 << (8 * seg.sample_width - 1))
    audio = samples.astype(np.float32) / max_val
    return audio, sr


def load_audio_mono(filepath):
    """Loads audio, downmixes to mono, returns (float32 array, sample rate).
    Tries soundfile first (fast, no external deps); falls back to
    pydub/ffmpeg for formats like M4A/AAC/MP3 that libsndfile can't decode."""
    try:
        audio, sr = sf.read(str(filepath), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return audio, sr
    except Exception:
        return _load_with_pydub(filepath)


def resample_if_needed(audio, sr, target_sr=TARGET_SR):
    if sr == target_sr:
        return audio, sr
    import librosa
    resampled = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return resampled, target_sr


def get_speech_region(audio, sr, model, get_speech_timestamps, threshold, strategy):
    """
    Runs VAD and returns (start_sample, end_sample) for the region of
    interest, or None if no speech was detected.
    """
    audio_tensor = torch.from_numpy(audio).float()
    timestamps = get_speech_timestamps(
        audio_tensor, model, sampling_rate=sr, threshold=threshold
    )
    if not timestamps:
        return None

    if strategy == "span":
        start = timestamps[0]["start"]
        end = timestamps[-1]["end"]
    else:  # "longest"
        longest = max(timestamps, key=lambda t: t["end"] - t["start"])
        start, end = longest["start"], longest["end"]

    return start, end


def build_one_second_window(audio, start, end, target_samples=TARGET_SAMPLES):
    """
    Given a detected speech region [start, end) in samples, returns a
    target_samples-long slice of `audio` that contains that region,
    centering it where possible and clamping/padding at the clip edges.
    """
    total_len = len(audio)
    speech_len = end - start

    if speech_len >= target_samples:
        # Speech itself is >= 1s: center-crop the window on the speech.
        center = (start + end) // 2
        new_start = center - target_samples // 2
    else:
        pad_total = target_samples - speech_len
        pad_left = pad_total // 2
        new_start = start - pad_left

    # Clamp to the clip's actual bounds.
    new_start = max(0, new_start)
    new_end = new_start + target_samples
    if new_end > total_len:
        new_end = total_len
        new_start = max(0, new_end - target_samples)

    clip = audio[new_start:new_end]

    if len(clip) < target_samples:
        # Source clip itself is under 1s total -> pad with silence.
        clip = np.pad(clip, (0, target_samples - len(clip)))

    return clip


def process_file(filepath, out_path, model, get_speech_timestamps,
                  threshold, min_speech_ms, strategy):
    try:
        audio, sr = load_audio_mono(filepath)
    except Exception as e:
        print(f"[ERROR reading] {filepath.name}: {e}")
        return "error"

    if len(audio) == 0:
        print(f"[SKIP - empty file] {filepath.name}")
        return "skipped"

    audio, sr = resample_if_needed(audio, sr, TARGET_SR)

    region = get_speech_region(audio, sr, model, get_speech_timestamps,
                                threshold, strategy)
    if region is None:
        print(f"[SKIP - no speech detected, likely pure noise/silence] {filepath.name}")
        return "skipped"

    start, end = region
    min_speech_samples = int((min_speech_ms / 1000) * sr)
    if (end - start) < min_speech_samples:
        print(f"[SKIP - detected speech too short ({(end-start)/sr*1000:.0f}ms), "
              f"treating as noise] {filepath.name}")
        return "skipped"

    clip = build_one_second_window(audio, start, end)
    sf.write(str(out_path), clip, sr)
    print(f"[OK] {filepath.name} -> {out_path.name}  "
          f"(speech {start/sr:.2f}s-{end/sr:.2f}s)")
    return "ok"


def main():
    parser = argparse.ArgumentParser(
        description="Trim audio clips to 1s windows containing the spoken word, "
                    "using VAD to tell speech apart from noise."
    )
    parser.add_argument("--input_dir", required=True, help="Folder of input audio clips")
    parser.add_argument("--output_dir", required=True, help="Folder to write trimmed clips to")
    parser.add_argument("--threshold", type=float, default=0.5,
                         help="VAD speech-probability threshold, 0-1 (default 0.5). "
                              "Raise it if noise is being misdetected as speech.")
    parser.add_argument("--min_speech_ms", type=int, default=100,
                         help="Minimum detected-speech duration (ms) to keep; "
                              "anything shorter is treated as noise (default 100)")
    parser.add_argument("--strategy", choices=["longest", "span"], default="longest",
                         help="How to pick the region if VAD finds multiple speech "
                              "segments (default: longest)")
    parser.add_argument("--extensions", nargs="+",
                         default=[".wav", ".flac", ".ogg", ".mp3"],
                         help="Audio file extensions to process")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if not input_dir.is_dir():
        sys.exit(f"Input directory not found: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Silero VAD model...")
    model, get_speech_timestamps = load_vad_model()

    files = sorted(
        f for f in input_dir.iterdir()
        if f.is_file() and f.suffix.lower() in args.extensions
    )
    print(f"Found {len(files)} audio file(s) in {input_dir}\n")

    counts = {"ok": 0, "skipped": 0, "error": 0}
    for f in files:
        out_path = output_dir / f.with_suffix(".wav").name
        result = process_file(
            f, out_path, model, get_speech_timestamps,
            args.threshold, args.min_speech_ms, args.strategy
        )
        counts[result] += 1

    print(f"\nDone. {counts['ok']} trimmed, {counts['skipped']} skipped "
          f"(no/too-short speech), {counts['error']} failed to read.")


if __name__ == "__main__":
    main()