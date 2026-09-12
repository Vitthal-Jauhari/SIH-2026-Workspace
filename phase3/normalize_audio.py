"""
Phase 3 - Step 1: Audio Normalization

Converts real Vaani recordings from various audio formats (M4A, MP3, AAC, and unusual extensions)
into a standardized format:
- WAV
- mono
- 16 kHz
- 16-bit PCM (PCM_16)
- Preserves speaker identity in subdirectory
- Preserves source samples as positive 'vaani' wake-word samples

Usage:
    python normalize_audio.py --in_dir ./Audio --out_dir ./data/normalized
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import soundfile as sf

try:
    import av
except ImportError:
    print("ERROR: 'av' (PyAV) is required for audio decoding. Install via: pip install av")
    sys.exit(1)

TARGET_SAMPLE_RATE = 16000


def decode_audio_robust(file_path: Path):
    """
    Decodes an audio file of any format (m4a, mp3, aac, raw/unusual extensions)
    using PyAV, resampling to 16kHz mono. Handles trailing metadata (e.g. Samsung
    voice recorder waveform/bookmark JSON tags) gracefully.
    
    Returns:
        audio (np.ndarray): 1D float32 numpy array normalized to [-1.0, 1.0]
        sr (int): 16000
        orig_sr (int): original sample rate
        orig_channels (int): original channel count
    """
    container = av.open(str(file_path))
    audio_stream = next((s for s in container.streams if s.type == "audio"), None)
    if not audio_stream:
        container.close()
        raise ValueError("No audio stream found in file")

    orig_sr = audio_stream.rate or audio_stream.codec_context.sample_rate or TARGET_SAMPLE_RATE
    orig_channels = audio_stream.channels or audio_stream.codec_context.channels or 1

    resampler = av.AudioResampler(format="fltp", layout="mono", rate=TARGET_SAMPLE_RATE)
    frames = []

    try:
        for frame in container.decode(audio_stream):
            frame.pts = None
            resampled_frames = resampler.resample(frame)
            if resampled_frames:
                for rf in resampled_frames:
                    frames.append(rf.to_ndarray())
    except Exception as e:
        # Trailing garbage or recorder metadata encountered after audio packets
        pass
    finally:
        container.close()

    if not frames:
        raise ValueError("No valid audio frames decoded")

    audio = np.concatenate(frames, axis=1)[0]
    # Ensure float32 in [-1.0, 1.0]
    audio = np.clip(audio, -1.0, 1.0).astype(np.float32)
    return audio, TARGET_SAMPLE_RATE, orig_sr, orig_channels


def normalize_dataset(in_dir: Path, out_dir: Path, expected_count: int = 206):
    in_dir = in_dir.resolve()
    out_dir = out_dir.resolve()

    if not in_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {in_dir}")

    # Ensure clean output directory
    if out_dir.exists():
        import shutil
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Discover all audio files per speaker
    speaker_dirs = sorted([d for d in in_dir.iterdir() if d.is_dir()])
    if not speaker_dirs:
        raise FileNotFoundError(f"No speaker directories found in {in_dir}")

    print("=" * 65)
    print("PHASE 3 - STEP 1: AUDIO NORMALIZATION AUDIT")
    print("=" * 65)
    print(f"Input source directory : {in_dir}")
    print(f"Output normalized dir  : {out_dir}")
    print(f"Target format          : WAV, mono, {TARGET_SAMPLE_RATE} Hz, PCM_16\n")

    total_found = 0
    total_converted = 0
    total_failed = 0
    failed_files = []

    orig_sr_stats = {}
    orig_ch_stats = {}
    per_speaker_stats = {}

    for s_dir in speaker_dirs:
        speaker_name = s_dir.name
        files = sorted([f for f in s_dir.iterdir() if f.is_file()])
        per_speaker_stats[speaker_name] = {
            "found": len(files),
            "converted": 0,
            "failed": 0,
            "durations": [],
            "extensions": {},
        }
        total_found += len(files)

        spk_out_dir = out_dir / speaker_name
        spk_out_dir.mkdir(parents=True, exist_ok=True)

        for f in files:
            ext = f.suffix.lower() if f.suffix else "(no-ext)"
            per_speaker_stats[speaker_name]["extensions"][ext] = (
                per_speaker_stats[speaker_name]["extensions"].get(ext, 0) + 1
            )

            # Avoid collision for unusual extensions (e.g. Recording.10, Recording.15)
            standard_exts = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg"}
            if ext in standard_exts:
                out_name = f"{f.stem}.wav"
            else:
                sanitized = f.name.replace(".", "_")
                out_name = f"{sanitized}.wav"

            out_wav = spk_out_dir / out_name

            try:
                audio, sr, orig_sr, orig_ch = decode_audio_robust(f)
                dur = len(audio) / sr

                # Write as standard 16-bit PCM WAV
                sf.write(str(out_wav), audio, sr, subtype="PCM_16")

                total_converted += 1
                per_speaker_stats[speaker_name]["converted"] += 1
                per_speaker_stats[speaker_name]["durations"].append(dur)

                orig_sr_stats[orig_sr] = orig_sr_stats.get(orig_sr, 0) + 1
                orig_ch_stats[orig_ch] = orig_ch_stats.get(orig_ch, 0) + 1

            except Exception as e:
                total_failed += 1
                per_speaker_stats[speaker_name]["failed"] += 1
                failed_files.append((str(f), str(e)))

    # Print Detailed Statistics
    print(f"{'Speaker':<12}{'Found':<8}{'Converted':<12}{'Failed':<8}{'Dur (min/mean/max)':<22}{'Extensions'}")
    print("-" * 75)
    all_durations = []
    for spk, stats in per_speaker_stats.items():
        durs = stats["durations"]
        all_durations.extend(durs)
        dur_str = f"{min(durs):.2f} / {np.mean(durs):.2f} / {max(durs):.2f}s" if durs else "N/A"
        ext_str = ", ".join(f"{k}:{v}" for k, v in stats["extensions"].items())
        print(f"{spk:<12}{stats['found']:<8}{stats['converted']:<12}{stats['failed']:<8}{dur_str:<22}{ext_str}")

    print("-" * 75)
    print(f"TOTAL FOUND     : {total_found}")
    print(f"TOTAL CONVERTED : {total_converted}")
    print(f"TOTAL FAILED    : {total_failed}")
    if all_durations:
        print(f"Overall Duration: min={min(all_durations):.2f}s, mean={np.mean(all_durations):.2f}s, "
              f"max={max(all_durations):.2f}s, total={sum(all_durations)/60:.2f} min")
    print(f"Original SRs    : {orig_sr_stats}")
    print(f"Original Channels: {orig_ch_stats}")
    print("=" * 65)

    if failed_files:
        print("\nFailed files:")
        for f, err in failed_files:
            print(f"  - {f}: {err}")

    # Guardrail Assertion
    actual_on_disk = len(list(out_dir.rglob("*.wav")))
    if expected_count is not None and (total_converted != expected_count or actual_on_disk != expected_count):
        msg = f"ERROR: Expected {expected_count} normalized files, but converted {total_converted} (on disk: {actual_on_disk})!"
        print(msg, file=sys.stderr)
        raise RuntimeError(msg)

    if total_failed > 0:
        msg = f"ERROR: Normalization failed on {total_failed} files!"
        print(msg, file=sys.stderr)
        raise RuntimeError(msg)

    print(f"[SUCCESS] All {actual_on_disk} files verified on disk ({expected_count}/{expected_count}) in {out_dir}\n")
    return per_speaker_stats


def main():
    parser = argparse.ArgumentParser(description="Standardize all Vaani recordings to 16kHz mono PCM WAV.")
    parser.add_argument("--in_dir", type=str, default="./Audio", help="Path to raw Audio/ directory")
    parser.add_argument("--out_dir", type=str, default="./data/normalized", help="Path to save normalized WAVs")
    parser.add_argument("--expected_count", type=int, default=206, help="Expected number of recordings (default: 206)")
    args = parser.parse_args()

    normalize_dataset(Path(args.in_dir), Path(args.out_dir), args.expected_count)


if __name__ == "__main__":
    main()
