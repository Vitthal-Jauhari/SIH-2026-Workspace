"""
Phase 4 - Steps 9 & 10: Continuous Audio Streaming & Trigger Policy Evaluation

Simulates continuous real-world audio stream:
- Assembles a realistic multi-minute continuous audio timeline
- Embeds positive "Vaani" utterances at known timestamps interspersed with
  conversational speech, ambient pauses, and household noise.
- Slides rolling 1.0s window with configurable hop size (100ms and 200ms).

Evaluates trigger policies:
  Policy A: Single-frame threshold (consecutive=1, cooldown=0.8s)
  Policy B: Two consecutive frames above threshold (consecutive=2, cooldown=0.8s)
  Policy C: Three consecutive frames with lower threshold (consecutive=3, th=0.35, cooldown=0.8s)

Measures:
- Detection Recall (% of embedded wake words detected)
- Missed wake words count
- Duplicate trigger count per utterance
- Detection latency (min, max, mean ms from end of utterance to detection)
- False Triggers Per Hour (FP/hour) in continuous non-target audio

Saves results to phase4/artifacts/continuous_eval.json.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import librosa

PHASE1_DIR = Path(__file__).resolve().parent.parent / "phase1"
PHASE4_DIR = Path(__file__).resolve().parent
for p in (PHASE1_DIR, PHASE4_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from features import load_and_pad  # noqa: E402
from streaming_inference import StreamingWakeWordDetector  # noqa: E402

SAMPLE_RATE = 16000


def build_continuous_test_stream(
    data_dir: Path,
    sc_dir: Path,
    duration_minutes: float = 10.0,
    n_embedded_vaani: int = 15,
) -> Tuple[np.ndarray, List[Dict[str, float]]]:
    """
    Synthesizes a continuous audio stream of duration_minutes containing
    embedded Vaani wake words at recorded timestamps, surrounded by Speech
    Commands negative words and ambient pauses.
    """
    total_samples = int(duration_minutes * 60 * SAMPLE_RATE)
    stream = np.zeros(total_samples, dtype=np.float32)

    # 1. Fill base stream with low ambient noise (room hum + gentle pink noise)
    t = np.linspace(0, duration_minutes * 60, total_samples, endpoint=False)
    hum = 0.003 * np.sin(2 * np.pi * 50 * t)
    noise = np.random.normal(0, 0.005, total_samples).astype(np.float32)
    stream += (hum + noise)

    # 2. Scatter negative speech words (non-Vaani conversations)
    sc_wavs = list(sc_dir.rglob("*.wav"))
    np.random.seed(42)
    np.random.shuffle(sc_wavs)

    # Add negative words every 3 - 6 seconds
    current_pos = int(2.0 * SAMPLE_RATE)
    sc_idx = 0
    while current_pos < total_samples - int(3.0 * SAMPLE_RATE) and sc_idx < len(sc_wavs):
        w_path = sc_wavs[sc_idx]
        sc_idx += 1
        y, sr = librosa.load(str(w_path), sr=SAMPLE_RATE, mono=True)
        length = min(len(y), total_samples - current_pos)
        stream[current_pos:current_pos + length] += y[:length] * 0.7
        current_pos += length + np.random.randint(int(1.5 * SAMPLE_RATE), int(4.0 * SAMPLE_RATE))

    # 3. Embed clean positive Vaani utterances at known, recorded intervals
    vitthal_wavs = list((data_dir / "test_unseen" / "Vitthal").glob("*.wav"))
    ishita_wavs = list((data_dir / "validation" / "Ishita").glob("*.wav"))
    umang_wavs = list((data_dir / "train" / "Umang").glob("*.wav"))
    ananya_wavs = list((data_dir / "train" / "Ananya").glob("*.wav"))
    vaani_pool = vitthal_wavs + ishita_wavs + umang_wavs + ananya_wavs
    np.random.shuffle(vaani_pool)

    # Space embedded wake words evenly
    interval = (duration_minutes * 60) / (n_embedded_vaani + 1)
    embedded_ground_truth = []

    for i in range(n_embedded_vaani):
        target_sec = (i + 1) * interval + np.random.uniform(-2.0, 2.0)
        start_sample = int(target_sec * SAMPLE_RATE)
        v_path = vaani_pool[i % len(vaani_pool)]
        y, sr = librosa.load(str(v_path), sr=SAMPLE_RATE, mono=True)

        # Trim to 1s or active segment
        if len(y) > SAMPLE_RATE:
            y = y[:SAMPLE_RATE]

        end_sample = min(start_sample + len(y), total_samples)
        v_len = end_sample - start_sample
        stream[start_sample:end_sample] += y[:v_len]

        embedded_ground_truth.append({
            "id": i + 1,
            "speaker": v_path.parent.name,
            "filename": v_path.name,
            "start_time": round(start_sample / SAMPLE_RATE, 3),
            "end_time": round(end_sample / SAMPLE_RATE, 3),
        })

    # Clip to legal float32 audio range
    stream = np.clip(stream, -1.0, 1.0)
    return stream, embedded_ground_truth


def evaluate_policy(
    tflite_path: Path,
    stream: np.ndarray,
    ground_truth: List[Dict[str, float]],
    policy_name: str,
    threshold: float,
    hop_seconds: float,
    cooldown_seconds: float,
    consecutive_frames: int,
) -> Dict[str, Any]:
    detector = StreamingWakeWordDetector(
        str(tflite_path),
        threshold=threshold,
        hop_seconds=hop_seconds,
        cooldown_seconds=cooldown_seconds,
        consecutive_frames_required=consecutive_frames,
    )

    t0 = time.perf_counter()
    triggers = detector.process_stream(stream)
    proc_time = time.perf_counter() - t0

    stream_duration_sec = len(stream) / SAMPLE_RATE
    stream_duration_hr = stream_duration_sec / 3600.0

    # Match triggers with ground truth:
    # A detection is a True Positive if it fires within [start_time, end_time + 1.2s]
    matched_gt = set()
    tp_events = []
    fp_events = []
    latencies_ms = []

    for trig in triggers:
        t_time = trig["timestamp"]
        matched = False
        for gt in ground_truth:
            # Tolerant window: trigger should occur between start of word and 1.2s after end of word
            if gt["start_time"] <= t_time <= (gt["end_time"] + 1.2):
                matched = True
                matched_gt.add(gt["id"])
                # Latency: time from end of utterance to trigger
                lat_ms = max(0.0, (t_time - gt["end_time"]) * 1000.0)
                latencies_ms.append(lat_ms)
                tp_events.append({"trigger": trig, "gt": gt, "latency_ms": lat_ms})
                break
        if not matched:
            fp_events.append(trig)

    # Duplicates check: extra triggers matching the same ground truth
    duplicate_count = max(0, len(tp_events) - len(matched_gt))
    recall = (len(matched_gt) / len(ground_truth)) * 100.0 if ground_truth else 0.0
    fp_per_hour = len(fp_events) / stream_duration_hr if stream_duration_hr > 0 else 0.0

    result = {
        "policy_name": policy_name,
        "threshold": threshold,
        "hop_seconds": hop_seconds,
        "cooldown_seconds": cooldown_seconds,
        "consecutive_frames": consecutive_frames,
        "embedded_wake_words": len(ground_truth),
        "detected_wake_words": len(matched_gt),
        "recall": round(recall, 1),
        "missed_count": len(ground_truth) - len(matched_gt),
        "duplicate_triggers": duplicate_count,
        "false_triggers": len(fp_events),
        "false_triggers_per_hour": round(fp_per_hour, 2),
        "mean_latency_ms": round(float(np.mean(latencies_ms)), 1) if latencies_ms else 0.0,
        "min_latency_ms": round(float(np.min(latencies_ms)), 1) if latencies_ms else 0.0,
        "max_latency_ms": round(float(np.max(latencies_ms)), 1) if latencies_ms else 0.0,
        "total_stream_minutes": round(stream_duration_sec / 60.0, 2),
        "realtime_factor": round((stream_duration_sec / proc_time), 1) if proc_time > 0 else 0.0,
    }
    return result


def run_continuous_benchmark(
    tflite_path: Path,
    data_dir: Path,
    sc_dir: Path,
    out_dir: Path,
    duration_minutes: float = 10.0,
):
    print("=" * 80)
    print("PHASE 4 - STEPS 9 & 10: CONTINUOUS AUDIO & TRIGGER POLICY EVALUATION")
    print("=" * 80)
    print(f"Model            : {tflite_path.name}")
    print(f"Stream Duration  : {duration_minutes} minutes")
    print("Synthesizing realistic continuous audio stream with embedded wake words...")

    stream, ground_truth = build_continuous_test_stream(
        data_dir, sc_dir, duration_minutes=duration_minutes, n_embedded_vaani=15
    )
    print(f"Stream synthesized ({len(stream):,} samples @ 16kHz). Embedded wake words: {len(ground_truth)}\n")

    policies = [
        {
            "name": "Policy A: Single-frame (th=0.40, hop=100ms, deb=0.8s)",
            "threshold": 0.40,
            "hop": 0.10,
            "cooldown": 0.80,
            "consecutive": 1,
        },
        {
            "name": "Policy A2: Single-frame (th=0.50, hop=100ms, deb=0.8s)",
            "threshold": 0.50,
            "hop": 0.10,
            "cooldown": 0.80,
            "consecutive": 1,
        },
        {
            "name": "Policy B: Two-consecutive (th=0.40, hop=100ms, deb=0.8s)",
            "threshold": 0.40,
            "hop": 0.10,
            "cooldown": 0.80,
            "consecutive": 2,
        },
        {
            "name": "Policy C: Larger hop (th=0.40, hop=200ms, deb=0.8s)",
            "threshold": 0.40,
            "hop": 0.20,
            "cooldown": 0.80,
            "consecutive": 1,
        },
    ]

    results = []
    print(f"{'Trigger Policy':<36}{'Recall':<10}{'Miss':<8}{'Duplicates':<12}{'FP/Hr':<10}{'Mean Latency'}")
    print("-" * 88)

    for pol in policies:
        res = evaluate_policy(
            tflite_path,
            stream,
            ground_truth,
            pol["name"],
            pol["threshold"],
            pol["hop"],
            pol["cooldown"],
            pol["consecutive"],
        )
        results.append(res)
        print(f"{pol['name'][:34]:<36}{res['recall']:>5.1f}%{'':<4}{res['missed_count']:<8}{res['duplicate_triggers']:<12}{res['false_triggers_per_hour']:<10.1f}{res['mean_latency_ms']:>6.1f} ms")

    print("-" * 88)

    # Identify optimal policy
    best_pol = min(results, key=lambda x: (x["false_triggers_per_hour"] - x["recall"] * 0.1))
    print(f"\nRECOMMENDED TRIGGER POLICY: {best_pol['policy_name']}")
    print(f"  - Recall            : {best_pol['recall']}%")
    print(f"  - Duplicate Triggers: {best_pol['duplicate_triggers']}")
    print(f"  - FP / Hour         : {best_pol['false_triggers_per_hour']}")
    print(f"  - Mean Latency      : {best_pol['mean_latency_ms']} ms")
    print("=" * 80 + "\n")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "continuous_eval.json"
    with open(out_file, "w") as f:
        json.dump({
            "stream_duration_minutes": duration_minutes,
            "embedded_wake_words": len(ground_truth),
            "recommended_policy": best_pol["policy_name"],
            "policy_results": results,
        }, f, indent=2)

    return results, best_pol


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tflite_path", type=str, default="../phase3/artifacts/vaani_int8.tflite")
    parser.add_argument("--data_dir", type=str, default="../phase3/data")
    parser.add_argument("--sc_dir", type=str, default="../phase3/data/sc_raw/mini_speech_commands")
    parser.add_argument("--out_dir", type=str, default="./artifacts")
    parser.add_argument("--duration_minutes", type=float, default=10.0)
    args = parser.parse_args()

    run_continuous_benchmark(
        Path(args.tflite_path),
        Path(args.data_dir),
        Path(args.sc_dir),
        Path(args.out_dir),
        args.duration_minutes,
    )


if __name__ == "__main__":
    main()
