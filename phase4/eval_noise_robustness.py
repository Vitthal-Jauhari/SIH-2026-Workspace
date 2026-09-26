"""
Phase 4 - Step 3: Comprehensive Noise Robustness Evaluation

Evaluates the quantized INT8 TFLite model on clean positive Vaani recordings
under various realistic acoustic noise conditions and SNR levels:
Noise types:
- fan (low-frequency blade hum + drone)
- keyboard_typing (mechanical typing transients)
- ambient_room (room air + 50Hz electrical hum)
- traffic (low rumble with passing vehicular AM modulation)
- speech_babble (multi-speaker cocktail party chatter)
- pink_noise (1/f spectral ambient decay)

SNR levels: Clean, 25 dB, 20 dB, 15 dB, 10 dB, 5 dB, 0 dB.

Measures:
- Vaani detection rate (Recall)
- False Negative Rate (FNR)
- Mean Vaani confidence
- Failure analysis per noise type

Saves results to phase4/artifacts/noise_evaluation.json.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy.signal import lfilter

PHASE1_DIR = Path(__file__).resolve().parent.parent / "phase1"
if str(PHASE1_DIR) not in sys.path:
    sys.path.insert(0, str(PHASE1_DIR))

from features import extract_mfcc, load_and_pad  # noqa: E402
from model import LABELS  # noqa: E402

SAMPLE_RATE = 16000
CLIP_LEN = 16000


def generate_synthetic_noise(noise_type: str, length: int) -> np.ndarray:
    """Generates realistic acoustic noise profiles."""
    t = np.linspace(0, length / SAMPLE_RATE, length, endpoint=False)
    np.random.seed(int(length) % 10000 + hash(noise_type) % 1000)

    if noise_type == "fan":
        # 120Hz fundamental + 240Hz blade harmonic + pink low-pass noise
        hum = 0.5 * np.sin(2 * np.pi * 120 * t) + 0.3 * np.sin(2 * np.pi * 240 * t)
        white = np.random.normal(0, 0.4, length)
        # Low-pass filter below 800 Hz
        b = [0.05, 0.1, 0.05]
        a = [1.0, -1.2, 0.4]
        lp_noise = lfilter(b, a, white)
        return (hum + lp_noise).astype(np.float32)

    elif noise_type == "keyboard_typing":
        # Ambient background + sharp mechanical click transients
        bg = np.random.normal(0, 0.1, length)
        clicks = np.zeros(length, dtype=np.float32)
        # Add random typing clicks every 0.15 - 0.35 seconds
        click_positions = np.random.choice(range(500, length - 1000), size=max(1, int(length / 4000)), replace=False)
        for pos in click_positions:
            click_len = min(400, length - pos)
            tc = np.linspace(0, 0.025, click_len)
            click_env = np.exp(-tc * 250) * np.sin(2 * np.pi * 2500 * tc)
            clicks[pos:pos + click_len] += click_env.astype(np.float32)
        return (bg + clicks).astype(np.float32)

    elif noise_type == "ambient_room":
        # 50Hz mains hum + gentle room rumble
        hum = 0.3 * np.sin(2 * np.pi * 50 * t) + 0.15 * np.sin(2 * np.pi * 100 * t)
        hiss = np.random.normal(0, 0.3, length)
        return (hum + hiss).astype(np.float32)

    elif noise_type == "traffic":
        # Low rumble modulated by slow AM (cars passing)
        slow_am = 0.5 + 0.5 * np.sin(2 * np.pi * 0.3 * t)
        rumble = np.random.normal(0, 0.5, length)
        # Low-pass filter below 400 Hz
        b = [0.02, 0.04, 0.02]
        a = [1.0, -1.5, 0.58]
        lp_rumble = lfilter(b, a, rumble)
        return (slow_am * lp_rumble).astype(np.float32)

    elif noise_type == "speech_babble":
        # Formant cluster noise simulating distant background conversation
        b1 = np.sin(2 * np.pi * 500 * t) * np.random.normal(0, 0.3, length)
        b2 = np.sin(2 * np.pi * 1500 * t) * np.random.normal(0, 0.2, length)
        b3 = np.sin(2 * np.pi * 2500 * t) * np.random.normal(0, 0.15, length)
        return (b1 + b2 + b3).astype(np.float32)

    elif noise_type == "pink_noise":
        # 1/f filtered noise
        white = np.random.normal(0, 1.0, length)
        b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
        a = [1, -2.494956002, 2.017265875, -0.522189400]
        return lfilter(b, a, white).astype(np.float32)

    else:
        return np.random.normal(0, 0.5, length).astype(np.float32)


def mix_noise(clean_audio: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Mixes noise into clean audio at specified SNR (dB)."""
    if len(noise) < len(clean_audio):
        noise = np.pad(noise, (0, len(clean_audio) - len(noise)), mode="wrap")
    else:
        noise = noise[:len(clean_audio)]

    sig_pwr = np.mean(clean_audio ** 2) + 1e-10
    noise_pwr = np.mean(noise ** 2) + 1e-10
    target_noise_pwr = sig_pwr / (10 ** (snr_db / 10))
    scale = np.sqrt(target_noise_pwr / noise_pwr)
    noisy_audio = clean_audio + noise * scale
    return np.clip(noisy_audio, -1.0, 1.0).astype(np.float32)


def run_noise_benchmark(
    tflite_path: Path,
    data_dir: Path,
    out_dir: Path,
    threshold: float = 0.40,
):
    import tensorflow as tf

    tflite_path = tflite_path.resolve()
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    inp_det = interpreter.get_input_details()[0]
    out_det = interpreter.get_output_details()[0]
    in_scale, in_zp = inp_det["quantization"]
    out_scale, out_zp = out_det["quantization"]
    vaani_idx = LABELS.index("vaani")

    def predict_audio(audio: np.ndarray):
        feat = extract_mfcc(audio)
        feat = np.expand_dims(feat, (0, -1)).astype(np.float32)
        q = feat / in_scale + in_zp if in_scale > 0 else feat
        q = np.clip(np.round(q), -128, 127).astype(np.int8)
        interpreter.set_tensor(inp_det["index"], q)
        interpreter.invoke()
        raw = interpreter.get_tensor(out_det["index"])[0]
        p = (raw.astype(np.float32) - out_zp) * out_scale if out_scale > 0 else raw.astype(np.float32)
        p = np.maximum(p, 0.0)
        p = p / max(p.sum(), 1e-8)
        return LABELS[int(np.argmax(p))], float(p[vaani_idx])

    # Load clean test audio:
    # We evaluate Unseen Vitthal (26), Validation Ishita (42), and sample from seen Umang/Ananya
    test_unseen_wavs = list((data_dir / "test_unseen" / "Vitthal").glob("*.wav"))
    val_wavs = list((data_dir / "validation" / "Ishita").glob("*.wav"))

    eval_sets = {
        "unseen_vitthal": [load_and_pad(str(w)) for w in test_unseen_wavs],
        "validation_ishita": [load_and_pad(str(w)) for w in val_wavs],
    }

    noise_types = ["fan", "keyboard_typing", "ambient_room", "traffic", "speech_babble", "pink_noise"]
    snr_levels = [25, 20, 15, 10, 5, 0]

    results_table = []
    print("=" * 75)
    print("PHASE 4 - STEP 3: NOISE ROBUSTNESS BENCHMARK")
    print("=" * 75)
    print(f"Evaluated Model  : {tflite_path.name}")
    print(f"Trigger Threshold: {threshold}")
    print(f"Target Speakers  : Vitthal (Unseen, N={len(test_unseen_wavs)}), Ishita (Validation, N={len(val_wavs)})\n")

    # Baseline Clean Performance
    clean_audio_pool = eval_sets["unseen_vitthal"]
    clean_hits = sum(1 for a in clean_audio_pool if predict_audio(a)[1] >= threshold)
    clean_recall = (clean_hits / len(clean_audio_pool)) * 100.0 if clean_audio_pool else 0.0
    print(f"CLEAN BASELINE RECALL (Vitthal @ th={threshold}): {clean_recall:.1f}%\n")

    print(f"{'Noise Condition':<20}{'SNR':<10}{'Unseen Recall':<18}{'Val Recall':<16}{'Mean Conf'}")
    print("-" * 75)

    noise_summary: Dict[str, Dict[str, Any]] = {}

    for ntype in noise_types:
        noise_summary[ntype] = {}
        for snr in snr_levels:
            # Generate continuous noise buffer
            noise_buf = generate_synthetic_noise(ntype, length=32000)

            # Evaluate on Unseen Vitthal
            vitthal_hits = 0
            vitthal_confs = []
            for a in eval_sets["unseen_vitthal"]:
                noisy_a = mix_noise(a, noise_buf, snr)
                _, conf = predict_audio(noisy_a)
                vitthal_confs.append(conf)
                if conf >= threshold:
                    vitthal_hits += 1
            unseen_rec = (vitthal_hits / len(eval_sets["unseen_vitthal"])) * 100.0

            # Evaluate on Validation Ishita
            val_hits = 0
            for a in eval_sets["validation_ishita"]:
                noisy_a = mix_noise(a, noise_buf, snr)
                _, conf = predict_audio(noisy_a)
                if conf >= threshold:
                    val_hits += 1
            val_rec = (val_hits / len(eval_sets["validation_ishita"])) * 100.0

            mean_c = float(np.mean(vitthal_confs)) * 100.0

            row = {
                "noise_type": ntype,
                "snr_db": snr,
                "unseen_vitthal_recall": round(unseen_rec, 1),
                "validation_ishita_recall": round(val_rec, 1),
                "unseen_mean_confidence": round(mean_c, 1),
            }
            results_table.append(row)
            noise_summary[ntype][f"{snr}dB"] = row

            print(f"{ntype:<20}{str(snr) + ' dB':<10}{unseen_rec:>6.1f}%{'':<11}{val_rec:>6.1f}%{'':<9}{mean_c:>6.1f}%")

    print("-" * 75)

    # Find worst condition
    worst = min(results_table, key=lambda x: x["unseen_vitthal_recall"])
    print(f"\nWORST NOISE CONDITION: {worst['noise_type']} @ {worst['snr_db']} dB SNR -> Recall: {worst['unseen_vitthal_recall']}%")
    print("=" * 75 + "\n")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "noise_evaluation.json"
    with open(out_file, "w") as f:
        json.dump({
            "threshold": threshold,
            "clean_baseline_recall": clean_recall,
            "worst_condition": worst,
            "detailed_results": results_table,
        }, f, indent=2)

    return results_table, worst


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tflite_path", type=str, default="../phase3/artifacts/vaani_int8.tflite")
    parser.add_argument("--data_dir", type=str, default="../phase3/data")
    parser.add_argument("--out_dir", type=str, default="./artifacts")
    parser.add_argument("--threshold", type=float, default=0.40)
    args = parser.parse_args()

    run_noise_benchmark(
        Path(args.tflite_path),
        Path(args.data_dir),
        Path(args.out_dir),
        args.threshold,
    )


if __name__ == "__main__":
    main()
