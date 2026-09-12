"""
Phase 4 - Steps 7 & 8: Negative Testing & Comprehensive Threshold Sweep

Evaluates the quantized INT8 model across a fine-grained threshold grid:
[0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]

Datasets evaluated:
1. Positive Vaani pool:
   - Unseen Vitthal (26)
   - Validation Ishita (42)
   - Seen speakers Ananya (39), Umang (50), Ark (49)
   - Total: 206 clean Vaani recordings
2. Negative non-target pool:
   - 800+ Speech Commands non-target speech clips
   - 200+ Ambient room noise and silence clips
   - Total: 1,000+ negative test clips

Calculates:
- TPR (Recall) on Unseen Vitthal and Overall Vaani
- FPR (False Positive Rate) on Negative Pool
- Precision, F1-Score, FNR
- Trade-off analysis for ESP32 deployment

Saves results to phase4/artifacts/threshold_sweep.json.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any

import numpy as np

PHASE1_DIR = Path(__file__).resolve().parent.parent / "phase1"
if str(PHASE1_DIR) not in sys.path:
    sys.path.insert(0, str(PHASE1_DIR))

from features import extract_mfcc, load_and_pad  # noqa: E402
from model import LABELS  # noqa: E402


def run_threshold_sweep(
    tflite_path: Path,
    data_dir: Path,
    sc_dir: Path,
    out_dir: Path,
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

    def get_vaani_conf(wav_path: str) -> float:
        audio = load_and_pad(str(wav_path))
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
        return float(p[vaani_idx])

    print("=" * 80)
    print("PHASE 4 - STEP 8: COMPREHENSIVE THRESHOLD OPTIMIZATION SWEEP")
    print("=" * 80)
    print(f"Model            : {tflite_path.name}")

    # Gather all positive recordings
    vitthal_wavs = list((data_dir / "test_unseen" / "Vitthal").glob("*.wav"))
    ishita_wavs = list((data_dir / "validation" / "Ishita").glob("*.wav"))
    other_wavs = list((data_dir / "train").rglob("*.wav"))
    all_vaani_wavs = vitthal_wavs + ishita_wavs + other_wavs

    # Gather negative pool
    # Speech commands words + silence from combined dataset
    neg_speech_wavs = list(sc_dir.rglob("*.wav"))[:800]
    silence_wavs = list((data_dir / "combined").rglob("silence_*.wav"))[:250]
    all_neg_wavs = neg_speech_wavs + silence_wavs

    print(f"Positive Vaani Clips : {len(all_vaani_wavs)} (Vitthal: {len(vitthal_wavs)}, Ishita: {len(ishita_wavs)})")
    print(f"Negative Clips Pool  : {len(all_neg_wavs)} ({len(neg_speech_wavs)} speech words + {len(silence_wavs)} silence)\n")

    print("Computing inference confidences across dataset...")
    vitthal_confs = [get_vaani_conf(str(w)) for w in vitthal_wavs]
    all_vaani_confs = [get_vaani_conf(str(w)) for w in all_vaani_wavs]
    neg_confs = [get_vaani_conf(str(w)) for w in all_neg_wavs]

    thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]

    sweep_table = []
    print(f"{'Threshold':<11}{'Vitthal Rec':<14}{'Overall Rec':<14}{'FPR':<12}{'Precision':<12}{'F1-Score':<12}{'FNR'}")
    print("-" * 80)

    for th in thresholds:
        # Unseen Vitthal
        v_tp = sum(1 for c in vitthal_confs if c >= th)
        v_rec = (v_tp / len(vitthal_confs)) * 100.0

        # Overall Vaani
        all_tp = sum(1 for c in all_vaani_confs if c >= th)
        all_rec = (all_tp / len(all_vaani_confs)) * 100.0
        all_fn = len(all_vaani_confs) - all_tp
        all_fnr = (all_fn / len(all_vaani_confs)) * 100.0

        # Negatives
        fp = sum(1 for c in neg_confs if c >= th)
        fpr = (fp / len(neg_confs)) * 100.0

        precision = (all_tp / (all_tp + fp)) * 100.0 if (all_tp + fp) > 0 else 0.0
        f1 = (2 * precision * all_rec) / (precision + all_rec) if (precision + all_rec) > 0 else 0.0

        row = {
            "threshold": th,
            "vitthal_recall": round(v_rec, 1),
            "overall_recall": round(all_rec, 1),
            "fpr": round(fpr, 2),
            "false_positives": fp,
            "total_negatives": len(neg_confs),
            "precision": round(precision, 1),
            "f1_score": round(f1, 1),
            "fnr": round(all_fnr, 1),
        }
        sweep_table.append(row)
        print(f"{th:<11.2f}{v_rec:>6.1f}%{'':<7}{all_rec:>6.1f}%{'':<7}{fpr:>5.2f}%{'':<6}{precision:>6.1f}%{'':<5}{f1:>6.1f}%{'':<5}{all_fnr:>5.1f}%")

    print("-" * 80)

    # Threshold selection analysis
    # Trade-off considerations:
    # - At 0.40: Vitthal recall is 100.0%, FPR is 0.48% (5 / 1050 false triggers), F1 is 90.9%
    # - At 0.45: Vitthal recall is 92.3%, FPR is 0.29%, F1 is 90.8%
    # - At 0.50: Vitthal recall is 84.6%, FPR is 0.19%, F1 is 87.8%
    # - At 0.60: Vitthal recall is 76.9%, FPR is 0.00% (0 false triggers), F1 is 86.6%

    best_f1_row = max(sweep_table, key=lambda x: x["f1_score"])
    print(f"\nOPTIMAL BALANCED THRESHOLD (Max F1): {best_f1_row['threshold']:.2f}")
    print(f"  - Unseen Vitthal Recall: {best_f1_row['vitthal_recall']}%")
    print(f"  - FPR                  : {best_f1_row['fpr']}% ({best_f1_row['false_positives']} / {best_f1_row['total_negatives']})")
    print(f"  - Precision            : {best_f1_row['precision']}%")
    print(f"  - F1-Score             : {best_f1_row['f1_score']}%")
    print("=" * 80 + "\n")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "threshold_sweep.json"
    with open(out_file, "w") as f:
        json.dump({
            "total_positives": len(all_vaani_wavs),
            "total_negatives": len(all_neg_wavs),
            "optimal_threshold": best_f1_row["threshold"],
            "sweep_results": sweep_table,
        }, f, indent=2)

    return sweep_table, best_f1_row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tflite_path", type=str, default="../phase3/artifacts/vaani_int8.tflite")
    parser.add_argument("--data_dir", type=str, default="../phase3/data")
    parser.add_argument("--sc_dir", type=str, default="../phase3/data/sc_raw/mini_speech_commands")
    parser.add_argument("--out_dir", type=str, default="./artifacts")
    args = parser.parse_args()

    run_threshold_sweep(
        Path(args.tflite_path),
        Path(args.data_dir),
        Path(args.sc_dir),
        Path(args.out_dir),
    )


if __name__ == "__main__":
    main()
