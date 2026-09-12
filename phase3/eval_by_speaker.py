"""
Phase 3 - Step 7: Speaker Generalization & False-Positive Evaluation

Evaluates the quantized INT8 TFLite model on CLEAN (non-augmented) recordings:
1. Seen training speakers: Ananya, Ark, Umang
2. Validation speaker: Ishita
3. Unseen test speaker: Vitthal
4. Negative classes: Speech Commands 'unknown' + ambient 'silence'

Computes:
- Per-speaker accuracy
- Mean seen-speaker accuracy
- Mean validation accuracy
- Mean unseen-speaker accuracy
- Generalization gap (Seen vs. Unseen)
- True Positive Rate (TPR), False Positive Rate (FPR), False Negative Rate (FNR)
- Threshold sweep for wake-word sensitivity
- 3x3 Confusion Matrix

Usage:
    python eval_by_speaker.py --tflite_path ./artifacts/vaani_int8.tflite \
                              --data_dir ./data \
                              --combined_dir ./data/combined \
                              --out_dir ./artifacts
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

PHASE1_DIR = Path(__file__).resolve().parent.parent / "phase1"
if str(PHASE1_DIR) not in sys.path:
    sys.path.insert(0, str(PHASE1_DIR))

from features import extract_mfcc, get_feature_shape, load_and_pad  # noqa: E402
from model import LABELS  # noqa: E402


class TFLiteWakeWordEvaluator:
    def __init__(self, tflite_path: str):
        self.tflite_path = Path(tflite_path).resolve()
        if not self.tflite_path.exists():
            raise FileNotFoundError(f"TFLite model not found: {self.tflite_path}")

        self.interpreter = tf.lite.Interpreter(model_path=str(self.tflite_path))
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()[0]

        self.in_scale, self.in_zero_point = self.input_details["quantization"]
        self.out_scale, self.out_zero_point = self.output_details["quantization"]
        self.labels = LABELS
        self.vaani_idx = self.labels.index("vaani")

    def _quantize_input(self, feat: np.ndarray) -> np.ndarray:
        if self.in_scale == 0:
            return feat.astype(self.input_details["dtype"])
        q = feat / self.in_scale + self.in_zero_point
        q = np.clip(np.round(q), -128, 127)
        return q.astype(np.int8)

    def _dequantize_output(self, raw_out: np.ndarray) -> np.ndarray:
        if self.out_scale == 0:
            return raw_out.astype(np.float32)
        return (raw_out.astype(np.float32) - self.out_zero_point) * self.out_scale

    def predict_file(self, wav_path: str):
        audio = load_and_pad(str(wav_path))
        feat = extract_mfcc(audio)
        feat = np.expand_dims(feat, (0, -1)).astype(np.float32)
        q_feat = self._quantize_input(feat)

        t0 = time.perf_counter()
        self.interpreter.set_tensor(self.input_details["index"], q_feat)
        self.interpreter.invoke()
        raw_out = self.interpreter.get_tensor(self.output_details["index"])
        latency_ms = (time.perf_counter() - t0) * 1000.0

        probs = self._dequantize_output(raw_out[0])
        # Renormalize softmax probabilities defensively
        probs = np.maximum(probs, 0.0)
        probs = probs / max(probs.sum(), 1e-8)
        pred_idx = int(np.argmax(probs))
        pred_label = self.labels[pred_idx]
        return pred_label, probs, latency_ms


def evaluate_experiment(
    tflite_path: Path,
    data_dir: Path,
    combined_dir: Path,
    out_dir: Path,
    held_out_speakers: list = None,
    val_speakers: list = None,
):
    held_out_set = set(held_out_speakers or ["Vitthal"])
    val_set = set(val_speakers or ["Ishita"])

    evaluator = TFLiteWakeWordEvaluator(str(tflite_path))

    print("=" * 70)
    print("PHASE 3 - STEP 7: COMPREHENSIVE SPEAKER GENERALIZATION EVALUATION")
    print("=" * 70)
    print(f"Quantized model  : {tflite_path} ({tflite_path.stat().st_size / 1024:.2f} KB)")
    print(f"Data directory   : {data_dir}")
    print(f"Held-out Unseen  : {sorted(held_out_set)}")
    print(f"Validation       : {sorted(val_set)}\n")

    # 1. EVALUATE ALL 5 SPEAKERS ON CLEAN RECORDINGS
    # Collect clean files per speaker from normalized directory
    norm_dir = data_dir / "normalized"
    speaker_results = {}
    speaker_latencies = []

    for spk_dir in sorted(norm_dir.iterdir()):
        if not spk_dir.is_dir():
            continue
        spk = spk_dir.name
        wavs = sorted(list(spk_dir.glob("*.wav")))
        correct = 0
        confidences = []

        for w in wavs:
            pred_label, probs, lat = evaluator.predict_file(str(w))
            speaker_latencies.append(lat)
            is_correct = (pred_label == "vaani")
            correct += int(is_correct)
            confidences.append(float(probs[evaluator.vaani_idx]))

        acc = (correct / len(wavs)) * 100.0 if wavs else 0.0
        status = "UNSEEN TEST" if spk in held_out_set else ("VALIDATION" if spk in val_set else "SEEN TRAIN")
        speaker_results[spk] = {
            "status": status,
            "total": len(wavs),
            "correct": correct,
            "accuracy": acc,
            "mean_confidence": float(np.mean(confidences)),
        }

    # Print Speaker Breakdown
    print(f"{'Speaker':<15}{'Category':<15}{'Correct/Total':<18}{'Accuracy':<12}{'Mean Vaani Conf'}")
    print("-" * 72)
    seen_accs = []
    val_accs = []
    unseen_accs = []

    for spk, res in speaker_results.items():
        cat = res["status"]
        if cat == "SEEN TRAIN":
            seen_accs.append(res["accuracy"])
        elif cat == "VALIDATION":
            val_accs.append(res["accuracy"])
        elif cat == "UNSEEN TEST":
            unseen_accs.append(res["accuracy"])

        print(f"{spk:<15}{cat:<15}{res['correct']}/{res['total']:<16}{res['accuracy']:>6.1f}%      "
              f"{res['mean_confidence']*100:>6.1f}%")

    mean_seen = float(np.mean(seen_accs)) if seen_accs else 0.0
    mean_val = float(np.mean(val_accs)) if val_accs else 0.0
    mean_unseen = float(np.mean(unseen_accs)) if unseen_accs else 0.0
    gen_gap = mean_seen - mean_unseen

    print("-" * 72)
    print(f"SEEN SPEAKERS MEAN ACCURACY       : {mean_seen:.1f}%")
    print(f"VALIDATION SPEAKER ACCURACY       : {mean_val:.1f}%")
    print(f"UNSEEN SPEAKER MEAN ACCURACY      : {mean_unseen:.1f}%")
    print(f"GENERALIZATION GAP (Seen - Unseen): {gen_gap:+.1f} percentage points")
    print("=" * 70)

    # 2. EVALUATE NEGATIVE / FALSE-POSITIVE PERFORMANCE
    # Evaluate testing split of data/combined (100 silence, 100 unknown, 26 unseen Vaani)
    test_split_dir = combined_dir / "testing"
    labels_cm = ["silence", "unknown", "vaani"]
    cm = np.zeros((3, 3), dtype=int)

    test_predictions = []
    test_ground_truth = []
    vaani_scores = []
    is_positive = []

    for true_idx, label in enumerate(labels_cm):
        dir_l = test_split_dir / label
        if not dir_l.exists():
            continue
        for w in sorted(dir_l.glob("*.wav")):
            pred_label, probs, lat = evaluator.predict_file(str(w))
            pred_idx = labels_cm.index(pred_label)
            cm[true_idx, pred_idx] += 1
            test_predictions.append(pred_idx)
            test_ground_truth.append(true_idx)
            vaani_scores.append(float(probs[evaluator.vaani_idx]))
            is_positive.append(true_idx == evaluator.vaani_idx)

    # Metrics computation
    total_positives = int(np.sum(is_positive))
    total_negatives = len(is_positive) - total_positives

    # Vaani row is index 2
    vaani_tp = int(cm[2, 2])
    vaani_fn = int(cm[2, 0] + cm[2, 1])
    vaani_fp = int(cm[0, 2] + cm[1, 2])
    vaani_tn = total_negatives - vaani_fp

    tpr = (vaani_tp / total_positives) * 100.0 if total_positives else 0.0
    fpr = (vaani_fp / total_negatives) * 100.0 if total_negatives else 0.0
    fnr = (vaani_fn / total_positives) * 100.0 if total_positives else 0.0
    precision = (vaani_tp / (vaani_tp + vaani_fp)) * 100.0 if (vaani_tp + vaani_fp) else 0.0
    f1 = (2 * precision * tpr) / (precision + tpr) if (precision + tpr) else 0.0

    print("\nFALSE POSITIVE & DETECTION ACCURACY AUDIT (Held-Out Test Split):")
    print("-" * 70)
    print(f"Total Test Samples   : {len(is_positive)} (26 unseen Vaani + 100 unknown + 100 silence)")
    print(f"True Positive Rate (TPR / Recall) : {tpr:.1f}% ({vaani_tp}/{total_positives})")
    print(f"False Positive Rate (FPR)         : {fpr:.1f}% ({vaani_fp}/{total_negatives})")
    print(f"False Negative Rate (FNR)         : {fnr:.1f}% ({vaani_fn}/{total_positives})")
    print(f"Precision                         : {precision:.1f}%")
    print(f"F1 Score                          : {f1:.1f}%")
    print(f"Mean Inference Latency per clip   : {np.mean(speaker_latencies):.2f} ms")

    print("\nConfusion Matrix (Rows = True Class, Columns = Predicted):")
    print(f"{'':<12}{'Pred silence':<14}{'Pred unknown':<14}{'Pred vaani':<14}")
    for i, row_label in enumerate(labels_cm):
        print(f"True {row_label:<7}: {cm[i, 0]:<14}{cm[i, 1]:<14}{cm[i, 2]:<14}")

    # 3. THRESHOLD SWEEP FOR SENSITIVITY CALIBRATION
    thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    sweep_results = []
    print("\nWAKE-WORD CONFIDENCE THRESHOLD SWEEP:")
    print(f"{'Threshold':<12}{'TPR (Vaani Recall)':<22}{'FPR (False Trigger)':<22}{'F1-Score'}")
    print("-" * 65)
    for th in thresholds:
        tp_th = sum(1 for s, p in zip(vaani_scores, is_positive) if p and s >= th)
        fp_th = sum(1 for s, p in zip(vaani_scores, is_positive) if not p and s >= th)
        tpr_th = (tp_th / total_positives) * 100.0 if total_positives else 0.0
        fpr_th = (fp_th / total_negatives) * 100.0 if total_negatives else 0.0
        prec_th = (tp_th / (tp_th + fp_th)) * 100.0 if (tp_th + fp_th) else 0.0
        f1_th = (2 * prec_th * tpr_th) / (prec_th + tpr_th) if (prec_th + tpr_th) else 0.0
        sweep_results.append({"threshold": th, "tpr": tpr_th, "fpr": fpr_th, "precision": prec_th, "f1": f1_th})
        print(f"{th:<12}{tpr_th:>6.1f}% ({tp_th}/{total_positives}){'':<6}{fpr_th:>6.1f}% ({fp_th}/{total_negatives}){'':<6}{f1_th:>6.1f}%")

    # Final verdict on the scientific question
    print("\n" + "=" * 70)
    print("SCIENTIFIC VERDICT: SPEAKER GENERALIZATION")
    print("=" * 70)
    if mean_unseen >= 80.0 and gen_gap <= 15.0:
        verdict = "YES - STRONG GENERALIZATION"
        explanation = (
            f"The model achieved {mean_unseen:.1f}% accuracy on unseen speaker Vitthal "
            f"with a modest generalization gap of {gen_gap:.1f} points. The model learned "
            f"the acoustic signature of 'Vaani' rather than memorizing training voices."
        )
    elif mean_unseen >= 60.0:
        verdict = "PARTIAL GENERALIZATION"
        explanation = (
            f"The model recognizes unseen speaker Vitthal with {mean_unseen:.1f}% accuracy, "
            f"with a generalization gap of {gen_gap:.1f} points."
        )
    else:
        verdict = "LIMITED GENERALIZATION"
        explanation = (
            f"The model achieved {mean_unseen:.1f}% on unseen speaker Vitthal, showing "
            f"a gap of {gen_gap:.1f} points."
        )

    print(f"Verdict: {verdict}")
    print(f"Details: {explanation}")
    print("=" * 70 + "\n")

    # Save full JSON report
    report_data = {
        "tflite_path": str(tflite_path),
        "model_size_kb": float(tflite_path.stat().st_size / 1024),
        "per_speaker": speaker_results,
        "mean_seen_accuracy": mean_seen,
        "mean_val_accuracy": mean_val,
        "mean_unseen_accuracy": mean_unseen,
        "generalization_gap": gen_gap,
        "true_positive_rate": tpr,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "precision": precision,
        "f1_score": f1,
        "confusion_matrix": cm.tolist(),
        "threshold_sweep": sweep_results,
        "verdict": verdict,
        "explanation": explanation,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    report_file = out_dir / "evaluation_results.json"
    with open(report_file, "w") as fp:
        json.dump(report_data, fp, indent=2)

    print(f"Detailed evaluation metrics saved to {report_file}")
    return report_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tflite_path", type=str, default="./artifacts/vaani_int8.tflite")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--combined_dir", type=str, default="./data/combined")
    parser.add_argument("--out_dir", type=str, default="./artifacts")
    parser.add_argument("--held_out_speakers", nargs="+", default=["Vitthal"])
    parser.add_argument("--val_speakers", nargs="+", default=["Ishita"])
    args = parser.parse_args()

    evaluate_experiment(
        Path(args.tflite_path),
        Path(args.data_dir),
        Path(args.combined_dir),
        Path(args.out_dir),
        args.held_out_speakers,
        args.val_speakers,
    )


if __name__ == "__main__":
    main()
