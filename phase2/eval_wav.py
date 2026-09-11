"""
Phase 2 - Step 1: WAV files -> Model -> Predictions, with the metrics that
matter for a wake-word system (not just accuracy).

    Accuracy               overall correctness across all 3 classes
    False Acceptance Rate  keyword fired when it shouldn't have
                            (true label is silence/unknown, predicted vaani)
    False Rejection Rate   keyword should have fired but didn't
                            (true label is vaani, predicted something else)
    Detection latency       per-inference wall-clock time (ms)
    Model size               size of the .tflite file on disk
    RAM usage                 peak resident memory during inference

Run:
    python eval_wav.py --tflite_path ../phase1/artifacts/vaani_int8.tflite \
                        --data_dir ../phase1/data/vaani_processed/testing
"""

import argparse
import os
from pathlib import Path

import numpy as np
import psutil
from sklearn.metrics import confusion_matrix

from inference import WakeWordModel

KEYWORDS = {"vaani"}  # classes that should trigger an "action"


def evaluate(tflite_path: str, data_dir: Path):
    model = WakeWordModel(tflite_path)
    proc = psutil.Process(os.getpid())

    y_true, y_pred, latencies = [], [], []
    peak_rss = proc.memory_info().rss

    for label in model.labels:
        label_dir = data_dir / label
        if not label_dir.exists():
            continue
        for wav in label_dir.glob("*.wav"):
            pred_label, probs, latency_ms = model.predict_from_file(str(wav))
            y_true.append(label)
            y_pred.append(pred_label)
            latencies.append(latency_ms)
            peak_rss = max(peak_rss, proc.memory_info().rss)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    accuracy = float(np.mean(y_true == y_pred))

    # False acceptance: true is non-keyword, predicted is a keyword
    non_kw_mask = ~np.isin(y_true, list(KEYWORDS))
    false_accepts = np.sum(non_kw_mask & np.isin(y_pred, list(KEYWORDS)))
    far = false_accepts / max(non_kw_mask.sum(), 1)

    # False rejection: true is a keyword, predicted != true label
    kw_mask = np.isin(y_true, list(KEYWORDS))
    false_rejects = np.sum(kw_mask & (y_pred != y_true))
    frr = false_rejects / max(kw_mask.sum(), 1)

    model_size_kb = Path(tflite_path).stat().st_size / 1024

    print(f"\n{'='*50}")
    print(f"Samples evaluated:      {len(y_true)}")
    print(f"Accuracy:               {accuracy*100:.2f}%")
    print(f"False Acceptance Rate:  {far*100:.2f}%  ({false_accepts}/{non_kw_mask.sum()})")
    print(f"False Rejection Rate:   {frr*100:.2f}%  ({false_rejects}/{kw_mask.sum()})")
    print(f"Latency (mean/p95/max): {np.mean(latencies):.2f} / "
          f"{np.percentile(latencies, 95):.2f} / {np.max(latencies):.2f} ms")
    print(f"Model size:             {model_size_kb:.1f} KB")
    print(f"Peak process RSS:       {peak_rss/1024:.1f} KB "
          f"(process-level, not a substitute for on-device tensor-arena sizing)")
    print(f"{'='*50}")

    print(f"\nConfusion matrix (rows=true, cols=pred), labels={model.labels}:")
    print(confusion_matrix(y_true, y_pred, labels=model.labels))

    return {
        "accuracy": accuracy,
        "far": far,
        "frr": frr,
        "latency_mean_ms": float(np.mean(latencies)),
        "latency_p95_ms": float(np.percentile(latencies, 95)),
        "model_size_kb": model_size_kb,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tflite_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True,
                         help="Directory containing yes/no/unknown/silence subfolders")
    args = parser.parse_args()
    evaluate(args.tflite_path, Path(args.data_dir))
