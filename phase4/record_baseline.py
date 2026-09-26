"""
Phase 4 - Step 1: Baseline Preservation

Fingerprints and records the exact Phase 3 INT8 TFLite model:
- Model filename & path
- SHA256 cryptographic hash
- Size in bytes and KB
- Input / output tensor specifications and quantization parameters
- Phase 3 baseline metrics

Saves results to phase4/artifacts/phase3_baseline.json.
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

PHASE3_MODEL = Path(__file__).resolve().parent.parent / "phase3" / "artifacts" / "vaani_int8.tflite"
PHASE3_RESULTS = Path(__file__).resolve().parent.parent / "phase3" / "artifacts" / "evaluation_results.json"
OUT_DIR = Path(__file__).resolve().parent / "artifacts"


def record_baseline():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not PHASE3_MODEL.exists():
        raise FileNotFoundError(f"Phase 3 model not found at {PHASE3_MODEL}")

    content = PHASE3_MODEL.read_bytes()
    sha256_hash = hashlib.sha256(content).hexdigest()
    size_bytes = len(content)
    size_kb = size_bytes / 1024.0

    interpreter = tf.lite.Interpreter(model_content=content)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    phase3_metrics = {}
    if PHASE3_RESULTS.exists():
        with open(PHASE3_RESULTS, "r") as f:
            phase3_metrics = json.load(f)

    baseline_record = {
        "model_path": str(PHASE3_MODEL.resolve()),
        "model_filename": PHASE3_MODEL.name,
        "sha256": sha256_hash,
        "size_bytes": size_bytes,
        "size_kb": round(size_kb, 2),
        "input_tensor": {
            "name": input_details["name"],
            "shape": input_details["shape"].tolist(),
            "dtype": str(input_details["dtype"]),
            "quantization": {
                "scale": float(input_details["quantization"][0]),
                "zero_point": int(input_details["quantization"][1]),
            },
        },
        "output_tensor": {
            "name": output_details["name"],
            "shape": output_details["shape"].tolist(),
            "dtype": str(output_details["dtype"]),
            "quantization": {
                "scale": float(output_details["quantization"][0]),
                "zero_point": int(output_details["quantization"][1]),
            },
        },
        "labels": ["silence", "unknown", "vaani"],
        "phase3_metrics": {
            "seen_mean_accuracy": phase3_metrics.get("mean_seen_accuracy"),
            "val_accuracy": phase3_metrics.get("mean_val_accuracy"),
            "unseen_vitthal_accuracy": phase3_metrics.get("mean_unseen_accuracy"),
            "generalization_gap": phase3_metrics.get("generalization_gap"),
            "false_positive_rate": phase3_metrics.get("false_positive_rate"),
            "true_positive_rate": phase3_metrics.get("true_positive_rate"),
            "model_size_kb": phase3_metrics.get("model_size_kb"),
        },
    }

    out_file = OUT_DIR / "phase3_baseline.json"
    with open(out_file, "w") as f:
        json.dump(baseline_record, f, indent=2)

    print("=" * 65)
    print("PHASE 4 - STEP 1: BASELINE PRESERVATION RECORD")
    print("=" * 65)
    print(f"Model Filename : {PHASE3_MODEL.name}")
    print(f"SHA256 Hash    : {sha256_hash}")
    print(f"Size           : {size_bytes:,} bytes ({size_kb:.2f} KB)")
    print(f"Input Tensor   : {input_details['shape'].tolist()} {input_details['dtype']}")
    print(f"Output Tensor  : {output_details['shape'].tolist()} {output_details['dtype']}")
    print(f"Baseline Saved : {out_file}\n")
    return baseline_record


if __name__ == "__main__":
    record_baseline()
