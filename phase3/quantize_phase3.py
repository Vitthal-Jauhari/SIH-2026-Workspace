"""
Phase 3 - Step 6: Post-Training INT8 Quantization

Converts the trained Keras model into a fully INT8-quantized TFLite model.
Features:
- Full integer quantization (int8 input, int8 output, int8 weights/activations)
- Representative dataset calibration from training split
- Automated guardrail: ABORTS IMMEDIATELY if INT8 model size >= 256 KB

Usage:
    python quantize_phase3.py --model_path ./artifacts/best_model.keras \
                              --data_dir ./data/combined \
                              --out_dir ./artifacts
"""

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import tensorflow as tf

PHASE1_DIR = Path(__file__).resolve().parent.parent / "phase1"
if str(PHASE1_DIR) not in sys.path:
    sys.path.insert(0, str(PHASE1_DIR))

from features import featurize_file, get_feature_shape  # noqa: E402
from model import LABELS  # noqa: E402

MAX_BUDGET_KB = 256.0


def representative_dataset_gen(data_dir: Path, n_samples: int = 200):
    paths = []
    for label in LABELS:
        label_dir = data_dir / "training" / label
        if label_dir.exists():
            wavs = list(label_dir.glob("*.wav"))
            paths.extend(wavs[: n_samples // len(LABELS) + 1])

    np.random.seed(42)
    np.random.shuffle(paths)

    def gen():
        for p in paths[:n_samples]:
            feat = featurize_file(str(p))
            feat = np.expand_dims(feat, (0, -1)).astype(np.float32)
            yield [feat]

    return gen


def quantize_model(model_path: Path, data_dir: Path, out_dir: Path):
    model_path = model_path.resolve()
    data_dir = data_dir.resolve()
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("PHASE 3 - STEP 6: INT8 QUANTIZATION & MICROCONTROLLER BUDGET CHECK")
    print("=" * 65)
    print(f"Source model : {model_path}")
    print(f"Data source  : {data_dir}")
    print(f"Target INT8  : {out_dir / 'vaani_int8.tflite'}\n")

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = tf.keras.models.load_model(model_path)

    # In Keras 3 / TF 2.16+, export to temporary SavedModel first
    # to avoid BatchNorm LLVM conversion issues
    with tempfile.TemporaryDirectory() as tmpdir:
        model.export(tmpdir)
        converter = tf.lite.TFLiteConverter.from_saved_model(tmpdir)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset_gen(data_dir, n_samples=200)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

        print("Converting to full INT8 TFLite...")
        tflite_model = converter.convert()

    out_path = out_dir / "vaani_int8.tflite"
    out_path.write_bytes(tflite_model)

    size_bytes = len(tflite_model)
    size_kb = size_bytes / 1024.0

    print("\nTFLITE INT8 MODEL SPECIFICATION:")
    print(f"File path         : {out_path}")
    print(f"Model size (bytes): {size_bytes:,} bytes")
    print(f"Model size (KB)   : {size_kb:.2f} KB")
    print(f"Budget ceiling    : {MAX_BUDGET_KB:.2f} KB")
    print(f"Budget headroom   : {MAX_BUDGET_KB - size_kb:.2f} KB under budget")

    # Verify input / output types using TFLite Interpreter
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    print(f"Input tensor name : {input_details['name']}")
    print(f"Input shape       : {input_details['shape'].tolist()}")
    print(f"Input dtype       : {input_details['dtype']}")
    print(f"Input scale/ZP    : {input_details['quantization']}")
    print(f"Output shape      : {output_details['shape'].tolist()}")
    print(f"Output dtype      : {output_details['dtype']}")
    print(f"Output scale/ZP   : {output_details['quantization']}")

    # STRICT GUARDRAIL ASSERTION
    if size_kb >= MAX_BUDGET_KB:
        msg = f"CRITICAL ERROR: INT8 model size ({size_kb:.2f} KB) exceeds maximum allowed budget ({MAX_BUDGET_KB} KB)!"
        print(msg, file=sys.stderr)
        raise RuntimeError(msg)

    print(f"\n[SUCCESS] Model is fully INT8 quantized and strictly under 256 KB ({size_kb:.2f} KB < 256 KB).\n")
    return out_path, size_kb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./artifacts/best_model.keras")
    parser.add_argument("--data_dir", type=str, default="./data/combined")
    parser.add_argument("--out_dir", type=str, default="./artifacts")
    args = parser.parse_args()

    # Fall back to final_model.keras if best_model does not exist
    p = Path(args.model_path)
    if not p.exists() and (p.parent / "final_model.keras").exists():
        p = p.parent / "final_model.keras"

    quantize_model(p, Path(args.data_dir), Path(args.out_dir))


if __name__ == "__main__":
    main()
