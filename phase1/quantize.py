"""
Phase 1 - Step 5: Post-training INT8 quantization -> TFLite.

Run:
    python quantize.py --model_path ./artifacts/final_model.keras \
                        --data_dir ./data/processed --out_dir ./artifacts
"""

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

from features import featurize_file, get_feature_shape
from model import LABELS


def representative_dataset_gen(data_dir: Path, n_samples=200):
    """Feeds real (unlabeled) calibration examples to the converter so it can
    pick good int8 activation ranges -- required for full-integer quantization."""
    paths = []
    for label in LABELS:
        label_dir = data_dir / "training" / label
        if label_dir.exists():
            paths.extend(list(label_dir.glob("*.wav"))[:n_samples // len(LABELS) + 1])
    np.random.shuffle(paths)

    def gen():
        for p in paths[:n_samples]:
            feat = featurize_file(str(p))
            feat = np.expand_dims(feat, (0, -1)).astype(np.float32)
            yield [feat]

    return gen


import tempfile

def quantize(model_path: Path, data_dir: Path, out_dir: Path):
    model = tf.keras.models.load_model(model_path)

    # In Keras 3 (TF 2.16+), converting directly via from_keras_model fails on
    # BatchNormalization layers (LLVM error: missing attribute 'value').
    # Exporting to a temporary SavedModel first resolves this cleanly.
    with tempfile.TemporaryDirectory() as tmpdir:
        model.export(tmpdir)
        converter = tf.lite.TFLiteConverter.from_saved_model(tmpdir)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset_gen(data_dir)
        # Full integer quantization -- weights AND activations int8, required
        # for the TFLite Micro int8 kernels on ESP32-S3.
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

        tflite_model = converter.convert()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "vikramedge_phase1_int8.tflite"
    out_path.write_bytes(tflite_model)

    size_kb = len(tflite_model) / 1024
    print(f"Saved quantized model to {out_path}")
    print(f"Model size: {size_kb:.1f} KB  (budget: 256 KB)")
    if size_kb > 256:
        print("WARNING: over budget -- try width=0.5 in model.py/train.py, "
              "fewer MFCC coefficients, or pruning.")
    else:
        print(f"Headroom: {256 - size_kb:.1f} KB under budget")

    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./artifacts/final_model.keras")
    parser.add_argument("--data_dir", type=str, default="./data/processed")
    parser.add_argument("--out_dir", type=str, default="./artifacts")
    args = parser.parse_args()
    quantize(Path(args.model_path), Path(args.data_dir), Path(args.out_dir))
