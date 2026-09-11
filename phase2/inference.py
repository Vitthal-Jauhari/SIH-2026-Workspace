"""
Phase 2 - shared inference wrapper around the INT8 TFLite model produced in
Phase 1 (quantize.py). Both eval_wav.py and mic_stream.py use this so the
exact same pre/post-processing path is exercised in both offline and live
testing -- that consistency matters, since Phase 2's whole point is
validating the model behaves the same off a WAV file as it will off a live
mic buffer.
"""

from pathlib import Path
import time

import numpy as np
import tensorflow as tf

import sys

# Locate Phase 1 directory containing features.py and model.py
for _candidate in [
    Path(__file__).resolve().parent.parent / "phase1",
    Path(__file__).resolve().parent.parent / "files",
    Path(__file__).resolve().parent.parent / "vikramedge_phase1",
    Path(__file__).resolve().parent,
]:
    if (_candidate / "features.py").exists():
        sys.path.insert(0, str(_candidate))
        break

from features import extract_mfcc, get_feature_shape  # noqa: E402
from model import LABELS  # noqa: E402


class WakeWordModel:
    def __init__(self, tflite_path: str):
        self.interpreter = tf.lite.Interpreter(model_path=tflite_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()[0]

        # int8 quantization params, needed to rescale float MFCCs -> int8
        self.in_scale, self.in_zero_point = self.input_details["quantization"]
        self.out_scale, self.out_zero_point = self.output_details["quantization"]

        self.labels = LABELS
        self.feature_shape = get_feature_shape()

    def _quantize_input(self, feat: np.ndarray) -> np.ndarray:
        if self.in_scale == 0:
            return feat.astype(self.input_details["dtype"])
        q = feat / self.in_scale + self.in_zero_point
        q = np.clip(np.round(q), -128, 127)
        return q.astype(np.int8)

    def _dequantize_output(self, out: np.ndarray) -> np.ndarray:
        if self.out_scale == 0:
            return out.astype(np.float32)
        return (out.astype(np.float32) - self.out_zero_point) * self.out_scale

    def predict_from_audio(self, audio: np.ndarray):
        """audio: 1D float32 array, 1 second @ 16kHz. Returns (label, probs, latency_ms)."""
        feat = extract_mfcc(audio)
        feat = np.expand_dims(feat, (0, -1)).astype(np.float32)
        q_feat = self._quantize_input(feat)

        t0 = time.perf_counter()
        self.interpreter.set_tensor(self.input_details["index"], q_feat)
        self.interpreter.invoke()
        raw_out = self.interpreter.get_tensor(self.output_details["index"])
        latency_ms = (time.perf_counter() - t0) * 1000

        probs = self._dequantize_output(raw_out[0])
        # softmax already applied in the model, but renormalize defensively
        # after dequantization rounding
        probs = probs / max(probs.sum(), 1e-8)
        label = self.labels[int(np.argmax(probs))]
        return label, probs, latency_ms

    def predict_from_file(self, path: str):
        from features import load_and_pad
        audio = load_and_pad(path)
        return self.predict_from_audio(audio)
