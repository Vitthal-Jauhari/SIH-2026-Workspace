"""
Phase 4 - Step 2: Streaming Audio Inference Engine

Implements rolling sliding-window inference mirroring on-device MCU operation:
- Rolling buffer of 1.0s (16,000 samples @ 16 kHz mono)
- Configurable hop size (default: 0.1s = 1,600 samples)
- Reuses exact Phase 3 MFCC feature pipeline (features.py)
- Reuses INT8 quantization scaling
- Stateful trigger policies:
    A) Single-frame threshold
    B) N consecutive frames above threshold
    C) Debounce / cooldown timer to prevent duplicate triggers
"""

import sys
import time
from collections import deque
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import tensorflow as tf

PHASE1_DIR = Path(__file__).resolve().parent.parent / "phase1"
if str(PHASE1_DIR) not in sys.path:
    sys.path.insert(0, str(PHASE1_DIR))

from features import extract_mfcc, get_feature_shape  # noqa: E402
from model import LABELS  # noqa: E402

SAMPLE_RATE = 16000
WINDOW_SECONDS = 1.0
WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_SECONDS)


class StreamingWakeWordDetector:
    def __init__(
        self,
        tflite_path: str,
        threshold: float = 0.40,
        hop_seconds: float = 0.10,
        cooldown_seconds: float = 0.80,
        consecutive_frames_required: int = 1,
    ):
        self.tflite_path = Path(tflite_path).resolve()
        if not self.tflite_path.exists():
            raise FileNotFoundError(f"Model not found: {self.tflite_path}")

        self.threshold = threshold
        self.hop_seconds = hop_seconds
        self.hop_samples = int(SAMPLE_RATE * hop_seconds)
        self.cooldown_seconds = cooldown_seconds
        self.consecutive_frames_required = consecutive_frames_required

        # Load TFLite Model
        self.interpreter = tf.lite.Interpreter(model_path=str(self.tflite_path))
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()[0]

        self.in_scale, self.in_zero_point = self.input_details["quantization"]
        self.out_scale, self.out_zero_point = self.output_details["quantization"]
        self.labels = LABELS
        self.vaani_idx = self.labels.index("vaani")

        # Streaming state buffers
        self.audio_buffer = deque(maxlen=WINDOW_SAMPLES)
        self.audio_buffer.extend(np.zeros(WINDOW_SAMPLES, dtype=np.float32))
        self.unprocessed_samples = 0

        # Detection state
        self.current_time = 0.0
        self.cooldown_until = 0.0
        self.consecutive_above_threshold = 0
        self.detection_events: List[Dict[str, Any]] = []

    def reset(self):
        self.audio_buffer.clear()
        self.audio_buffer.extend(np.zeros(WINDOW_SAMPLES, dtype=np.float32))
        self.unprocessed_samples = 0
        self.current_time = 0.0
        self.cooldown_until = 0.0
        self.consecutive_above_threshold = 0
        self.detection_events.clear()

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

    def infer_window(self, audio_window: np.ndarray) -> Tuple[str, np.ndarray, float]:
        """Runs inference on a 1.0s audio window. Returns (pred_label, probs, latency_ms)."""
        feat = extract_mfcc(audio_window)
        feat = np.expand_dims(feat, (0, -1)).astype(np.float32)
        q_feat = self._quantize_input(feat)

        t0 = time.perf_counter()
        self.interpreter.set_tensor(self.input_details["index"], q_feat)
        self.interpreter.invoke()
        raw_out = self.interpreter.get_tensor(self.output_details["index"])
        latency_ms = (time.perf_counter() - t0) * 1000.0

        probs = self._dequantize_output(raw_out[0])
        probs = np.maximum(probs, 0.0)
        probs = probs / max(probs.sum(), 1e-8)
        pred_label = self.labels[int(np.argmax(probs))]
        return pred_label, probs, latency_ms

    def process_chunk(self, chunk: np.ndarray) -> List[Dict[str, Any]]:
        """
        Ingests a new chunk of audio (any size), advances rolling buffer,
        runs inference every hop_samples, and returns any triggered events.
        """
        triggers = []
        for sample in chunk:
            self.audio_buffer.append(sample)
            self.unprocessed_samples += 1
            self.current_time += 1.0 / SAMPLE_RATE

            if self.unprocessed_samples >= self.hop_samples:
                self.unprocessed_samples = 0
                window_audio = np.array(self.audio_buffer, dtype=np.float32)
                pred_label, probs, latency_ms = self.infer_window(window_audio)
                vaani_conf = float(probs[self.vaani_idx])

                # Check policy
                is_above_threshold = (vaani_conf >= self.threshold)
                if is_above_threshold:
                    self.consecutive_above_threshold += 1
                else:
                    self.consecutive_above_threshold = 0

                # Trigger Condition:
                # 1. Consecutive frames met
                # 2. Cooldown expired (debounce)
                if (
                    self.consecutive_above_threshold >= self.consecutive_frames_required
                    and self.current_time >= self.cooldown_until
                ):
                    event = {
                        "timestamp": round(self.current_time, 3),
                        "vaani_confidence": round(vaani_conf, 4),
                        "predicted_label": pred_label,
                        "probs": [round(float(p), 4) for p in probs],
                        "latency_ms": round(latency_ms, 2),
                    }
                    triggers.append(event)
                    self.detection_events.append(event)
                    self.cooldown_until = self.current_time + self.cooldown_seconds
                    self.consecutive_above_threshold = 0

        return triggers

    def process_stream(self, full_audio: np.ndarray) -> List[Dict[str, Any]]:
        """Processes an entire audio array sequentially through the sliding window pipeline."""
        self.reset()
        triggers = []
        n_samples = len(full_audio)
        if n_samples < WINDOW_SAMPLES:
            return triggers

        for start_idx in range(0, n_samples - WINDOW_SAMPLES + 1, self.hop_samples):
            window_audio = full_audio[start_idx : start_idx + WINDOW_SAMPLES]
            self.current_time = (start_idx + WINDOW_SAMPLES) / SAMPLE_RATE

            pred_label, probs, latency_ms = self.infer_window(window_audio)
            vaani_conf = float(probs[self.vaani_idx])

            # Check policy
            is_above_threshold = (vaani_conf >= self.threshold)
            if is_above_threshold:
                self.consecutive_above_threshold += 1
            else:
                self.consecutive_above_threshold = 0

            # Trigger condition: consecutive frames met AND cooldown expired
            if (
                self.consecutive_above_threshold >= self.consecutive_frames_required
                and self.current_time >= self.cooldown_until
            ):
                event = {
                    "timestamp": round(self.current_time, 3),
                    "vaani_confidence": round(vaani_conf, 4),
                    "predicted_label": pred_label,
                    "probs": [round(float(p), 4) for p in probs],
                    "latency_ms": round(latency_ms, 2),
                }
                triggers.append(event)
                self.detection_events.append(event)
                self.cooldown_until = self.current_time + self.cooldown_seconds
                self.consecutive_above_threshold = 0

        return triggers
