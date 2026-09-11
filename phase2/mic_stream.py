"""
Phase 2 - Step 2: 🎤 PC microphone -> Preprocessing -> Model -> Prediction, live.

This runs a sliding 1-second window over a continuous mic stream, which is
the closest PC analogue to what the ESP32 will do in Phase 5 (no fixed
"press to record" -- the model has to make a decision on rolling audio).

Run:
    python mic_stream.py --tflite_path ../phase1/artifacts/vaani_int8.tflite

Press Ctrl+C to stop. Prints a line whenever "yes" or "no" is detected
above --threshold, plus a running detections/minute count you can use as a
rough live FAR proxy (say nothing / say other words for a while and count
false triggers).
"""

import argparse
import queue
import time
from collections import deque

import numpy as np
import sounddevice as sd

from inference import WakeWordModel

SAMPLE_RATE = 16000
WINDOW_SECONDS = 1.0
HOP_SECONDS = 0.2  # how often we run inference -- smaller = lower latency, more CPU


def main(args):
    model = WakeWordModel(args.tflite_path)

    window_len = int(SAMPLE_RATE * WINDOW_SECONDS)
    hop_len = int(SAMPLE_RATE * HOP_SECONDS)

    audio_buffer = deque(maxlen=window_len)
    audio_buffer.extend(np.zeros(window_len, dtype=np.float32))
    audio_q = queue.Queue()

    def callback(indata, frames, time_info, status):
        if status:
            print(status)
        audio_q.put(indata[:, 0].copy())

    device_info = sd.query_devices(args.device, 'input') if args.device is not None else sd.query_devices(kind='input')
    print(f"Using input device: {device_info['name']}")
    print(f"Listening... (window={WINDOW_SECONDS}s, hop={HOP_SECONDS}s, "
          f"threshold={args.threshold})")
    print("Say 'yes' or 'no'. Ctrl+C to stop.\n")

    detections = 0
    start_time = time.time()
    samples_since_infer = 0
    cooldown_until = 0.0  # debounce: one spoken word spans several windows

    try:
        with sd.InputStream(device=args.device, samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                             blocksize=hop_len, callback=callback):
            while True:
                chunk = audio_q.get()
                audio_buffer.extend(chunk)
                samples_since_infer += len(chunk)

                if samples_since_infer >= hop_len:
                    samples_since_infer = 0
                    audio = np.array(audio_buffer, dtype=np.float32)
                    label, probs, latency_ms = model.predict_from_audio(audio)
                    confidence = float(np.max(probs))

                    now = time.time()
                    if (label in ("yes", "no") and confidence >= args.threshold
                            and now >= cooldown_until):
                        detections += 1
                        cooldown_until = now + args.cooldown
                        elapsed_min = max((now - start_time) / 60, 1e-6)
                        print(f"[DETECTED] '{label}'  conf={confidence:.2f}  "
                              f"latency={latency_ms:.1f}ms  "
                              f"({detections} total, {detections/elapsed_min:.1f}/min)")
                    elif args.verbose:
                        print(f"  ... {label} ({confidence:.2f})  {latency_ms:.1f}ms")
    except KeyboardInterrupt:
        elapsed_min = (time.time() - start_time) / 60
        print(f"\nStopped. {detections} detections over {elapsed_min:.1f} min "
              f"({detections/max(elapsed_min, 1e-6):.1f}/min).")
        print("If you were silent/talking about other things the whole time, "
              "this rate is your live false-accept rate.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tflite_path", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=0.7,
                         help="Min softmax confidence to count as a detection")
    parser.add_argument("--verbose", action="store_true",
                         help="Print every window's prediction, not just detections")
    parser.add_argument("--cooldown", type=float, default=0.8,
                         help="Seconds to suppress repeat detections after one fires "
                              "(a single spoken word spans several sliding windows)")
    parser.add_argument("--device", type=int, default=None,
                         help="Audio input device index (run with --list_devices to view options)")
    parser.add_argument("--list_devices", action="store_true",
                         help="List available audio input devices and exit")
    args = parser.parse_args()

    if args.list_devices:
        print(sd.query_devices())
    else:
        if not args.tflite_path:
            parser.error("--tflite_path is required unless using --list_devices")
        main(args)
