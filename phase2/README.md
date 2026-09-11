# VikramEdge — Phase 2: Test on PC

Validates the Phase 1 `.tflite` model off real audio before any microcontroller work:
first on static WAV files, then on a live PC mic stream.

## Setup
```bash
pip install -r requirements.txt
```
Expects Phase 1's model at `../phase1/artifacts/vikramedge_phase1_int8.tflite`
(pass a different path with `--tflite_path` if yours lives elsewhere).

## 1. WAV files → Model → Prediction
Batch-evaluates every WAV in a labeled directory (e.g. Phase 1's `testing`
split, or your own held-out recordings) and reports the metrics that
actually matter for a wake-word system:

```bash
python eval_wav.py \
    --tflite_path ../phase1/artifacts/vikramedge_phase1_int8.tflite \
    --data_dir ../phase1/data/vaani_processed/testing
```

Reports: accuracy, **False Acceptance Rate** (silence/unknown misfired as
the wake word), **False Rejection Rate** (wake word missed), per-inference latency
(mean/p95/max), `.tflite` file size, and peak process RSS during inference.

> Note on RAM: peak process RSS is a rough PC-side proxy, not what the
> ESP32 will actually use — TFLite Micro's tensor arena on-device is sized
> very differently from a desktop Python process. Real RAM validation
> happens in Phase 5; this number is just useful for spotting regressions
> as you iterate on the model here.

## 2. 🎤 Live PC microphone → Preprocessing → Model → Prediction
```bash
python mic_stream.py --tflite_path ../phase1/artifacts/vikramedge_phase1_int8.tflite
```
Runs a sliding 1-second window (default 0.2s hop) over your mic input —
the same rolling-buffer pattern the ESP32 will use, rather than
a push-to-talk recording. Say your wake word ("Vaani") to see detections print live,
with a running detections/minute rate.

Useful things to try:
- Stay silent or talk about something else for a couple minutes and watch
  the detection count — that rate is your **live false-accept rate**.
- Say your wake word repeatedly and check the **miss rate** and reported latency.
- `--threshold` (default 0.7) trades FAR against FRR — raise it to cut false
  accepts at the cost of more misses, lower it the other way.
- `--cooldown` (default 0.8s) debounces so one spoken word doesn't count as
  several detections across overlapping windows.

## What's next (Phase 3)
Once accuracy/FAR/FRR/latency look reasonable here, Phase 3 introduces real
speaker diversity — multiple genuine speakers, held-out speakers for
testing, and augmentation (noise/speed/pitch/volume) — to see how much
these PC numbers hold up outside the training distribution.
