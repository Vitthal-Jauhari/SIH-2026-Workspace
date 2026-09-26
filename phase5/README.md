# Phase 5: Targeted Data Collection + Model V2

Phase 5 addresses the empirical failure modes identified during Phase 4 stress-testing of the Phase 3 V1 INT8 wake-word model (`vaani_int8.tflite`).

---

## 1. System Audit & Baseline Preservation

Prior to code execution, an exhaustive audit of Phase 1, Phase 3, and Phase 4 established the following ground truth parameters:

| Parameter | Specification | Source |
| :--- | :--- | :--- |
| **Target Audio Format** | 16,000 Hz, Single Channel (Mono), 16-bit signed PCM WAV | `phase3/normalize_audio.py` |
| **Clip Window Duration** | 1.0 second (16,000 samples) | `phase1/features.py` |
| **FFT Size (`n_fft`)** | 512 samples (32 ms window) | `phase1/features.py` |
| **Hop Length (`hop_length`)** | 256 samples (16 ms hop) | `phase1/features.py` |
| **Mel Filterbanks (`n_mels`)** | 40 | `phase1/features.py` |
| **MFCC Coefficients (`n_mfcc`)**| 13 | `phase1/features.py` |
| **Feature Map Shape** | `(63, 13, 1)` | `phase1/features.py` |
| **Class Labels** | `["silence", "unknown", "vaani"]` (strictly 3 classes) | `phase1/model.py` |
| **Model Architecture** | Tiny Depthwise-Separable CNN (DS-CNN), 4,643 params | `phase1/model.py` |
| **Baseline Model (V1)** | `phase3/artifacts/vaani_int8.tflite` | Phase 3 / Phase 4 |
| **Baseline Model SHA256** | `2c05085e251a800f8176fdb33b3ed4d812782dd10305788ddcffe0132d59d88e` | `phase4/artifacts/phase3_baseline.json` |
| **Baseline Model Size** | 13,712 bytes (13.39 KB) | Flash budget: < 256.00 KB |
| **Quantization Method** | Full Integer Quantization (INT8 input, INT8 output, INT8 ops) | `phase3/quantize_phase3.py` |

> [!IMPORTANT]
> **Baseline V1 Preservation**: The Phase 3 INT8 model (`phase3/artifacts/vaani_int8.tflite`) is treated as an immutable baseline and is never modified or overwritten.

---

## 2. Phase 4 Weaknesses Targeted in Phase 5

1. **Quiet Speech Vulnerability**:
   - Volume attenuation of -12 dB dropped recall to 61.5%.
   - *Targeted Fix*: Include volume-attenuated training samples (-12 dB, -6 dB) during train augmentation.
2. **Low-Frequency Noise (Traffic Rumble)**:
   - 0 dB SNR traffic noise dropped recall to 50.0% (5 dB SNR to 73.1%).
   - *Targeted Fix*: Augment training with low-pass filtered noise mirroring vehicle engine frequencies across 0–25 dB SNR.
3. **Slow / Prolonged Speech**:
   - Tempo stretch of 0.85x–0.92x dropped recall to 88.5%.
   - *Targeted Fix*: Incorporate 0.85x–1.15x speed variations into training data.
4. **Timing Windowing & Leading Silence**:
   - Ark's clips performed poorly (26.5%) due to 2–3 seconds of leading silence before speaking, causing naive 1.0s window cuts to miss the wake word.
   - *Targeted Fix*: Preserve natural timing in validation/testing untouched. Generate timing-shifted positive variations during training only (little, moderate, and substantial leading/trailing silence). An energy-based onset detector is provided as an analysis diagnostic tool only.
5. **Hard Negatives**:
   - Resistance against phonetically similar Indian words (*"Pani"*, *"Rani"*, *"Mani"*, *"Vani"*) mapped to the `unknown` class.

---

## 3. Directory Layout

```
phase5/
├── README.md
├── data/
│   ├── positives/          # Subdirectories per new speaker (e.g. speaker_06/)
│   ├── hard_negatives/     # Subdirectories: pani/, rani/, mani/, vani/, other/
│   ├── background/         # Subdirectories: traffic/, conversation/, household/, other/
│   └── continuous/         # Continuous audio for sliding-window stream testing
├── scripts/
│   ├── prepare_dataset.py  # Audio normalization & dataset validation
│   ├── split_speakers.py   # Strict speaker split with zero-leakage check
│   ├── augment.py          # Train-only targeted augmentations
│   ├── train_v2.py         # Tiny DS-CNN training with feature caching
│   ├── quantize_v2.py      # Post-training INT8 quantization (<256 KB check)
│   ├── evaluate_v1_v2.py   # Side-by-side apples-to-apples benchmark
│   ├── threshold_sweep.py  # Threshold sweep (0.30 - 0.90)
│   ├── continuous_eval.py  # Sliding-window stream simulation
│   └── resource_estimator.py # Flash, SRAM tensor arena, RAM estimation
├── artifacts/
│   ├── speaker_split.json
│   ├── dataset_stats.json
│   ├── eval_manifest.json
│   ├── vaani_v2_best.keras
│   ├── vaani_v2_int8.tflite
│   ├── v1_vs_v2_results.json
│   └── resource_results.json
└── reports/
    └── phase5_report.md
```

---

## 4. Pipeline Execution Sequence

```bash
# 1. Prepare and inspect data structure
python scripts/prepare_dataset.py

# 2. Split speakers and verify zero leakage
python scripts/split_speakers.py

# 3. Apply train-only targeted augmentations
python scripts/augment.py

# 4. Smoke test (2 epochs)
python scripts/train_v2.py --smoke_test

# 5. Full V2 training
python scripts/train_v2.py --epochs 35

# 6. Quantize V2 to INT8
python scripts/quantize_v2.py

# 7. Apples-to-apples V1 vs. V2 benchmark
python scripts/evaluate_v1_v2.py

# 8. Threshold sweep & continuous streaming evaluation
python scripts/threshold_sweep.py
python scripts/continuous_eval.py

# 9. ESP32 microcontroller resource audit
python scripts/resource_estimator.py
```
