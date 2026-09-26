<p align="center">
  <h1 align="center">🎤 Vaani</h1>
  <p align="center">
    <strong>Edge-Native Wake-Word Detection for Microcontrollers</strong>
  </p>
  <p align="center">
    A production-grade Keyword Spotting (KWS) pipeline that trains, evaluates, and deploys a custom wake-word detector to ESP32 / ESP32-S3 — fitting the entire neural network in <strong>13.38 KB</strong> of Flash with <strong>14.5 ms</strong> inference latency.
  </p>
  <p align="center">
    <a href="#-key-results"><img src="https://img.shields.io/badge/Unseen_Speaker_Recall-96.1%25-brightgreen?style=for-the-badge" alt="Recall"></a>
    <a href="#-microcontroller-resource-budget"><img src="https://img.shields.io/badge/Model_Size-13.38_KB-blue?style=for-the-badge" alt="Size"></a>
    <a href="#-microcontroller-resource-budget"><img src="https://img.shields.io/badge/Latency-14.5_ms-orange?style=for-the-badge" alt="Latency"></a>
    <a href="#-microcontroller-resource-budget"><img src="https://img.shields.io/badge/Parameters-4,643-purple?style=for-the-badge" alt="Params"></a>
  </p>
</p>

---

## 📌 Problem Statement

Voice-activated edge devices need a lightweight, always-on wake-word detector that runs **entirely on-device** — no cloud, no Wi-Fi, no latency. The challenge: build a neural network that fits within the extreme memory constraints of a $3 microcontroller while generalizing to **unseen speakers**, **noisy environments**, and **phonetically similar decoy words** ("Paani", "Rani", "Naani").

---

## 🏗️ Architecture

```
                  ┌──────────────┐
  16kHz Mono PCM  │  I2S MEMS    │   1.0s sliding window
  ───────────────▶│  Microphone  │───────────────────────▶ MFCC Features (63 × 13)
                  └──────────────┘        100ms hop
                                              │
                                              ▼
                                   ┌─────────────────────┐
                                   │   DS-CNN (INT8)      │
                                   │   13.38 KB Flash     │
                                   │   4,643 parameters   │
                                   │   14.5 ms inference  │
                                   └─────────────────────┘
                                              │
                              ┌────────────────┼────────────────┐
                              ▼                ▼                ▼
                         [ silence ]      [ unknown ]      [ vaani ✓ ]
                                                               │
                                                               ▼
                                                    Multi-Window Confirm
                                                    + Cooldown Debounce
                                                               │
                                                               ▼
                                                      🎯 WAKE WORD!
```

**Model:** Depthwise Separable Convolutional Neural Network (DS-CNN) — the same family used by Google's "Hey Google" on-device detector — optimized down to just **4,643 trainable parameters** with full INT8 post-training quantization.

---

## ✨ Key Results

### Model V1 → V2 Improvement

| Metric | V1 (Baseline) | V2 (Final) | Improvement |
| :--- | :---: | :---: | :---: |
| **Unseen Speaker Recall** (Vitthal, th=0.40) | 82.7% | **96.1%** | +13.4 pp |
| **Quiet Speech Recall** (−12 dB) | 52.0% | **89.8%** | +37.8 pp |
| **Slow Tempo Recall** (0.85×) | 63.8% | **90.6%** | +26.8 pp |
| **Test Set F1 Score** (th=0.40) | 77.2% | **86.5%** | +9.3 pp |
| **Test Set Precision** (th=0.40) | 72.4% | **78.7%** | +6.3 pp |
| **Continuous Stream False Triggers/hr** | 2,322 | **1,800** | −22.5% |

### Per-Speaker Recall @ threshold 0.50

| Speaker | Role | Clips | V1 Recall | V2 Recall |
| :--- | :--- | :---: | :---: | :---: |
| Ananya | Train | 39 | 59.0% | **92.3%** |
| Ark | Train | 49 | 8.2% | **100.0%** |
| Umang | Train | 50 | 50.0% | **100.0%** |
| Mayank | Train | 48 | 20.8% | **100.0%** |
| Ishita | Validation | 42 | 50.0% | **83.3%** |
| **Vitthal** | **🔒 Unseen Test** | **127** | 70.9% | **93.7%** |

> Vitthal's recordings were **never** seen during training — this is the true generalization benchmark.

### Noise Robustness (Unseen Speaker, th=0.40)

| Noise Type | 0 dB SNR | 5 dB | 10 dB | 15 dB | 20 dB |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Traffic | 100% | 100% | 100% | 100% | 100% |
| Pink Noise | 100% | 100% | 100% | 100% | 99.2% |
| Babble | 52.0% | 59.8% | 68.5% | 75.6% | 81.9% |

---

## 💾 Microcontroller Resource Budget

| Resource | Budget | Vaani V2 | Utilization |
| :--- | ---: | ---: | :---: |
| **Flash (Model)** | 256.00 KB | **13.38 KB** | 5.2% |
| **SRAM (Runtime)** | 512.00 KB | **43.37 KB** | 8.5% |
| **Inference Latency** | < 100 ms | **14.5 ms** | ✅ |
| **Parameters** | — | **4,643** | — |
| **Quantization** | — | Full INT8 | — |

> The model occupies just **5.2% of Flash** — leaving **94.8% free** for your application firmware, Wi-Fi stack, Bluetooth, OTA updates, and more.

---

## 📂 Repository Structure

```text
vaani/
├── 📁 esp32_deployment/           # Ready-to-flash Arduino/PlatformIO package
│   ├── esp32_vaani_wakeword.ino   #   Complete I2S + TFLM inference sketch
│   ├── model_data.h               #   INT8 model as C byte array (13.38 KB)
│   ├── model_config.h             #   Preprocessing & quantization constants
│   └── README.md                  #   Wiring diagram & flash instructions
│
├── 📁 phase1/                     # Foundation: DS-CNN architecture & training
├── 📁 phase2/                     # PC validation: offline WAV & live mic testing
├── 📁 phase3/                     # Speaker diversity: 352 recordings, 5 speakers
│   └── 📁 Audio/                  #   Raw recordings (Ananya, Ark, Ishita, Mayank, Vitthal)
├── 📁 phase4/                     # Robustness: noise injection, threshold sweep, ESP32 profiling
├── 📁 phase5/                     # Hard negatives, V2 training, comprehensive benchmarks
│   ├── 📁 scripts/                #   10-step automated pipeline
│   ├── 📁 artifacts/              #   Trained models, JSON results, sweep data
│   └── 📁 reports/                #   Detailed technical report
│
├── export_model_to_c.py           # Convert .tflite → C headers for ESP32
├── test_live_mic.py               # Real-time PC microphone wake-word tester
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites

```bash
pip install tensorflow numpy librosa soundfile sounddevice
```

### 1. Test on Your PC Microphone

```bash
# Run with Model V2 (default) — speak "Vaani" into your mic
python test_live_mic.py

# Compare against V1 baseline
python test_live_mic.py --model v1

# Adjust detection sensitivity
python test_live_mic.py --threshold 0.45 --consecutive 2

# List available microphones
python test_live_mic.py --list-devices
```

### 2. Deploy to ESP32 / ESP32-S3

#### Hardware Wiring (I2S MEMS Microphone)

| INMP441 Pin | ESP32 GPIO | Function |
| :--- | :--- | :--- |
| **VDD** | **3V3** | Power |
| **GND** | **GND** | Ground |
| **SD / DOUT** | **GPIO 32** | Serial Data |
| **WS / LRCLK** | **GPIO 15** | Word Select |
| **SCK / BCLK** | **GPIO 14** | Bit Clock |
| **L/R** | **GND** | Left Channel |

#### Flash via Arduino IDE

1. Install **esp32** board package (by Espressif Systems)
2. Install **TensorFlowLite_ESP32** library
3. Open `esp32_deployment/esp32_vaani_wakeword.ino`
4. Select your board & COM port → **Upload**
5. Open Serial Monitor at **115200 baud** → Speak "Vaani"!

### 3. Re-train the Model

```bash
# Phase 3: Normalize audio, split speakers, augment, train
cd phase3
python run_pipeline.py

# Phase 5: Hard negative mining, V2 training, full benchmark suite
cd ../phase5
python run_phase5_suite.py --full_pipeline

# Export updated model to ESP32 C headers
cd ..
python export_model_to_c.py
```

---

## 🔬 Pipeline Deep Dive

### Phase 1 — Foundation
- DS-CNN architecture design targeting sub-256KB INT8 models
- 3-class classification: `silence` | `unknown` | `vaani`
- MFCC feature extraction: 63 time frames × 13 coefficients
- Google Speech Commands v0.02 as negative class source

### Phase 2 — PC Validation
- Offline evaluation with confusion matrices, FAR, FRR
- Live microphone streaming with rolling 1-second buffer
- Confidence threshold calibration

### Phase 3 — Real Speaker Diversity
- **352 recordings** from 5 diverse speakers across different microphones, accents, and speaking styles
- Automated normalization pipeline (M4A, MP3, AAC → 16kHz mono PCM WAV)
- **Zero-leakage speaker split**: Train (Ananya, Ark, Umang, Mayank) → Validation (Ishita) → Unseen Test (Vitthal)
- Data augmentation: noise injection (babble, traffic, pink), speed warping (0.85×–1.15×), volume scaling (−12dB to +6dB), onset alignment

### Phase 4 — Robustness Benchmarking
- Continuous sliding-window simulation over hours of mixed audio
- Background noise stress testing at 0–25 dB SNR
- Volume and speaking-tempo sensitivity analysis
- ESP32-S3 resource profiling (Flash, SRAM, CPU cycles)
- Threshold sweep across 0.30–0.95 for optimal precision–recall operating point

### Phase 5 — Hard Negative Mining & Model V2
- Phonetically similar near-miss words ("Paani", "Rani", "Naani", "Kahaani") as hard negatives
- Speaker-aware hard negative partitioning (zero leakage preserved)
- Dual-target optimization: maximize recall + minimize false triggers
- **12,825 training samples** (balanced 3-class: 4,275 each)
- Comprehensive V1 vs V2 benchmark across 6 test dimensions

---

## 📊 Dataset Summary

| Split | Vaani | Unknown | Silence | Total | Speakers |
| :--- | ---: | ---: | ---: | ---: | :--- |
| **Training** | 4,275 | 4,275 | 4,275 | 12,825 | Ananya, Ark, Umang, Mayank |
| **Validation** | 42 | 100 | 100 | 242 | Ishita |
| **Testing** | 127 | 381 | 381 | 889 | Vitthal (unseen) |
| **Total** | 4,444 | 4,756 | 4,756 | **13,956** | **5 speakers** |

> ✅ **Zero speaker leakage verified** — no audio from any test speaker appears in training or validation.

---

## 🛠️ Technical Specifications

| Parameter | Value |
| :--- | :--- |
| **Audio Input** | 16 kHz, mono, 16-bit PCM |
| **Inference Window** | 1.0 second (16,000 samples) |
| **Sliding Hop** | 100 ms (1,600 samples) |
| **Feature Extraction** | 13 MFCCs × 63 time frames |
| **FFT** | 512-point (32 ms frame), 256-point hop (16 ms) |
| **Mel Filter Banks** | 40 |
| **Model Architecture** | Depthwise Separable CNN (DS-CNN) |
| **Quantization** | Full INT8 (weights + activations) |
| **Trigger Logic** | Threshold (0.60) + 2-frame consecutive confirmation + 800 ms cooldown |
| **Supported MCUs** | ESP32, ESP32-S3 (Xtensa LX6/LX7) |
| **Supported Microphones** | INMP441, ICS-43434, SPH0645 (I2S digital MEMS) |

---

## 📜 License

This project was developed for SIH (Smart India Hackathon) 2026.
