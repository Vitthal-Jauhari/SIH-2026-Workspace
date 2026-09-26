<p align="center">
  <h1 align="center">🎤 Vaani</h1>
  <p align="center">
    <strong>Edge-Native Wake-Word Detection & Dual-Core ESP-IDF Pipeline</strong>
  </p>
  <p align="center">
    A production-grade Keyword Spotting (KWS) system that trains, evaluates, and deploys a custom wake-word detector to ESP32 / ESP32-S3 — fitting the entire neural network in <strong>13.38 KB</strong> of Flash with <strong>14.5 ms</strong> inference latency on a dual-core FreeRTOS architecture.
  </p>
  <p align="center">
    <a href="#-key-results"><img src="https://img.shields.io/badge/Unseen_Speaker_Recall-96.1%25-brightgreen?style=for-the-badge" alt="Recall"></a>
    <a href="#-microcontroller-resource-budget"><img src="https://img.shields.io/badge/Model_Size-13.38_KB-blue?style=for-the-badge" alt="Size"></a>
    <a href="#-microcontroller-resource-budget"><img src="https://img.shields.io/badge/Latency-14.5_ms-orange?style=for-the-badge" alt="Latency"></a>
    <a href="#-microcontroller-resource-budget"><img src="https://img.shields.io/badge/Parameters-4,643-purple?style=for-the-badge" alt="Params"></a>
    <a href="#-dual-core-esp-idf-architecture"><img src="https://img.shields.io/badge/Runtime-ESP--IDF_Dual--Core-red?style=for-the-badge" alt="ESP-IDF"></a>
  </p>
</p>

---

## 📌 Problem Statement

Voice-activated edge devices require a lightweight, always-on wake-word detector running **completely on-device** — without cloud latency, mandatory internet connectivity, or power-hungry coprocessors.

The core challenge is fitting an accurate neural network within the strict memory constraints of an ultra-low-cost ESP32 microcontroller, while generalizing reliably across **unseen speakers**, **noisy acoustic environments**, and **phonetically similar hard decoy words** (*"Paani"*, *"Rani"*, *"Naani"*, *"Kahaani"*).

---

## 🏗️ Dual-Core ESP-IDF Architecture

The production firmware in `vaani-wakeword/` leverages both 240 MHz Xtensa cores of the ESP32 in an asymmetric FreeRTOS pipeline:

```
                        ┌─────────────────────────────────────────────────────────┐
                        │             CORE 0 (PRO_CPU) — Capture & Gate           │
                        │                                                         │
  INMP441 I2S MEMS      │  Blocking DMA Read     1.0s Rolling Buffer              │
  ─────────────────────▶│ ───────────────────▶ [ 16,000 samples ]                │
  16 kHz Mono PCM       │   (~0% CPU idle)              │                         │
                        │                               ▼                         │
                        │                       Energy VAD Gate                   │
                        │                      (Speech START / END)               │
                        └───────────────────────────────┬─────────────────────────┘
                                                        │
                                    Window Ready & VAD Active (200ms Hop)
                                    [Single Snapshot Buffer | s_infer_busy]
                                                        │
                                                        ▼
                        ┌─────────────────────────────────────────────────────────┐
                        │             CORE 1 (APP_CPU) — DSP & Inference          │
                        │                                                         │
                        │   MFCC Feature Extraction (63 frames × 13 coeffs)       │
                        │                       │                                 │
                        │                       ▼                                 │
                        │   DS-CNN (INT8 TFLite Micro)                            │
                        │   [13.38 KB Flash | 4,643 parameters | 14.5ms latency]  │
                        │                       │                                 │
                        │                       ▼                                 │
                        │          [ Silence | Unknown | Vaani ]                  │
                        │                       │                                 │
                        │                       ▼                                 │
                        │   Multi-Window Confirmation (KWS_MIN_CONSECUTIVE = 2)   │
                        │                       │                                 │
                        │                       ▼                                 │
                        │             🎯 WAKE WORD CONFIRMED!                     │
                        │         (GPIO 2 Status LED + ASR Trigger)               │
                        └─────────────────────────────────────────────────────────┘
```

### Key Engineering Optimizations

1. **Zero-Lag Dual-Core Decoupling**: Core 0 stays DMA-bound capturing audio and running lightweight Voice Activity Detection (VAD). Core 1 only activates when a full, speech-active window is available.
2. **DRAM-Safe Snapshot Buffer**: A single snapshot buffer guarded by an atomic busy gate (`s_infer_busy`) replaces memory-heavy double ping-pong buffers, avoiding static `dram0_0_seg` overflow on ESP32-WROOM-32.
3. **Drop-on-Busy Duty Cycle Bounding**: If Core 1 is still processing inference when the next hop arrives, the hop is dropped rather than queued, eliminating latency queues.
4. **2-Window Spike Filter**: Requires two consecutive positive window triggers (`KWS_MIN_CONSECUTIVE = 2`) to confirm an activation, suppressing isolated false positives.

---

## ✨ Key Results & Benchmarks

### Model V1 → V2 Improvement

| Metric | V1 (Baseline) | V2 (Final) | Improvement |
| :--- | :---: | :---: | :---: |
| **Unseen Speaker Recall** (Vitthal, th=0.40) | 82.7% | **96.1%** | **+13.4 pp** |
| **Quiet Speech Recall** (−12 dB) | 52.0% | **89.8%** | **+37.8 pp** |
| **Slow Tempo Recall** (0.85×) | 63.8% | **90.6%** | **+26.8 pp** |
| **Test Set F1 Score** (th=0.40) | 77.2% | **86.5%** | **+9.3 pp** |
| **Test Set Precision** (th=0.40) | 72.4% | **78.7%** | **+6.3 pp** |
| **Continuous Stream False Triggers/hr** | 2,322 | **1,800** | **−22.5%** |

### Per-Speaker Recall @ threshold 0.50

| Speaker | Role | Clips | V1 Recall | V2 Recall |
| :--- | :--- | :---: | :---: | :---: |
| Ananya | Train | 39 | 59.0% | **92.3%** |
| Ark | Train | 49 | 8.2% | **100.0%** |
| Umang | Train | 50 | 50.0% | **100.0%** |
| Mayank | Train | 48 | 20.8% | **100.0%** |
| Ishita | Validation | 42 | 50.0% | **83.3%** |
| **Vitthal** | **🔒 Unseen Test** | **127** | 70.9% | **93.7%** |

> **Zero Leakage:** Vitthal's recordings were **never seen** during training or validation, establishing the true generalization benchmark.

### Noise Robustness (Unseen Speaker, th=0.40)

| Noise Type | 0 dB SNR | 5 dB | 10 dB | 15 dB | 20 dB |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Traffic | 100% | 100% | 100% | 100% | 100% |
| Pink Noise | 100% | 100% | 100% | 100% | 99.2% |
| Babble | 52.0% | 59.8% | 68.5% | 75.6% | 81.9% |

---

## 💾 Microcontroller Resource Budget

| Resource | Budget | Vaani V2 (ESP32) | Utilization |
| :--- | ---: | ---: | :---: |
| **Flash (Model)** | 256.00 KB | **13.38 KB** (13,696 B) | **5.2%** |
| **SRAM (Tensor Arena)** | 512.00 KB | **~43.40 KB** | **8.5%** |
| **Inference Latency** | < 100 ms | **14.5 ms** | ✅ |
| **Model Parameters** | — | **4,643** | — |
| **Quantization** | — | Full INT8 | — |

> The model occupies only **5.2% of Flash**, leaving over **94% headroom** for application logic, Wi-Fi networking, Bluetooth stacks, and OTA updates.

---

## 📂 Repository Structure

```text
SIH-2026-Workspace/
│
├── vaani-wakeword/                # ⚡ PRODUCTION ESP-IDF FIRMWARE
│   ├── CMakeLists.txt             #   Root CMake project configuration
│   ├── sdkconfig.defaults         #   240MHz CPU frequency & FreeRTOS settings
│   ├── dependencies.lock          #   Locked ESP-IDF component dependencies
│   ├── receive_mic.py             #   High-speed serial audio recorder
│   ├── test_model_offline.py      #   Firmware DSP & quantization parity tester
│   └── main/
│       ├── CMakeLists.txt         #   Source component registry
│       ├── idf_component.yml      #   Managed dependency: espressif/esp-tflite-micro
│       ├── main.c                 #   Dual-core FreeRTOS coordinator & task pinning
│       ├── i2s_mic.c / .h         #   INMP441 I2S DMA driver (GPIO 26/25/22)
│       ├── vad.c / .h             #   Energy VAD speech gate & state machine
│       ├── mel_features.c / .h    #   On-device 63×13 MFCC DSP engine
│       ├── mel_tables.h           #   Precomputed FFT & Mel filterbank tables
│       ├── kws_model.cpp / .h     #   TFLite Micro C++ wrapper & inference engine
│       └── model_data.cc / .h     #   Compiled 13.38 KB INT8 model byte array
│
├── phase1/                        # 🧠 Baseline DS-CNN architecture & training
├── phase2/                        # 🧪 PC validation, offline WAV & live mic testing
├── phase3/                        # 🎙️ Speaker diversity dataset (352 clips, 5 speakers)
├── phase4/                        # 📊 Robustness benchmarking (noise injection, SNR sweeps)
├── phase5/                        # 🎯 Hard negative mining ("Paani", "Rani"), Model V2
│   ├── scripts/                   #   Automated training & evaluation pipeline
│   ├── artifacts/                 #   Trained weights, JSON results & sweep data
│   └── reports/                   #   Detailed benchmark reports
│
├── esp32_deployment/              # 📦 Arduino / PlatformIO prototype package (.ino)
├── test-servers/                  # 📡 Host streaming server utilities (Flask audio receiver)
├── export_model_to_c.py           # 🔄 Convert .tflite to C headers for ESP32
├── test_live_mic.py               # 🎙️ PC microphone real-time wake-word tester
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start Guide

### Option 1: Flash Production ESP-IDF Firmware (Recommended)

#### 1. Hardware Wiring (INMP441 MEMS Microphone to ESP32)

| INMP441 Pin | ESP32 GPIO | Description |
| :--- | :--- | :--- |
| **VDD / 3V3** | **3V3** | 3.3V Power |
| **GND** | **GND** | Ground |
| **SD / DIN** | **GPIO 22** | Serial Data Input |
| **WS / LRCLK** | **GPIO 25** | Word Select (Left/Right Clock) |
| **SCK / BCK** | **GPIO 26** | Bit Clock |
| **L/R** | **GND** | Left Channel Select |
| **Indicator LED** | **GPIO 2** | Built-in / External Status LED |

#### 2. Build & Flash with ESP-IDF

```bash
# Navigate to the ESP-IDF project
cd vaani-wakeword

# Set target to standard ESP32 (or esp32s3)
idf.py set-target esp32

# Build the firmware
idf.py build

# Flash to device and open serial monitor (replace COM3 with your port)
idf.py -p COM3 flash monitor
```

Speak **"Vaani"** near the microphone. When detected:
- The GPIO 2 status LED turns on for 2 seconds.
- Serial logs display confidence scores, feature extraction time, and inference latency.

---

### Option 2: Test on Your PC Microphone

Test the trained model directly on your computer before touching hardware:

```bash
# Install dependencies
pip install tensorflow numpy librosa soundfile sounddevice

# Run live tester with Model V2 — speak "Vaani" into your mic
python test_live_mic.py

# Adjust sensitivity threshold and confirmation windows
python test_live_mic.py --threshold 0.45 --consecutive 2
```

---

### Option 3: Flash Arduino / PlatformIO Prototype

If you prefer testing with Arduino IDE:
1. Install the **esp32** board package in Arduino IDE.
2. Install the **TensorFlowLite_ESP32** library.
3. Open `esp32_deployment/esp32_vaani_wakeword.ino`.
4. Upload to your ESP32 board and monitor at **115200 baud**.

---

## 🔬 Research & Training Pipeline (Phases 1–5)

To reproduce or re-train the models from scratch:

```bash
# 1. Run Phase 3: Normalize audio, split speakers, apply augmentations
cd phase3
python run_pipeline.py

# 2. Run Phase 5: Hard negative mining, V2 training, full benchmark suite
cd ../phase5
python run_phase5_suite.py --full_pipeline

# 3. Export the resulting INT8 .tflite model to ESP32 C headers
cd ..
python export_model_to_c.py
```

### Training Pipeline Summary

* **Phase 1 (Foundation):** Designed sub-256KB DS-CNN architecture targeting 3 classes (`silence`, `unknown`, `vaani`).
* **Phase 2 (PC Validation):** Calibrated FAR/FRR curves and confidence thresholds on sliding-window audio.
* **Phase 3 (Speaker Diversity):** 352 recordings across 5 speakers (Ananya, Ark, Umang, Mayank, Ishita) with strict zero-leakage test partitioning for Vitthal.
* **Phase 4 (Robustness):** Continuous stress testing at 0–25 dB SNR with babble, traffic, and pink noise.
* **Phase 5 (Hard Negatives & V2):** Integrated phonetically similar near-miss words (*"Paani"*, *"Rani"*, *"Naani"*, *"Kahaani"*) to eliminate false activations on rhyming speech.

---

## 📊 Dataset Distribution

| Split | Vaani | Unknown | Silence | Total | Speakers |
| :--- | ---: | ---: | ---: | ---: | :--- |
| **Training** | 4,275 | 4,275 | 4,275 | 12,825 | Ananya, Ark, Umang, Mayank |
| **Validation** | 42 | 100 | 100 | 242 | Ishita |
| **Testing** | 127 | 381 | 381 | 889 | **Vitthal (unseen)** |
| **Total** | 4,444 | 4,756 | 4,756 | **13,956** | **5 speakers** |

---

## 🗺️ Next Steps: ASR Server Integration

The firmware is structured to act as the frontline wake-up trigger for a complete offline/edge-cloud voice assistant:

```
[ User speaks "Vaani" ] 
         │
         ▼
[ ESP32 Wake-Word Trigger (Core 1) ]
         │
         ▼
[ Open Wi-Fi / WebSocket Stream to ASR Server (asr_server/) ]
         │
         ▼
[ Stream 16kHz PCM Audio ──▶ Transcribe with Whisper/Vosk ──▶ Execute Intent ]
```

---

## 👥 Authors & Acknowledgments

Developed for **SIH (Smart India Hackathon) 2026**:
- **Vitthal Jauhari** — Dual-core ESP-IDF firmware, I2S DMA, VAD gating, and hardware deployment
- **Umang** — Model research, Phase 1–5 training pipeline, hard negative mining, benchmarks
- **Mayank Singh** — Model architecture, dataset processing, and quantization pipeline
- **Dataset Contributors** — Ananya, Ark, Ishita, Mayank, Umang, Vitthal

---

## 📜 License

This project was developed for SIH 2026. Distributed under the [MIT License](LICENSE).
