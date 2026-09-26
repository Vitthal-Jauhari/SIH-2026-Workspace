# Vaani — Edge Wake-Word Detection Pipeline

An end-to-end Keyword Spotting (KWS) pipeline designed for ultra-low-power microcontrollers (e.g. ESP32 / ESP32-S3) with an INT8 model size under 256 KB (~13.4 KB) and ultra-low inference latency (~14.5 ms on ESP32-S3, ~0.12 ms on PC).

---

## Repository Structure

```text
vaani/
├── esp32_deployment/     # Microcontroller deployment package (.ino sketch, C headers, wiring guide)
├── export_model_to_c.py   # Packager script converting .tflite to C headers (model_data.h, model_config.h)
├── test_live_mic.py       # Real-time PC microphone wake-word & near-miss testing suite
├── phase1/                # Prototype, DS-CNN architecture, initial training & INT8 quantization
├── phase2/                # PC-side validation (offline WAV evaluation & live mic stream)
├── phase3/                # Real speaker diversity (352 recordings across 5 speakers), augmentation & zero-leakage split
├── phase4/                # Robustness benchmarking (continuous sliding window, noise injection, threshold sweep, ESP32 profiler)
├── phase5/                # Hard negative mining, Model V2 (dual-target optimization, 100% unseen test recall, 100% 'Paani' rejection)
└── .gitignore             # Ignores large raw audio datasets, virtualenvs, IDE caches, and temporary files
```

---

## Quick Start: Testing & Deployment

### 1. Test Live on PC Microphone

Experience real-time wake-word detection and near-miss rejection ("Paani", "Rani", conversational speech):

```bash
# Run Model V2 (default)
python test_live_mic.py

# Compare against Model V1 (Phase 3 baseline)
python test_live_mic.py --model v1

# List available input devices
python test_live_mic.py --list-devices
```

### 2. Deploy to ESP32 / ESP32-S3

The `esp32_deployment/` folder is ready to flash via Arduino IDE or PlatformIO:

1. Connect an I2S MEMS microphone (e.g., **INMP441**, **ICS-43434**, or **SPH0645**):
   * `SCK / BCLK` $\rightarrow$ GPIO 14
   * `WS / LRCLK` $\rightarrow$ GPIO 15
   * `SD / DOUT`  $\rightarrow$ GPIO 32
   * `VDD` $\rightarrow$ 3.3V | `GND` $\rightarrow$ GND
2. Open `esp32_deployment/esp32_vaani_wakeword.ino` in Arduino IDE.
3. Install **esp32** board support and **TensorFlowLite_ESP32** library.
4. Upload to your board and open Serial Monitor at **115200 baud**.

To regenerate microcontroller C headers after any model retrain:
```bash
python export_model_to_c.py
```

---

## Pipeline Walkthrough (Phases 1 – 5)

### Phase 1: Prototype & Model Architecture
* DS-CNN (Depthwise Separable Convolutional Neural Network) architecture optimized for microcontrollers.
* Full-integer INT8 quantization reducing model footprint from float32 to ~13.4 KB.

### Phase 2: PC Validation
* Offline multi-class confusion matrix, FAR, FRR, and latency benchmarks.
* Streaming audio circular buffer evaluation with variable confidence thresholds.

### Phase 3: Speaker Diversity & Zero-Leakage Partition
* 352 real-world recordings across diverse team speakers (Ananya, Ark, Ishita, Mayank, Vitthal).
* Guaranteed zero speaker leakage between train, validation, and unseen holdout splits.
* Automated normalization to 16kHz mono 16-bit PCM WAV.

### Phase 4: Robustness & Microcontroller Profiling
* Continuous sliding window simulation over hours of mixed audio.
* Stress testing with background noise (babble, traffic, white noise) at varying SNR levels (0 to 15 dB).
* Microcontroller resource estimation: Flash memory, SRAM Tensor Arena headroom, and CPU cycles.

### Phase 5: Hard Negative Mining & Model V2
* Hard negative mining on phonetically similar near-misses ("Paani", "Rani", "Naani", "Kahaani").
* Dual-target optimization balancing high recall for varied speech tempos with near-miss rejection.
* Model V2 achieved **100.0% recall on unseen test speaker** (Vitthal) and **100.0% false trigger rejection** on near-miss words.

---

## Hardware Resource Specifications

| Metric | Target Budget | Vaani Model V2 (INT8) | Status |
| :--- | :--- | :--- | :--- |
| **Model Flash Footprint** | $< 256\text{ KB}$ | **13.38 KB** (13,696 bytes) | PASS (94.8% headroom) |
| **SRAM (Tensor Arena)** | $< 100\text{ KB}$ | **~26 KB** (~45 KB reserved) | PASS (> 250 KB free SRAM) |
| **Inference Latency** | $< 100\text{ ms}$ | **~14.5 ms** (ESP32-S3) / **0.12 ms** (PC) | PASS |
| **Audio Format** | 16 kHz Mono | 16 kHz Mono PCM (1.0s window, 100ms hop) | PASS |
