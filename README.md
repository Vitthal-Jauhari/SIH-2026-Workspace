# VikramEdge — Edge Wake-Word Detection Pipeline

An end-to-end keyword spotting (KWS) pipeline designed for ultra-low-power microcontrollers (e.g. ESP32-S3) with an INT8 model size under 256 KB.

---

## Project Structure

This repository is structured into three self-contained development phases:

```text
vikramedge/
├── phase1/    # Prototype, Model Architecture & INT8 Quantization
├── phase2/    # PC-Side Validation (Offline WAVs & Live Mic Stream)
├── phase3/    # Real Speaker Diversity, Audio Augmentation & Holdout Evaluation
└── .gitignore # Git ignore rules (filters out large raw audio >100MB)
```

---

## Phases Overview

### [Phase 1: Prototype & Model Export](./phase1/)
- **Target Keyword**: Configured for 3-class detection (`silence`, `unknown`, `vaani`).
- **Features**: MFCC (13 coefficients, 512 FFT, 256 hop @ 16 kHz).
- **Architecture**: Depthwise-Separable CNN (DS-CNN).
- **Quantization**: Full-integer INT8 post-training quantization (`~13.7 KB`, <256 KB budget).
- **Quickstart**:
  ```bash
  cd phase1
  pip install -r requirements.txt
  python train.py --data_dir ./data/processed --out_dir ./artifacts
  python quantize.py --model_path ./artifacts/final_model.keras
  ```

### [Phase 2: PC-Side Validation](./phase2/)
- **Offline Batch Evaluation**: Measures accuracy, False Acceptance Rate (FAR), False Rejection Rate (FRR), latency, and RAM usage on WAV test sets.
- **Real-Time Mic Stream**: Rolling 1-second buffer detection with debounce cooldown.
- **Quickstart**:
  ```bash
  cd phase2
  pip install -r requirements.txt
  python eval_wav.py --tflite_path ../phase1/artifacts/model_int8.tflite --data_dir ../phase1/data/processed/testing
  python mic_stream.py --tflite_path ../phase1/artifacts/model_int8.tflite
  ```

### [Phase 3: Real Speaker Diversity](./phase3/)
- **Speaker Recording**: Ingest multiple real voices with variations in distance, pace, and volume.
- **Augmentation**: Offline noise mixing, pitch shifting, speed variation, and gain adjustment.
- **Speaker Holdout Split**: Partitions recordings by speaker identity to guarantee testing on completely unseen voices.
- **Generalization Gap**: Evaluates model performance on seen vs. unseen speakers.
- **Quickstart**:
  ```bash
  cd phase3
  pip install -r requirements.txt
  python record_speakers.py --speaker_id alice --reps 15
  python augment.py --in_dir ./data/speakers --out_dir ./data/speakers_augmented
  python speaker_split.py --in_dir ./data/speakers_augmented --out_dir ./data/speakers_processed --held_out_speakers charlie
  ```
