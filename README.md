# Vaani — Edge Wake-Word Detection Pipeline

An end-to-end keyword spotting (KWS) pipeline designed for ultra-low-power microcontrollers (e.g. ESP32-S3) with an INT8 model size under 256 KB (~13.7 KB).

---

## Project Structure

This repository is structured into three self-contained development phases:

```text
vaani/
├── phase1/    # Prototype, DS-CNN Model, Training & INT8 Quantization
├── phase2/    # PC-Side Validation (Offline WAV Evaluation & Live Mic Stream)
├── phase3/    # Real Speaker Diversity, Audio Augmentation & Holdout Evaluation
└── .gitignore # Filters out local data downloads & temporary caches
```

---

## End-to-End Walkthrough (When Your Dataset Arrives)

### 1. Phase 1: Setup & Train the Wake-Word Model

```bash
cd phase1
pip install -r requirements.txt

# Step A: Download Speech Commands v0.02 to generate negative classes ("unknown" & "silence")
python data_prep.py --data_dir ./data

# Step B: Drop your Vaani recordings into phase1/data/vaani_raw organized by speaker:
# data/vaani_raw/
# ├── speaker_01/ (*.wav)
# ├── speaker_02/ (*.wav)
# └── speaker_03/ (*.wav)

# Step C: Ingest Vaani recordings, hold out test speakers, and merge with negative classes
python prepare_vaani_data.py \
    --vaani_dir ./data/vaani_raw \
    --speech_commands_dir ./data/processed \
    --out_dir ./data/vaani_processed \
    --held_out_speakers speaker_03

# Step D: Train the 3-class DS-CNN model
python train.py --data_dir ./data/vaani_processed --out_dir ./artifacts --epochs 40

# Step E: Quantize to full-integer INT8 TFLite model (~13.7 KB)
python quantize.py --model_path ./artifacts/final_model.keras --data_dir ./data/vaani_processed --out_dir ./artifacts
```

---

### 2. Phase 2: Test on PC (WAVs & Live Mic)

```bash
cd ../phase2
pip install -r requirements.txt

# Offline batch evaluation: accuracy, FAR, FRR on held-out Vaani test voices
python eval_wav.py \
    --tflite_path ../phase1/artifacts/vaani_int8.tflite \
    --data_dir ../phase1/data/vaani_processed/testing

# Live microphone detection: rolling 1-second buffer testing
python mic_stream.py --tflite_path ../phase1/artifacts/vaani_int8.tflite --threshold 0.7
```

---

### 3. Phase 3: Real Speaker Diversity & Generalization

```bash
cd ../phase3
pip install -r requirements.txt

# Step A: Record real speakers saying target words
python record_speakers.py --speaker_id alice --reps 15

# Step B: Multiply recordings with noise/speed/pitch augmentation
python augment.py \
    --in_dir ./data/speakers \
    --out_dir ./data/speakers_augmented \
    --noise_dir ../phase1/data/raw/_background_noise_

# Step C: Split speakers with unseen holdouts
python speaker_split.py \
    --in_dir ./data/speakers_augmented \
    --out_dir ./data/speakers_processed \
    --held_out_speakers charlie

# Step D: Merge with base negative data
python merge_with_phase1.py \
    --phase1_dir ../phase1/data/processed \
    --phase3_dir ./data/speakers_processed \
    --out_dir ./data/combined

# Step E: Evaluate generalization gap on seen vs. unseen voices
python eval_by_speaker.py \
    --tflite_path ../phase1/artifacts/vaani_int8.tflite \
    --speakers_dir ./data/speakers \
    --held_out_speakers charlie
```
