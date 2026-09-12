# Vaani — Phase 3: Real Wake-Word & Speaker Generalization

Evaluates whether the tiny ESP32 wake-word model recognizes **"Vaani"** from a genuinely unseen speaker voice it never heard during training or validation, while remaining strictly **under 256 KB**.

## Architecture & Constraints
- **Target Microcontroller**: ESP32 / ESP32-S3
- **Model Size Ceiling**: 256.0 KB (Achieved: **13.39 KB INT8**)
- **Phonetic Target**: "Vaani" (Positive class)
- **Negative Classes**: `unknown` (Google Speech Commands non-target words) + `silence` (ambient room noise)
- **Speakers**: Ananya, Ark, Ishita, Umang, Vitthal (206 total recordings)

---

## The Speaker Generalization Setup
- **Train (Seen)**: Ananya, Ark, Umang (138 clips → augmented to 690 clips)
- **Validation**: Ishita (42 clean clips)
- **Unseen Test**: Vitthal (26 clean clips — completely held out)

---

## One-Click Pipeline Runner

Run the entire pipeline end-to-end with automated guardrails:
```bash
python run_pipeline.py --epochs 35
```

---

## Step-by-Step Pipeline

### 1. Audio Normalization
Converts all 206 raw recordings (M4A, MP3, AAC, and unusual extensions like `.10`, `.15`, `.2`) into standard 16 kHz mono 16-bit PCM WAVs:
```bash
python normalize_audio.py --in_dir ./Audio --out_dir ./data/normalized --expected_count 206
```
*(Automated guardrail: aborts immediately if normalized files != 206)*.

### 2. Speaker-Level Split (Strictly Before Augmentation)
Partitions clean normalized audio by speaker to guarantee zero speaker leakage:
```bash
python speaker_split.py --in_dir ./data/normalized --out_dir ./data \
    --train_speakers Ananya Ark Umang \
    --val_speakers Ishita \
    --held_out_speakers Vitthal
```
*(Automated guardrail: aborts immediately if any held-out speaker is found in train or val)*.

### 3. Augment Train Only
Applies realistic time stretching, pitch shifting, volume scaling, and ambient noise **only to training speakers**:
```bash
python augment.py --in_dir ./data/train --out_dir ./data/train_augmented \
    --copies_per_file 4 --held_out_speakers Vitthal --val_speakers Ishita
```

### 4. Merge with Negative Classes (3-Class Balanced)
Combines positive Vaani recordings with non-target speech words and ambient silence:
```bash
python merge_with_phase1.py --vaani_dir ./data \
    --sc_dir ./data/sc_raw/mini_speech_commands \
    --out_dir ./data/combined \
    --held_out_speakers Vitthal --val_speakers Ishita
```

### 5. Retrain Model
Trains the 3-class tiny DS-CNN:
```bash
python train_phase3.py --data_dir ./data/combined --out_dir ./artifacts --epochs 35
```

### 6. INT8 Quantization
Converts the model to full integer INT8 TFLite:
```bash
python quantize_phase3.py --model_path ./artifacts/best_model.keras \
    --data_dir ./data/combined --out_dir ./artifacts
```
*(Automated guardrail: aborts immediately if INT8 model size >= 256 KB)*.

### 7. Comprehensive Evaluation
Measures seen-speaker vs. unseen-speaker accuracy, false positives, and sensitivity thresholds:
```bash
python eval_by_speaker.py --tflite_path ./artifacts/vaani_int8.tflite \
    --data_dir ./data --combined_dir ./data/combined --out_dir ./artifacts \
    --held_out_speakers Vitthal --val_speakers Ishita
```

---

## Key Results
- **Unseen Speaker Accuracy (Vitthal)**: **84.6%** (22/26) at 0.5 threshold, **100.0%** (26/26) at 0.4 threshold
- **Validation Speaker Accuracy (Ishita)**: **83.3%** (35/42)
- **False Positive Rate**: **0.5%** (1 false trigger in 200 non-wake-word test clips)
- **Silence Rejection**: **100.0%** (100/100 correct)
- **INT8 Model Size**: **13.39 KB** (< 256 KB budget, 242.6 KB headroom)
- **Mean Inference Latency**: **0.12 ms**
