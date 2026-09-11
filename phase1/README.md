# VikramEdge — Phase 1: Prototype (no ESP32)

Pipeline: Google Speech Commands (negative classes) + Vaani recordings (wake-word) → MFCC features → tiny DS-CNN → INT8 quant → TFLite.  
Classes: `silence`, `unknown`, `vaani` (3-class model).

## Setup
```bash
pip install -r requirements.txt
```

## Run the pipeline

### 1. Prepare negative classes (Speech Commands)
Downloads Speech Commands v0.02 and builds negative classes (`unknown` and `silence`):
```bash
python data_prep.py --data_dir ./data
```
- Extracts background noise for the `silence` class.
- Samples across words for the `unknown` class.

### 2. Ingest & split Vaani dataset
Once your Vaani recordings are ready, organize them by speaker folder:
```text
data/vaani_raw/
├── speaker_01/
│   ├── sample_01.wav
│   └── sample_02.wav
├── speaker_02/
│   └── ...
```
Then run `prepare_vaani_data.py` to split speakers (with holdout for unseen-speaker testing) and pull in `unknown` and `silence` from `./data/processed`:
```bash
python prepare_vaani_data.py \
    --vaani_dir ./data/vaani_raw \
    --speech_commands_dir ./data/processed \
    --out_dir ./data/vaani_processed \
    --held_out_speakers speaker_03 speaker_04
```
*(If `--vaani_dir` does not exist yet, the script exits cleanly with instructions).*

### 3. Sanity-check feature shape & model
```bash
# Verify feature extraction shape ((time_frames, n_mfcc) e.g. (63, 13))
python features.py

# Inspect 3-class DS-CNN architecture + parameter count
python model.py
```

### 4. Train
Train the 3-class DS-CNN on the combined Vaani dataset:
```bash
python train.py --data_dir ./data/vaani_processed --out_dir ./artifacts --epochs 40
```

### 5. Quantize to INT8 TFLite
Full-integer post-training quantization to keep within the 256KB microcontroller budget:
```bash
python quantize.py --model_path ./artifacts/final_model.keras --data_dir ./data/vaani_processed --out_dir ./artifacts
```

### 6. Evaluate WAV-level performance
Evaluate accuracy, False Acceptance Rate (FAR), False Rejection Rate (FRR), and latency on held-out test data:
```bash
python eval_wav.py --tflite_path ./artifacts/vikramedge_phase1_int8.tflite --data_dir ./data/vaani_processed/testing
```

## Notes / design decisions
- **3-Class Setup**: Labels are `["silence", "unknown", "vaani"]`. Target wake-word detection is focused exclusively on `"vaani"`.
- **Speaker Holdout**: `prepare_vaani_data.py` splits recordings by speaker ID so unseen speakers are completely held out in `testing/`, guaranteeing that test metrics measure real voice generalization without speaker leakage.
- **`unknown`**: Sampled broadly across non-target Speech Commands words so the model learns a robust "not Vaani" acoustic boundary.
- **`silence`**: 1-second random crops of background noise recordings to reject non-speech ambient room noise on edge devices.
- **MFCC parameters** (`features.py`: 13 coefficients, 512-pt FFT, 256 hop, 16 kHz): Sized for lightweight execution on microcontrollers (e.g. ESP32-S3 via ESP-DSP).
- **Model** (`model.py`): Small DS-CNN (depthwise-separable convolutions) optimized for KWS on microcontrollers. Lands well within the 256KB budget (approx. 13–14 KB INT8).
- **Quantization** (`quantize.py`): Full INT8 (weights and activations) with representative dataset calibration for TFLite Micro kernels.
