# Vaani — Phase 3: Speaker Diversity Experiment

Tests whether the model generalizes beyond the Speech Commands crowd to your
own real speakers, with some speakers held out entirely so "testing"
means a genuinely unseen voice.

## Setup
```bash
pip install -r requirements.txt
```

## Pipeline

### 1. Record real speakers
Run once per person (rerun with a different `--speaker_id` for each):
```bash
python record_speakers.py --speaker_id alice --reps 15
```
Deliberately vary distance from the mic, pace, and volume across sessions —
that's the variation Phase 3 is meant to stress-test. Aim for ~8 speakers.

### 2. Augment
Multiplies each recording with noise/speed/pitch/volume variations:
```bash
python augment.py --in_dir ./data/speakers --out_dir ./data/speakers_augmented \
    --noise_dir ../phase1/data/raw/_background_noise_ \
    --copies_per_file 4
```

### 3. Speaker-level split (hold some speakers out completely)
```bash
python speaker_split.py --in_dir ./data/speakers_augmented \
    --out_dir ./data/speakers_processed \
    --held_out_speakers charlie dana
```
`--held_out_speakers` should be real, unmodified voices you never train or
validate on — that's what makes the eventual test-set accuracy meaningful.

### 4. Merge with Phase 1 data
```bash
python merge_with_phase1.py \
    --phase1_dir ../phase1/data/processed \
    --phase3_dir ./data/speakers_processed \
    --out_dir ./data/combined
```

### 5. Retrain (reuses Phase 1's train.py / quantize.py)
```bash
python ../phase1/train.py --data_dir ./data/combined --out_dir ./artifacts
python ../phase1/quantize.py --model_path ./artifacts/final_model.keras \
    --data_dir ./data/combined --out_dir ./artifacts
```

### 6. The metric that actually matters: seen vs. unseen accuracy
```bash
python eval_by_speaker.py \
    --tflite_path ../phase1/artifacts/vaani_int8.tflite \
    --speakers_dir ./data/speakers \
    --held_out_speakers charlie dana
```
This is the important output of Phase 3 — not aggregate accuracy (Phase 2's
`eval_wav.py` already gives you that), but the **gap between speakers the
model trained on and speakers it never heard**. A small gap means the model
is learning the wake word in general, not memorizing your specific voices.
