"""
Phase 3 - End-to-End Orchestrator

Runs the complete Phase 3 pipeline in the exact required sequence:
1. Normalize audio (M4A, MP3, AAC, unusual extensions -> 16kHz mono PCM WAV)
   [Guardrail: asserts exactly 352 files normalized]
2. Split speakers BEFORE augmentation (Train: Ananya, Ark, Umang, Mayank | Val: Ishita | Unseen: Vitthal)
   [Guardrail: asserts zero speaker leakage]
3. Augment TRAINING speakers only (Validation & Unseen strictly untouched)
4. Merge Vaani + Phase 1 Speech Commands negative data (3-class balance)
5. Model Training (Tiny DS-CNN)
6. Post-training INT8 Quantization
   [Guardrail: asserts INT8 model size < 256 KB]
7. Comprehensive Speaker Generalization & Negative Evaluation

Usage:
    python run_pipeline.py [--smoke_test] [--epochs 35]
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd, desc):
    print("\n" + "=" * 75)
    print(f">> {desc.upper()}")
    print(f">> Command: {' '.join(cmd)}")
    print("=" * 75)
    res = subprocess.run(cmd)
    if res.returncode != 0:
        print(f"\n[PIPELINE ABORTED] Step failed: {desc}", file=sys.stderr)
        sys.exit(res.returncode)


def main():
    parser = argparse.ArgumentParser(description="Phase 3 Pipeline Orchestrator")
    parser.add_argument("--smoke_test", action="store_true", help="Run quick 2-epoch smoke test")
    parser.add_argument("--epochs", type=int, default=35, help="Number of full training epochs")
    args = parser.parse_args()

    python_exe = sys.executable
    phase3_dir = Path(__file__).resolve().parent

    norm_script = phase3_dir / "normalize_audio.py"
    split_script = phase3_dir / "speaker_split.py"
    aug_script = phase3_dir / "augment.py"
    merge_script = phase3_dir / "merge_with_phase1.py"
    train_script = phase3_dir / "train_phase3.py"
    quant_script = phase3_dir / "quantize_phase3.py"
    eval_script = phase3_dir / "eval_by_speaker.py"

    # Step 1: Normalize
    run_cmd(
        [python_exe, str(norm_script), "--in_dir", str(phase3_dir / "Audio"),
         "--out_dir", str(phase3_dir / "data" / "normalized"), "--expected_count", "352"],
        "Step 1: Audio Normalization (Assert 352/352 files)",
    )

    # Step 2: Speaker Split
    run_cmd(
        [python_exe, str(split_script), "--in_dir", str(phase3_dir / "data" / "normalized"),
         "--out_dir", str(phase3_dir / "data"), "--train_speakers", "Ananya", "Ark", "Umang", "Mayank",
         "--val_speakers", "Ishita", "--held_out_speakers", "Vitthal"],
        "Step 2: Speaker Split & Zero-Leakage Audit",
    )

    # Step 3: Augment Train Only
    run_cmd(
        [python_exe, str(aug_script), "--in_dir", str(phase3_dir / "data" / "train"),
         "--out_dir", str(phase3_dir / "data" / "train_augmented"),
         "--copies_per_file", "4", "--held_out_speakers", "Vitthal", "--val_speakers", "Ishita"],
        "Step 3: Augment Training Data Only",
    )

    # Step 4: Merge with Negatives
    run_cmd(
        [python_exe, str(merge_script), "--vaani_dir", str(phase3_dir / "data"),
         "--sc_dir", str(phase3_dir / "data" / "sc_raw" / "mini_speech_commands"),
         "--out_dir", str(phase3_dir / "data" / "combined"),
         "--held_out_speakers", "Vitthal", "--val_speakers", "Ishita"],
        "Step 4: Merge Vaani with Phase 1 Negatives (3-Class Balanced)",
    )

    # Step 5: Train
    train_args = [python_exe, str(train_script), "--data_dir", str(phase3_dir / "data" / "combined"),
                  "--out_dir", str(phase3_dir / "artifacts"), "--epochs", str(args.epochs)]
    if args.smoke_test:
        train_args.append("--smoke_test")
    run_cmd(train_args, "Step 5: Train Tiny DS-CNN Model")

    # Step 6: Quantize
    run_cmd(
        [python_exe, str(quant_script), "--model_path", str(phase3_dir / "artifacts" / "best_model.keras"),
         "--data_dir", str(phase3_dir / "data" / "combined"), "--out_dir", str(phase3_dir / "artifacts")],
        "Step 6: Post-Training INT8 Quantization (Assert < 256 KB)",
    )

    # Step 7: Evaluate
    run_cmd(
        [python_exe, str(eval_script), "--tflite_path", str(phase3_dir / "artifacts" / "vaani_int8.tflite"),
         "--data_dir", str(phase3_dir / "data"), "--combined_dir", str(phase3_dir / "data" / "combined"),
         "--out_dir", str(phase3_dir / "artifacts"),
         "--held_out_speakers", "Vitthal", "--val_speakers", "Ishita"],
        "Step 7: Speaker Generalization & False-Positive Evaluation",
    )

    print("\n" + "*" * 75)
    print("PHASE 3 PIPELINE COMPLETED SUCCESSFULLY WITH ALL GUARDRAILS VERIFIED!")
    print("*" * 75)


if __name__ == "__main__":
    main()
