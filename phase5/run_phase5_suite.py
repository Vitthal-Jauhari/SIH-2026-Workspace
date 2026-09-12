"""
Phase 5 - Comprehensive Pipeline & Suite Runner

Orchestrates the complete Phase 5 workflow:
1. prepare_dataset.py
2. split_speakers.py
3. augment.py
4. prepare_combined_dataset.py
5. train_v2.py
6. quantize_v2.py
7. evaluate_v1_v2.py
8. threshold_sweep.py
9. continuous_eval.py
10. resource_estimator.py

Usage:
    python run_phase5_suite.py --eval_only
    python run_phase5_suite.py --full_pipeline
"""

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"
PYTHON_EXE = sys.executable


def run_cmd(cmd_list, desc):
    print("\n" + "=" * 75)
    print(f"RUNNING: {desc}")
    print("=" * 75)
    print("Command:", " ".join(cmd_list))
    res = subprocess.run(cmd_list)
    if res.returncode != 0:
        print(f"\nERROR: Step '{desc}' failed with exit code {res.returncode}", file=sys.stderr)
        sys.exit(res.returncode)


def main():
    parser = argparse.ArgumentParser(description="Run Phase 5 Pipeline & Evaluation Suite")
    parser.add_argument("--eval_only", action="store_true", help="Run only benchmark and evaluation steps")
    parser.add_argument("--full_pipeline", action="store_true", help="Run end-to-end data prep, training, and eval")
    args = parser.parse_args()

    if args.full_pipeline:
        run_cmd([PYTHON_EXE, str(SCRIPTS_DIR / "prepare_dataset.py"), "--analyze_onsets"], "1. Dataset Preparation & Normalization")
        run_cmd([PYTHON_EXE, str(SCRIPTS_DIR / "split_speakers.py")], "2. Speaker-Level Split & Zero-Leakage Verification")
        run_cmd([PYTHON_EXE, str(SCRIPTS_DIR / "augment.py")], "3. Targeted Train Augmentations")
        run_cmd([PYTHON_EXE, str(SCRIPTS_DIR / "prepare_combined_dataset.py")], "4. Combined 3-Class Dataset Assembly")
        run_cmd([PYTHON_EXE, str(SCRIPTS_DIR / "train_v2.py"), "--epochs", "35"], "5. Train Model V2")
        run_cmd([PYTHON_EXE, str(SCRIPTS_DIR / "quantize_v2.py")], "6. Quantize V2 to Full INT8 TFLite")

    # Evaluation Suite (Always run)
    run_cmd([PYTHON_EXE, str(SCRIPTS_DIR / "evaluate_v1_v2.py")], "7. Side-by-Side V1 vs V2 Benchmark")
    run_cmd([PYTHON_EXE, str(SCRIPTS_DIR / "threshold_sweep.py")], "8. Threshold Sweep (0.30 - 0.90)")
    run_cmd([PYTHON_EXE, str(SCRIPTS_DIR / "continuous_eval.py")], "9. Continuous 10-Minute Streaming Evaluation")
    run_cmd([PYTHON_EXE, str(SCRIPTS_DIR / "resource_estimator.py")], "10. ESP32 Microcontroller Resource Audit")

    print("\n" + "=" * 75)
    print("[ALL STEPS COMPLETED SUCCESSFULLY]")
    print("=" * 75)


if __name__ == "__main__":
    main()
