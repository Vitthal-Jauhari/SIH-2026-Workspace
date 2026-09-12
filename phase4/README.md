# Vaani — Phase 4: Real-World Robustness & Deployment Evaluation

Evaluates the real-world operational robustness of the trained Phase 3 INT8 TFLite model (`vaani_int8.tflite`) without performing any retraining.

## Primary Objective
Stress-test the existing model under realistic deployment conditions to identify specific acoustic failure modes and formulate a targeted data collection plan for the next training cycle.

---

## One-Click Suite Runner

Execute all Phase 4 evaluations and output the comprehensive summary:
```bash
python run_phase4_suite.py
```

---

## Benchmark Modules

### 1. Baseline Preservation
Records model SHA256, tensor shapes, quantization parameters, and baseline metrics:
```bash
python record_baseline.py
```
*Output: `artifacts/phase3_baseline.json`*

### 2. Noise Robustness Benchmark
Evaluates recall across 6 noise types (fan, typing, room, traffic, babble, pink noise) at SNRs from 25 dB to 0 dB:
```bash
python eval_noise_robustness.py --threshold 0.40
```
*Output: `artifacts/noise_evaluation.json`*

### 3. Speaking Volume & Tempo Benchmark
Evaluates volume attenuation (-12 dB to +6 dB) and tempo variation (0.85x to 1.15x):
```bash
python eval_volume_speed.py --threshold 0.40
```
*Output: `artifacts/volume_speed_evaluation.json`*

### 4. Negative Testing & Threshold Sweep
Sweeps detection thresholds from 0.30 to 0.90 across 206 Vaani recordings and 1,050 negative clips:
```bash
python eval_threshold_sweep.py
```
*Output: `artifacts/threshold_sweep.json`*

### 5. Continuous Streaming & Trigger Policy Analysis
Simulates sliding-window streaming over a 10-minute continuous audio timeline, measuring latency and false triggers:
```bash
python eval_continuous.py --duration_minutes 10.0
```
*Output: `artifacts/continuous_eval.json`*

### 6. ESP32 Microcontroller Resource Audit
Calculates Flash usage, TFLite Micro Tensor Arena, rolling audio buffer, and projected ESP32-S3 inference cycles:
```bash
python esp32_resource_estimator.py
```
*Output: `artifacts/esp32_resources.json`*

---

## Deliverables & Documentation
- **Comprehensive Scientific Report**: [`artifacts/phase4_report.md`](artifacts/phase4_report.md)
- **Targeted Data Collection Plan**: [`artifacts/data_collection_plan.md`](artifacts/data_collection_plan.md)
