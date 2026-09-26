# ESP32 / ESP32-S3 Deployment Package for Vaani Wake-Word

This folder contains everything needed to deploy the trained **Vaani Model V2 (INT8)** to an ESP32 or ESP32-S3 microcontroller.

---

## 1. Files in This Directory

| File | Purpose | Size |
| :--- | :--- | :--- |
| **`model_data.h`** | The complete INT8 neural network exported as a C byte array (`g_vaani_model_data[]`) stored directly in microcontroller Flash memory. | **13.38 KB** (13,696 B) |
| **`model_config.h`** | C `#define` constants for sample rate, MFCC parameters, quantization scale, zero-points, and trigger thresholds. | ~1.5 KB |
| **`esp32_vaani_wakeword.ino`** | Ready-to-flash Arduino / PlatformIO sketch with I2S microphone sampling, rolling circular buffer, and TFLM inference. | ~6 KB |

---

## 2. Microcontroller Resource Footprint

* **Flash Utilization**: **13.38 KB** (Budget: < 256.00 KB — **94.8% free Flash headroom**).
* **Runtime SRAM (Tensor Arena)**: **~26 KB** (allocated 45 KB for safety, leaves > 250 KB free for Wi-Fi/Bluetooth/App).
* **Inference Latency**: **~14.5 ms** on ESP32-S3 (well within the 100 ms hop budget).

---

## 3. Hardware Setup (I2S MEMS Microphone)

Recommended microphone: **INMP441**, **ICS-43434**, or **SPH0645**.

| INMP441 Pin | ESP32 Pin (Default) | Function |
| :--- | :--- | :--- |
| **VDD / 3V3** | **3V3** | 3.3V Power |
| **GND** | **GND** | Ground |
| **SD / DOUT** | **GPIO 32** | Serial Data |
| **WS / LRCLK**| **GPIO 15** | Word Select (Left/Right clock) |
| **SCK / BCLK**| **GPIO 14** | Bit Clock |
| **L/R** | **GND** | Left Channel Select |

*(Pins can be modified in `esp32_vaani_wakeword.ino` to match any available GPIOs on your board).*

---

## 4. How to Flash via Arduino IDE

1. **Install Arduino IDE & ESP32 Board Package**:
   * Tools $\rightarrow$ Board $\rightarrow$ Boards Manager $\rightarrow$ Install **esp32 by Espressif Systems**.
2. **Install TensorFlow Lite Micro Library**:
   * Sketch $\rightarrow$ Include Library $\rightarrow$ Manage Libraries...
   * Search and install **TensorFlowLite_ESP32** (or use the official ESP-TFLite-Micro repository).
3. **Open the Sketch**:
   * Open `esp32_vaani_wakeword.ino`.
   * Ensure `model_data.h` and `model_config.h` are located in the same folder as the `.ino` file.
4. **Select Board & Port**:
   * Tools $\rightarrow$ Board $\rightarrow$ Select your board (e.g. *ESP32-S3 Dev Module* or *ESP32 Dev Module*).
   * Tools $\rightarrow$ Port $\rightarrow$ Select COM port.
5. **Upload & Monitor**:
   * Click **Upload**.
   * Open the Serial Monitor at **115200 baud** to see real-time detections!

---

## 5. Re-generating Header Files in the Future

If you retrain the model in the future, run:

```bash
python export_model_to_c.py
```
This will automatically refresh `model_data.h` and `model_config.h` from your latest `.tflite` model.
