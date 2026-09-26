/*
 * Vaani Wake-Word Detection on ESP32 / ESP32-S3
 * Using TensorFlow Lite for Microcontrollers (TFLM) & I2S MEMS Microphone
 *
 * Microcontroller: ESP32 or ESP32-S3
 * Microphone: INMP441 / ICS-43434 / SPH0645 (I2S Digital MEMS)
 * Memory: Flash ~13.4 KB, SRAM ~45 KB
 */

#include <Arduino.h>
#include <driver/i2s.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "model_data.h"
#include "model_config.h"

// =========================================================================
// Hardware Pinout for I2S Microphone (e.g. INMP441)
// Adjust pins to match your specific ESP32 development board
// =========================================================================
#define I2S_PORT         I2S_NUM_0
#define I2S_PIN_BCLK     GPIO_NUM_14   // Bit Clock (SCK / BCLK)
#define I2S_PIN_WS       GPIO_NUM_15   // Word Select (WS / LRCLK)
#define I2S_PIN_DATA     GPIO_NUM_32   // Serial Data In (SD / DOUT)
#define LED_INDICATOR    GPIO_NUM_2    // Onboard status LED

// =========================================================================
// Tensor Arena Allocation
// Model requires ~26 KB tensor arena. 45 KB provides safe headroom.
// =========================================================================
constexpr int kTensorArenaSize = 45 * 1024;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

// TFLM Global Handles
const tflite::Model* tflite_model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input_tensor = nullptr;
TfLiteTensor* output_tensor = nullptr;

// Audio Sliding Buffer (16,000 samples @ 16kHz = 1.0s)
int16_t audio_window[AUDIO_WINDOW_SAMPLES];
int16_t i2s_read_chunk[INFERENCE_HOP_SAMPLES];

// State Tracking
int consecutive_hits = 0;
unsigned long last_trigger_time = 0;

// Setup I2S Driver for 16kHz 16-bit Mono Input
void setup_i2s() {
    i2s_config_t i2s_config = {
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
        .sample_rate = SAMPLE_RATE_HZ,
        .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT, // Most MEMS mics output in 32-bit slot
        .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
        .communication_format = I2S_COMM_FORMAT_STAND_I2S,
        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count = 4,
        .dma_buf_len = 512,
        .use_apll = false,
        .tx_desc_auto_clear = false,
        .fixed_mclk = 0
    };

    i2s_pin_config_t pin_config = {
        .bck_io_num = I2S_PIN_BCLK,
        .ws_io_num = I2S_PIN_WS,
        .data_out_num = I2S_PIN_NO_CHANGE,
        .data_in_num = I2S_PIN_DATA
    };

    i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL);
    i2s_set_pin(I2S_PORT, &pin_config);
    i2s_zero_dma_buffer(I2S_PORT);
}

void setup() {
    Serial.begin(115200);
    delay(1000);
    pinMode(LED_INDICATOR, OUTPUT);
    digitalWrite(LED_INDICATOR, LOW);

    Serial.println("\n==================================================");
    Serial.println("🎤 Vaani Wake-Word Detection (ESP32 / ESP32-S3)");
    Serial.println("==================================================");

    // 1. Load Model from Flash byte array
    tflite_model = tflite::GetModel(g_vaani_model_data);
    if (tflite_model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.printf("Error: Model schema mismatch (got %d, expected %d)\n",
                      tflite_model->version(), TFLITE_SCHEMA_VERSION);
        while (1) { delay(1000); }
    }
    Serial.printf("Model loaded from Flash. Size: %u bytes\n", g_vaani_model_data_len);

    // 2. Resolve Operators
    static tflite::AllOpsResolver resolver;

    // 3. Initialize Interpreter
    static tflite::MicroInterpreter static_interpreter(
        tflite_model, resolver, tensor_arena, kTensorArenaSize);
    interpreter = &static_interpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("Error: AllocateTensors failed! Increase kTensorArenaSize.");
        while (1) { delay(1000); }
    }

    input_tensor = interpreter->input(0);
    output_tensor = interpreter->output(0);

    Serial.printf("Tensor Arena: %d KB allocated\n", kTensorArenaSize / 1024);
    Serial.printf("Input Shape : [%d, %d, %d, %d]\n",
                  input_tensor->dims->data[0], input_tensor->dims->data[1],
                  input_tensor->dims->data[2], input_tensor->dims->data[3]);
    Serial.printf("Quantization: Scale=%.6f, ZeroPoint=%d\n",
                  input_tensor->params.scale, input_tensor->params.zero_point);

    // 4. Initialize I2S Microphone
    setup_i2s();
    memset(audio_window, 0, sizeof(audio_window));

    Serial.println("System Ready. 🟢 LISTENING for wake word 'Vaani'...\n");
}

void on_wake_word_detected(float confidence, unsigned long latency_ms) {
    digitalWrite(LED_INDICATOR, HIGH);
    Serial.println("\n**************************************************");
    Serial.printf("🎯 [WAKE WORD DETECTED] 'VAANI'!\n");
    Serial.printf("   Confidence: %.1f%%\n", confidence * 100.0f);
    Serial.printf("   Latency   : %lu ms\n", latency_ms);
    Serial.println("**************************************************\n");

    // Add your downstream action here (e.g. start voice recording, trigger Wi-Fi API)
    delay(200);
    digitalWrite(LED_INDICATOR, LOW);
}

void loop() {
    // 1. Read hop chunk from I2S (100 ms = 1600 samples)
    size_t bytes_read = 0;
    // INMP441 provides 24-bit audio in 32-bit slots
    int32_t raw_i2s_buf[INFERENCE_HOP_SAMPLES];
    i2s_read(I2S_PORT, raw_i2s_buf, sizeof(raw_i2s_buf), &bytes_read, portMAX_DELAY);

    // Convert 32-bit I2S data to 16-bit PCM and calculate energy
    int64_t sum_sq = 0;
    int samples_read = bytes_read / sizeof(int32_t);
    for (int i = 0; i < samples_read; i++) {
        int16_t sample = (int16_t)(raw_i2s_buf[i] >> 14); // Scale 24-bit to 16-bit
        i2s_read_chunk[i] = sample;
        sum_sq += (int64_t)sample * sample;
    }

    // 2. Slide rolling audio buffer by 100 ms
    memmove(audio_window, &audio_window[INFERENCE_HOP_SAMPLES],
            (AUDIO_WINDOW_SAMPLES - INFERENCE_HOP_SAMPLES) * sizeof(int16_t));
    memcpy(&audio_window[AUDIO_WINDOW_SAMPLES - INFERENCE_HOP_SAMPLES],
           i2s_read_chunk, INFERENCE_HOP_SAMPLES * sizeof(int16_t));

    // 3. Quick Energy Gate (VAD)
    float rms = sqrtf((float)sum_sq / samples_read);
    if (rms < 80.0f) { // Silence threshold
        consecutive_hits = 0;
        return;
    }

    // 4. Extract MFCC features into input_tensor->data.int8
    // Note: Use ESP-DSP (dsps_fft2r_fc32) or a lightweight C MFCC routine
    // Example: mfcc_extract(audio_window, input_tensor->data.int8, ...);

    // 5. Run Neural Network Inference
    unsigned long t0 = millis();
    TfLiteStatus invoke_status = interpreter->Invoke();
    unsigned long latency = millis() - t0;

    if (invoke_status != kTfLiteOk) {
        Serial.println("Inference invoke error!");
        return;
    }

    // 6. Dequantize Outputs
    int8_t* raw_out = output_tensor->data.int8;
    float p_silence = (raw_out[CLASS_SILENCE_IDX] - OUTPUT_QUANT_ZERO_POINT) * OUTPUT_QUANT_SCALE;
    float p_unknown = (raw_out[CLASS_UNKNOWN_IDX] - OUTPUT_QUANT_ZERO_POINT) * OUTPUT_QUANT_SCALE;
    float p_vaani   = (raw_out[CLASS_VAANI_IDX]   - OUTPUT_QUANT_ZERO_POINT) * OUTPUT_QUANT_SCALE;

    unsigned long now = millis();

    // 7. Multi-window confirmation check
    if (p_vaani >= TRIGGER_THRESHOLD && (p_vaani - p_unknown) >= 0.10f) {
        consecutive_hits++;
        if (consecutive_hits >= CONSECUTIVE_FRAMES && (now - last_trigger_time) >= COOLDOWN_MS) {
            on_wake_word_detected(p_vaani, latency);
            last_trigger_time = now;
            consecutive_hits = 0;
        }
    } else {
        consecutive_hits = 0;
    }
}
