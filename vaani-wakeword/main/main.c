/**
 * @file main.c
 * @brief Vaani Wakeword Detection — Main Pipeline
 *
 * Full wakeword detection loop on ESP32:
 *
 *   I2S Mic  →  VAD Gate  →  Log-Mel Features  →  TFLite DSCNN  →  Detection
 *
 * Audio flow:
 *   1. Read 30 ms frames from INMP441 via I2S
 *   2. Shift into a rolling 1-second audio buffer
 *   3. Run energy-based VAD on each frame
 *   4. Every 200 ms, if VAD is active:
 *      - Extract log-mel spectrogram (40 bins × 49 frames)
 *      - Feed into the INT8 quantized DSCNN model
 *      - If P(Vaani) ≥ threshold → detection!
 *
 * Classes:
 *   0 = Negative (not Vaani)
 *   1 = Vaani (wakeword detected)
 */

#include <stdio.h>
#include <string.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "driver/gpio.h"

#include "i2s_mic.h"
#include "vad.h"
#include "mel_features.h"
#include "kws_model.h"

static const char *TAG = "VAANI";


/* ================================================================
 * Configuration
 * ================================================================ */

#define SAMPLE_RATE         16000
#define FRAME_SAMPLES       480         /* 30 ms at 16 kHz */
#define AUDIO_BUF_SAMPLES   16000       /* 1 second */
#define KWS_HOP_SAMPLES     3200        /* 200 ms at 16 kHz */
#define KWS_THRESHOLD       0.90f

#define LED_INDICATOR_GPIO  2           /* Built-in LED on most ESP32 boards */
#define LED_HOLD_TIME_US    2000000ULL  /* 2 seconds in microseconds */


/* ================================================================
 * Static buffers
 * ================================================================ */

/* Rolling 1-second audio buffer (shifted left on each frame). */
static int16_t s_audio_buf[AUDIO_BUF_SAMPLES];

/* Single frame read buffer. */
static int16_t s_frame_buf[FRAME_SAMPLES];

/* Log-mel feature output: 40 × 49 floats. */
static float s_features[MEL_NUM_BINS * MEL_NUM_FRAMES];


/* ================================================================
 * Main
 * ================================================================ */

void app_main(void)
{
    /* ----------------------------------------------------------
     * Banner
     * ---------------------------------------------------------- */

    ESP_LOGI(TAG, "====================================");
    ESP_LOGI(TAG, "   VAANI WAKEWORD DETECTION v1");
    ESP_LOGI(TAG, "====================================");


    /* ----------------------------------------------------------
     * Initialize subsystems
     * ---------------------------------------------------------- */

    ESP_LOGI(TAG, "Initializing I2S microphone...");
    esp_err_t ret = i2s_mic_init();
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "I2S init failed! Halting.");
        while (1) { vTaskDelay(pdMS_TO_TICKS(1000)); }
    }

    ESP_LOGI(TAG, "Initializing mel feature extraction...");
    mel_features_init();

    ESP_LOGI(TAG, "Initializing KWS model...");
    if (kws_model_init() != 0) {
        ESP_LOGE(TAG, "Model init failed! Halting.");
        while (1) { vTaskDelay(pdMS_TO_TICKS(1000)); }
    }

    ESP_LOGI(TAG, "Initializing VAD...");
    vad_init();

    ESP_LOGI(TAG, "Configuring LED indicator (GPIO %d)...", LED_INDICATOR_GPIO);
    gpio_config_t led_cfg = {
        .pin_bit_mask = (1ULL << LED_INDICATOR_GPIO),
        .mode         = GPIO_MODE_OUTPUT,
        .pull_up_en   = GPIO_PULLUP_DISABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type    = GPIO_INTR_DISABLE,
    };
    gpio_config(&led_cfg);
    gpio_set_level(LED_INDICATOR_GPIO, 0);


    /* ----------------------------------------------------------
     * Clear audio buffer
     * ---------------------------------------------------------- */

    memset(s_audio_buf, 0, sizeof(s_audio_buf));


    /* ----------------------------------------------------------
     * State
     * ---------------------------------------------------------- */

    int total_samples_read = 0;     /* tracks buffer fill */
    int samples_since_kws  = 0;     /* accumulator for 200 ms hop */
    int total_detections   = 0;
    int64_t led_off_time   = 0;     /* non-blocking timer for LED hold */


    /* ----------------------------------------------------------
     * Detection loop
     * ---------------------------------------------------------- */

    ESP_LOGI(TAG, "");
    ESP_LOGI(TAG, "====================================");
    ESP_LOGI(TAG, "   LISTENING...  Say \"Vaani\"!");
    ESP_LOGI(TAG, "====================================");
    ESP_LOGI(TAG, "");

    while (1) {

        /* Check if LED indicator 2-second hold has elapsed */
        if (led_off_time > 0 && esp_timer_get_time() >= led_off_time) {
            gpio_set_level(LED_INDICATOR_GPIO, 0);
            led_off_time = 0;
        }

        /* ======================================================
         * 1. Read 30 ms frame from I2S microphone
         * ====================================================== */

        ret = i2s_mic_read(s_frame_buf, FRAME_SAMPLES);
        if (ret != ESP_OK) {
            ESP_LOGW(TAG, "I2S read error, skipping frame");
            continue;
        }


        /* ======================================================
         * 2. Shift audio buffer left and append new frame
         *
         *    [old ... | frame N-1 | frame N ]
         *        ←── shift ───←
         *    [... | frame N-1 | frame N | NEW FRAME ]
         * ====================================================== */

        memmove(
            s_audio_buf,
            s_audio_buf + FRAME_SAMPLES,
            (AUDIO_BUF_SAMPLES - FRAME_SAMPLES) * sizeof(int16_t)
        );

        memcpy(
            s_audio_buf + (AUDIO_BUF_SAMPLES - FRAME_SAMPLES),
            s_frame_buf,
            FRAME_SAMPLES * sizeof(int16_t)
        );

        total_samples_read += FRAME_SAMPLES;


        /* ======================================================
         * 3. VAD — process this frame
         * ====================================================== */

        float db = vad_process_frame(s_frame_buf, FRAME_SAMPLES);


        /* ======================================================
         * 4. Check if it's time to run KWS (every ~200 ms)
         * ====================================================== */

        samples_since_kws += FRAME_SAMPLES;

        if (samples_since_kws < KWS_HOP_SAMPLES) {
            continue;
        }

        samples_since_kws = 0;


        /* Don't run until we have a full 1-second buffer. */
        if (total_samples_read < AUDIO_BUF_SAMPLES) {
            ESP_LOGI(TAG, "Filling buffer... %d / %d samples",
                     total_samples_read, AUDIO_BUF_SAMPLES);
            continue;
        }


        /* ======================================================
         * 5. VAD gate — skip KWS if silence
         * ====================================================== */

        if (!vad_is_active()) {
            ESP_LOGI(TAG, "Level %7.1f dB | VAD: SILENCE | KWS: SKIP",
                     db);
            continue;
        }


        /* ======================================================
         * 6. Extract log-mel features
         * ====================================================== */

        int64_t t0 = esp_timer_get_time();

        mel_features_extract(s_audio_buf, AUDIO_BUF_SAMPLES, s_features);

        int64_t t_feat = esp_timer_get_time() - t0;


        /* ======================================================
         * 7. Run TFLite Micro inference
         * ====================================================== */

        float p_negative = 0.0f;
        float p_vaani    = 0.0f;

        int64_t t1 = esp_timer_get_time();

        int rc = kws_model_run(s_features, &p_negative, &p_vaani);

        int64_t t_infer = esp_timer_get_time() - t1;

        if (rc != 0) {
            ESP_LOGE(TAG, "Inference failed!");
            continue;
        }


        /* ======================================================
         * 8. Detection decision
         * ====================================================== */

        bool detected = (p_vaani >= KWS_THRESHOLD);

        if (detected) {
            total_detections++;

            /* Light LED indicator for 2 seconds */
            gpio_set_level(LED_INDICATOR_GPIO, 1);
            led_off_time = esp_timer_get_time() + LED_HOLD_TIME_US;

            ESP_LOGW(TAG,
                "***  VAANI DETECTED!  ***  "
                "P(Vaani)=%.4f  P(Neg)=%.4f  "
                "[feat %lld us | infer %lld us]  "
                "[total detections: %d]",
                p_vaani, p_negative,
                t_feat, t_infer,
                total_detections
            );

        } else {

            ESP_LOGI(TAG,
                "Level %6.1f dB | VAD: SPEECH | "
                "P(Vaani)=%.4f  P(Neg)=%.4f  "
                "[feat %lld us | infer %lld us]",
                db,
                p_vaani, p_negative,
                t_feat, t_infer
            );
        }
    }
}
