/**
 * @file mel_features.c
 * @brief On-device log-mel spectrogram feature extraction
 *
 * Replicates the V1 Python feature pipeline:
 *
 *   1. Hann-windowed 30 ms frames, 20 ms hop, over 1 second of audio
 *   2. 480-sample window zero-padded to 512 for radix-2 FFT
 *   3. Power spectrum → 40-band mel filterbank (20–7600 Hz)
 *   4. Convert to dB relative to per-spectrogram maximum
 *   5. Normalize (dB + 80) / 80, clipped to [0, 1]
 *
 * Output: 40 × 49 float array matching model input (1, 40, 49, 1).
 *
 * NOTE: Using 512-point FFT instead of the original 480 introduces
 *       a minor frequency resolution difference (~31.25 Hz vs 33.33 Hz
 *       per bin).  This is negligible for wakeword detection.
 */

#include "mel_features.h"

#include <math.h>
#include <string.h>
#include "esp_log.h"

static const char *TAG = "MEL_FEAT";


/* ================================================================
 * V1 feature parameters
 * ================================================================ */

#define SAMPLE_RATE     16000
#define FFT_SIZE        512         /* zero-padded from 480 */
#define WINDOW_LEN      480         /* 30 ms at 16 kHz */
#define HOP_LEN         320         /* 20 ms at 16 kHz */
#define N_MELS          40
#define FMIN            20.0f
#define FMAX            7600.0f
#define NUM_FFT_BINS    (FFT_SIZE / 2 + 1)  /* 257 */
#define NUM_FRAMES      49

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif


/* ================================================================
 * Static buffers (avoid stack allocation)
 * ================================================================ */

static float s_hann_window[WINDOW_LEN];

/* FFT working buffer: interleaved [re, im, re, im, …] */
static float s_fft_buf[FFT_SIZE * 2];

/* Power spectrum for one frame */
static float s_power[NUM_FFT_BINS];

/* Full mel spectrogram [mel_bin * NUM_FRAMES + frame] */
static float s_mel_spec[N_MELS * NUM_FRAMES];


/* ================================================================
 * Mel filterbank (sparse representation)
 *
 * Each triangular filter covers only a few FFT bins, so we store
 * start_bin + weights for just those bins.
 * ================================================================ */

typedef struct {
    int start_bin;      /* first FFT bin this filter touches */
    int num_weights;    /* number of consecutive bins */
    int weight_offset;  /* index into s_mel_weights[] */
} mel_filter_t;

static mel_filter_t s_filters[N_MELS];

#define MAX_TOTAL_WEIGHTS 2000
static float s_mel_weights[MAX_TOTAL_WEIGHTS];
static int   s_total_weights = 0;


/* ================================================================
 * Mel scale conversions (HTK formula)
 * ================================================================ */

static float hz_to_mel(float hz)
{
    return 2595.0f * log10f(1.0f + hz / 700.0f);
}

static float mel_to_hz(float mel)
{
    return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f);
}


/* ================================================================
 * Radix-2 Cooley–Tukey FFT (in-place, complex interleaved)
 *
 * data[]: [re0, im0, re1, im1, …]  length = 2 * n
 * n must be a power of 2.
 * ================================================================ */

static void fft_bit_reverse(float *data, int n)
{
    int j = 0;

    for (int i = 0; i < n; i++) {
        if (j > i) {
            float tmp;
            tmp = data[2 * i];
            data[2 * i] = data[2 * j];
            data[2 * j] = tmp;

            tmp = data[2 * i + 1];
            data[2 * i + 1] = data[2 * j + 1];
            data[2 * j + 1] = tmp;
        }

        int m = n >> 1;
        while (m >= 1 && j >= m) {
            j -= m;
            m >>= 1;
        }
        j += m;
    }
}


static void fft_compute(float *data, int n)
{
    fft_bit_reverse(data, n);

    for (int step = 1; step < n; step <<= 1) {

        float theta = -M_PI / (float)step;
        float w_re  = cosf(theta);
        float w_im  = sinf(theta);

        for (int group = 0; group < n; group += (step << 1)) {

            float tw_re = 1.0f;
            float tw_im = 0.0f;

            for (int pair = 0; pair < step; pair++) {

                int i = group + pair;
                int j = i + step;

                float t_re = data[2 * j]     * tw_re
                           - data[2 * j + 1] * tw_im;

                float t_im = data[2 * j]     * tw_im
                           + data[2 * j + 1] * tw_re;

                data[2 * j]     = data[2 * i]     - t_re;
                data[2 * j + 1] = data[2 * i + 1] - t_im;
                data[2 * i]     += t_re;
                data[2 * i + 1] += t_im;

                float new_tw_re = tw_re * w_re - tw_im * w_im;
                tw_im           = tw_re * w_im + tw_im * w_re;
                tw_re           = new_tw_re;
            }
        }
    }
}


/* ================================================================
 * Init — build Hann window and mel filterbank
 * ================================================================ */

void mel_features_init(void)
{
    ESP_LOGI(TAG, "Building Hann window (%d samples)...", WINDOW_LEN);

    for (int i = 0; i < WINDOW_LEN; i++) {
        s_hann_window[i] = 0.5f * (1.0f - cosf(2.0f * M_PI * (float)i
                                                 / (float)WINDOW_LEN));
    }

    /* ----------------------------------------------------------
     * Build mel filterbank
     *
     * 42 equally-spaced points on the mel scale (40 filters +
     * 2 edge points).  Each filter is a triangle spanning three
     * consecutive mel points.
     * ---------------------------------------------------------- */

    ESP_LOGI(TAG, "Building mel filterbank (%d bins, %.0f–%.0f Hz)...",
             N_MELS, FMIN, FMAX);

    float mel_low  = hz_to_mel(FMIN);
    float mel_high = hz_to_mel(FMAX);

    float mel_points[N_MELS + 2];
    float hz_points[N_MELS + 2];
    int   bin_points[N_MELS + 2];

    for (int i = 0; i < N_MELS + 2; i++) {
        mel_points[i] = mel_low
                       + (mel_high - mel_low) * (float)i / (float)(N_MELS + 1);
        hz_points[i]  = mel_to_hz(mel_points[i]);
        bin_points[i] = (int)floorf(
            (float)(FFT_SIZE + 1) * hz_points[i] / (float)SAMPLE_RATE
        );
    }

    /* Construct triangular weights (sparse). */
    s_total_weights = 0;

    for (int m = 0; m < N_MELS; m++) {

        int start  = bin_points[m];
        int center = bin_points[m + 1];
        int end    = bin_points[m + 2];

        s_filters[m].start_bin     = start;
        s_filters[m].num_weights   = end - start + 1;
        s_filters[m].weight_offset = s_total_weights;

        for (int k = start; k <= end; k++) {
            float weight;

            if (k <= center && center > start) {
                weight = (float)(k - start) / (float)(center - start);
            } else if (k > center && end > center) {
                weight = (float)(end - k) / (float)(end - center);
            } else {
                weight = 0.0f;
            }

            if (s_total_weights < MAX_TOTAL_WEIGHTS) {
                s_mel_weights[s_total_weights++] = weight;
            }
        }
    }

    ESP_LOGI(TAG, "Mel filterbank ready (%d total weights).",
             s_total_weights);
}


/* ================================================================
 * Extract features
 * ================================================================ */

void mel_features_extract(const int16_t *audio, size_t num_samples,
                          float *output)
{
    /* ----------------------------------------------------------
     * Process each of the 49 frames
     * ---------------------------------------------------------- */

    for (int frame = 0; frame < NUM_FRAMES; frame++) {

        int offset = frame * HOP_LEN;

        /* Clear FFT buffer (real and imaginary). */
        memset(s_fft_buf, 0, sizeof(s_fft_buf));

        /* Apply Hann window and convert INT16 → float [-1, +1]. */
        for (int i = 0; i < WINDOW_LEN; i++) {

            int idx = offset + i;

            if (idx < (int)num_samples) {
                s_fft_buf[2 * i] = ((float)audio[idx] / 32768.0f)
                                 * s_hann_window[i];
            }
            /* Imaginary part stays 0 (memset above). */
        }
        /* Samples [WINDOW_LEN .. FFT_SIZE-1] are zero-padded. */


        /* FFT */
        fft_compute(s_fft_buf, FFT_SIZE);


        /* Power spectrum: |X[k]|^2 */
        for (int k = 0; k < NUM_FFT_BINS; k++) {
            float re = s_fft_buf[2 * k];
            float im = s_fft_buf[2 * k + 1];
            s_power[k] = re * re + im * im;
        }


        /* Apply mel filterbank */
        for (int m = 0; m < N_MELS; m++) {

            float energy = 0.0f;
            int   w_off  = s_filters[m].weight_offset;

            for (int j = 0; j < s_filters[m].num_weights; j++) {
                int bin = s_filters[m].start_bin + j;
                if (bin < NUM_FFT_BINS) {
                    energy += s_power[bin] * s_mel_weights[w_off + j];
                }
            }

            /* Store in [mel_bin][frame] layout */
            s_mel_spec[m * NUM_FRAMES + frame] = energy;
        }
    }


    /* ----------------------------------------------------------
     * Convert to log-dB (relative to spectrogram maximum)
     *
     * Matches:  librosa.power_to_db(mel, ref=np.max)
     * ---------------------------------------------------------- */

    float max_val = 1e-10f;
    for (int i = 0; i < N_MELS * NUM_FRAMES; i++) {
        if (s_mel_spec[i] > max_val) {
            max_val = s_mel_spec[i];
        }
    }

    for (int i = 0; i < N_MELS * NUM_FRAMES; i++) {

        /* 10 * log10(S / S_max), floored at -80 dB */
        float db = 10.0f * log10f(
            fmaxf(s_mel_spec[i], 1e-10f) / max_val
        );

        if (db < -80.0f) {
            db = -80.0f;
        }

        /* Normalize: [-80, 0] → [0, 1] */
        float norm = (db + 80.0f) / 80.0f;

        if (norm < 0.0f) norm = 0.0f;
        if (norm > 1.0f) norm = 1.0f;

        output[i] = norm;
    }
}
