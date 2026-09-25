/**
 * @file mel_features.h
 * @brief On-device log-mel spectrogram feature extraction
 *
 * Replicates the Python V1 feature pipeline on ESP32:
 *   16 kHz mono → 480-pt FFT (zero-padded to 512) → 40 mel bins
 *   → log dB (ref=max) → normalize to [0,1]
 *
 * Output shape: 40 × 49 (mel_bins × time_frames)
 * Matches the model input shape (1, 40, 49, 1).
 *
 * V1 parameters:
 *   Sample rate : 16000 Hz
 *   FFT size    : 480 (zero-padded to 512 for radix-2)
 *   Window      : 30 ms (480 samples), Hann
 *   Hop         : 20 ms (320 samples)
 *   Mel bins    : 40
 *   Freq range  : 20 – 7600 Hz
 */

#ifndef MEL_FEATURES_H_
#define MEL_FEATURES_H_

#include <stdint.h>
#include <stddef.h>

#define MEL_NUM_BINS    40
#define MEL_NUM_FRAMES  49

/**
 * @brief Pre-compute Hann window and mel filterbank weights.
 *
 * Must be called once at startup before mel_features_extract().
 */
void mel_features_init(void);

/**
 * @brief Extract log-mel features from a 1-second audio buffer.
 *
 * @param audio       Pointer to 16-bit PCM samples (16 kHz, mono).
 * @param num_samples Number of samples (should be 16000 for 1 second).
 * @param output      Output buffer, must hold MEL_NUM_BINS * MEL_NUM_FRAMES
 *                    floats (1960 elements). Layout: [mel_bin][frame].
 */
void mel_features_extract(const int16_t *audio, size_t num_samples,
                          float *output);

#endif  /* MEL_FEATURES_H_ */
