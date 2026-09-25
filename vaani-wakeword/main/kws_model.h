/**
 * @file kws_model.h
 * @brief TFLite Micro wrapper for the Vaani DSCNN wakeword model
 *
 * Loads the INT8 quantized model, registers the required operators,
 * and provides a simple C-callable interface for running inference.
 *
 * Internally uses C++ (TFLite Micro API), exposed via extern "C".
 */

#ifndef KWS_MODEL_H_
#define KWS_MODEL_H_

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Load the model, register ops, create interpreter, allocate tensors.
 *
 * Must be called once at startup. Logs input/output tensor metadata.
 *
 * @return 0 on success, -1 on failure.
 */
int kws_model_init(void);

/**
 * @brief Run wakeword inference on extracted features.
 *
 * Quantizes float32 features to INT8 using the model's input
 * scale/zero-point, invokes the interpreter, and dequantizes
 * the output probabilities.
 *
 * @param features   Pointer to 40×49 = 1960 float32 log-mel features.
 *                   Layout: [mel_bin][frame], matching (1,40,49,1) tensor.
 * @param p_negative Output: probability of class 0 (negative / not Vaani).
 * @param p_vaani    Output: probability of class 1 (Vaani detected).
 * @return 0 on success, -1 on inference failure.
 */
int kws_model_run(const float *features, float *p_negative, float *p_vaani);

#ifdef __cplusplus
}
#endif

#endif  /* KWS_MODEL_H_ */
