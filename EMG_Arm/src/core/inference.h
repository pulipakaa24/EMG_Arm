/**
 * @file inference.h
 * @brief On-device inference engine for EMG gesture recognition.
 */

#ifndef INFERENCE_H
#define INFERENCE_H

#include <stdbool.h>
#include <stdint.h>

// --- Configuration ---
#define INFERENCE_WINDOW_SIZE 150 // Window size in samples (must match Python)
#define NUM_CHANNELS 4            // Number of EMG channels

/**
 * @brief Initialize the inference engine.
 */
void inference_init(void);

/**
 * @brief Add a sample to the inference buffer.
 *
 * @param channels Array of 4 channel values (raw ADC)
 * @return true if a full window is ready for processing
 */
bool inference_add_sample(uint16_t *channels);

/**
 * @brief Run inference on the current window.
 *
 * @param confidence Output pointer for confidence score (0.0 - 1.0)
 * @return Detected class index (-1 if error)
 */
int inference_predict(float *confidence);

/**
 * @brief Get the name of a class index.
 */
const char *inference_get_class_name(int class_idx);

/**
 * @brief Map class index to gesture_t enum.
 */
int inference_get_gesture_enum(int class_idx);

#endif /* INFERENCE_H */
