/**
 * @file inference.c
 * @brief Implementation of EMG inference engine.
 */

#include "inference.h"
#include "config/config.h"
#include "model_weights.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

// --- Constants ---
#define SMOOTHING_FACTOR 0.7f // EMA factor for probability (matches Python)
#define VOTE_WINDOW 5         // Majority vote window size
#define DEBOUNCE_COUNT 3      // Confirmations needed to change output

// --- State ---
static uint16_t window_buffer[INFERENCE_WINDOW_SIZE][NUM_CHANNELS];
static int buffer_head = 0;
static int samples_collected = 0;

// Smoothing State
static float smoothed_probs[MODEL_NUM_CLASSES];
static int vote_history[VOTE_WINDOW];
static int vote_head = 0;
static int current_output = -1;
static int pending_output = -1;
static int pending_count = 0;

void inference_init(void) {
  memset(window_buffer, 0, sizeof(window_buffer));
  buffer_head = 0;
  samples_collected = 0;

  // Initialize smoothing
  for (int i = 0; i < MODEL_NUM_CLASSES; i++) {
    smoothed_probs[i] = 1.0f / MODEL_NUM_CLASSES;
  }
  for (int i = 0; i < VOTE_WINDOW; i++) {
    vote_history[i] = -1;
  }
  vote_head = 0;
  current_output = -1;
  pending_output = -1;
  pending_count = 0;
}

bool inference_add_sample(uint16_t *channels) {
  // Add to circular buffer
  for (int i = 0; i < NUM_CHANNELS; i++) {
    window_buffer[buffer_head][i] = channels[i];
  }

  buffer_head = (buffer_head + 1) % INFERENCE_WINDOW_SIZE;

  if (samples_collected < INFERENCE_WINDOW_SIZE) {
    samples_collected++;
    return false;
  }

  return true; // Buffer is full (always ready in sliding window, but caller
               // controls stride)
}

// --- Feature Extraction ---

static void compute_features(float *features_out) {
  // Process each channel
  // We need to iterate over the logical window (unrolling circular buffer)

  for (int ch = 0; ch < NUM_CHANNELS; ch++) {
    float sum = 0;
    float sq_sum = 0;

    // Pass 1: Mean (for centering) and raw values collection
    // We could optimize by not copying, but accessing logically is safer
    float signal[INFERENCE_WINDOW_SIZE];

    int idx = buffer_head; // Oldest sample
    for (int i = 0; i < INFERENCE_WINDOW_SIZE; i++) {
      signal[i] = (float)window_buffer[idx][ch];
      sum += signal[i];
      idx = (idx + 1) % INFERENCE_WINDOW_SIZE;
    }

    float mean = sum / INFERENCE_WINDOW_SIZE;

    // Pass 2: Centering and Features
    float wl = 0;
    int zc = 0;
    int ssc = 0;

    // Center the signal
    for (int i = 0; i < INFERENCE_WINDOW_SIZE; i++) {
      signal[i] -= mean;
      sq_sum += signal[i] * signal[i];
    }

    float rms = sqrtf(sq_sum / INFERENCE_WINDOW_SIZE);

    // Thresholds
    float zc_thresh = FEAT_ZC_THRESH * rms;
    float ssc_thresh = (FEAT_SSC_THRESH * rms) *
                       (FEAT_SSC_THRESH * rms); // threshold is on diff product

    for (int i = 0; i < INFERENCE_WINDOW_SIZE - 1; i++) {
      // WL
      wl += fabsf(signal[i + 1] - signal[i]);

      // ZC
      if ((signal[i] > 0 && signal[i + 1] < 0) ||
          (signal[i] < 0 && signal[i + 1] > 0)) {
        if (fabsf(signal[i] - signal[i + 1]) > zc_thresh) {
          zc++;
        }
      }

      // SSC (needs 3 points, so loop to N-2)
      if (i < INFERENCE_WINDOW_SIZE - 2) {
        float diff1 = signal[i + 1] - signal[i];
        float diff2 = signal[i + 1] - signal[i + 2];
        if ((diff1 * diff2) > ssc_thresh) {
          ssc++;
        }
      }
    }

    // Store features: [RMS, WL, ZC, SSC] per channel
    int base = ch * 4;
    features_out[base + 0] = rms;
    features_out[base + 1] = wl;
    features_out[base + 2] = (float)zc;
    features_out[base + 3] = (float)ssc;
  }
}

// --- Prediction ---

int inference_predict(float *confidence) {
  if (samples_collected < INFERENCE_WINDOW_SIZE) {
    return -1;
  }

  // 1. Extract Features
  float features[MODEL_NUM_FEATURES];
  compute_features(features);

  // 2. LDA Inference (Linear Score)
  float raw_scores[MODEL_NUM_CLASSES];
  float max_score = -1e9;
  int max_idx = 0;

  // Calculate raw discriminative scores
  for (int c = 0; c < MODEL_NUM_CLASSES; c++) {
    float score = LDA_INTERCEPTS[c];
    for (int f = 0; f < MODEL_NUM_FEATURES; f++) {
      score += features[f] * LDA_WEIGHTS[c][f];
    }
    raw_scores[c] = score;
  }

  // Convert scores to probabilities (Softmax)
  // LDA scores are log-likelihoods + const. Softmax is appropriate.
  float sum_exp = 0;
  float probas[MODEL_NUM_CLASSES];

  // Numerical stability: subtract max
  // Create temp copy for max finding
  for (int c = 0; c < MODEL_NUM_CLASSES; c++) {
    if (raw_scores[c] > max_score)
      max_score = raw_scores[c];
  }

  for (int c = 0; c < MODEL_NUM_CLASSES; c++) {
    probas[c] = expf(raw_scores[c] - max_score);
    sum_exp += probas[c];
  }
  for (int c = 0; c < MODEL_NUM_CLASSES; c++) {
    probas[c] /= sum_exp;
  }

  // 3. Smoothing
  // 3a. Probability EMA
  float max_smoothed_prob = 0;
  int smoothed_winner = 0;

  for (int c = 0; c < MODEL_NUM_CLASSES; c++) {
    smoothed_probs[c] = (SMOOTHING_FACTOR * smoothed_probs[c]) +
                        ((1.0f - SMOOTHING_FACTOR) * probas[c]);

    if (smoothed_probs[c] > max_smoothed_prob) {
      max_smoothed_prob = smoothed_probs[c];
      smoothed_winner = c;
    }
  }

  // 3b. Majority Vote
  vote_history[vote_head] = smoothed_winner;
  vote_head = (vote_head + 1) % VOTE_WINDOW;

  int counts[MODEL_NUM_CLASSES];
  memset(counts, 0, sizeof(counts));

  for (int i = 0; i < VOTE_WINDOW; i++) {
    if (vote_history[i] != -1) {
      counts[vote_history[i]]++;
    }
  }

  int majority_winner = 0;
  int majority_count = 0;
  for (int c = 0; c < MODEL_NUM_CLASSES; c++) {
    if (counts[c] > majority_count) {
      majority_count = counts[c];
      majority_winner = c;
    }
  }

  // 3c. Debounce
  int final_result = current_output;

  if (current_output == -1) {
    current_output = majority_winner;
    pending_output = majority_winner;
    pending_count = 1;
    final_result = majority_winner;
  } else if (majority_winner == current_output) {
    pending_output = majority_winner;
    pending_count = 1;
  } else if (majority_winner == pending_output) {
    pending_count++;
    if (pending_count >= DEBOUNCE_COUNT) {
      current_output = majority_winner;
      final_result = majority_winner;
    }
  } else {
    pending_output = majority_winner;
    pending_count = 1;
  }

  // Use smoothed probability of the final winner as confidence
  // Or simpler: use fraction of votes
  *confidence = (float)majority_count / VOTE_WINDOW;

  return final_result;
}

const char *inference_get_class_name(int class_idx) {
  if (class_idx >= 0 && class_idx < MODEL_NUM_CLASSES) {
    return MODEL_CLASS_NAMES[class_idx];
  }
  return "UNKNOWN";
}

int inference_get_gesture_enum(int class_idx) {
  const char *name = inference_get_class_name(class_idx);

  // Map string name to gesture_t enum
  // Strings must match those in Python list: ["fist", "hook_em", "open",
  // "rest", "thumbs_up"] Note: Python strings are lowercase, config.h enums
  // are: GESTURE_NONE=0, REST=1, FIST=2, OPEN=3, HOOK_EM=4, THUMBS_UP=5

  // Case-insensitive check would be safer, but let's assume Python output is
  // lowercase as seen in scripts or uppercase if specified. In
  // learning_data_collection.py, they seem to be "rest", "open", "fist", etc.

  // Simple string matching
  if (strcmp(name, "rest") == 0 || strcmp(name, "REST") == 0)
    return GESTURE_REST;
  if (strcmp(name, "fist") == 0 || strcmp(name, "FIST") == 0)
    return GESTURE_FIST;
  if (strcmp(name, "open") == 0 || strcmp(name, "OPEN") == 0)
    return GESTURE_OPEN;
  if (strcmp(name, "hook_em") == 0 || strcmp(name, "HOOK_EM") == 0)
    return GESTURE_HOOK_EM;
  if (strcmp(name, "thumbs_up") == 0 || strcmp(name, "THUMBS_UP") == 0)
    return GESTURE_THUMBS_UP;

  return GESTURE_NONE;
}
