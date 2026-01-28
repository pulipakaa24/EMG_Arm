/**
 * @file model_weights.h
 * @brief Trained LDA model weights exported from Python.
 * @date 2026-01-27 21:35:17
 */

#ifndef MODEL_WEIGHTS_H
#define MODEL_WEIGHTS_H

#include <stdint.h>

/* Metadata */
#define MODEL_NUM_CLASSES 5
#define MODEL_NUM_FEATURES 16

/* Class Names */
static const char* MODEL_CLASS_NAMES[MODEL_NUM_CLASSES] = {
    "fist",
    "hook_em",
    "open",
    "rest",
    "thumbs_up",
};

/* Feature Extractor Parameters */
#define FEAT_ZC_THRESH 0.1f
#define FEAT_SSC_THRESH 0.1f

/* LDA Intercepts/Biases */
static const float LDA_INTERCEPTS[MODEL_NUM_CLASSES] = {
    -14.097581f, -2.018629f, -4.478267f, 1.460458f, -5.562349f
};

/* LDA Coefficients (Weights) */
static const float LDA_WEIGHTS[MODEL_NUM_CLASSES][MODEL_NUM_FEATURES] = {
    /* fist */
    {
        0.070110f, -0.002554f, 0.043924f, 0.020555f, -0.660305f, 0.010691f, -0.074429f, -0.037253f, 
        0.057908f, -0.002655f, 0.042119f, -0.052956f, 0.063822f, 0.006184f, -0.025462f, 0.040815f, 
    },
    /* hook_em */
    {
        -0.002511f, 0.001034f, 0.027889f, 0.026006f, 0.183681f, -0.000773f, 0.016791f, -0.027926f, 
        -0.023321f, 0.000770f, 0.059023f, -0.056021f, 0.237063f, -0.007423f, 0.082101f, -0.021472f, 
    },
    /* open */
    {
        -0.006170f, 0.000208f, -0.041151f, 0.013271f, 0.054508f, -0.002356f, 0.000170f, 0.012941f, 
        -0.106180f, 0.003538f, -0.013656f, -0.017712f, 0.131131f, -0.002623f, -0.007022f, 0.024497f, 
    },
    /* rest */
    {
        -0.011094f, 0.000160f, -0.012547f, -0.011058f, 0.130577f, -0.001942f, 0.020823f, -0.001961f, 
        0.018021f, -0.000404f, -0.065598f, 0.039676f, 0.018679f, -0.001522f, 0.023302f, -0.008474f, 
    },
    /* thumbs_up */
    {
        -0.016738f, 0.000488f, 0.024199f, -0.024643f, -0.044912f, 0.000153f, -0.011080f, 0.043487f, 
        0.051828f, -0.001670f, 0.109633f, 0.004154f, -0.460694f, 0.008616f, -0.104097f, -0.020886f, 
    },
};

#endif /* MODEL_WEIGHTS_H */