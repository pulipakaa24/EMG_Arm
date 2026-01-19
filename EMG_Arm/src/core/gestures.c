/**
 * @file gestures.c
 * @brief Named gesture implementation.
 *
 * Implements gesture functions using the hand driver.
 */

#include "gestures.h"
#include "drivers/hand.h"
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>

/*******************************************************************************
 * Private Data
 ******************************************************************************/

/** @brief Gesture name lookup table. */
static const char* gesture_names[GESTURE_COUNT] = {
    "NONE",
    "REST",
    "FIST",
    "OPEN",
    "HOOK_EM",
    "THUMBS_UP"
};

/*******************************************************************************
 * Public Functions
 ******************************************************************************/

void gestures_execute(gesture_t gesture)
{
    switch (gesture) {
        case GESTURE_REST:
            gesture_rest();
            break;
        case GESTURE_FIST:
            gesture_fist();
            break;
        case GESTURE_OPEN:
            gesture_open();
            break;
        case GESTURE_HOOK_EM:
            gesture_hook_em();
            break;
        case GESTURE_THUMBS_UP:
            gesture_thumbs_up();
            break;
        default:
            break;
    }
}

const char* gestures_get_name(gesture_t gesture)
{
    if (gesture >= GESTURE_COUNT) {
        return "UNKNOWN";
    }
    return gesture_names[gesture];
}

/*******************************************************************************
 * Individual Gesture Functions
 ******************************************************************************/

void gesture_open(void)
{
    hand_unflex_all();
}

void gesture_fist(void)
{
    hand_flex_all();
}

void gesture_hook_em(void)
{
    /* Index and pinky extended, others flexed */
    hand_flex_finger(FINGER_THUMB);
    hand_unflex_finger(FINGER_INDEX);
    hand_flex_finger(FINGER_MIDDLE);
    hand_flex_finger(FINGER_RING);
    hand_unflex_finger(FINGER_PINKY);
}

void gesture_thumbs_up(void)
{
    /* Thumb extended, others flexed */
    hand_unflex_finger(FINGER_THUMB);
    hand_flex_finger(FINGER_INDEX);
    hand_flex_finger(FINGER_MIDDLE);
    hand_flex_finger(FINGER_RING);
    hand_flex_finger(FINGER_PINKY);
}

void gesture_rest(void)
{
    /* Rest is same as open - neutral position */
    gesture_open();
}

/*******************************************************************************
 * Demo Functions
 ******************************************************************************/

void gestures_demo_fingers(uint32_t delay_ms)
{
    for (int finger = 0; finger < FINGER_COUNT; finger++) {
        hand_flex_finger((finger_t)finger);
        vTaskDelay(pdMS_TO_TICKS(delay_ms));
        hand_unflex_finger((finger_t)finger);
        vTaskDelay(pdMS_TO_TICKS(delay_ms));
    }
}

void gestures_demo_fist(uint32_t delay_ms)
{
    gesture_fist();
    vTaskDelay(pdMS_TO_TICKS(delay_ms));
    gesture_open();
    vTaskDelay(pdMS_TO_TICKS(delay_ms));
}
