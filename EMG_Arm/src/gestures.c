/**
 * @file gestures.c
 * @brief Gesture control implementation for the EMG-controlled robotic hand.
 *
 * This module implements high-level gesture functions using the low-level
 * servo control interface. Each gesture function translates intuitive
 * commands into appropriate servo positions.
 */

#include "gestures.h"
#include "servo.h"
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>

/*******************************************************************************
 * Individual Finger Control - Flex Functions
 ******************************************************************************/

void flex_thumb(void)
{
    servo_set_finger(FINGER_THUMB, SERVO_POS_MAX);
}

void flex_index(void)
{
    servo_set_finger(FINGER_INDEX, SERVO_POS_MAX);
}

void flex_middle(void)
{
    servo_set_finger(FINGER_MIDDLE, SERVO_POS_MAX);
}

void flex_ring(void)
{
    servo_set_finger(FINGER_RING, SERVO_POS_MAX);
}

void flex_pinky(void)
{
    servo_set_finger(FINGER_PINKY, SERVO_POS_MAX);
}

/*******************************************************************************
 * Individual Finger Control - Unflex Functions
 ******************************************************************************/

void unflex_thumb(void)
{
    servo_set_finger(FINGER_THUMB, SERVO_POS_MIN);
}

void unflex_index(void)
{
    servo_set_finger(FINGER_INDEX, SERVO_POS_MIN);
}

void unflex_middle(void)
{
    servo_set_finger(FINGER_MIDDLE, SERVO_POS_MIN);
}

void unflex_ring(void)
{
    servo_set_finger(FINGER_RING, SERVO_POS_MIN);
}

void unflex_pinky(void)
{
    servo_set_finger(FINGER_PINKY, SERVO_POS_MIN);
}

/*******************************************************************************
 * Composite Gestures
 ******************************************************************************/

void gesture_make_fist(void)
{
    /* Flex all fingers simultaneously */
    flex_thumb();
    flex_index();
    flex_middle();
    flex_ring();
    flex_pinky();
}

void gesture_open_hand(void)
{
    /* Extend all fingers simultaneously */
    unflex_thumb();
    unflex_index();
    unflex_middle();
    unflex_ring();
    unflex_pinky();
}

/*******************************************************************************
 * Demo/Test Sequences
 ******************************************************************************/

void demo_individual_fingers(uint32_t delay_ms)
{
    /* Thumb */
    flex_thumb();
    vTaskDelay(pdMS_TO_TICKS(delay_ms));
    unflex_thumb();
    vTaskDelay(pdMS_TO_TICKS(delay_ms));

    /* Index */
    flex_index();
    vTaskDelay(pdMS_TO_TICKS(delay_ms));
    unflex_index();
    vTaskDelay(pdMS_TO_TICKS(delay_ms));

    /* Middle */
    flex_middle();
    vTaskDelay(pdMS_TO_TICKS(delay_ms));
    unflex_middle();
    vTaskDelay(pdMS_TO_TICKS(delay_ms));

    /* Ring */
    flex_ring();
    vTaskDelay(pdMS_TO_TICKS(delay_ms));
    unflex_ring();
    vTaskDelay(pdMS_TO_TICKS(delay_ms));

    /* Pinky */
    flex_pinky();
    vTaskDelay(pdMS_TO_TICKS(delay_ms));
    unflex_pinky();
    vTaskDelay(pdMS_TO_TICKS(delay_ms));
}

void demo_close_open(uint32_t delay_ms)
{
    gesture_make_fist();
    vTaskDelay(pdMS_TO_TICKS(delay_ms));
    gesture_open_hand();
    vTaskDelay(pdMS_TO_TICKS(delay_ms));
}
