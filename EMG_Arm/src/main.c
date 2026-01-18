/**
 * @file main.c
 * @brief Main application entry point for the EMG-controlled robotic hand.
 *
 * This application controls a 5-finger robotic hand using servo motors.
 * The servos are driven by PWM signals generated through the ESP32's LEDC
 * peripheral. Future versions will integrate EMG signal processing to
 * translate muscle activity into hand gestures.
 *
 * Hardware Platform: ESP32-S3-DevKitC-1
 * Framework: ESP-IDF with FreeRTOS
 */

#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include "servo.h"
#include "gestures.h"

/*******************************************************************************
 * Configuration
 ******************************************************************************/

/** @brief Delay between demo movements in milliseconds. */
#define DEMO_DELAY_MS  1000

/*******************************************************************************
 * Application Entry Point
 ******************************************************************************/

/**
 * @brief Main application entry point.
 *
 * Initializes the servo hardware and runs a demo sequence.
 * The demo can be configured to test individual finger control
 * or simultaneous hand gestures.
 */
void app_main(void)
{
    /* Initialize servo motors */
    servo_init();

    /* Run demo sequence */
    while (1) {
        /* Option 1: Test individual finger control */
        // demo_individual_fingers(DEMO_DELAY_MS);

        /* Option 2: Test simultaneous hand gestures */
        demo_close_open(DEMO_DELAY_MS);
    }
}
