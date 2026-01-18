/**
 * @file servo.h
 * @brief Servo motor control interface for the EMG-controlled robotic hand.
 *
 * This module provides low-level servo motor control using the ESP32's LEDC
 * (LED Controller) peripheral for PWM generation. It handles initialization
 * and position control for all five finger servos.
 *
 * Hardware Configuration:
 *   - Thumb:  GPIO 1, LEDC Channel 0
 *   - Index:  GPIO 4, LEDC Channel 1
 *   - Middle: GPIO 5, LEDC Channel 2
 *   - Ring:   GPIO 6, LEDC Channel 3
 *   - Pinky:  GPIO 7, LEDC Channel 4
 *
 * @note All servos share a single LEDC timer configured for 50Hz (standard servo frequency).
 */

#ifndef SERVO_H
#define SERVO_H

#include "driver/ledc.h"
#include "driver/gpio.h"

/*******************************************************************************
 * GPIO Pin Definitions
 ******************************************************************************/

#define THUMB_SERVO_PIN   GPIO_NUM_1
#define INDEX_SERVO_PIN   GPIO_NUM_4
#define MIDDLE_SERVO_PIN  GPIO_NUM_5
#define RING_SERVO_PIN    GPIO_NUM_6
#define PINKY_SERVO_PIN   GPIO_NUM_7

/*******************************************************************************
 * LEDC Channel Definitions
 ******************************************************************************/

#define THUMB_CHANNEL     LEDC_CHANNEL_0
#define INDEX_CHANNEL     LEDC_CHANNEL_1
#define MIDDLE_CHANNEL    LEDC_CHANNEL_2
#define RING_CHANNEL      LEDC_CHANNEL_3
#define PINKY_CHANNEL     LEDC_CHANNEL_4

/*******************************************************************************
 * Servo Position Definitions
 *
 * These duty cycle values correspond to servo positions at 14-bit resolution.
 * At 50Hz with 14-bit resolution (16384 counts per 20ms period):
 *   - 1ms pulse (~0 degrees)   = ~819 counts
 *   - 2ms pulse (~180 degrees) = ~1638 counts
 *
 * Actual values may vary based on specific servo characteristics.
 ******************************************************************************/

#define SERVO_POS_MIN     430   /**< Duty cycle for 0 degrees (finger extended) */
#define SERVO_POS_MAX     2048  /**< Duty cycle for 180 degrees (finger flexed) */

/*******************************************************************************
 * Finger Index Enumeration
 ******************************************************************************/

/**
 * @brief Enumeration for finger identification.
 */
typedef enum {
    FINGER_THUMB = 0,
    FINGER_INDEX,
    FINGER_MIDDLE,
    FINGER_RING,
    FINGER_PINKY,
    FINGER_COUNT  /**< Total number of fingers (5) */
} finger_t;

/*******************************************************************************
 * Public Function Declarations
 ******************************************************************************/

/**
 * @brief Initialize all servo motors.
 *
 * Configures the LEDC timer and channels for all five finger servos.
 * All servos are initialized to the extended (open) position.
 *
 * @note Must be called before any other servo functions.
 */
void servo_init(void);

/**
 * @brief Set a specific servo to a given duty cycle position.
 *
 * @param channel The LEDC channel corresponding to the servo.
 * @param duty    The duty cycle value (use SERVO_POS_MIN to SERVO_POS_MAX).
 */
void servo_set_position(ledc_channel_t channel, uint32_t duty);

/**
 * @brief Set a finger servo to a specific position.
 *
 * @param finger The finger to control (use finger_t enumeration).
 * @param duty   The duty cycle value (use SERVO_POS_MIN to SERVO_POS_MAX).
 */
void servo_set_finger(finger_t finger, uint32_t duty);

/**
 * @brief Convert an angle in degrees to the corresponding duty cycle value.
 *
 * Performs linear interpolation between SERVO_POS_MIN (0 degrees) and
 * SERVO_POS_MAX (180 degrees). Input values are clamped to the valid range.
 *
 * @param degrees The desired angle in degrees (0 to 180).
 * @return The corresponding duty cycle value for the LEDC peripheral.
 *
 * @example
 *   uint32_t duty = servo_degrees_to_duty(90);  // Get duty for 90 degrees
 *   servo_set_finger(FINGER_INDEX, duty);
 */
uint32_t servo_degrees_to_duty(float degrees);

/**
 * @brief Set a finger servo to a specific angle in degrees.
 *
 * Convenience function that combines degree-to-duty conversion with
 * finger positioning. This is the recommended function for most use cases.
 *
 * @param finger  The finger to control (use finger_t enumeration).
 * @param degrees The desired angle in degrees (0 = extended, 180 = flexed).
 *
 * @example
 *   servo_set_finger_degrees(FINGER_THUMB, 90);   // Move thumb to 90 degrees
 *   servo_set_finger_degrees(FINGER_INDEX, 45);   // Move index to 45 degrees
 */
void servo_set_finger_degrees(finger_t finger, float degrees);

#endif /* SERVO_H */
