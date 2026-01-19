/**
 * @file servo_hal.h
 * @brief Hardware Abstraction Layer for servo PWM control.
 *
 * This module provides low-level access to the ESP32's LEDC peripheral
 * for generating PWM signals to control servo motors.
 *
 * @note This is Layer 1 (HAL). Only drivers/ should use this directly.
 */

#ifndef SERVO_HAL_H
#define SERVO_HAL_H

#include <stdint.h>
#include "config/config.h"

/*******************************************************************************
 * Public Functions
 ******************************************************************************/

/**
 * @brief Initialize the LEDC peripheral for all servo channels.
 *
 * Configures the timer and all 5 channels for 50Hz PWM output.
 * All servos start in the extended (open) position.
 *
 * @note Must be called once before any other servo_hal functions.
 */
void servo_hal_init(void);

/**
 * @brief Set the duty cycle for a specific finger's servo.
 *
 * @param finger Which finger (use finger_t enum from config.h)
 * @param duty   Duty cycle value (SERVO_DUTY_MIN to SERVO_DUTY_MAX)
 */
void servo_hal_set_duty(finger_t finger, uint32_t duty);

/**
 * @brief Convert degrees to duty cycle value.
 *
 * @param degrees Angle in degrees (0 to 180)
 * @return Corresponding duty cycle value
 */
uint32_t servo_hal_degrees_to_duty(float degrees);

#endif /* SERVO_HAL_H */
