/**
 * @file config.h
 * @brief Centralized configuration for the EMG-controlled robotic hand.
 *
 * All hardware pin definitions and system constants in one place.
 * Modify this file to adapt to different hardware configurations.
 */

#ifndef CONFIG_H
#define CONFIG_H

#include "driver/gpio.h"
#include "driver/ledc.h"

/*******************************************************************************
 * Feature Flags
 *
 * Compile-time switches to enable/disable features.
 * Set to 1 to enable, 0 to disable.
 ******************************************************************************/

/**
 * @brief Use fake EMG data (random values) instead of real ADC reads.
 *
 * Set to 1 while waiting for EMG sensors to arrive.
 * Set to 0 when ready to use real sensors.
 */
#define FEATURE_FAKE_EMG          0

/*******************************************************************************
 * GPIO Pin Definitions - Servos
 ******************************************************************************/

#define PIN_SERVO_THUMB           GPIO_NUM_1
#define PIN_SERVO_INDEX           GPIO_NUM_4
#define PIN_SERVO_MIDDLE          GPIO_NUM_5
#define PIN_SERVO_RING            GPIO_NUM_6
#define PIN_SERVO_PINKY           GPIO_NUM_7

/*******************************************************************************
 * Servo PWM Configuration
 ******************************************************************************/

#define SERVO_PWM_FREQ_HZ         50                  /**< Standard servo frequency */
#define SERVO_PWM_RESOLUTION      LEDC_TIMER_14_BIT   /**< 14-bit = 16384 levels */
#define SERVO_PWM_TIMER           LEDC_TIMER_0        /**< LEDC timer for servos */
#define SERVO_PWM_SPEED_MODE      LEDC_LOW_SPEED_MODE /**< ESP32-S3 uses low-speed */

#define SERVO_DUTY_MIN            430   /**< Duty cycle for 0 degrees (extended) */
#define SERVO_DUTY_MAX            2048  /**< Duty cycle for 180 degrees (flexed) */

/*******************************************************************************
 * LEDC Channel Assignments
 ******************************************************************************/

#define LEDC_CH_THUMB             LEDC_CHANNEL_0
#define LEDC_CH_INDEX             LEDC_CHANNEL_1
#define LEDC_CH_MIDDLE            LEDC_CHANNEL_2
#define LEDC_CH_RING              LEDC_CHANNEL_3
#define LEDC_CH_PINKY             LEDC_CHANNEL_4

/*******************************************************************************
 * EMG Configuration
 ******************************************************************************/

#define EMG_NUM_CHANNELS          4     /**< Number of EMG sensor channels */
#define EMG_SAMPLE_RATE_HZ        1000  /**< Samples per second per channel */

/*******************************************************************************
 * Common Type Definitions
 ******************************************************************************/

/**
 * @brief Finger identification.
 */
typedef enum {
    FINGER_THUMB = 0,
    FINGER_INDEX,
    FINGER_MIDDLE,
    FINGER_RING,
    FINGER_PINKY,
    FINGER_COUNT    /**< Total number of fingers (5) */
} finger_t;

/**
 * @brief Recognized gestures.
 */
typedef enum {
    GESTURE_NONE = 0,
    GESTURE_REST,
    GESTURE_FIST,
    GESTURE_OPEN,
    GESTURE_HOOK_EM,
    GESTURE_THUMBS_UP,
    GESTURE_COUNT
} gesture_t;

/**
 * @brief System operating modes.
 */
typedef enum {
    MODE_IDLE = 0,      /**< Waiting for commands */
    MODE_DATA_STREAM,   /**< Streaming EMG data to laptop */
    MODE_COMMAND,       /**< Executing gesture commands from laptop */
    MODE_DEMO,          /**< Running demo sequence */
    MODE_COUNT
} system_mode_t;

#endif /* CONFIG_H */
