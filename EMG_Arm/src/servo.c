/**
 * @file servo.c
 * @brief Servo motor control implementation for the EMG-controlled robotic hand.
 *
 * This module implements low-level servo control using the ESP32's LEDC peripheral.
 * The LEDC is configured to generate 50Hz PWM signals suitable for standard hobby servos.
 */

#include "servo.h"
#include "esp_err.h"

/*******************************************************************************
 * Private Constants
 ******************************************************************************/

/** @brief PWM frequency for servo control (standard is 50Hz). */
#define SERVO_PWM_FREQ_HZ     50

/** @brief PWM resolution in bits (14-bit = 16384 levels). */
#define SERVO_PWM_RESOLUTION  LEDC_TIMER_14_BIT

/** @brief LEDC speed mode (ESP32-S3 only supports low-speed mode). */
#define SERVO_SPEED_MODE      LEDC_LOW_SPEED_MODE

/** @brief LEDC timer used for all servos. */
#define SERVO_TIMER           LEDC_TIMER_0

/*******************************************************************************
 * Private Data
 ******************************************************************************/

/**
 * @brief Mapping of finger indices to GPIO pins.
 */
static const int servo_pins[FINGER_COUNT] = {
    THUMB_SERVO_PIN,
    INDEX_SERVO_PIN,
    MIDDLE_SERVO_PIN,
    RING_SERVO_PIN,
    PINKY_SERVO_PIN
};

/**
 * @brief Mapping of finger indices to LEDC channels.
 */
static const ledc_channel_t servo_channels[FINGER_COUNT] = {
    THUMB_CHANNEL,
    INDEX_CHANNEL,
    MIDDLE_CHANNEL,
    RING_CHANNEL,
    PINKY_CHANNEL
};

/*******************************************************************************
 * Public Function Implementations
 ******************************************************************************/

void servo_init(void)
{
    /* Configure LEDC timer (shared by all servo channels) */
    ledc_timer_config_t timer_config = {
        .speed_mode      = SERVO_SPEED_MODE,
        .timer_num       = SERVO_TIMER,
        .duty_resolution = SERVO_PWM_RESOLUTION,
        .freq_hz         = SERVO_PWM_FREQ_HZ,
        .clk_cfg         = LEDC_AUTO_CLK
    };
    ESP_ERROR_CHECK(ledc_timer_config(&timer_config));

    /* Configure LEDC channel for each finger servo */
    for (int i = 0; i < FINGER_COUNT; i++) {
        ledc_channel_config_t channel_config = {
            .speed_mode = SERVO_SPEED_MODE,
            .channel    = servo_channels[i],
            .timer_sel  = SERVO_TIMER,
            .intr_type  = LEDC_INTR_DISABLE,
            .gpio_num   = servo_pins[i],
            .duty       = SERVO_POS_MIN,  /* Start with fingers extended */
            .hpoint     = 0
        };
        ESP_ERROR_CHECK(ledc_channel_config(&channel_config));
    }
}

void servo_set_position(ledc_channel_t channel, uint32_t duty)
{
    ledc_set_duty(SERVO_SPEED_MODE, channel, duty);
    ledc_update_duty(SERVO_SPEED_MODE, channel);
}

void servo_set_finger(finger_t finger, uint32_t duty)
{
    if (finger >= FINGER_COUNT) {
        return;  /* Invalid finger index */
    }
    servo_set_position(servo_channels[finger], duty);
}

uint32_t servo_degrees_to_duty(float degrees)
{
    /* Clamp input to valid range */
    if (degrees < 0.0f) {
        degrees = 0.0f;
    } else if (degrees > 180.0f) {
        degrees = 180.0f;
    }

    /*
     * Linear interpolation formula:
     * duty = min + (degrees / 180) * (max - min)
     *
     * Where:
     *   - min = SERVO_POS_MIN (duty at 0 degrees)
     *   - max = SERVO_POS_MAX (duty at 180 degrees)
     */
    float duty = SERVO_POS_MIN + (degrees / 180.0f) * (SERVO_POS_MAX - SERVO_POS_MIN);

    return (uint32_t)duty;
}

void servo_set_finger_degrees(finger_t finger, float degrees)
{
    uint32_t duty = servo_degrees_to_duty(degrees);
    servo_set_finger(finger, duty);
}
