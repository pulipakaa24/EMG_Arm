/**
 * @file emg_sensor.c
 * @brief EMG sensor driver implementation.
 *
 * Provides EMG readings - fake data for now, real ADC when sensors arrive.
 */

#include "emg_sensor.h"
#include "esp_timer.h"
#include <stdlib.h>
#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_adc/adc_oneshot.h"
#include "esp_adc/adc_cali.h"
#include "esp_adc/adc_cali_scheme.h"
#include "esp_err.h"

adc_oneshot_unit_handle_t adc1_handle; 
adc_cali_handle_t cali_handle = NULL;

const uint8_t emg_channels[EMG_NUM_CHANNELS] = {
    ADC_CHANNEL_1,  // GPIO 2 - EMG Channel 0
    ADC_CHANNEL_2,  // GPIO 3 - EMG Channel 1
    ADC_CHANNEL_8,  // GPIO 9 - EMG Channel 2
    ADC_CHANNEL_9   // GPIO 10 - EMG Channel 3
};

/*******************************************************************************
 * Public Functions
 ******************************************************************************/

void emg_sensor_init(void)
{
#if FEATURE_FAKE_EMG
    /* Seed random number generator for fake data */
    srand((unsigned int)esp_timer_get_time());
#else
    // 1. --- ADC Unit Setup ---
    adc_oneshot_unit_init_cfg_t init_config1 = {
        .unit_id = ADC_UNIT_1,
        .ulp_mode = ADC_ULP_MODE_DISABLE,
    };
    ESP_ERROR_CHECK(adc_oneshot_new_unit(&init_config1, &adc1_handle));

    // 2. --- ADC Channel Setup (GPIO 1?) ---
    // Ensure the channel matches your GPIO in pinmap. For ADC1, GPIO1 is usually not CH0.
    // Check your datasheet! (e.g., on S3, GPIO 1 is ADC1_CH0)
    adc_oneshot_chan_cfg_t config = {
        .bitwidth = ADC_BITWIDTH_DEFAULT, // 12-bit for S3
        .atten = ADC_ATTEN_DB_12,         // Allows up to ~3.1V
    };
    for (uint8_t i = 0; i < EMG_NUM_CHANNELS; i++)
      ESP_ERROR_CHECK(adc_oneshot_config_channel(adc1_handle, emg_channels[i], &config));

    // 3. --- Calibration Setup (CORRECTED for S3) ---
    // ESP32-S3 uses Curve Fitting, not Line Fitting
    adc_cali_curve_fitting_config_t cali_config = {
        .unit_id = ADC_UNIT_1,
        .atten = ADC_ATTEN_DB_12,
        .bitwidth = ADC_BITWIDTH_DEFAULT,
    };
    ESP_ERROR_CHECK(adc_cali_create_scheme_curve_fitting(&cali_config, &cali_handle));
    
    // while (1) {
    //     int raw_val, voltage_mv;
        
    //     // Read Raw
    //     ESP_ERROR_CHECK(adc_oneshot_read(adc1_handle, ADC_CHANNEL_1, &raw_val));
        
    //     // Convert to mV using calibration
    //     ESP_ERROR_CHECK(adc_cali_raw_to_voltage(cali_handle, raw_val, &voltage_mv));

    //     printf("Raw: %d | Voltage: %d mV\n", raw_val, voltage_mv);
    //     vTaskDelay(pdMS_TO_TICKS(500));
    // }
#endif
}

void emg_sensor_read(emg_sample_t *sample)
{
    sample->timestamp_ms = emg_sensor_get_timestamp_ms();

#if FEATURE_FAKE_EMG
    /*
     * Generate fake EMG data:
     * - Base value around 1650 (middle of 3.3V millivolt range)
     * - Random noise of +/- 50
     * - Mimics real EMG baseline noise
     */
    for (int i = 0; i < EMG_NUM_CHANNELS; i++) {
        int noise = (rand() % 101) - 50;  /* -50 to +50 */
        sample->channels[i] = (uint16_t)(1650 + noise);
    }
#else
    int raw_val, voltage_mv;
    for (uint8_t i = 0; i < EMG_NUM_CHANNELS; i++) {
      ESP_ERROR_CHECK(adc_oneshot_read(adc1_handle, emg_channels[i], &raw_val));
      ESP_ERROR_CHECK(adc_cali_raw_to_voltage(cali_handle, raw_val, &voltage_mv));
      sample->channels[i] = (uint16_t) voltage_mv;
    }

#endif
}

uint32_t emg_sensor_get_timestamp_ms(void)
{
    return (uint32_t)(esp_timer_get_time() / 1000);
}
