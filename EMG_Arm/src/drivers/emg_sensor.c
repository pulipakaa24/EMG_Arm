/**
 * @file emg_sensor.c
 * @brief EMG sensor driver implementation.
 *
 * Provides EMG readings - fake data for now, real ADC when sensors arrive.
 */

#include "emg_sensor.h"
#include "esp_timer.h"
#include <stdlib.h>

/*******************************************************************************
 * Public Functions
 ******************************************************************************/

void emg_sensor_init(void)
{
#if FEATURE_FAKE_EMG
    /* Seed random number generator for fake data */
    srand((unsigned int)esp_timer_get_time());
#else
    /* TODO: Configure ADC channels when sensors arrive */
    /* adc1_config_width(EMG_ADC_WIDTH); */
    /* adc1_config_channel_atten(ADC_EMG_CH0, EMG_ADC_ATTEN); */
    /* ... */
#endif
}

void emg_sensor_read(emg_sample_t *sample)
{
    sample->timestamp_ms = emg_sensor_get_timestamp_ms();

#if FEATURE_FAKE_EMG
    /*
     * Generate fake EMG data:
     * - Base value around 512 (middle of 10-bit range, matching Python sim)
     * - Random noise of +/- 50
     * - Mimics real EMG baseline noise
     */
    for (int i = 0; i < EMG_NUM_CHANNELS; i++) {
        int noise = (rand() % 101) - 50;  /* -50 to +50 */
        sample->channels[i] = (uint16_t)(512 + noise);
    }
#else
    /* TODO: Real ADC reads when sensors arrive */
    /* sample->channels[0] = adc1_get_raw(ADC_EMG_CH0); */
    /* sample->channels[1] = adc1_get_raw(ADC_EMG_CH1); */
    /* ... */
#endif
}

uint32_t emg_sensor_get_timestamp_ms(void)
{
    return (uint32_t)(esp_timer_get_time() / 1000);
}
