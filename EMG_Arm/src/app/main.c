/**
 * @file main.c
 * @brief Application entry point for the EMG-controlled robotic hand.
 *
 * This is the top-level application that initializes all subsystems
 * and runs the main loop. Currently configured to stream EMG data
 * over USB serial for Python to receive.
 *
 * @note This is Layer 4 (Application).
 */

#include <stdio.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include "esp_timer.h"

#include "config/config.h"
#include "drivers/hand.h"
#include "drivers/emg_sensor.h"
#include "core/gestures.h"

/*******************************************************************************
 * Private Functions
 ******************************************************************************/

/**
 * @brief Stream EMG data over USB serial.
 *
 * Outputs data in format: "timestamp_ms,ch0,ch1,ch2,ch3\n"
 * This matches what Python's SimulatedEMGStream produces.
 */
static void stream_emg_data(void)
{
    emg_sample_t sample;

    printf("\n[EMG] Starting data stream at %d Hz...\n", EMG_SAMPLE_RATE_HZ);
    printf("[EMG] Format: timestamp_ms,ch0,ch1,ch2,ch3\n\n");

    /*
     * FreeRTOS tick rate is set to 1000 Hz in sdkconfig.defaults (1ms per tick).
     * Delay of 1 tick = 1ms, giving us the full 1000 Hz sample rate.
     */
    const TickType_t delay_ticks = 1;  /* 1 tick = 1ms at 1000 Hz tick rate */

    while (1) {
        /* Read EMG (fake or real depending on FEATURE_FAKE_EMG) */
        emg_sensor_read(&sample);

        /* Output in CSV format matching Python expectation */
        printf("%lu,%u,%u,%u,%u\n",
               (unsigned long)sample.timestamp_ms,
               sample.channels[0],
               sample.channels[1],
               sample.channels[2],
               sample.channels[3]);

        /* Yield to FreeRTOS scheduler - prevents watchdog timeout */
        vTaskDelay(delay_ticks);
    }
}

/**
 * @brief Run demo mode - cycle through gestures.
 */
static void run_demo(void)
{
    printf("\n[DEMO] Running gesture demo...\n");

    while (1) {
        gestures_demo_fist(1000);
    }
}

/*******************************************************************************
 * Application Entry Point
 ******************************************************************************/

void app_main(void)
{
    printf("\n");
    printf("========================================\n");
    printf("  Bucky Arm - EMG Robotic Hand\n");
    printf("  Firmware v1.0.0\n");
    printf("========================================\n\n");

    /* Initialize subsystems */
    printf("[INIT] Initializing hand (servos)...\n");
    hand_init();

    printf("[INIT] Initializing EMG sensor...\n");
    emg_sensor_init();

#if FEATURE_FAKE_EMG
    printf("[INIT] Using FAKE EMG data (sensors not connected)\n");
#else
    printf("[INIT] Using REAL EMG sensors\n");
#endif

    printf("[INIT] Done!\n\n");

    /*
     * Choose what to run:
     * - stream_emg_data(): Send EMG data to laptop (Phase 1)
     * - run_demo(): Test servo movement
     *
     * For now, we stream EMG data.
     * Comment out and use run_demo() to test servos.
     */
    stream_emg_data();

    /* Alternative: run servo demo */
    // run_demo();
}
