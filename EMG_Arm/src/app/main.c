/**
 * @file main.c
 * @brief Application entry point for the EMG-controlled robotic hand.
 *
 * Implements a robust handshake protocol with the host computer:
 * 1. Wait for "connect" command
 * 2. Acknowledge connection
 * 3. Wait for "start" command
 * 4. Stream EMG data
 * 5. Handle "stop" and "disconnect" commands
 *
 * @note This is Layer 4 (Application).
 */

#include <stdio.h>
#include <string.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <freertos/queue.h>
#include "esp_timer.h"

#include "config/config.h"
#include "drivers/hand.h"
#include "drivers/emg_sensor.h"
#include "core/gestures.h"

/*******************************************************************************
 * Constants
 ******************************************************************************/

#define CMD_BUFFER_SIZE 128
#define JSON_RESPONSE_SIZE 128

/*******************************************************************************
 * Types
 ******************************************************************************/

/**
 * @brief Device state machine.
 */
typedef enum {
    STATE_IDLE = 0,       /**< Waiting for connect command */
    STATE_CONNECTED,      /**< Connected, waiting for start command */
    STATE_STREAMING,      /**< Actively streaming EMG data */
} device_state_t;

/**
 * @brief Commands from host.
 */
typedef enum {
    CMD_NONE = 0,
    CMD_CONNECT,
    CMD_START,
    CMD_STOP,
    CMD_DISCONNECT,
} command_t;

/*******************************************************************************
 * Global State
 ******************************************************************************/

static volatile device_state_t g_device_state = STATE_IDLE;
static QueueHandle_t g_cmd_queue = NULL;

/*******************************************************************************
 * Forward Declarations
 ******************************************************************************/

static void send_ack_connect(void);

/*******************************************************************************
 * Command Parsing
 ******************************************************************************/

/**
 * @brief Parse incoming command from JSON.
 *
 * Expected format: {"cmd": "connect"}
 *
 * @param line Input line from serial
 * @return Parsed command
 */
static command_t parse_command(const char* line)
{
    /* Simple JSON parsing - look for "cmd" field */
    const char* cmd_start = strstr(line, "\"cmd\"");
    if (!cmd_start) {
        return CMD_NONE;
    }

    /* Find the value after "cmd": */
    const char* value_start = strchr(cmd_start, ':');
    if (!value_start) {
        return CMD_NONE;
    }

    /* Skip whitespace and opening quote */
    value_start++;
    while (*value_start == ' ' || *value_start == '"') {
        value_start++;
    }

    /* Match command strings */
    if (strncmp(value_start, "connect", 7) == 0) {
        return CMD_CONNECT;
    } else if (strncmp(value_start, "start", 5) == 0) {
        return CMD_START;
    } else if (strncmp(value_start, "stop", 4) == 0) {
        return CMD_STOP;
    } else if (strncmp(value_start, "disconnect", 10) == 0) {
        return CMD_DISCONNECT;
    }

    return CMD_NONE;
}

/*******************************************************************************
 * Serial Input Task
 ******************************************************************************/

/**
 * @brief FreeRTOS task to read serial input and parse commands.
 *
 * This task runs continuously, reading lines from stdin (USB serial)
 * and updating device state directly. This allows commands to interrupt
 * streaming immediately via the volatile state variable.
 *
 * @param pvParameters Unused
 */
static void serial_input_task(void* pvParameters)
{
    char line_buffer[CMD_BUFFER_SIZE];
    int line_idx = 0;

    while (1) {
        /* Read one character at a time */
        int c = getchar();

        if (c == EOF || c == 0xFF) {
            /* No data available, yield to other tasks */
            vTaskDelay(pdMS_TO_TICKS(10));
            continue;
        }

        if (c == '\n' || c == '\r') {
            /* End of line - process command */
            if (line_idx > 0) {
                line_buffer[line_idx] = '\0';

                command_t cmd = parse_command(line_buffer);

                if (cmd != CMD_NONE) {
                    /* Handle state transitions directly */
                    /* This allows streaming loop to see state changes immediately */
                    switch (g_device_state) {
                        case STATE_IDLE:
                            if (cmd == CMD_CONNECT) {
                                g_device_state = STATE_CONNECTED;
                                send_ack_connect();
                                printf("[STATE] IDLE -> CONNECTED\n");
                            }
                            break;

                        case STATE_CONNECTED:
                            if (cmd == CMD_START) {
                                g_device_state = STATE_STREAMING;
                                printf("[STATE] CONNECTED -> STREAMING\n");
                                /* Signal state machine to start streaming */
                                xQueueSend(g_cmd_queue, &cmd, 0);
                            } else if (cmd == CMD_DISCONNECT) {
                                g_device_state = STATE_IDLE;
                                printf("[STATE] CONNECTED -> IDLE\n");
                            }
                            break;

                        case STATE_STREAMING:
                            if (cmd == CMD_STOP) {
                                g_device_state = STATE_CONNECTED;
                                printf("[STATE] STREAMING -> CONNECTED\n");
                                /* Streaming loop will exit when it sees state change */
                            } else if (cmd == CMD_DISCONNECT) {
                                g_device_state = STATE_IDLE;
                                printf("[STATE] STREAMING -> IDLE\n");
                                /* Streaming loop will exit when it sees state change */
                            }
                            break;
                    }
                }

                line_idx = 0;
            }
        } else if (line_idx < CMD_BUFFER_SIZE - 1) {
            /* Add character to buffer */
            line_buffer[line_idx++] = (char)c;
        } else {
            /* Buffer overflow - reset */
            line_idx = 0;
        }
    }
}

/*******************************************************************************
 * State Machine
 ******************************************************************************/

/**
 * @brief Send JSON acknowledgment for connection.
 */
static void send_ack_connect(void)
{
    printf("{\"status\":\"ack_connect\",\"device\":\"ESP32-EMG\",\"channels\":%d}\n",
           EMG_NUM_CHANNELS);
    fflush(stdout);
}

/**
 * @brief Stream EMG data continuously until stopped.
 *
 * This function blocks and streams data at the configured sample rate.
 * Returns when state changes from STREAMING.
 */
static void stream_emg_data(void)
{
    emg_sample_t sample;
    const TickType_t delay_ticks = 1;  /* 1 tick = 1ms at 1000 Hz tick rate */

    while (g_device_state == STATE_STREAMING) {
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
 * @brief Main state machine loop.
 *
 * Monitors device state and starts streaming when requested.
 * Serial input task handles all state transitions directly.
 */
static void state_machine_loop(void)
{
    command_t cmd;
    const TickType_t poll_interval = pdMS_TO_TICKS(50);

    while (1) {
        /* Check if we should start streaming */
        if (g_device_state == STATE_STREAMING) {
            /* Stream until state changes (via serial input task) */
            stream_emg_data();
            /* Returns when state is no longer STREAMING */
        }

        /* Wait for start command or just poll state */
        /* Timeout allows checking state even if queue is empty */
        xQueueReceive(g_cmd_queue, &cmd, poll_interval);

        /* Note: State transitions are handled by serial_input_task */
        /* This loop only triggers streaming when state becomes STREAMING */
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
    printf("  Firmware v2.0.0 (Handshake Protocol)\n");
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

    /* Create command queue */
    g_cmd_queue = xQueueCreate(10, sizeof(command_t));
    if (g_cmd_queue == NULL) {
        printf("[ERROR] Failed to create command queue!\n");
        return;
    }

    /* Launch serial input task */
    xTaskCreate(
        serial_input_task,
        "serial_input",
        4096,              /* Stack size */
        NULL,              /* Parameters */
        5,                 /* Priority */
        NULL               /* Task handle */
    );

    printf("[PROTOCOL] Waiting for host to connect...\n");
    printf("[PROTOCOL] Send: {\"cmd\": \"connect\"}\n\n");

    /* Run main state machine */
    state_machine_loop();
}
