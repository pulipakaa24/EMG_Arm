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

#include "esp_timer.h"
#include <freertos/FreeRTOS.h>
#include <freertos/queue.h>
#include <freertos/task.h>
#include <stdio.h>
#include <string.h>

#include "config/config.h"
#include "core/gestures.h"
#include "core/inference.h" // [NEW]
#include "drivers/emg_sensor.h"
#include "drivers/hand.h"

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
  STATE_IDLE = 0,   /**< Waiting for connect command */
  STATE_CONNECTED,  /**< Connected, waiting for start command */
  STATE_STREAMING,  /**< Actively streaming raw EMG data (for training) */
  STATE_PREDICTING, /**< [NEW] On-device inference and control */
} device_state_t;

/**
 * @brief Commands from host.
 */
typedef enum {
  CMD_NONE = 0,
  CMD_CONNECT,
  CMD_START,         /**< Start raw streaming */
  CMD_START_PREDICT, /**< [NEW] Start on-device prediction */
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
static command_t parse_command(const char *line) {
  /* Simple JSON parsing - look for "cmd" field */
  const char *cmd_start = strstr(line, "\"cmd\"");
  if (!cmd_start) {
    return CMD_NONE;
  }

  /* Find the value after "cmd": */
  const char *value_start = strchr(cmd_start, ':');
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
  } else if (strncmp(value_start, "start_predict", 13) == 0) {
    return CMD_START_PREDICT;
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
 */
static void serial_input_task(void *pvParameters) {
  char line_buffer[CMD_BUFFER_SIZE];
  int line_idx = 0;

  while (1) {
    int c = getchar();

    if (c == EOF || c == 0xFF) {
      vTaskDelay(pdMS_TO_TICKS(10));
      continue;
    }

    if (c == '\n' || c == '\r') {
      if (line_idx > 0) {
        line_buffer[line_idx] = '\0';
        command_t cmd = parse_command(line_buffer);

        if (cmd != CMD_NONE) {
          if (cmd == CMD_CONNECT) {
            g_device_state = STATE_CONNECTED;
            send_ack_connect();
            printf("[STATE] ANY -> CONNECTED (reconnect)\n");
          } else {
            switch (g_device_state) {
            case STATE_IDLE:
              break;

            case STATE_CONNECTED:
              if (cmd == CMD_START) {
                g_device_state = STATE_STREAMING;
                printf("[STATE] CONNECTED -> STREAMING\n");
                xQueueSend(g_cmd_queue, &cmd, 0);
              } else if (cmd == CMD_START_PREDICT) {
                g_device_state = STATE_PREDICTING;
                printf("[STATE] CONNECTED -> PREDICTING\n");
                xQueueSend(g_cmd_queue, &cmd, 0);
              } else if (cmd == CMD_DISCONNECT) {
                g_device_state = STATE_IDLE;
                printf("[STATE] CONNECTED -> IDLE\n");
              }
              break;

            case STATE_STREAMING:
            case STATE_PREDICTING:
              if (cmd == CMD_STOP) {
                g_device_state = STATE_CONNECTED;
                printf("[STATE] ACTIVE -> CONNECTED\n");
              } else if (cmd == CMD_DISCONNECT) {
                g_device_state = STATE_IDLE;
                printf("[STATE] ACTIVE -> IDLE\n");
              }
              break;
            }
          }
        }
        line_idx = 0;
      }
    } else if (line_idx < CMD_BUFFER_SIZE - 1) {
      line_buffer[line_idx++] = (char)c;
    } else {
      line_idx = 0;
    }
  }
}

/*******************************************************************************
 * State Machine
 ******************************************************************************/

static void send_ack_connect(void) {
  printf(
      "{\"status\":\"ack_connect\",\"device\":\"ESP32-EMG\",\"channels\":%d}\n",
      EMG_NUM_CHANNELS);
  fflush(stdout);
}

/**
 * @brief Stream raw EMG data (Training Mode).
 */
static void stream_emg_data(void) {
  emg_sample_t sample;
  const TickType_t delay_ticks = 1;

  while (g_device_state == STATE_STREAMING) {
    emg_sensor_read(&sample);
    printf("%u,%u,%u,%u\n", sample.channels[0], sample.channels[1],
           sample.channels[2], sample.channels[3]);
    vTaskDelay(delay_ticks);
  }
}

/**
 * @brief Run on-device inference (Prediction Mode).
 */
static void run_inference_loop(void) {
  emg_sample_t sample;
  const TickType_t delay_ticks = 1; // 1ms @ 1kHz
  int last_gesture = -1;

  // Reset inference state
  inference_init();
  printf("{\"status\":\"info\",\"msg\":\"Inference started\"}\n");

  while (g_device_state == STATE_PREDICTING) {
    emg_sensor_read(&sample);

    // Add to buffer
    // Note: sample.channels is uint16_t, matching inference engine expectation
    if (inference_add_sample(sample.channels)) {
      // Buffer full (sliding window), run prediction
      // We can optimize stride here (e.g. valid prediction only every N
      // samples) For now, let's predict every sample (sliding window) or
      // throttle if too slow. ESP32S3 is fast enough for 4ch features @ 1kHz?
      // maybe. Let's degrade to 50Hz updates (20ms stride) to be safe and avoid
      // UART spam.

      static int stride_counter = 0;
      stride_counter++;

      if (stride_counter >= 20) { // 20ms stride
        float confidence = 0;
        int gesture_idx = inference_predict(&confidence);
        stride_counter = 0;

        if (gesture_idx >= 0) {
          // Map class index (0-N) to gesture enum (correct hardware action)
          int gesture_enum = inference_get_gesture_enum(gesture_idx);

          // Execute gesture on hand
          gestures_execute((gesture_t)gesture_enum);

          // Send telemetry if changed or periodically?
          // "Live prediction flow should change to only have each new output...
          // sent"
          if (gesture_idx != last_gesture) {
            printf("{\"gesture\":\"%s\",\"conf\":%.2f}\n",
                   inference_get_class_name(gesture_idx), confidence);
            last_gesture = gesture_idx;
          }
        }
      }
    }

    vTaskDelay(delay_ticks);
  }
}

static void state_machine_loop(void) {
  command_t cmd;
  const TickType_t poll_interval = pdMS_TO_TICKS(50);

  while (1) {
    if (g_device_state == STATE_STREAMING) {
      stream_emg_data();
    } else if (g_device_state == STATE_PREDICTING) {
      run_inference_loop();
    }

    xQueueReceive(g_cmd_queue, &cmd, poll_interval);
  }
}

void appConnector() {
  g_cmd_queue = xQueueCreate(10, sizeof(command_t));
  if (g_cmd_queue == NULL) {
    printf("[ERROR] Failed to create command queue!\n");
    return;
  }

  xTaskCreate(serial_input_task, "serial_input", 4096, NULL, 5, NULL);

  printf("[PROTOCOL] Waiting for host to connect...\n");
  printf("[PROTOCOL] Send: {\"cmd\": \"connect\"}\n");
  printf("[PROTOCOL] Send: {\"cmd\": \"start_predict\"} for on-device "
         "inference\n\n");

  state_machine_loop();
}

/*******************************************************************************
 * Application Entry Point
 ******************************************************************************/

void app_main(void) {
  printf("\n");
  printf("========================================\n");
  printf("  Bucky Arm - EMG Robotic Hand\n");
  printf("  Firmware v2.1.0 (Inference Enabled)\n");
  printf("========================================\n\n");

  printf("[INIT] Initializing hand (servos)...\n");
  hand_init();

  printf("[INIT] Initializing EMG sensor...\n");
  emg_sensor_init();

  printf("[INIT] Initializing Inference Engine...\n");
  inference_init();

#if FEATURE_FAKE_EMG
  printf("[INIT] Using FAKE EMG data (sensors not connected)\n");
#else
  printf("[INIT] Using REAL EMG sensors\n");
#endif

  printf("[INIT] Done!\n\n");

  appConnector();
}
